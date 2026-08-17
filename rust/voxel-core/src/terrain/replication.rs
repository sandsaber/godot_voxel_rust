//! `terrain::replication` — transport-agnostic replication protocol
//! (ROADMAP R3). Implements the boundary decided in
//! [doc/source/multiplayer.md](../../../doc/source/multiplayer.md):
//!
//! - The **server** is authoritative. It tracks one interest box per peer in
//!   a [`VoxelAreaFinder`] and produces outbound messages: block snapshots
//!   for edited blocks entering a peer's interest, and snapshots of edited
//!   blocks whose revision advanced since the last poll. Never-generated
//!   (pure generator) blocks are never replicated — clients generate them
//!   locally, exactly like upstream's `are_voxels_edited` shortcut.
//! - The **client** applies snapshots through
//!   [`VoxelTerrainCore::try_install_remote_block`], which replaces the
//!   resident block atomically (revision-ordered; stale snapshots are
//!   dropped by comparing the carried revision against the client's block
//!   revision).
//!
//! Messages are length-prefixed byte frames so any transport works: the
//! reference in-memory channel here, Godot's MultiplayerAPI RPCs, ENet, or
//! plain UDP in a game crate. This module deliberately contains **no
//! sockets** — the gdext node bridges frames to a peer connection of the
//! game's choosing.
//!
//! What is intentionally *not* here (recorded in multiplayer.md): edit
//! *deltas* (edits replicate as whole-block snapshots of the affected LOD0
//! blocks; the batch-edit per-block-revision API that deltas need is the
//! recorded follow-up), LOD>0 replication (clients mesh locally), and
//! hand-off/reconciliation beyond revision ordering.

use crate::math::{Box3i, Vector3i};
use crate::storage::VoxelBuffer;
use crate::streams::block_serializer;
use crate::streams::compressed_data::Compression;
use crate::terrain::area_finder::{AreaId, VoxelAreaFinder};
use crate::terrain::voxel_terrain_core::VoxelTerrainCore;
use std::collections::HashMap;

pub const PROTOCOL_VERSION: u16 = 1;

const MSG_KIND_SNAPSHOT: u8 = 1;

/// A serialized block snapshot: one LOD0 block in the v4 serializer bytes
/// (the same envelope region files store), with its server-side revision.
#[derive(Debug, Clone, PartialEq)]
pub struct BlockSnapshot {
    pub position_in_blocks: Vector3i,
    pub lod_index: u8,
    pub block_revision: u64,
    pub payload: Vec<u8>,
}

impl BlockSnapshot {
    /// Serialize as a length-prefixed frame: `u8 kind | u16 version |
    /// i32 x | i32 y | i32 z | u8 lod | u64 revision | u32 payload_len |
    /// payload`.
    pub fn encode(&self, dst: &mut Vec<u8>) {
        use crate::io::serialization::MemoryWriter;
        let mut w = MemoryWriter::little(dst);
        w.store_8(MSG_KIND_SNAPSHOT);
        w.store_16(PROTOCOL_VERSION);
        w.store_32(self.position_in_blocks.x as u32);
        w.store_32(self.position_in_blocks.y as u32);
        w.store_32(self.position_in_blocks.z as u32);
        w.store_8(self.lod_index);
        w.store_64(self.block_revision);
        w.store_32(self.payload.len() as u32);
        w.store_buffer(&self.payload);
    }

    /// Decode one frame from `src`; returns `None` on any mismatch
    /// (malformed frame, unknown version, truncated payload).
    pub fn decode(src: &[u8]) -> Option<BlockSnapshot> {
        #[allow(unused_imports)]
        use crate::io::serialization::Endianness as _Endianness;
        use crate::io::serialization::{Endianness, MemoryReader};
        let mut r = MemoryReader::new(src, Endianness::LittleEndian);
        if r.try_get_8()? != MSG_KIND_SNAPSHOT {
            return None;
        }
        if r.try_get_16()? != PROTOCOL_VERSION {
            return None;
        }
        let x = r.try_get_32()? as i32;
        let y = r.try_get_32()? as i32;
        let z = r.try_get_32()? as i32;
        let lod_index = r.try_get_8()?;
        // The boundary is LOD0-only by design (multiplayer.md); a hostile
        // or version-skewed frame must die here, not two layers down in a
        // storage assert.
        if lod_index != 0 {
            return None;
        }
        let block_revision = r.try_get_64()?;
        let len = r.try_get_32()? as usize;
        let payload = r.try_take(len)?.to_vec();
        if r.remaining() != 0 {
            return None;
        }
        Some(BlockSnapshot {
            position_in_blocks: Vector3i::new(x, y, z),
            lod_index,
            block_revision,
            payload,
        })
    }

    /// Decompress + deserialize the payload into `out` (the exact path a
    /// region-file load takes, `DecodeLimits`-guarded).
    pub fn materialize(&self, out: &mut VoxelBuffer) -> bool {
        block_serializer::decompress_and_deserialize(&self.payload, out).is_ok()
    }
}

/// Server-side replication state: peer interest boxes + per-block
/// last-sent revisions.
/// Per-peer "last sent" revisions: block key -> server revision.
type SentMap = HashMap<(i32, i32, i32, u8), u64>;

pub struct ReplicationServer {
    interests: VoxelAreaFinder,
    /// Last server revision already sent per peer per block.
    sent: HashMap<AreaId, SentMap>,
}

impl ReplicationServer {
    /// `cell_size` is in blocks; see [`VoxelAreaFinder`] for guidance
    /// (choose near the typical snapshot query size, e.g. one block).
    pub fn new(cell_size: i32) -> Self {
        Self {
            interests: VoxelAreaFinder::new(cell_size),
            sent: HashMap::new(),
        }
    }

    /// Set (or move) a peer's interest box, in LOD0 block coordinates.
    /// Returns entered/exited block boxes so the transport can send
    /// interest updates if it wants to (snapshots are produced by
    /// [`Self::poll_outbound`] regardless).
    pub fn set_peer_area(
        &mut self,
        peer: AreaId,
        area: Box3i,
    ) -> Result<(Vec<Box3i>, Vec<Box3i>), crate::terrain::area_finder::AreaError> {
        let previous = self.interests.area(peer);
        // `update` requires an existing peer; a new one inserts instead.
        if previous.is_some() {
            self.interests.update(peer, area)?;
        } else {
            self.interests.insert(peer, area)?;
        }
        let entered = match previous {
            Some(old) => crate::terrain::area_finder::box_subtraction(area, old),
            None => vec![area],
        };
        let exited = match previous {
            Some(old) => crate::terrain::area_finder::box_subtraction(old, area),
            None => Vec::new(),
        };
        // Re-entered blocks must resend even at an unchanged revision: the
        // client evicted them while outside its interest, and its generator
        // cannot reproduce edited state.
        if let Some(sent) = self.sent.get_mut(&peer) {
            sent.retain(|&(x, y, z, _lod), _| {
                !entered
                    .iter()
                    .any(|b| b.contains_point(Vector3i::new(x, y, z)))
            });
        }
        Ok((entered, exited))
    }

    pub fn remove_peer(&mut self, peer: AreaId) -> bool {
        self.sent.remove(&peer);
        self.interests.remove(peer)
    }

    /// Produce snapshots for `terrain`: for each peer, every resident LOD0
    /// block that is edited, whose revision advanced since the last poll,
    /// and whose position intersects the peer's interest box. Order is
    /// deterministic (sorted by position, then peer id). Pure generator
    /// blocks (`!is_edited()`) are never sent.
    pub fn poll_outbound(&mut self, terrain: &VoxelTerrainCore) -> Vec<(AreaId, BlockSnapshot)> {
        // One pass, no per-block clones: serialize each changed edited
        // block once and share the payload across overlapping peers.
        struct Pending {
            revision: u64,
            payload: Option<Vec<u8>>,
        }
        let view = terrain.data();
        let mut edited: Vec<(Vector3i, Pending)> = Vec::new();
        view.for_each_edited_block(0, |position, revision, _voxels| {
            edited.push((
                position,
                Pending {
                    revision,
                    payload: None,
                },
            ));
        });
        edited.sort_unstable_by_key(|(p, _)| (p.x, p.y, p.z));
        let peers = self.peer_areas_sorted();
        let mut out = Vec::new();
        for peer in peers {
            let Some(area) = self.interests.area(peer) else {
                continue;
            };
            let sent = self.sent.entry(peer).or_default();
            for (position, pending) in &mut edited {
                if !area.contains_point(*position) {
                    continue;
                }
                let key = (position.x, position.y, position.z, 0);
                if sent.get(&key).is_some_and(|&last| last >= pending.revision) {
                    continue;
                }
                if pending.payload.is_none() {
                    let Some(block) = view.block_snapshot(*position, 0) else {
                        continue;
                    };
                    let voxels = block.into_voxels();
                    let Some(voxels) = voxels.as_ref() else {
                        continue;
                    };
                    let mut payload = Vec::new();
                    if block_serializer::serialize_and_compress(
                        voxels,
                        &mut payload,
                        Compression::Lz4,
                    )
                    .is_err()
                    {
                        continue;
                    }
                    pending.payload = Some(payload);
                }
                sent.insert(key, pending.revision);
                out.push((
                    peer,
                    BlockSnapshot {
                        position_in_blocks: *position,
                        lod_index: 0,
                        block_revision: pending.revision,
                        payload: pending.payload.clone().expect("just serialized"),
                    },
                ));
            }
        }
        out
    }

    /// Like [`Self::poll_outbound`] but computes and marks exactly one
    /// peer. The per-peer bridge API must use this: calling the all-peers
    /// version and filtering the result marks every peer's frames as sent
    /// on the first poll, starving everyone else.
    pub fn poll_outbound_for_peer(
        &mut self,
        terrain: &VoxelTerrainCore,
        peer: AreaId,
    ) -> Vec<BlockSnapshot> {
        let Some(area) = self.interests.area(peer) else {
            return Vec::new();
        };
        let view = terrain.data();
        let mut candidates: Vec<(Vector3i, u64)> = Vec::new();
        view.for_each_edited_block(0, |position, revision, _voxels| {
            if area.contains_point(position) {
                candidates.push((position, revision));
            }
        });
        let sent = self.sent.entry(peer).or_default();
        let mut out = Vec::new();
        for (position, revision) in candidates {
            if sent
                .get(&(position.x, position.y, position.z, 0))
                .is_some_and(|&last| last >= revision)
            {
                continue;
            }
            let Some(block) = view.block_snapshot(position, 0) else {
                continue;
            };
            let Some(voxels) = block.into_voxels() else {
                continue;
            };
            let mut payload = Vec::new();
            if block_serializer::serialize_and_compress(&voxels, &mut payload, Compression::Lz4)
                .is_err()
            {
                continue;
            }
            sent.insert((position.x, position.y, position.z, 0), revision);
            out.push(BlockSnapshot {
                position_in_blocks: position,
                lod_index: 0,
                block_revision: revision,
                payload,
            });
        }
        out
    }

    fn peer_areas_sorted(&self) -> Vec<AreaId> {
        let mut peers: Vec<AreaId> = self.sent.keys().copied().collect();
        // Include peers with areas but no traffic yet.
        self.interests
            .areas_in_box(Box3i::new(
                Vector3i::new(i32::MIN / 2, i32::MIN / 2, i32::MIN / 2),
                Vector3i::new(i32::MAX, i32::MAX, i32::MAX),
            ))
            .iter()
            .for_each(|(peer, _)| {
                if !peers.contains(peer) {
                    peers.push(*peer);
                }
            });
        peers.sort_unstable();
        peers
    }
}

/// Client-side inbound ordering. Server and client revisions live in
/// different processes (multiplayer.md: revisions are per server process),
/// so the client tracks the last server revision it applied per block and
/// drops anything not strictly newer — stale or duplicated delivery.
pub struct ReplicationClient {
    last_applied: HashMap<(i32, i32, i32, u8), u64>,
}

impl Default for ReplicationClient {
    fn default() -> Self {
        Self::new()
    }
}

impl ReplicationClient {
    pub fn new() -> Self {
        Self {
            last_applied: HashMap::new(),
        }
    }

    /// Apply one frame. Returns `true` when the snapshot was installed.
    pub fn apply_frame(&mut self, terrain: &mut VoxelTerrainCore, frame: &[u8]) -> bool {
        let Some(snapshot) = BlockSnapshot::decode(frame) else {
            return false;
        };
        self.apply_snapshot(terrain, &snapshot)
    }

    pub fn apply_snapshot(
        &mut self,
        terrain: &mut VoxelTerrainCore,
        snapshot: &BlockSnapshot,
    ) -> bool {
        let key = (
            snapshot.position_in_blocks.x,
            snapshot.position_in_blocks.y,
            snapshot.position_in_blocks.z,
            snapshot.lod_index,
        );
        if self
            .last_applied
            .get(&key)
            .is_some_and(|&applied| snapshot.block_revision <= applied)
        {
            return false;
        }
        let mut voxels = VoxelBuffer::new(crate::storage::Allocator::Default);
        if !snapshot.materialize(&mut voxels) {
            return false;
        }
        if !terrain.try_install_remote_block(
            snapshot.position_in_blocks,
            snapshot.lod_index,
            voxels,
        ) {
            return false;
        }
        self.last_applied.insert(key, snapshot.block_revision);
        true
    }

    /// Drop the revision table (rejoin / server restart; see multiplayer.md).
    pub fn reset(&mut self) {
        self.last_applied.clear();
    }

    /// Last server revision applied for a block (debug/introspection).
    pub fn last_applied_revision(&self, position: Vector3i, lod_index: u8) -> Option<u64> {
        self.last_applied
            .get(&(position.x, position.y, position.z, lod_index))
            .copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::edition::ops::EditMode;
    use crate::storage::{ChannelId, MetadataValue};
    use crate::terrain::voxel_terrain_core::ViewerUpdate;

    fn viewer_at(id: u32) -> ViewerUpdate {
        ViewerUpdate {
            id,
            world_position_voxels: Vector3i::zero(),
            horizontal_view_distance_voxels: 16,
            vertical_view_distance_voxels: 16,
            demand: crate::terrain::clipbox_coordinator::MeshDemand {
                visuals: true,
                collisions: true,
            },
        }
    }

    fn make_terrain(edited: bool) -> VoxelTerrainCore {
        let mut data = crate::storage::VoxelData::new();
        data.set_bounds(Box3i::new(Vector3i::splat(-1024), Vector3i::splat(2048)));
        data.set_streaming_enabled(false);
        data.set_full_load_completed(true);
        let mut core = VoxelTerrainCore::legacy_variable_lod_for_parity(
            data,
            std::sync::Arc::new(crate::streams::MemoryStream::new()),
            crate::engine::MeshingDependency::new(
                std::sync::Arc::new(crate::meshers::TransvoxelMesher::new()),
                None,
            ),
            1,
        );
        if edited {
            let _ = core.try_process(&[viewer_at(1)]).unwrap();
            let _ = core
                .try_edit_box(
                    Vector3i::new(1, 1, 1),
                    Vector3i::new(4, 4, 4),
                    ChannelId::Sdf.index(),
                    EditMode::Set,
                    0,
                )
                .unwrap();
            let _ = core
                .try_edit_voxel_metadata(
                    Vector3i::new(2, 2, 2),
                    Some(MetadataValue::Text("server".into())),
                )
                .unwrap();
        }
        core
    }

    #[test]
    fn nonzero_block_replicates_to_same_position() {
        // Regression: for_each_edited_block once yielded voxel origins —
        // only block (0,0,0) ever matched its own snapshot lookup.
        let mut server = make_terrain(true);
        // A second viewer covering block (2,2,2) so the edit materializes
        // there; paging needs several ticks (task schedule + install).
        let viewer2 = ViewerUpdate {
            id: 2,
            world_position_voxels: Vector3i::splat(40),
            horizontal_view_distance_voxels: 16,
            vertical_view_distance_voxels: 16,
            demand: crate::terrain::clipbox_coordinator::MeshDemand {
                visuals: true,
                collisions: true,
            },
        };
        for _ in 0..20 {
            let _ = server.try_process(&[viewer2]).unwrap();
            if server
                .data()
                .block_snapshot(Vector3i::new(2, 2, 2), 0)
                .is_some()
            {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        assert!(
            server
                .try_edit_voxel(5, Vector3i::new(33, 34, 35), ChannelId::Sdf.index())
                .unwrap()
                .is_some(),
            "edit must materialize block (2,2,2)"
        );
        let mut client = make_terrain(false);
        let mut protocol = ReplicationServer::new(8);
        protocol
            .set_peer_area(9, Box3i::new(Vector3i::splat(-4), Vector3i::splat(8)))
            .unwrap();
        eprintln!(
            "DEBUG block_size={} resident={:?} edited(2,2,2)={:?} area_contains={}",
            server.data().block_size(),
            server.data().block_positions(0),
            server
                .data()
                .block_snapshot(Vector3i::new(2, 2, 2), 0)
                .map(|b| b.is_edited()),
            Box3i::new(Vector3i::splat(-4), Vector3i::splat(8))
                .contains_point(Vector3i::new(2, 2, 2)),
        );
        let frames = protocol.poll_outbound_for_peer(&server, 9);
        assert!(
            frames
                .iter()
                .any(|s| s.position_in_blocks == Vector3i::new(2, 2, 2)),
            "block (2,2,2) must be captured, got {:?}",
            frames
                .iter()
                .map(|f| f.position_in_blocks)
                .collect::<Vec<_>>()
        );
        let mut rx = ReplicationClient::new();
        let mut installed = 0;
        for snapshot in frames {
            if rx.apply_snapshot(&mut client, &snapshot) {
                installed += 1;
            }
        }
        assert!(installed >= 1);
        assert_eq!(
            client
                .data()
                .block_snapshot(Vector3i::new(2, 2, 2), 0)
                .map(|b| b.is_edited()),
            Some(true)
        );
    }

    #[test]
    fn per_peer_poll_does_not_starve_other_peers() {
        // Regression: the all-peers poll marked every peer's frames sent on
        // the first call, so the second peer's poll came back empty.
        let server = make_terrain(true);
        let mut protocol = ReplicationServer::new(8);
        let area = Box3i::new(Vector3i::splat(-4), Vector3i::splat(8));
        protocol.set_peer_area(1, area).unwrap();
        protocol.set_peer_area(2, area).unwrap();
        assert!(!protocol.poll_outbound_for_peer(&server, 1).is_empty());
        assert!(
            !protocol.poll_outbound_for_peer(&server, 2).is_empty(),
            "peer 2 must not be starved by peer 1's poll"
        );
        // And both are quiet now.
        assert!(protocol.poll_outbound_for_peer(&server, 1).is_empty());
        assert!(protocol.poll_outbound_for_peer(&server, 2).is_empty());
    }

    #[test]
    fn frame_round_trips() {
        let snapshot = BlockSnapshot {
            position_in_blocks: Vector3i::new(-1, 2, 3),
            lod_index: 0,
            block_revision: 9,
            payload: vec![1, 2, 3, 4, 5],
        };
        let mut frame = Vec::new();
        snapshot.encode(&mut frame);
        assert_eq!(BlockSnapshot::decode(&frame), Some(snapshot));
        // Truncated / junk frames are rejected.
        assert_eq!(BlockSnapshot::decode(&frame[..frame.len() - 1]), None);
        assert_eq!(BlockSnapshot::decode(&[9, 1, 0]), None);
    }

    #[test]
    fn edited_blocks_reach_interested_peer_with_metadata() {
        let server = make_terrain(true);
        let mut client = make_terrain(false);
        let mut protocol = ReplicationServer::new(8);
        protocol
            .set_peer_area(7, Box3i::new(Vector3i::splat(-4), Vector3i::splat(8)))
            .unwrap();

        let outbound = protocol.poll_outbound(&server);
        assert!(!outbound.is_empty(), "edited block must be sent");

        let (peer, snapshot) = &outbound[0];
        assert_eq!(*peer, 7);
        assert_eq!(snapshot.position_in_blocks, Vector3i::zero());
        assert!(snapshot.block_revision >= 1);

        let mut frame = Vec::new();
        snapshot.encode(&mut frame);
        let mut rx = ReplicationClient::new();
        assert!(rx.apply_frame(&mut client, &frame));

        assert_eq!(
            client.voxel_metadata(Vector3i::new(2, 2, 2)),
            Some(MetadataValue::Text("server".into()))
        );

        // Second poll without edits sends nothing new.
        assert!(protocol.poll_outbound(&server).is_empty());
        // Re-delivering the same frame is dropped by revision ordering.
        assert!(!rx.apply_frame(&mut client, &frame));
    }

    #[test]
    fn unedited_blocks_are_never_sent() {
        let server = make_terrain(false);
        let mut protocol = ReplicationServer::new(8);
        protocol
            .set_peer_area(1, Box3i::new(Vector3i::splat(-8), Vector3i::splat(16)))
            .unwrap();
        assert!(
            protocol.poll_outbound(&server).is_empty(),
            "pure generator blocks replicate as nothing"
        );
    }

    #[test]
    fn peer_area_exit_stops_sending() {
        let server = make_terrain(true);
        let mut protocol = ReplicationServer::new(8);
        protocol
            .set_peer_area(2, Box3i::new(Vector3i::splat(-4), Vector3i::splat(8)))
            .unwrap();
        assert!(!protocol.poll_outbound(&server).is_empty());
        // Edit again -> revision advances; move the peer away first.
        let mut server = server;
        protocol
            .set_peer_area(
                2,
                Box3i::new(Vector3i::new(5000, 5000, 5000), Vector3i::splat(8)),
            )
            .unwrap();
        let _ = server
            .try_edit_voxel(9, Vector3i::new(6, 6, 6), ChannelId::Sdf.index())
            .unwrap();
        assert!(
            protocol.poll_outbound(&server).is_empty(),
            "blocks outside the interest box are not sent"
        );
    }

    #[test]
    fn later_revision_wins_on_reordered_delivery() {
        let mut server = make_terrain(true);
        let mut client = make_terrain(false);
        let mut protocol = ReplicationServer::new(8);
        protocol
            .set_peer_area(3, Box3i::new(Vector3i::splat(-4), Vector3i::splat(8)))
            .unwrap();
        let first = protocol.poll_outbound(&server).remove(0).1;

        // Server edits again; second snapshot carries a higher revision.
        let _ = server
            .try_edit_voxel_metadata(Vector3i::new(3, 3, 3), Some(MetadataValue::Int(42)))
            .unwrap();
        let second = protocol.poll_outbound(&server).remove(0).1;
        assert!(second.block_revision > first.block_revision);

        // Deliver out of order: newer first, stale second must be dropped.
        let mut rx = ReplicationClient::new();
        assert!(rx.apply_snapshot(&mut client, &second));
        assert!(!rx.apply_snapshot(&mut client, &first));
        assert_eq!(
            client.voxel_metadata(Vector3i::new(3, 3, 3)),
            Some(MetadataValue::Int(42))
        );
    }
}
