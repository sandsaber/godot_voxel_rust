# Multiplayer — replication boundary (design)

Status: **design pass, no transport**. This page is the R3 decision record
required by the roadmap before any networking code is written. It defines what
is authoritative, what crosses the wire (conceptually), and where that meets
the current `voxel-core` code. A transport / interest protocol on top of this
boundary is explicitly **not started** — see "Out of scope".

Upstream context: the C++ engine shipped a server-authoritative setup built
from `VoxelTerrainMultiplayerSynchronizer` plus viewer notifications
(`doc/source/multiplayer.md` at the C++ removal commit). That design is the
baseline; this page adapts it to the Rust port's real APIs.

## Authority model

- **The server is the only writer.** Clients never send voxel data. The only
  client→server inputs are viewer state (position, view distance) and edit
  *requests*, which the server validates and applies through the same code
  path as local edits.
- **Edits are authoritative facts; generated terrain is not.** A block that
  was never edited is reproducible from (generator, seed, position) and must
  not be replicated — the client generates it locally, exactly like the C++
  `are_voxels_edited()` shortcut. Only edited blocks and edit deltas cross
  the server→client boundary.
- **Metadata replicates with its block.** Per-voxel/block `MetadataValue`
  persists in the v4 block payload (R7 narrow); the same bytes are the wire
  payload, so metadata needs no separate protocol.

## Unit of replication

Two granularities, both already expressible in the core:

1. **Whole blocks** — initial or rejoin state for *edited* blocks. Payload:
   the v4 block-serializer bytes (`block_serializer::serialize_and_compress`,
   LZ4 by default — the same envelope region files store). Block framing:
   position (`Vector3i`, block coordinates in LOD0), `block_revision` (u64,
   see below), and the serializer bytes.
2. **Edit deltas** — the steady state. An edit applied through
   `VoxelTerrainCore::try_edit_sphere/box/hemisphere/smooth` or
   `try_edit_voxel_metadata` is replicated as `(operation, parameters,
   revision)`, e.g. `(Sphere, center, radius, channel, mode, value)`, not as
   resulting voxels. Clients replay operations against their local copy
   through the same public edit API, so client-side replay needs no new core
   code. (*Pastes excepted:* `try_paste` carries an arbitrary buffer and
   replicates as a snapshot — see Consistency rules. Per-voxel metadata edits
   are deltas too; successive metadata deltas for one block may be coalesced
   or superseded by a block snapshot at the latest revision.)

`block_revision` is the per-block ordering token, carried publicly by
`VoxelEditOutcome.block_revision` (single-block edit paths) and
`BlockToSave.block_revision` (the save path); internally both come from the
same strictly-monotonic per-block counter, advancing by exactly one per
committed edit. **Scope: one server process lifetime.** The counter is not
persisted to disk (the v4 block format stores no revision), so after a server
restart counters reset while disk state is whatever was last saved — a client
must therefore drop its revision table on rejoin or server restart and
resynchronize from whole-block snapshots (rejoin already implies that).

## Where replication meets the core (all hooks exist)

| Concern | Core API today |
|---|---|
| Apply an edit server-side | `try_edit_sphere` / `try_edit_box` / hemisphere / smooth / `try_paste` / `try_edit_voxel_metadata` — one transaction per touched data block |
| Detect "block entered a peer's interest" | `VoxelTerrainEvent::DataBlockLoaded` / `DataBlockUnloaded` on the server core |
| Detect "block was edited" | edit results + block `is_edited()` (blocks only become non-generated through edits/paste) |
| Serialize a block to bytes | `block_serializer::serialize_and_compress` |
| Revision of a single-block edit | `VoxelEditOutcome.block_revision` (returned by `try_edit_voxel` / `try_edit_voxel_metadata`) |
| Interest index ("who cares about this box") | `VoxelAreaFinder` (`terrain::area_finder`) |
| Interest diff on viewer movement | `box_subtraction(old, new)` / `box_subtraction(new, old)` (same module) |
| Client applies received data | existing block-install path used by stream loads (`BlockDataOutput::loaded`) — a network source is a stream, see below |
| **Revision of a multi-block edit** | **Gap.** `try_edit_sphere/box/hemisphere/smooth` return only a touched-block count; per-block revisions of a batch are not publicly exposed. Decision for the transport stage: extend the batch edit APIs to return per-block `VoxelEditOutcome`s (preferred, mirrors the single-block path) — a transport-level sequence number is the fallback if that proves invasive. This must land before any transport work. |

**Client-side integration shape:** a client terrain runs with
`automatic loading` off and a network "stream" that answers
`load_voxel_block(position)` from the peer's received-block cache, falling
back to the local generator for never-edited blocks. This keeps the client on
the standard paging/meshing path — the clipbox planner, meshers, and unloading
all work unchanged, mirroring the C++ `try_set_block_data` flow.

**Viewer hysteresis:** the client's local `VoxelViewer` keeps a slightly
larger view distance than the server-side interest box so blocks are never
unloaded by the client before the server stops sending updates (same
recommendation as upstream).

## Interest management (VoxelAreaFinder)

The server keeps one `VoxelAreaFinder` entry per peer: the peer's interest box
(viewer position ± view distance, in block coordinates). Boxes are half-open
`Box3i`s; an edit's dirty box arrives in voxel coordinates and must be
downscaled to LOD0 block coordinates (divide by the data block size, floor
toward negative infinity) before querying — the same downscale upstream's
`get_viewers_in_area` performed.

- **Edits fan out** through `for_each_area_in_box(dirty_box)`: every edit
  returns/implies the dirty box; the server sends the delta to each matching
  peer. Deterministic ascending-id iteration keeps send order reproducible.
- **Viewer movement** reindexes via `update(peer_id, new_box)`;
  `box_subtraction(new, old)` is the set of block boxes to load-and-send
  (entered), `box_subtraction(old, new)` the set allowed to be dropped
  client-side (exited).
- **Queries are box-based, not block-based**: bulk operations (block entered,
  area edited) compute the box once and ask the finder once.
- **Bounds**: the server clamps peer view distances to a configured maximum
  before indexing (the transport's job); the finder independently rejects
  areas covering more than `MAX_CELLS_PER_AREA` spatial-hash cells with
  `TooManyCells`, so a malformed box can never wedge the server.

`VoxelAreaFinder` is pure `voxel-core` (no sockets, no Godot types): the
network layer is expected to live in `voxel-gdext` or a game-side crate and
own peers, RPCs, and reliability choices.

## Consistency rules

- **Deltas are ordered per block by `block_revision`**; a whole-block
  snapshot supersedes any queued deltas for that block (client checks
  revision before applying).
- **Pastes are snapshots, not deltas**: `try_paste` can carry arbitrary
  channel data, so it replicates as a block snapshot (or box of snapshots)
  at its revision.
- **LOD: replicate LOD0 only.** Clients mesh locally; higher LODs are a pure
  function of LOD0 + the local clipbox planner. Nothing about LOD crosses
  the wire. Concretely: an LOD0 block dirty for saving is due for
  replication at the same revision; LOD>0 blocks dirtied by the edit cascade
  are save-only and never replicate.
- **Stream save on the server is untouched**: replication observes the same
  transactions that feed `VoxelSaveQuery` on unload; it does not fork them.

## Security posture

Client→server inputs are validated shapes only: viewer position/distances
(bounded), edit requests (bounded radius/box per permission policy), block
requests (positions). No serialized voxel data is ever accepted from a
client, so the untrusted-input parsers (`block_serializer`, region files)
stay server/disk-side only. Received-from-server block bytes on the client go
through the same `DecodeLimits`-guarded decode as disk data.

## Out of scope (explicit non-goals for now)

- Any transport: ENet/WebSocket/MultiplayerAPI wiring, channels, reliability.
- Client-side prediction or edit reconciliation beyond revision ordering.
- Replicating instancer state (R5) and graph/generator *changes* at runtime
  (generator identity is assumed synced at world start).
- Variant metadata rides inside block snapshots via the R7 wide codec (tag 32); engine-only Variant types (objects, callables) remain unsupported on both sides.
