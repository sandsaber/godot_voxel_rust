//! `terrain::area_finder` — interest-area index for multiplayer replication
//! (ROADMAP R3).
//!
//! Answers the two questions a server-authoritative voxel game keeps asking:
//!
//! - *An area changed* (an edit dirtied a box of voxels, a block was loaded or
//!   unloaded) — **which viewers/peers care?** → [`VoxelAreaFinder`].
//! - *A viewer moved* — **which blocks entered or left its interest box?**
//!   → [`box_subtraction`] on the old/new boxes.
//!
//! The structure is pure `voxel-core`: no sockets, peers, or Godot types. A
//! [`VoxelAreaFinder`] maps stable ids (peer ids, viewer ids) to axis-aligned
//! [`Box3i`] interest areas and answers box-intersection queries in
//! (sub)linear time via a coarse spatial hash, instead of scanning every
//! area. The replication design that consumes it lives in
//! `doc/source/multiplayer.md`.
//!
//! Port note: upstream C++ never had a standalone `VoxelAreaFinder`; the same
//! logic lived inside `VoxelTerrain::get_viewer_network_peer_ids_in_area` and
//! the server-side `VoxelTerrainMultiplayerSynchronizer`. This module is the
//! extracted, engine-agnostic core of those.

use crate::math::{Box3i, Vector3i};
use std::collections::HashMap;

/// Identifier of a tracked interest area. Opaque here; callers typically use
/// network peer ids or viewer ids.
pub type AreaId = u32;

/// Coarse spatial-hash cell coordinate. Cells are cubes of `cell_size` blocks;
/// their coordinates can be negative.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct CellKey {
    x: i32,
    y: i32,
    z: i32,
}

/// Index over axis-aligned interest areas answering "which areas intersect
/// this box?".
///
/// Areas are half-open boxes in block coordinates (`position` inclusive,
/// `position + size` exclusive), matching [`Box3i`] semantics everywhere else
/// in the core. Empty areas (any non-positive axis) never match and are not
/// indexed.
///
/// Query results are **deterministic**: intersecting areas are reported in
/// ascending id order, so network code that fans edits out over a finder
/// behaves identically run to run.
///
/// Complexity: insertion/removal/update touches every cell an area overlaps
/// (`O((box/cell_size)^3)`); a query touches every cell the query box
/// overlaps plus one intersection test per candidate — independent of the
/// total number of tracked areas. Pick `cell_size` near the typical *query*
/// box diameter (e.g. one edit brush or one block), not the (much larger)
/// viewer areas.
#[derive(Debug, Clone)]
pub struct VoxelAreaFinder {
    areas: HashMap<AreaId, Box3i>,
    cells: HashMap<CellKey, Vec<AreaId>>,
    cell_size: i32,
}

impl VoxelAreaFinder {
    /// Build a finder with a spatial-hash cell size in blocks. Panics if
    /// `cell_size < 1` — a non-positive cell size cannot index anything.
    pub fn new(cell_size: i32) -> Self {
        assert!(
            cell_size >= 1,
            "cell_size must be positive, got {cell_size}"
        );
        Self {
            areas: HashMap::new(),
            cells: HashMap::new(),
            cell_size,
        }
    }

    /// Number of tracked areas.
    pub fn len(&self) -> usize {
        self.areas.len()
    }

    /// Whether no area is tracked.
    pub fn is_empty(&self) -> bool {
        self.areas.is_empty()
    }

    /// The box of a tracked area, if present.
    pub fn area(&self, id: AreaId) -> Option<Box3i> {
        self.areas.get(&id).copied()
    }

    /// Insert a new area. Returns `false` (and does nothing) if `id` is
    /// already in use — updates go through [`VoxelAreaFinder::update`].
    pub fn insert(&mut self, id: AreaId, area: Box3i) -> bool {
        if self.areas.contains_key(&id) {
            return false;
        }
        self.areas.insert(id, area);
        if !area.is_empty() {
            for cell in self.cells_of(area) {
                self.cells.entry(cell).or_default().push(id);
            }
        }
        true
    }

    /// Remove an area. Returns `false` if it was not tracked.
    pub fn remove(&mut self, id: AreaId) -> bool {
        let Some(area) = self.areas.remove(&id) else {
            return false;
        };
        if !area.is_empty() {
            for cell in self.cells_of(area) {
                self.remove_from_cell(cell, id);
            }
        }
        true
    }

    /// Move or resize an area. Returns `false` if it was not tracked.
    ///
    /// Cheaper than remove+insert only in that the id mapping survives; every
    /// overlapping cell of both boxes is still visited.
    pub fn update(&mut self, id: AreaId, area: Box3i) -> bool {
        if self.remove(id) {
            self.insert(id, area)
        } else {
            false
        }
    }

    /// Visit every tracked area whose box intersects `query`, in ascending id
    /// order. Empty query boxes match nothing.
    pub fn for_each_area_in_box(&self, query: Box3i, mut visit: impl FnMut(AreaId, Box3i)) {
        if query.is_empty() {
            return;
        }
        let mut matches: Vec<AreaId> = Vec::new();
        for cell in self.cells_of(query) {
            if let Some(ids) = self.cells.get(&cell) {
                for &id in ids {
                    if self
                        .areas
                        .get(&id)
                        .is_some_and(|area| area.intersects(&query))
                        && !matches.contains(&id)
                    {
                        matches.push(id);
                    }
                }
            }
        }
        matches.sort_unstable();
        for id in matches {
            // The finder is immutable during the query; the entry cannot
            // disappear between collection and this read.
            if let Some(area) = self.areas.get(&id) {
                visit(id, *area);
            }
        }
    }

    /// [`VoxelAreaFinder::for_each_area_in_box`] as a vector, for tests and
    /// callers that need ownership.
    pub fn areas_in_box(&self, query: Box3i) -> Vec<(AreaId, Box3i)> {
        let mut out = Vec::new();
        self.for_each_area_in_box(query, |id, area| out.push((id, area)));
        out
    }

    /// Cells covered by a non-empty half-open box, in xyz order. Returned as
    /// a vector so callers may mutate `self` while iterating.
    fn cells_of(&self, area: Box3i) -> Vec<CellKey> {
        debug_assert!(!area.is_empty());
        let min = area.position;
        let max = area.position + area.size;
        let to_cell = |v: i32| v.div_euclid(self.cell_size);
        // Half-open box: the last covered coordinate is max-1, which may be
        // negative, so the endpoint cell uses (max-1) too. An axis of size
        // zero cannot happen here (is_empty check upstream).
        let x0 = to_cell(min.x);
        let x1 = to_cell(max.x - 1);
        let y0 = to_cell(min.y);
        let y1 = to_cell(max.y - 1);
        let z0 = to_cell(min.z);
        let z1 = to_cell(max.z - 1);
        let mut cells = Vec::with_capacity(
            (x1 - x0 + 1).unsigned_abs() as usize
                * (y1 - y0 + 1).unsigned_abs() as usize
                * (z1 - z0 + 1).unsigned_abs() as usize,
        );
        for x in x0..=x1 {
            for y in y0..=y1 {
                for z in z0..=z1 {
                    cells.push(CellKey { x, y, z });
                }
            }
        }
        cells
    }

    fn remove_from_cell(&mut self, cell: CellKey, id: AreaId) {
        let Some(ids) = self.cells.get_mut(&cell) else {
            return;
        };
        ids.retain(|&other| other != id);
        if ids.is_empty() {
            self.cells.remove(&cell);
        }
    }
}

/// Subtract `hole` from `whole` and return the remaining boxes (0 to 6
/// axis-aligned slabs, xyz order). Used to diff a viewer's interest box
/// before/after a move: `box_subtraction(old, new)` is the set of block boxes
/// that *left* the interest area, and `box_subtraction(new, old)` the set
/// that *entered* it.
///
/// Returns a single unchanged `whole` when the boxes are disjoint, and
/// nothing when `hole` covers `whole`.
pub fn box_subtraction(whole: Box3i, hole: Box3i) -> Vec<Box3i> {
    let overlap = whole.clipped(hole);
    if overlap.is_empty() {
        return vec![whole];
    }
    if overlap == whole {
        return Vec::new();
    }

    let mut out = Vec::new();
    // Slab split along x, then y, then z around the overlap. Each slab keeps
    // the full extent on later axes; sizes are always positive by clipping.
    let w_min = whole.position;
    let w_max = whole.position + whole.size;
    let o_min = overlap.position;
    let o_max = overlap.position + overlap.size;
    let push = |out: &mut Vec<Box3i>, min: Vector3i, max: Vector3i| {
        let size = max - min;
        if size.x > 0 && size.y > 0 && size.z > 0 {
            out.push(Box3i::new(min, size));
        }
    };

    // Left/right x slabs spanning full y/z.
    push(
        &mut out,
        Vector3i::new(w_min.x, w_min.y, w_min.z),
        Vector3i::new(o_min.x, w_max.y, w_max.z),
    );
    push(
        &mut out,
        Vector3i::new(o_max.x, w_min.y, w_min.z),
        Vector3i::new(w_max.x, w_max.y, w_max.z),
    );
    // Bottom/top y slabs spanning the overlap's x range and full z.
    push(
        &mut out,
        Vector3i::new(o_min.x, w_min.y, w_min.z),
        Vector3i::new(o_max.x, o_min.y, w_max.z),
    );
    push(
        &mut out,
        Vector3i::new(o_min.x, o_max.y, w_min.z),
        Vector3i::new(o_max.x, w_max.y, w_max.z),
    );
    // Back/front z slabs spanning the overlap's x/y range.
    push(
        &mut out,
        Vector3i::new(o_min.x, o_min.y, w_min.z),
        Vector3i::new(o_max.x, o_max.y, o_min.z),
    );
    push(
        &mut out,
        Vector3i::new(o_min.x, o_min.y, o_max.z),
        Vector3i::new(o_max.x, o_max.y, w_max.z),
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn box_at(x: i32, y: i32, z: i32, w: i32, h: i32, d: i32) -> Box3i {
        Box3i::new(Vector3i::new(x, y, z), Vector3i::new(w, h, d))
    }

    #[test]
    fn new_rejects_non_positive_cell_size() {
        let result = std::panic::catch_unwind(|| VoxelAreaFinder::new(0));
        assert!(result.is_err());
    }

    #[test]
    fn insert_query_remove_round_trip() {
        let mut finder = VoxelAreaFinder::new(8);
        assert!(finder.insert(1, box_at(0, 0, 0, 16, 16, 16)));
        assert_eq!(finder.len(), 1);
        assert_eq!(finder.area(1), Some(box_at(0, 0, 0, 16, 16, 16)));

        // Inside.
        assert_eq!(
            finder.areas_in_box(box_at(4, 4, 4, 2, 2, 2)),
            vec![(1, box_at(0, 0, 0, 16, 16, 16))]
        );
        // Touching from outside (half-open boxes) matches nothing.
        assert!(finder.areas_in_box(box_at(16, 0, 0, 4, 4, 4)).is_empty());
        assert!(finder.areas_in_box(box_at(-4, 0, 0, 4, 4, 4)).is_empty());
        // Disjoint.
        assert!(finder
            .areas_in_box(box_at(100, 100, 100, 4, 4, 4))
            .is_empty());
        // Empty query matches nothing.
        assert!(finder.areas_in_box(box_at(0, 0, 0, 0, 4, 4)).is_empty());

        assert!(finder.remove(1));
        assert!(finder.is_empty());
        assert!(finder.cells.is_empty(), "cells must not leak");
        assert!(!finder.remove(1));
    }

    #[test]
    fn duplicate_insert_is_rejected() {
        let mut finder = VoxelAreaFinder::new(8);
        assert!(finder.insert(7, box_at(0, 0, 0, 4, 4, 4)));
        assert!(!finder.insert(7, box_at(100, 0, 0, 4, 4, 4)));
        assert_eq!(finder.area(7), Some(box_at(0, 0, 0, 4, 4, 4)));
    }

    #[test]
    fn query_reports_ascending_ids_and_deduplicates() {
        let mut finder = VoxelAreaFinder::new(4);
        // All three overlap the central query box and each other's cells.
        finder.insert(30, box_at(-8, -8, -8, 24, 24, 24));
        finder.insert(10, box_at(0, 0, 0, 8, 8, 8));
        finder.insert(20, box_at(4, 4, 4, 8, 8, 8));

        let hits = finder.areas_in_box(box_at(2, 2, 2, 4, 4, 4));
        let ids: Vec<AreaId> = hits.iter().map(|(id, _)| *id).collect();
        assert_eq!(ids, vec![10, 20, 30], "ids must be sorted and unique");
        for (_, area) in hits {
            assert!(area.intersects(&box_at(2, 2, 2, 4, 4, 4)));
        }
    }

    #[test]
    fn update_moves_the_area_and_cleans_old_cells() {
        let mut finder = VoxelAreaFinder::new(8);
        assert!(finder.insert(1, box_at(0, 0, 0, 8, 8, 8)));
        assert!(finder.update(1, box_at(64, 64, 64, 8, 8, 8)));

        assert_eq!(finder.area(1), Some(box_at(64, 64, 64, 8, 8, 8)));
        assert!(finder.areas_in_box(box_at(0, 0, 0, 8, 8, 8)).is_empty());
        assert_eq!(
            finder.areas_in_box(box_at(64, 64, 64, 8, 8, 8)),
            vec![(1, box_at(64, 64, 64, 8, 8, 8))]
        );
        // No stale cells from the old position.
        assert_eq!(finder.cells.len(), 1);
        assert!(!finder.update(2, box_at(0, 0, 0, 1, 1, 1)));
    }

    #[test]
    fn negative_coordinates_index_correctly() {
        let mut finder = VoxelAreaFinder::new(8);
        finder.insert(1, box_at(-24, -24, -24, 16, 16, 16));
        // Query entirely inside negative cells.
        assert_eq!(finder.areas_in_box(box_at(-20, -20, -20, 4, 4, 4)).len(), 1);
        // Query crossing the origin boundary does not match.
        assert!(finder.areas_in_box(box_at(0, 0, 0, 4, 4, 4)).is_empty());
        // A second area on the positive side is independent; a query spanning
        // the gap hits both.
        finder.insert(2, box_at(0, 0, 0, 8, 8, 8));
        let ids: Vec<AreaId> = finder
            .areas_in_box(box_at(-10, -10, -10, 20, 20, 20))
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        assert_eq!(ids, vec![1, 2]);
    }

    #[test]
    fn large_area_spans_many_cells_without_duplicates() {
        let mut finder = VoxelAreaFinder::new(4);
        finder.insert(1, box_at(0, 0, 0, 64, 64, 64)); // 16^3 cells
        assert_eq!(finder.cells.len(), 16 * 16 * 16);
        let hits = finder.areas_in_box(box_at(0, 0, 0, 64, 64, 64));
        assert_eq!(hits, vec![(1, box_at(0, 0, 0, 64, 64, 64))]);
    }

    #[test]
    fn empty_areas_are_stored_but_never_match() {
        let mut finder = VoxelAreaFinder::new(8);
        assert!(finder.insert(1, box_at(5, 5, 5, 0, 8, 8)));
        assert!(finder.cells.is_empty());
        assert!(finder
            .areas_in_box(box_at(0, 0, 0, 100, 100, 100))
            .is_empty());
        // An empty area can still be updated into a real one.
        assert!(finder.update(1, box_at(0, 0, 0, 4, 4, 4)));
        assert_eq!(finder.areas_in_box(box_at(0, 0, 0, 1, 1, 1)).len(), 1);
    }

    // ------------------------------------------------------------------
    // box_subtraction
    // ------------------------------------------------------------------

    fn volume(boxes: &[Box3i]) -> i64 {
        boxes
            .iter()
            .map(|b| b.size.volume_u64() as i64)
            .sum::<i64>()
    }

    #[test]
    fn subtraction_disjoint_returns_whole() {
        let whole = box_at(0, 0, 0, 8, 8, 8);
        let out = box_subtraction(whole, box_at(100, 100, 100, 4, 4, 4));
        assert_eq!(out, vec![whole]);
    }

    #[test]
    fn subtraction_covered_returns_nothing() {
        let out = box_subtraction(box_at(0, 0, 0, 8, 8, 8), box_at(-4, -4, -4, 16, 16, 16));
        assert!(out.is_empty());
    }

    #[test]
    fn subtraction_equal_returns_nothing() {
        let out = box_subtraction(box_at(2, 2, 2, 8, 8, 8), box_at(2, 2, 2, 8, 8, 8));
        assert!(out.is_empty());
    }

    #[test]
    fn subtraction_slab_volumes_add_up_and_do_not_overlap() {
        let whole = box_at(-8, -8, -8, 24, 24, 24);
        let hole = box_at(0, 0, 0, 8, 8, 8); // centered octant
        let out = box_subtraction(whole, hole);

        assert_eq!(
            volume(&out),
            whole.size.volume_u64() as i64 - hole.size.volume_u64() as i64
        );
        // No result box may intersect the hole.
        for b in &out {
            assert!(!b.intersects(&hole), "{b:?} intersects the hole");
        }
        // Sampling: no remaining voxel lies inside the hole.
        for x in -8..16 {
            for y in -8..16 {
                for z in -8..16 {
                    let p = Vector3i::new(x, y, z);
                    let in_hole = hole.contains_point(p);
                    let in_rest = out.iter().any(|b| b.contains_point(p));
                    assert_ne!(
                        in_hole, in_rest,
                        "voxel {p:?} must be in the hole or the remainder, not both/neither"
                    );
                }
            }
        }
    }

    #[test]
    fn subtraction_partial_edge_overlap() {
        // Hole overlaps one face of the whole: a single slab must remain.
        let whole = box_at(0, 0, 0, 8, 8, 8);
        let hole = box_at(4, 0, 0, 8, 8, 8);
        let out = box_subtraction(whole, hole);
        assert_eq!(out, vec![box_at(0, 0, 0, 4, 8, 8)]);
    }
}
