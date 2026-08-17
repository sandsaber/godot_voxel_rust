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
//! [`Box3i`] interest areas and answers box-intersection queries via a coarse
//! spatial hash instead of scanning every area. The replication design that
//! consumes it lives in `doc/source/multiplayer.md`.
//!
//! Port note: upstream C++ answered the same "which viewer boxes intersect
//! this box" query with a linear scan (`VoxelTerrain::get_viewers_in_area`,
//! used by the server-side `VoxelTerrainMultiplayerSynchronizer` for edit
//! fan-out). This module reimplements that query with a spatial hash,
//! deterministic result order, and a box-diff helper; it is not a literal
//! port of any single C++ class.

use crate::math::Box3i;
use std::collections::HashMap;

/// Identifier of a tracked interest area. Opaque here; callers typically use
/// network peer ids or viewer ids (the core's `ViewerId` is also `u32`).
pub type AreaId = u32;

/// Default upper bound on how many spatial-hash cells one area may cover.
/// A legitimate interest box is a viewer radius — a few hundred cells at
/// worst. The bound exists so that a peer-fed absurd box (a malformed
/// "position ± distance" on the server) is rejected with
/// [`AreaError::TooManyCells`] instead of enumerating billions of cells.
pub const MAX_CELLS_PER_AREA: u64 = 1 << 20;

/// Why an area could not be inserted or moved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AreaError {
    /// [`VoxelAreaFinder::insert`] was called with an id that is already
    /// tracked.
    DuplicateId,
    /// [`VoxelAreaFinder::update`] was called with an id that is not tracked.
    NotFound,
    /// The area covers more than `max_cells` spatial-hash cells (or its cell
    /// count is unrepresentable, e.g. a `position + size` corner overflowing
    /// `i32`). The finder is unchanged.
    TooManyCells,
}

/// Coarse spatial-hash cell coordinate. Cells are cubes of `cell_size` blocks;
/// their coordinates can be negative.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct CellKey {
    x: i64,
    y: i64,
    z: i64,
}

/// Half-open box corners in i64, so boxes whose `position + size` corner
/// would overflow i32 can still be tested exactly.
#[derive(Debug, Clone, Copy)]
struct WideBox {
    min: [i64; 3],
    max: [i64; 3], // exclusive
}

impl WideBox {
    fn from_box(b: Box3i) -> Self {
        WideBox {
            min: [
                b.position.x as i64,
                b.position.y as i64,
                b.position.z as i64,
            ],
            max: [
                b.position.x as i64 + b.size.x as i64,
                b.position.y as i64 + b.size.y as i64,
                b.position.z as i64 + b.size.z as i64,
            ],
        }
    }

    fn intersects(&self, other: &WideBox) -> bool {
        for a in 0..3 {
            if self.min[a] >= other.max[a] || other.min[a] >= self.max[a] {
                return false;
            }
        }
        true
    }
}

/// Index over axis-aligned interest areas answering "which areas intersect
/// this box?".
///
/// Areas are half-open boxes in block coordinates (`position` inclusive,
/// `position + size` exclusive), matching [`Box3i`] semantics everywhere else
/// in the core. Empty areas (any non-positive axis) never match and are not
/// indexed. Areas whose covered-cell count exceeds `max_cells` (default
/// [`MAX_CELLS_PER_AREA`]) are rejected with [`AreaError::TooManyCells`] —
/// the finder's defense against peer-derived absurd boxes; servers should
/// still clamp view distances before indexing them.
///
/// Query results are **deterministic**: intersecting areas are reported in
/// ascending id order, so network code that fans edits out over a finder
/// behaves identically run to run.
///
/// Complexity: insertion/removal/update touches every cell an area overlaps
/// (`O((box/cell_size)^3)`); a query touches the cells of the query box — or,
/// when that would be more work than scanning, every *occupied* cell — plus
/// one intersection test per matching area. Cost scales with the number of
/// matching areas and locally colocated ones, not with the total number of
/// tracked areas. Pick `cell_size` near the typical *query* box diameter
/// (e.g. one edit brush or one block), not the (much larger) viewer areas.
#[derive(Debug, Clone)]
pub struct VoxelAreaFinder {
    areas: HashMap<AreaId, Box3i>,
    cells: HashMap<CellKey, Vec<AreaId>>,
    cell_size: i32,
    max_cells: u64,
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
            max_cells: MAX_CELLS_PER_AREA,
        }
    }

    /// Override the per-area cell budget. Mostly for tests; see
    /// [`MAX_CELLS_PER_AREA`].
    pub fn with_max_cells(mut self, max_cells: u64) -> Self {
        self.max_cells = max_cells;
        self
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

    /// Insert a new area. Errors (leaving the finder unchanged) if `id` is
    /// already in use — moves and resizes go through
    /// [`VoxelAreaFinder::update`].
    pub fn insert(&mut self, id: AreaId, area: Box3i) -> Result<(), AreaError> {
        if self.areas.contains_key(&id) {
            return Err(AreaError::DuplicateId);
        }
        if !area.is_empty() {
            let cells = self.validated_cells(area)?;
            for cell in cells {
                self.cells.entry(cell).or_default().push(id);
            }
        }
        self.areas.insert(id, area);
        Ok(())
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

    /// Move or resize an area. Equivalent to `remove` + `insert`; provided as
    /// a single call so callers cannot observe the intermediate removed
    /// state. Errors (leaving the *old* area in place) if the id is unknown
    /// or the new area exceeds the cell budget.
    pub fn update(&mut self, id: AreaId, area: Box3i) -> Result<(), AreaError> {
        // Validate before mutating so a rejection never drops the old area.
        let cells = if area.is_empty() {
            Vec::new()
        } else {
            self.validated_cells(area)?
        };
        if !self.remove(id) {
            return Err(AreaError::NotFound);
        }
        for cell in cells {
            self.cells.entry(cell).or_default().push(id);
        }
        self.areas.insert(id, area);
        Ok(())
    }

    /// Visit every tracked area whose box intersects `query`, in ascending id
    /// order. Empty query boxes match nothing.
    pub fn for_each_area_in_box(&self, query: Box3i, mut visit: impl FnMut(AreaId, Box3i)) {
        if query.is_empty() {
            return;
        }
        let mut matches: Vec<AreaId> = Vec::new();
        let wide_query = WideBox::from_box(query);
        if self.query_cell_count(query) > self.cells.len() as u64 {
            // The query box is huge relative to the occupied grid (or its
            // extent is hostile): scan occupied cells instead of enumerating
            // the query's cells.
            for (&cell, ids) in &self.cells {
                if wide_query.intersects(&self.cell_box(cell)) {
                    matches.extend_from_slice(ids);
                }
            }
        } else {
            for cell in self.cells_of(query) {
                if let Some(ids) = self.cells.get(&cell) {
                    matches.extend_from_slice(ids);
                }
            }
        }
        matches.sort_unstable();
        matches.dedup();
        for id in matches {
            // The finder is immutable during the query; the entry cannot
            // disappear between collection and this read. WideBox keeps the
            // precise filter exact even for boxes with wrapping i32 corners.
            if let Some(area) = self.areas.get(&id) {
                if WideBox::from_box(*area).intersects(&wide_query) {
                    visit(id, *area);
                }
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

    /// Number of cells a non-empty box would cover; `None` if that count
    /// overflows `u64` (which by construction also exceeds any sane budget).
    fn cell_count(&self, area: Box3i) -> Option<u64> {
        let wide = WideBox::from_box(area);
        let s = self.cell_size as i64;
        let span = |lo: i64, hi_exclusive: i64| -> Option<u64> {
            let a = lo.div_euclid(s);
            let b = (hi_exclusive - 1).div_euclid(s);
            (b - a + 1).try_into().ok()
        };
        let sx = span(wide.min[0], wide.max[0])?;
        let sy = span(wide.min[1], wide.max[1])?;
        let sz = span(wide.min[2], wide.max[2])?;
        sx.checked_mul(sy)?.checked_mul(sz)
    }

    /// Cells of an area after the budget check; `Err(TooManyCells)` otherwise.
    fn validated_cells(&self, area: Box3i) -> Result<Vec<CellKey>, AreaError> {
        match self.cell_count(area) {
            Some(count) if count <= self.max_cells => Ok(self.cells_of(area)),
            _ => Err(AreaError::TooManyCells),
        }
    }

    /// Upper bound on cells a query would enumerate, saturating at `u64::MAX`
    /// so hostile query boxes always take the occupied-cell path.
    fn query_cell_count(&self, query: Box3i) -> u64 {
        self.cell_count(query).unwrap_or(u64::MAX)
    }

    /// The box covered by one cell, as i64 corners.
    fn cell_box(&self, cell: CellKey) -> WideBox {
        let s = self.cell_size as i64;
        WideBox {
            min: [cell.x * s, cell.y * s, cell.z * s],
            max: [cell.x * s + s, cell.y * s + s, cell.z * s + s],
        }
    }

    /// Cells covered by a non-empty half-open box, in xyz order. Returned as
    /// a vector so callers may mutate `self` while iterating. Only called
    /// after [`VoxelAreaFinder::cell_count`] has bounded the result.
    fn cells_of(&self, area: Box3i) -> Vec<CellKey> {
        debug_assert!(!area.is_empty());
        let wide = WideBox::from_box(area);
        let s = self.cell_size as i64;
        // Half-open box: the last covered coordinate is max-1, which may be
        // negative, so the endpoint cell uses (max-1) too.
        let to_cell = |v: i64| v.div_euclid(s);
        let (x0, x1) = (to_cell(wide.min[0]), to_cell(wide.max[0] - 1));
        let (y0, y1) = (to_cell(wide.min[1]), to_cell(wide.max[1] - 1));
        let (z0, z1) = (to_cell(wide.min[2]), to_cell(wide.max[2] - 1));
        let mut cells = Vec::new();
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
/// axis-aligned slabs). Delegates to [`Box3i::difference`] after handling the
/// empty-hole case (a zero-extent hole "inside" the whole removes nothing,
/// which `intersects`-based `difference` alone would mishandle). Used to
/// diff a viewer's interest box before/after a move:
/// `box_subtraction(old, new)` is the set of block boxes that *left* the
/// interest area, and `box_subtraction(new, old)` the set that *entered* it.
///
/// Precondition: `whole` and `hole` must have representable corners
/// (`position + size` within `i32`); the slab arithmetic inside `difference`
/// is the faithful C++ port and assumes that, like the rest of the core's
/// box math.
pub fn box_subtraction(whole: Box3i, hole: Box3i) -> Vec<Box3i> {
    let overlap = whole.clipped(hole);
    if overlap.is_empty() {
        return vec![whole];
    }
    if overlap == whole {
        return Vec::new();
    }
    whole.difference(hole)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Vector3i;

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
        finder.insert(1, box_at(0, 0, 0, 16, 16, 16)).unwrap();
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
        finder.insert(7, box_at(0, 0, 0, 4, 4, 4)).unwrap();
        assert_eq!(
            finder.insert(7, box_at(100, 0, 0, 4, 4, 4)),
            Err(AreaError::DuplicateId)
        );
        assert_eq!(finder.area(7), Some(box_at(0, 0, 0, 4, 4, 4)));
    }

    #[test]
    fn update_of_unknown_id_is_rejected() {
        let mut finder = VoxelAreaFinder::new(8);
        assert_eq!(
            finder.update(9, box_at(0, 0, 0, 4, 4, 4)),
            Err(AreaError::NotFound)
        );
    }

    #[test]
    fn query_reports_ascending_ids_and_deduplicates() {
        let mut finder = VoxelAreaFinder::new(4);
        // All three overlap the central query box and each other's cells.
        finder.insert(30, box_at(-8, -8, -8, 24, 24, 24)).unwrap();
        finder.insert(10, box_at(0, 0, 0, 8, 8, 8)).unwrap();
        finder.insert(20, box_at(4, 4, 4, 8, 8, 8)).unwrap();

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
        finder.insert(1, box_at(0, 0, 0, 8, 8, 8)).unwrap();
        finder.update(1, box_at(64, 64, 64, 8, 8, 8)).unwrap();

        assert_eq!(finder.area(1), Some(box_at(64, 64, 64, 8, 8, 8)));
        assert!(finder.areas_in_box(box_at(0, 0, 0, 8, 8, 8)).is_empty());
        assert_eq!(
            finder.areas_in_box(box_at(64, 64, 64, 8, 8, 8)),
            vec![(1, box_at(64, 64, 64, 8, 8, 8))]
        );
        // No stale cells from the old position.
        assert_eq!(finder.cells.len(), 1);
    }

    #[test]
    fn negative_coordinates_index_correctly() {
        let mut finder = VoxelAreaFinder::new(8);
        finder.insert(1, box_at(-24, -24, -24, 16, 16, 16)).unwrap();
        // Query entirely inside negative cells.
        assert_eq!(finder.areas_in_box(box_at(-20, -20, -20, 4, 4, 4)).len(), 1);
        // Query crossing the origin boundary does not match.
        assert!(finder.areas_in_box(box_at(0, 0, 0, 4, 4, 4)).is_empty());
        // A second area on the positive side is independent; a query spanning
        // the gap hits both.
        finder.insert(2, box_at(0, 0, 0, 8, 8, 8)).unwrap();
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
        finder.insert(1, box_at(0, 0, 0, 64, 64, 64)).unwrap(); // 16^3 cells
        assert_eq!(finder.cells.len(), 16 * 16 * 16);
        let hits = finder.areas_in_box(box_at(0, 0, 0, 64, 64, 64));
        assert_eq!(hits, vec![(1, box_at(0, 0, 0, 64, 64, 64))]);
    }

    #[test]
    fn empty_areas_are_stored_but_never_match() {
        let mut finder = VoxelAreaFinder::new(8);
        finder.insert(1, box_at(5, 5, 5, 0, 8, 8)).unwrap();
        assert!(finder.cells.is_empty());
        assert!(finder
            .areas_in_box(box_at(0, 0, 0, 100, 100, 100))
            .is_empty());
        // An empty area can still be updated into a real one.
        finder.update(1, box_at(0, 0, 0, 4, 4, 4)).unwrap();
        assert_eq!(finder.areas_in_box(box_at(0, 0, 0, 1, 1, 1)).len(), 1);
    }

    #[test]
    fn hostile_area_boxes_are_rejected_not_enumerated() {
        let mut finder = VoxelAreaFinder::new(8);
        // i32-spanning box: ~2^84 cells. Must be rejected instantly.
        let absurd = Box3i::new(
            Vector3i::new(i32::MIN, i32::MIN, i32::MIN),
            Vector3i::new(i32::MAX, i32::MAX, i32::MAX),
        );
        assert_eq!(finder.insert(1, absurd), Err(AreaError::TooManyCells));
        assert!(finder.is_empty());
        assert!(finder.cells.is_empty());

        // Corner-overflow box (position + size wraps i32): also rejected via
        // the budget path instead of silently indexing nothing.
        let wrapping = Box3i::new(Vector3i::new(10, 0, 0), Vector3i::new(i32::MAX, 4, 4));
        assert_eq!(finder.insert(2, wrapping), Err(AreaError::TooManyCells));

        // A small budget rejects boxes that are large but representable.
        let mut strict = VoxelAreaFinder::new(1).with_max_cells(8);
        assert_eq!(
            strict.insert(1, box_at(0, 0, 0, 3, 3, 3)),
            Err(AreaError::TooManyCells) // 27 cells
        );
        assert!(strict.insert(1, box_at(0, 0, 0, 2, 2, 2)).is_ok()); // 8 cells
    }

    #[test]
    fn update_rejection_keeps_the_old_area() {
        let mut finder = VoxelAreaFinder::new(4).with_max_cells(8);
        finder.insert(1, box_at(0, 0, 0, 4, 4, 4)).unwrap();
        assert_eq!(
            finder.update(1, box_at(0, 0, 0, 64, 64, 64)),
            Err(AreaError::TooManyCells)
        );
        assert_eq!(finder.area(1), Some(box_at(0, 0, 0, 4, 4, 4)));
        assert_eq!(
            finder.cells.len(),
            1,
            "old cells must survive the rejection"
        );
    }

    #[test]
    fn huge_query_box_falls_back_to_occupied_cells() {
        // One tiny area, one absurd query: the occupied-cell path must answer
        // quickly instead of enumerating the query's cells.
        let mut finder = VoxelAreaFinder::new(8);
        finder.insert(1, box_at(0, 0, 0, 8, 8, 8)).unwrap();

        // A query with representable corners but billions of cells: the
        // occupied-cell path must answer instead of enumerating them.
        let absurd_query = box_at(
            -1_000_000_000,
            -1_000_000_000,
            -1_000_000_000,
            2_000_000_000,
            2_000_000_000,
            2_000_000_000,
        );
        assert_eq!(
            finder.areas_in_box(absurd_query),
            vec![(1, box_at(0, 0, 0, 8, 8, 8))]
        );

        // Huge-but-finite query that only overlaps through one occupied cell.
        let wide = box_at(-1_000_000, 0, 0, 2_000_000, 1, 1);
        assert_eq!(
            finder.areas_in_box(wide),
            vec![(1, box_at(0, 0, 0, 8, 8, 8))]
        );
        // And one that misses everything.
        let miss = box_at(-1_000_000, 900_000, 0, 2_000_000, 1, 1);
        assert!(finder.areas_in_box(miss).is_empty());
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
    fn subtraction_zero_extent_hole_inside_removes_nothing() {
        // `difference`'s intersects() gate mishandles zero-extent holes; our
        // clipped-empty gate must not.
        let whole = box_at(0, 0, 0, 8, 8, 8);
        let hole = box_at(4, 4, 4, 0, 4, 4);
        assert_eq!(box_subtraction(whole, hole), vec![whole]);
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
    fn subtraction_matches_box3i_difference() {
        // Cross-check against the faithful C++ port so the two cannot drift
        // (different slab shapes, same union).
        let cases = [
            (box_at(0, 0, 0, 8, 8, 8), box_at(4, 4, 4, 8, 8, 8)),
            (box_at(-6, -2, 0, 12, 8, 4), box_at(-4, 0, 1, 4, 4, 2)),
            (box_at(0, 0, 0, 10, 10, 10), box_at(3, 3, 3, 1, 1, 1)),
            (box_at(0, 0, 0, 8, 8, 8), box_at(-1, 2, 2, 16, 2, 2)),
        ];
        for (whole, hole) in cases {
            let ours = box_subtraction(whole, hole);
            let reference = whole.difference(hole);
            assert_eq!(volume(&ours), volume(&reference), "{whole:?} - {hole:?}");
            for x in -8..16 {
                for y in -8..16 {
                    for z in -8..16 {
                        let p = Vector3i::new(x, y, z);
                        assert_eq!(
                            ours.iter().any(|b| b.contains_point(p)),
                            reference.iter().any(|b| b.contains_point(p)),
                            "coverage differs at {p:?} for {whole:?} - {hole:?}"
                        );
                    }
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
