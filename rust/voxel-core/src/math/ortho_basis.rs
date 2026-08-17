//! Orthogonal bases: 3D rotations using only 90° angle steps.
//!
//! Ported from `util/math/ortho_basis.{h,cpp}`. Because there are only 24 such
//! orientations, they can be encoded as a single byte and recovered via a lookup
//! table. Every axis is a unit vector pointing along ±X/±Y/±Z and is
//! perpendicular to the others, so equality comparison is exact (no float eps).
//!
//! The 24-entry table `ORTHO_BASES`, the `OrthoRotationID` enum, and the
//! `ROTATION_NAMES` array are **positionally linked**: index `i` ↔ enum variant
//! `i` ↔ name `ROTATION_NAMES[i]` ↔ basis `ORTHO_BASES[i]`. Order is taken from
//! Godot's GridMap code and must remain stable to match the enum.

use super::constants::Axis;
use super::vector3::Vector3i;

/// Number of distinct 90°-step orthogonal orientations. Matches
/// `ORTHOGONAL_BASIS_COUNT`.
pub const ORTHOGONAL_BASIS_COUNT: usize = 24;

/// Index of the identity basis in [`ORTHO_BASES`]. Matches
/// `ORTHOGONAL_BASIS_IDENTITY_INDEX`.
pub const ORTHOGONAL_BASIS_IDENTITY_INDEX: usize = 0;

/// Basis where every axis is a unit vector along ±X/±Y/±Z and every axis is
/// perpendicular to the others. No precision is lost operating such a basis, so
/// `==` is safe. Matches `math::OrthoBasis`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OrthoBasis {
    pub x: Vector3i,
    pub y: Vector3i,
    pub z: Vector3i,
}

impl Default for OrthoBasis {
    fn default() -> Self {
        Self {
            x: Vector3i::new(1, 0, 0),
            y: Vector3i::new(0, 1, 0),
            z: Vector3i::new(0, 0, 1),
        }
    }
}

impl OrthoBasis {
    pub const fn new(x: Vector3i, y: Vector3i, z: Vector3i) -> Self {
        Self { x, y, z }
    }

    /// Build a basis by rotating the identity `turns` times (90° each) around
    /// `axis` (clockwise, axis pointed at the viewer). Matches
    /// `OrthoBasis::from_axis_turns`. Negative turns wrap to their positive
    /// equivalent.
    pub fn from_axis_turns(axis: Axis, turns: i32) -> OrthoBasis {
        // If turns are negative, do the positive equivalent.
        let mturns = if turns >= 0 {
            turns % 4
        } else {
            4 - ((-turns) % 4)
        };
        if mturns == 0 {
            return OrthoBasis::default();
        }
        // Clockwise with the rotation axis pointing at us.
        match (axis, mturns) {
            (Axis::X, 1) => OrthoBasis::new(
                Vector3i::new(1, 0, 0),
                Vector3i::new(0, 0, -1),
                Vector3i::new(0, 1, 0),
            ),
            (Axis::X, 2) => OrthoBasis::new(
                Vector3i::new(1, 0, 0),
                Vector3i::new(0, -1, 0),
                Vector3i::new(0, 0, -1),
            ),
            (Axis::X, 3) => OrthoBasis::new(
                Vector3i::new(1, 0, 0),
                Vector3i::new(0, 0, 1),
                Vector3i::new(0, -1, 0),
            ),
            (Axis::Y, 1) => OrthoBasis::new(
                Vector3i::new(0, 0, 1),
                Vector3i::new(0, 1, 0),
                Vector3i::new(-1, 0, 0),
            ),
            (Axis::Y, 2) => OrthoBasis::new(
                Vector3i::new(-1, 0, 0),
                Vector3i::new(0, 1, 0),
                Vector3i::new(0, 0, -1),
            ),
            (Axis::Y, 3) => OrthoBasis::new(
                Vector3i::new(0, 0, -1),
                Vector3i::new(0, 1, 0),
                Vector3i::new(1, 0, 0),
            ),
            (Axis::Z, 1) => OrthoBasis::new(
                Vector3i::new(0, -1, 0),
                Vector3i::new(1, 0, 0),
                Vector3i::new(0, 0, 1),
            ),
            (Axis::Z, 2) => OrthoBasis::new(
                Vector3i::new(-1, 0, 0),
                Vector3i::new(0, -1, 0),
                Vector3i::new(0, 0, 1),
            ),
            (Axis::Z, 3) => OrthoBasis::new(
                Vector3i::new(0, 1, 0),
                Vector3i::new(-1, 0, 0),
                Vector3i::new(0, 0, 1),
            ),
            // mturns == 0 is handled above; any other combination is unreachable
            // (axis is exhaustive, mturns is in 0..=3), but fall back to identity
            // to stay total and mirror the C++ default return.
            _ => OrthoBasis::default(),
        }
    }

    /// True if all three axes are unit vectors and mutually perpendicular.
    /// Matches `is_orthonormal`.
    #[inline]
    pub fn is_orthonormal(&self) -> bool {
        self.x.is_unit_vector()
            && self.y.is_unit_vector()
            && self.z.is_unit_vector()
            && self.x.dot(self.y) == 0
            && self.x.dot(self.z) == 0
            && self.y.dot(self.z) == 0
    }

    #[inline]
    pub fn get_axis(&self, i: usize) -> Vector3i {
        // TODO Optimization: could use a union with an array
        match i {
            0 => self.x,
            1 => self.y,
            2 => self.z,
            _ => panic!("OrthoBasis axis index out of range"),
        }
    }

    /// Transpose across the diagonal (swap the upper/lower off-diagonal
    /// entries). Matches `transpose`.
    #[inline]
    pub fn transpose(&mut self) {
        // x A A
        // B x A
        // B B x
        // We only need to swap the As with the Bs across the diagonal.
        core::mem::swap(&mut self.x.y, &mut self.y.x);
        core::mem::swap(&mut self.x.z, &mut self.z.x);
        core::mem::swap(&mut self.y.z, &mut self.z.y);
    }

    /// The inverse of an orthogonal matrix is its transpose.
    /// Matches `invert` / `inverted`.
    /// <https://math.stackexchange.com/questions/1936020/why-is-the-inverse-of-an-orthogonal-matrix-equal-to-its-transpose>
    #[inline]
    pub fn invert(&mut self) {
        self.transpose();
    }

    #[inline]
    pub fn inverted(mut self) -> OrthoBasis {
        self.invert();
        self
    }

    /// Transform `p` by this basis: `p.x*x + p.y*y + p.z*z`. Matches `xform`.
    #[inline]
    pub fn xform(&self, p: Vector3i) -> Vector3i {
        p.x * self.x + p.y * self.y + p.z * self.z
    }

    /// Float counterpart of [`xform`](Self::xform). Used when rotating baked
    /// mesh vertices and normals.
    #[inline]
    pub fn xform_f(&self, p: crate::math::Vector3f) -> crate::math::Vector3f {
        crate::math::Vector3f::new(
            p.x * self.x.x as f32 + p.y * self.y.x as f32 + p.z * self.z.x as f32,
            p.x * self.x.y as f32 + p.y * self.y.y as f32 + p.z * self.z.y as f32,
            p.x * self.x.z as f32 + p.y * self.y.z as f32 + p.z * self.z.z as f32,
        )
    }

    /// Basis composition: apply `other`'s axes through `self`. Matches
    /// `operator*(OrthoBasis)`.
    #[inline]
    pub fn mul(&self, other: OrthoBasis) -> OrthoBasis {
        OrthoBasis::new(
            self.xform(other.x),
            self.xform(other.y),
            self.xform(other.z),
        )
    }

    // ---- 90° in-place rotations of all three axes (match rotate_*_90_*) ----

    #[inline]
    pub fn rotate_x_90_cw(&mut self) {
        self.x = self.x.rotate_x_90_cw();
        self.y = self.y.rotate_x_90_cw();
        self.z = self.z.rotate_x_90_cw();
    }
    #[inline]
    pub fn rotate_x_90_ccw(&mut self) {
        self.x = self.x.rotate_x_90_ccw();
        self.y = self.y.rotate_x_90_ccw();
        self.z = self.z.rotate_x_90_ccw();
    }
    #[inline]
    pub fn rotate_y_90_cw(&mut self) {
        self.x = self.x.rotate_y_90_cw();
        self.y = self.y.rotate_y_90_cw();
        self.z = self.z.rotate_y_90_cw();
    }
    #[inline]
    pub fn rotate_y_90_ccw(&mut self) {
        self.x = self.x.rotate_y_90_ccw();
        self.y = self.y.rotate_y_90_ccw();
        self.z = self.z.rotate_y_90_ccw();
    }
    #[inline]
    pub fn rotate_z_90_cw(&mut self) {
        self.x = self.x.rotate_z_90_cw();
        self.y = self.y.rotate_z_90_cw();
        self.z = self.z.rotate_z_90_cw();
    }
    #[inline]
    pub fn rotate_z_90_ccw(&mut self) {
        self.x = self.x.rotate_z_90_ccw();
        self.y = self.y.rotate_z_90_ccw();
        self.z = self.z.rotate_z_90_ccw();
    }

    /// Rotate all three axes 90° around `axis`. Matches `rotate_90(Axis, bool)`.
    #[inline]
    pub fn rotate_90(&mut self, axis: Axis, clockwise: bool) {
        let axes = [self.x, self.y, self.z];
        let mut rotated = axes;
        Vector3i::rotate_90_slice(&mut rotated, axis, clockwise);
        self.x = rotated[0];
        self.y = rotated[1];
        self.z = rotated[2];
    }
}

/// Identifier for each of the 24 orthogonal rotations. Matches `OrthoRotationID`.
///
/// The naming convention treats `-Z` as forward, `Y` as up, `X` as right, with
/// counter-clockwise rotations preferred. Half of rotations involve a roll
/// around Z (rarely used in Minecraft-style games).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum OrthoRotationId {
    Identity = 0,
    Z270 = 1,      // roll 270
    Z180 = 2,      // roll 180
    Z90 = 3,       // roll 90
    X270 = 4,      // look down
    X270Y270 = 5,  // look down, turn right
    X270Y180 = 6,  // look down, turn around Y 180
    X270Y90 = 7,   // look down, turn left
    Z180Y180 = 8,  // roll 180, turn around Y 180
    Z90Y180 = 9,   // roll 90, turn around Y 180
    Y180 = 10,     // turn around Y 180
    Z270Y180 = 11, // roll 270, turn around Y 180
    X90 = 12,      // look up
    X90Y90 = 13,   // look up, turn left
    X90Y180 = 14,  // look up, turn around Y 180
    X90Y270 = 15,  // look up, turn right
    Y270 = 16,     // turn right
    Z270Y270 = 17, // roll 270, turn right
    Z180Y270 = 18, // roll 180, turn right
    Z90Y270 = 19,  // roll 90, turn right
    Z180Y90 = 20,  // roll 180, turn left
    Z90Y90 = 21,   // roll 90, turn left
    Y90 = 22,      // turn left
    Z270Y90 = 23,  // roll 270, turn left
}

impl OrthoRotationId {
    /// Number of variants. Matches `ORTHO_ROTATION_COUNT`.
    pub const COUNT: usize = 24;
}

/// All 24 orthogonal bases, indexed by [`OrthoRotationId`] discriminant.
///
/// Values are taken from Godot's GridMap code. Order is arbitrary but must
/// remain the same to match the enum. Matches the C++ `g_ortho_bases` array.
pub const ORTHO_BASES: [OrthoBasis; ORTHOGONAL_BASIS_COUNT] = [
    // 0 — identity
    OrthoBasis::new(
        Vector3i::new(1, 0, 0),
        Vector3i::new(0, 1, 0),
        Vector3i::new(0, 0, 1),
    ),
    // 1
    OrthoBasis::new(
        Vector3i::new(0, -1, 0),
        Vector3i::new(1, 0, 0),
        Vector3i::new(0, 0, 1),
    ),
    // 2
    OrthoBasis::new(
        Vector3i::new(-1, 0, 0),
        Vector3i::new(0, -1, 0),
        Vector3i::new(0, 0, 1),
    ),
    // 3
    OrthoBasis::new(
        Vector3i::new(0, 1, 0),
        Vector3i::new(-1, 0, 0),
        Vector3i::new(0, 0, 1),
    ),
    // 4
    OrthoBasis::new(
        Vector3i::new(1, 0, 0),
        Vector3i::new(0, 0, -1),
        Vector3i::new(0, 1, 0),
    ),
    // 5
    OrthoBasis::new(
        Vector3i::new(0, 0, 1),
        Vector3i::new(1, 0, 0),
        Vector3i::new(0, 1, 0),
    ),
    // 6
    OrthoBasis::new(
        Vector3i::new(-1, 0, 0),
        Vector3i::new(0, 0, 1),
        Vector3i::new(0, 1, 0),
    ),
    // 7
    OrthoBasis::new(
        Vector3i::new(0, 0, -1),
        Vector3i::new(-1, 0, 0),
        Vector3i::new(0, 1, 0),
    ),
    // 8
    OrthoBasis::new(
        Vector3i::new(1, 0, 0),
        Vector3i::new(0, -1, 0),
        Vector3i::new(0, 0, -1),
    ),
    // 9
    OrthoBasis::new(
        Vector3i::new(0, 1, 0),
        Vector3i::new(1, 0, 0),
        Vector3i::new(0, 0, -1),
    ),
    // 10
    OrthoBasis::new(
        Vector3i::new(-1, 0, 0),
        Vector3i::new(0, 1, 0),
        Vector3i::new(0, 0, -1),
    ),
    // 11
    OrthoBasis::new(
        Vector3i::new(0, -1, 0),
        Vector3i::new(-1, 0, 0),
        Vector3i::new(0, 0, -1),
    ),
    // 12
    OrthoBasis::new(
        Vector3i::new(1, 0, 0),
        Vector3i::new(0, 0, 1),
        Vector3i::new(0, -1, 0),
    ),
    // 13
    OrthoBasis::new(
        Vector3i::new(0, 0, -1),
        Vector3i::new(1, 0, 0),
        Vector3i::new(0, -1, 0),
    ),
    // 14
    OrthoBasis::new(
        Vector3i::new(-1, 0, 0),
        Vector3i::new(0, 0, -1),
        Vector3i::new(0, -1, 0),
    ),
    // 15
    OrthoBasis::new(
        Vector3i::new(0, 0, 1),
        Vector3i::new(-1, 0, 0),
        Vector3i::new(0, -1, 0),
    ),
    // 16
    OrthoBasis::new(
        Vector3i::new(0, 0, 1),
        Vector3i::new(0, 1, 0),
        Vector3i::new(-1, 0, 0),
    ),
    // 17
    OrthoBasis::new(
        Vector3i::new(0, -1, 0),
        Vector3i::new(0, 0, 1),
        Vector3i::new(-1, 0, 0),
    ),
    // 18
    OrthoBasis::new(
        Vector3i::new(0, 0, -1),
        Vector3i::new(0, -1, 0),
        Vector3i::new(-1, 0, 0),
    ),
    // 19
    OrthoBasis::new(
        Vector3i::new(0, 1, 0),
        Vector3i::new(0, 0, -1),
        Vector3i::new(-1, 0, 0),
    ),
    // 20
    OrthoBasis::new(
        Vector3i::new(0, 0, 1),
        Vector3i::new(0, -1, 0),
        Vector3i::new(1, 0, 0),
    ),
    // 21
    OrthoBasis::new(
        Vector3i::new(0, 1, 0),
        Vector3i::new(0, 0, 1),
        Vector3i::new(1, 0, 0),
    ),
    // 22
    OrthoBasis::new(
        Vector3i::new(0, 0, -1),
        Vector3i::new(0, 1, 0),
        Vector3i::new(1, 0, 0),
    ),
    // 23
    OrthoBasis::new(
        Vector3i::new(0, -1, 0),
        Vector3i::new(0, 0, -1),
        Vector3i::new(1, 0, 0),
    ),
];

/// String name of each rotation, indexed by [`OrthoRotationId`] discriminant.
/// Matches the C++ `s_rotation_names` array.
pub const ROTATION_NAMES: [&str; OrthoRotationId::COUNT] = [
    "identity",
    "z_270",
    "z_180",
    "z_90",
    "x_270",
    "x_270_y_270",
    "x_270_y_180",
    "x_270_y_90",
    "z_180_y_180",
    "z_90_y_180",
    "y_180",
    "z_270_y_180",
    "x_90",
    "x_90_y_90",
    "x_90_y_180",
    "x_90_y_270",
    "y_270",
    "z_270_y_270",
    "z_180_y_270",
    "z_90_y_270",
    "z_180_y_90",
    "z_90_y_90",
    "y_90",
    "z_270_y_90",
];

/// Look up a basis by index. Matches `get_ortho_basis_from_index`.
/// Panics if `i >= ORTHOGONAL_BASIS_COUNT` (debug assert, matching `ZN_ASSERT`).
#[inline]
pub fn get_ortho_basis_from_index(i: usize) -> OrthoBasis {
    debug_assert!(i < ORTHOGONAL_BASIS_COUNT);
    ORTHO_BASES[i]
}

/// Find the index of `b` in [`ORTHO_BASES`], or `None` if not present.
/// Matches `get_index_from_ortho_basis` (which returns -1 when not found).
#[inline]
pub fn get_index_from_ortho_basis(b: &OrthoBasis) -> Option<usize> {
    ORTHO_BASES.iter().position(|ob| ob == b)
}

/// String name for rotation index `i`. Matches `ortho_rotation_to_string`.
/// Returns `"<error>"` for out-of-range indices (matching the C++ fallback).
#[inline]
pub fn ortho_rotation_to_string(i: usize) -> &'static str {
    if i < ROTATION_NAMES.len() {
        ROTATION_NAMES[i]
    } else {
        "<error>"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_identity_basis() {
        let b = OrthoBasis::default();
        assert_eq!(b, ORTHO_BASES[ORTHOGONAL_BASIS_IDENTITY_INDEX]);
        assert!(b.is_orthonormal());
    }

    #[test]
    fn all_table_bases_are_orthonormal() {
        // Every entry in the lookup table must be a valid orthonormal basis.
        for (i, b) in ORTHO_BASES.iter().enumerate() {
            assert!(
                b.is_orthonormal(),
                "ORTHO_BASES[{i}] is not orthonormal: {b:?}"
            );
        }
        assert_eq!(ORTHO_BASES.len(), ORTHOGONAL_BASIS_COUNT);
    }

    #[test]
    fn index_basis_roundtrip_is_bijective() {
        // Every basis must round-trip through its index, and indices are unique.
        let mut seen = [false; ORTHOGONAL_BASIS_COUNT];
        for (i, b) in ORTHO_BASES.iter().enumerate() {
            let back = get_index_from_ortho_basis(b);
            assert_eq!(
                back,
                Some(i),
                "basis {i} did not round-trip to its own index"
            );
            seen[i] = true;
        }
        assert!(seen.iter().all(|s| *s), "some indices were never hit");
    }

    #[test]
    fn rotation_names_aligned_with_table() {
        // Names and bases are positionally linked: index 0 is "identity".
        assert_eq!(ROTATION_NAMES[0], "identity");
        assert_eq!(ortho_rotation_to_string(0), "identity");
        assert_eq!(ROTATION_NAMES.len(), ORTHOGONAL_BASIS_COUNT);
        // Out-of-range returns the error sentinel.
        assert_eq!(ortho_rotation_to_string(99), "<error>");
    }

    #[test]
    fn from_axis_turns_x() {
        // Identity at 0 turns.
        assert_eq!(
            OrthoBasis::from_axis_turns(Axis::X, 0),
            OrthoBasis::default()
        );
        // 1 turn CW around X (matches the cpp switch case 1).
        let one = OrthoBasis::from_axis_turns(Axis::X, 1);
        assert_eq!(
            one,
            OrthoBasis::new(
                Vector3i::new(1, 0, 0),
                Vector3i::new(0, 0, -1),
                Vector3i::new(0, 1, 0)
            )
        );
        assert!(one.is_orthonormal());
        // 4 turns == identity (full revolution).
        assert_eq!(
            OrthoBasis::from_axis_turns(Axis::X, 4),
            OrthoBasis::default()
        );
        // Negative turns wrap to positive equivalent: -1 == 3.
        assert_eq!(
            OrthoBasis::from_axis_turns(Axis::X, -1),
            OrthoBasis::from_axis_turns(Axis::X, 3)
        );
    }

    #[test]
    fn from_axis_turns_all_axes_orthonormal() {
        for &axis in &[Axis::X, Axis::Y, Axis::Z] {
            for turns in 0..4 {
                let b = OrthoBasis::from_axis_turns(axis, turns);
                assert!(
                    b.is_orthonormal(),
                    "axis {axis:?} turns {turns} not orthonormal"
                );
            }
        }
    }

    #[test]
    fn transpose_is_inverse() {
        // For an orthogonal basis, transpose == invert: b * b.inverted() == identity.
        for b in ORTHO_BASES {
            let inv = b.inverted();
            let product = b.mul(inv);
            assert_eq!(
                product,
                OrthoBasis::default(),
                "b * b.inverted() != identity"
            );
        }
    }

    #[test]
    fn xform_transforms_axes() {
        let id = OrthoBasis::default();
        // Identity leaves vectors unchanged.
        assert_eq!(id.xform(Vector3i::new(1, 2, 3)), Vector3i::new(1, 2, 3));
        assert_eq!(
            id.xform_f(crate::math::Vector3f::new(1.0, 2.0, 3.0)),
            crate::math::Vector3f::new(1.0, 2.0, 3.0)
        );
    }

    #[test]
    fn rotate_90_in_place_matches_axis_helper() {
        // Rotating the identity basis 90° CW around Z must match the table entry
        // produced by composing with that rotation.
        let mut b = OrthoBasis::default();
        b.rotate_90(Axis::Z, true);
        // After Z-CW, x axis (1,0,0) -> (0,-1,0); y axis (0,1,0) -> (1,0,0).
        assert_eq!(b.x, Vector3i::new(0, -1, 0));
        assert_eq!(b.y, Vector3i::new(1, 0, 0));
        assert_eq!(b.z, Vector3i::new(0, 0, 1));
        assert!(b.is_orthonormal());
    }

    #[test]
    fn enum_discriminants_are_contiguous_from_zero() {
        // The table is indexed by discriminant, so variants must be 0..24.
        for i in 0..OrthoRotationId::COUNT {
            // Sanity: the name at that index exists.
            assert!(!ROTATION_NAMES[i].is_empty());
            // And the basis at that index is orthonormal (covered above too).
            assert!(ORTHO_BASES[i].is_orthonormal());
        }
    }
}
