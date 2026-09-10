//! Modifier stack + sphere modifier. Ports `modifiers/voxel_modifier*.{h,cpp}`.
//!
//! MVP: `VoxelModifier` trait + `SphereModifier` + `ModifierStack` with the
//! SoA (slice-of-f32) apply path. The `VoxelBuffer` decompression apply path
//! is deferred.

use crate::math::Vector3f;

/// Per-voxel context passed to `VoxelModifier::apply`.
pub struct ModifierContext<'a> {
    pub sdf: &'a mut [f32],
    pub positions: &'a [Vector3f],
}

/// SDF blend operation. Matches C++ `VoxelModifierSdf::Operation`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SdfOperation {
    /// Smooth union: `min(a, b)` with optional smoothing.
    Add,
    /// Smooth subtract: `max(a, -b)` with optional smoothing.
    Subtract,
}

/// A voxel modifier: post-generation SDF transform.
pub trait VoxelModifier: Send + Sync {
    /// Apply this modifier to the given SDF values + world positions.
    fn apply(&self, ctx: &mut ModifierContext<'_>);
}

/// A sphere SDF modifier. Computes `length(pos - center) - radius` and blends
/// it into the existing SDF using smooth union or subtract.
pub struct SphereModifier {
    pub center: Vector3f,
    pub radius: f32,
    pub operation: SdfOperation,
    pub smoothness: f32,
}

impl VoxelModifier for SphereModifier {
    fn apply(&self, ctx: &mut ModifierContext<'_>) {
        let r2 = self.radius * self.radius;
        for (sdf, pos) in ctx.sdf.iter_mut().zip(ctx.positions.iter()) {
            let dx = pos.x - self.center.x;
            let dy = pos.y - self.center.y;
            let dz = pos.z - self.center.z;
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            let shape_sdf = dist - self.radius;
            *sdf = sdf_blend(*sdf, shape_sdf, self.operation, self.smoothness);
            let _ = r2; // suppress unused warning
        }
    }
}

/// An ordered stack of modifiers applied in sequence.
pub struct ModifierStack {
    modifiers: Vec<Box<dyn VoxelModifier>>,
}

impl Default for ModifierStack {
    fn default() -> Self {
        Self::new()
    }
}

impl ModifierStack {
    pub fn new() -> Self {
        Self {
            modifiers: Vec::new(),
        }
    }

    pub fn add(&mut self, modifier: Box<dyn VoxelModifier>) {
        self.modifiers.push(modifier);
    }

    pub fn len(&self) -> usize {
        self.modifiers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.modifiers.is_empty()
    }

    /// Apply all modifiers in sequence to the given SDF slice + positions.
    pub fn apply(&self, sdf: &mut [f32], positions: &[Vector3f]) {
        for modifier in &self.modifiers {
            let mut ctx = ModifierContext { sdf, positions };
            modifier.apply(&mut ctx);
        }
    }
}

/// Smooth SDF blending. Matches `util/math/sdf.h` smooth_union / smooth_subtract.
/// Public so binding layers blend with the exact same math as the core
/// modifier stack instead of re-implementing it.
pub fn sdf_blend(existing: f32, shape: f32, op: SdfOperation, smoothness: f32) -> f32 {
    if smoothness <= 0.0 {
        // Hard blend (no smoothing).
        return match op {
            SdfOperation::Add => existing.min(shape),
            SdfOperation::Subtract => existing.max(-shape),
        };
    }
    match op {
        SdfOperation::Add => crate::math::sdf::sdf_smooth_union(existing, shape, smoothness),
        SdfOperation::Subtract => {
            crate::math::sdf::sdf_smooth_subtract(existing, shape, smoothness)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sphere_modifier_add_creates_solid() {
        let modifier = SphereModifier {
            center: Vector3f::new(5.0, 5.0, 5.0),
            radius: 3.0,
            operation: SdfOperation::Add,
            smoothness: 0.0,
        };
        // Start fully outside (air).
        let mut sdf = [100.0f32; 3];
        let positions = [
            Vector3f::new(5.0, 5.0, 5.0),    // center: inside sphere
            Vector3f::new(10.0, 10.0, 10.0), // far outside
            Vector3f::new(6.0, 5.0, 5.0),    // just inside
        ];
        let mut ctx = ModifierContext {
            sdf: &mut sdf,
            positions: &positions,
        };
        modifier.apply(&mut ctx);
        // Center should be solid (negative SDF).
        assert!(sdf[0] < 0.0, "center should be solid: {}", sdf[0]);
        // Far point should remain air.
        assert!(sdf[1] > 0.0, "far point should be air: {}", sdf[1]);
    }

    #[test]
    fn sphere_modifier_subtract_carves_hole() {
        let modifier = SphereModifier {
            center: Vector3f::new(0.0, 0.0, 0.0),
            radius: 5.0,
            operation: SdfOperation::Subtract,
            smoothness: 0.0,
        };
        // Start fully inside (solid).
        let mut sdf = [-100.0f32; 1];
        let positions = [Vector3f::new(0.0, 0.0, 0.0)];
        let mut ctx = ModifierContext {
            sdf: &mut sdf,
            positions: &positions,
        };
        modifier.apply(&mut ctx);
        // Center should now be carved (positive SDF).
        assert!(sdf[0] > 0.0, "center should be carved: {}", sdf[0]);
    }

    #[test]
    fn modifier_stack_applies_in_order() {
        let mut stack = ModifierStack::new();
        stack.add(Box::new(SphereModifier {
            center: Vector3f::new(0.0, 0.0, 0.0),
            radius: 10.0,
            operation: SdfOperation::Add,
            smoothness: 0.0,
        }));
        stack.add(Box::new(SphereModifier {
            center: Vector3f::new(0.0, 0.0, 0.0),
            radius: 3.0,
            operation: SdfOperation::Subtract,
            smoothness: 0.0,
        }));
        let mut sdf = [100.0f32];
        let positions = [Vector3f::new(0.0, 0.0, 0.0)];
        stack.apply(&mut sdf, &positions);
        // First add makes it solid, then subtract carves the center.
        assert!(
            sdf[0] > 0.0,
            "should be carved after add+subtract: {}",
            sdf[0]
        );
    }

    #[test]
    fn smooth_union_matches_hard_at_zero_smoothness() {
        let a = -2.0;
        let b = 3.0;
        assert_eq!(sdf_blend(a, b, SdfOperation::Add, 0.0), a.min(b));
    }
}
