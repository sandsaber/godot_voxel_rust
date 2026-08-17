//! Region copy/fill/transform helpers and snorm quantization.
//!
//! Ported from `storage/funcs.{h,cpp}`. These are the low-level 3D-array
//! operations used by `VoxelBuffer` (and elsewhere) to move sub-regions between
//! flat ZXY-laid-out buffers, fill rectangular regions, and rotate/flip volumes
//! via an [`OrthoBasis`]. Also includes the signed-normalized (`snorm`)
//! int↔float conversions used to pack SDF data into 8/16-bit channels.

use crate::math::funcs;
use crate::math::ortho_basis::OrthoBasis;
use crate::math::Vector3i;

/// Clip a 1D copy region so both source and destination stay in bounds. Mutates
/// `src_min`/`src_max`/`dst_min` in place. The resulting `[src_min, src_max)`
/// range may be empty or inverted (negative size) — the caller must check.
/// Matches `clip_copy_region_coord`.
fn clip_copy_region_coord(
    src_min: &mut i32,
    src_max: &mut i32,
    src_size: i32,
    dst_min: &mut i32,
    dst_size: i32,
) {
    // Clamp source and shrink destination for moved borders.
    if *src_min < 0 {
        *dst_min += -*src_min;
        *src_min = 0;
    }
    if *src_max > src_size {
        *src_max = src_size;
    }
    // Clamp destination and shrink source for moved borders.
    if *dst_min < 0 {
        *src_min += -*dst_min;
        *dst_min = 0;
    }
    let dst_w = *src_max - *src_min;
    let dst_max = *dst_min + dst_w;
    if dst_max > dst_size {
        *src_max -= dst_max - dst_size;
    }
}

/// 3D version of [`clip_copy_region_coord`]. Mutates `src_min`/`src_max`/`dst_min`
/// per-axis. Matches `clip_copy_region`.
pub fn clip_copy_region(
    src_min: &mut Vector3i,
    src_max: &mut Vector3i,
    src_size: Vector3i,
    dst_min: &mut Vector3i,
    dst_size: Vector3i,
) {
    clip_copy_region_coord(
        &mut src_min.x,
        &mut src_max.x,
        src_size.x,
        &mut dst_min.x,
        dst_size.x,
    );
    clip_copy_region_coord(
        &mut src_min.y,
        &mut src_max.y,
        src_size.y,
        &mut dst_min.y,
        dst_size.y,
    );
    clip_copy_region_coord(
        &mut src_min.z,
        &mut src_max.z,
        src_size.z,
        &mut dst_min.z,
        dst_size.z,
    );
}

/// Copy a rectangular sub-region from `src` to `dst`, both flat byte buffers in
/// ZXY layout (Y innermost). `item_size` is the bytes per element (e.g. 1, 2, 4).
/// Matches `copy_3d_region_zxy`.
///
/// Panics in debug if the same buffer is copied onto an overlapping region, or
/// if the region exceeds either buffer (matching the C++ `ZN_ASSERT_RETURN`s).
#[allow(clippy::too_many_arguments)] // 8 params match the C++ API surface
pub fn copy_3d_region_zxy(
    dst: &mut [u8],
    dst_size: Vector3i,
    dst_min: Vector3i,
    src: &[u8],
    src_size: Vector3i,
    mut src_min: Vector3i,
    mut src_max: Vector3i,
    item_size: usize,
) {
    Vector3i::sort_min_max(&mut src_min, &mut src_max);
    let mut dst_min = dst_min;
    clip_copy_region(&mut src_min, &mut src_max, src_size, &mut dst_min, dst_size);
    let area_size = src_max - src_min;
    if area_size.x <= 0 || area_size.y <= 0 || area_size.z <= 0 {
        return; // Degenerate area.
    }

    debug_assert!(
        !(core::ptr::eq(src.as_ptr(), dst.as_ptr())
            && crate::math::Box3i::from_min_max(src_min, src_max).intersects(
                &crate::math::Box3i::from_min_max(dst_min, dst_min + area_size)
            )),
        "Copy across the same buffer to an overlapping area is not supported"
    );

    let area_bytes = area_size.volume_u64() as usize * item_size;
    debug_assert!(area_bytes <= dst.len());
    debug_assert!(area_bytes <= src.len());

    if area_size == src_size && area_size == dst_size {
        debug_assert_eq!(dst.len(), src.len());
        dst.copy_from_slice(src);
    } else {
        // Copy row by row (Y is the row direction).
        let src_row_offset = (src_size.y as usize) * item_size;
        let dst_row_offset = (dst_size.y as usize) * item_size;
        let row_bytes = (area_size.y as usize) * item_size;
        for z in 0..area_size.z {
            let mut src_ri =
                (src_min + Vector3i::new(0, 0, z)).zxy_index(src_size) as usize * item_size;
            let mut dst_ri =
                (dst_min + Vector3i::new(0, 0, z)).zxy_index(dst_size) as usize * item_size;
            for _x in 0..area_size.x {
                dst[dst_ri..dst_ri + row_bytes].copy_from_slice(&src[src_ri..src_ri + row_bytes]);
                src_ri += src_row_offset;
                dst_ri += dst_row_offset;
            }
        }
    }
}

/// Fill a rectangular sub-region of `dst` (ZXY layout) with `value`. `dst_min`/
/// `dst_max` are clamped to `[0, dst_size]` and may be inverted (sorted first).
/// Matches `fill_3d_region_zxy<T>`.
pub fn fill_3d_region_zxy<T: Copy>(
    dst: &mut [T],
    dst_size: Vector3i,
    mut dst_min: Vector3i,
    mut dst_max: Vector3i,
    value: T,
) {
    Vector3i::sort_min_max(&mut dst_min, &mut dst_max);
    dst_min.x = funcs::clamp(dst_min.x, 0, dst_size.x);
    dst_min.y = funcs::clamp(dst_min.y, 0, dst_size.y);
    dst_min.z = funcs::clamp(dst_min.z, 0, dst_size.z);
    dst_max.x = funcs::clamp(dst_max.x, 0, dst_size.x);
    dst_max.y = funcs::clamp(dst_max.y, 0, dst_size.y);
    dst_max.z = funcs::clamp(dst_max.z, 0, dst_size.z);
    let area_size = dst_max - dst_min;
    if area_size.x <= 0 || area_size.y <= 0 || area_size.z <= 0 {
        return;
    }
    debug_assert!(area_size.volume_u64() as usize <= dst.len());

    if area_size == dst_size {
        for v in dst.iter_mut() {
            *v = value;
        }
    } else {
        let dst_row_offset = dst_size.y as usize;
        for z in 0..area_size.z {
            let mut dst_ri = (dst_min + Vector3i::new(0, 0, z)).zxy_index(dst_size) as usize;
            for _x in 0..area_size.x {
                for y in 0..area_size.y {
                    dst[dst_ri + y as usize] = value;
                }
                dst_ri += dst_row_offset;
            }
        }
    }
}

// ---- snorm quantization (matches the inline constexpr functions in funcs.h) ----

/// int8 → float in `[-1, 1]`, Vulkan convention. `-128` clamps to `-1`.
/// Matches `s8_to_snorm`.
#[inline]
pub fn s8_to_snorm(v: i8) -> f32 {
    funcs::max(v as f32 / 127.0, -1.0)
}

/// int8 → float in `[-1, 1]` without the `-1` clamp (so `-128` → slightly < -1).
/// Matches `s8_to_snorm_noclamp`.
#[inline]
pub fn s8_to_snorm_noclamp(v: i8) -> f32 {
    v as f32 / 127.0
}

/// float `[-1, 1]` → int8 (clamped). Matches `snorm_to_s8`.
#[inline]
pub fn snorm_to_s8(v: f32) -> i8 {
    (funcs::clamp(v, -1.0, 1.0) * 127.0) as i8
}

/// int16 → float in `[-1, 1]`, `-32768` clamps to `-1`. Matches `s16_to_snorm`.
#[inline]
pub fn s16_to_snorm(v: i16) -> f32 {
    funcs::max(v as f32 / 32767.0, -1.0)
}

/// int16 → float in `[-1, 1]` without the clamp. Matches `s16_to_snorm_noclamp`.
#[inline]
pub fn s16_to_snorm_noclamp(v: i16) -> f32 {
    v as f32 / 32767.0
}

/// float `[-1, 1]` → int16 (clamped). Matches `snorm_to_s16`.
#[inline]
/// Legacy v2 SDF unsigned encoding: `(u - 127) / 127`. Used only by the
/// v2→v3 block migration.
pub fn u8_to_snorm(u: u8) -> f32 {
    (f32::from(u) - 127.0) * (1.0 / 127.0)
}

/// Legacy v2 SDF unsigned encoding: `(u - 32767) / 32767`.
pub fn u16_to_snorm(u: u16) -> f32 {
    (f32::from(u) - 32767.0) * (1.0 / 32767.0)
}

pub fn snorm_to_s16(v: f32) -> i16 {
    (funcs::clamp(v, -1.0, 1.0) * 32767.0) as i16
}

/// Origin to add to transformed 3D coords so an [`OrthoBasis`] rotation keeps
/// cells inside the destination array. Returns `(origin, dst_size)`.
/// Matches `get_3d_array_transform_origin`.
pub fn get_3d_array_transform_origin(
    basis: &OrthoBasis,
    src_size: Vector3i,
) -> (Vector3i, Vector3i) {
    let xa = if basis.x.x != 0 {
        0
    } else if basis.x.y != 0 {
        1
    } else {
        2
    };
    let ya = if basis.y.x != 0 {
        0
    } else if basis.y.y != 0 {
        1
    } else {
        2
    };
    let za = if basis.z.x != 0 {
        0
    } else if basis.z.y != 0 {
        1
    } else {
        2
    };

    let mut dst_size = Vector3i::zero();
    dst_size[xa] = src_size.x;
    dst_size[ya] = src_size.y;
    dst_size[za] = src_size.z;

    // If an axis is negative, iteration starts from the end.
    let ox = if basis.get_axis(xa).x < 0 {
        dst_size.x - 1
    } else {
        0
    };
    let oy = if basis.get_axis(ya).y < 0 {
        dst_size.y - 1
    } else {
        0
    };
    let oz = if basis.get_axis(za).z < 0 {
        dst_size.z - 1
    } else {
        0
    };

    (Vector3i::new(ox, oy, oz), dst_size)
}

/// Rotate/flip/transpose a 3D array (ZXY layout) using `basis`. `src_grid` and
/// `dst_grid` must have the same volume (`src_size.x*y*z`). Returns the
/// transformed size. Matches `transform_3d_array_zxy<T>`.
pub fn transform_3d_array_zxy<T: Copy>(
    src_grid: &[T],
    dst_grid: &mut [T],
    src_size: Vector3i,
    basis: OrthoBasis,
) -> Vector3i {
    debug_assert!(basis.x.is_unit_vector());
    debug_assert!(basis.y.is_unit_vector());
    debug_assert!(basis.z.is_unit_vector());
    let vol = src_size.volume_u64() as usize;
    debug_assert_eq!(src_grid.len(), vol);
    debug_assert_eq!(dst_grid.len(), vol);

    let (origin, dst_size) = get_3d_array_transform_origin(&basis, src_size);
    let mut src_i = 0usize;
    for z in 0..src_size.z {
        for x in 0..src_size.x {
            for y in 0..src_size.y {
                let dst_x = origin.x + x * basis.x.x + y * basis.y.x + z * basis.z.x;
                let dst_y = origin.y + x * basis.x.y + y * basis.y.y + z * basis.z.y;
                let dst_z = origin.z + x * basis.x.z + y * basis.y.z + z * basis.z.z;
                let dst_i = (dst_y + dst_size.y * (dst_x + dst_size.x * dst_z)) as usize;
                dst_grid[dst_i] = src_grid[src_i];
                src_i += 1;
            }
        }
    }
    dst_size
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clip_copy_region_clamps_both_sides() {
        let mut src_min = Vector3i::new(-2, 0, 0);
        let mut src_max = Vector3i::new(10, 4, 4);
        let src_size = Vector3i::new(8, 4, 4);
        let mut dst_min = Vector3i::new(0, 0, 0);
        let dst_size = Vector3i::new(4, 4, 4);
        clip_copy_region(&mut src_min, &mut src_max, src_size, &mut dst_min, dst_size);
        // src clamped to [0,8), dst shifted by 2, then clamped to dst_size 4.
        assert!(src_min.x >= 0);
        assert!(src_max.x <= src_size.x);
        assert!(dst_min.x >= 0);
    }

    #[test]
    fn copy_region_full_buffer() {
        let src = (0..64u8).collect::<Vec<_>>();
        let mut dst = vec![0u8; 64];
        let size = Vector3i::new(4, 4, 4);
        copy_3d_region_zxy(
            &mut dst,
            size,
            Vector3i::zero(),
            &src,
            size,
            Vector3i::zero(),
            size,
            1,
        );
        assert_eq!(dst, src);
    }

    #[test]
    fn copy_region_subrect_with_item_size_4() {
        // 2x2x2 source of u32, copy a 1x1x1 corner.
        let size = Vector3i::new(2, 2, 2);
        let src: Vec<u32> = (0..8).collect();
        let mut dst = vec![0u32; 8];
        // Reinterpret as bytes for copy_3d_region_zxy.
        let mut dst_bytes = vec![0u8; 8 * 4];
        copy_3d_region_zxy(
            &mut dst_bytes,
            size,
            Vector3i::new(1, 0, 0),
            bytemuck_cast(&src),
            size,
            Vector3i::new(1, 0, 0),
            Vector3i::new(2, 1, 1),
            4,
        );
        // Reinterpret back.
        dst = bytemuck_cast_back(&dst_bytes);
        // Position (1,0,0) in src has value 2 (zxy index: y=0 + sy*(x=1 + sx*z=0) = 0 + 2*(1+0)=2).
        assert_eq!(dst[2], src[2]);
    }

    #[test]
    fn fill_region_full() {
        let mut buf = vec![0u32; 27];
        let size = Vector3i::new(3, 3, 3);
        fill_3d_region_zxy(&mut buf, size, Vector3i::zero(), size, 42u32);
        assert!(buf.iter().all(|&v| v == 42));
    }

    #[test]
    fn fill_region_subrect() {
        let mut buf = vec![0u32; 27];
        let size = Vector3i::new(3, 3, 3);
        // Fill the cell at (1,1,1) only.
        fill_3d_region_zxy(
            &mut buf,
            size,
            Vector3i::new(1, 1, 1),
            Vector3i::new(2, 2, 2),
            7u32,
        );
        // zxy index of (1,1,1) = 1 + 3*(1 + 3*1) = 1 + 12 = 13.
        assert_eq!(buf[13], 7);
        let nonzero = buf.iter().filter(|&&v| v != 0).count();
        assert_eq!(nonzero, 1);
    }

    #[test]
    fn snorm_round_trip_s8() {
        // 0 maps to exactly 0.0 (Vulkan property).
        assert_eq!(s8_to_snorm(0), 0.0);
        assert!((s8_to_snorm(127) - 1.0).abs() < 1e-6);
        assert!((s8_to_snorm(-127) + 1.0).abs() < 1e-6);
        // -128 clamps to -1.
        assert_eq!(s8_to_snorm(-128), -1.0);
        // Round-trip a value within range.
        let v = 0.5f32;
        let q = snorm_to_s8(v);
        assert!((s8_to_snorm(q) - v).abs() < 1.0 / 127.0 + 1e-6);
    }

    #[test]
    fn snorm_round_trip_s16() {
        assert_eq!(s16_to_snorm(0), 0.0);
        assert!((s16_to_snorm(32767) - 1.0).abs() < 1e-6);
        assert_eq!(s16_to_snorm(-32768), -1.0); // clamped
        let q = snorm_to_s16(0.25);
        assert!((s16_to_snorm(q) - 0.25).abs() < 1.0 / 32767.0 + 1e-6);
    }

    #[test]
    fn transform_identity_returns_same() {
        let size = Vector3i::new(2, 3, 4);
        let src: Vec<u32> = (0..24).collect();
        let mut dst = vec![0u32; 24];
        let out_size = transform_3d_array_zxy(&src, &mut dst, size, OrthoBasis::default());
        assert_eq!(out_size, size);
        assert_eq!(dst, src);
    }

    #[test]
    fn transform_preserves_volume() {
        let size = Vector3i::new(2, 2, 2);
        let src: Vec<u32> = (0..8).collect();
        let mut dst = vec![0u32; 8];
        // Rotate 90 around Z (cw): basis from the ortho table.
        let basis = OrthoBasis::from_axis_turns(crate::math::Axis::Z, 1);
        let out_size = transform_3d_array_zxy(&src, &mut dst, size, basis);
        assert_eq!(out_size, size);
        // All elements present (a permutation).
        let mut sorted = dst.clone();
        sorted.sort();
        assert_eq!(sorted, src);
    }

    // Helper: cast a &[u32] to &[u8] without an external dependency.
    fn bytemuck_cast(src: &[u32]) -> &[u8] {
        unsafe { core::slice::from_raw_parts(src.as_ptr() as *const u8, src.len() * 4) }
    }
    fn bytemuck_cast_back(src: &[u8]) -> Vec<u32> {
        assert!(src.len().is_multiple_of(4));
        let mut out = vec![0u32; src.len() / 4];
        unsafe {
            core::ptr::copy_nonoverlapping(src.as_ptr(), out.as_mut_ptr() as *mut u8, src.len());
        }
        out
    }
}
