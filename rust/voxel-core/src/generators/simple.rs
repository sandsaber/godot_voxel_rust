//! `generators::simple` — math-pure terrain generators.
//!
//! Ported from `generators/simple/{voxel_generator_waves,voxel_generator_flat,
//! voxel_generator_noise}.{h,cpp}`. [`Waves`] uses a sinusoid, [`Flat`] uses a
//! constant plane, [`Noise`] samples 3D [`fastnoise_lite`] into an SDF terrain
//! slab. None of them goes through the 2D heightmap helper except [`Waves`].
//!
//! The C++ versions inherit `VoxelGenerator` (a Godot `Resource` with an
//! `RWLock` around their parameter struct) and `VoxelGeneratorHeightmap`. Here
//! each generator owns its parameters by value and implements [`VoxelGenerator`];
//! parameter mutation is single-threaded by Rust's `&mut` borrow rules.

use crate::generators::base::{
    generate_heightmap, GenResult, HeightmapParams, VoxelGenerator, VoxelQueryData,
};
use crate::math::funcs;
use crate::math::{Vector2f, Vector2i, Vector3i};
use crate::storage::voxel_buffer::ChannelId;
use std::sync::Arc;

// ===========================================================================
// Waves
// ===========================================================================

/// Sinusoidal heightmap generator. Ported from `VoxelGeneratorWaves`.
///
/// Produces terrain height `0.5 + 0.25 * (cos((x+ox)*fx) + sin((z+oz)*fz))`
/// where `f = pi / pattern_size`, before the heightmap range remap.
#[derive(Debug, Clone, PartialEq)]
pub struct Waves {
    /// Period of the wave pattern along each axis. Clamped to `>= 0` on set.
    pub pattern_size: Vector2f,
    /// Phase offset (in voxels) of the pattern along each axis.
    pub pattern_offset: Vector2f,
    /// Shared heightmap parameters (channel, range, iso_scale, …).
    pub heightmap: HeightmapParams,
}

impl Default for Waves {
    fn default() -> Self {
        // C++ ctor: pattern_size (30, 30), height_range 30.
        Self {
            pattern_size: Vector2f::new(30.0, 30.0),
            pattern_offset: Vector2f::new(0.0, 0.0),
            heightmap: HeightmapParams {
                height_start: 0.0,
                height_range: 30.0,
                ..Default::default()
            },
        }
    }
}

impl Waves {
    /// Compute the raw (pre-range-remap) height at world `(x, z)`. Exposed for
    /// unit tests; matches the C++ lambda inside `generate_block`.
    pub fn height_at(&self, x: i32, z: i32) -> f32 {
        let fx = std::f32::consts::PI / self.pattern_size.x;
        let fz = std::f32::consts::PI / self.pattern_size.y;
        let ox = self.pattern_offset.x;
        let oz = self.pattern_offset.y;
        0.5 + 0.25 * (((x as f32 + ox) * fx).cos() + ((z as f32 + oz) * fz).sin())
    }

    /// `set_pattern_size` — clamps both components to `>= 0`.
    pub fn set_pattern_size(&mut self, size: Vector2f) {
        self.pattern_size = Vector2f::new(funcs::max(size.x, 0.0), funcs::max(size.y, 0.0));
    }

    /// `set_pattern_offset`.
    pub fn set_pattern_offset(&mut self, offset: Vector2f) {
        self.pattern_offset = offset;
    }
}

impl VoxelGenerator for Waves {
    fn generate_block(&self, input: VoxelQueryData<'_>) -> GenResult {
        let ps = self.pattern_size;
        let po = self.pattern_offset;
        let hp = self.heightmap;
        // Capture by value so the closure is `Fn` and borrows nothing.
        let height_fn = move |x: i32, z: i32| {
            let fx = std::f32::consts::PI / ps.x;
            let fz = std::f32::consts::PI / ps.y;
            0.5 + 0.25 * (((x as f32 + po.x) * fx).cos() + ((z as f32 + po.y) * fz).sin())
        };
        generate_heightmap(
            input.buffer,
            height_fn,
            &hp,
            input.origin_in_voxels,
            input.lod,
        )
    }
}

// ===========================================================================
// Flat
// ===========================================================================

/// A flat ground plane at a fixed height. Ported from `VoxelGeneratorFlat`.
///
/// Unlike [`Waves`], this generator does **not** go through the shared
/// heightmap helper: it has its own `generate_block` with an SDF and a blocky
/// path, plus two early-exit branches (block entirely above / below the
/// plane). The C++ version is the same — it overrides `generate_block`
/// directly rather than using `VoxelGeneratorHeightmap::generate`.
#[derive(Debug, Clone, PartialEq)]
pub struct Flat {
    /// Channel to write. Defaults to SDF.
    pub channel: ChannelId,
    /// Voxel id used when filling blocky terrain below `height`.
    pub voxel_type: u64,
    /// World-space Y of the ground plane.
    pub height: f32,
    /// SDF iso-surface scale (multiplies `y - height`).
    pub iso_scale: f32,
}

impl Default for Flat {
    fn default() -> Self {
        Self {
            channel: ChannelId::Sdf,
            voxel_type: 1,
            height: 0.0,
            iso_scale: 1.0,
        }
    }
}

impl Flat {
    pub fn set_channel(&mut self, channel: ChannelId) {
        self.channel = channel;
    }
    pub fn set_voxel_type(&mut self, t: u64) {
        self.voxel_type = t;
    }
    pub fn set_height(&mut self, h: f32) {
        self.height = h;
    }
    pub fn set_iso_scale(&mut self, s: f32) {
        self.iso_scale = s;
    }
}

impl VoxelGenerator for Flat {
    fn generate_block(&self, input: VoxelQueryData<'_>) -> GenResult {
        let channel = self.channel.index();
        let origin = input.origin_in_voxels;
        let bs = input.buffer.size();
        let use_sdf = self.channel == ChannelId::Sdf;
        let margin = 1i32 << input.lod;
        let lod = input.lod;

        // Block bottom above the highest ground → air.
        if (origin.y as f32) > self.height + margin as f32 {
            return GenResult::max_lod();
        }
        // Block top below the lowest ground → uniform fill.
        let block_top = origin.y + (bs.y << lod);
        if (block_top as f32) < self.height - margin as f32 {
            if use_sdf {
                // "Not consistent SDF but should work ok" — matches C++.
                input.buffer.clear_channel_f(channel, -100.0);
            } else {
                input.buffer.clear_channel(channel, self.voxel_type);
            }
            return GenResult::max_lod();
        }

        let stride = 1i32 << lod;
        if use_sdf {
            // Flat plane: height is constant, so the SDF depends only on Y.
            // (The C++ loop still tracks gx/gz for parity with the heightmap
            // generators, but they don't affect the output; we drop them.)
            for z in 0..bs.z {
                for x in 0..bs.x {
                    let mut gy = origin.y;
                    for y in 0..bs.y {
                        let sdf = self.iso_scale * (gy as f32 - self.height);
                        input.buffer.set_voxel_f(sdf, x, y, z, channel);
                        gy += stride;
                    }
                }
            }
        } else {
            // Blocky: fill [0, irh_voxels) across the whole block footprint.
            let rh_world = self.height - origin.y as f32;
            let irh_world = rh_world as i32;
            if irh_world > 0 {
                let irh_voxels = funcs::min(funcs::arithmetic_rshift(irh_world, lod), bs.y);
                input.buffer.fill_area(
                    self.voxel_type,
                    Vector3i::new(0, 0, 0),
                    Vector3i::new(bs.x, irh_voxels, bs.z),
                    channel,
                );
            }
        }

        GenResult::default()
    }

    fn used_channels_mask(&self) -> u32 {
        1 << self.channel.index()
    }
}

// ===========================================================================
// Noise
// ===========================================================================

/// SDF sentinel for voxels "far outside" the surface (deep air).
/// Matches `constants::voxel_constants.h::SDF_FAR_OUTSIDE`.
const SDF_FAR_OUTSIDE: f32 = 100.0;
/// SDF sentinel for voxels "far inside" the surface (deep solid).
/// Matches `constants::voxel_constants.h::SDF_FAR_INSIDE`.
const SDF_FAR_INSIDE: f32 = -100.0;

/// 3D noise SDF generator. Ported from `VoxelGeneratorNoise`.
///
/// Builds a terrain slab of height [`Self::height_range`] starting at
/// [`Self::height_start`]: per voxel `(x,y,z)` the SDF is
/// `(noise_3d(x,y,z) + bias) * noise_period`, where `bias` is a linear ramp
/// in Y (-1 at the bottom, +1 at the top) that turns noise into a surface,
/// and `noise_period = 1/frequency` rescales the gradient so 16-bit SDF
/// encoding doesn't produce blocky artifacts.
///
/// This is **not** a 2D heightmap — it samples full 3D noise and therefore
/// has its own `generate_block` loop (like [`Flat`]), bypassing
/// [`generate_heightmap`].
pub struct Noise {
    /// Channel to write. Defaults to SDF.
    pub channel: ChannelId,
    /// The noise sampler. Configured via `set_*` before generation.
    pub noise: fastnoise_lite::FastNoiseLite,
    /// World-space Y where the terrain slab begins (bottom).
    pub height_start: f32,
    /// Vertical extent of the slab. Clamped to `>= 0.1` on set.
    pub height_range: f32,
}

impl Default for Noise {
    fn default() -> Self {
        // C++ defaults: height_start=-100, height_range=200, channel=SDF.
        // The noise resource starts unconfigured; callers must set a seed,
        // frequency and noise type before generating.
        Self {
            channel: ChannelId::Sdf,
            noise: fastnoise_lite::FastNoiseLite::new(),
            height_start: -100.0,
            height_range: 200.0,
        }
    }
}

impl Noise {
    pub fn set_channel(&mut self, channel: ChannelId) {
        self.channel = channel;
    }
    /// `set_height_range` — clamps to `>= 0.1` (matches C++).
    pub fn set_height_range(&mut self, range: f32) {
        self.height_range = funcs::max(range, 0.1);
    }
    pub fn set_height_start(&mut self, start: f32) {
        self.height_start = start;
    }
    /// Borrow the underlying noise sampler for configuration (seed, frequency,
    /// noise type, fractal settings). Returns `&mut` so the caller can call
    /// `set_*` methods.
    pub fn noise_mut(&mut self) -> &mut fastnoise_lite::FastNoiseLite {
        &mut self.noise
    }

    /// Sample the raw 3D noise (not the terrain SDF) at a world point. Used by
    /// the Godot binding ([`FastNoiseLiteGD`](../../voxel_gdext/...)) to expose
    /// noise sampling through the binding without depending on `fastnoise-lite`
    /// directly.
    pub fn sample_noise_3d(&self, x: f32, y: f32, z: f32) -> f32 {
        self.noise.get_noise_3d(x, y, z)
    }

    /// The noise period derived from the configured frequency. Matches the C++
    /// `noise_period = 1.0 / max(frequency, 0.0001)`.
    fn noise_period(&self) -> f32 {
        // fastnoise-lite defaults frequency to None → 0.01 internally; we read
        // it back the same way the C++ reads `noise.get_frequency()`.
        let freq = self.frequency();
        1.0 / funcs::max(freq, 0.0001)
    }

    /// Accessor for the configured frequency. The crate stores the effective
    /// value directly and maps `set_frequency(None)` back to the default 0.01.
    fn frequency(&self) -> f32 {
        self.noise.frequency
    }
}

impl VoxelGenerator for Noise {
    fn generate_block(&self, input: VoxelQueryData<'_>) -> GenResult {
        let channel = self.channel.index();
        let origin = input.origin_in_voxels;
        let bs = input.buffer.size();
        let use_sdf = self.channel == ChannelId::Sdf;
        let lod = input.lod;

        let noise_period = self.noise_period();
        let lower_bound = (self.height_start).floor() as i32;
        let upper_bound = (self.height_start + self.height_range).ceil() as i32;

        // Early-exit A: block entirely above the terrain slab → air.
        if origin.y >= upper_bound {
            if use_sdf {
                input.buffer.clear_channel_f(channel, SDF_FAR_OUTSIDE);
            } else {
                input.buffer.clear_channel(channel, 0);
            }
            return GenResult::max_lod();
        }
        // Early-exit B: block entirely below the slab → solid.
        if origin.y + (bs.y << lod) <= lower_bound {
            if use_sdf {
                input.buffer.clear_channel_f(channel, SDF_FAR_INSIDE);
            } else {
                input.buffer.clear_channel(channel, 1);
            }
            return GenResult::max_lod();
        }

        let stride = 1i32 << lod;
        let inv_height_range = 1.0 / self.height_range;

        let mut gz = origin.z;
        for z in 0..bs.z {
            let mut gx = origin.x;
            for x in 0..bs.x {
                let mut gy = origin.y;
                for y in 0..bs.y {
                    if gy < lower_bound {
                        if use_sdf {
                            input.buffer.set_voxel_f(SDF_FAR_INSIDE, x, y, z, channel);
                        } else {
                            input.buffer.set_voxel(1, x, y, z, channel);
                        }
                    } else if gy >= upper_bound {
                        if use_sdf {
                            input.buffer.set_voxel_f(SDF_FAR_OUTSIDE, x, y, z, channel);
                        } else {
                            input.buffer.set_voxel(0, x, y, z, channel);
                        }
                    } else {
                        // Inside the slab: sample 3D noise + bias ramp.
                        let t = (gy as f32 - self.height_start) * inv_height_range;
                        let bias = 2.0 * t - 1.0;
                        let n = self.noise.get_noise_3d(gx as f32, gy as f32, gz as f32);
                        let d = (n + bias) * noise_period;
                        if use_sdf {
                            input.buffer.set_voxel_f(d, x, y, z, channel);
                        } else {
                            input
                                .buffer
                                .set_voxel(if d < 0.0 { 1 } else { 0 }, x, y, z, channel);
                        }
                    }
                    gy += stride;
                }
                gx += stride;
            }
            gz += stride;
        }

        GenResult::default()
    }

    fn used_channels_mask(&self) -> u32 {
        1 << self.channel.index()
    }
}

// ===========================================================================
// HeightmapNoise (2D noise → heightmap)
// ===========================================================================

/// A baked 1D curve mapping noise output `[0,1]` to a height value. Ported
/// from Godot's `Curve` resource (the subset used by `VoxelGeneratorNoise2D`):
/// a lookup table sampled with linear interpolation.
#[derive(Debug, Clone)]
pub struct Curve {
    /// Sampled values at evenly-spaced points in `[0, 1]`.
    points: Vec<f32>,
}

impl Curve {
    /// Build a curve from evenly-spaced sample points covering `[0, 1]`.
    /// At least 2 points are required. Matches Godot `Curve::bake()` output
    /// (the C++ generator calls `curve.sample_baked(t)`).
    pub fn from_points(points: Vec<f32>) -> Self {
        assert!(points.len() >= 2, "curve needs at least 2 points");
        Self { points }
    }

    /// Identity curve: `sample(t) = t`. Useful as a default when no curve is
    /// set (the generator applies the curve only if present).
    pub fn identity(point_count: usize) -> Self {
        let n = point_count.max(2);
        Self::from_points((0..n).map(|i| i as f32 / (n - 1) as f32).collect())
    }

    /// `sample_baked(t)` — linear interpolation of the baked points. `t` is
    /// clamped to `[0, 1]`. Matches Godot `Curve::sample_baked`.
    pub fn sample(&self, t: f32) -> f32 {
        let t = funcs::clamp(t, 0.0, 1.0);
        let n = self.points.len();
        let f = t * (n - 1) as f32;
        let i = f as usize;
        if i >= n - 1 {
            return self.points[n - 1];
        }
        let frac = f - i as f32;
        self.points[i] * (1.0 - frac) + self.points[i + 1] * frac
    }
}

impl Default for Curve {
    fn default() -> Self {
        Self::identity(256)
    }
}

/// Serializable noise configuration that can be cheaply cloned into a
/// `FastNoiseLite` instance per `generate_block` call (the crate's
/// `FastNoiseLite` doesn't implement `Clone`, so we store the settings and
/// rebuild the sampler).
///
/// The `fractal_*` fields are passthrough settings read from the user-assigned
/// noise resource (upstream `ZN_FastNoiseLite`): when a generator is driven by
/// such a resource, its fractal configuration genuinely changes the output.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct NoiseConfig {
    pub seed: Option<i32>,
    pub frequency: Option<f32>,
    pub noise_type: Option<fastnoise_lite::NoiseType>,
    pub fractal_type: Option<fastnoise_lite::FractalType>,
    pub fractal_octaves: Option<i32>,
    pub fractal_lacunarity: Option<f32>,
    pub fractal_gain: Option<f32>,
    pub fractal_weighted_strength: Option<f32>,
    pub fractal_ping_pong_strength: Option<f32>,
}

impl NoiseConfig {
    /// Build a configured `FastNoiseLite` from these settings. `None` fields
    /// keep the sampler's own defaults.
    pub fn build(&self) -> fastnoise_lite::FastNoiseLite {
        let mut n = fastnoise_lite::FastNoiseLite::new();
        n.set_seed(self.seed);
        n.set_frequency(self.frequency);
        n.set_noise_type(self.noise_type);
        n.set_fractal_type(self.fractal_type);
        n.set_fractal_octaves(self.fractal_octaves);
        n.set_fractal_lacunarity(self.fractal_lacunarity);
        n.set_fractal_gain(self.fractal_gain);
        n.set_fractal_weighted_strength(self.fractal_weighted_strength);
        n.set_fractal_ping_pong_strength(self.fractal_ping_pong_strength);
        n
    }
}

/// 2D-noise heightmap generator. Ported from `VoxelGeneratorNoise2D`.
///
/// Samples 2D noise per `(x, z)` column, optionally remaps it through a
/// [`Curve`], then builds terrain via the shared [`generate_heightmap`] helper
/// (the same path [`Waves`] uses). Unlike [`Noise`] (which does full 3D SDF),
/// this one produces a heightmap surface.
#[derive(Default)]
pub struct HeightmapNoise {
    /// Noise configuration (seed, frequency, type). Cloned into a sampler
    /// per generation call.
    pub noise_config: NoiseConfig,
    /// Optional curve remapping noise `[0,1]` → height. When `None`, the raw
    /// `0.5 + 0.5*noise` value is used.
    pub curve: Option<Arc<Curve>>,
    /// Shared heightmap parameters (channel, range, iso_scale, offset).
    pub heightmap: HeightmapParams,
}

impl HeightmapNoise {
    /// Set the optional curve.
    pub fn set_curve(&mut self, curve: Option<Curve>) {
        self.curve = curve.map(Arc::new);
    }

    /// Set the optional curve from shared storage. This keeps per-block
    /// generation at an O(1) refcount clone instead of copying baked points.
    pub fn set_curve_arc(&mut self, curve: Option<Arc<Curve>>) {
        self.curve = curve;
    }
}

impl VoxelGenerator for HeightmapNoise {
    fn generate_block(&self, input: VoxelQueryData<'_>) -> GenResult {
        // Rebuild a fresh sampler from the config (FastNoiseLite isn't Clone).
        let noise = self.noise_config.build();
        let curve = self.curve.clone();
        let hp = self.heightmap;

        let result = match curve {
            Some(c) => generate_heightmap(
                input.buffer,
                move |x, z| {
                    let n = noise.get_noise_2d(x as f32, z as f32);
                    c.sample(0.5 + 0.5 * n)
                },
                &hp,
                input.origin_in_voxels,
                input.lod,
            ),
            None => generate_heightmap(
                input.buffer,
                move |x, z| {
                    let n = noise.get_noise_2d(x as f32, z as f32);
                    0.5 + 0.5 * n
                },
                &hp,
                input.origin_in_voxels,
                input.lod,
            ),
        };

        // The C++ compresses uniform channels after generation; do the same.
        input.buffer.compress_uniform_channels();
        result
    }

    fn used_channels_mask(&self) -> u32 {
        1 << self.heightmap.channel.index()
    }
}

// ===========================================================================
// Image
// ===========================================================================

/// How world coordinates outside the image extent are sampled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ImageWrapMode {
    /// Coordinates outside the image clamp to the nearest edge pixel.
    #[default]
    Clamp,
    /// Coordinates wrap around (tiling terrain).
    Repeat,
}

/// 5-tap plus/cross blur over a row-major `width * height` grid: the centre
/// plus its 4 orthogonal neighbours, each weighted `0.2`, with wrap-around
/// fetches across image borders (modulo indexing). Mirrors upstream
/// `voxel_generator_image.cpp::get_height_blurred` + `get_height_repeat`
/// exactly (upstream multiplies the 5-tap sum by `0.2f` and wraps via
/// `math::wrap`). Sum-preserving, since every pixel contributes its full
/// value to its own and its four neighbours' outputs. Backing of the pinned
/// `VoxelGeneratorImage.blur_enabled` property.
fn blur_heights(values: Vec<f32>, width: i32, height: i32) -> Vec<f32> {
    let w = width;
    let h = height;
    // Wrap-around fetch, mirroring `get_height_repeat` (`math::wrap`).
    let repeat = |values: &[f32], x: i32, z: i32| -> f32 {
        values[(funcs::wrap_i32(x, w) + funcs::wrap_i32(z, h) * w) as usize]
    };
    let mut blurred = vec![0.0; values.len()];
    for z in 0..h {
        for x in 0..w {
            let sum = repeat(&values, x, z)
                + repeat(&values, x + 1, z)
                + repeat(&values, x - 1, z)
                + repeat(&values, x, z + 1)
                + repeat(&values, x, z - 1);
            blurred[(x + z * w) as usize] = sum * 0.2;
        }
    }
    blurred
}

/// 2D image heightmap generator. Rust port of the `VoxelGeneratorImage`
/// concept: a stored grayscale image (`values` in `0..1`, row-major with
/// `x + z * width`) is sampled as the terrain height at world `(x, z)`.
///
/// With an SDF channel this produces smooth terrain; with a non-SDF channel
/// (e.g. `Type`) it fills blocky terrain up to the height, which lets it
/// drive cubes/blocky meshers.
#[derive(Debug, Clone, Default)]
pub struct Image {
    /// Grayscale heights in `0..1`, row-major (`values[x + z * width]`).
    values: Arc<[f32]>,
    /// Image width (X) and height (Z), in pixels/voxels.
    size: Vector2i,
    /// Sampling behaviour outside the image extent.
    pub wrap: ImageWrapMode,
    /// Apply the upstream 5-tap plus/cross blur (wrap-around at borders) to
    /// values as they are loaded (pinned `blur_enabled`; must be set before
    /// `set_image`).
    pub blur_enabled: bool,
    /// Shared heightmap parameters (channel, range, iso_scale, offset).
    pub heightmap: HeightmapParams,
}

impl Image {
    /// Set the image data. `values` must contain exactly `width * height`
    /// entries; they are clamped to `0..1` (and blurred when
    /// [`Self::blur_enabled`] is set at load time). Returns `false` on size
    /// mismatch.
    pub fn set_image(&mut self, values: Vec<f32>, width: i32, height: i32) -> bool {
        if width <= 0 || height <= 0 || values.len() != (width as usize) * (height as usize) {
            return false;
        }
        let clamped: Vec<f32> = values.iter().map(|v| v.clamp(0.0, 1.0)).collect();
        let clamped = if self.blur_enabled {
            blur_heights(clamped, width, height)
        } else {
            clamped
        };
        self.values = clamped.into();
        self.size = Vector2i::new(width, height);
        true
    }

    /// Whether an image is loaded.
    pub fn has_image(&self) -> bool {
        !self.values.is_empty()
    }

    /// Sample the normalized (`0..1`) height at world `(x, z)` applying the
    /// configured wrap mode. Empty images sample as `0.0`.
    pub fn height_at(&self, x: i32, z: i32) -> f32 {
        if self.values.is_empty() {
            return 0.0;
        }
        let w = self.size.x;
        let h = self.size.y;
        let ix = match self.wrap {
            ImageWrapMode::Repeat => funcs::wrap_i32(x, w),
            ImageWrapMode::Clamp => x.clamp(0, w - 1),
        };
        let iz = match self.wrap {
            ImageWrapMode::Repeat => funcs::wrap_i32(z, h),
            ImageWrapMode::Clamp => z.clamp(0, h - 1),
        };
        self.values[(ix + iz * w) as usize]
    }
}

impl VoxelGenerator for Image {
    fn generate_block(&self, input: VoxelQueryData<'_>) -> GenResult {
        let hp = self.heightmap;
        let result = generate_heightmap(
            input.buffer,
            |x, z| self.height_at(x, z),
            &hp,
            input.origin_in_voxels,
            input.lod,
        );
        input.buffer.compress_uniform_channels();
        result
    }

    fn used_channels_mask(&self) -> u32 {
        1 << self.heightmap.channel.index()
    }
}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;
    use crate::math::{Vector2f, Vector2i, Vector3i};
    use crate::storage::voxel_buffer::{ChannelId, Compression};
    use crate::storage::VoxelBuffer;
    use std::sync::Arc;

    /// Build a fresh SDF-channel buffer of the given size.
    fn sdf_buffer(size: Vector3i) -> VoxelBuffer {
        // SDF starts uniform; the generators decompress channels as they write.
        VoxelBuffer::with_size(size)
    }

    // ---- Waves: height function ------------------------------------------

    #[test]
    fn waves_height_at_zero_offset_is_half_plus_quarter_cos_sin() {
        let w = Waves {
            pattern_size: Vector2f::new(30.0, 30.0),
            pattern_offset: Vector2f::new(0.0, 0.0),
            ..Default::default()
        };
        // At (0, 0): cos(0) + sin(0) = 1 + 0 = 1 → 0.5 + 0.25 = 0.75.
        assert!((w.height_at(0, 0) - 0.75).abs() < 1e-5);
    }

    #[test]
    fn waves_height_is_bounded_between_0_and_1() {
        let w = Waves::default();
        // The sinusoid's range is 0.5 ± 0.5, so any integer (x, z) lands inside.
        for x in -100..100 {
            for z in -100..100 {
                let h = w.height_at(x, z);
                assert!(
                    (-1e-5..=1.0 + 1e-5).contains(&h),
                    "height {h} out of [0,1] at ({x},{z})"
                );
            }
        }
    }

    #[test]
    fn waves_set_pattern_size_clamps_negative_to_zero() {
        let mut w = Waves::default();
        w.set_pattern_size(Vector2f::new(-5.0, -10.0));
        assert_eq!(w.pattern_size, Vector2f::new(0.0, 0.0));
    }

    #[test]
    fn waves_pattern_offset_shifts_the_phase() {
        let mut w = Waves::default();
        w.pattern_size = Vector2f::new(30.0, 30.0);
        let h0 = w.height_at(0, 0);
        // Shifting by exactly the pattern period (2*pi * size, but our freq is
        // pi/size so the period is 2*size) must return the same height.
        w.pattern_offset = Vector2f::new(60.0, 60.0);
        let h_shifted = w.height_at(0, 0);
        assert!(
            (h0 - h_shifted).abs() < 1e-4,
            "period shift mismatch: {h0} vs {h_shifted}"
        );
    }

    // ---- Flat: SDF path --------------------------------------------------

    #[test]
    fn flat_sdf_gradient_grows_with_y_and_crosses_zero_at_height() {
        let mut gen = Flat::default();
        gen.height = 4.0;
        let mut buf = sdf_buffer(Vector3i::new(2, 8, 2));
        // Block spans world Y 0..8 with the plane at y=4.
        gen.generate_block(VoxelQueryData {
            buffer: &mut buf,
            origin_in_voxels: Vector3i::new(0, 0, 0),
            lod: 0,
        });
        // Below the plane: negative SDF (solid); at the plane: ~0; above: positive.
        assert!(buf.get_voxel_f(0, 0, 0, ChannelId::Sdf.index()) < 0.0);
        assert!(buf.get_voxel_f(0, 7, 0, ChannelId::Sdf.index()) > 0.0);
        assert!(buf.get_voxel_f(0, 4, 0, ChannelId::Sdf.index()).abs() < 1.0);
    }

    #[test]
    fn flat_sdf_uses_iso_scale() {
        let mut gen = Flat::default();
        gen.height = 0.0;
        gen.iso_scale = 0.1;
        let mut buf = sdf_buffer(Vector3i::new(1, 4, 1));
        // Use 32-bit depth so the SDF round-trips without 16-bit snorm
        // quantization (storage quantizes Bit16 SDFs to [-1,1] via snorm).
        buf.set_channel_depth(ChannelId::Sdf.index(), crate::storage::ChannelDepth::Bit32);
        gen.generate_block(VoxelQueryData {
            buffer: &mut buf,
            origin_in_voxels: Vector3i::new(0, 0, 0),
            lod: 0,
        });
        // At y=1 with iso_scale=0.1: sdf = 0.1 * (1 - 0) = 0.1.
        let v = buf.get_voxel_f(0, 1, 0, ChannelId::Sdf.index());
        assert!((v - 0.1).abs() < 1e-5, "sdf at y=1: {v}");
    }

    // ---- Flat: blocky path ----------------------------------------------

    #[test]
    fn flat_blocky_fills_below_height() {
        let mut gen = Flat::default();
        gen.channel = ChannelId::Type;
        gen.voxel_type = 7;
        gen.height = 3.0;
        let mut buf = VoxelBuffer::with_size(Vector3i::new(2, 8, 2));
        gen.generate_block(VoxelQueryData {
            buffer: &mut buf,
            origin_in_voxels: Vector3i::new(0, 0, 0),
            lod: 0,
        });
        // y < 3 should be solid (7), y >= 3 should be default (0).
        assert_eq!(buf.get_voxel(0, 0, 0, ChannelId::Type.index()), 7);
        assert_eq!(buf.get_voxel(0, 2, 0, ChannelId::Type.index()), 7);
        assert_eq!(buf.get_voxel(0, 3, 0, ChannelId::Type.index()), 0);
        assert_eq!(buf.get_voxel(0, 7, 0, ChannelId::Type.index()), 0);
    }

    // ---- Flat: early-exit branches --------------------------------------

    #[test]
    fn flat_early_exit_above_ground_leaves_air() {
        let mut gen = Flat::default();
        gen.height = 0.0;
        let mut buf = VoxelBuffer::with_size(Vector3i::new(2, 2, 2));
        let result = gen.generate_block(VoxelQueryData {
            buffer: &mut buf,
            origin_in_voxels: Vector3i::new(0, 100, 0), // well above the plane
            lod: 0,
        });
        assert!(result.max_lod_hint);
        // Buffer untouched: stays at default uniform value (0).
        assert_eq!(
            buf.channel_compression(ChannelId::Sdf.index()),
            Compression::Uniform
        );
    }

    #[test]
    fn flat_early_exit_below_ground_fills_uniform_sdf() {
        let mut gen = Flat::default();
        gen.height = 0.0;
        let mut buf = VoxelBuffer::with_size(Vector3i::new(2, 2, 2));
        // 32-bit depth so the -100 sentinel survives the round-trip
        // (Bit16 SDF is quantized to [-1,1] via snorm and would saturate).
        buf.set_channel_depth(ChannelId::Sdf.index(), crate::storage::ChannelDepth::Bit32);
        let result = gen.generate_block(VoxelQueryData {
            buffer: &mut buf,
            origin_in_voxels: Vector3i::new(0, -200, 0), // well below the plane
            lod: 0,
        });
        assert!(result.max_lod_hint);
        // SDF below ground is the C++ "not consistent" sentinel -100.
        let v = buf.get_voxel_f(0, 0, 0, ChannelId::Sdf.index());
        assert!(
            (v - (-100.0)).abs() < 1e-3,
            "below-ground SDF sentinel: {v}"
        );
    }

    #[test]
    fn flat_early_exit_below_ground_fills_uniform_blocky() {
        let mut gen = Flat::default();
        gen.channel = ChannelId::Type;
        gen.voxel_type = 9;
        gen.height = 0.0;
        let mut buf = VoxelBuffer::with_size(Vector3i::new(2, 2, 2));
        gen.generate_block(VoxelQueryData {
            buffer: &mut buf,
            origin_in_voxels: Vector3i::new(0, -200, 0),
            lod: 0,
        });
        assert_eq!(buf.get_voxel(0, 0, 0, ChannelId::Type.index()), 9);
    }

    // ---- used_channels_mask ---------------------------------------------

    #[test]
    fn flat_used_channels_mask_reflects_configured_channel() {
        let mut gen = Flat::default();
        assert_eq!(gen.used_channels_mask(), 1 << ChannelId::Sdf.index());
        gen.set_channel(ChannelId::Type);
        assert_eq!(gen.used_channels_mask(), 1 << ChannelId::Type.index());
    }

    #[test]
    fn waves_used_channels_mask_defaults_to_sdf() {
        let gen = Waves::default();
        // Waves uses the shared heightmap helper, which always writes the
        // channel from HeightmapParams (default SDF).
        let g: &dyn VoxelGenerator = &gen;
        assert_eq!(g.used_channels_mask(), 1 << ChannelId::Sdf.index());
    }

    // ---- heightmap range remap (via Waves integration) ------------------

    #[test]
    fn waves_applies_height_range_remap() {
        // Default Waves: height_range = 30, height_start = 0.
        // At a peak (h≈1) the world height is ~30; at a trough (h≈0) it's ~0.
        let mut gen = Waves::default();
        gen.heightmap.height_start = 0.0;
        gen.heightmap.height_range = 30.0;
        gen.heightmap.iso_scale = 1.0;
        // Force a peak by placing the block where the sinusoid is maximal.
        // We can't easily pick a peak in integer coords, so just verify the
        // SDF at the very top of a tall block crosses zero somewhere — i.e.
        // the heightmap surface is inside the block, not above/below it.
        let mut buf = sdf_buffer(Vector3i::new(1, 40, 1));
        let origin = Vector3i::new(0, 0, 0);
        gen.generate_block(VoxelQueryData {
            buffer: &mut buf,
            origin_in_voxels: origin,
            lod: 0,
        });
        // Find the sign-change row (the surface). Heights range ~[0,30], so the
        // crossing must be between y=0 and y=30.
        let mut found_crossing = false;
        for y in 0..39 {
            let a = buf.get_voxel_f(0, y, 0, ChannelId::Sdf.index());
            let b = buf.get_voxel_f(0, y + 1, 0, ChannelId::Sdf.index());
            if (a < 0.0) != (b < 0.0) {
                found_crossing = true;
                assert!(y < 30, "surface crossing at y={y} exceeds height_range 30");
                break;
            }
        }
        assert!(
            found_crossing,
            "no SDF sign change found; heightmap surface missing"
        );
    }

    // ---- heightmap offset (via Waves integration) -----------------------

    #[test]
    fn waves_heightmap_offset_shifts_origin() {
        let mut gen = Waves::default();
        gen.heightmap.offset = Vector2i::new(100, 0);
        let mut buf = sdf_buffer(Vector3i::new(1, 40, 1));
        gen.generate_block(VoxelQueryData {
            buffer: &mut buf,
            origin_in_voxels: Vector3i::new(0, 0, 0),
            lod: 0,
        });
        // With offset 100, sampling at world x=0 is the same as sampling at
        // x=-100 with no offset. Just verify the generator runs and produces a
        // crossing inside the height range (sanity check the offset path).
        let mut found_crossing = false;
        for y in 0..39 {
            let a = buf.get_voxel_f(0, y, 0, ChannelId::Sdf.index());
            let b = buf.get_voxel_f(0, y + 1, 0, ChannelId::Sdf.index());
            if (a < 0.0) != (b < 0.0) {
                found_crossing = true;
                break;
            }
        }
        assert!(found_crossing, "offset path produced no surface");
    }

    // ---- Noise: 3D SDF generator ----------------------------------------

    fn configured_noise() -> Noise {
        let mut gen = Noise::default();
        // Small height range so a small buffer spans the slab; default seed.
        gen.set_height_start(0.0);
        gen.set_height_range(10.0);
        gen.noise_mut()
            .set_noise_type(Some(fastnoise_lite::NoiseType::OpenSimplex2));
        gen.noise_mut().set_frequency(Some(0.1));
        gen.noise_mut().set_seed(Some(1337));
        gen
    }

    #[test]
    fn noise_set_height_range_clamps_below_minimum() {
        let mut gen = Noise::default();
        gen.set_height_range(0.01);
        assert!((gen.height_range - 0.1).abs() < 1e-5);
    }

    #[test]
    fn noise_used_channels_mask_defaults_to_sdf() {
        let gen = configured_noise();
        let g: &dyn VoxelGenerator = &gen;
        assert_eq!(g.used_channels_mask(), 1 << ChannelId::Sdf.index());
    }

    #[test]
    fn noise_period_tracks_configured_frequency() {
        let mut gen = Noise::default();
        assert!((gen.noise_period() - 100.0).abs() < 1e-5);

        gen.noise_mut().set_frequency(Some(0.05));
        assert!((gen.noise_period() - 20.0).abs() < 1e-5);

        gen.noise_mut().set_frequency(Some(0.0));
        assert!((gen.noise_period() - 10_000.0).abs() < 1e-3);
    }

    #[test]
    fn noise_early_exit_above_slab_is_far_outside() {
        let gen = configured_noise();
        // Slab is [0, 10]; place the block well above it.
        let mut buf = VoxelBuffer::with_size(Vector3i::new(2, 2, 2));
        buf.set_channel_depth(ChannelId::Sdf.index(), crate::storage::ChannelDepth::Bit32);
        let result = gen.generate_block(VoxelQueryData {
            buffer: &mut buf,
            origin_in_voxels: Vector3i::new(0, 100, 0),
            lod: 0,
        });
        assert!(result.max_lod_hint);
        let v = buf.get_voxel_f(0, 0, 0, ChannelId::Sdf.index());
        assert!((v - SDF_FAR_OUTSIDE).abs() < 1e-3, "above-slab SDF: {v}");
    }

    #[test]
    fn noise_early_exit_below_slab_is_far_inside() {
        let gen = configured_noise();
        let mut buf = VoxelBuffer::with_size(Vector3i::new(2, 2, 2));
        buf.set_channel_depth(ChannelId::Sdf.index(), crate::storage::ChannelDepth::Bit32);
        let result = gen.generate_block(VoxelQueryData {
            buffer: &mut buf,
            origin_in_voxels: Vector3i::new(0, -100, 0),
            lod: 0,
        });
        assert!(result.max_lod_hint);
        let v = buf.get_voxel_f(0, 0, 0, ChannelId::Sdf.index());
        assert!((v - SDF_FAR_INSIDE).abs() < 1e-3, "below-slab SDF: {v}");
    }

    #[test]
    fn noise_deterministic_for_same_seed() {
        // Same seed + params must produce identical SDF at the same voxel.
        let gen_a = configured_noise();
        let gen_b = configured_noise();
        let mut buf_a = VoxelBuffer::with_size(Vector3i::new(2, 4, 2));
        let mut buf_b = VoxelBuffer::with_size(Vector3i::new(2, 4, 2));
        buf_a.set_channel_depth(ChannelId::Sdf.index(), crate::storage::ChannelDepth::Bit32);
        buf_b.set_channel_depth(ChannelId::Sdf.index(), crate::storage::ChannelDepth::Bit32);
        gen_a.generate_block(VoxelQueryData {
            buffer: &mut buf_a,
            origin_in_voxels: Vector3i::new(0, 0, 0),
            lod: 0,
        });
        gen_b.generate_block(VoxelQueryData {
            buffer: &mut buf_b,
            origin_in_voxels: Vector3i::new(0, 0, 0),
            lod: 0,
        });
        for z in 0..2 {
            for y in 0..4 {
                for x in 0..2 {
                    assert_eq!(
                        buf_a.get_voxel_f(x, y, z, ChannelId::Sdf.index()).to_bits(),
                        buf_b.get_voxel_f(x, y, z, ChannelId::Sdf.index()).to_bits(),
                        "noise diverged at ({x},{y},{z})"
                    );
                }
            }
        }
    }

    #[test]
    fn noise_slab_contains_sign_change() {
        // The slab spans y=[0,10]; the SDF must transition from negative
        // (solid, near the bottom) to positive (air, near the top).
        let gen = configured_noise();
        let mut buf = VoxelBuffer::with_size(Vector3i::new(4, 12, 4));
        buf.set_channel_depth(ChannelId::Sdf.index(), crate::storage::ChannelDepth::Bit32);
        gen.generate_block(VoxelQueryData {
            buffer: &mut buf,
            origin_in_voxels: Vector3i::new(0, 0, 0),
            lod: 0,
        });
        // Sample the centre column; expect a crossing somewhere in [0,12).
        let mut found_solid = false;
        let mut found_air = false;
        for y in 0..12 {
            let v = buf.get_voxel_f(2, y, 2, ChannelId::Sdf.index());
            if v < 0.0 {
                found_solid = true;
            } else {
                found_air = true;
            }
        }
        assert!(found_solid, "no solid voxels in the slab column");
        assert!(found_air, "no air voxels in the slab column");
    }

    #[test]
    fn noise_blocky_channel_writes_zero_or_one() {
        let mut gen = configured_noise();
        gen.set_channel(ChannelId::Type);
        let mut buf = VoxelBuffer::with_size(Vector3i::new(2, 12, 2));
        gen.generate_block(VoxelQueryData {
            buffer: &mut buf,
            origin_in_voxels: Vector3i::new(0, 0, 0),
            lod: 0,
        });
        // Blocky channel only ever writes 0 (air) or 1 (matter).
        for z in 0..2 {
            for y in 0..12 {
                for x in 0..2 {
                    let v = buf.get_voxel(x, y, z, ChannelId::Type.index());
                    assert!(v == 0 || v == 1, "blocky noise wrote {v} at ({x},{y},{z})");
                }
            }
        }
    }

    // ---- HeightmapNoise: 2D noise → heightmap --------------------------

    #[test]
    fn curve_identity_samples_linearly() {
        let c = Curve::identity(11);
        for i in 0..=10 {
            let t = i as f32 / 10.0;
            assert!(
                (c.sample(t) - t).abs() < 1e-4,
                "identity curve at {t}: {}",
                c.sample(t)
            );
        }
    }

    #[test]
    fn curve_clamps_out_of_range() {
        let c = Curve::identity(11);
        assert!((c.sample(-1.0) - 0.0).abs() < 1e-4);
        assert!((c.sample(2.0) - 1.0).abs() < 1e-4);
    }

    #[test]
    fn curve_from_points_interpolates() {
        // Points: [0.0, 10.0] → at t=0.5 should be 5.0.
        let c = Curve::from_points(vec![0.0, 10.0]);
        assert!((c.sample(0.5) - 5.0).abs() < 1e-5);
    }

    #[test]
    fn heightmap_noise_produces_surface_in_height_range() {
        let mut gen = HeightmapNoise::default();
        gen.heightmap.height_start = 0.0;
        gen.heightmap.height_range = 20.0;
        gen.noise_config.frequency = Some(0.1);
        gen.noise_config.noise_type = Some(fastnoise_lite::NoiseType::OpenSimplex2);
        gen.noise_config.seed = Some(42);

        let mut buf = VoxelBuffer::with_size(Vector3i::new(4, 30, 4));
        buf.set_channel_depth(ChannelId::Sdf.index(), crate::storage::ChannelDepth::Bit32);
        gen.generate_block(VoxelQueryData {
            buffer: &mut buf,
            origin_in_voxels: Vector3i::new(0, 0, 0),
            lod: 0,
        });

        // Heightmap in [0, 20] → SDF must transition from solid to air.
        let mut found_solid = false;
        let mut found_air = false;
        for y in 0..30 {
            let v = buf.get_voxel_f(2, y, 2, ChannelId::Sdf.index());
            if v < 0.0 {
                found_solid = true;
            } else {
                found_air = true;
            }
        }
        assert!(found_solid, "no solid voxels from heightmap noise");
        assert!(found_air, "no air voxels from heightmap noise");
    }

    #[test]
    fn heightmap_noise_with_curve_remaps_height() {
        // A curve that inverts: [0, 1] → [1, 0]. The surface should still
        // appear (solid below, air above), just at inverted heights.
        let mut gen = HeightmapNoise::default();
        gen.heightmap.height_start = 0.0;
        gen.heightmap.height_range = 20.0;
        gen.noise_config.frequency = Some(0.1);
        gen.noise_config.seed = Some(7);
        gen.set_curve(Some(Curve::from_points(vec![1.0, 0.0])));

        let mut buf = VoxelBuffer::with_size(Vector3i::new(2, 30, 2));
        buf.set_channel_depth(ChannelId::Sdf.index(), crate::storage::ChannelDepth::Bit32);
        gen.generate_block(VoxelQueryData {
            buffer: &mut buf,
            origin_in_voxels: Vector3i::new(0, 0, 0),
            lod: 0,
        });
        let mut found_solid = false;
        let mut found_air = false;
        for y in 0..30 {
            let v = buf.get_voxel_f(0, y, 0, ChannelId::Sdf.index());
            if v < 0.0 {
                found_solid = true;
            } else {
                found_air = true;
            }
        }
        assert!(
            found_solid && found_air,
            "curve-remapped heightmap produced no surface"
        );
    }

    #[test]
    fn heightmap_noise_curve_can_be_arc_shared() {
        let curve = Arc::new(Curve::identity(257));
        let mut gen = HeightmapNoise::default();

        gen.set_curve_arc(Some(Arc::clone(&curve)));

        let stored = gen.curve.as_ref().expect("curve should be stored");
        assert!(Arc::ptr_eq(stored, &curve));
        assert_eq!(Arc::strong_count(&curve), 2);
    }

    #[test]
    fn heightmap_noise_used_channels_mask_defaults_to_sdf() {
        let gen = HeightmapNoise::default();
        let g: &dyn VoxelGenerator = &gen;
        assert_eq!(g.used_channels_mask(), 1 << ChannelId::Sdf.index());
    }

    // ---- Image: heightmap sampling ---------------------------------------

    fn ramp_image(width: i32, height: i32) -> Vec<f32> {
        // Height 0 at z=0, 1 at the last row; constant along x.
        (0..height)
            .flat_map(|z| vec![z as f32 / (height - 1).max(1) as f32; width as usize])
            .collect()
    }

    #[test]
    fn image_set_rejects_size_mismatch() {
        let mut gen = Image::default();
        assert!(!gen.set_image(vec![0.5; 3], 2, 2));
        assert!(!gen.set_image(Vec::new(), 0, 0));
        assert!(gen.set_image(vec![0.5; 4], 2, 2));
        assert!(gen.has_image());
    }

    #[test]
    fn image_clamp_wraps_out_of_range_coordinates() {
        let mut gen = Image::default();
        gen.wrap = ImageWrapMode::Clamp;
        assert!(gen.set_image(ramp_image(4, 4), 4, 4));
        assert_eq!(gen.height_at(0, 0), 0.0);
        assert_eq!(gen.height_at(0, 3), 1.0);
        // Clamped beyond the edges.
        assert_eq!(gen.height_at(-10, -10), 0.0);
        assert_eq!(gen.height_at(100, 100), 1.0);
    }

    #[test]
    fn image_repeat_wraps_around() {
        let mut gen = Image::default();
        gen.wrap = ImageWrapMode::Repeat;
        assert!(gen.set_image(ramp_image(4, 4), 4, 4));
        assert_eq!(gen.height_at(0, 4), gen.height_at(0, 0));
        assert_eq!(gen.height_at(0, 7), gen.height_at(0, 3));
        assert_eq!(gen.height_at(0, -1), gen.height_at(0, 3));
    }

    #[test]
    fn image_sdf_generation_follows_heightmap() {
        let mut gen = Image::default();
        assert!(gen.set_image(ramp_image(4, 4), 4, 4));
        gen.heightmap.height_start = 0.0;
        gen.heightmap.height_range = 8.0;
        let mut buf = sdf_buffer(Vector3i::new(4, 8, 4));
        buf.set_channel_depth(ChannelId::Sdf.index(), crate::storage::ChannelDepth::Bit32);
        gen.generate_block(VoxelQueryData {
            buffer: &mut buf,
            origin_in_voxels: Vector3i::new(0, 0, 0),
            lod: 0,
        });
        // z=0 row has height 0 → solid below y=0 only; deep in the block
        // (z=3 row, height 8) the bottom voxels must be solid (sdf < 0).
        assert!(buf.get_voxel_f(0, 0, 3, ChannelId::Sdf.index()) < 0.0);
        // Above the maximum height everything is air.
        let mut buf2 = sdf_buffer(Vector3i::new(4, 8, 4));
        buf2.set_channel_depth(ChannelId::Sdf.index(), crate::storage::ChannelDepth::Bit32);
        gen.generate_block(VoxelQueryData {
            buffer: &mut buf2,
            origin_in_voxels: Vector3i::new(0, 32, 0),
            lod: 0,
        });
        assert!(buf2.get_voxel_f(0, 0, 0, ChannelId::Sdf.index()) > 0.0);
    }

    #[test]
    fn image_blocky_fills_type_channel_up_to_height() {
        let mut gen = Image::default();
        // Uniform image at height 1.0 → fills the whole 4-voxel column.
        assert!(gen.set_image(vec![1.0; 16], 4, 4));
        gen.heightmap.channel = ChannelId::Type;
        gen.heightmap.height_start = 0.0;
        gen.heightmap.height_range = 4.0;
        gen.heightmap.matter_type = 7;
        let mut buf = sdf_buffer(Vector3i::new(4, 4, 4));
        gen.generate_block(VoxelQueryData {
            buffer: &mut buf,
            origin_in_voxels: Vector3i::new(0, 0, 0),
            lod: 0,
        });
        assert_eq!(buf.get_voxel(1, 0, 1, ChannelId::Type.index()), 7);
        assert_eq!(buf.get_voxel(1, 3, 1, ChannelId::Type.index()), 7);
        let g: &dyn VoxelGenerator = &gen;
        assert_eq!(g.used_channels_mask(), 1 << ChannelId::Type.index());
    }

    // ---- Image: blur_enabled (pinned VoxelGeneratorImage property) --------

    #[test]
    fn image_blur_preserves_mean_and_smooths_delta_peak() {
        // 5x5 field of zeros with a single delta peak at the centre.
        let mut peak = vec![0.0; 25];
        peak[2 + 2 * 5] = 1.0;

        let mut blurred = Image::default();
        blurred.blur_enabled = true;
        assert!(blurred.set_image(peak.clone(), 5, 5));

        let mut plain = Image::default();
        assert!(plain.set_image(peak, 5, 5));

        let mean = |image: &Image| -> f32 {
            let mut sum = 0.0;
            for z in 0..5 {
                for x in 0..5 {
                    sum += image.height_at(x, z);
                }
            }
            sum / 25.0
        };
        // The 5-tap kernel is sum-preserving (each pixel's value is spread
        // over itself and its four neighbours with total weight 1), so the
        // mean is unchanged.
        assert!(
            (mean(&blurred) - mean(&plain)).abs() < 1e-5,
            "blur changed the mean: {} vs {}",
            mean(&blurred),
            mean(&plain)
        );
        // The delta peak spreads to exactly the cross neighbours: upstream's
        // `get_height_blurred` weights the centre and the 4 orthogonal taps by
        // 0.2 each; diagonal pixels receive nothing.
        assert!((blurred.height_at(2, 2) - 0.2).abs() < 1e-6);
        assert!((blurred.height_at(3, 2) - 0.2).abs() < 1e-6);
        assert!((blurred.height_at(1, 2) - 0.2).abs() < 1e-6);
        assert!((blurred.height_at(2, 3) - 0.2).abs() < 1e-6);
        assert!((blurred.height_at(2, 1) - 0.2).abs() < 1e-6);
        assert_eq!(blurred.height_at(1, 1), 0.0, "diagonals stay untouched");
        assert_eq!(blurred.height_at(3, 3), 0.0);
        assert_eq!(blurred.height_at(0, 0), 0.0);
    }

    #[test]
    fn image_blur_fetches_wrap_around_across_borders() {
        // Upstream `get_height_repeat` wraps coordinates modulo the image
        // size: a peak at the left edge must bleed into the right edge.
        let mut peak = vec![0.0; 16];
        peak[2 * 4] = 1.0; // 4x4, peak at (0, 2): index x + z * 4.
        let mut blurred = Image::default();
        blurred.blur_enabled = true;
        assert!(blurred.set_image(peak, 4, 4));
        // (3, 2) is the left peak's x-1 neighbour via wrap-around.
        assert!(
            (blurred.height_at(3, 2) - 0.2).abs() < 1e-6,
            "peak must wrap to the opposite edge, got {}",
            blurred.height_at(3, 2)
        );
        // Same vertically: peak at (2, 0) bleeds into (2, 3).
        let mut peak_v = vec![0.0; 16];
        peak_v[2] = 1.0; // 4x4, peak at (2, 0): index x + z * 4 = 2.
        let mut blurred_v = Image::default();
        blurred_v.blur_enabled = true;
        assert!(blurred_v.set_image(peak_v, 4, 4));
        assert!((blurred_v.height_at(2, 3) - 0.2).abs() < 1e-6);
    }

    #[test]
    fn image_blur_keeps_values_clamped_in_unit_range() {
        let mut blurred = Image::default();
        blurred.blur_enabled = true;
        assert!(blurred.set_image(vec![1.0; 16], 4, 4));
        for z in 0..4 {
            for x in 0..4 {
                assert!((0.0..=1.0).contains(&blurred.height_at(x, z)));
            }
        }
    }

    // ---- NoiseConfig: fractal passthrough from noise resources -----------

    #[test]
    fn noise_config_build_applies_fractal_settings() {
        use fastnoise_lite::FractalType;
        let base = NoiseConfig {
            seed: Some(11),
            frequency: Some(0.05),
            ..Default::default()
        };
        let fractal = NoiseConfig {
            fractal_type: Some(FractalType::FBm),
            fractal_octaves: Some(4),
            fractal_lacunarity: Some(2.0),
            fractal_gain: Some(0.5),
            fractal_weighted_strength: Some(0.3),
            ..base.clone()
        };
        // Deterministic: the same config rebuilds to identical samples.
        let a = fractal.build().get_noise_3d(1.0, 2.0, 3.0);
        let b = fractal.build().get_noise_3d(1.0, 2.0, 3.0);
        assert_eq!(a.to_bits(), b.to_bits());
        // The fractal settings genuinely change the sampler output.
        let plain = base.build().get_noise_3d(1.0, 2.0, 3.0);
        assert_ne!(a.to_bits(), plain.to_bits());
        // PingPong strength only affects the PingPong fractal.
        let ping_pong = NoiseConfig {
            fractal_type: Some(FractalType::PingPong),
            fractal_ping_pong_strength: Some(4.0),
            ..fractal.clone()
        };
        let c = ping_pong.build().get_noise_3d(1.0, 2.0, 3.0);
        assert_ne!(c.to_bits(), a.to_bits());
    }
}
