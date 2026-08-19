//! Godot Resource bindings for voxel generators.
//!
//! Each `VoxelGenerator*` is a Godot `Resource` that wraps the corresponding
//! `voxel_core::generators::simple::*` type. When attached to a
//! [`VoxelTerrain`](crate::terrain::VoxelTerrain) via the `generator` property,
//! it produces voxel data on demand.
//!
//! ## Class hierarchy note (deviation from upstream)
//!
//! Upstream inherits: `VoxelGenerator` → Flat/Noise/Heightmap → Image/Waves.
//! gdext 0.5 (the pinned `godot` crate) does not support inheriting
//! user-defined Rust classes — a class attribute with `base = <own struct>`
//! expands to a `godot::classes::<Struct>` lookup that only covers engine
//! classes — so all six bindings remain flat `base = Resource`. The inherited
//! pinned members (e.g. `height_start`/`iso_scale`/`offset` on Image and
//! Waves) are re-declared on each subclass instead, keeping the pinned
//! *surface* complete even though ClassDB parentage is `Resource`.

use godot::classes::Image as GodotImage;
use godot::classes::Object;
use godot::prelude::*;
use std::sync::Arc;

use voxel_core::fastnoise_lite::{FractalType, NoiseType};
use voxel_core::generators::base::{
    HeightmapParams, VoxelGenerator as CoreVoxelGenerator, VoxelQueryData,
};
use voxel_core::generators::simple::{
    Flat, HeightmapNoise, Image, ImageWrapMode, Noise, NoiseConfig, Waves,
};
use voxel_core::storage::voxel_buffer::channel_id_from_index;
use voxel_core::storage::{ChannelId, SharedVoxelGenerator};

fn validate_finite_float(value: f32) -> Result<f32, &'static str> {
    if !value.is_finite() {
        return Err("value must be finite");
    }
    Ok(value)
}

fn validate_positive_finite_float(value: f32) -> Result<f32, &'static str> {
    if !value.is_finite() || value <= 0.0 {
        return Err("value must be finite and strictly positive");
    }
    Ok(value)
}

fn validate_wave_configuration(
    amplitude: f32,
    frequency: f32,
    period: f32,
) -> Result<(f32, f32, f32), &'static str> {
    Ok((
        validate_finite_float(amplitude)?,
        validate_positive_finite_float(frequency)?,
        validate_positive_finite_float(period)?,
    ))
}

fn validate_sample_coordinates(coordinates: &[f32]) -> Result<(), &'static str> {
    if coordinates.iter().any(|coordinate| !coordinate.is_finite()) {
        return Err("sample coordinates must be finite");
    }
    Ok(())
}

/// Validate a canonical `channel` value. Must be a valid `ChannelId` (0..=7),
/// matching `voxel_core::storage::voxel_buffer::ChannelId` (`#[repr(u8)]`).
fn validate_channel_id(value: i32) -> Result<i32, &'static str> {
    if (0..=7).contains(&value) {
        Ok(value)
    } else {
        Err("channel must be a valid ChannelId (0..=7)")
    }
}

/// Saturating `i64 → i32` conversion for forwarding the pinned `seed`
/// property into core noise configs. A plain `as` cast would wrap modulo
/// 2³², silently producing a different seed; out-of-range values saturate
/// to `i32::MAX`/`i32::MIN` instead. Free function so the behavior is
/// pinnable in plain `cargo test`.
fn saturating_seed_i32(seed: i64) -> i32 {
    i32::try_from(seed).unwrap_or(if seed >= 0 { i32::MAX } else { i32::MIN })
}

// ---------------------------------------------------------------------------
// Shared generate_block plumbing (pinned VoxelGenerator method)
// ---------------------------------------------------------------------------

/// Convert the pinned `generate_block` `origin_in_voxels` parameter (upstream
/// binds it as `Vector3`) into the integer voxel origin used by the core
/// generators. Components are truncated toward zero, mirroring Godot's
/// `Vector3 → Vector3i` conversion (and upstream's C++ cast); `as` saturates
/// out-of-range inputs and maps NaN to 0.
fn core_origin_from_vector3(origin: Vector3) -> voxel_core::math::Vector3i {
    voxel_core::math::Vector3i::new(origin.x as i32, origin.y as i32, origin.z as i32)
}

/// Run a core generator into the Godot-facing buffer. Shared body of the
/// pinned `generate_block` method implemented by every concrete generator.
pub(crate) fn generate_core_block_into_gd(
    generator: &dyn CoreVoxelGenerator,
    out_buffer: &mut Gd<crate::voxel_buffer::VoxelBufferGD>,
    origin_in_voxels: Vector3,
    lod: i32,
) {
    let lod = u32::try_from(lod.max(0)).unwrap_or(0);
    let mut bound = out_buffer.bind_mut();
    generator.generate_block(VoxelQueryData {
        buffer: bound.core_buffer_mut(),
        origin_in_voxels: core_origin_from_vector3(origin_in_voxels),
        lod,
    });
}

/// Assemble the shared heightmap parameter block from the pinned
/// `VoxelGeneratorHeightmap` members (channel, height_start, height_range,
/// iso_scale, offset). Free function so the forwarding is pinnable in plain
/// `cargo test` without a live engine.
fn heightmap_params_from_godot(
    channel: i32,
    height_start: f32,
    height_range: f32,
    iso_scale: f32,
    offset: Vector2i,
) -> HeightmapParams {
    HeightmapParams {
        channel: channel_id_from_index(channel as usize).unwrap_or(ChannelId::Sdf),
        matter_type: 1,
        height_start,
        height_range,
        iso_scale,
        offset: voxel_core::math::Vector2i::new(offset.x, offset.y),
    }
}

// ---------------------------------------------------------------------------
// VoxelGeneratorWaves
// ---------------------------------------------------------------------------

/// A simple SDF terrain generator that produces rolling waves along the X axis.
/// Wraps [`voxel_core::generators::simple::Waves`].
///
/// In GDScript: create a `VoxelGeneratorWaves` resource and assign it to a
/// `VoxelTerrain`'s `generator` property.
///
/// Upstream inherits `VoxelGeneratorHeightmap`; since gdext cannot inherit
/// user classes (see the module-level note), the inherited pinned members are
/// re-declared here. `amplitude`/`frequency`/`period` are legacy extras kept
/// for source compatibility: `amplitude` and `period` write through to
/// `height_range` and `pattern_size`; `frequency` never influenced generation
/// and remains a stored no-op.
#[derive(GodotClass)]
#[class(base = Resource, tool)]
pub struct VoxelGeneratorWaves {
    base: Base<Resource>,
    /// Canonical `height_start` (backing field; finite).
    height_start_value: f32,
    /// Canonical `height_range` (backing field; finite). Effective default 30
    /// (upstream `VoxelGeneratorWaves` overrides the base default).
    height_range_value: f32,
    /// Canonical `iso_scale` (backing field; strictly positive finite).
    iso_scale_value: f32,
    /// Canonical `channel` (VoxelBuffer channel id 0..=7).
    channel_value: i32,
    /// Canonical `offset` (backing field).
    offset_value: Vector2i,
    /// Canonical `pattern_offset` (backing field; finite components).
    pattern_offset_value: Vector2,
    /// Canonical `pattern_size` (backing field; strictly positive finite).
    pattern_size_value: Vector2,
    /// Deprecated legacy alias of `height_range`.
    amplitude_value: f32,
    /// Deprecated legacy no-op (never influenced generation).
    frequency_value: f32,
    /// Deprecated legacy alias of `pattern_size` (both components).
    period_value: f32,
    #[var(get = get_height_start, set = set_height_start)]
    height_start: PhantomVar<f32>,
    #[var(get = get_height_range, set = set_height_range)]
    height_range: PhantomVar<f32>,
    #[var(get = get_iso_scale, set = set_iso_scale)]
    iso_scale: PhantomVar<f32>,
    #[var(get = get_channel, set = set_channel)]
    channel: PhantomVar<i32>,
    #[var(get = get_offset, set = set_offset)]
    offset: PhantomVar<Vector2i>,
    #[var(get = get_pattern_offset, set = set_pattern_offset)]
    pattern_offset: PhantomVar<Vector2>,
    #[var(get = get_pattern_size, set = set_pattern_size)]
    pattern_size: PhantomVar<Vector2>,
    #[var(get = get_amplitude, set = set_amplitude)]
    amplitude: PhantomVar<f32>,
    #[var(get = get_frequency, set = set_frequency)]
    frequency: PhantomVar<f32>,
    #[var(get = get_period, set = set_period)]
    period: PhantomVar<f32>,
}

#[godot_api]
impl IResource for VoxelGeneratorWaves {
    fn init(base: Base<Resource>) -> Self {
        // Upstream defaults: the Waves ctor sets pattern_size (30, 30) and
        // height_range 30 on top of the heightmap base defaults
        // (height_start -50, iso_scale 1, offset 0, channel SDF).
        Self {
            base,
            height_start_value: -50.0,
            height_range_value: 30.0,
            iso_scale_value: 1.0,
            channel_value: 1,
            offset_value: Vector2i::new(0, 0),
            pattern_offset_value: Vector2::new(0.0, 0.0),
            pattern_size_value: Vector2::new(30.0, 30.0),
            amplitude_value: 30.0,
            frequency_value: 0.02,
            period_value: 30.0,
            height_start: PhantomVar::default(),
            height_range: PhantomVar::default(),
            iso_scale: PhantomVar::default(),
            channel: PhantomVar::default(),
            offset: PhantomVar::default(),
            pattern_offset: PhantomVar::default(),
            pattern_size: PhantomVar::default(),
            amplitude: PhantomVar::default(),
            frequency: PhantomVar::default(),
            period: PhantomVar::default(),
        }
    }
}

/// Build the core Waves generator from the canonical pinned parameters. Free
/// function so the canonical forwarding is pinnable in plain `cargo test`
/// (regression guard: `pattern_size`/`pattern_offset` used to be ignored by
/// the wrapper in favour of legacy `period`/`amplitude`).
fn waves_generator_from_params(
    pattern_size: Vector2,
    pattern_offset: Vector2,
    params: HeightmapParams,
) -> Waves {
    Waves {
        pattern_size: voxel_core::math::Vector2f::new(pattern_size.x, pattern_size.y),
        pattern_offset: voxel_core::math::Vector2f::new(pattern_offset.x, pattern_offset.y),
        heightmap: params,
    }
}

#[godot_api]
impl VoxelGeneratorWaves {
    /// Construct the engine-agnostic generator from the current parameters.
    pub fn create_core_generator(&self) -> SharedVoxelGenerator {
        if validate_wave_configuration(
            self.amplitude_value,
            self.frequency_value,
            self.period_value,
        )
        .is_err()
            || validate_finite_float(self.height_start_value).is_err()
            || validate_finite_float(self.height_range_value).is_err()
            || validate_positive_finite_float(self.iso_scale_value).is_err()
            || validate_positive_finite_float(self.pattern_size_value.x).is_err()
            || validate_positive_finite_float(self.pattern_size_value.y).is_err()
            || !self.pattern_offset_value.x.is_finite()
            || !self.pattern_offset_value.y.is_finite()
        {
            godot_error!("VoxelGeneratorWaves: heights must be finite, iso_scale/pattern_size/frequency/period must be finite and positive");
            return Arc::new(Waves::default());
        }
        let params = heightmap_params_from_godot(
            self.channel_value,
            self.height_start_value,
            self.height_range_value,
            self.iso_scale_value,
            self.offset_value,
        );
        Arc::new(waves_generator_from_params(
            self.pattern_size_value,
            self.pattern_offset_value,
            params,
        ))
    }

    /// Generates a block of voxels within the specified world area (pinned
    /// `generate_block`, upstream `VoxelGenerator.xml`).
    #[func]
    fn generate_block(
        &self,
        mut out_buffer: Gd<crate::voxel_buffer::VoxelBufferGD>,
        origin_in_voxels: Vector3,
        lod: i32,
    ) {
        let generator = self.create_core_generator();
        generate_core_block_into_gd(generator.as_ref(), &mut out_buffer, origin_in_voxels, lod);
    }

    /// Canonical `height_start` (world Y origin of the slab; finite).
    #[func]
    fn get_height_start(&self) -> f32 {
        self.height_start_value
    }

    #[func]
    fn set_height_start(&mut self, value: f32) {
        if validate_finite_float(value).is_err() {
            godot_error!("VoxelGeneratorWaves.set_height_start: value must be finite");
            return;
        }
        self.height_start_value = value;
    }

    /// Canonical `height_range` (finite; effective default 30).
    #[func]
    fn get_height_range(&self) -> f32 {
        self.height_range_value
    }

    #[func]
    fn set_height_range(&mut self, value: f32) {
        if validate_finite_float(value).is_err() {
            godot_error!("VoxelGeneratorWaves.set_height_range: value must be finite");
            return;
        }
        self.height_range_value = value;
        // Keep the deprecated amplitude alias in sync.
        self.amplitude_value = value;
    }

    /// Canonical `iso_scale` (strictly positive finite).
    #[func]
    fn get_iso_scale(&self) -> f32 {
        self.iso_scale_value
    }

    #[func]
    fn set_iso_scale(&mut self, value: f32) {
        if validate_positive_finite_float(value).is_err() {
            godot_error!("VoxelGeneratorWaves.set_iso_scale: value must be finite and positive");
            return;
        }
        self.iso_scale_value = value;
    }

    /// Canonical `channel` (valid VoxelBuffer channel id 0..=7).
    #[func]
    fn get_channel(&self) -> i32 {
        self.channel_value
    }

    #[func]
    fn set_channel(&mut self, value: i32) {
        if validate_channel_id(value).is_err() {
            godot_error!("VoxelGeneratorWaves.set_channel: must be a valid channel id (0..=7)");
            return;
        }
        self.channel_value = value;
    }

    /// Canonical `offset` (block-aligned chunk origin offset).
    #[func]
    fn get_offset(&self) -> Vector2i {
        self.offset_value
    }

    #[func]
    fn set_offset(&mut self, value: Vector2i) {
        self.offset_value = value;
    }

    /// Canonical `pattern_offset` (both components must be finite).
    #[func]
    fn get_pattern_offset(&self) -> Vector2 {
        self.pattern_offset_value
    }

    #[func]
    fn set_pattern_offset(&mut self, value: Vector2) {
        if !value.x.is_finite() || !value.y.is_finite() {
            godot_error!("VoxelGeneratorWaves.set_pattern_offset: components must be finite");
            return;
        }
        self.pattern_offset_value = value;
    }

    /// Canonical `pattern_size` (both components must be finite and positive).
    #[func]
    fn get_pattern_size(&self) -> Vector2 {
        self.pattern_size_value
    }

    #[func]
    fn set_pattern_size(&mut self, value: Vector2) {
        if validate_positive_finite_float(value.x).is_err()
            || validate_positive_finite_float(value.y).is_err()
        {
            godot_error!(
                "VoxelGeneratorWaves.set_pattern_size: components must be finite and positive"
            );
            return;
        }
        self.pattern_size_value = value;
    }

    /// Deprecated alias of `height_range` (any finite real, including
    /// zero/negative). Writes through to the canonical property.
    #[func]
    fn get_amplitude(&self) -> f32 {
        self.amplitude_value
    }

    #[func]
    fn set_amplitude(&mut self, value: f32) {
        if validate_finite_float(value).is_err() {
            godot_error!("VoxelGeneratorWaves.set_amplitude: value must be finite");
            return;
        }
        self.amplitude_value = value;
        self.height_range_value = value;
    }

    /// Deprecated legacy property (strictly positive finite). It never
    /// influenced generation and is kept only for source compatibility.
    #[func]
    fn get_frequency(&self) -> f32 {
        self.frequency_value
    }

    #[func]
    fn set_frequency(&mut self, value: f32) {
        if validate_positive_finite_float(value).is_err() {
            godot_error!("VoxelGeneratorWaves.set_frequency: value must be finite and positive");
            return;
        }
        self.frequency_value = value;
    }

    /// Deprecated alias of `pattern_size` (writes the same value to both
    /// components; strictly positive finite).
    #[func]
    fn get_period(&self) -> f32 {
        self.period_value
    }

    #[func]
    fn set_period(&mut self, value: f32) {
        if validate_positive_finite_float(value).is_err() {
            godot_error!("VoxelGeneratorWaves.set_period: value must be finite and positive");
            return;
        }
        self.period_value = value;
        self.pattern_size_value = Vector2::new(value, value);
    }
}

// ---------------------------------------------------------------------------
// VoxelGeneratorFlat
// ---------------------------------------------------------------------------

/// A flat terrain generator that fills SDF as a horizontal plane at a given
/// height. Wraps [`voxel_core::generators::simple::Flat`].
#[derive(GodotClass)]
#[class(base = Resource, tool)]
pub struct VoxelGeneratorFlat {
    base: Base<Resource>,
    /// Integer height of the flat surface (back-compat alias of the canonical
    /// float `height`). Exposed as the `height_int` GDScript property.
    #[var(get = get_height_int, set = set_height_int)]
    pub height_int: i64,
    /// Canonical `height` (float) of the flat surface (backing field; finite).
    height_value: f32,
    /// Canonical `channel` (VoxelBuffer channel id 0..=7).
    channel_value: i32,
    /// Canonical `voxel_type` (block type / material id written into the
    /// channel; default 1).
    voxel_type_value: i32,
    #[var(get = get_height, set = set_height)]
    height: PhantomVar<f32>,
    #[var(get = get_channel, set = set_channel)]
    channel: PhantomVar<i32>,
    #[var(get = get_voxel_type, set = set_voxel_type)]
    voxel_type: PhantomVar<i32>,
}

#[godot_api]
impl IResource for VoxelGeneratorFlat {
    fn init(base: Base<Resource>) -> Self {
        Self {
            base,
            height_int: 0,
            height_value: 0.0,
            channel_value: 1,
            voxel_type_value: 1,
            height: PhantomVar::default(),
            channel: PhantomVar::default(),
            voxel_type: PhantomVar::default(),
        }
    }
}

/// Build the core Flat generator from the canonical pinned parameters. Free
/// function so the parameter wiring is pinnable in plain `cargo test`
/// (pattern: `noise_generator_from_params`).
fn flat_generator_from_params(height: f32, channel: i32, voxel_type: i32) -> Flat {
    let channel = channel_id_from_index(channel as usize).unwrap_or(ChannelId::Sdf);
    let voxel_type = u64::try_from(voxel_type.max(0)).unwrap_or(1);
    Flat {
        height,
        channel,
        voxel_type,
        ..Flat::default()
    }
}

#[godot_api]
impl VoxelGeneratorFlat {
    /// Construct the engine-agnostic generator from the current parameters.
    pub fn create_core_generator(&self) -> SharedVoxelGenerator {
        Arc::new(flat_generator_from_params(
            self.height_value,
            self.channel_value,
            self.voxel_type_value,
        ))
    }

    /// Generates a block of voxels within the specified world area (pinned
    /// `generate_block`, upstream `VoxelGenerator.xml`).
    #[func]
    fn generate_block(
        &self,
        mut out_buffer: Gd<crate::voxel_buffer::VoxelBufferGD>,
        origin_in_voxels: Vector3,
        lod: i32,
    ) {
        let generator = self.create_core_generator();
        generate_core_block_into_gd(generator.as_ref(), &mut out_buffer, origin_in_voxels, lod);
    }

    /// Canonical `height` (float; finite).
    #[func]
    fn get_height(&self) -> f32 {
        self.height_value
    }

    #[func]
    fn set_height(&mut self, value: f32) {
        if validate_finite_float(value).is_err() {
            godot_error!("VoxelGeneratorFlat.set_height: value must be finite");
            return;
        }
        self.height_value = value;
        // Keep the integer alias in sync (round-trips through the canonical value).
        self.height_int = value.round() as i64;
    }

    /// Integer alias `height_int` (back-compat).
    #[func]
    fn get_height_int(&self) -> i64 {
        self.height_int
    }

    #[func]
    fn set_height_int(&mut self, value: i64) {
        self.height_int = value;
        self.height_value = value as f32;
    }

    /// Canonical `channel` (valid VoxelBuffer channel id 0..=7).
    #[func]
    fn get_channel(&self) -> i32 {
        self.channel_value
    }

    #[func]
    fn set_channel(&mut self, value: i32) {
        if validate_channel_id(value).is_err() {
            godot_error!("VoxelGeneratorFlat.set_channel: must be a valid channel id (0..=7)");
            return;
        }
        self.channel_value = value;
    }

    /// Canonical `voxel_type` (block type / material id).
    #[func]
    fn get_voxel_type(&self) -> i32 {
        self.voxel_type_value
    }

    #[func]
    fn set_voxel_type(&mut self, value: i32) {
        self.voxel_type_value = value;
    }
}

// ---------------------------------------------------------------------------
// VoxelGeneratorNoise
// ---------------------------------------------------------------------------

/// A 3D noise terrain generator. Produces caves / overhangs via 3D FastNoiseLite.
/// Wraps [`voxel_core::generators::simple::Noise`].
#[derive(GodotClass)]
#[class(base = Resource, tool)]
pub struct VoxelGeneratorNoise {
    base: Base<Resource>,
    /// Random seed for the noise.
    #[var]
    pub seed: i64,
    /// Noise frequency (backing field; strictly positive finite).
    frequency_value: f32,
    /// Bottom of the noise slab, world Y (backing field; finite).
    height_start_value: f32,
    /// Vertical extent of the slab (backing field; finite, may be 0).
    height_range_value: f32,
    /// Canonical `channel` (VoxelBuffer channel id 0..=7).
    channel_value: i32,
    /// Canonical `noise` resource (typically a `ZN_FastNoiseLite`).
    noise_resource: Option<Gd<Resource>>,
    #[var(get = get_frequency, set = set_frequency)]
    frequency: PhantomVar<f32>,
    #[var(get = get_height_start, set = set_height_start)]
    height_start: PhantomVar<f32>,
    #[var(get = get_height_range, set = set_height_range)]
    height_range: PhantomVar<f32>,
    #[var(get = get_channel, set = set_channel)]
    channel: PhantomVar<i32>,
    #[var(get = get_noise, set = set_noise)]
    noise: PhantomVar<Option<Gd<Resource>>>,
}

#[godot_api]
impl IResource for VoxelGeneratorNoise {
    fn init(base: Base<Resource>) -> Self {
        Self {
            base,
            seed: 0,
            frequency_value: 0.05,
            height_start_value: -100.0,
            height_range_value: 200.0,
            channel_value: 1,
            noise_resource: None,
            frequency: PhantomVar::default(),
            height_start: PhantomVar::default(),
            height_range: PhantomVar::default(),
            channel: PhantomVar::default(),
            noise: PhantomVar::default(),
        }
    }
}

/// Build the core noise generator from a complete sampler configuration. The
/// fractal passthrough fields let an assigned noise resource genuinely drive
/// generation. Free function so the wiring is pinnable in plain `cargo test`.
fn noise_generator_from_config(
    config: NoiseConfig,
    height_start: f32,
    height_range: f32,
    channel: i32,
) -> Noise {
    let channel = channel_id_from_index(channel as usize).unwrap_or(ChannelId::Sdf);
    Noise {
        noise: config.build(),
        height_start,
        height_range,
        channel,
    }
}

/// Build the core noise generator from the wrapper's own fallback parameters.
/// Free function so the parameter wiring (especially `channel`, whose
/// ignore-everything bug was fixed alongside this extraction) is pinnable in
/// plain `cargo test` without a live engine.
fn noise_generator_from_params(
    seed: i64,
    frequency: f32,
    height_start: f32,
    height_range: f32,
    channel: i32,
) -> Noise {
    // Use NoiseConfig.build() to avoid direct fastnoise_lite dependency.
    let config = NoiseConfig {
        seed: Some(saturating_seed_i32(seed)),
        frequency: Some(frequency),
        ..NoiseConfig::default()
    };
    noise_generator_from_config(config, height_start, height_range, channel)
}

/// Map a pinned `ZN_FastNoiseLite.NoiseType` enum value (0..=5) onto the
/// fastnoise-lite sampler enum. Unknown encodings fall back to `None` (the
/// sampler default). Free function so the mapping is pinnable in tests.
fn noise_type_from_godot_enum(value: i32) -> Option<NoiseType> {
    match value {
        0 => Some(NoiseType::OpenSimplex2),
        1 => Some(NoiseType::OpenSimplex2S),
        2 => Some(NoiseType::Cellular),
        3 => Some(NoiseType::Perlin),
        4 => Some(NoiseType::ValueCubic),
        5 => Some(NoiseType::Value),
        _ => None,
    }
}

/// Map a pinned `FastNoise2.NoiseType` enum value (0..=6) onto the
/// fastnoise-lite sampler enum, mirroring `FastNoise2GD::build_sampler`:
/// `TYPE_SIMPLEX`(1) maps to OpenSimplex2S, `TYPE_PERLIN`(2) to Perlin,
/// `TYPE_VALUE`(3) to Value and `TYPE_CELLULAR`(4) to Cellular. The
/// `TYPE_ENCODED_NODE_TREE`(5) and `TYPE_CELLULAR_VALUE`(6) sentinels have no
/// fastnoise-lite equivalent and fall back to OpenSimplex2 there — mirrored
/// here. NOTE: this encoding differs from `ZN_FastNoiseLite`'s for the same
/// integers (e.g. 2 means Cellular there, Perlin here). Free function so the
/// mapping is pinnable in tests.
fn fastnoise2_noise_type_from_godot_enum(value: i32) -> Option<NoiseType> {
    match value {
        1 => Some(NoiseType::OpenSimplex2S),
        2 => Some(NoiseType::Perlin),
        3 => Some(NoiseType::Value),
        4 => Some(NoiseType::Cellular),
        // 0, 5 (encoded node tree) and 6 (cellular value): OpenSimplex2
        // fallback, exactly like FastNoise2GD::build_sampler's catch-all.
        _ => Some(NoiseType::OpenSimplex2),
    }
}

/// Decode a resource's `get_noise_type` integer according to the resource's
/// class: the two supported noise resources use different `NoiseType` enum
/// encodings (see the two mapping functions). Any other class has no known
/// encoding, so the decode yields `None` (sampler default) and the caller
/// logs. Free function so the per-class dispatch is pinnable in tests.
fn noise_type_from_resource_enum(class_name: &str, value: i32) -> Option<NoiseType> {
    match class_name {
        "ZN_FastNoiseLite" => noise_type_from_godot_enum(value),
        "FastNoise2" => fastnoise2_noise_type_from_godot_enum(value),
        _ => None,
    }
}

/// Map a pinned `ZN_FastNoiseLite.FractalType` enum value (0..=3) onto the
/// fastnoise-lite sampler enum. Unknown encodings fall back to `None`.
fn fractal_type_from_godot_enum(value: i32) -> Option<FractalType> {
    match value {
        0 => Some(FractalType::None),
        1 => Some(FractalType::FBm),
        2 => Some(FractalType::Ridged),
        3 => Some(FractalType::PingPong),
        _ => None,
    }
}

/// Pure parameter bundle extracted from an assigned noise resource; converts
/// into a [`NoiseConfig`]. Kept separate from the (engine-dependent)
/// ClassDB calls so the mapping itself is pinnable in plain `cargo test`.
#[derive(Debug, Clone, PartialEq)]
struct NoiseConfigParts {
    seed: i32,
    frequency: f32,
    noise_type: Option<NoiseType>,
    fractal_type: Option<FractalType>,
    fractal_octaves: Option<i32>,
    fractal_lacunarity: Option<f32>,
    fractal_gain: Option<f32>,
    fractal_weighted_strength: Option<f32>,
    fractal_ping_pong_strength: Option<f32>,
}

impl From<NoiseConfigParts> for NoiseConfig {
    fn from(parts: NoiseConfigParts) -> Self {
        NoiseConfig {
            seed: Some(parts.seed),
            frequency: Some(parts.frequency),
            noise_type: parts.noise_type,
            fractal_type: parts.fractal_type,
            fractal_octaves: parts.fractal_octaves,
            fractal_lacunarity: parts.fractal_lacunarity,
            fractal_gain: parts.fractal_gain,
            fractal_weighted_strength: parts.fractal_weighted_strength,
            fractal_ping_pong_strength: parts.fractal_ping_pong_strength,
        }
    }
}

/// Call a pinned getter on the assigned noise resource through ClassDB.
/// Returns `None` when the method is missing or the call fails.
fn call_resource_get(resource: &Gd<Resource>, method: &str) -> Option<Variant> {
    let mut object = resource.clone().upcast::<Object>();
    object.try_call(method, &[]).ok()
}

/// Read a complete noise configuration from the user-assigned `noise`
/// resource by calling its pinned getters (`get_seed`, `get_frequency`,
/// `get_noise_type`, `get_fractal_*`). Works for `ZN_FastNoiseLite` and
/// `FastNoise2` resources — but note their `NoiseType` enums use *different*
/// integer encodings, so `get_noise_type` is decoded per resource class (see
/// [`noise_type_from_resource_enum`]). Any other resource class falls back to
/// the sampler's default noise type (with an error logged). Returns `None`
/// when the resource does not provide at least `get_seed` and
/// `get_frequency`.
///
/// NOTE: `Gd` values cannot exist under plain `cargo test`, so this function
/// is deliberately a thin shell — everything testable lives in
/// [`NoiseConfigParts`] and the enum-mapping functions above.
fn noise_config_from_resource(resource: &Gd<Resource>) -> Option<NoiseConfig> {
    let seed: i32 = call_resource_get(resource, "get_seed")?.try_to().ok()?;
    let frequency: f32 = call_resource_get(resource, "get_frequency")?
        .try_to()
        .ok()?;
    if !frequency.is_finite() || frequency <= 0.0 {
        return None;
    }
    let class_name = resource.get_class().to_string();
    let read_i32 = |method: &str| -> Option<i32> {
        call_resource_get(resource, method).and_then(|variant| variant.try_to::<i32>().ok())
    };
    let read_f32 = |method: &str| -> Option<f32> {
        call_resource_get(resource, method)
            .and_then(|variant| variant.try_to::<f32>().ok())
            .filter(|value| value.is_finite())
    };
    let noise_type = read_i32("get_noise_type").and_then(|value| {
        let decoded = noise_type_from_resource_enum(&class_name, value);
        if decoded.is_none() {
            godot_error!(
                "VoxelGeneratorNoise: noise resource of class '{class_name}' has no known \
                 NoiseType encoding; using the sampler's default noise type"
            );
        }
        decoded
    });
    let parts = NoiseConfigParts {
        seed,
        frequency,
        noise_type,
        fractal_type: read_i32("get_fractal_type").and_then(fractal_type_from_godot_enum),
        fractal_octaves: read_i32("get_fractal_octaves").map(|octaves| octaves.max(1)),
        fractal_lacunarity: read_f32("get_fractal_lacunarity"),
        fractal_gain: read_f32("get_fractal_gain"),
        fractal_weighted_strength: read_f32("get_fractal_weighted_strength"),
        fractal_ping_pong_strength: read_f32("get_fractal_ping_pong_strength"),
    };
    Some(parts.into())
}

#[godot_api]
impl VoxelGeneratorNoise {
    pub fn create_core_generator(&self) -> SharedVoxelGenerator {
        if validate_positive_finite_float(self.frequency_value).is_err()
            || validate_finite_float(self.height_start_value).is_err()
            || validate_finite_float(self.height_range_value).is_err()
        {
            godot_error!(
                "VoxelGeneratorNoise: frequency must be positive and all ranges must be finite"
            );
            return Arc::new(Noise::default());
        }
        // Upstream requires a `noise` resource for the generator to work; it
        // owns seed/frequency/noise_type/fractal settings. When one is
        // assigned and exposes the pinned getters, it alone drives the
        // sampler. Ours additionally logs and falls back to the wrapper's own
        // seed/frequency when no usable resource is assigned, so scripts that
        // only set `seed`/`frequency` keep working.
        let generator = match &self.noise_resource {
            Some(resource) => match noise_config_from_resource(resource) {
                Some(config) => noise_generator_from_config(
                    config,
                    self.height_start_value,
                    self.height_range_value,
                    self.channel_value,
                ),
                None => {
                    godot_error!(
                        "VoxelGeneratorNoise: assigned noise resource does not expose \
                         get_seed/get_frequency; falling back to own seed/frequency"
                    );
                    noise_generator_from_params(
                        self.seed,
                        self.frequency_value,
                        self.height_start_value,
                        self.height_range_value,
                        self.channel_value,
                    )
                }
            },
            None => noise_generator_from_params(
                self.seed,
                self.frequency_value,
                self.height_start_value,
                self.height_range_value,
                self.channel_value,
            ),
        };
        Arc::new(generator)
    }

    /// Generates a block of voxels within the specified world area (pinned
    /// `generate_block`, upstream `VoxelGenerator.xml`).
    #[func]
    fn generate_block(
        &self,
        mut out_buffer: Gd<crate::voxel_buffer::VoxelBufferGD>,
        origin_in_voxels: Vector3,
        lod: i32,
    ) {
        let generator = self.create_core_generator();
        generate_core_block_into_gd(generator.as_ref(), &mut out_buffer, origin_in_voxels, lod);
    }

    /// Noise frequency (higher = more detail; strictly positive finite).
    #[func]
    fn get_frequency(&self) -> f32 {
        self.frequency_value
    }

    #[func]
    fn set_frequency(&mut self, value: f32) {
        if validate_positive_finite_float(value).is_err() {
            godot_error!("VoxelGeneratorNoise.set_frequency: value must be finite and positive");
            return;
        }
        self.frequency_value = value;
    }

    /// Bottom of the noise slab (world Y; finite).
    #[func]
    fn get_height_start(&self) -> f32 {
        self.height_start_value
    }

    #[func]
    fn set_height_start(&mut self, value: f32) {
        if validate_finite_float(value).is_err() {
            godot_error!("VoxelGeneratorNoise.set_height_start: value must be finite");
            return;
        }
        self.height_start_value = value;
    }

    /// Vertical extent of the slab (finite, may be 0).
    #[func]
    fn get_height_range(&self) -> f32 {
        self.height_range_value
    }

    #[func]
    fn set_height_range(&mut self, value: f32) {
        if validate_finite_float(value).is_err() {
            godot_error!("VoxelGeneratorNoise.set_height_range: value must be finite");
            return;
        }
        self.height_range_value = value;
    }

    /// Canonical `channel` (valid VoxelBuffer channel id 0..=7).
    #[func]
    fn get_channel(&self) -> i32 {
        self.channel_value
    }

    #[func]
    fn set_channel(&mut self, value: i32) {
        if validate_channel_id(value).is_err() {
            godot_error!("VoxelGeneratorNoise.set_channel: must be a valid channel id (0..=7)");
            return;
        }
        self.channel_value = value;
    }

    /// Canonical `noise` resource (typically a `ZN_FastNoiseLite`).
    #[func]
    fn get_noise(&self) -> Option<Gd<Resource>> {
        self.noise_resource.clone()
    }

    #[func]
    fn set_noise(&mut self, value: Option<Gd<Resource>>) {
        self.noise_resource = value;
    }
}

// ---------------------------------------------------------------------------
// VoxelGeneratorHeightmap
// ---------------------------------------------------------------------------

/// A heightmap terrain generator driven by 2D noise. Produces rolling hills
/// with controllable seed, frequency, and height range.
/// Wraps [`voxel_core::generators::simple::HeightmapNoise`].
#[derive(GodotClass)]
#[class(base = Resource, tool)]
pub struct VoxelGeneratorHeightmap {
    base: Base<Resource>,
    /// Random seed.
    #[var]
    pub seed: i64,
    /// Noise frequency (backing field; strictly positive finite).
    frequency_value: f32,
    /// Height range of the terrain / amplitude (backing field; finite).
    height_range_value: f32,
    /// Canonical `height_start` (backing field; finite).
    height_start_value: f32,
    /// Canonical `iso_scale` (backing field; strictly positive finite).
    iso_scale_value: f32,
    /// Canonical `channel` (VoxelBuffer channel id 0..=7).
    channel_value: i32,
    /// Canonical `offset` (backing field).
    offset_value: Vector2i,
    #[var(get = get_frequency, set = set_frequency)]
    frequency: PhantomVar<f32>,
    #[var(get = get_height_range, set = set_height_range)]
    height_range: PhantomVar<f32>,
    #[var(get = get_height_start, set = set_height_start)]
    height_start: PhantomVar<f32>,
    #[var(get = get_iso_scale, set = set_iso_scale)]
    iso_scale: PhantomVar<f32>,
    #[var(get = get_channel, set = set_channel)]
    channel: PhantomVar<i32>,
    #[var(get = get_offset, set = set_offset)]
    offset: PhantomVar<Vector2i>,
}

#[godot_api]
impl IResource for VoxelGeneratorHeightmap {
    fn init(base: Base<Resource>) -> Self {
        Self {
            base,
            seed: 0,
            frequency_value: 0.02,
            // Pinned upstream default (VoxelGeneratorHeightmap.xml): 30.0.
            height_range_value: 30.0,
            height_start_value: -50.0,
            iso_scale_value: 1.0,
            channel_value: 1,
            offset_value: Vector2i::new(0, 0),
            frequency: PhantomVar::default(),
            height_range: PhantomVar::default(),
            height_start: PhantomVar::default(),
            iso_scale: PhantomVar::default(),
            channel: PhantomVar::default(),
            offset: PhantomVar::default(),
        }
    }
}

/// Build the core 2D-noise heightmap generator from the pinned parameters.
/// Forwards *all* pinned members (height_range, height_start, iso_scale,
/// channel, offset) into the core `HeightmapParams`. Free function so the
/// forwarding is pinnable in plain `cargo test` (regression guard: the
/// wrapper used to forward only `height_range`).
fn heightmap_generator_from_params(
    seed: i64,
    frequency: f32,
    params: HeightmapParams,
) -> HeightmapNoise {
    HeightmapNoise {
        noise_config: NoiseConfig {
            seed: Some(saturating_seed_i32(seed)),
            frequency: Some(frequency),
            ..NoiseConfig::default()
        },
        curve: None,
        heightmap: params,
    }
}

#[godot_api]
impl VoxelGeneratorHeightmap {
    pub fn create_core_generator(&self) -> SharedVoxelGenerator {
        if validate_positive_finite_float(self.frequency_value).is_err()
            || validate_finite_float(self.height_range_value).is_err()
            || validate_finite_float(self.height_start_value).is_err()
            || validate_positive_finite_float(self.iso_scale_value).is_err()
        {
            godot_error!("VoxelGeneratorHeightmap: frequency/iso_scale must be positive and heights must be finite");
            return Arc::new(HeightmapNoise::default());
        }
        let params = heightmap_params_from_godot(
            self.channel_value,
            self.height_start_value,
            self.height_range_value,
            self.iso_scale_value,
            self.offset_value,
        );
        Arc::new(heightmap_generator_from_params(
            self.seed,
            self.frequency_value,
            params,
        ))
    }

    /// Generates a block of voxels within the specified world area (pinned
    /// `generate_block`, upstream `VoxelGenerator.xml`).
    #[func]
    fn generate_block(
        &self,
        mut out_buffer: Gd<crate::voxel_buffer::VoxelBufferGD>,
        origin_in_voxels: Vector3,
        lod: i32,
    ) {
        let generator = self.create_core_generator();
        generate_core_block_into_gd(generator.as_ref(), &mut out_buffer, origin_in_voxels, lod);
    }

    /// Noise frequency (strictly positive finite).
    #[func]
    fn get_frequency(&self) -> f32 {
        self.frequency_value
    }

    #[func]
    fn set_frequency(&mut self, value: f32) {
        if validate_positive_finite_float(value).is_err() {
            godot_error!(
                "VoxelGeneratorHeightmap.set_frequency: value must be finite and positive"
            );
            return;
        }
        self.frequency_value = value;
    }

    /// Height range of the terrain (amplitude; finite, may be 0).
    #[func]
    fn get_height_range(&self) -> f32 {
        self.height_range_value
    }

    #[func]
    fn set_height_range(&mut self, value: f32) {
        if validate_finite_float(value).is_err() {
            godot_error!("VoxelGeneratorHeightmap.set_height_range: value must be finite");
            return;
        }
        self.height_range_value = value;
    }

    /// Canonical `height_start` (world Y origin of the slab; finite).
    #[func]
    fn get_height_start(&self) -> f32 {
        self.height_start_value
    }

    #[func]
    fn set_height_start(&mut self, value: f32) {
        if validate_finite_float(value).is_err() {
            godot_error!("VoxelGeneratorHeightmap.set_height_start: value must be finite");
            return;
        }
        self.height_start_value = value;
    }

    /// Canonical `iso_scale` (strictly positive finite).
    #[func]
    fn get_iso_scale(&self) -> f32 {
        self.iso_scale_value
    }

    #[func]
    fn set_iso_scale(&mut self, value: f32) {
        if validate_positive_finite_float(value).is_err() {
            godot_error!(
                "VoxelGeneratorHeightmap.set_iso_scale: value must be finite and positive"
            );
            return;
        }
        self.iso_scale_value = value;
    }

    /// Canonical `channel` (valid VoxelBuffer channel id 0..=7).
    #[func]
    fn get_channel(&self) -> i32 {
        self.channel_value
    }

    #[func]
    fn set_channel(&mut self, value: i32) {
        if validate_channel_id(value).is_err() {
            godot_error!("VoxelGeneratorHeightmap.set_channel: must be a valid channel id (0..=7)");
            return;
        }
        self.channel_value = value;
    }

    /// Canonical `offset` (block-aligned chunk origin offset).
    #[func]
    fn get_offset(&self) -> Vector2i {
        self.offset_value
    }

    #[func]
    fn set_offset(&mut self, value: Vector2i) {
        self.offset_value = value;
    }

    /// Sample the terrain height at world `(x, z)`. Returns the height value
    /// (noise remapped to `[0, height_range]`). Deterministic for a fixed
    /// seed/frequency.
    #[func]
    fn sample_height(&self, x: f32, z: f32) -> f32 {
        if validate_sample_coordinates(&[x, z]).is_err()
            || validate_positive_finite_float(self.frequency_value).is_err()
            || validate_finite_float(self.height_range_value).is_err()
        {
            godot_error!("VoxelGeneratorHeightmap.sample_height: coordinates/range must be finite and frequency positive");
            return 0.0;
        }
        let config = NoiseConfig {
            seed: Some(saturating_seed_i32(self.seed)),
            frequency: Some(self.frequency_value),
            ..NoiseConfig::default()
        };
        let noise = config.build();
        let n = noise.get_noise_2d(x, z);
        // Match HeightmapNoise's default (no curve): 0.5 + 0.5*noise → height_range.
        (0.5 + 0.5 * n) * self.height_range_value
    }
}

#[cfg(test)]
mod input_validation_tests {
    use super::*;
    #[test]
    fn noise_generator_from_params_wires_the_channel() {
        // Regression: the wrapper used to ignore `channel` and always build
        // an SDF noise generator, so Type-channel terrain never meshed under
        // a blocky mesher.
        let type_gen = noise_generator_from_params(7, 0.05, -8.0, 16.0, 0);
        assert_eq!(type_gen.channel, ChannelId::Type);
        let sdf_gen = noise_generator_from_params(7, 0.05, -8.0, 16.0, 1);
        assert_eq!(sdf_gen.channel, ChannelId::Sdf);
        // Out-of-range ids fall back to SDF like every other generator.
        let bad = noise_generator_from_params(7, 0.05, -8.0, 16.0, 99);
        assert_eq!(bad.channel, ChannelId::Sdf);
    }

    #[test]
    fn generator_float_validation_rejects_every_nonfinite_value() {
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(validate_finite_float(invalid).is_err());
            assert!(validate_positive_finite_float(invalid).is_err());
        }
        assert_eq!(validate_finite_float(-10.0), Ok(-10.0));
    }

    #[test]
    fn wave_period_and_noise_frequency_must_be_strictly_positive() {
        for invalid in [-1.0, -0.0, 0.0] {
            assert!(validate_positive_finite_float(invalid).is_err());
        }
        assert_eq!(validate_positive_finite_float(0.02), Ok(0.02));
        assert_eq!(validate_positive_finite_float(128.0), Ok(128.0));
    }

    #[test]
    fn rejected_generator_assignment_preserves_previous_state() {
        let mut period = 128.0;
        if let Ok(next) = validate_positive_finite_float(0.0) {
            period = next;
        }
        assert_eq!(period, 128.0);

        let mut height_range = 100.0;
        if let Ok(next) = validate_finite_float(f32::INFINITY) {
            height_range = next;
        }
        assert_eq!(height_range, 100.0);
    }

    #[test]
    fn wave_core_configuration_never_accepts_zero_period() {
        assert!(validate_wave_configuration(60.0, 0.02, 0.0).is_err());
        assert!(validate_wave_configuration(60.0, 0.02, f32::NAN).is_err());
        assert_eq!(
            validate_wave_configuration(60.0, 0.02, 128.0),
            Ok((60.0, 0.02, 128.0))
        );
    }

    #[test]
    fn sampling_coordinates_must_be_finite() {
        assert!(validate_sample_coordinates(&[0.0, 1.0, -2.0]).is_ok());
        assert!(validate_sample_coordinates(&[0.0, f32::NAN]).is_err());
        assert!(validate_sample_coordinates(&[f32::INFINITY]).is_err());
    }

    #[test]
    fn channel_id_must_be_in_zero_to_seven_inclusive() {
        // Bounds of the voxel-core ChannelId enum (Type=0 .. Data7=7).
        for valid in 0..=7 {
            assert_eq!(validate_channel_id(valid), Ok(valid));
        }
        // Out-of-range and negative rejected; previous state preserved.
        for invalid in [-1, 8, 100, i32::MIN, i32::MAX] {
            assert!(validate_channel_id(invalid).is_err());
        }
        let mut channel_value = 1;
        if let Ok(next) = validate_channel_id(8) {
            channel_value = next;
        }
        assert_eq!(channel_value, 1);
    }

    // Setter-transaction tests. `Gd<T>` cannot be constructed in pure unit
    // tests (constructing it requires a live Godot engine; `Gd::from_init_fn`
    // panics with "Godot binding accessed before initialization"). Each
    // `#[func] set_*` body applies the exact transaction under test — validate
    // first, assign only on success, log `godot_error!` and leave all state
    // untouched on reject — so we exercise that transaction directly against a
    // backing field, mirroring the established
    // `rejected_generator_assignment_preserves_previous_state` pattern.

    #[test]
    fn waves_period_setter_rejects_nonpositive_and_preserves_previous_state() {
        let mut period_value = 128.0;
        // Valid assignment accepted.
        if validate_positive_finite_float(64.0).is_ok() {
            period_value = 64.0;
        }
        assert_eq!(period_value, 64.0);
        // Invalid assignment must preserve the previously-set value.
        for invalid in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            if validate_positive_finite_float(invalid).is_ok() {
                period_value = invalid;
            }
        }
        assert_eq!(period_value, 64.0);
    }

    #[test]
    fn waves_amplitude_setter_accepts_any_finite_real_but_rejects_nonfinite() {
        let mut amplitude_value = 60.0;
        // Amplitude is a multiplier: any finite real is valid (incl. 0/negative).
        if validate_finite_float(0.0).is_ok() {
            amplitude_value = 0.0;
        }
        assert_eq!(amplitude_value, 0.0);
        if validate_finite_float(-12.5).is_ok() {
            amplitude_value = -12.5;
        }
        assert_eq!(amplitude_value, -12.5);
        // Non-finite values must be rejected and preserve previous state.
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            if validate_finite_float(invalid).is_ok() {
                amplitude_value = invalid;
            }
        }
        assert_eq!(amplitude_value, -12.5);
    }

    #[test]
    fn noise_frequency_setter_rejects_nonpositive_and_preserves_previous_state() {
        let mut frequency_value = 0.05;
        if validate_positive_finite_float(0.1).is_ok() {
            frequency_value = 0.1;
        }
        assert_eq!(frequency_value, 0.1);
        for invalid in [0.0, -0.05, f32::NAN, f32::NEG_INFINITY] {
            if validate_positive_finite_float(invalid).is_ok() {
                frequency_value = invalid;
            }
        }
        assert_eq!(frequency_value, 0.1);
    }

    #[test]
    fn noise_height_start_and_height_range_setters_accept_finite_only() {
        let mut height_start_value = -100.0;
        let mut height_range_value = 200.0;
        // Finite assignments (including 0 / negative) accepted.
        if validate_finite_float(-50.0).is_ok() {
            height_start_value = -50.0;
        }
        if validate_finite_float(0.0).is_ok() {
            height_range_value = 0.0;
        }
        assert_eq!(height_start_value, -50.0);
        assert_eq!(height_range_value, 0.0);
        // Non-finite rejected; previous state preserved.
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            if validate_finite_float(invalid).is_ok() {
                height_start_value = invalid;
            }
            if validate_finite_float(invalid).is_ok() {
                height_range_value = invalid;
            }
        }
        assert_eq!(height_start_value, -50.0);
        assert_eq!(height_range_value, 0.0);
    }
}

// ---------------------------------------------------------------------------
// VoxelGeneratorImage
// ---------------------------------------------------------------------------

/// Normalized height read from one image pixel. Upstream uses the RED channel
/// only (`get_pixel(...).r`), not luminance. Free function so the pinned
/// channel choice is pinnable in plain `cargo test`.
fn image_pixel_height(pixel: Color) -> f32 {
    pixel.r
}

/// Pure guard extracted from [`VoxelGeneratorImage::load_image_values`]: the
/// reason an image cannot be loaded, or `None` when it can. Compressed images
/// are rejected because pixel access would fail — upstream's `set_image`
/// guards with `ERR_FAIL_COND(im->is_compressed())`. Free function so the
/// guard is pinnable in plain `cargo test`.
fn image_load_error(width: i32, height: i32, compressed: bool) -> Option<&'static str> {
    if width <= 0 || height <= 0 {
        return Some("image must have positive dimensions");
    }
    if compressed {
        return Some("image must not be compressed (call decompress() first)");
    }
    None
}

/// Build the core image generator from the wrapper's parameters. Free
/// function so the wiring (wrap mode, blur, full heightmap parameter block)
/// is pinnable in plain `cargo test`.
fn image_generator_from_params(
    values: Vec<f32>,
    size: [i32; 2],
    repeat: bool,
    blur_enabled: bool,
    params: HeightmapParams,
) -> Image {
    let mut gen = Image::default();
    gen.blur_enabled = blur_enabled;
    gen.set_image(values, size[0], size[1]);
    gen.wrap = if repeat {
        ImageWrapMode::Repeat
    } else {
        ImageWrapMode::Clamp
    };
    gen.heightmap = params;
    gen
}

/// A heightmap terrain generator driven by an image. The RED channel of each
/// pixel becomes the terrain height:
/// `height = height_start + red * height_range`. Wraps
/// [`voxel_core::generators::simple::Image`].
///
/// With `channel = 1` (SDF) this produces smooth transvoxel terrain; with
/// `channel = 0` (Type) it fills blocky voxels, which can drive the cubes
/// mesher (Minecraft-style terrain from an image). Upstream repeats the image
/// beyond its extent, so `repeat` defaults to `true`.
///
/// Upstream inherits `VoxelGeneratorHeightmap`; since gdext cannot inherit
/// user classes (see the module-level note), the inherited pinned members
/// (`height_range` with the pinned override default 200, `iso_scale`,
/// `offset`) are re-declared here.
#[derive(GodotClass)]
#[class(base = Resource, tool)]
pub struct VoxelGeneratorImage {
    base: Base<Resource>,
    /// Vertical extent of the terrain; pixel values `0..1` scale by this.
    /// Pinned subclass override default: 200 (base class default is 30).
    #[var]
    pub height_range: f32,
    /// World Y that a black pixel (0) maps to.
    #[var]
    pub height_start: f32,
    /// Output channel: `0` = Type (blocky), `1` = Sdf (smooth).
    #[var]
    pub channel: i32,
    /// Tile the image beyond its edges (upstream repeats; default true).
    #[var]
    pub repeat: bool,
    /// Canonical `offset` (inherited from `VoxelGeneratorHeightmap`).
    #[var]
    pub offset: Vector2i,
    /// Canonical `iso_scale` (backing field; strictly positive finite).
    iso_scale_value: f32,
    /// Canonical `blur_enabled` (backing field).
    blur_enabled_value: bool,
    /// Last image assigned through the `image` property.
    image_resource: Option<Gd<GodotImage>>,
    /// Normalized heights (red channel `0..1`), row-major `x + z * width`.
    values: Vec<f32>,
    /// Image size: `[width, height]`.
    size: [i32; 2],
    #[var(get = get_iso_scale, set = set_iso_scale)]
    iso_scale: PhantomVar<f32>,
    #[var(get = is_blur_enabled, set = set_blur_enabled)]
    blur_enabled: PhantomVar<bool>,
    #[var(get = get_image, set = set_image)]
    image: PhantomVar<Option<Gd<GodotImage>>>,
}

#[godot_api]
impl IResource for VoxelGeneratorImage {
    fn init(base: Base<Resource>) -> Self {
        Self {
            base,
            height_range: 200.0,
            height_start: -50.0,
            channel: ChannelId::Sdf.index() as i32,
            repeat: true,
            offset: Vector2i::new(0, 0),
            iso_scale_value: 1.0,
            blur_enabled_value: false,
            image_resource: None,
            values: Vec::new(),
            size: [0, 0],
            iso_scale: PhantomVar::default(),
            blur_enabled: PhantomVar::default(),
            image: PhantomVar::default(),
        }
    }
}

#[godot_api]
impl VoxelGeneratorImage {
    /// Load heights from a Godot `Image`: the RED channel of each pixel
    /// becomes the normalized height at that `(x, z)`. Replaces any previously
    /// loaded image. Pinned `image` property setter (upstream
    /// `VoxelGeneratorImage.xml`).
    #[func]
    fn set_image(&mut self, image: Option<Gd<GodotImage>>) {
        match image {
            Some(image) => {
                if self.load_image_values(&image) {
                    self.image_resource = Some(image);
                }
            }
            None => {
                // Assigning null clears the stored image.
                self.image_resource = None;
                self.values.clear();
                self.size = [0, 0];
            }
        }
    }

    /// The image currently used as a heightmap (pinned `image` property
    /// getter; `null` when none was assigned or the assignment failed).
    #[func]
    fn get_image(&self) -> Option<Gd<GodotImage>> {
        self.image_resource.clone()
    }

    /// Bool-reporting variant of the pinned `set_image` property setter (kept
    /// as an extra for scripts that want to detect failed loads: `false` on
    /// an empty or invalid image).
    #[func]
    fn set_image_checked(&mut self, image: Gd<GodotImage>) -> bool {
        let loaded = self.load_image_values(&image);
        if loaded {
            self.image_resource = Some(image);
        }
        loaded
    }

    /// Extract the per-pixel red-channel heights. Returns `false` without
    /// touching state when the image is empty or compressed (upstream rejects
    /// compressed images in `set_image` with `ERR_FAIL_COND`).
    fn load_image_values(&mut self, image: &Gd<GodotImage>) -> bool {
        let width = image.get_width();
        let height = image.get_height();
        if let Some(reason) = image_load_error(width, height, image.is_compressed()) {
            godot_error!("VoxelGeneratorImage.set_image: {reason}");
            return false;
        }
        let mut values = Vec::with_capacity((width * height) as usize);
        for z in 0..height {
            for x in 0..width {
                values.push(image_pixel_height(image.get_pixel(x, z)));
            }
        }
        self.values = values;
        self.size = [width, height];
        true
    }

    /// Load heights from raw bytes (`0..255` → `0..1`), row-major.
    /// Returns `false` if `data.len() != width * height`.
    #[func]
    fn set_heights(&mut self, data: PackedByteArray, width: i32, height: i32) -> bool {
        if width <= 0 || height <= 0 || data.len() != (width * height) as usize {
            return false;
        }
        self.values = data.as_slice().iter().map(|&b| b as f32 / 255.0).collect();
        self.size = [width, height];
        true
    }

    /// Whether an image/heightmap is loaded.
    #[func]
    fn has_image(&self) -> bool {
        !self.values.is_empty()
    }

    /// Canonical `iso_scale` (inherited from `VoxelGeneratorHeightmap`;
    /// strictly positive finite).
    #[func]
    fn get_iso_scale(&self) -> f32 {
        self.iso_scale_value
    }

    #[func]
    fn set_iso_scale(&mut self, value: f32) {
        if validate_positive_finite_float(value).is_err() {
            godot_error!("VoxelGeneratorImage.set_iso_scale: value must be finite and positive");
            return;
        }
        self.iso_scale_value = value;
    }

    /// Canonical `blur_enabled` (pinned `VoxelGeneratorImage.xml`). When
    /// enabled, loaded heights are smoothed with upstream's 5-tap plus/cross
    /// kernel (center + 4 orthogonal neighbours, weight 0.2 each, wrap-around
    /// across image borders) before generating terrain.
    #[func]
    fn is_blur_enabled(&self) -> bool {
        self.blur_enabled_value
    }

    #[func]
    fn set_blur_enabled(&mut self, value: bool) {
        self.blur_enabled_value = value;
    }

    /// Construct the engine-agnostic generator from the current parameters.
    pub fn create_core_generator(&self) -> SharedVoxelGenerator {
        let params = heightmap_params_from_godot(
            self.channel,
            self.height_start,
            self.height_range,
            self.iso_scale_value,
            self.offset,
        );
        Arc::new(image_generator_from_params(
            self.values.clone(),
            self.size,
            self.repeat,
            self.blur_enabled_value,
            params,
        ))
    }

    /// Generates a block of voxels within the specified world area (pinned
    /// `generate_block`, upstream `VoxelGenerator.xml`).
    #[func]
    fn generate_block(
        &self,
        mut out_buffer: Gd<crate::voxel_buffer::VoxelBufferGD>,
        origin_in_voxels: Vector3,
        lod: i32,
    ) {
        let generator = self.create_core_generator();
        generate_core_block_into_gd(generator.as_ref(), &mut out_buffer, origin_in_voxels, lod);
    }
}

#[cfg(test)]
mod generator_contract_tests {
    use super::*;
    use voxel_core::storage::{ChannelDepth, VoxelBuffer};

    /// Fresh 32-bit-depth SDF buffer (Bit16 snorm would quantize to [-1,1]).
    fn sdf_buffer(size: voxel_core::math::Vector3i) -> VoxelBuffer {
        let mut buffer = VoxelBuffer::with_size(size);
        buffer.set_channel_depth(ChannelId::Sdf.index(), ChannelDepth::Bit32);
        buffer
    }

    fn origin(x: i32, y: i32, z: i32) -> voxel_core::math::Vector3i {
        voxel_core::math::Vector3i::new(x, y, z)
    }

    #[test]
    fn noise_type_decoding_is_per_resource_class() {
        // ZN_FastNoiseLite and FastNoise2 use DIFFERENT NoiseType encodings:
        // the same integer must decode to different fastnoise-lite types
        // depending on the assigned resource's class.
        for (value, lite, fastnoise2) in [
            (2, NoiseType::Cellular, NoiseType::Perlin),
            (3, NoiseType::Perlin, NoiseType::Value),
        ] {
            assert_eq!(
                noise_type_from_resource_enum("ZN_FastNoiseLite", value),
                Some(lite)
            );
            assert_eq!(
                noise_type_from_resource_enum("FastNoise2", value),
                Some(fastnoise2),
                "same int {value} must decode differently per class"
            );
        }
        // FastNoise2's sentinels mirror build_sampler's OpenSimplex2 fallback;
        // ZN_FastNoiseLite rejects out-of-range encodings.
        assert_eq!(
            noise_type_from_resource_enum("FastNoise2", 5),
            Some(NoiseType::OpenSimplex2)
        );
        assert_eq!(
            noise_type_from_resource_enum("ZN_FastNoiseLite", 5),
            Some(NoiseType::Value)
        );
        assert_eq!(noise_type_from_resource_enum("ZN_FastNoiseLite", 9), None);
        // Unknown classes fall back to the sampler default.
        assert_eq!(noise_type_from_resource_enum("FastNoiseLite", 0), None);
        assert_eq!(noise_type_from_resource_enum("Resource", 2), None);
    }

    #[test]
    fn seed_forwarding_saturates_instead_of_wrapping() {
        // A plain `as i32` cast would wrap 2^32 + 7 back to 7 (a different
        // seed); the saturated conversion clamps to i32 bounds.
        assert_eq!(saturating_seed_i32(7), 7);
        assert_eq!(saturating_seed_i32(-3), -3);
        assert_eq!(saturating_seed_i32(i64::from(i32::MAX)), i32::MAX);
        assert_eq!(saturating_seed_i32(i64::from(i32::MAX) + 1), i32::MAX);
        assert_eq!(saturating_seed_i32(i64::MAX), i32::MAX);
        assert_eq!(saturating_seed_i32(i64::from(i32::MIN) - 1), i32::MIN);
        assert_eq!(saturating_seed_i32(i64::MIN), i32::MIN);
        // The generators forward the pinned seed through the saturating path.
        let gen = heightmap_generator_from_params(
            i64::from(i32::MAX) + 1,
            0.02,
            heightmap_params_from_godot(1, -5.0, 30.0, 1.0, Vector2i::new(0, 0)),
        );
        assert_eq!(gen.noise_config.seed, Some(i32::MAX));
    }

    #[test]
    fn image_load_error_reports_empty_and_compressed_images() {
        // The pure guard behind `load_image_values`: empty images and
        // compressed images (upstream ERR_FAIL_COND(is_compressed)) are
        // rejected, well-formed uncompressed ones are accepted.
        assert_eq!(
            image_load_error(0, 16, false),
            Some("image must have positive dimensions")
        );
        assert_eq!(
            image_load_error(16, -1, false),
            Some("image must have positive dimensions")
        );
        assert_eq!(
            image_load_error(16, 16, true),
            Some("image must not be compressed (call decompress() first)")
        );
        assert_eq!(image_load_error(1, 1, false), None);
    }

    #[test]
    fn voxel_generator_generate_block_contract() {
        // The pinned VoxelGenerator.generate_block contract, exercised through
        // the core Flat generator built by the extracted free fn: solid below
        // the height, air above, and an SDF sign flip at the surface.
        let generator = flat_generator_from_params(4.0, 1, 1);
        let mut buffer = sdf_buffer(origin(2, 8, 2));
        generator.generate_block(VoxelQueryData {
            buffer: &mut buffer,
            origin_in_voxels: origin(0, 0, 0),
            lod: 0,
        });
        let sdf = |x: i32, y: i32, z: i32| buffer.get_voxel_f(x, y, z, ChannelId::Sdf.index());
        assert!(sdf(0, 0, 0) < 0.0, "below the plane must be solid");
        assert!(sdf(0, 7, 0) > 0.0, "above the plane must be air");
        let mut flipped = false;
        for y in 0..7 {
            if (sdf(0, y, 0) < 0.0) != (sdf(0, y + 1, 0) < 0.0) {
                flipped = true;
            }
        }
        assert!(flipped, "SDF must cross zero at the height");
    }

    #[test]
    fn flat_generator_from_params_fills_blocky_and_maps_channel() {
        // Blocky path: custom voxel type fills below the height, air above.
        let generator = flat_generator_from_params(3.0, 0, 7);
        assert_eq!(generator.channel, ChannelId::Type);
        assert_eq!(generator.voxel_type, 7);
        let mut buffer = VoxelBuffer::with_size(origin(2, 6, 2));
        generator.generate_block(VoxelQueryData {
            buffer: &mut buffer,
            origin_in_voxels: origin(0, 0, 0),
            lod: 0,
        });
        assert_eq!(buffer.get_voxel(0, 2, 0, ChannelId::Type.index()), 7);
        assert_eq!(buffer.get_voxel(0, 3, 0, ChannelId::Type.index()), 0);
        assert_eq!(buffer.get_voxel(1, 5, 1, ChannelId::Type.index()), 0);
        // Negative voxel types clamp to 0; out-of-range channels fall back to
        // SDF like every other generator binding.
        let clamped = flat_generator_from_params(3.0, 0, -5);
        assert_eq!(clamped.voxel_type, 0);
        let fallback = flat_generator_from_params(3.0, 99, 1);
        assert_eq!(fallback.channel, ChannelId::Sdf);
    }

    #[test]
    fn noise_generator_honors_noise_resource_params() {
        // The configuration path create_core_generator uses when a noise
        // resource is assigned: the pure mapping of the pinned ZN_FastNoiseLite
        // getters into NoiseConfigParts. (Reading the Gd resource itself needs
        // a live engine; this mapping is everything below that call.) Identical
        // parameters must give bit-identical blocks; a different seed must
        // change the output.
        let parts = NoiseConfigParts {
            seed: 7,
            frequency: 0.05,
            noise_type: noise_type_from_godot_enum(0),
            fractal_type: fractal_type_from_godot_enum(1),
            fractal_octaves: Some(3),
            fractal_lacunarity: Some(2.0),
            fractal_gain: Some(0.5),
            fractal_weighted_strength: Some(0.0),
            fractal_ping_pong_strength: Some(2.0),
        };
        // Pinned enum encodings map onto the sampler enums.
        assert_eq!(noise_type_from_godot_enum(2), Some(NoiseType::Cellular));
        assert_eq!(noise_type_from_godot_enum(5), Some(NoiseType::Value));
        assert_eq!(noise_type_from_godot_enum(9), None);
        assert_eq!(fractal_type_from_godot_enum(3), Some(FractalType::PingPong));
        assert_eq!(fractal_type_from_godot_enum(4), None);

        let generate = |parts: &NoiseConfigParts| -> VoxelBuffer {
            let generator = noise_generator_from_config(
                parts.clone().into(),
                -8.0,
                16.0,
                ChannelId::Sdf.index() as i32,
            );
            let mut buffer = sdf_buffer(origin(4, 8, 4));
            generator.generate_block(VoxelQueryData {
                buffer: &mut buffer,
                origin_in_voxels: origin(0, 0, 0),
                lod: 0,
            });
            buffer
        };

        let block_a = generate(&parts);
        let block_b = generate(&parts);
        for z in 0..4usize {
            for y in 0..8usize {
                for x in 0..4usize {
                    assert_eq!(
                        block_a
                            .get_voxel_f(x as i32, y as i32, z as i32, ChannelId::Sdf.index())
                            .to_bits(),
                        block_b
                            .get_voxel_f(x as i32, y as i32, z as i32, ChannelId::Sdf.index())
                            .to_bits(),
                        "identical noise parameters diverged at ({x},{y},{z})"
                    );
                }
            }
        }

        let mut other_seed = parts.clone();
        other_seed.seed = 8;
        let block_c = generate(&other_seed);
        let mut differs = false;
        for z in 0..4usize {
            for y in 0..8usize {
                for x in 0..4usize {
                    if block_a
                        .get_voxel_f(x as i32, y as i32, z as i32, ChannelId::Sdf.index())
                        .to_bits()
                        != block_c
                            .get_voxel_f(x as i32, y as i32, z as i32, ChannelId::Sdf.index())
                            .to_bits()
                    {
                        differs = true;
                    }
                }
            }
        }
        assert!(differs, "a different noise seed must change the block");
    }

    #[test]
    fn heightmap_generator_forwards_all_pinned_params() {
        // Regression: the wrapper used to forward only height_range. All five
        // pinned members must reach the core HeightmapParams.
        let params = heightmap_params_from_godot(1, -5.0, 30.0, 2.0, Vector2i::new(0, 0));
        let generator = heightmap_generator_from_params(42, 0.02, params);
        assert_eq!(generator.heightmap, params);
        assert_eq!(generator.heightmap.channel, ChannelId::Sdf);
        assert_eq!(generator.heightmap.height_start, -5.0);
        assert_eq!(generator.heightmap.height_range, 30.0);
        assert_eq!(generator.heightmap.iso_scale, 2.0);
        assert_eq!(generator.noise_config.seed, Some(42));
        assert_eq!(generator.noise_config.frequency, Some(0.02));

        // offset shifts the sampled field: generating at origin (0, 0) with
        // offset (32, 0) equals generating at origin (-32, 0) without offset
        // (the core helper subtracts the offset from the block origin).
        let run_at = |offset: Vector2i, origin_x: i32| -> VoxelBuffer {
            let generator = heightmap_generator_from_params(
                42,
                0.02,
                heightmap_params_from_godot(1, -5.0, 30.0, 1.0, offset),
            );
            let mut buffer = sdf_buffer(origin(4, 40, 4));
            generator.generate_block(VoxelQueryData {
                buffer: &mut buffer,
                origin_in_voxels: origin(origin_x, -6, 0),
                lod: 0,
            });
            buffer
        };
        let shifted = run_at(Vector2i::new(32, 0), 0);
        let direct = run_at(Vector2i::new(0, 0), -32);
        for z in 0..4usize {
            for y in 0..40usize {
                for x in 0..4usize {
                    assert_eq!(
                        shifted
                            .get_voxel_f(x as i32, y as i32, z as i32, ChannelId::Sdf.index())
                            .to_bits(),
                        direct
                            .get_voxel_f(x as i32, y as i32, z as i32, ChannelId::Sdf.index())
                            .to_bits(),
                        "offset did not shift the sampled field at ({x},{y},{z})"
                    );
                }
            }
        }
    }

    #[test]
    fn image_generator_contract() {
        // Upstream reads the RED channel (regression: this wrapper used to
        // read Rec.709 luminance, which would return 0.6437 for pure green).
        assert_eq!(image_pixel_height(Color::from_rgb(0.8, 0.1, 0.6)), 0.8);
        assert_eq!(image_pixel_height(Color::from_rgb(0.0, 0.9, 0.0)), 0.0);

        // values row-major (x + z * width): rows z=0 → [0.0, 0.25], z=1 → [0.5, 0.75].
        let values = vec![0.0, 0.25, 0.5, 0.75];
        let params = heightmap_params_from_godot(1, 0.0, 8.0, 1.5, Vector2i::new(2, 3));

        let clamped = image_generator_from_params(values.clone(), [2, 2], false, false, params);
        // iso_scale and offset are forwarded into the heightmap params.
        assert_eq!(clamped.heightmap, params);
        assert_eq!(clamped.wrap, ImageWrapMode::Clamp);
        assert_eq!(clamped.height_at(-10, -10), 0.0);
        assert_eq!(clamped.height_at(99, 99), 0.75);

        let repeating = image_generator_from_params(values, [2, 2], true, false, params);
        assert_eq!(repeating.wrap, ImageWrapMode::Repeat);
        assert_eq!(repeating.height_at(0, 2), repeating.height_at(0, 0));
        assert_eq!(repeating.height_at(1, -1), repeating.height_at(1, 1));

        // blur_enabled forwards into the core generator: a delta peak spreads
        // to the cross neighbours only (upstream 5-tap kernel), not diagonals.
        let mut peak = vec![0.0; 9];
        peak[1 + 3] = 1.0; // 3x3, centre (1, 1).
        let blurred = image_generator_from_params(peak.clone(), [3, 3], false, true, params);
        let sharp = image_generator_from_params(peak, [3, 3], false, false, params);
        assert!(blurred.height_at(1, 1) < sharp.height_at(1, 1));
        assert!(blurred.height_at(0, 1) > sharp.height_at(0, 1));
        assert!(blurred.height_at(1, 0) > sharp.height_at(1, 0));
        assert_eq!(blurred.height_at(0, 0), sharp.height_at(0, 0));
        assert_eq!(blurred.height_at(2, 2), sharp.height_at(2, 2));
    }

    #[test]
    fn waves_generator_forwards_canonical_params() {
        // Regression: the wrapper used to ignore pattern_size/pattern_offset
        // and drive the core from legacy period/amplitude instead, so setting
        // pattern_size from GDScript did nothing.
        let params = heightmap_params_from_godot(1, -10.0, 20.0, 1.0, Vector2i::new(0, 0));
        let waves =
            waves_generator_from_params(Vector2::new(40.0, 50.0), Vector2::new(3.0, 4.0), params);
        assert_eq!(
            waves.pattern_size,
            voxel_core::math::Vector2f::new(40.0, 50.0)
        );
        assert_eq!(
            waves.pattern_offset,
            voxel_core::math::Vector2f::new(3.0, 4.0)
        );
        assert_eq!(waves.heightmap, params);

        // pattern_offset shifts the phase: sampling with offset (3, 4) equals
        // sampling without it at (x + 3, z + 4), for the same pattern size.
        let base =
            waves_generator_from_params(Vector2::new(40.0, 50.0), Vector2::new(0.0, 0.0), params);
        for (x, z) in [(0, 0), (5, -7), (13, 21)] {
            assert!(
                (waves.height_at(x, z) - base.height_at(x + 3, z + 4)).abs() < 1e-5,
                "pattern_offset did not shift the phase at ({x},{z})"
            );
        }

        // The surface crossing lands inside
        // [height_start, height_start + height_range) (heights remap to
        // [-10, 10) here).
        let mut buffer = sdf_buffer(origin(1, 32, 1));
        waves.generate_block(VoxelQueryData {
            buffer: &mut buffer,
            origin_in_voxels: origin(0, -12, 0),
            lod: 0,
        });
        let mut crossing: Option<i32> = None;
        for y in 0..31 {
            let a = buffer.get_voxel_f(0, y, 0, ChannelId::Sdf.index());
            let b = buffer.get_voxel_f(0, y + 1, 0, ChannelId::Sdf.index());
            if (a < 0.0) != (b < 0.0) {
                crossing = Some(y);
                break;
            }
        }
        let crossing = crossing.expect("waves produced no SDF sign change");
        let world_y = -12 + crossing;
        assert!(
            (-11..10).contains(&world_y),
            "surface crossing at {world_y} outside the height range [-10, 10)"
        );
    }

    #[test]
    fn generate_block_origin_truncates_toward_zero() {
        // The pinned generate_block binds origin_in_voxels as Vector3
        // (upstream VoxelGenerator.xml); the core consumes integers. The
        // conversion truncates toward zero, matching Godot's Vector3 →
        // Vector3i cast (10.6 → 10, -2.5 → -2).
        assert_eq!(
            core_origin_from_vector3(Vector3::new(10.6, -2.5, 3.49)),
            origin(10, -2, 3)
        );
        assert_eq!(
            core_origin_from_vector3(Vector3::new(-0.9, 0.9, -1.0)),
            origin(0, 0, -1)
        );
    }
}
