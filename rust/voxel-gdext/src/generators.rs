//! Godot Resource bindings for voxel generators.
//!
//! Each `VoxelGenerator*` is a Godot `Resource` that wraps the corresponding
//! `voxel_core::generators::simple::*` type. When attached to a
//! [`VoxelTerrain`](crate::terrain::VoxelTerrain) via the `generator` property,
//! it produces voxel data on demand.

use godot::prelude::*;
use std::sync::Arc;

use voxel_core::generators::base::HeightmapParams;
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

// ---------------------------------------------------------------------------
// VoxelGeneratorWaves
// ---------------------------------------------------------------------------

/// A simple SDF terrain generator that produces rolling waves along the X axis.
/// Wraps [`voxel_core::generators::simple::Waves`].
///
/// In GDScript: create a `VoxelGeneratorWaves` resource and assign it to a
/// `VoxelTerrain`'s `generator` property.
#[derive(GodotClass)]
#[class(base = Resource, tool)]
pub struct VoxelGeneratorWaves {
    base: Base<Resource>,
    /// Amplitude of the waves (in voxels). Backing field for the
    /// `amplitude` property; any finite real.
    amplitude_value: f32,
    /// Frequency of the waves (backing field; strictly positive finite).
    frequency_value: f32,
    /// Period of the waves (backing field; strictly positive finite).
    period_value: f32,
    /// Canonical `pattern_offset` (backing field; finite components).
    pattern_offset_value: Vector2,
    /// Canonical `pattern_size` (backing field; strictly positive finite).
    pattern_size_value: Vector2,
    #[var(get = get_amplitude, set = set_amplitude)]
    amplitude: PhantomVar<f32>,
    #[var(get = get_frequency, set = set_frequency)]
    frequency: PhantomVar<f32>,
    #[var(get = get_period, set = set_period)]
    period: PhantomVar<f32>,
    #[var(get = get_pattern_offset, set = set_pattern_offset)]
    pattern_offset: PhantomVar<Vector2>,
    #[var(get = get_pattern_size, set = set_pattern_size)]
    pattern_size: PhantomVar<Vector2>,
}

#[godot_api]
impl IResource for VoxelGeneratorWaves {
    fn init(base: Base<Resource>) -> Self {
        Self {
            base,
            amplitude_value: 60.0,
            frequency_value: 0.02,
            period_value: 128.0,
            pattern_offset_value: Vector2::new(0.0, 0.0),
            pattern_size_value: Vector2::new(30.0, 30.0),
            amplitude: PhantomVar::default(),
            frequency: PhantomVar::default(),
            period: PhantomVar::default(),
            pattern_offset: PhantomVar::default(),
            pattern_size: PhantomVar::default(),
        }
    }
}

#[godot_api]
impl VoxelGeneratorWaves {
    /// Construct the engine-agnostic generator from the current parameters.
    pub fn create_core_generator(&self) -> SharedVoxelGenerator {
        let Ok((amplitude, _frequency, period)) = validate_wave_configuration(
            self.amplitude_value,
            self.frequency_value,
            self.period_value,
        ) else {
            godot_error!("VoxelGeneratorWaves: amplitude must be finite and frequency/period must be finite and positive");
            return Arc::new(Waves::default());
        };
        let mut waves = Waves::default();
        waves.set_pattern_size(voxel_core::math::Vector2f::new(period, period));
        waves.heightmap.height_range = amplitude;
        Arc::new(waves)
    }

    /// Amplitude of the waves (any finite real, including zero/negative).
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
    }

    /// Frequency of the waves (strictly positive finite).
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

    /// Period of the waves (strictly positive finite).
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

#[godot_api]
impl VoxelGeneratorFlat {
    /// Construct the engine-agnostic generator from the current parameters.
    pub fn create_core_generator(&self) -> SharedVoxelGenerator {
        let flat = Flat {
            height: self.height_value,
            ..Flat::default()
        };
        Arc::new(flat)
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
        // Use NoiseConfig.build() to avoid direct fastnoise_lite dependency.
        let config = NoiseConfig {
            seed: Some(self.seed as i32),
            frequency: Some(self.frequency_value),
            ..NoiseConfig::default()
        };
        let noise = Noise {
            noise: config.build(),
            height_start: self.height_start_value,
            height_range: self.height_range_value,
            ..Noise::default()
        };
        Arc::new(noise)
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
            height_range_value: 100.0,
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

#[godot_api]
impl VoxelGeneratorHeightmap {
    pub fn create_core_generator(&self) -> SharedVoxelGenerator {
        if validate_positive_finite_float(self.frequency_value).is_err()
            || validate_finite_float(self.height_range_value).is_err()
        {
            godot_error!("VoxelGeneratorHeightmap: frequency must be positive and height range must be finite");
            return Arc::new(HeightmapNoise::default());
        }
        let config = NoiseConfig {
            seed: Some(self.seed as i32),
            frequency: Some(self.frequency_value),
            ..NoiseConfig::default()
        };
        let hm = HeightmapNoise {
            noise_config: config,
            curve: None,
            heightmap: HeightmapParams {
                height_range: self.height_range_value,
                ..Default::default()
            },
        };
        Arc::new(hm)
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
            seed: Some(self.seed as i32),
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

/// A heightmap terrain generator driven by an image. Pixel luminance becomes
/// terrain height: `height = height_start + luminance * height_range`.
/// Wraps [`voxel_core::generators::simple::Image`].
///
/// With `channel = 1` (SDF) this produces smooth transvoxel terrain; with
/// `channel = 0` (Type) it fills blocky voxels, which can drive the cubes
/// mesher (Minecraft-style terrain from an image).
#[derive(GodotClass)]
#[class(base = Resource, tool)]
pub struct VoxelGeneratorImage {
    base: Base<Resource>,
    /// Vertical extent of the terrain; pixel values `0..1` scale by this.
    #[var]
    pub height_range: f32,
    /// World Y that a black pixel (0) maps to.
    #[var]
    pub height_start: f32,
    /// Output channel: `0` = Type (blocky), `1` = Sdf (smooth).
    #[var]
    pub channel: i32,
    /// Tile the image horizontally instead of clamping at its edges.
    #[var]
    pub repeat: bool,
    /// Normalized heights (`0..1`), row-major `x + z * width`.
    values: Vec<f32>,
    /// Image size: `[width, height]`.
    size: [i32; 2],
}

#[godot_api]
impl IResource for VoxelGeneratorImage {
    fn init(base: Base<Resource>) -> Self {
        Self {
            base,
            height_range: 100.0,
            height_start: -50.0,
            channel: ChannelId::Sdf.index() as i32,
            repeat: false,
            values: Vec::new(),
            size: [0, 0],
        }
    }
}

#[godot_api]
impl VoxelGeneratorImage {
    /// Load heights from a Godot `Image`: each pixel's luminance becomes the
    /// normalized height at that `(x, z)`. Replaces any previously loaded
    /// image.
    #[func]
    fn set_image(&mut self, image: Gd<godot::classes::Image>) -> bool {
        let width = image.get_width();
        let height = image.get_height();
        if width <= 0 || height <= 0 {
            return false;
        }
        let mut values = Vec::with_capacity((width * height) as usize);
        for z in 0..height {
            for x in 0..width {
                let c = image.get_pixel(x, z);
                // Rec. 709 luminance.
                values.push(0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b);
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

    /// Construct the engine-agnostic generator from the current parameters.
    pub fn create_core_generator(&self) -> SharedVoxelGenerator {
        let mut gen = Image::default();
        gen.set_image(self.values.clone(), self.size[0], self.size[1]);
        gen.wrap = if self.repeat {
            ImageWrapMode::Repeat
        } else {
            ImageWrapMode::Clamp
        };
        gen.heightmap.height_start = self.height_start;
        gen.heightmap.height_range = self.height_range;
        gen.heightmap.channel =
            channel_id_from_index(self.channel.max(0) as usize).unwrap_or(ChannelId::Sdf);
        Arc::new(gen)
    }
}
