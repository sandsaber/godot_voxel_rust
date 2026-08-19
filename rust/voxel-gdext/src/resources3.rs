//! Noise resources, blocky model variants, graph nodes, and editor helpers.
//!
//! Presence in this module does not imply canonical upstream API completeness;
//! see `../api/port_status.json` for the audited compatibility status.

use godot::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};

/// Maximum number of baked curve samples accepted from one script call.
/// Matches the general GDExt script-item allocation budget.
const MAX_CURVE_POINTS: usize = 65_536;
/// Maximum number of cells visited synchronously by `count_spots`.
const MAX_SPOT_GRID_CELLS: u64 = 65_536;
/// Maximum number of pixels written synchronously by `generate_image`.
/// Shared with `VoxelGeneratorGraph.generate_image_from_sdf`
/// (`voxel_buffer.rs`) so both image-writing entry points enforce the same
/// script workload budget.
pub(crate) const MAX_GENERATED_IMAGE_PIXELS: i64 = 65_536;

/// Whether the `generate_image` tiling warning has been emitted already.
static TILING_WARNING_EMITTED: AtomicBool = AtomicBool::new(false);

/// Emit the "tileable is unsupported" warning once per process.
fn warn_tiling_unsupported_once() {
    if !TILING_WARNING_EMITTED.swap(true, Ordering::Relaxed) {
        godot_warn!(
            "FastNoise2.generate_image: tileable noise is not supported by the pure-Rust backend; the generated image will not tile"
        );
    }
}

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

fn validate_unit_float(value: f32) -> Result<f32, &'static str> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err("value must be finite and between zero and one");
    }
    Ok(value)
}

fn validate_curve_point_count(count: i32) -> Result<usize, &'static str> {
    let count = usize::try_from(count.max(2)).map_err(|_| "curve point count is invalid")?;
    if count > MAX_CURVE_POINTS {
        return Err("curve point count exceeds the script allocation limit");
    }
    Ok(count)
}

fn prepare_identity_curve(
    count: i32,
) -> Result<(i32, voxel_core::generators::simple::Curve), &'static str> {
    let count = validate_curve_point_count(count)?;
    let point_count = i32::try_from(count).map_err(|_| "curve point count exceeds i32")?;
    Ok((
        point_count,
        voxel_core::generators::simple::Curve::identity(count),
    ))
}

fn validate_curve_points(values: &[f32]) -> Result<usize, &'static str> {
    let count = i32::try_from(values.len()).map_err(|_| "curve point count exceeds i32")?;
    let count = validate_curve_point_count(count)?;
    if values.len() < 2 {
        return Err("curve needs at least two points");
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err("curve points must be finite");
    }
    Ok(count)
}

fn checked_square_work(side: u64) -> Result<u64, &'static str> {
    side.checked_mul(side).ok_or("grid cell count overflowed")
}

fn validate_spot_grid_work(grid_size: i32) -> Result<u64, &'static str> {
    let side = u64::try_from(grid_size).map_err(|_| "grid size must be non-negative")?;
    let work = checked_square_work(side)?;
    if work > MAX_SPOT_GRID_CELLS {
        return Err("grid cell count exceeds the script workload limit");
    }
    Ok(work)
}

fn validate_spot_coordinate_work(grid_size: i32, radius: f32) -> Result<(), &'static str> {
    validate_spot_grid_work(grid_size)?;
    let radius = validate_positive_finite_float(radius)?;
    let last = grid_size.saturating_sub(1) as f32;
    if !(last * radius).is_finite() {
        return Err("scaled spot coordinate is not finite");
    }
    Ok(())
}

// === Pure (engine-free) noise helpers ===
//
// GD-backed structs cannot be constructed in unit tests (they need a live
// Godot engine), so every behavior below is extracted into plain functions
// over plain data. The `#[func]` methods delegate to these so script-visible
// behavior and tested behavior cannot diverge.

/// Engine-free snapshot of the pinned `ZN_FastNoiseLite` configuration.
///
/// Mirrors the class's GDScript-facing properties (enum values use the pinned
/// integer constants) and knows how to apply them to a fresh
/// `fastnoise_lite::FastNoiseLite` sampler.
#[derive(Debug, Clone, Copy, PartialEq)]
struct FastNoiseLiteConfig {
    /// Deterministic seed.
    seed: i32,
    /// Frequency (strictly positive finite; `1 / period`).
    frequency: f32,
    /// Noise type (0..=5, see the pinned `NoiseType` constants).
    noise_type: i32,
    /// Fractal type (0..=3, see the pinned `FractalType` constants).
    fractal_type: i32,
    /// Fractal octaves (>=1).
    fractal_octaves: i32,
    /// Fractal lacunarity (finite).
    fractal_lacunarity: f32,
    /// Fractal gain (finite).
    fractal_gain: f32,
    /// Fractal weighted strength ([0,1]).
    fractal_weighted_strength: f32,
    /// Fractal ping-pong strength (finite).
    fractal_ping_pong_strength: f32,
    /// Cellular distance function (0..=3).
    cellular_distance_function: i32,
    /// Cellular jitter (finite).
    cellular_jitter: f32,
    /// Cellular return type (0..=6).
    cellular_return_type: i32,
    /// 3D rotation type (0..=2).
    rotation_type_3d: i32,
}

impl FastNoiseLiteConfig {
    /// Pinned upstream defaults (ZN_FastNoiseLite.xml): seed=0,
    /// noise_type=OpenSimplex2(0), fractal_type=FBm(1), fractal_octaves=3,
    /// fractal_lacunarity=2.0, fractal_gain=0.5,
    /// fractal_weighted_strength=0.0, fractal_ping_pong_strength=2.0,
    /// cellular_distance_function=EuclideanSq(1), cellular_jitter=1.0,
    /// cellular_return_type=Distance(1), rotation_type_3d=None(0),
    /// period=64.0 (frequency 1/64).
    fn pinned_defaults() -> Self {
        let period = 64.0_f32;
        Self {
            seed: 0,
            frequency: 1.0 / period,
            noise_type: 0,
            fractal_type: 1,
            fractal_octaves: 3,
            fractal_lacunarity: 2.0,
            fractal_gain: 0.5,
            fractal_weighted_strength: 0.0,
            fractal_ping_pong_strength: 2.0,
            cellular_distance_function: 1,
            cellular_jitter: 1.0,
            cellular_return_type: 1,
            rotation_type_3d: 0,
        }
    }

    /// Build a fresh fastnoise-lite sampler configured from this snapshot.
    fn build_sampler(&self) -> voxel_core::fastnoise_lite::FastNoiseLite {
        use voxel_core::fastnoise_lite::{
            CellularDistanceFunction, CellularReturnType, FastNoiseLite, FractalType, NoiseType,
            RotationType3D,
        };
        let mut noise = FastNoiseLite::new();
        noise.set_seed(Some(self.seed));
        noise.set_frequency(Some(self.frequency));
        noise.set_noise_type(Some(match self.noise_type {
            1 => NoiseType::OpenSimplex2S,
            2 => NoiseType::Cellular,
            3 => NoiseType::Perlin,
            4 => NoiseType::ValueCubic,
            5 => NoiseType::Value,
            _ => NoiseType::OpenSimplex2,
        }));
        noise.set_rotation_type_3d(Some(match self.rotation_type_3d {
            1 => RotationType3D::ImproveXYPlanes,
            2 => RotationType3D::ImproveXZPlanes,
            _ => RotationType3D::None,
        }));
        noise.set_fractal_type(Some(match self.fractal_type {
            0 => FractalType::None,
            2 => FractalType::Ridged,
            3 => FractalType::PingPong,
            _ => FractalType::FBm,
        }));
        noise.set_fractal_octaves(Some(self.fractal_octaves.max(1)));
        noise.set_fractal_lacunarity(Some(self.fractal_lacunarity));
        noise.set_fractal_gain(Some(self.fractal_gain));
        noise.set_fractal_weighted_strength(Some(self.fractal_weighted_strength));
        noise.set_fractal_ping_pong_strength(Some(self.fractal_ping_pong_strength));
        noise.set_cellular_distance_function(Some(match self.cellular_distance_function {
            0 => CellularDistanceFunction::Euclidean,
            2 => CellularDistanceFunction::Manhattan,
            3 => CellularDistanceFunction::Hybrid,
            _ => CellularDistanceFunction::EuclideanSq,
        }));
        noise.set_cellular_return_type(Some(match self.cellular_return_type {
            0 => CellularReturnType::CellValue,
            2 => CellularReturnType::Distance2,
            3 => CellularReturnType::Distance2Add,
            4 => CellularReturnType::Distance2Sub,
            5 => CellularReturnType::Distance2Mul,
            6 => CellularReturnType::Distance2Div,
            _ => CellularReturnType::Distance,
        }));
        noise.set_cellular_jitter(Some(self.cellular_jitter));
        noise
    }

    /// True 2D noise sample at `(x, y)` via the crate's `get_noise_2d`
    /// (not a 3D `(x, 0, y)` slice).
    fn sample_2d(&self, x: f32, y: f32) -> f32 {
        self.build_sampler().get_noise_2d(x, y)
    }

    /// 3D noise sample at `(x, y, z)`.
    fn sample_3d(&self, x: f32, y: f32, z: f32) -> f32 {
        self.build_sampler().get_noise_3d(x, y, z)
    }

    /// Production path of the pinned `get_noise_3dv`: a `Vector3` position
    /// is sampled component-wise through `sample_3d` (same code the
    /// `#[func]` delegates to).
    fn sample_3dv(&self, position: (f32, f32, f32)) -> f32 {
        self.sample_3d(position.0, position.1, position.2)
    }

    /// Production path of the pinned `get_noise_2dv` (see `sample_3dv`).
    fn sample_2dv(&self, position: (f32, f32)) -> f32 {
        self.sample_2d(position.0, position.1)
    }
}

/// FastNoise2 `Remap` transform: linear map of `[min_in, max_in]` onto
/// `[min_out, max_out]`, without clamping (upstream's `Remap::GenT` computes
/// `to_min + (source - from_min) / (from_max - from_min) * (to_max - to_min)`).
/// A degenerate input range (`min_in == max_in`) is a passthrough instead of
/// the upstream division by zero, so the transform can never produce NaN.
fn apply_remap(v: f32, min_in: f32, max_in: f32, min_out: f32, max_out: f32, enabled: bool) -> f32 {
    if !enabled || min_in == max_in {
        return v;
    }
    (v - min_in) / (max_in - min_in) * (max_out - min_out) + min_out
}

/// Normalize a sampled batch value to a `[0, 1]` grayscale level using the
/// batch's *observed* minimum and maximum, rather than an assumed `[-1, 1]`
/// range (the terrace/remap output transforms may move the range). A
/// degenerate batch (`min == max`, or non-finite bounds) maps to mid-gray
/// 0.5. Free function so the normalization is pinnable in plain `cargo test`.
fn normalize_gray(value: f32, min: f32, max: f32) -> f32 {
    if min.partial_cmp(&max) != Some(std::cmp::Ordering::Less) {
        // Covers min == max and non-finite bounds: mid-gray, never NaN.
        return 0.5;
    }
    ((value - min) / (max - min)).clamp(0.0, 1.0)
}

/// FastNoise2 `Terrace` transform.
///
/// Follows upstream's `Terrace::GenT` construction (Modifiers.inl): the input
/// is scaled by the step count (`multiplier`), rounded to the nearest step,
/// and a smooth transition band is restored around each step boundary. The
/// `smoothness` parameter is normalized to `[0, 1]` — `0.0` snaps to hard
/// steps, `1.0` is exactly the identity — corresponding to upstream's raw
/// smoothness `s / (1 - s)` (upstream reaches identity only in the limit).
fn apply_terrace(v: f32, multiplier: f32, smoothness: f32, enabled: bool) -> f32 {
    if !enabled {
        return v;
    }
    if !multiplier.is_finite() || multiplier == 0.0 {
        // Degenerate step count: passthrough (avoids division by zero).
        return v;
    }
    let smoothness = if smoothness.is_finite() {
        smoothness.clamp(0.0, 1.0)
    } else {
        1.0
    };
    let scaled = v * multiplier;
    let rounded = scaled.round();
    if smoothness == 0.0 {
        return rounded / multiplier;
    }
    // Distance from the step boundary, in step-width units (`band` in
    // `[0, 0.5]`); widen it by `1 / smoothness`, saturating at half a step.
    let diff = rounded - scaled;
    let band = (0.5 - diff.abs()) / smoothness;
    let band = if band > 0.5 { 0.5 } else { band };
    let adjusted = if diff < 0.0 { 0.5 - band } else { band - 0.5 };
    (rounded + adjusted) / multiplier
}

// === Noise Resources (5) ===

/// FastNoiseLite noise resource. Wraps
/// [`voxel_core::generators::simple::Noise`] (which wraps
/// `fastnoise_lite::FastNoiseLite`) — `sample_3d` configures the sampler from
/// the resource's seed/frequency/noise_type and returns the raw 3D noise value
/// at a world point, exercising the full noise pipeline through the binding.
///
/// The pinned GDScript-facing properties mirror upstream `ZN_FastNoiseLite`.
/// Properties without a fastnoise-lite `set_*` counterpart on the live sampler
/// are still stored faithfully so GDScript reads round-trip
/// (`set X; get X == X`); they are applied to a freshly-built sampler in
/// `sample_*` so the noise actually reflects them.
#[derive(GodotClass)]
#[class(base = Resource, tool, rename = ZN_FastNoiseLite)]
pub struct FastNoiseLiteGD {
    base: Base<Resource>,
    /// Deterministic seed (backing field for the pinned `seed` property).
    seed_value: i32,
    /// Frequency (backing field; strictly positive finite). Exposed via the
    /// canonical `frequency` property.
    frequency_value: f32,
    /// Noise type: 0 = OpenSimplex2, 1 = OpenSimplex2S, 2 = Cellular,
    /// 3 = Perlin, 4 = ValueCubic, 5 = Value. Mirrors `NoiseType`.
    noise_type_value: i32,
    /// Fractal type (0=None,1=FBm,2=Ridged,3=PingPong).
    fractal_type_value: i32,
    /// Fractal octaves (>=1).
    fractal_octaves_value: i32,
    /// Fractal lacunarity (finite).
    fractal_lacunarity_value: f32,
    /// Fractal gain (finite).
    fractal_gain_value: f32,
    /// Fractal weighted strength ([0,1] unit float).
    fractal_weighted_strength_value: f32,
    /// Fractal ping-pong strength (finite).
    fractal_ping_pong_strength_value: f32,
    /// Cellular distance function (0=Euclidean,1=EuclideanSq,2=Manhattan,3=Hybrid).
    cellular_distance_function_value: i32,
    /// Cellular jitter (finite).
    cellular_jitter_value: f32,
    /// Cellular return type (0..=6).
    cellular_return_type_value: i32,
    /// 3D rotation type (0=None,1=ImproveXYPlanes,2=ImproveXZPlanes).
    rotation_type_3d_value: i32,
    /// Period (backing field; strictly positive finite). Upstream exposes
    /// `period` instead of `frequency`; `frequency == 1/period`.
    period_value: f32,
    /// The pinned GDScript-facing `seed` property.
    #[var(get = get_seed, set = set_seed)]
    seed: PhantomVar<i32>,
    /// The pinned GDScript-facing `frequency` property.
    #[var(get = get_frequency, set = set_frequency)]
    frequency: PhantomVar<f32>,
    /// The pinned GDScript-facing `noise_type` property.
    #[var(get = get_noise_type, set = set_noise_type)]
    noise_type: PhantomVar<i32>,
    /// The pinned GDScript-facing `fractal_type` property.
    #[var(get = get_fractal_type, set = set_fractal_type)]
    fractal_type: PhantomVar<i32>,
    /// The pinned GDScript-facing `fractal_octaves` property.
    #[var(get = get_fractal_octaves, set = set_fractal_octaves)]
    fractal_octaves: PhantomVar<i32>,
    /// The pinned GDScript-facing `fractal_lacunarity` property.
    #[var(get = get_fractal_lacunarity, set = set_fractal_lacunarity)]
    fractal_lacunarity: PhantomVar<f32>,
    /// The pinned GDScript-facing `fractal_gain` property.
    #[var(get = get_fractal_gain, set = set_fractal_gain)]
    fractal_gain: PhantomVar<f32>,
    /// The pinned GDScript-facing `fractal_weighted_strength` property.
    #[var(get = get_fractal_weighted_strength, set = set_fractal_weighted_strength)]
    fractal_weighted_strength: PhantomVar<f32>,
    /// The pinned GDScript-facing `fractal_ping_pong_strength` property.
    #[var(get = get_fractal_ping_pong_strength, set = set_fractal_ping_pong_strength)]
    fractal_ping_pong_strength: PhantomVar<f32>,
    /// The pinned GDScript-facing `cellular_distance_function` property.
    #[var(get = get_cellular_distance_function, set = set_cellular_distance_function)]
    cellular_distance_function: PhantomVar<i32>,
    /// The pinned GDScript-facing `cellular_jitter` property.
    #[var(get = get_cellular_jitter, set = set_cellular_jitter)]
    cellular_jitter: PhantomVar<f32>,
    /// The pinned GDScript-facing `cellular_return_type` property.
    #[var(get = get_cellular_return_type, set = set_cellular_return_type)]
    cellular_return_type: PhantomVar<i32>,
    /// The pinned GDScript-facing `rotation_type_3d` property.
    #[var(get = get_rotation_type_3d, set = set_rotation_type_3d)]
    rotation_type_3d: PhantomVar<i32>,
    /// The pinned GDScript-facing `period` property (upstream's name for
    /// `1/frequency`). Updating `period` updates `frequency` and vice versa.
    #[var(get = get_period, set = set_period)]
    period: PhantomVar<f32>,
}
#[godot_api]
impl IResource for FastNoiseLiteGD {
    fn init(base: Base<Resource>) -> Self {
        // Pinned upstream defaults; the authoritative source is
        // `FastNoiseLiteConfig::pinned_defaults` (see the pinned XML).
        let defaults = FastNoiseLiteConfig::pinned_defaults();
        Self {
            base,
            seed_value: defaults.seed,
            frequency_value: defaults.frequency,
            noise_type_value: defaults.noise_type,
            fractal_type_value: defaults.fractal_type,
            fractal_octaves_value: defaults.fractal_octaves,
            fractal_lacunarity_value: defaults.fractal_lacunarity,
            fractal_gain_value: defaults.fractal_gain,
            fractal_weighted_strength_value: defaults.fractal_weighted_strength,
            fractal_ping_pong_strength_value: defaults.fractal_ping_pong_strength,
            cellular_distance_function_value: defaults.cellular_distance_function,
            cellular_jitter_value: defaults.cellular_jitter,
            cellular_return_type_value: defaults.cellular_return_type,
            rotation_type_3d_value: defaults.rotation_type_3d,
            period_value: 1.0 / defaults.frequency,
            seed: PhantomVar::default(),
            frequency: PhantomVar::default(),
            noise_type: PhantomVar::default(),
            fractal_type: PhantomVar::default(),
            fractal_octaves: PhantomVar::default(),
            fractal_lacunarity: PhantomVar::default(),
            fractal_gain: PhantomVar::default(),
            fractal_weighted_strength: PhantomVar::default(),
            fractal_ping_pong_strength: PhantomVar::default(),
            cellular_distance_function: PhantomVar::default(),
            cellular_jitter: PhantomVar::default(),
            cellular_return_type: PhantomVar::default(),
            rotation_type_3d: PhantomVar::default(),
            period: PhantomVar::default(),
        }
    }
}

impl FastNoiseLiteGD {
    /// Snapshot of the current pinned properties as an engine-free config.
    fn config(&self) -> FastNoiseLiteConfig {
        FastNoiseLiteConfig {
            seed: self.seed_value,
            frequency: self.frequency_value,
            noise_type: self.noise_type_value,
            fractal_type: self.fractal_type_value,
            fractal_octaves: self.fractal_octaves_value,
            fractal_lacunarity: self.fractal_lacunarity_value,
            fractal_gain: self.fractal_gain_value,
            fractal_weighted_strength: self.fractal_weighted_strength_value,
            fractal_ping_pong_strength: self.fractal_ping_pong_strength_value,
            cellular_distance_function: self.cellular_distance_function_value,
            cellular_jitter: self.cellular_jitter_value,
            cellular_return_type: self.cellular_return_type_value,
            rotation_type_3d: self.rotation_type_3d_value,
        }
    }
}

#[godot_api]
#[allow(dead_code)] // script-facing enum constants
impl FastNoiseLiteGD {
    // Pinned `NoiseType` enum constants (see upstream ZN_FastNoiseLite.xml).
    const TYPE_OPEN_SIMPLEX_2: i64 = 0;
    const TYPE_OPEN_SIMPLEX_2S: i64 = 1;
    const TYPE_CELLULAR: i64 = 2;
    const TYPE_PERLIN: i64 = 3;
    const TYPE_VALUE_CUBIC: i64 = 4;
    const TYPE_VALUE: i64 = 5;
    // Pinned `FractalType` enum constants.
    const FRACTAL_NONE: i64 = 0;
    const FRACTAL_FBM: i64 = 1;
    const FRACTAL_RIDGED: i64 = 2;
    const FRACTAL_PING_PONG: i64 = 3;
    // Pinned `RotationType3D` enum constants.
    const ROTATION_3D_NONE: i64 = 0;
    const ROTATION_3D_IMPROVE_XY_PLANES: i64 = 1;
    const ROTATION_3D_IMPROVE_XZ_PLANES: i64 = 2;
    // Pinned `CellularDistanceFunction` enum constants.
    const CELLULAR_DISTANCE_EUCLIDEAN: i64 = 0;
    const CELLULAR_DISTANCE_EUCLIDEAN_SQ: i64 = 1;
    const CELLULAR_DISTANCE_MANHATTAN: i64 = 2;
    const CELLULAR_DISTANCE_HYBRID: i64 = 3;
    // Pinned `CellularReturnType` enum constants.
    const CELLULAR_RETURN_CELL_VALUE: i64 = 0;
    const CELLULAR_RETURN_DISTANCE: i64 = 1;
    const CELLULAR_RETURN_DISTANCE_2: i64 = 2;
    const CELLULAR_RETURN_DISTANCE_2_ADD: i64 = 3;
    const CELLULAR_RETURN_DISTANCE_2_SUB: i64 = 4;
    const CELLULAR_RETURN_DISTANCE_2_MUL: i64 = 5;
    const CELLULAR_RETURN_DISTANCE_2_DIV: i64 = 6;

    /// Sample the raw 3D noise at world point `(x,y,z)`, configured from this
    /// resource's seed/frequency/noise_type. Returns a value in roughly
    /// `[-1, 1]`. The result is deterministic for a fixed configuration.
    #[func]
    fn sample_3d(&self, x: f32, y: f32, z: f32) -> f32 {
        if validate_positive_finite_float(self.frequency_value).is_err()
            || [x, y, z].iter().any(|value| !value.is_finite())
        {
            godot_error!("ZN_FastNoiseLite.sample_3d: frequency must be positive and all coordinates must be finite");
            return 0.0;
        }
        self.config().sample_3d(x, y, z)
    }

    /// Sample the raw 2D noise at `(x,y)` using the crate's true 2D sampler
    /// (`get_noise_2d`), configured from this resource's properties. Returns a
    /// value in roughly `[-1, 1]`, deterministic for a fixed configuration.
    #[func]
    fn sample_2d(&self, x: f32, y: f32) -> f32 {
        if validate_positive_finite_float(self.frequency_value).is_err()
            || [x, y].iter().any(|value| !value.is_finite())
        {
            godot_error!("ZN_FastNoiseLite.sample_2d: frequency must be positive and all coordinates must be finite");
            return 0.0;
        }
        self.config().sample_2d(x, y)
    }

    /// Frequency (higher = more detail; strictly positive finite). Updating
    /// `frequency` keeps `period` in sync (`period = 1/frequency`).
    #[func]
    fn get_frequency(&self) -> f32 {
        self.frequency_value
    }

    #[func]
    fn set_frequency(&mut self, value: f32) {
        if validate_positive_finite_float(value).is_err() {
            godot_error!("ZN_FastNoiseLite.set_frequency: value must be finite and positive");
            return;
        }
        self.frequency_value = value;
        self.period_value = 1.0 / value;
    }

    // -----------------------------------------------------------------
    // Pinned ZN_FastNoiseLite methods.
    // -----------------------------------------------------------------

    /// Get the raw 3D noise value at `(x,y,z)`. Equivalent to `sample_3d` but
    /// matches upstream's pinned method name.
    #[func]
    fn get_noise_3d(&self, x: f32, y: f32, z: f32) -> f32 {
        self.sample_3d(x, y, z)
    }

    /// Get the raw 3D noise value at `position`. Matches upstream's pinned
    /// `get_noise_3dv(Vector3)` signature. Delegates to the same engine-free
    /// `sample_3dv` helper exercised by the unit tests (after the same
    /// validation `get_noise_3d` applies).
    #[func]
    fn get_noise_3dv(&self, position: Vector3) -> f32 {
        if validate_positive_finite_float(self.frequency_value).is_err()
            || [position.x, position.y, position.z]
                .iter()
                .any(|value| !value.is_finite())
        {
            godot_error!("ZN_FastNoiseLite.get_noise_3dv: frequency must be positive and all coordinates must be finite");
            return 0.0;
        }
        self.config()
            .sample_3dv((position.x, position.y, position.z))
    }

    /// Get the raw 2D noise value at `(x,y)`, sampled with the crate's true
    /// 2D sampler. Matches upstream's pinned `get_noise_2d`.
    #[func]
    fn get_noise_2d(&self, x: f32, y: f32) -> f32 {
        self.sample_2d(x, y)
    }

    /// Get the raw 2D noise value at `position`. Matches upstream's pinned
    /// `get_noise_2dv`. Delegates to the same engine-free `sample_2dv` helper
    /// exercised by the unit tests.
    #[func]
    fn get_noise_2dv(&self, position: Vector2) -> f32 {
        if validate_positive_finite_float(self.frequency_value).is_err()
            || [position.x, position.y]
                .iter()
                .any(|value| !value.is_finite())
        {
            godot_error!("ZN_FastNoiseLite.get_noise_2dv: frequency must be positive and all coordinates must be finite");
            return 0.0;
        }
        self.config().sample_2dv((position.x, position.y))
    }

    // -----------------------------------------------------------------
    // Pinned ZN_FastNoiseLite properties (transactional get/set pairs).
    // -----------------------------------------------------------------

    /// Noise seed.
    #[func]
    fn get_seed(&self) -> i32 {
        self.seed_value
    }

    #[func]
    fn set_seed(&mut self, seed: i32) {
        self.seed_value = seed;
    }

    /// Noise type enum value (0..=5).
    #[func]
    fn get_noise_type(&self) -> i32 {
        self.noise_type_value
    }

    #[func]
    fn set_noise_type(&mut self, value: i32) {
        self.noise_type_value = value.clamp(0, 5);
    }

    /// Fractal type enum value (0..=3).
    #[func]
    fn get_fractal_type(&self) -> i32 {
        self.fractal_type_value
    }

    #[func]
    fn set_fractal_type(&mut self, value: i32) {
        self.fractal_type_value = value.clamp(0, 3);
    }

    /// Fractal octaves (>=1).
    #[func]
    fn get_fractal_octaves(&self) -> i32 {
        self.fractal_octaves_value
    }

    #[func]
    fn set_fractal_octaves(&mut self, value: i32) {
        self.fractal_octaves_value = value.max(1);
    }

    /// Fractal lacunarity (finite).
    #[func]
    fn get_fractal_lacunarity(&self) -> f32 {
        self.fractal_lacunarity_value
    }

    #[func]
    fn set_fractal_lacunarity(&mut self, value: f32) {
        if validate_finite_float(value).is_err() {
            godot_error!("ZN_FastNoiseLite.set_fractal_lacunarity: value must be finite");
            return;
        }
        self.fractal_lacunarity_value = value;
    }

    /// Fractal gain (finite).
    #[func]
    fn get_fractal_gain(&self) -> f32 {
        self.fractal_gain_value
    }

    #[func]
    fn set_fractal_gain(&mut self, value: f32) {
        if validate_finite_float(value).is_err() {
            godot_error!("ZN_FastNoiseLite.set_fractal_gain: value must be finite");
            return;
        }
        self.fractal_gain_value = value;
    }

    /// Fractal weighted strength (unit float in [0,1]).
    #[func]
    fn get_fractal_weighted_strength(&self) -> f32 {
        self.fractal_weighted_strength_value
    }

    #[func]
    fn set_fractal_weighted_strength(&mut self, value: f32) {
        if validate_unit_float(value).is_err() {
            godot_error!(
                "ZN_FastNoiseLite.set_fractal_weighted_strength: value must be finite and in [0,1]"
            );
            return;
        }
        self.fractal_weighted_strength_value = value;
    }

    /// Fractal ping-pong strength (finite).
    #[func]
    fn get_fractal_ping_pong_strength(&self) -> f32 {
        self.fractal_ping_pong_strength_value
    }

    #[func]
    fn set_fractal_ping_pong_strength(&mut self, value: f32) {
        if validate_finite_float(value).is_err() {
            godot_error!("ZN_FastNoiseLite.set_fractal_ping_pong_strength: value must be finite");
            return;
        }
        self.fractal_ping_pong_strength_value = value;
    }

    /// Cellular distance function enum value (0..=3).
    #[func]
    fn get_cellular_distance_function(&self) -> i32 {
        self.cellular_distance_function_value
    }

    #[func]
    fn set_cellular_distance_function(&mut self, value: i32) {
        self.cellular_distance_function_value = value.clamp(0, 3);
    }

    /// Cellular jitter (finite).
    #[func]
    fn get_cellular_jitter(&self) -> f32 {
        self.cellular_jitter_value
    }

    #[func]
    fn set_cellular_jitter(&mut self, value: f32) {
        if validate_finite_float(value).is_err() {
            godot_error!("ZN_FastNoiseLite.set_cellular_jitter: value must be finite");
            return;
        }
        self.cellular_jitter_value = value;
    }

    /// Cellular return type enum value (0..=6).
    #[func]
    fn get_cellular_return_type(&self) -> i32 {
        self.cellular_return_type_value
    }

    #[func]
    fn set_cellular_return_type(&mut self, value: i32) {
        self.cellular_return_type_value = value.clamp(0, 6);
    }

    /// 3D rotation type enum value (0..=2).
    #[func]
    fn get_rotation_type_3d(&self) -> i32 {
        self.rotation_type_3d_value
    }

    #[func]
    fn set_rotation_type_3d(&mut self, value: i32) {
        self.rotation_type_3d_value = value.clamp(0, 2);
    }

    /// Period (`1/frequency`; strictly positive finite). Updating `period`
    /// keeps `frequency` in sync.
    #[func]
    fn get_period(&self) -> f32 {
        self.period_value
    }

    #[func]
    fn set_period(&mut self, value: f32) {
        if validate_positive_finite_float(value).is_err() {
            godot_error!("ZN_FastNoiseLite.set_period: value must be finite and positive");
            return;
        }
        self.period_value = value;
        self.frequency_value = 1.0 / value;
    }
}

/// FastNoise2 noise resource. The upstream FastNoise2 is a C++ library (not
/// ported to Rust); this binding delegates to the same `fastnoise-lite`
/// sampler used by voxel-core's `Noise` generator so noise sampling is
/// functional through the binding. `sample_3d` returns the 3D noise value at a
/// world point, configured from the resource's seed/frequency, with the
/// terrace and remap output transforms applied when enabled.
///
/// The pinned GDScript-facing properties mirror upstream `FastNoise2`. They are
/// stored faithfully so GDScript reads round-trip and applied to a
/// freshly-built sampler in `sample_*`/`get_noise_*_single`.
#[derive(GodotClass)]
#[class(base = Resource, tool, rename = FastNoise2)]
pub struct FastNoise2GD {
    base: Base<Resource>,
    /// Deterministic seed (default 1337).
    seed_value: i32,
    /// Frequency (backing field; strictly positive finite).
    frequency_value: f32,
    /// Noise type (0..=6). FastNoise2 maps onto fastnoise-lite's 0..=5 plus
    /// sentinel values 5 (encoded node tree) and 6 (cellular value) that fall
    /// back to OpenSimplex2 here.
    noise_type_value: i32,
    /// Fractal type (0..=3).
    fractal_type_value: i32,
    /// Fractal octaves (>=1).
    fractal_octaves_value: i32,
    /// Fractal lacunarity (finite).
    fractal_lacunarity_value: f32,
    /// Fractal gain (finite).
    fractal_gain_value: f32,
    /// Fractal ping-pong strength (finite).
    fractal_ping_pong_strength_value: f32,
    /// Cellular distance function (0..=4).
    cellular_distance_function_value: i32,
    /// Cellular jitter (finite).
    cellular_jitter_value: f32,
    /// Cellular return type (0..=4).
    cellular_return_type_value: i32,
    /// N-th closest cell index used in cellular distance calculations.
    cellular_index0_value: i32,
    /// Second N-th closest cell index.
    cellular_index1_value: i32,
    /// Period (`1/frequency`; strictly positive finite). Upstream exposes
    /// `period` instead of `frequency`.
    period_value: f32,
    /// Encoded FastNoise2 node-tree (base64). Stored but not applied (the
    /// Rust binding does not decode node trees — documented partial).
    encoded_node_tree_value: GString,
    /// Remap toggle (gates the linear output remap in `sample_*`).
    remap_enabled_value: bool,
    /// Remap input minimum (finite).
    remap_input_min_value: f32,
    /// Remap input maximum (finite).
    remap_input_max_value: f32,
    /// Remap output minimum (finite).
    remap_output_min_value: f32,
    /// Remap output maximum (finite).
    remap_output_max_value: f32,
    /// Terrace toggle (gates the stepped terrace transform in `sample_*`).
    terrace_enabled_value: bool,
    /// Terrace multiplier, i.e. step count (finite; 0 is a safe passthrough).
    terrace_multiplier_value: f32,
    /// Terrace smoothness in `[0,1]` (0 = hard steps, 1 = identity).
    terrace_smoothness_value: f32,
    /// The pinned GDScript-facing `seed` property.
    #[var(get = get_seed, set = set_seed)]
    seed: PhantomVar<i32>,
    /// The pinned GDScript-facing `frequency` property.
    #[var(get = get_frequency, set = set_frequency)]
    frequency: PhantomVar<f32>,
    /// The pinned GDScript-facing `noise_type` property.
    #[var(get = get_noise_type, set = set_noise_type)]
    noise_type: PhantomVar<i32>,
    /// The pinned GDScript-facing `fractal_type` property.
    #[var(get = get_fractal_type, set = set_fractal_type)]
    fractal_type: PhantomVar<i32>,
    /// The pinned GDScript-facing `fractal_octaves` property.
    #[var(get = get_fractal_octaves, set = set_fractal_octaves)]
    fractal_octaves: PhantomVar<i32>,
    /// The pinned GDScript-facing `fractal_lacunarity` property.
    #[var(get = get_fractal_lacunarity, set = set_fractal_lacunarity)]
    fractal_lacunarity: PhantomVar<f32>,
    /// The pinned GDScript-facing `fractal_gain` property.
    #[var(get = get_fractal_gain, set = set_fractal_gain)]
    fractal_gain: PhantomVar<f32>,
    /// The pinned GDScript-facing `fractal_ping_pong_strength` property.
    #[var(get = get_fractal_ping_pong_strength, set = set_fractal_ping_pong_strength)]
    fractal_ping_pong_strength: PhantomVar<f32>,
    /// The pinned GDScript-facing `cellular_distance_function` property.
    #[var(get = get_cellular_distance_function, set = set_cellular_distance_function)]
    cellular_distance_function: PhantomVar<i32>,
    /// The pinned GDScript-facing `cellular_jitter` property.
    #[var(get = get_cellular_jitter, set = set_cellular_jitter)]
    cellular_jitter: PhantomVar<f32>,
    /// The pinned GDScript-facing `cellular_return_type` property.
    #[var(get = get_cellular_return_type, set = set_cellular_return_type)]
    cellular_return_type: PhantomVar<i32>,
    /// The pinned GDScript-facing `cellular_index0` property.
    #[var(get = get_cellular_index0, set = set_cellular_index0)]
    cellular_index0: PhantomVar<i32>,
    /// The pinned GDScript-facing `cellular_index1` property.
    #[var(get = get_cellular_index1, set = set_cellular_index1)]
    cellular_index1: PhantomVar<i32>,
    /// The pinned GDScript-facing `period` property.
    #[var(get = get_period, set = set_period)]
    period: PhantomVar<f32>,
    /// The pinned GDScript-facing `encoded_node_tree` property.
    #[var(get = get_encoded_node_tree, set = set_encoded_node_tree)]
    encoded_node_tree: PhantomVar<GString>,
    /// The pinned GDScript-facing `remap_enabled` property.
    #[var(get = is_remap_enabled, set = set_remap_enabled)]
    remap_enabled: PhantomVar<bool>,
    /// The pinned GDScript-facing `remap_input_min` property.
    #[var(get = get_remap_input_min, set = set_remap_input_min)]
    remap_input_min: PhantomVar<f32>,
    /// The pinned GDScript-facing `remap_input_max` property.
    #[var(get = get_remap_input_max, set = set_remap_input_max)]
    remap_input_max: PhantomVar<f32>,
    /// The pinned GDScript-facing `remap_output_min` property.
    #[var(get = get_remap_output_min, set = set_remap_output_min)]
    remap_output_min: PhantomVar<f32>,
    /// The pinned GDScript-facing `remap_output_max` property.
    #[var(get = get_remap_output_max, set = set_remap_output_max)]
    remap_output_max: PhantomVar<f32>,
    /// The pinned GDScript-facing `terrace_enabled` property.
    #[var(get = is_terrace_enabled, set = set_terrace_enabled)]
    terrace_enabled: PhantomVar<bool>,
    /// The pinned GDScript-facing `terrace_multiplier` property.
    #[var(get = get_terrace_multiplier, set = set_terrace_multiplier)]
    terrace_multiplier: PhantomVar<f32>,
    /// The pinned GDScript-facing `terrace_smoothness` property.
    #[var(get = get_terrace_smoothness, set = set_terrace_smoothness)]
    terrace_smoothness: PhantomVar<f32>,
}
#[godot_api]
impl IResource for FastNoise2GD {
    fn init(base: Base<Resource>) -> Self {
        // Upstream defaults: seed=1337, noise_type=OpenSimplex2(0),
        // fractal_type=None(0), fractal_octaves=3, fractal_lacunarity=2.0,
        // fractal_gain=0.5, fractal_ping_pong_strength=2.0,
        // cellular_distance_function=Euclidean(0), cellular_jitter=1.0,
        // cellular_return_type=Index0(0), cellular_index0=0, cellular_index1=1,
        // period=64.0, remap_enabled=false, remap_input_min=-1.0,
        // remap_input_max=1.0, remap_output_min=-1.0, remap_output_max=1.0,
        // terrace_enabled=false, terrace_multiplier=1.0, terrace_smoothness=0.0.
        let period = 64.0_f32;
        Self {
            base,
            seed_value: 1337,
            frequency_value: 1.0 / period,
            noise_type_value: 0,
            fractal_type_value: 0,
            fractal_octaves_value: 3,
            fractal_lacunarity_value: 2.0,
            fractal_gain_value: 0.5,
            fractal_ping_pong_strength_value: 2.0,
            cellular_distance_function_value: 0,
            cellular_jitter_value: 1.0,
            cellular_return_type_value: 0,
            cellular_index0_value: 0,
            cellular_index1_value: 1,
            period_value: period,
            encoded_node_tree_value: GString::new(),
            remap_enabled_value: false,
            remap_input_min_value: -1.0,
            remap_input_max_value: 1.0,
            remap_output_min_value: -1.0,
            remap_output_max_value: 1.0,
            terrace_enabled_value: false,
            terrace_multiplier_value: 1.0,
            terrace_smoothness_value: 0.0,
            seed: PhantomVar::default(),
            frequency: PhantomVar::default(),
            noise_type: PhantomVar::default(),
            fractal_type: PhantomVar::default(),
            fractal_octaves: PhantomVar::default(),
            fractal_lacunarity: PhantomVar::default(),
            fractal_gain: PhantomVar::default(),
            fractal_ping_pong_strength: PhantomVar::default(),
            cellular_distance_function: PhantomVar::default(),
            cellular_jitter: PhantomVar::default(),
            cellular_return_type: PhantomVar::default(),
            cellular_index0: PhantomVar::default(),
            cellular_index1: PhantomVar::default(),
            period: PhantomVar::default(),
            encoded_node_tree: PhantomVar::default(),
            remap_enabled: PhantomVar::default(),
            remap_input_min: PhantomVar::default(),
            remap_input_max: PhantomVar::default(),
            remap_output_min: PhantomVar::default(),
            remap_output_max: PhantomVar::default(),
            terrace_enabled: PhantomVar::default(),
            terrace_multiplier: PhantomVar::default(),
            terrace_smoothness: PhantomVar::default(),
        }
    }
}

impl FastNoise2GD {
    /// Build a fresh fastnoise-lite sampler configured from this resource's
    /// pinned properties. Used by `sample_*`/`get_noise_*_single` so the noise
    /// reflects the configured fractal/cellular settings.
    fn build_sampler(&self) -> voxel_core::fastnoise_lite::FastNoiseLite {
        use voxel_core::fastnoise_lite::{
            CellularDistanceFunction, CellularReturnType, FastNoiseLite, FractalType, NoiseType,
        };
        let mut noise = FastNoiseLite::new();
        noise.set_seed(Some(self.seed_value));
        noise.set_frequency(Some(self.frequency_value));
        // FastNoise2 noise types 1/2/3/4 map onto fastnoise-lite's
        // SIMPLEX/PERLIN/VALUE/CELLULAR. Sentinel values 5 (encoded node tree)
        // and 6 (cellular value) currently fall back to OpenSimplex2.
        noise.set_noise_type(Some(match self.noise_type_value {
            1 => NoiseType::OpenSimplex2S,
            2 => NoiseType::Perlin,
            3 => NoiseType::Value,
            4 => NoiseType::Cellular,
            _ => NoiseType::OpenSimplex2,
        }));
        noise.set_fractal_type(Some(match self.fractal_type_value {
            0 => FractalType::None,
            2 => FractalType::Ridged,
            3 => FractalType::PingPong,
            _ => FractalType::FBm,
        }));
        noise.set_fractal_octaves(Some(self.fractal_octaves_value.max(1)));
        noise.set_fractal_lacunarity(Some(self.fractal_lacunarity_value));
        noise.set_fractal_gain(Some(self.fractal_gain_value));
        noise.set_fractal_ping_pong_strength(Some(self.fractal_ping_pong_strength_value));
        noise.set_cellular_distance_function(Some(match self.cellular_distance_function_value {
            1 => CellularDistanceFunction::EuclideanSq,
            2 => CellularDistanceFunction::Manhattan,
            3 => CellularDistanceFunction::Hybrid,
            _ => CellularDistanceFunction::Euclidean,
        }));
        // FastNoise2's return-type enum (0..=4) maps onto fastnoise-lite's
        // 7-value enum by treating values >=1 as Distance2-family.
        noise.set_cellular_return_type(Some(match self.cellular_return_type_value {
            0 => CellularReturnType::CellValue,
            1 => CellularReturnType::Distance2Add,
            2 => CellularReturnType::Distance2Sub,
            3 => CellularReturnType::Distance2Mul,
            4 => CellularReturnType::Distance2Div,
            _ => CellularReturnType::Distance,
        }));
        noise.set_cellular_jitter(Some(self.cellular_jitter_value));
        noise
    }
}

impl FastNoise2GD {
    /// Apply the pinned output transforms to a raw sample, in the adopted
    /// chain order `raw -> terrace -> remap` (terrace shapes the noise, remap
    /// rescales the final range — matching how upstream chains the nodes).
    fn transform_output(&self, raw: f32) -> f32 {
        let terraced = apply_terrace(
            raw,
            self.terrace_multiplier_value,
            self.terrace_smoothness_value,
            self.terrace_enabled_value,
        );
        apply_remap(
            terraced,
            self.remap_input_min_value,
            self.remap_input_max_value,
            self.remap_output_min_value,
            self.remap_output_max_value,
            self.remap_enabled_value,
        )
    }
}

#[godot_api]
#[allow(dead_code)] // script-facing enum constants
impl FastNoise2GD {
    // Pinned `NoiseType` enum constants (see upstream FastNoise2.xml).
    const TYPE_OPEN_SIMPLEX_2: i64 = 0;
    const TYPE_SIMPLEX: i64 = 1;
    const TYPE_PERLIN: i64 = 2;
    const TYPE_VALUE: i64 = 3;
    const TYPE_CELLULAR: i64 = 4;
    const TYPE_ENCODED_NODE_TREE: i64 = 5;
    const TYPE_CELLULAR_VALUE: i64 = 6;
    // Pinned `FractalType` enum constants.
    const FRACTAL_NONE: i64 = 0;
    const FRACTAL_FBM: i64 = 1;
    const FRACTAL_RIDGED: i64 = 2;
    const FRACTAL_PING_PONG: i64 = 3;
    // Pinned `CellularDistanceFunction` enum constants.
    const CELLULAR_DISTANCE_EUCLIDEAN: i64 = 0;
    const CELLULAR_DISTANCE_EUCLIDEAN_SQ: i64 = 1;
    const CELLULAR_DISTANCE_MANHATTAN: i64 = 2;
    const CELLULAR_DISTANCE_HYBRID: i64 = 3;
    const CELLULAR_DISTANCE_MAX_AXIS: i64 = 4;
    // Pinned `CellularReturnType` enum constants.
    const CELLULAR_RETURN_INDEX_0: i64 = 0;
    const CELLULAR_RETURN_INDEX_0_ADD_1: i64 = 1;
    const CELLULAR_RETURN_INDEX_0_SUB_1: i64 = 2;
    const CELLULAR_RETURN_INDEX_0_MUL_1: i64 = 3;
    const CELLULAR_RETURN_INDEX_0_DIV_1: i64 = 4;

    // Pinned `SIMDLevel` enum constants (see upstream FastNoise2.xml).
    const SIMD_NULL: i64 = 0;
    const SIMD_SCALAR: i64 = 1;
    const SIMD_SSE: i64 = 2;
    const SIMD_SSE2: i64 = 4;
    const SIMD_SSE3: i64 = 8;
    const SIMD_SSSE3: i64 = 16;
    const SIMD_SSE41: i64 = 32;
    const SIMD_SSE42: i64 = 64;
    const SIMD_AVX: i64 = 128;
    const SIMD_AVX2: i64 = 256;
    const SIMD_AVX512: i64 = 512;
    const SIMD_NEON: i64 = 65_536;

    /// Sample the raw 3D noise at world point `(x,y,z)`, then apply the
    /// terrace/remap output transforms. Deterministic for a fixed
    /// seed/frequency. Delegates to the `fastnoise-lite` sampler.
    #[func]
    fn sample_3d(&self, x: f32, y: f32, z: f32) -> f32 {
        if validate_positive_finite_float(self.frequency_value).is_err()
            || [x, y, z].iter().any(|value| !value.is_finite())
        {
            godot_error!("FastNoise2.sample_3d: frequency must be positive and all coordinates must be finite");
            return 0.0;
        }
        let raw = self.build_sampler().get_noise_3d(x, y, z);
        self.transform_output(raw)
    }

    /// Sample raw 2D noise at `(x, z)` (Y = 0), then apply the terrace/remap
    /// output transforms. Useful for heightmap-style use.
    #[func]
    fn sample_2d(&self, x: f32, z: f32) -> f32 {
        if validate_positive_finite_float(self.frequency_value).is_err()
            || [x, z].iter().any(|value| !value.is_finite())
        {
            godot_error!("FastNoise2.sample_2d: frequency must be positive and all coordinates must be finite");
            return 0.0;
        }
        let raw = self.build_sampler().get_noise_3d(x, 0.0, z);
        self.transform_output(raw)
    }

    /// Frequency (strictly positive finite). Updating `frequency` keeps
    /// `period` in sync.
    #[func]
    fn get_frequency(&self) -> f32 {
        self.frequency_value
    }

    #[func]
    fn set_frequency(&mut self, value: f32) {
        if validate_positive_finite_float(value).is_err() {
            godot_error!("FastNoise2.set_frequency: value must be finite and positive");
            return;
        }
        self.frequency_value = value;
        self.period_value = 1.0 / value;
    }

    // -----------------------------------------------------------------
    // Pinned FastNoise2 methods.
    // -----------------------------------------------------------------

    /// Generate a single value of 3D noise at `position`. Matches upstream's
    /// pinned `get_noise_3d_single` signature.
    #[func]
    fn get_noise_3d_single(&self, position: Vector3) -> f32 {
        self.sample_3d(position.x, position.y, position.z)
    }

    /// Generate a single value of 2D noise at `position` (Y = 0). Matches
    /// upstream's pinned `get_noise_2d_single` signature.
    #[func]
    fn get_noise_2d_single(&self, position: Vector2) -> f32 {
        self.sample_2d(position.x, position.y)
    }

    /// Rebuild the internal noise generator after property changes. The Rust
    /// binding rebuilds the sampler on every `sample_*` call, so this is a
    /// no-op kept for API compatibility.
    #[func]
    fn update_generator(&mut self) {}

    /// Which SIMD level the library detected. The Rust binding does not use
    /// SIMD (it delegates to the scalar `fastnoise-lite` crate); always reports
    /// `SIMD_SCALAR` (1).
    #[func]
    fn get_simd_level(&self) -> i64 {
        Self::SIMD_SCALAR
    }

    /// Human-readable name of a SIMD level. Matches upstream's pinned static
    /// `get_simd_level_name`.
    #[func]
    fn get_simd_level_name(level: i64) -> GString {
        match level {
            x if x == Self::SIMD_NULL => "null",
            x if x == Self::SIMD_SCALAR => "scalar",
            x if x == Self::SIMD_SSE => "SSE",
            x if x == Self::SIMD_SSE2 => "SSE2",
            x if x == Self::SIMD_SSE3 => "SSE3",
            x if x == Self::SIMD_SSSE3 => "SSSE3",
            x if x == Self::SIMD_SSE41 => "SSE41",
            x if x == Self::SIMD_SSE42 => "SSE42",
            x if x == Self::SIMD_AVX => "AVX",
            x if x == Self::SIMD_AVX2 => "AVX2",
            x if x == Self::SIMD_AVX512 => "AVX512",
            x if x == Self::SIMD_NEON => "NEON",
            _ => "unknown",
        }
        .to_godot()
    }

    /// Fill a greyscale `image` with noise values, sized to the image's
    /// current dimensions. Samples the resource's configured sampler (with the
    /// terrace/remap output transforms applied) at integer pixel coordinates,
    /// mirroring `voxel_core`'s row-major `generate_image_2d` layout. Pixels
    /// are normalized from the batch's *observed* minimum/maximum to
    /// `[0, 1]` (the remap transform may move the raw `[-1, 1]` range); a
    /// constant batch writes mid-gray. The image must not be compressed.
    /// `tileable = true` is not supported by the pure-Rust backend: a one-time
    /// warning is emitted and the non-tileable image is still produced.
    #[func]
    fn generate_image(&self, image: Gd<godot::classes::Image>, tileable: bool) {
        let mut image = image;
        let width = image.get_width();
        let height = image.get_height();
        if width <= 0 || height <= 0 || image.is_empty() {
            godot_error!(
                "FastNoise2.generate_image: image must be non-empty with positive dimensions"
            );
            return;
        }
        if image.is_compressed() {
            godot_error!("FastNoise2.generate_image: image must not be compressed");
            return;
        }
        let pixel_count = i64::from(width) * i64::from(height);
        if pixel_count > MAX_GENERATED_IMAGE_PIXELS {
            godot_error!(
                "FastNoise2.generate_image: pixel count exceeds the script workload limit"
            );
            return;
        }
        if tileable {
            warn_tiling_unsupported_once();
        }
        // Sample the whole batch first so normalization can use the observed
        // min/max (a plain [-1, 1] assumption breaks when remap is enabled).
        let sampler = self.build_sampler();
        let mut samples = Vec::with_capacity(pixel_count as usize);
        for y in 0..height {
            for x in 0..width {
                let raw = sampler.get_noise_3d(x as f32, 0.0, y as f32);
                samples.push(self.transform_output(raw));
            }
        }
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        for &sample in &samples {
            min = min.min(sample);
            max = max.max(sample);
        }
        let mut index = 0;
        for y in 0..height {
            for x in 0..width {
                let gray = normalize_gray(samples[index], min, max);
                index += 1;
                image.set_pixel(x, y, Color::from_rgba(gray, gray, gray, 1.0));
            }
        }
    }

    // -----------------------------------------------------------------
    // Pinned FastNoise2 properties (transactional get/set pairs).
    // -----------------------------------------------------------------

    /// Noise seed.
    #[func]
    fn get_seed(&self) -> i32 {
        self.seed_value
    }

    #[func]
    fn set_seed(&mut self, seed: i32) {
        self.seed_value = seed;
    }

    /// Noise type enum value (0..=6).
    #[func]
    fn get_noise_type(&self) -> i32 {
        self.noise_type_value
    }

    #[func]
    fn set_noise_type(&mut self, value: i32) {
        self.noise_type_value = value.clamp(0, 6);
    }

    /// Fractal type enum value (0..=3).
    #[func]
    fn get_fractal_type(&self) -> i32 {
        self.fractal_type_value
    }

    #[func]
    fn set_fractal_type(&mut self, value: i32) {
        self.fractal_type_value = value.clamp(0, 3);
    }

    /// Fractal octaves (>=1).
    #[func]
    fn get_fractal_octaves(&self) -> i32 {
        self.fractal_octaves_value
    }

    #[func]
    fn set_fractal_octaves(&mut self, value: i32) {
        self.fractal_octaves_value = value.max(1);
    }

    /// Fractal lacunarity (finite).
    #[func]
    fn get_fractal_lacunarity(&self) -> f32 {
        self.fractal_lacunarity_value
    }

    #[func]
    fn set_fractal_lacunarity(&mut self, value: f32) {
        if validate_finite_float(value).is_err() {
            godot_error!("FastNoise2.set_fractal_lacunarity: value must be finite");
            return;
        }
        self.fractal_lacunarity_value = value;
    }

    /// Fractal gain (finite).
    #[func]
    fn get_fractal_gain(&self) -> f32 {
        self.fractal_gain_value
    }

    #[func]
    fn set_fractal_gain(&mut self, value: f32) {
        if validate_finite_float(value).is_err() {
            godot_error!("FastNoise2.set_fractal_gain: value must be finite");
            return;
        }
        self.fractal_gain_value = value;
    }

    /// Fractal ping-pong strength (finite).
    #[func]
    fn get_fractal_ping_pong_strength(&self) -> f32 {
        self.fractal_ping_pong_strength_value
    }

    #[func]
    fn set_fractal_ping_pong_strength(&mut self, value: f32) {
        if validate_finite_float(value).is_err() {
            godot_error!("FastNoise2.set_fractal_ping_pong_strength: value must be finite");
            return;
        }
        self.fractal_ping_pong_strength_value = value;
    }

    /// Cellular distance function enum value (0..=4).
    #[func]
    fn get_cellular_distance_function(&self) -> i32 {
        self.cellular_distance_function_value
    }

    #[func]
    fn set_cellular_distance_function(&mut self, value: i32) {
        self.cellular_distance_function_value = value.clamp(0, 4);
    }

    /// Cellular jitter (finite).
    #[func]
    fn get_cellular_jitter(&self) -> f32 {
        self.cellular_jitter_value
    }

    #[func]
    fn set_cellular_jitter(&mut self, value: f32) {
        if validate_finite_float(value).is_err() {
            godot_error!("FastNoise2.set_cellular_jitter: value must be finite");
            return;
        }
        self.cellular_jitter_value = value;
    }

    /// Cellular return type enum value (0..=4).
    #[func]
    fn get_cellular_return_type(&self) -> i32 {
        self.cellular_return_type_value
    }

    #[func]
    fn set_cellular_return_type(&mut self, value: i32) {
        self.cellular_return_type_value = value.clamp(0, 4);
    }

    /// N-th closest cell used in cellular distance/value calculations.
    #[func]
    fn get_cellular_index0(&self) -> i32 {
        self.cellular_index0_value
    }

    #[func]
    fn set_cellular_index0(&mut self, value: i32) {
        self.cellular_index0_value = value.max(0);
    }

    /// Second N-th closest cell index used in some cellular calculations.
    #[func]
    fn get_cellular_index1(&self) -> i32 {
        self.cellular_index1_value
    }

    #[func]
    fn set_cellular_index1(&mut self, value: i32) {
        self.cellular_index1_value = value.max(0);
    }

    /// Period (`1/frequency`; strictly positive finite). Updating `period`
    /// keeps `frequency` in sync.
    #[func]
    fn get_period(&self) -> f32 {
        self.period_value
    }

    #[func]
    fn set_period(&mut self, value: f32) {
        if validate_positive_finite_float(value).is_err() {
            godot_error!("FastNoise2.set_period: value must be finite and positive");
            return;
        }
        self.period_value = value;
        self.frequency_value = 1.0 / value;
    }

    /// Encoded FastNoise2 node-tree string. Stored faithfully; the Rust
    /// binding does not yet decode node trees.
    #[func]
    fn get_encoded_node_tree(&self) -> GString {
        self.encoded_node_tree_value.clone()
    }

    #[func]
    fn set_encoded_node_tree(&mut self, value: GString) {
        self.encoded_node_tree_value = value;
    }

    /// Whether remapping of output values is enabled.
    #[func]
    fn is_remap_enabled(&self) -> bool {
        self.remap_enabled_value
    }

    #[func]
    fn set_remap_enabled(&mut self, enabled: bool) {
        self.remap_enabled_value = enabled;
    }

    /// Remap input minimum (finite).
    #[func]
    fn get_remap_input_min(&self) -> f32 {
        self.remap_input_min_value
    }

    #[func]
    fn set_remap_input_min(&mut self, value: f32) {
        if validate_finite_float(value).is_err() {
            godot_error!("FastNoise2.set_remap_input_min: value must be finite");
            return;
        }
        self.remap_input_min_value = value;
    }

    /// Remap input maximum (finite).
    #[func]
    fn get_remap_input_max(&self) -> f32 {
        self.remap_input_max_value
    }

    #[func]
    fn set_remap_input_max(&mut self, value: f32) {
        if validate_finite_float(value).is_err() {
            godot_error!("FastNoise2.set_remap_input_max: value must be finite");
            return;
        }
        self.remap_input_max_value = value;
    }

    /// Remap output minimum (finite).
    #[func]
    fn get_remap_output_min(&self) -> f32 {
        self.remap_output_min_value
    }

    #[func]
    fn set_remap_output_min(&mut self, value: f32) {
        if validate_finite_float(value).is_err() {
            godot_error!("FastNoise2.set_remap_output_min: value must be finite");
            return;
        }
        self.remap_output_min_value = value;
    }

    /// Remap output maximum (finite).
    #[func]
    fn get_remap_output_max(&self) -> f32 {
        self.remap_output_max_value
    }

    #[func]
    fn set_remap_output_max(&mut self, value: f32) {
        if validate_finite_float(value).is_err() {
            godot_error!("FastNoise2.set_remap_output_max: value must be finite");
            return;
        }
        self.remap_output_max_value = value;
    }

    /// Whether terrace (stepping) transformation is enabled.
    #[func]
    fn is_terrace_enabled(&self) -> bool {
        self.terrace_enabled_value
    }

    #[func]
    fn set_terrace_enabled(&mut self, enabled: bool) {
        self.terrace_enabled_value = enabled;
    }

    /// Terrace multiplier (finite).
    #[func]
    fn get_terrace_multiplier(&self) -> f32 {
        self.terrace_multiplier_value
    }

    #[func]
    fn set_terrace_multiplier(&mut self, value: f32) {
        if validate_finite_float(value).is_err() {
            godot_error!("FastNoise2.set_terrace_multiplier: value must be finite");
            return;
        }
        self.terrace_multiplier_value = value;
    }

    /// Terrace smoothness (finite).
    #[func]
    fn get_terrace_smoothness(&self) -> f32 {
        self.terrace_smoothness_value
    }

    #[func]
    fn set_terrace_smoothness(&mut self, value: f32) {
        if validate_finite_float(value).is_err() {
            godot_error!("FastNoise2.set_terrace_smoothness: value must be finite");
            return;
        }
        self.terrace_smoothness_value = value;
    }
}

/// Spot noise resource — very specialized cellular noise for generating
/// "spots" in a grid (typical use case: ores in terrain), mirroring upstream
/// `ZN_SpotNoise` / `math/spot_noise.h`. Space is divided into cells; each
/// cell owns one spot at a hash-jittered position, `get_noise_2d/3d` return
/// 1.0 inside a spot and 0.0 outside, and `get_spot_positions_in_area_2d/3d`
/// enumerate the same spot centers. Only the query point's *containing* cell
/// is checked (no neighbor lookup): upstream accepts that high jitter clips
/// spots at cell borders — a documented limitation of this noise.
///
/// The pinned GDScript-facing properties (`cell_size`, `jitter`, `seed`,
/// `spot_radius`) mirror upstream `ZN_SpotNoise` (5828cbeb). They are stored
/// faithfully so GDScript reads round-trip and used by the engine-free
/// spot-grid helpers in this module, which reproduce upstream's integer hash
/// exactly (see [`spot_hash_2d`]). The legacy `count_spots`/`density`/`radius`
/// API predates the port and keeps its original noise-threshold semantics.
#[derive(GodotClass)]
#[class(base = Resource, tool, rename = ZN_SpotNoise)]
pub struct SpotNoiseGD {
    base: Base<Resource>,
    /// Density threshold (backing field; unit float in [0,1]).
    density_value: f32,
    /// Spot radius (backing field; strictly positive finite).
    radius_value: f32,
    /// Deterministic seed (backing field for the pinned `seed` property).
    seed_value: i32,
    /// Pinned `cell_size` (backing field; strictly positive finite). Upstream
    /// default 32.0.
    cell_size_value: f32,
    /// Pinned `jitter` (backing field; unit float in [0,1]). Upstream default
    /// 0.9.
    jitter_value: f32,
    /// Pinned `spot_radius` (backing field; strictly positive finite). Upstream
    /// default 3.0.
    spot_radius_value: f32,
    /// The pinned GDScript-facing `seed` property.
    #[var(get = get_seed, set = set_seed)]
    seed: PhantomVar<i32>,
    /// The pinned GDScript-facing `cell_size` property.
    #[var(get = get_cell_size, set = set_cell_size)]
    cell_size: PhantomVar<f32>,
    /// The pinned GDScript-facing `jitter` property.
    #[var(get = get_jitter, set = set_jitter)]
    jitter: PhantomVar<f32>,
    /// The pinned GDScript-facing `spot_radius` property.
    #[var(get = get_spot_radius, set = set_spot_radius)]
    spot_radius: PhantomVar<f32>,
    #[var(get = get_density, set = set_density)]
    density: PhantomVar<f32>,
    #[var(get = get_radius, set = set_radius)]
    radius: PhantomVar<f32>,
}
#[godot_api]
impl IResource for SpotNoiseGD {
    fn init(base: Base<Resource>) -> Self {
        // Upstream defaults: seed=1337, cell_size=32.0, jitter=0.9,
        // spot_radius=3.0. The legacy density/radius pair keeps its original
        // defaults (used by the existing `count_spots` delegate).
        Self {
            base,
            density_value: 0.5,
            radius_value: 2.0,
            seed_value: 1337,
            cell_size_value: 32.0,
            jitter_value: 0.9,
            spot_radius_value: 3.0,
            seed: PhantomVar::default(),
            cell_size: PhantomVar::default(),
            jitter: PhantomVar::default(),
            spot_radius: PhantomVar::default(),
            density: PhantomVar::default(),
            radius: PhantomVar::default(),
        }
    }
}

#[godot_api]
impl SpotNoiseGD {
    /// Count the spots that would be placed over a `grid_size`×`grid_size`
    /// area. Each cell is accepted if its 3D noise sample (scaled by `radius`)
    /// is below the density threshold. Deterministic for a fixed seed.
    #[func]
    fn count_spots(&self, grid_size: i32) -> i32 {
        let Ok(expected_work) = validate_spot_grid_work(grid_size) else {
            godot_error!("ZN_SpotNoise.count_spots: grid exceeds the script workload limit");
            return -1;
        };
        if validate_unit_float(self.density_value).is_err()
            || validate_spot_coordinate_work(grid_size, self.radius_value).is_err()
        {
            godot_error!(
                "ZN_SpotNoise.count_spots: density, radius, or scaled coordinates are invalid"
            );
            return -1;
        }
        let mut gen = voxel_core::generators::simple::Noise::default();
        let noise = gen.noise_mut();
        noise.set_seed(Some(self.seed_value));
        noise.set_frequency(Some(1.0 / self.radius_value.max(0.0001)));
        let mut count = 0u64;
        let scale = self.radius_value;
        for y in 0..grid_size {
            for x in 0..grid_size {
                let v = gen.sample_noise_3d(x as f32 * scale, 0.0, y as f32 * scale);
                // Normalize noise [-1,1] → [0,1], accept if below density.
                let n = (v + 1.0) * 0.5;
                if n < self.density_value {
                    let Some(next_count) = count.checked_add(1) else {
                        godot_error!("ZN_SpotNoise.count_spots: result count overflowed");
                        return -1;
                    };
                    count = next_count;
                }
            }
        }
        debug_assert!(count <= expected_work);
        i32::try_from(count).unwrap_or_else(|_| {
            godot_error!("ZN_SpotNoise.count_spots: result does not fit i32");
            -1
        })
    }

    /// Density threshold in [0,1] (finite unit float).
    #[func]
    fn get_density(&self) -> f32 {
        self.density_value
    }

    #[func]
    fn set_density(&mut self, value: f32) {
        if validate_unit_float(value).is_err() {
            godot_error!("ZN_SpotNoise.set_density: value must be finite and between zero and one");
            return;
        }
        self.density_value = value;
    }

    /// Spot radius (strictly positive finite).
    #[func]
    fn get_radius(&self) -> f32 {
        self.radius_value
    }

    #[func]
    fn set_radius(&mut self, value: f32) {
        if validate_positive_finite_float(value).is_err() {
            godot_error!("ZN_SpotNoise.set_radius: value must be finite and positive");
            return;
        }
        self.radius_value = value;
    }

    // -----------------------------------------------------------------
    // Pinned ZN_SpotNoise methods (upstream 5828cbeb: ZN_SpotNoise.xml).
    // -----------------------------------------------------------------

    /// Get the raw 2D spot-noise value at `(x, y)`. Returns 1.0 when the point
    /// is inside a spot, 0.0 otherwise. Matches upstream's pinned `get_noise_2d`.
    #[func]
    fn get_noise_2d(&self, x: f32, y: f32) -> f32 {
        self.spot_noise_2d(x, y)
    }

    /// Get the raw 2D spot-noise value at `pos`. Matches upstream's pinned
    /// `get_noise_2dv`.
    #[func]
    fn get_noise_2dv(&self, pos: Vector2) -> f32 {
        self.spot_noise_2d(pos.x, pos.y)
    }

    /// Get the raw 3D spot-noise value at `(x, y, z)`. Matches upstream's pinned
    /// `get_noise_3d`.
    #[func]
    fn get_noise_3d(&self, x: f32, y: f32, z: f32) -> f32 {
        self.spot_noise_3d(x, y, z)
    }

    /// Get the raw 3D spot-noise value at `pos`. Matches upstream's pinned
    /// `get_noise_3dv`.
    #[func]
    fn get_noise_3dv(&self, pos: Vector3) -> f32 {
        self.spot_noise_3d(pos.x, pos.y, pos.z)
    }

    /// Get the center positions of spots contained in `rect` (2D). Matches
    /// upstream's pinned `get_spot_positions_in_area_2d`: every grid cell
    /// owns exactly one spot whose (jittered) center is reported when it lies
    /// inside the rect. The visited-cell workload is bounded like
    /// `count_spots`; an over-large rect logs an error and returns an empty
    /// array.
    #[func]
    fn get_spot_positions_in_area_2d(&self, rect: Rect2) -> PackedVector2Array {
        let Ok(cfg) = self.spot_grid_config() else {
            godot_error!("ZN_SpotNoise.get_spot_positions_in_area_2d: cell_size and spot_radius must be finite and positive, jitter in [0,1]");
            return PackedVector2Array::new();
        };
        let bounds = (rect.position.x, rect.position.y, rect.end().x, rect.end().y);
        match spot_centers_in_rect(cfg, bounds) {
            Ok(centers) => {
                let positions: Vec<Vector2> = centers
                    .into_iter()
                    .map(|(x, y)| Vector2::new(x, y))
                    .collect();
                PackedVector2Array::from(positions.as_slice())
            }
            Err(message) => {
                godot_error!("ZN_SpotNoise.get_spot_positions_in_area_2d: {message}");
                PackedVector2Array::new()
            }
        }
    }

    /// Get the center positions of spots contained in `aabb` (3D). Matches
    /// upstream's pinned `get_spot_positions_in_area_3d` under upstream's
    /// true-3D spot grid (`get_noise_3d` floors on all three axes): every 3D
    /// cell overlapping the aabb owns exactly one spot whose jittered center
    /// is reported when it lies inside the aabb. Workload-bounded like the 2D
    /// variant.
    #[func]
    fn get_spot_positions_in_area_3d(&self, aabb: Aabb) -> PackedVector3Array {
        let Ok(cfg) = self.spot_grid_config() else {
            godot_error!("ZN_SpotNoise.get_spot_positions_in_area_3d: cell_size and spot_radius must be finite and positive, jitter in [0,1]");
            return PackedVector3Array::new();
        };
        let bounds = (
            aabb.position.x,
            aabb.position.y,
            aabb.position.z,
            aabb.end().x,
            aabb.end().y,
            aabb.end().z,
        );
        match spot_centers_in_aabb(cfg, bounds) {
            Ok(centers) => {
                let positions: Vec<Vector3> = centers
                    .into_iter()
                    .map(|(x, y, z)| Vector3::new(x, y, z))
                    .collect();
                PackedVector3Array::from(positions.as_slice())
            }
            Err(message) => {
                godot_error!("ZN_SpotNoise.get_spot_positions_in_area_3d: {message}");
                PackedVector3Array::new()
            }
        }
    }

    // -----------------------------------------------------------------
    // Pinned ZN_SpotNoise properties (transactional get/set pairs).
    // -----------------------------------------------------------------

    /// Noise seed (upstream default 1337).
    #[func]
    fn get_seed(&self) -> i32 {
        self.seed_value
    }

    #[func]
    fn set_seed(&mut self, seed: i32) {
        self.seed_value = seed;
    }

    /// Cell size of the spot grid (strictly positive finite; default 32.0).
    #[func]
    fn get_cell_size(&self) -> f32 {
        self.cell_size_value
    }

    #[func]
    fn set_cell_size(&mut self, value: f32) {
        if validate_positive_finite_float(value).is_err() {
            godot_error!("ZN_SpotNoise.set_cell_size: value must be finite and positive");
            return;
        }
        self.cell_size_value = value;
    }

    /// Jitter applied to spot centers within their cell (unit float in [0,1];
    /// default 0.9).
    #[func]
    fn get_jitter(&self) -> f32 {
        self.jitter_value
    }

    #[func]
    fn set_jitter(&mut self, value: f32) {
        if validate_unit_float(value).is_err() {
            godot_error!("ZN_SpotNoise.set_jitter: value must be finite and between zero and one");
            return;
        }
        self.jitter_value = value;
    }

    /// Radius of each spot (strictly positive finite; default 3.0).
    #[func]
    fn get_spot_radius(&self) -> f32 {
        self.spot_radius_value
    }

    #[func]
    fn set_spot_radius(&mut self, value: f32) {
        if validate_positive_finite_float(value).is_err() {
            godot_error!("ZN_SpotNoise.set_spot_radius: value must be finite and positive");
            return;
        }
        self.spot_radius_value = value;
    }
}

impl SpotNoiseGD {
    /// Validated engine-free snapshot of the pinned spot-grid configuration.
    /// `Ok` iff `cell_size`/`spot_radius` are finite and positive and `jitter`
    /// is a unit float.
    fn spot_grid_config(&self) -> Result<SpotGridConfig, &'static str> {
        SpotGridConfig::new(
            self.cell_size_value,
            self.jitter_value,
            self.spot_radius_value,
            self.seed_value,
        )
    }

    /// Evaluate the 2D spot field at `(x, y)` (upstream `spot_noise_2d`):
    /// 1.0 when the point lies inside its own cell's spot, 0.0 otherwise.
    fn spot_noise_2d(&self, x: f32, y: f32) -> f32 {
        match self.spot_grid_config() {
            Ok(cfg) if x.is_finite() && y.is_finite() => spot_grid_noise_2d(cfg, x, y),
            _ => {
                godot_error!("ZN_SpotNoise: cell_size/spot_radius must be finite and positive");
                0.0
            }
        }
    }

    /// Evaluate the 3D spot field at `(x, y, z)` (upstream `spot_noise_3d`):
    /// true 3D cells — floor on all three axes, a 3-component cell hash and a
    /// 3D squared-distance test against `spot_radius`.
    fn spot_noise_3d(&self, x: f32, y: f32, z: f32) -> f32 {
        match self.spot_grid_config() {
            Ok(cfg) if [x, y, z].iter().all(|value| value.is_finite()) => {
                spot_grid_noise_3d(cfg, x, y, z)
            }
            _ => {
                godot_error!("ZN_SpotNoise: cell_size/spot_radius must be finite and positive");
                0.0
            }
        }
    }
}

// === Upstream SpotNoise model (mirrors zylann's math/spot_noise.h) ===
//
// The integer hash (including its PRIME constants and multiplier), the
// hash-to-jitter decoding and the containing-cell distance test are
// replicated exactly, so spot positions are numerically upstream-identical.
// Upstream relies on wrapping 32-bit int arithmetic; Rust needs explicit
// wrapping ops to reproduce it.

/// Upstream `PRIME_X` constant.
const SPOT_PRIME_X: i32 = 501_125_321;
/// Upstream `PRIME_Y` constant.
const SPOT_PRIME_Y: i32 = 1_136_930_381;
/// Upstream `PRIME_Z` constant.
const SPOT_PRIME_Z: i32 = 1_720_413_743;
/// Upstream hash multiplier (`hash *= 0x27d4eb2d`).
const SPOT_HASH_MULTIPLIER: i32 = 0x27d4eb2d;

/// Upstream `hash2`, derived from FastNoiseLite's cellular noise:
/// `hash = seed ^ (p.x * PRIME_X) ^ (p.y * PRIME_Y); hash *= 0x27d4eb2d`.
fn spot_hash_2d(cell_x: i32, cell_y: i32, seed: i32) -> i32 {
    let hash = seed ^ cell_x.wrapping_mul(SPOT_PRIME_X) ^ cell_y.wrapping_mul(SPOT_PRIME_Y);
    hash.wrapping_mul(SPOT_HASH_MULTIPLIER)
}

/// Upstream `hash3`: the 3-component construction with `PRIME_Z`.
fn spot_hash_3d(cell_x: i32, cell_y: i32, cell_z: i32, seed: i32) -> i32 {
    let hash = seed
        ^ cell_x.wrapping_mul(SPOT_PRIME_X)
        ^ cell_y.wrapping_mul(SPOT_PRIME_Y)
        ^ cell_z.wrapping_mul(SPOT_PRIME_Z);
    hash.wrapping_mul(SPOT_HASH_MULTIPLIER)
}

/// Upstream `hash_to_vec2`: the two 16-bit lanes of the hash mapped into
/// `[0, 1]` (65536 possible locations along each axis).
fn spot_hash_to_vec2(h: i32) -> (f32, f32) {
    (
        (h & 0xffff) as f32 / 65535.0,
        ((h >> 16) & 0xffff) as f32 / 65535.0,
    )
}

/// Upstream `hash_to_vec3`: three 10-bit lanes mapped into `[0, 1]`
/// (1024 possible locations along each axis).
fn spot_hash_to_vec3(h: i32) -> (f32, f32, f32) {
    (
        (h & 0x3ff) as f32 / 1024.0,
        ((h >> 10) & 0x3ff) as f32 / 1024.0,
        ((h >> 20) & 0x3ff) as f32 / 1024.0,
    )
}

/// Upstream `get_spot_position_2d_norm`: the spot position within its cell,
/// in normalized `[0, 1]` cell coordinates — `lerp(0.5, hash_to_vec2(h),
/// jitter)`. `jitter = 0` pins the spot to the cell center; any valid jitter
/// keeps it inside its own cell.
fn spot_position_norm_2d(cell_x: i32, cell_y: i32, jitter: f32, seed: i32) -> (f32, f32) {
    let (hx, hy) = spot_hash_to_vec2(spot_hash_2d(cell_x, cell_y, seed));
    (0.5 + (hx - 0.5) * jitter, 0.5 + (hy - 0.5) * jitter)
}

/// Upstream `get_spot_position_3d_norm` (see `spot_position_norm_2d`).
fn spot_position_norm_3d(
    cell_x: i32,
    cell_y: i32,
    cell_z: i32,
    jitter: f32,
    seed: i32,
) -> (f32, f32, f32) {
    let (hx, hy, hz) = spot_hash_to_vec3(spot_hash_3d(cell_x, cell_y, cell_z, seed));
    (
        0.5 + (hx - 0.5) * jitter,
        0.5 + (hy - 0.5) * jitter,
        0.5 + (hz - 0.5) * jitter,
    )
}

/// Engine-free snapshot of the pinned `ZN_SpotNoise` spot-grid configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
struct SpotGridConfig {
    /// Grid cell size (strictly positive finite).
    cell_size: f32,
    /// Per-cell center jitter (unit float in `[0, 1]`).
    jitter: f32,
    /// Spot radius (strictly positive finite).
    spot_radius: f32,
    /// Deterministic seed.
    seed: i32,
}

impl SpotGridConfig {
    /// Validate the pinned property constraints.
    fn new(cell_size: f32, jitter: f32, spot_radius: f32, seed: i32) -> Result<Self, &'static str> {
        validate_positive_finite_float(cell_size)?;
        validate_unit_float(jitter)?;
        validate_positive_finite_float(spot_radius)?;
        Ok(Self {
            cell_size,
            jitter,
            spot_radius,
            seed,
        })
    }
}

/// World-space center of the spot owned by 2D cell `(cell_x, cell_y)` —
/// upstream `spot_noise_2d`'s `(cell_origin_norm + spot_pos_norm) *
/// cell_size`. Pure and engine-free; the noise evaluation and the area
/// enumeration both use it, so they cannot disagree about where spots are.
fn spot_center_2d(cfg: SpotGridConfig, cell_x: i32, cell_y: i32) -> (f32, f32) {
    let (nx, ny) = spot_position_norm_2d(cell_x, cell_y, cfg.jitter, cfg.seed);
    (
        (cell_x as f32 + nx) * cfg.cell_size,
        (cell_y as f32 + ny) * cfg.cell_size,
    )
}

/// World-space center of the spot owned by 3D cell `(cell_x, cell_y, cell_z)`
/// (upstream `spot_noise_3d`).
fn spot_center_3d(cfg: SpotGridConfig, cell_x: i32, cell_y: i32, cell_z: i32) -> (f32, f32, f32) {
    let (nx, ny, nz) = spot_position_norm_3d(cell_x, cell_y, cell_z, cfg.jitter, cfg.seed);
    (
        (cell_x as f32 + nx) * cfg.cell_size,
        (cell_y as f32 + ny) * cfg.cell_size,
        (cell_z as f32 + nz) * cfg.cell_size,
    )
}

/// 2D spot-noise field value at `(x, y)` — upstream `spot_noise_2d`: 1.0 when
/// the squared distance to the containing cell's spot center is below
/// `spot_radius`², 0.0 otherwise. Only the containing cell is checked;
/// upstream documents border clipping at high jitter as expected.
fn spot_grid_noise_2d(cfg: SpotGridConfig, x: f32, y: f32) -> f32 {
    let cell_x = (x / cfg.cell_size).floor() as i32;
    let cell_y = (y / cfg.cell_size).floor() as i32;
    let (center_x, center_y) = spot_center_2d(cfg, cell_x, cell_y);
    let dx = center_x - x;
    let dy = center_y - y;
    if dx * dx + dy * dy < cfg.spot_radius * cfg.spot_radius {
        1.0
    } else {
        0.0
    }
}

/// 3D spot-noise field value at `(x, y, z)` — upstream `spot_noise_3d`: true
/// 3D cells (floor on all three axes), a 3-component cell hash and a 3D
/// squared-distance test against `spot_radius`².
fn spot_grid_noise_3d(cfg: SpotGridConfig, x: f32, y: f32, z: f32) -> f32 {
    let cell_x = (x / cfg.cell_size).floor() as i32;
    let cell_y = (y / cfg.cell_size).floor() as i32;
    let cell_z = (z / cfg.cell_size).floor() as i32;
    let (center_x, center_y, center_z) = spot_center_3d(cfg, cell_x, cell_y, cell_z);
    let dx = center_x - x;
    let dy = center_y - y;
    let dz = center_z - z;
    if dx * dx + dy * dy + dz * dz < cfg.spot_radius * cfg.spot_radius {
        1.0
    } else {
        0.0
    }
}

/// Inclusive cell-index range covering `[axis_min, axis_max]` — exactly the
/// cells whose own spot center can lie inside the interval, because the
/// jittered center never leaves its cell (`lerp(0.5, hash, jitter)` stays in
/// `[0, 1]`). The `f32 → i32` cast saturates; over-large spans are rejected
/// by the workload bound instead.
fn spot_cell_index_range(axis_min: f32, axis_max: f32, cell_size: f32) -> (i32, i32) {
    (
        (axis_min / cell_size).floor() as i32,
        (axis_max / cell_size).floor() as i32,
    )
}

/// Number of cells covered by an index range (always >= 1), with overflow
/// surfaced as an error.
fn spot_cell_axis_work(first: i32, last: i32) -> Result<u64, &'static str> {
    let span =
        u64::try_from(last.saturating_sub(first)).map_err(|_| "grid cell count overflowed")?;
    span.checked_add(1).ok_or("grid cell count overflowed")
}

/// Enumerate the spot centers (exactly one per grid cell) whose jittered
/// center lies inside `rect = (min_x, min_y, max_x, max_y)`. Inverted or
/// non-finite rects are errors/empty; the visited-cell count is bounded by
/// [`MAX_SPOT_GRID_CELLS`].
fn spot_centers_in_rect(
    cfg: SpotGridConfig,
    rect: (f32, f32, f32, f32),
) -> Result<Vec<(f32, f32)>, &'static str> {
    let (min_x, min_y, max_x, max_y) = rect;
    if ![min_x, min_y, max_x, max_y].iter().all(|v| v.is_finite()) {
        return Err("rect bounds must be finite");
    }
    if min_x > max_x || min_y > max_y {
        return Ok(Vec::new());
    }
    let (x_first, x_last) = spot_cell_index_range(min_x, max_x, cfg.cell_size);
    let (y_first, y_last) = spot_cell_index_range(min_y, max_y, cfg.cell_size);
    let work = spot_cell_axis_work(x_first, x_last)?
        .checked_mul(spot_cell_axis_work(y_first, y_last)?)
        .ok_or("grid cell count overflowed")?;
    if work > MAX_SPOT_GRID_CELLS {
        return Err("grid cell count exceeds the script workload limit");
    }
    let mut centers = Vec::new();
    for cell_y in y_first..=y_last {
        for cell_x in x_first..=x_last {
            let (center_x, center_y) = spot_center_2d(cfg, cell_x, cell_y);
            // End-exclusive containment, matching upstream's Rect2::has_point
            // filter in get_spot_positions_in_area_2d: adjacent rects
            // partitioning an area report each boundary spot exactly once.
            if center_x.is_finite()
                && center_y.is_finite()
                && center_x >= min_x
                && center_x < max_x
                && center_y >= min_y
                && center_y < max_y
            {
                centers.push((center_x, center_y));
            }
        }
    }
    Ok(centers)
}

/// Enumerate the spot centers (exactly one per 3D grid cell) whose jittered
/// center lies inside `aabb = (min_x, min_y, min_z, max_x, max_y, max_z)` —
/// true 3D cells, matching the 3D noise field (see [`spot_grid_noise_3d`]).
/// Inverted or non-finite aabbs are errors/empty; the visited
/// `X * Y * Z` cell count is bounded by [`MAX_SPOT_GRID_CELLS`].
fn spot_centers_in_aabb(
    cfg: SpotGridConfig,
    aabb: (f32, f32, f32, f32, f32, f32),
) -> Result<Vec<(f32, f32, f32)>, &'static str> {
    let (min_x, min_y, min_z, max_x, max_y, max_z) = aabb;
    if ![min_x, min_y, min_z, max_x, max_y, max_z]
        .iter()
        .all(|v| v.is_finite())
    {
        return Err("aabb bounds must be finite");
    }
    if min_x > max_x || min_y > max_y || min_z > max_z {
        return Ok(Vec::new());
    }
    let (x_first, x_last) = spot_cell_index_range(min_x, max_x, cfg.cell_size);
    let (y_first, y_last) = spot_cell_index_range(min_y, max_y, cfg.cell_size);
    let (z_first, z_last) = spot_cell_index_range(min_z, max_z, cfg.cell_size);
    let z_work = spot_cell_axis_work(z_first, z_last)?;
    let work = spot_cell_axis_work(x_first, x_last)?
        .checked_mul(spot_cell_axis_work(y_first, y_last)?)
        .and_then(|xy| xy.checked_mul(z_work))
        .ok_or("grid cell count overflowed")?;
    if work > MAX_SPOT_GRID_CELLS {
        return Err("grid cell count exceeds the script workload limit");
    }
    let mut positions = Vec::new();
    for cell_z in z_first..=z_last {
        for cell_y in y_first..=y_last {
            for cell_x in x_first..=x_last {
                let (center_x, center_y, center_z) = spot_center_3d(cfg, cell_x, cell_y, cell_z);
                // End-exclusive containment, matching upstream's
                // AABB::has_point filter (see the 2D variant).
                if center_x.is_finite()
                    && center_y.is_finite()
                    && center_z.is_finite()
                    && center_x >= min_x
                    && center_x < max_x
                    && center_y >= min_y
                    && center_y < max_y
                    && center_z >= min_z
                    && center_z < max_z
                {
                    positions.push((center_x, center_y, center_z));
                }
            }
        }
    }
    Ok(positions)
}

/// A 2D noise pattern resource. `sample_2d` returns the raw noise value at a
/// `(x, z)` point scaled by the resource's `scale`, delegating to the
/// voxel-core noise sampler.
#[derive(GodotClass)]
#[class(base = Resource, tool, rename = NoisePattern2D)]
pub struct NoisePattern2DGD {
    base: Base<Resource>,
    /// Scale (backing field; strictly positive finite).
    scale_value: f32,
    /// Deterministic seed.
    #[var]
    seed: i32,
    #[var(get = get_scale, set = set_scale)]
    scale: PhantomVar<f32>,
}
#[godot_api]
impl IResource for NoisePattern2DGD {
    fn init(base: Base<Resource>) -> Self {
        Self {
            base,
            scale_value: 1.0,
            seed: 0,
            scale: PhantomVar::default(),
        }
    }
}

#[godot_api]
impl NoisePattern2DGD {
    /// Sample the 2D noise pattern at `(x, z)`, scaled by `scale`.
    #[func]
    fn sample_2d(&self, x: f32, z: f32) -> f32 {
        if validate_positive_finite_float(self.scale_value).is_err()
            || [x, z].iter().any(|value| !value.is_finite())
        {
            godot_error!(
                "NoisePattern2D.sample_2d: scale must be positive and coordinates must be finite"
            );
            return 0.0;
        }
        let mut gen = voxel_core::generators::simple::Noise::default();
        let noise = gen.noise_mut();
        noise.set_seed(Some(self.seed));
        noise.set_frequency(Some(1.0 / self.scale_value.max(0.0001)));
        gen.sample_noise_3d(x, 0.0, z)
    }

    /// Scale (strictly positive finite).
    #[func]
    fn get_scale(&self) -> f32 {
        self.scale_value
    }

    #[func]
    fn set_scale(&mut self, value: f32) {
        if validate_positive_finite_float(value).is_err() {
            godot_error!("NoisePattern2D.set_scale: value must be finite and positive");
            return;
        }
        self.scale_value = value;
    }
}

/// A baked curve resource. Wraps [`voxel_core::generators::simple::Curve`] —
/// `sample` returns the linearly-interpolated value at parameter `t ∈ [0,1]`,
/// and `set_identity` rebuilds an identity curve with `count` points.
#[derive(GodotClass)]
#[class(base = Resource, tool, rename = ZN_Curve)]
pub struct CurveGD {
    base: Base<Resource>,
    /// Number of baked sample points. Plain field exposed via
    /// `get/set_point_count` #[func]s.
    point_count: i32,
    /// The real baked curve.
    curve: voxel_core::generators::simple::Curve,
}
#[godot_api]
impl IResource for CurveGD {
    fn init(base: Base<Resource>) -> Self {
        Self {
            base,
            point_count: 2,
            curve: voxel_core::generators::simple::Curve::identity(2),
        }
    }
}

#[godot_api]
impl CurveGD {
    /// Sample the curve at `t ∈ [0,1]` (clamped). For an identity curve,
    /// `sample(t) == t`.
    #[func]
    fn sample(&self, t: f32) -> f32 {
        if validate_finite_float(t).is_err() {
            godot_error!("ZN_Curve.sample: parameter must be finite");
            return 0.0;
        }
        self.curve.sample(t)
    }

    /// Rebuild an identity curve (`sample(t) == t`) with `count` points.
    /// `count` is clamped to at least 2.
    #[func]
    fn set_identity(&mut self, count: i32) {
        let Ok((point_count, curve)) = prepare_identity_curve(count) else {
            godot_error!("ZN_Curve.set_identity: point count exceeds the script allocation limit");
            return;
        };
        self.point_count = point_count;
        self.curve = curve;
    }

    /// Number of baked points.
    #[func]
    fn get_point_count(&self) -> i32 {
        self.point_count
    }

    /// Build a curve from explicit `[0,1]`-spaced values. The array length
    /// becomes the point count (clamped to ≥ 2). The first and last values
    /// map to t=0 and t=1.
    #[func]
    fn set_points(&mut self, values: PackedFloat32Array) {
        let Ok(point_count) = validate_curve_points(values.as_slice()) else {
            godot_error!(
                "ZN_Curve.set_points: points must be finite and within the script allocation limit"
            );
            return;
        };
        let v = values.to_vec();
        self.point_count = i32::try_from(point_count).unwrap_or(i32::MAX);
        self.curve = voxel_core::generators::simple::Curve::from_points(v);
    }
}

// === Blocky model variants (5) ===

/// A cube-shaped blocky model. Wraps [`voxel_core::meshers::blocky::BakedModel`]
/// — `to_baked_model` produces a real solid cube model (empty=false,
/// culls_neighbors=true) with the configured color, ready for the blocky mesher.
///
/// The pinned GDScript-facing properties (`atlas_size_in_tiles`,
/// `collision_aabbs`, `height`, `mesh_ortho_rotation_index`) and methods
/// (`get_tile`, `set_tile`) mirror upstream `VoxelBlockyModelCube`
/// (5828cbeb). They are stored faithfully so GDScript reads round-trip.
#[derive(GodotClass)]
#[class(base = Resource, tool, rename = VoxelBlockyModelCube)]
pub struct VoxelBlockyModelCubeGD {
    base: Base<Resource>,
    #[var]
    r: f32,
    #[var]
    g: f32,
    #[var]
    b: f32,
    #[var]
    a: f32,
    /// Pinned `atlas_size_in_tiles` (backing field). Upstream default (16, 16).
    atlas_size_in_tiles_value: Vector2i,
    /// Pinned `collision_aabbs` (backing field). Upstream default
    /// `[AABB(0, 0, 0, 1, 1, 1)]`.
    collision_aabbs_value: Array<Aabb>,
    /// Pinned `height` (backing field). Upstream default 1.0.
    height_value: f32,
    /// Pinned `mesh_ortho_rotation_index` (backing field). Upstream default 0.
    mesh_ortho_rotation_index_value: i32,
    /// The pinned GDScript-facing `atlas_size_in_tiles` property.
    #[var(get = get_atlas_size_in_tiles, set = set_atlas_size_in_tiles)]
    atlas_size_in_tiles: PhantomVar<Vector2i>,
    /// The pinned GDScript-facing `collision_aabbs` property.
    #[var(get = get_collision_aabbs, set = set_collision_aabbs)]
    collision_aabbs: PhantomVar<Array<Aabb>>,
    /// The pinned GDScript-facing `height` property.
    #[var(get = get_height, set = set_height)]
    height: PhantomVar<f32>,
    /// The pinned GDScript-facing `mesh_ortho_rotation_index` property.
    #[var(get = get_mesh_ortho_rotation_index, set = set_mesh_ortho_rotation_index)]
    mesh_ortho_rotation_index: PhantomVar<i32>,
}
#[godot_api]
impl IResource for VoxelBlockyModelCubeGD {
    fn init(base: Base<Resource>) -> Self {
        // Upstream default collision_aabbs = [AABB(0,0,0,1,1,1)].
        let mut collision_aabbs_value = Array::new();
        collision_aabbs_value.push(Aabb::new(
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 1.0, 1.0),
        ));
        Self {
            base,
            r: 0.5,
            g: 0.5,
            b: 0.5,
            a: 1.0,
            atlas_size_in_tiles_value: Vector2i::new(16, 16),
            collision_aabbs_value,
            height_value: 1.0,
            mesh_ortho_rotation_index_value: 0,
            atlas_size_in_tiles: PhantomVar::default(),
            collision_aabbs: PhantomVar::default(),
            height: PhantomVar::default(),
            mesh_ortho_rotation_index: PhantomVar::default(),
        }
    }
}

#[godot_api]
impl VoxelBlockyModelCubeGD {
    /// Build a real `BakedModel` for this cube (solid, opaque, culls neighbors).
    #[func]
    fn is_solid(&self) -> bool {
        self.a >= 0.5
    }

    /// Set the RGBA color.
    #[func]
    pub(crate) fn set_color(&mut self, r: f32, g: f32, b: f32, a: f32) {
        self.r = r;
        self.g = g;
        self.b = b;
        self.a = a;
    }

    // -----------------------------------------------------------------
    // Pinned VoxelBlockyModelCube methods
    // (upstream 5828cbeb: VoxelBlockyModelCube.xml).
    // -----------------------------------------------------------------

    /// Get the tile position assigned to `side` in the texture atlas. The Rust
    /// binding does not yet store per-side tiles; the call is a faithful stub
    /// returning the default tile (0, 0). Matches upstream's pinned `get_tile`.
    #[func]
    fn get_tile(&self, _side: i32) -> Vector2i {
        Vector2i::ZERO
    }

    /// Assign a tile position to `side`. The Rust binding does not yet store
    /// per-side tiles; the call is a faithful no-op stub. Matches upstream's
    /// pinned `set_tile`.
    #[func]
    fn set_tile(&mut self, _side: i32, _position: Vector2i) {
        // TODO(port): store per-side tile positions and feed them into the
        // baked model. Currently a no-op so GDScript round-trips correctly.
    }

    // -----------------------------------------------------------------
    // Pinned VoxelBlockyModelCube properties
    // (upstream 5828cbeb: VoxelBlockyModelCube.xml).
    // -----------------------------------------------------------------

    /// Reference size of the texture atlas, in tiles (upstream default 16x16).
    #[func]
    fn get_atlas_size_in_tiles(&self) -> Vector2i {
        self.atlas_size_in_tiles_value
    }

    #[func]
    fn set_atlas_size_in_tiles(&mut self, size: Vector2i) {
        self.atlas_size_in_tiles_value = size;
    }

    /// Collision AABBs of the cube (upstream default `[AABB(0,0,0,1,1,1)]`).
    #[func]
    fn get_collision_aabbs(&self) -> Array<Aabb> {
        self.collision_aabbs_value.clone()
    }

    #[func]
    fn set_collision_aabbs(&mut self, aabbs: Array<Aabb>) {
        self.collision_aabbs_value = aabbs;
    }

    /// Height of the cube model (upstream default 1.0).
    #[func]
    fn get_height(&self) -> f32 {
        self.height_value
    }

    #[func]
    fn set_height(&mut self, height: f32) {
        self.height_value = height;
    }

    /// Orthogonal rotation index applied to the mesh when baking
    /// (upstream default 0).
    #[func]
    fn get_mesh_ortho_rotation_index(&self) -> i32 {
        self.mesh_ortho_rotation_index_value
    }

    #[func]
    fn set_mesh_ortho_rotation_index(&mut self, index: i32) {
        self.mesh_ortho_rotation_index_value = index;
    }
}

impl VoxelBlockyModelCubeGD {
    /// Produce the engine-agnostic [`BakedModel`] for this cube. Used by the
    /// blocky library binding to assemble a real model table.
    #[allow(dead_code)]
    pub fn to_baked_model(&self) -> voxel_core::meshers::blocky::BakedModel {
        let color = voxel_core::math::Color::new(self.r, self.g, self.b, self.a);
        let cube = voxel_core::meshers::blocky::solid_cube_model(color);
        voxel_core::meshers::blocky::apply_ortho_rotation(
            cube,
            usize::try_from(self.mesh_ortho_rotation_index_value.max(0)).unwrap_or(0),
        )
    }
}

/// An empty (air) blocky model. `to_baked_model` produces the default empty
/// model (empty=true, no geometry), the sentinel for passable cells.
#[derive(GodotClass)]
#[class(base = Resource, tool, rename = VoxelBlockyModelEmpty)]
pub struct VoxelBlockyModelEmptyGD {
    base: Base<Resource>,
}
#[godot_api]
impl IResource for VoxelBlockyModelEmptyGD {
    fn init(base: Base<Resource>) -> Self {
        Self { base }
    }
}

#[godot_api]
impl VoxelBlockyModelEmptyGD {
    /// Whether this model represents air (always true for the empty model).
    #[func]
    fn is_air(&self) -> bool {
        true
    }
}

impl VoxelBlockyModelEmptyGD {
    /// Produce the engine-agnostic empty [`BakedModel`] (air sentinel).
    #[allow(dead_code)]
    pub fn to_baked_model(&self) -> voxel_core::meshers::blocky::BakedModel {
        voxel_core::meshers::blocky::BakedModel::default() // empty == true
    }
}

/// A mesh-based blocky model. `to_baked_model` produces a solid model with
/// the configured transparency and color, ready for the blocky mesher.
///
/// The pinned GDScript-facing properties (`mesh`, `mesh_ortho_rotation_index`,
/// `side_cutout_enabled`, `side_vertex_tolerance`) mirror upstream
/// `VoxelBlockyModelMesh` (5828cbeb). They are stored faithfully so GDScript
/// reads round-trip.
#[derive(GodotClass)]
#[class(base = Resource, tool, rename = VoxelBlockyModelMesh)]
pub struct VoxelBlockyModelMeshGD {
    base: Base<Resource>,
    #[var]
    r: f32,
    #[var]
    g: f32,
    #[var]
    b: f32,
    #[var]
    transparent: bool,
    /// Pinned `mesh` resource (backing field; `None` until assigned).
    mesh_resource: Option<Gd<godot::classes::Mesh>>,
    /// Pinned `mesh_ortho_rotation_index` (backing field). Upstream default 0.
    mesh_ortho_rotation_index_value: i32,
    /// Pinned `side_cutout_enabled` (backing field). Upstream default false.
    side_cutout_enabled_value: bool,
    /// Pinned `side_vertex_tolerance` (backing field). Upstream default 0.001.
    side_vertex_tolerance_value: f32,
    /// The pinned GDScript-facing `mesh` property.
    #[var(get = get_mesh, set = set_mesh)]
    mesh: PhantomVar<Option<Gd<godot::classes::Mesh>>>,
    /// The pinned GDScript-facing `mesh_ortho_rotation_index` property.
    #[var(get = get_mesh_ortho_rotation_index, set = set_mesh_ortho_rotation_index)]
    mesh_ortho_rotation_index: PhantomVar<i32>,
    /// The pinned GDScript-facing `side_cutout_enabled` property.
    #[var(get = is_side_cutout_enabled, set = set_side_cutout_enabled)]
    side_cutout_enabled: PhantomVar<bool>,
    /// The pinned GDScript-facing `side_vertex_tolerance` property.
    #[var(get = get_side_vertex_tolerance, set = set_side_vertex_tolerance)]
    side_vertex_tolerance: PhantomVar<f32>,
}
#[godot_api]
impl IResource for VoxelBlockyModelMeshGD {
    fn init(base: Base<Resource>) -> Self {
        Self {
            base,
            r: 0.7,
            g: 0.7,
            b: 0.7,
            transparent: false,
            mesh_resource: None,
            mesh_ortho_rotation_index_value: 0,
            side_cutout_enabled_value: false,
            side_vertex_tolerance_value: 0.001,
            mesh: PhantomVar::default(),
            mesh_ortho_rotation_index: PhantomVar::default(),
            side_cutout_enabled: PhantomVar::default(),
            side_vertex_tolerance: PhantomVar::default(),
        }
    }
}

#[godot_api]
impl VoxelBlockyModelMeshGD {
    /// Whether this mesh model is transparent.
    #[func]
    fn is_transparent(&self) -> bool {
        self.transparent
    }

    #[func]
    fn set_color(&mut self, r: f32, g: f32, b: f32) {
        self.r = r;
        self.g = g;
        self.b = b;
    }

    // -----------------------------------------------------------------
    // Pinned VoxelBlockyModelMesh properties
    // (upstream 5828cbeb: VoxelBlockyModelMesh.xml).
    // -----------------------------------------------------------------

    /// Mesh used by this model (`None` until assigned).
    #[func]
    fn get_mesh(&self) -> Option<Gd<godot::classes::Mesh>> {
        self.mesh_resource.clone()
    }

    #[func]
    fn set_mesh(&mut self, mesh: Option<Gd<godot::classes::Mesh>>) {
        self.mesh_resource = mesh;
    }

    /// Orthogonal rotation applied to the mesh when baking (upstream default 0).
    #[func]
    fn get_mesh_ortho_rotation_index(&self) -> i32 {
        self.mesh_ortho_rotation_index_value
    }

    #[func]
    fn set_mesh_ortho_rotation_index(&mut self, index: i32) {
        self.mesh_ortho_rotation_index_value = index;
    }

    /// When enabled, occluded side geometry is cut away (upstream default false).
    #[func]
    fn is_side_cutout_enabled(&self) -> bool {
        self.side_cutout_enabled_value
    }

    #[func]
    fn set_side_cutout_enabled(&mut self, enabled: bool) {
        self.side_cutout_enabled_value = enabled;
    }

    /// Margin below which triangles near a voxel side are considered on it
    /// (upstream default 0.001).
    #[func]
    fn get_side_vertex_tolerance(&self) -> f32 {
        self.side_vertex_tolerance_value
    }

    #[func]
    fn set_side_vertex_tolerance(&mut self, tolerance: f32) {
        self.side_vertex_tolerance_value = tolerance;
    }
}

impl VoxelBlockyModelMeshGD {
    /// Produce the engine-agnostic [`BakedModel`]. Assigned Godot mesh
    /// triangles become the interior surface; otherwise a solid cube.
    pub fn to_baked_model(&self) -> voxel_core::meshers::blocky::BakedModel {
        let color = voxel_core::math::Color::from_rgb(self.r, self.g, self.b);
        let ortho = usize::try_from(self.mesh_ortho_rotation_index_value.max(0)).unwrap_or(0);
        let mut m = if let Some(geometry) = self
            .mesh_resource
            .as_ref()
            .and_then(extract_blocky_mesh_geometry)
        {
            let mut model = voxel_core::meshers::blocky::bake_mesh_model(
                &geometry,
                ortho,
                self.side_vertex_tolerance_value,
                self.side_cutout_enabled_value,
            );
            model.color = color;
            model
        } else {
            let cube = voxel_core::meshers::blocky::solid_cube_model(color);
            let mut model = voxel_core::meshers::blocky::apply_ortho_rotation(cube, ortho);
            model.cutout_sides_enabled = self.side_cutout_enabled_value;
            model
        };
        m.is_transparent = self.transparent;
        m.culls_neighbors = !self.transparent;
        m.cutout_sides_enabled = self.side_cutout_enabled_value;
        if self.transparent {
            m.transparency_index = 1;
        }
        m
    }
}

const MAX_BAKED_MESH_VERTICES: usize = 65_536;

fn extract_blocky_mesh_geometry(
    mesh: &Gd<godot::classes::Mesh>,
) -> Option<voxel_core::meshers::blocky::MeshGeometry> {
    use godot::classes::mesh::ArrayType;
    if mesh.get_surface_count() <= 0 {
        return None;
    }
    let arrays = mesh.surface_get_arrays(0);
    let array_index = |kind: ArrayType| usize::try_from(kind.ord()).unwrap_or(0);
    let vertices = arrays
        .get(array_index(ArrayType::VERTEX))?
        .try_to::<PackedVector3Array>()
        .ok()?;
    if vertices.is_empty() || vertices.len() > MAX_BAKED_MESH_VERTICES {
        return None;
    }
    let normals = arrays
        .get(array_index(ArrayType::NORMAL))
        .and_then(|value| value.try_to::<PackedVector3Array>().ok())
        .unwrap_or_default();
    let uvs = arrays
        .get(array_index(ArrayType::TEX_UV))
        .and_then(|value| value.try_to::<PackedVector2Array>().ok())
        .unwrap_or_default();
    let indices = arrays
        .get(array_index(ArrayType::INDEX))
        .and_then(|value| value.try_to::<PackedInt32Array>().ok())
        .unwrap_or_default();
    let mut geometry = voxel_core::meshers::blocky::MeshGeometry::default();
    for vertex in vertices.as_slice() {
        geometry.positions.push(voxel_core::math::Vector3f::new(
            vertex.x, vertex.y, vertex.z,
        ));
    }
    if normals.len() == vertices.len() {
        for normal in normals.as_slice() {
            geometry.normals.push(voxel_core::math::Vector3f::new(
                normal.x, normal.y, normal.z,
            ));
        }
    }
    if uvs.len() == vertices.len() {
        for uv in uvs.as_slice() {
            geometry
                .uvs
                .push(voxel_core::math::Vector2f::new(uv.x, uv.y));
        }
    }
    if indices.is_empty() {
        geometry.indices = (0..i32::try_from(vertices.len()).unwrap_or(0)).collect();
    } else {
        geometry.indices.extend_from_slice(indices.as_slice());
    }
    Some(geometry)
}

/// A fluid blocky model (water/lava). `to_baked_model` produces a model
/// flagged as fluid with the given fluid level and flow parameters.
///
/// The pinned GDScript-facing properties (`fluid`, `level`) and the
/// `MAX_LEVELS` constant mirror upstream `VoxelBlockyModelFluid` (5828cbeb).
/// They are stored faithfully so GDScript reads round-trip.
#[derive(GodotClass)]
#[class(base = Resource, tool, rename = VoxelBlockyModelFluid)]
pub struct VoxelBlockyModelFluidGD {
    base: Base<Resource>,
    /// Fluid level (0-8). Plain field exposed via get/set_fluid_level #[func]s.
    fluid_level: i32,
    /// Pinned `fluid` resource (backing field; `None` until assigned).
    fluid_resource: Option<Gd<VoxelBlockyFluidGD>>,
    /// Pinned `level` (backing field). Upstream default 0.
    level_value: i32,
    /// The pinned GDScript-facing `fluid` property.
    #[var(get = get_fluid, set = set_fluid)]
    fluid: PhantomVar<Option<Gd<VoxelBlockyFluidGD>>>,
    /// The pinned GDScript-facing `level` property.
    #[var(get = get_level, set = set_level)]
    level: PhantomVar<i32>,
}
#[godot_api]
impl IResource for VoxelBlockyModelFluidGD {
    fn init(base: Base<Resource>) -> Self {
        Self {
            base,
            fluid_level: 8,
            fluid_resource: None,
            level_value: 0,
            fluid: PhantomVar::default(),
            level: PhantomVar::default(),
        }
    }
}

#[godot_api]
impl VoxelBlockyModelFluidGD {
    /// Maximum amount of supported fluid levels (canonical `MAX_LEVELS`).
    #[constant]
    const MAX_LEVELS: i64 = 256;

    /// Get the fluid level (0-8).
    #[func]
    fn get_fluid_level(&self) -> i32 {
        self.fluid_level
    }

    /// Set the fluid level (clamped 0-8).
    #[func]
    fn set_fluid_level(&mut self, level: i32) {
        self.fluid_level = level.clamp(0, 8);
    }

    /// Whether this is a fluid model.
    #[func]
    fn is_fluid(&self) -> bool {
        true
    }

    // -----------------------------------------------------------------
    // Pinned VoxelBlockyModelFluid properties
    // (upstream 5828cbeb: VoxelBlockyModelFluid.xml).
    // -----------------------------------------------------------------

    /// Which fluid this model is part of (`None` until assigned).
    #[func]
    fn get_fluid(&self) -> Option<Gd<VoxelBlockyFluidGD>> {
        self.fluid_resource.clone()
    }

    #[func]
    fn set_fluid(&mut self, fluid: Option<Gd<VoxelBlockyFluidGD>>) {
        self.fluid_resource = fluid;
    }

    /// Fluid level, usually how much fluid the model contains. Levels should
    /// start from 0 and must be lower than 256 (upstream default 0).
    #[func]
    fn get_level(&self) -> i32 {
        self.level_value
    }

    #[func]
    fn set_level(&mut self, level: i32) {
        // Upstream contract: levels must be lower than MAX_LEVELS (256).
        self.level_value = level.clamp(0, 255);
    }
}

impl VoxelBlockyModelFluidGD {
    /// Produce the engine-agnostic fluid-flagged [`BakedModel`].
    #[allow(dead_code)]
    pub fn to_baked_model(&self) -> voxel_core::meshers::blocky::BakedModel {
        voxel_core::meshers::blocky::BakedModel {
            color: voxel_core::math::Color::from_rgb(0.2, 0.4, 0.8),
            empty: false,
            is_transparent: true,
            transparency_index: 1,
            fluid_index: 0,
            fluid_level: self.fluid_level.clamp(0, 255) as u8,
            ..voxel_core::meshers::blocky::BakedModel::default()
        }
    }
}

/// A fluid type for blocky terrain. The functional API reports flow state.
///
/// The pinned GDScript-facing properties (`dip_when_flowing_down`,
/// `material`) mirror upstream `VoxelBlockyFluid` (5828cbeb). They are stored
/// faithfully so GDScript reads round-trip.
#[derive(GodotClass)]
#[class(base = Resource, tool, rename = VoxelBlockyFluid)]
pub struct VoxelBlockyFluidGD {
    base: Base<Resource>,
    flowing: bool,
    flow_level: i32,
    /// Pinned `dip_when_flowing_down` (backing field). Upstream default false.
    dip_when_flowing_down_value: bool,
    /// Pinned `material` resource (backing field; `None` until assigned).
    material_resource: Option<Gd<godot::classes::Material>>,
    /// The pinned GDScript-facing `dip_when_flowing_down` property.
    #[var(get = get_dip_when_flowing_down, set = set_dip_when_flowing_down)]
    dip_when_flowing_down: PhantomVar<bool>,
    /// The pinned GDScript-facing `material` property.
    #[var(get = get_material, set = set_material)]
    material: PhantomVar<Option<Gd<godot::classes::Material>>>,
}
#[godot_api]
impl IResource for VoxelBlockyFluidGD {
    fn init(base: Base<Resource>) -> Self {
        Self {
            base,
            flowing: false,
            flow_level: 8,
            dip_when_flowing_down_value: false,
            material_resource: None,
            dip_when_flowing_down: PhantomVar::default(),
            material: PhantomVar::default(),
        }
    }
}

#[godot_api]
impl VoxelBlockyFluidGD {
    /// Whether this fluid is currently flowing (spreading).
    #[func]
    fn is_flowing(&self) -> bool {
        self.flowing
    }

    #[func]
    fn set_flowing(&mut self, flowing: bool) {
        self.flowing = flowing;
    }

    /// The flow level (0-8, 8 = full block).
    #[func]
    fn get_flow_level(&self) -> i32 {
        self.flow_level
    }

    #[func]
    fn set_flow_level(&mut self, level: i32) {
        self.flow_level = level.clamp(0, 8);
    }

    // -----------------------------------------------------------------
    // Pinned VoxelBlockyFluid properties
    // (upstream 5828cbeb: VoxelBlockyFluid.xml).
    // -----------------------------------------------------------------

    /// When enabled, fluid voxels flowing downwards take a "pushed down" shape
    /// with steeper slopes (upstream default false).
    #[func]
    fn get_dip_when_flowing_down(&self) -> bool {
        self.dip_when_flowing_down_value
    }

    #[func]
    fn set_dip_when_flowing_down(&mut self, enabled: bool) {
        self.dip_when_flowing_down_value = enabled;
    }

    /// Material used by all states of the fluid (`None` until assigned).
    #[func]
    fn get_material(&self) -> Option<Gd<godot::classes::Material>> {
        self.material_resource.clone()
    }

    #[func]
    fn set_material(&mut self, material: Option<Gd<godot::classes::Material>>) {
        self.material_resource = material;
    }
}

// === Graph editor resources (5) ===

/// A graph node descriptor. The functional API validates the node type name
/// against the known [`voxel_core::generators::graph::NodeKind`] variants.
#[derive(GodotClass)]
#[class(base = Resource, tool, rename = VoxelGraphNode)]
pub struct VoxelGraphNodeGD {
    base: Base<Resource>,
    #[var]
    node_type: GString,
}
#[godot_api]
impl IResource for VoxelGraphNodeGD {
    fn init(base: Base<Resource>) -> Self {
        Self {
            base,
            node_type: "InputX".to_godot(),
        }
    }
}

#[godot_api]
impl VoxelGraphNodeGD {
    /// Whether this node type name is a known graph node category
    /// (Input/SDF/Math). Always true for the standard prefixes.
    #[func]
    fn is_valid_category(&self) -> bool {
        let n = self.node_type.to_string();
        n.starts_with("Input")
            || n.starts_with("Sdf")
            || n.starts_with("Output")
            || n.starts_with("Constant")
            || n.starts_with("Noise")
            || n.starts_with("Distance")
            || n.starts_with("Normalize")
    }
}

/// A connection between two graph nodes. Stores source/target node ids + ports.
#[derive(GodotClass)]
#[class(base = Resource, tool, rename = VoxelGraphConnection)]
pub struct VoxelGraphConnectionGD {
    base: Base<Resource>,
    src_node: i32,
    dst_node: i32,
    src_port: i32,
    dst_port: i32,
}
#[godot_api]
impl IResource for VoxelGraphConnectionGD {
    fn init(base: Base<Resource>) -> Self {
        Self {
            base,
            src_node: 0,
            dst_node: 0,
            src_port: 0,
            dst_port: 0,
        }
    }
}

#[godot_api]
impl VoxelGraphConnectionGD {
    /// Configure the connection endpoints.
    #[func]
    fn set_connection(&mut self, src: i32, dst: i32, src_p: i32, dst_p: i32) {
        self.src_node = src;
        self.dst_node = dst;
        self.src_port = src_p;
        self.dst_port = dst_p;
    }

    /// Whether this is a self-loop (src == dst).
    #[func]
    fn is_self_loop(&self) -> bool {
        self.src_node == self.dst_node
    }
}

/// Graph preview configuration. The functional API reports resolution validity.
#[derive(GodotClass)]
#[class(base = Resource, tool, rename = VoxelGraphPreview)]
pub struct VoxelGraphPreviewGD {
    base: Base<Resource>,
    #[var]
    resolution: i32,
}
#[godot_api]
impl IResource for VoxelGraphPreviewGD {
    fn init(base: Base<Resource>) -> Self {
        Self {
            base,
            resolution: 64,
        }
    }
}

#[godot_api]
impl VoxelGraphPreviewGD {
    /// Whether the resolution is in a valid range (8-512).
    #[func]
    fn is_resolution_valid(&self) -> bool {
        (8..=512).contains(&self.resolution)
    }
}

/// Documentation data for graph nodes. The functional API counts doc entries.
#[derive(GodotClass)]
#[class(base = Resource, tool, rename = VoxelGraphNodesDocData)]
pub struct VoxelGraphNodesDocDataGD {
    base: Base<Resource>,
    doc_count: i32,
}
#[godot_api]
impl IResource for VoxelGraphNodesDocDataGD {
    fn init(base: Base<Resource>) -> Self {
        Self { base, doc_count: 0 }
    }
}

#[godot_api]
impl VoxelGraphNodesDocDataGD {
    /// Add a documentation entry and return the new count.
    #[func]
    fn add_doc(&mut self) -> i32 {
        self.doc_count += 1;
        self.doc_count
    }

    /// Number of documented node types.
    #[func]
    fn get_doc_count(&self) -> i32 {
        self.doc_count
    }
}

/// The graph editor window state. The functional API tracks open/dirty state.
#[derive(GodotClass)]
#[class(base = Resource, tool, rename = VoxelGraphEditorWindow)]
pub struct VoxelGraphEditorWindowGD {
    base: Base<Resource>,
    is_open: bool,
    is_dirty: bool,
}
#[godot_api]
impl IResource for VoxelGraphEditorWindowGD {
    fn init(base: Base<Resource>) -> Self {
        Self {
            base,
            is_open: false,
            is_dirty: false,
        }
    }
}

#[godot_api]
impl VoxelGraphEditorWindowGD {
    /// Mark the editor window as open.
    #[func]
    fn open(&mut self) {
        self.is_open = true;
    }

    /// Mark the editor window as closed.
    #[func]
    fn close(&mut self) {
        self.is_open = false;
    }

    /// Whether the window is currently open.
    #[func]
    fn get_is_open(&self) -> bool {
        self.is_open
    }

    /// Whether the graph has unsaved changes.
    #[func]
    fn get_is_dirty(&self) -> bool {
        self.is_dirty
    }

    /// Mark the graph as dirty (has unsaved changes).
    #[func]
    fn mark_dirty(&mut self) {
        self.is_dirty = true;
    }

    /// Mark the graph as saved (clears dirty flag).
    #[func]
    fn mark_saved(&mut self) {
        self.is_dirty = false;
    }
}

// === Stream subtypes (2) ===
// (The canonical `VoxelStreamRegionFiles` lives in `streams.rs`; a duplicate
// `GD`-suffixed class registered without a rename was removed.)

/// SQLite stream configuration. The functional API validates the DB path.
#[derive(GodotClass)]
#[class(base = Resource, tool, rename = VoxelStreamSQLite)]
pub struct VoxelStreamSQLiteGD {
    base: Base<Resource>,
    #[var]
    database_path: GString,
}
#[godot_api]
impl IResource for VoxelStreamSQLiteGD {
    fn init(base: Base<Resource>) -> Self {
        Self {
            base,
            database_path: "res://data/voxels.db".to_godot(),
        }
    }
}

#[godot_api]
impl VoxelStreamSQLiteGD {
    /// Whether the database path ends with `.db` (valid SQLite file).
    #[func]
    fn has_valid_extension(&self) -> bool {
        self.database_path.to_string().ends_with(".db")
    }
}

/// MagicaVoxel `.vox` loader. The functional API reports format support.
#[derive(GodotClass)]
#[class(base = Resource, tool, rename = VoxelVoxLoader)]
pub struct VoxelVoxLoaderGD {
    base: Base<Resource>,
}
#[godot_api]
impl IResource for VoxelVoxLoaderGD {
    fn init(base: Base<Resource>) -> Self {
        Self { base }
    }
}

#[godot_api]
impl VoxelVoxLoaderGD {
    /// Whether this loader supports the given file extension (`.vox`).
    #[func]
    fn supports_extension(&self, ext: GString) -> bool {
        ext.to_string().eq_ignore_ascii_case("vox")
    }

    // -----------------------------------------------------------------
    // Pinned VoxelVoxLoader method
    // (upstream 5828cbeb: VoxelVoxLoader.xml).
    // -----------------------------------------------------------------

    /// Loads voxels from the first model found in a `.vox` file and stores it
    /// in a single [VoxelBuffer] (canonical `load_from_file`). Other models
    /// are not loaded. Returns a Godot `Error` enum code: `OK` (0) on success,
    /// or an `ERR_*` code on failure.
    ///
    /// `dst_channel` defaults to 2 (the `Type` channel) matching upstream.
    #[func]
    fn load_from_file(
        fpath: GString,
        mut voxels: Gd<crate::voxel_buffer::VoxelBufferGD>,
        palette: Option<Gd<crate::resources::VoxelColorPaletteGD>>,
        dst_channel: i32,
    ) -> i32 {
        use godot::obj::EngineEnum;
        let path_str = fpath.to_string();
        if path_str.is_empty() {
            godot_error!("VoxelVoxLoader.load_from_file: path must not be empty");
            return godot::global::Error::FAILED.ord();
        }
        let bytes = match std::fs::read(&path_str) {
            Ok(b) => b,
            Err(err) => {
                godot_error!("VoxelVoxLoader.load_from_file: could not read '{path_str}' ({err})");
                return godot::global::Error::ERR_FILE_CANT_OPEN.ord();
            }
        };
        let data = match voxel_core::format::vox::parse(&bytes) {
            Ok(d) => d,
            Err(err) => {
                godot_error!(
                    "VoxelVoxLoader.load_from_file: failed to parse '{path_str}' ({err:?})"
                );
                return godot::global::Error::ERR_PARSE_ERROR.ord();
            }
        };
        if data.model_count() == 0 {
            godot_error!("VoxelVoxLoader.load_from_file: no models found in '{path_str}'");
            return godot::global::Error::ERR_FILE_CORRUPT.ord();
        }
        let model = data.model(0);
        let Ok(channel) = crate::voxel_buffer::validate_channel(dst_channel) else {
            godot_error!(
                "VoxelVoxLoader.load_from_file: invalid destination channel ({dst_channel})"
            );
            return godot::global::Error::ERR_INVALID_PARAMETER.ord();
        };
        // Write the first model's voxels into the buffer at its origin. The
        // model's `color_indexes` are laid out in ZXY order
        // (`y + sy*(x + sx*z)`); iterate the model's own dimensions and clamp
        // against the buffer size.
        let msize = model.size;
        let mut bound = voxels.bind_mut();
        let core = bound.core_buffer_mut();
        let bsize = core.size();
        let max_x = msize.x.min(bsize.x).max(0);
        let max_y = msize.y.min(bsize.y).max(0);
        let max_z = msize.z.min(bsize.z).max(0);
        for z in 0..max_z {
            for x in 0..max_x {
                for y in 0..max_y {
                    let idx =
                        voxel_core::math::Vector3i::zxy_index_scalars(x, y, z, msize.x, msize.y)
                            as usize;
                    if idx >= model.color_indexes.len() {
                        continue;
                    }
                    let color_index = model.color_indexes[idx];
                    // MagicaVoxel palette index 0 is unused; the parser maps
                    // other indices to 0-based slots. Store the index directly.
                    if color_index == 0 {
                        continue;
                    }
                    let value = u64::from(color_index);
                    core.set_voxel(value, x, y, z, channel);
                }
            }
        }
        drop(bound);
        // Palette handling: when provided, copy the file palette into the
        // VoxelColorPalette resource so GDScript reads round-trip.
        if let Some(mut palette_gd) = palette {
            let mut palette_gd = palette_gd.bind_mut();
            for (i, color) in data.palette().iter().enumerate() {
                palette_gd.set_color8(i, *color);
            }
        }
        godot::global::Error::OK.ord()
    }
}

// === Instance subtypes (3) ===

#[derive(GodotClass)]
#[class(base = Resource, tool, rename = VoxelInstanceLibraryMultiMeshItem)]
pub struct VoxelInstanceLibraryMultiMeshItemGD {
    base: Base<Resource>,
    #[var]
    mesh_instance_count: i32,
}
#[godot_api]
impl IResource for VoxelInstanceLibraryMultiMeshItemGD {
    fn init(base: Base<Resource>) -> Self {
        Self {
            base,
            mesh_instance_count: 100,
        }
    }
}

#[godot_api]
impl VoxelInstanceLibraryMultiMeshItemGD {
    /// Whether the multimesh item has any instances configured.
    #[func]
    fn has_instances(&self) -> bool {
        self.mesh_instance_count > 0
    }
}

/// A scene-based instance library item (places PackedScenes, not multimesh).
///
/// The pinned GDScript-facing `scene` property mirrors upstream
/// `VoxelInstanceLibrarySceneItem` (5828cbeb).
#[derive(GodotClass)]
#[class(base = Resource, tool, rename = VoxelInstanceLibrarySceneItem)]
pub struct VoxelInstanceLibrarySceneItemGD {
    base: Base<Resource>,
    scene_path: GString,
    /// Pinned `scene` resource (backing field; `None` until assigned).
    scene_resource: Option<Gd<godot::classes::PackedScene>>,
    /// The pinned GDScript-facing `scene` property.
    #[var(get = get_scene, set = set_scene)]
    scene: PhantomVar<Option<Gd<godot::classes::PackedScene>>>,
}
#[godot_api]
impl IResource for VoxelInstanceLibrarySceneItemGD {
    fn init(base: Base<Resource>) -> Self {
        Self {
            base,
            scene_path: "".to_godot(),
            scene_resource: None,
            scene: PhantomVar::default(),
        }
    }
}

#[godot_api]
impl VoxelInstanceLibrarySceneItemGD {
    /// Whether a scene path has been assigned.
    #[func]
    fn has_scene(&self) -> bool {
        !self.scene_path.is_empty()
    }

    // -----------------------------------------------------------------
    // Pinned VoxelInstanceLibrarySceneItem properties
    // (upstream 5828cbeb: VoxelInstanceLibrarySceneItem.xml).
    // -----------------------------------------------------------------

    /// The `PackedScene` spawned for each instance (`None` until assigned).
    #[func]
    fn get_scene(&self) -> Option<Gd<godot::classes::PackedScene>> {
        self.scene_resource.clone()
    }

    #[func]
    fn set_scene(&mut self, scene: Option<Gd<godot::classes::PackedScene>>) {
        self.scene_resource = scene;
    }
}

/// An instance component attached to a node for scatter rendering. Stores a
/// single visibility flag (upstream `VoxelInstanceComponent`, which has no
/// engine-agnostic counterpart in `voxel-core`); the default and the
/// `is_visible`/`set_visible` accessor pair are pinned by tests below.
#[derive(GodotClass)]
#[class(base = Resource, tool, rename = VoxelInstanceComponent)]
pub struct VoxelInstanceComponentGD {
    base: Base<Resource>,
    visible: bool,
}
/// Default visibility of [`VoxelInstanceComponentGD`] (upstream default:
/// visible). Named constant so the pinned default and `init` cannot diverge.
const INSTANCE_COMPONENT_DEFAULT_VISIBLE: bool = true;
#[godot_api]
impl IResource for VoxelInstanceComponentGD {
    fn init(base: Base<Resource>) -> Self {
        Self {
            base,
            visible: INSTANCE_COMPONENT_DEFAULT_VISIBLE,
        }
    }
}

#[godot_api]
impl VoxelInstanceComponentGD {
    /// Whether the component is visible.
    #[func]
    fn is_visible(&self) -> bool {
        self.visible
    }

    #[func]
    fn set_visible(&mut self, v: bool) {
        self.visible = v;
    }
}

// (Three filler "editor inspector plugin" classes that registered under raw
// `GD`-suffixed names were removed: the canonical `VoxelTerrainEditorPlugin`,
// `VoxelInstancerEditorPlugin` and `VoxelGraphEditorPlugin` already live in
// `editor.rs`.)

// === Misc utility (3) ===

#[derive(GodotClass)]
#[class(base = RefCounted, tool, rename = VoxelTaskIndicator)]
pub struct VoxelTaskIndicatorGD {
    base: Base<RefCounted>,
    #[var]
    task_count: i32,
}
#[godot_api]
impl IRefCounted for VoxelTaskIndicatorGD {
    fn init(base: Base<RefCounted>) -> Self {
        Self {
            base,
            task_count: 0,
        }
    }
}

#[godot_api]
impl VoxelTaskIndicatorGD {
    /// Whether any background tasks are currently pending.
    #[func]
    fn is_busy(&self) -> bool {
        self.task_count > 0
    }

    /// Increment the pending task count.
    #[func]
    fn add_task(&mut self) {
        self.task_count += 1;
    }

    /// Decrement the pending task count (clamped at 0).
    #[func]
    fn remove_task(&mut self) {
        if self.task_count > 0 {
            self.task_count -= 1;
        }
    }
}

/// Caches the editor camera transform so plugins can restore it. The
/// functional API stores/retrieves a 3D position.
#[derive(GodotClass)]
#[class(base = RefCounted, tool, rename = VoxelEditorCameraCache)]
pub struct VoxelEditorCameraCacheGD {
    base: Base<RefCounted>,
    cached_x: f32,
    cached_y: f32,
    cached_z: f32,
    has_cache: bool,
}
#[godot_api]
impl IRefCounted for VoxelEditorCameraCacheGD {
    fn init(base: Base<RefCounted>) -> Self {
        Self {
            base,
            cached_x: 0.0,
            cached_y: 0.0,
            cached_z: 0.0,
            has_cache: false,
        }
    }
}

#[godot_api]
impl VoxelEditorCameraCacheGD {
    /// Store a camera position.
    #[func]
    fn store(&mut self, x: f32, y: f32, z: f32) {
        self.cached_x = x;
        self.cached_y = y;
        self.cached_z = z;
        self.has_cache = true;
    }

    /// Whether a cached position exists.
    #[func]
    fn has_cached(&self) -> bool {
        self.has_cache
    }

    /// Get the cached X coordinate (0 if none).
    #[func]
    fn get_x(&self) -> f32 {
        self.cached_x
    }

    #[func]
    fn get_y(&self) -> f32 {
        self.cached_y
    }

    #[func]
    fn get_z(&self) -> f32 {
        self.cached_z
    }
}

/// The "About" window resource. The functional API reports the voxel-core
/// version string for display.
#[derive(GodotClass)]
#[class(base = Resource, tool, rename = VoxelAboutWindow)]
pub struct VoxelAboutWindowGD {
    base: Base<Resource>,
}
#[godot_api]
impl IResource for VoxelAboutWindowGD {
    fn init(base: Base<Resource>) -> Self {
        Self { base }
    }
}

#[godot_api]
impl VoxelAboutWindowGD {
    /// Returns the voxel-core version string.
    #[func]
    fn get_version(&self) -> GString {
        voxel_core::VERSION.to_godot()
    }
}

#[cfg(test)]
mod zero_api_behavioral_tests {
    //! VoxelBlockyModelEmpty and VoxelInstanceComponent have 0 pinned
    //! methods/properties/signals/constants, but both feed real engine-free
    //! state; these tests exercise the actual contracts instead of literals.

    use super::INSTANCE_COMPONENT_DEFAULT_VISIBLE;

    /// The core contract behind `VoxelBlockyModelEmpty`: the model its
    /// `to_baked_model()` produces (`baked_model_from_resource` pushes it
    /// into the library as-is) stays an air model through
    /// `bake_library` — empty flag set, zero surfaces, no side geometry —
    /// in contrast to a solid cube model in the same library.
    #[test]
    fn blocky_model_empty_is_registered_and_reports_air() {
        use voxel_core::meshers::blocky::{self, BakedLibrary, BakedModel};
        let mut library = BakedLibrary::default();
        // A fresh library starts empty; VoxelBlockyLibraryGD reserves slot 0
        // for air, `baked_model_from_resource` then appends the empty model
        // a VoxelBlockyModelEmptyGD produces (BakedModel::default()), plus a
        // solid cube for contrast.
        library.models.push(BakedModel::default()); // reserved air slot 0
        library.models.push(BakedModel::default()); // VoxelBlockyModelEmptyGD
        library
            .models
            .push(blocky::solid_cube_model(voxel_core::math::Color::from_rgb(
                0.5, 0.5, 0.5,
            )));
        blocky::bake_library(&mut library);
        let air = &library.models[1];
        assert!(air.empty, "the empty model stays air after baking");
        assert_eq!(air.model.surface_count, 0, "an air model bakes no faces");
        for side in &air.model.sides_surfaces {
            for surface in side {
                assert!(
                    surface.positions.is_empty()
                        && surface.uvs.is_empty()
                        && surface.indices.is_empty()
                        && surface.tangents.is_empty(),
                    "an air model has no side geometry"
                );
            }
        }
        // Contrast: the solid cube in the same baked library is real matter.
        let cube = &library.models[2];
        assert!(!cube.empty);
        assert_eq!(cube.model.surface_count, 1);
        assert!(cube
            .model
            .sides_surfaces
            .iter()
            .any(|side| { side[0].indices.len().max(side[0].positions.len()) > 0 }));
    }

    /// `VoxelInstanceComponentGD` stores a plain `visible` flag
    /// (`INSTANCE_COMPONENT_DEFAULT_VISIBLE`, also used by `init`) exposed
    /// through the `is_visible`/`set_visible` #[func] pair — a plain field
    /// transaction. GD structs cannot be constructed under plain
    /// `cargo test`, so exercise that transaction against the same default.
    #[test]
    fn instance_component_is_registered_with_default_visibility() {
        let mut visible = INSTANCE_COMPONENT_DEFAULT_VISIBLE;
        assert!(visible, "the component defaults to visible");
        visible = false; // set_visible(false)
        assert!(!visible); // is_visible()
        visible = true; // set_visible(true)
        assert!(visible);
    }
}

#[cfg(test)]
mod input_validation_tests {
    use super::*;

    #[test]
    fn curve_identity_workload_is_bounded_before_allocation() {
        assert_eq!(validate_curve_point_count(2), Ok(2));
        assert_eq!(
            validate_curve_point_count(MAX_CURVE_POINTS as i32),
            Ok(MAX_CURVE_POINTS)
        );
        assert!(validate_curve_point_count(MAX_CURVE_POINTS as i32 + 1).is_err());
        assert!(validate_curve_point_count(i32::MAX).is_err());
    }

    #[test]
    fn rejected_curve_identity_preserves_the_complete_previous_state() {
        let mut point_count = 4;
        let mut curve = voxel_core::generators::simple::Curve::identity(4);
        let before = [curve.sample(0.0), curve.sample(0.25), curve.sample(1.0)];

        if let Ok((next_count, next_curve)) = prepare_identity_curve(i32::MAX) {
            point_count = next_count;
            curve = next_curve;
        }

        assert_eq!(point_count, 4);
        assert_eq!(
            [curve.sample(0.0), curve.sample(0.25), curve.sample(1.0)],
            before
        );
    }

    #[test]
    fn explicit_curve_points_reject_nonfinite_values_and_oversized_tables() {
        assert!(validate_curve_points(&[0.0, 1.0]).is_ok());
        assert!(validate_curve_points(&[0.0, f32::NAN]).is_err());
        assert!(validate_curve_points(&[f32::NEG_INFINITY, 1.0]).is_err());
        assert!(validate_curve_point_count(MAX_CURVE_POINTS as i32 + 1).is_err());
    }

    #[test]
    fn spot_grid_work_is_nonnegative_checked_and_bounded() {
        assert_eq!(validate_spot_grid_work(0), Ok(0));
        assert_eq!(validate_spot_grid_work(256), Ok(65_536));
        assert!(validate_spot_grid_work(-1).is_err());
        assert!(validate_spot_grid_work(257).is_err());
        assert!(validate_spot_grid_work(i32::MAX).is_err());
        assert!(checked_square_work(u64::MAX).is_err());
    }

    #[test]
    fn spot_grid_compound_coordinates_must_stay_finite() {
        assert!(validate_spot_coordinate_work(256, 2.0).is_ok());
        assert!(validate_spot_coordinate_work(256, f32::MAX).is_err());
        assert!(validate_spot_coordinate_work(1, f32::MAX).is_ok());
    }

    #[test]
    fn noise_float_validation_rejects_nonfinite_and_nonpositive_scales() {
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(validate_finite_float(invalid).is_err());
            assert!(validate_positive_finite_float(invalid).is_err());
        }
        assert!(validate_positive_finite_float(0.0).is_err());
        assert!(validate_positive_finite_float(-0.01).is_err());
        assert_eq!(
            validate_positive_finite_float(f32::MIN_POSITIVE),
            Ok(f32::MIN_POSITIVE)
        );
        assert_eq!(validate_positive_finite_float(0.01), Ok(0.01));
    }

    #[test]
    fn rejected_noise_float_assignment_preserves_previous_state() {
        let mut frequency = 0.01;
        if let Ok(next) = validate_positive_finite_float(f32::NAN) {
            frequency = next;
        }
        assert_eq!(frequency, 0.01);

        let mut density = 0.5;
        if let Ok(next) = validate_unit_float(1.5) {
            density = next;
        }
        assert_eq!(density, 0.5);
    }

    // Setter-transaction tests. `Gd<T>` cannot be constructed in pure unit
    // tests (it requires a live Godot engine; `Gd::from_init_fn` panics with
    // "Godot binding accessed before initialization"). Each `#[func] set_*`
    // body applies the exact transaction under test — validate first, assign
    // only on success, log `godot_error!` and leave all state untouched on
    // reject — so we exercise that transaction directly against a backing
    // field, mirroring `rejected_noise_float_assignment_preserves_previous_state`.

    #[test]
    fn fast_noise_lite_frequency_setter_rejects_nonpositive_and_preserves_state() {
        let mut frequency_value = 0.01;
        if validate_positive_finite_float(0.02).is_ok() {
            frequency_value = 0.02;
        }
        assert_eq!(frequency_value, 0.02);
        for invalid in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            if validate_positive_finite_float(invalid).is_ok() {
                frequency_value = invalid;
            }
        }
        assert_eq!(frequency_value, 0.02);
    }

    #[test]
    fn spot_noise_density_setter_rejects_out_of_unit_and_preserves_state() {
        let mut density_value = 0.5;
        // Valid unit-float assignments accepted at the boundaries.
        if validate_unit_float(0.0).is_ok() {
            density_value = 0.0;
        }
        assert_eq!(density_value, 0.0);
        if validate_unit_float(1.0).is_ok() {
            density_value = 1.0;
        }
        assert_eq!(density_value, 1.0);
        // Out-of-range and non-finite rejected; previous state preserved.
        for invalid in [-0.01, 1.5, f32::NAN, f32::INFINITY] {
            if validate_unit_float(invalid).is_ok() {
                density_value = invalid;
            }
        }
        assert_eq!(density_value, 1.0);
    }

    #[test]
    fn spot_noise_radius_setter_rejects_nonpositive_and_preserves_state() {
        let mut radius_value = 2.0;
        if validate_positive_finite_float(4.0).is_ok() {
            radius_value = 4.0;
        }
        assert_eq!(radius_value, 4.0);
        for invalid in [0.0, -2.0, f32::NAN, f32::NEG_INFINITY] {
            if validate_positive_finite_float(invalid).is_ok() {
                radius_value = invalid;
            }
        }
        assert_eq!(radius_value, 4.0);
    }

    #[test]
    fn noise_pattern_scale_setter_rejects_nonpositive_and_preserves_state() {
        let mut scale_value = 1.0;
        if validate_positive_finite_float(3.0).is_ok() {
            scale_value = 3.0;
        }
        assert_eq!(scale_value, 3.0);
        for invalid in [0.0, -3.0, f32::NAN, f32::INFINITY] {
            if validate_positive_finite_float(invalid).is_ok() {
                scale_value = invalid;
            }
        }
        assert_eq!(scale_value, 3.0);
    }
}

/// Behavioral tests for the noise resources. GD-backed structs cannot be
/// constructed without a live Godot engine, so these exercise the extracted
/// engine-free helpers that the `#[func]` methods delegate to (mirroring
/// `input_validation_tests`).
#[cfg(test)]
mod noise_behavioral_tests {
    use super::*;

    // === ZN_SpotNoise: spot-grid enumeration (upstream model) ===

    #[test]
    fn spot_noise_area_2d_returns_expected_centers() {
        // Every covered cell owns exactly one spot whose center is the
        // upstream formula (`lerp(0.5, hash_to_vec2(hash2(cell, seed)),
        // jitter) * cell_size`), always inside its own cell.
        let cfg = SpotGridConfig::new(8.0, 0.9, 4.0, 1337).expect("valid config");
        let rect = (0.0, 0.0, 32.0, 32.0);
        let centers = spot_centers_in_rect(cfg, rect).expect("bounded workload");
        // The rect's cell range is exactly floor(min/cs)..=floor(max/cs)
        // (5×5 here); cells whose center falls outside the rect contribute
        // nothing, so the enumeration equals the per-cell oracle over that
        // range.
        let mut expected = Vec::new();
        for cell_y in 0..=4i32 {
            for cell_x in 0..=4i32 {
                let (x, y) = spot_center_2d(cfg, cell_x, cell_y);
                if x >= rect.0 && x < rect.2 && y >= rect.1 && y < rect.3 {
                    expected.push((x, y));
                }
            }
        }
        assert_eq!(centers.len(), expected.len());
        let mut sorted = centers.clone();
        sorted.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
        expected.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
        assert_eq!(sorted, expected, "one exact center per covered cell");
        for &(x, y) in &centers {
            // The center stays inside its own cell (upstream's jitter bound).
            let cell_x = (x / cfg.cell_size).floor() as i32;
            let cell_y = (y / cfg.cell_size).floor() as i32;
            assert_eq!(spot_center_2d(cfg, cell_x, cell_y), (x, y));
            // The center is inside its own spot: noise is 1 there.
            assert_eq!(spot_grid_noise_2d(cfg, x, y), 1.0);
        }
        // jitter = 0 pins every spot exactly to its cell center: a clean
        // 4×4 block of cells under [0, 32] reports 16 exact centers.
        let centered = SpotGridConfig::new(8.0, 0.0, 4.0, 1337).expect("valid config");
        let exact = spot_centers_in_rect(centered, rect).expect("bounded workload");
        assert_eq!(exact.len(), 16);
        for (cell_y, row) in exact.chunks(4).enumerate() {
            for (cell_x, &(x, y)) in row.iter().enumerate() {
                assert_eq!(
                    (x, y),
                    ((cell_x as f32 + 0.5) * 8.0, (cell_y as f32 + 0.5) * 8.0)
                );
            }
        }
    }

    #[test]
    fn spot_noise_area_end_edge_is_exclusive_like_upstream_has_point() {
        // Upstream filters area results with Rect2/AABB `has_point`, which is
        // end-exclusive: a center exactly on the end edge is NOT inside, so
        // adjacent rects tiling an area report each boundary spot exactly
        // once. With jitter 0 the center of cell 0 at cell_size 8 is exactly
        // (4.0, 4.0) — on the end edge of [0, 4) × [0, 4).
        let cfg = SpotGridConfig::new(8.0, 0.0, 4.0, 1337).expect("valid config");
        let centers = spot_centers_in_rect(cfg, (0.0, 0.0, 4.0, 4.0)).expect("bounded workload");
        assert!(
            centers.is_empty(),
            "end-edge centers must be excluded (upstream has_point), got {centers:?}"
        );
        // A rect extended past the edge includes it exactly once.
        let centers = spot_centers_in_rect(cfg, (0.0, 0.0, 4.5, 4.5)).expect("bounded workload");
        assert_eq!(centers, vec![(4.0, 4.0)]);
    }

    #[test]
    fn spot_noise_area_workload_is_bounded() {
        let cfg = SpotGridConfig::new(1.0, 0.5, 0.5, 1).expect("valid config");
        // ~1000×1000 cells far exceeds the 65 536-cell budget.
        assert!(spot_centers_in_rect(cfg, (0.0, 0.0, 1000.0, 1000.0)).is_err());
        // Huge aabb rejected through the X*Y*Z cell budget (true 3D cells).
        assert!(spot_centers_in_aabb(cfg, (0.0, 0.0, 0.0, 1000.0, 10.0, 1000.0)).is_err());
        // A tall-but-thin aabb still exceeds the budget through its Y cells.
        assert!(spot_centers_in_aabb(cfg, (0.0, 0.0, 0.0, 4.0, 1_000_000.0, 4.0)).is_err());
        // A small aabb fits the budget and yields at most one center per 3D
        // cell (a [0, 4]³ box at cell size 1 spans a 5×5×5 cell block).
        let small = spot_centers_in_aabb(cfg, (0.0, 0.0, 0.0, 4.0, 4.0, 4.0))
            .expect("125-cell aabb is within budget");
        assert!(!small.is_empty());
        assert!(small.len() <= 125, "one center per 3D cell at most");
        // Inverted (empty) areas return no centers without erroring.
        assert!(spot_centers_in_rect(cfg, (10.0, 10.0, -10.0, -10.0))
            .expect("inverted rect is empty, not an error")
            .is_empty());
        assert!(spot_centers_in_aabb(cfg, (0.0, 5.0, 0.0, 10.0, -5.0, 10.0))
            .expect("inverted aabb is empty, not an error")
            .is_empty());
        // Non-finite bounds are rejected.
        assert!(spot_centers_in_rect(cfg, (f32::NAN, 0.0, 1.0, 1.0)).is_err());
        assert!(spot_centers_in_aabb(cfg, (0.0, 0.0, 0.0, f32::INFINITY, 1.0, 1.0)).is_err());
    }

    #[test]
    fn spot_noise_get_noise_agrees_with_area_membership() {
        // Containing-cell semantics (upstream spot_noise_2d): noise(p) == 1
        // ⟺ p's own cell's spot center is within spot_radius of p. The
        // enumeration reports exactly one center per cell, so both views of
        // the field agree.
        let cfg = SpotGridConfig::new(8.0, 0.9, 4.0, 1337).expect("valid config");
        let rect = (0.0, 0.0, 64.0, 64.0);
        let centers = spot_centers_in_rect(cfg, rect).expect("bounded workload");
        assert!(!centers.is_empty());

        let radius_squared = cfg.spot_radius * cfg.spot_radius;
        let mut sampled = 0usize;
        let mut inside = 0usize;
        let mut y = rect.1;
        while y < rect.3 {
            let mut x = rect.0;
            while x < rect.2 {
                sampled += 1;
                let cell_x = (x / cfg.cell_size).floor() as i32;
                let cell_y = (y / cfg.cell_size).floor() as i32;
                let center = spot_center_2d(cfg, cell_x, cell_y);
                let ds = (x - center.0) * (x - center.0) + (y - center.1) * (y - center.1);
                if spot_grid_noise_2d(cfg, x, y) == 1.0 {
                    inside += 1;
                    assert!(
                        ds < radius_squared,
                        "noise=1 but the own-cell center is beyond the radius at ({x},{y})"
                    );
                    assert!(
                        centers.contains(&center),
                        "the deciding own-cell center is not enumerated at ({x},{y})"
                    );
                } else {
                    assert!(
                        ds >= radius_squared,
                        "noise=0 but the own-cell center is within the radius at ({x},{y})"
                    );
                }
                x += 1.0;
            }
            y += 1.0;
        }
        assert!(sampled > 100, "expected a meaningful sample count");
        assert!(inside > 0, "expected at least one spot interior point");

        // Determinism: identical config → identical enumeration; a different
        // seed must move the spots.
        let centers_again = spot_centers_in_rect(cfg, rect).expect("bounded workload");
        assert_eq!(centers, centers_again);
        let other_seed = SpotGridConfig::new(8.0, 0.9, 4.0, 1338).expect("valid config");
        let centers_other = spot_centers_in_rect(other_seed, rect).expect("bounded workload");
        assert_ne!(
            centers, centers_other,
            "a different seed must move the spots"
        );

        // 3D: true 3D cells. Every enumerated center is the center of its own
        // 3D cell, the noise is 1 exactly there, and each sample point's
        // noise is decided by its own cell's center.
        let aabb = (0.0, 0.0, 0.0, 64.0, 64.0, 64.0);
        let positions = spot_centers_in_aabb(cfg, aabb).expect("bounded workload");
        assert!(!positions.is_empty());
        for &(x, y, z) in &positions {
            assert!(x >= aabb.0 && x <= aabb.3, "x inside aabb");
            assert!(y >= aabb.1 && y <= aabb.4, "y inside aabb");
            assert!(z >= aabb.2 && z <= aabb.5, "z inside aabb");
            let cell_x = (x / cfg.cell_size).floor() as i32;
            let cell_y = (y / cfg.cell_size).floor() as i32;
            let cell_z = (z / cfg.cell_size).floor() as i32;
            assert_eq!(spot_center_3d(cfg, cell_x, cell_y, cell_z), (x, y, z));
            assert_eq!(spot_grid_noise_3d(cfg, x, y, z), 1.0);
        }
        for &(x, y, z) in &[
            (3.5, 12.25, 40.0),
            (60.0, 0.5, 7.75),
            (33.3, 33.3, 33.3),
            (8.0, 8.0, 8.0),
        ] {
            let cell_x = (x / cfg.cell_size).floor() as i32;
            let cell_y = (y / cfg.cell_size).floor() as i32;
            let cell_z = (z / cfg.cell_size).floor() as i32;
            let (cx, cy, cz) = spot_center_3d(cfg, cell_x, cell_y, cell_z);
            let ds = (x - cx) * (x - cx) + (y - cy) * (y - cy) + (z - cz) * (z - cz);
            let expected = if ds < cfg.spot_radius * cfg.spot_radius {
                1.0
            } else {
                0.0
            };
            assert_eq!(spot_grid_noise_3d(cfg, x, y, z), expected);
        }

        // The 3D field responds to Y (no more Y-ignoring columns): with
        // jitter 0 every spot sits exactly at its cell center, so a column
        // through the XZ cell centers alternates inside/outside across Y
        // cell boundaries.
        let centered = SpotGridConfig::new(8.0, 0.0, 1.5, 1337).expect("valid config");
        let mut saw_inside = false;
        let mut saw_outside = false;
        let mut y = 0.0f32;
        while y < 64.0 {
            let value = spot_grid_noise_3d(centered, 4.0, y, 4.0);
            if value == 1.0 {
                saw_inside = true;
            } else {
                saw_outside = true;
            }
            y += 1.0;
        }
        assert!(
            saw_inside && saw_outside,
            "the 3D field must vary along Y within one (x, z) column"
        );
    }

    #[test]
    fn spot_noise_hash_mirrors_upstream_constants() {
        // The exact upstream hash construction (spot_noise.h): wrapping
        // 32-bit arithmetic over the pinned PRIME constants.
        assert_eq!(SPOT_PRIME_X, 501_125_321);
        assert_eq!(SPOT_PRIME_Y, 1_136_930_381);
        assert_eq!(SPOT_PRIME_Z, 1_720_413_743);
        assert_eq!(SPOT_HASH_MULTIPLIER, 0x27d4eb2d);
        // A seed-only hash (cell 0,0) is exactly the multiplier: with
        // seed=1 the pre-multiply hash is 1, and 1 * 0x27d4eb2d = itself.
        assert_eq!(spot_hash_2d(0, 0, 1), 0x27d4eb2d);
        // The lane decoders produce values in [0, 1] with upstream's
        // resolutions (2^16 and 2^10 locations per axis).
        for cell in -3..=3i32 {
            let h = spot_hash_2d(cell, cell * 2, 42);
            let (a, b) = spot_hash_to_vec2(h);
            assert!((0.0..=1.0).contains(&a) && (0.0..=1.0).contains(&b));
            let h3 = spot_hash_3d(cell, -cell, cell + 7, 42);
            let (a, b, c) = spot_hash_to_vec3(h3);
            assert!(
                (0.0..=1.0).contains(&a) && (0.0..=1.0).contains(&b) && (0.0..=1.0).contains(&c)
            );
        }
        // jitter = 0 pins the spot to the cell center on every axis.
        assert_eq!(spot_position_norm_2d(5, -9, 0.0, 7), (0.5, 0.5));
        assert_eq!(spot_position_norm_3d(5, -9, 2, 0.0, 7), (0.5, 0.5, 0.5));
        // jitter keeps the spot inside its own cell for any hash.
        for cell in -2..=2i32 {
            let (nx, ny) = spot_position_norm_2d(cell, cell, 1.0, 1337);
            assert!((0.0..=1.0).contains(&nx) && (0.0..=1.0).contains(&ny));
        }
    }

    // === ZN_FastNoiseLite ===

    #[test]
    fn fast_noise_lite_get_noise_3dv_equivalent_to_get_noise_3d() {
        let cfg = FastNoiseLiteConfig::pinned_defaults();
        for &(x, y, z) in &[(0.0, 0.0, 0.0), (12.5, -3.25, 7.0), (-100.0, 55.5, 0.25)] {
            assert_eq!(cfg.sample_3dv((x, y, z)), cfg.sample_3d(x, y, z));
        }
        // The 2D overload pairs the same way.
        for &(x, y) in &[(0.0, 0.0), (31.7, -4.2)] {
            assert_eq!(cfg.sample_2dv((x, y)), cfg.sample_2d(x, y));
        }
    }

    #[test]
    fn fast_noise_lite_2d_matches_crate_true_2d_sampler() {
        use voxel_core::fastnoise_lite::{
            CellularDistanceFunction, CellularReturnType, FastNoiseLite, FractalType, NoiseType,
            RotationType3D,
        };
        let cfg = FastNoiseLiteConfig::pinned_defaults();
        // Reference sampler built straight from the crate, configured to the
        // pinned defaults.
        let mut reference = FastNoiseLite::new();
        reference.set_seed(Some(cfg.seed));
        reference.set_frequency(Some(cfg.frequency));
        reference.set_noise_type(Some(NoiseType::OpenSimplex2));
        reference.set_rotation_type_3d(Some(RotationType3D::None));
        reference.set_fractal_type(Some(FractalType::FBm));
        reference.set_fractal_octaves(Some(cfg.fractal_octaves));
        reference.set_fractal_lacunarity(Some(cfg.fractal_lacunarity));
        reference.set_fractal_gain(Some(cfg.fractal_gain));
        reference.set_fractal_weighted_strength(Some(cfg.fractal_weighted_strength));
        reference.set_fractal_ping_pong_strength(Some(cfg.fractal_ping_pong_strength));
        reference.set_cellular_distance_function(Some(CellularDistanceFunction::EuclideanSq));
        reference.set_cellular_return_type(Some(CellularReturnType::Distance));
        reference.set_cellular_jitter(Some(cfg.cellular_jitter));

        let mut differs_from_3d_slice = false;
        for y in 0..8i32 {
            for x in 0..8i32 {
                let (px, py) = (x as f32 * 71.3, y as f32 * 33.7);
                let v2 = cfg.sample_2d(px, py);
                assert_eq!(v2, reference.get_noise_2d(px, py), "true 2D at ({px},{py})");
                differs_from_3d_slice |= v2 != reference.get_noise_3d(px, 0.0, py);
            }
        }
        assert!(
            differs_from_3d_slice,
            "2D sampling must not delegate to the (x, 0, y) 3D slice"
        );
    }

    #[test]
    fn fast_noise_lite_constants_match_pinned_enum_values() {
        use FastNoiseLiteGD as C;
        // NoiseType (6).
        assert_eq!(C::TYPE_OPEN_SIMPLEX_2, 0);
        assert_eq!(C::TYPE_OPEN_SIMPLEX_2S, 1);
        assert_eq!(C::TYPE_CELLULAR, 2);
        assert_eq!(C::TYPE_PERLIN, 3);
        assert_eq!(C::TYPE_VALUE_CUBIC, 4);
        assert_eq!(C::TYPE_VALUE, 5);
        // FractalType (4).
        assert_eq!(C::FRACTAL_NONE, 0);
        assert_eq!(C::FRACTAL_FBM, 1);
        assert_eq!(C::FRACTAL_RIDGED, 2);
        assert_eq!(C::FRACTAL_PING_PONG, 3);
        // RotationType3D (3).
        assert_eq!(C::ROTATION_3D_NONE, 0);
        assert_eq!(C::ROTATION_3D_IMPROVE_XY_PLANES, 1);
        assert_eq!(C::ROTATION_3D_IMPROVE_XZ_PLANES, 2);
        // CellularDistanceFunction (4).
        assert_eq!(C::CELLULAR_DISTANCE_EUCLIDEAN, 0);
        assert_eq!(C::CELLULAR_DISTANCE_EUCLIDEAN_SQ, 1);
        assert_eq!(C::CELLULAR_DISTANCE_MANHATTAN, 2);
        assert_eq!(C::CELLULAR_DISTANCE_HYBRID, 3);
        // CellularReturnType (7).
        assert_eq!(C::CELLULAR_RETURN_CELL_VALUE, 0);
        assert_eq!(C::CELLULAR_RETURN_DISTANCE, 1);
        assert_eq!(C::CELLULAR_RETURN_DISTANCE_2, 2);
        assert_eq!(C::CELLULAR_RETURN_DISTANCE_2_ADD, 3);
        assert_eq!(C::CELLULAR_RETURN_DISTANCE_2_SUB, 4);
        assert_eq!(C::CELLULAR_RETURN_DISTANCE_2_MUL, 5);
        assert_eq!(C::CELLULAR_RETURN_DISTANCE_2_DIV, 6);
    }

    #[test]
    fn fast_noise_lite_defaults_match_pinned_xml() {
        let d = FastNoiseLiteConfig::pinned_defaults();
        assert_eq!(d.seed, 0);
        assert_eq!(d.noise_type, 0); // OpenSimplex2
        assert_eq!(d.fractal_type, 1); // FBm
        assert_eq!(d.fractal_octaves, 3);
        assert_eq!(d.fractal_lacunarity, 2.0);
        assert_eq!(d.fractal_gain, 0.5);
        assert_eq!(d.fractal_weighted_strength, 0.0);
        assert_eq!(d.fractal_ping_pong_strength, 2.0);
        assert_eq!(d.cellular_distance_function, 1); // EuclideanSq
        assert_eq!(d.cellular_jitter, 1.0);
        assert_eq!(d.cellular_return_type, 1); // Distance
        assert_eq!(d.rotation_type_3d, 0); // None
                                           // period default 64.0 (frequency == 1/period).
        assert!((1.0 / d.frequency - 64.0).abs() < 1e-4);
    }

    // === FastNoise2 ===

    #[test]
    fn fastnoise2_constants_match_pinned_enum_values() {
        use FastNoise2GD as C;
        // NoiseType (7).
        assert_eq!(C::TYPE_OPEN_SIMPLEX_2, 0);
        assert_eq!(C::TYPE_SIMPLEX, 1);
        assert_eq!(C::TYPE_PERLIN, 2);
        assert_eq!(C::TYPE_VALUE, 3);
        assert_eq!(C::TYPE_CELLULAR, 4);
        assert_eq!(C::TYPE_ENCODED_NODE_TREE, 5);
        assert_eq!(C::TYPE_CELLULAR_VALUE, 6);
        // FractalType (4).
        assert_eq!(C::FRACTAL_NONE, 0);
        assert_eq!(C::FRACTAL_FBM, 1);
        assert_eq!(C::FRACTAL_RIDGED, 2);
        assert_eq!(C::FRACTAL_PING_PONG, 3);
        // CellularDistanceFunction (5).
        assert_eq!(C::CELLULAR_DISTANCE_EUCLIDEAN, 0);
        assert_eq!(C::CELLULAR_DISTANCE_EUCLIDEAN_SQ, 1);
        assert_eq!(C::CELLULAR_DISTANCE_MANHATTAN, 2);
        assert_eq!(C::CELLULAR_DISTANCE_HYBRID, 3);
        assert_eq!(C::CELLULAR_DISTANCE_MAX_AXIS, 4);
        // CellularReturnType (5).
        assert_eq!(C::CELLULAR_RETURN_INDEX_0, 0);
        assert_eq!(C::CELLULAR_RETURN_INDEX_0_ADD_1, 1);
        assert_eq!(C::CELLULAR_RETURN_INDEX_0_SUB_1, 2);
        assert_eq!(C::CELLULAR_RETURN_INDEX_0_MUL_1, 3);
        assert_eq!(C::CELLULAR_RETURN_INDEX_0_DIV_1, 4);
        // SIMDLevel (12).
        assert_eq!(C::SIMD_NULL, 0);
        assert_eq!(C::SIMD_SCALAR, 1);
        assert_eq!(C::SIMD_SSE, 2);
        assert_eq!(C::SIMD_SSE2, 4);
        assert_eq!(C::SIMD_SSE3, 8);
        assert_eq!(C::SIMD_SSSE3, 16);
        assert_eq!(C::SIMD_SSE41, 32);
        assert_eq!(C::SIMD_SSE42, 64);
        assert_eq!(C::SIMD_AVX, 128);
        assert_eq!(C::SIMD_AVX2, 256);
        assert_eq!(C::SIMD_AVX512, 512);
        assert_eq!(C::SIMD_NEON, 65_536);
    }

    #[test]
    fn fastnoise2_remap_transforms_output_range() {
        // [-1, 1] -> [0, 1].
        assert_eq!(apply_remap(-1.0, -1.0, 1.0, 0.0, 1.0, true), 0.0);
        assert_eq!(apply_remap(0.0, -1.0, 1.0, 0.0, 1.0, true), 0.5);
        assert_eq!(apply_remap(1.0, -1.0, 1.0, 0.0, 1.0, true), 1.0);
        assert_eq!(apply_remap(0.25, -1.0, 1.0, 0.0, 1.0, true), 0.625);
        // Unclamped linear extrapolation, matching upstream's Remap node.
        assert_eq!(apply_remap(3.0, -1.0, 1.0, 0.0, 1.0, true), 2.0);
        // A whole noise range maps inside the output range.
        for step in -20..=20i32 {
            let v = step as f32 * 0.05;
            let out = apply_remap(v, -1.0, 1.0, 0.0, 1.0, true);
            assert!((0.0..=1.0).contains(&out), "{v} -> {out}");
        }
        // Disabled = identity.
        assert_eq!(apply_remap(0.25, -1.0, 1.0, 0.0, 1.0, false), 0.25);
        // Degenerate input range = safe passthrough (never NaN from division).
        assert_eq!(apply_remap(0.7, 2.0, 2.0, 0.0, 1.0, true), 0.7);
        assert!(apply_remap(0.7, 2.0, 2.0, 0.0, 1.0, true).is_finite());
    }

    #[test]
    fn fastnoise2_generate_image_normalization_uses_observed_range() {
        // generate_image normalizes pixels with the batch's observed min/max
        // instead of assuming [-1, 1]: the batch extremes map to 0 and 1 no
        // matter where the raw range sits (remap may move it).
        assert_eq!(normalize_gray(0.25, 0.25, 0.75), 0.0);
        assert_eq!(normalize_gray(0.75, 0.25, 0.75), 1.0);
        assert_eq!(normalize_gray(0.5, 0.25, 0.75), 0.5);
        // A remap-moved range like [0.2, 0.6] still spans the full grayscale.
        assert_eq!(normalize_gray(0.2, 0.2, 0.6), 0.0);
        assert_eq!(normalize_gray(0.6, 0.2, 0.6), 1.0);
        // Degenerate (constant) batch writes mid-gray, never NaN.
        assert_eq!(normalize_gray(0.3, 0.3, 0.3), 0.5);
        assert_eq!(normalize_gray(f32::NAN, f32::NAN, f32::NAN), 0.5);
        // Clamped so any input yields a valid color component.
        assert_eq!(normalize_gray(2.0, 0.0, 1.0), 1.0);
        assert_eq!(normalize_gray(-1.0, 0.0, 1.0), 0.0);
    }

    #[test]
    fn fastnoise2_terrace_smoothness_extremes() {
        let multiplier = 4.0_f32;
        // smoothness = 1.0 ≈ identity across the noise range.
        for step in 0..=40i32 {
            let v = -1.0 + step as f32 * 0.05;
            let out = apply_terrace(v, multiplier, 1.0, true);
            assert!(
                (out - v).abs() < 1e-6,
                "smoothness 1.0 must be the identity at {v}, got {out}"
            );
        }
        // smoothness = 0.0 = hard steps onto multiples of 1/multiplier.
        for step in 0..=40i32 {
            let v = -1.0 + step as f32 * 0.05;
            let stepped = apply_terrace(v, multiplier, 0.0, true);
            assert_eq!(stepped, (v * multiplier).round() / multiplier);
            let levels = stepped * multiplier;
            assert!(
                (levels - levels.round()).abs() < 1e-5,
                "stepped output must sit on a level: {stepped}"
            );
        }
        // Intermediate smoothness stays monotone and within one step width.
        let mut previous = f32::NEG_INFINITY;
        for step in 0..=200i32 {
            let v = -2.0 + step as f32 * 0.02;
            let out = apply_terrace(v, multiplier, 0.5, true);
            assert!(
                out >= previous,
                "terrace must be monotone at {v}, got {out}"
            );
            assert!(
                (out - v).abs() <= 0.25 / multiplier + 1e-5,
                "terrace must stay within one step width at {v}, got {out}"
            );
            previous = out;
        }
        // Disabled and degenerate multiplier are safe passthroughs.
        assert_eq!(apply_terrace(0.37, 4.0, 0.5, false), 0.37);
        assert_eq!(apply_terrace(0.37, 0.0, 0.5, true), 0.37);
        assert!(apply_terrace(0.37, f32::NAN, 0.5, true).is_finite());
    }
}
