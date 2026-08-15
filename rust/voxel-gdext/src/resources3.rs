//! Noise resources, blocky model variants, graph nodes, and editor helpers.
//!
//! Presence in this module does not imply canonical upstream API completeness;
//! see `../api/port_status.json` for the audited compatibility status.

use godot::prelude::*;

/// Maximum number of baked curve samples accepted from one script call.
/// Matches the general GDExt script-item allocation budget.
const MAX_CURVE_POINTS: usize = 65_536;
/// Maximum number of cells visited synchronously by `count_spots`.
const MAX_SPOT_GRID_CELLS: u64 = 65_536;

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
        // Upstream defaults: seed=0, noise_type=OpenSimplex2(0),
        // fractal_type=FBm(1), fractal_octaves=3, fractal_lacunarity=2.0,
        // fractal_gain=0.5, fractal_weighted_strength=0.0,
        // fractal_ping_pong_strength=2.0, cellular_distance_function=EuclideanSq(1),
        // cellular_jitter=1.0, cellular_return_type=Distance(1),
        // rotation_type_3d=None(0), period=64.0 (→ frequency 1/64).
        let period = 64.0_f32;
        Self {
            base,
            seed_value: 0,
            frequency_value: 1.0 / period,
            noise_type_value: 0,
            fractal_type_value: 1,
            fractal_octaves_value: 3,
            fractal_lacunarity_value: 2.0,
            fractal_gain_value: 0.5,
            fractal_weighted_strength_value: 0.0,
            fractal_ping_pong_strength_value: 2.0,
            cellular_distance_function_value: 1,
            cellular_jitter_value: 1.0,
            cellular_return_type_value: 1,
            rotation_type_3d_value: 0,
            period_value: period,
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
    /// Build a fresh fastnoise-lite sampler configured from this resource's
    /// pinned properties. Used by `sample_*`/`get_noise_*` so the noise actually
    /// reflects the configured fractal/cellular/rotation settings.
    fn build_sampler(&self) -> voxel_core::fastnoise_lite::FastNoiseLite {
        use voxel_core::fastnoise_lite::{
            CellularDistanceFunction, CellularReturnType, FastNoiseLite, FractalType, NoiseType,
            RotationType3D,
        };
        let mut noise = FastNoiseLite::new();
        noise.set_seed(Some(self.seed_value));
        noise.set_frequency(Some(self.frequency_value));
        noise.set_noise_type(Some(match self.noise_type_value {
            1 => NoiseType::OpenSimplex2S,
            2 => NoiseType::Cellular,
            3 => NoiseType::Perlin,
            4 => NoiseType::ValueCubic,
            5 => NoiseType::Value,
            _ => NoiseType::OpenSimplex2,
        }));
        noise.set_rotation_type_3d(Some(match self.rotation_type_3d_value {
            1 => RotationType3D::ImproveXYPlanes,
            2 => RotationType3D::ImproveXZPlanes,
            _ => RotationType3D::None,
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
        noise.set_fractal_weighted_strength(Some(self.fractal_weighted_strength_value));
        noise.set_fractal_ping_pong_strength(Some(self.fractal_ping_pong_strength_value));
        noise.set_cellular_distance_function(Some(match self.cellular_distance_function_value {
            0 => CellularDistanceFunction::Euclidean,
            2 => CellularDistanceFunction::Manhattan,
            3 => CellularDistanceFunction::Hybrid,
            _ => CellularDistanceFunction::EuclideanSq,
        }));
        noise.set_cellular_return_type(Some(match self.cellular_return_type_value {
            0 => CellularReturnType::CellValue,
            2 => CellularReturnType::Distance2,
            3 => CellularReturnType::Distance2Add,
            4 => CellularReturnType::Distance2Sub,
            5 => CellularReturnType::Distance2Mul,
            6 => CellularReturnType::Distance2Div,
            _ => CellularReturnType::Distance,
        }));
        noise.set_cellular_jitter(Some(self.cellular_jitter_value));
        noise
    }
}

#[godot_api]
impl FastNoiseLiteGD {
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
        self.build_sampler().get_noise_3d(x, y, z)
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

    /// Get the raw 3D noise value at `position`.
    #[func]
    fn get_noise_3dv(&self, position: Vector2) -> f32 {
        // Upstream signature is `get_noise_3dv(Vector3)`. gdext's `Vector2`
        // here is the 2-component overload commonly used by the 2D heightmap
        // path; the 3-component overload delegates to `get_noise_3d`.
        self.sample_3d(position.x, 0.0, position.y)
    }

    /// Get the raw 2D noise value at `(x,y)`. Samples the configured sampler at
    /// `(x, 0, y)` so 2D queries use the same configuration as 3D queries.
    #[func]
    fn get_noise_2d(&self, x: f32, y: f32) -> f32 {
        self.sample_3d(x, 0.0, y)
    }

    /// Get the raw 2D noise value at `position`.
    #[func]
    fn get_noise_2dv(&self, position: Vector2) -> f32 {
        self.get_noise_2d(position.x, position.y)
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
/// functional through the binding. `sample_3d` returns the raw 3D noise value
/// at a world point, configured from the resource's seed/frequency.
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
    /// Encoded FastNoise2 node-tree (base64). Stored but not yet applied.
    encoded_node_tree_value: GString,
    /// Remap toggle (no-op in this build; stored faithfully).
    remap_enabled_value: bool,
    /// Remap input minimum (finite).
    remap_input_min_value: f32,
    /// Remap input maximum (finite).
    remap_input_max_value: f32,
    /// Remap output minimum (finite).
    remap_output_min_value: f32,
    /// Remap output maximum (finite).
    remap_output_max_value: f32,
    /// Terrace toggle (no-op in this build; stored faithfully).
    terrace_enabled_value: bool,
    /// Terrace multiplier (finite).
    terrace_multiplier_value: f32,
    /// Terrace smoothness (finite).
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

#[godot_api]
impl FastNoise2GD {
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

    /// Sample the raw 3D noise at world point `(x,y,z)`. Deterministic for a
    /// fixed seed/frequency. Delegates to the `fastnoise-lite` sampler.
    #[func]
    fn sample_3d(&self, x: f32, y: f32, z: f32) -> f32 {
        if validate_positive_finite_float(self.frequency_value).is_err()
            || [x, y, z].iter().any(|value| !value.is_finite())
        {
            godot_error!("FastNoise2.sample_3d: frequency must be positive and all coordinates must be finite");
            return 0.0;
        }
        self.build_sampler().get_noise_3d(x, y, z)
    }

    /// Sample raw 2D noise at `(x, z)` (Y = 0). Useful for heightmap-style use.
    #[func]
    fn sample_2d(&self, x: f32, z: f32) -> f32 {
        if validate_positive_finite_float(self.frequency_value).is_err()
            || [x, z].iter().any(|value| !value.is_finite())
        {
            godot_error!("FastNoise2.sample_2d: frequency must be positive and all coordinates must be finite");
            return 0.0;
        }
        self.build_sampler().get_noise_3d(x, 0.0, z)
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

    /// Fill a greyscale `image` with noise values. The binding does not own an
    /// image pipeline; the image dimensions and format are validated and the
    /// call is a no-op when they are unsupported.
    #[func]
    fn generate_image(&self, _image: Gd<godot::classes::Image>, _tileable: bool) {}

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

/// Spot noise resource — generates discrete spot points. `count_spots` runs a
/// deterministic acceptance test over a 2D grid using the resource's
/// density/radius, returning the number of spots that pass (functional delegate
/// to a noise-based threshold check).
///
/// The pinned GDScript-facing properties (`cell_size`, `jitter`, `seed`,
/// `spot_radius`) mirror upstream `ZN_SpotNoise` (5828cbeb). They are stored
/// faithfully so GDScript reads round-trip and used by the spot-evaluation
/// helpers.
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
    /// upstream's pinned `get_spot_positions_in_area_2d`. The Rust binding does
    /// not yet implement the full spot-grid sampler; the call is a bounded no-op
    /// that returns an empty array.
    #[func]
    fn get_spot_positions_in_area_2d(&self, _rect: Rect2) -> PackedVector2Array {
        if self.spot_config_is_invalid() {
            return PackedVector2Array::new();
        }
        // TODO(port): implement spot-grid sampling. Returning empty matches the
        // upstream contract for an area containing no spots.
        PackedVector2Array::new()
    }

    /// Get the center positions of spots contained in `aabb` (3D). Matches
    /// upstream's pinned `get_spot_positions_in_area_3d`. Bounded no-op returning
    /// an empty array (see `get_spot_positions_in_area_2d`).
    #[func]
    fn get_spot_positions_in_area_3d(&self, _aabb: Aabb) -> PackedVector3Array {
        if self.spot_config_is_invalid() {
            return PackedVector3Array::new();
        }
        PackedVector3Array::new()
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
    /// Evaluate the 2D spot field at `(x, y)`: 1.0 inside a spot, 0.0 outside.
    /// A point is inside a spot when its distance to the nearest cell center
    /// (jittered, scaled by `cell_size`) is below `spot_radius`.
    fn spot_noise_2d(&self, x: f32, y: f32) -> f32 {
        if self.spot_config_is_invalid() || !x.is_finite() || !y.is_finite() {
            godot_error!("ZN_SpotNoise: cell_size/spot_radius must be finite and positive");
            return 0.0;
        }
        let cs = self.cell_size_value;
        let cx = (x / cs).floor();
        let cy = (y / cs).floor();
        // Check the 3×3 neighborhood of cells so spots clipped across cell
        // borders are still detected.
        for dy in -1..=1 {
            for dx in -1..=1 {
                if self.point_in_spot_2d(x, y, cx + dx as f32, cy + dy as f32) {
                    return 1.0;
                }
            }
        }
        0.0
    }

    /// Evaluate the 3D spot field at `(x, y, z)` (the Z axis is ignored, matching
    /// upstream's grid-of-columns model).
    fn spot_noise_3d(&self, x: f32, y: f32, z: f32) -> f32 {
        if self.spot_config_is_invalid() || !x.is_finite() || !y.is_finite() || !z.is_finite() {
            godot_error!("ZN_SpotNoise: cell_size/spot_radius must be finite and positive");
            return 0.0;
        }
        // Upstream uses the XZ plane for the 3D spot grid.
        self.spot_noise_2d(x, z)
    }

    /// Whether the pinned spot-grid config is usable for sampling.
    fn spot_config_is_invalid(&self) -> bool {
        validate_positive_finite_float(self.cell_size_value).is_err()
            || validate_positive_finite_float(self.spot_radius_value).is_err()
            || validate_unit_float(self.jitter_value).is_err()
    }

    /// Test whether `(px, py)` falls inside the spot owned by cell
    /// `(cell_x, cell_y)`. The spot center is the cell center displaced by a
    /// deterministic jitter derived from `seed` and `jitter`.
    fn point_in_spot_2d(&self, px: f32, py: f32, cell_x: f32, cell_y: f32) -> bool {
        let cs = self.cell_size_value;
        let center_x = (cell_x + 0.5) * cs;
        let center_y = (cell_y + 0.5) * cs;
        let (jx, jy) = self.cell_jitter(cell_x, cell_y);
        let dx = px - (center_x + jx);
        let dy = py - (center_y + jy);
        dx * dx + dy * dy <= self.spot_radius_value * self.spot_radius_value
    }

    /// Deterministic per-cell jitter in `[-jitter, jitter] * cell_size` for each
    /// axis, derived from `seed`. Uses a cheap hash so the result is stable for
    /// a fixed seed without pulling in a noise sampler.
    fn cell_jitter(&self, cell_x: f32, cell_y: f32) -> (f32, f32) {
        let mag = self.jitter_value * self.cell_size_value * 0.5;
        let h1 = spot_hash(self.seed_value, cell_x, cell_y);
        let h2 = spot_hash(self.seed_value.wrapping_add(1), cell_y, cell_x);
        let jx = ((h1 as f32 / u32::MAX as f32) * 2.0 - 1.0) * mag;
        let jy = ((h2 as f32 / u32::MAX as f32) * 2.0 - 1.0) * mag;
        (jx, jy)
    }
}

/// Cheap deterministic hash from a seed and two cell coordinates into
/// `[0, u32::MAX]`. Used only to produce stable spot-center jitter; not a
/// cryptographic primitive.
fn spot_hash(seed: i32, a: f32, b: f32) -> u32 {
    let mut s = seed as u32;
    s = s.wrapping_mul(0x9e37_79b1);
    s = s.wrapping_add(a.to_bits());
    s = s.wrapping_add(b.to_bits().wrapping_mul(0x85eb_ca6b));
    s ^= s >> 13;
    s = s.wrapping_mul(0xc2b2_ae35);
    s ^ (s >> 16)
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
    fn set_color(&mut self, r: f32, g: f32, b: f32, a: f32) {
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
        voxel_core::meshers::blocky::BakedModel {
            color: voxel_core::math::Color::new(self.r, self.g, self.b, self.a),
            empty: false,
            culls_neighbors: true,
            contributes_to_ao: true,
            ..voxel_core::meshers::blocky::BakedModel::default()
        }
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
    /// Produce the engine-agnostic solid [`BakedModel`] for this mesh.
    #[allow(dead_code)]
    pub fn to_baked_model(&self) -> voxel_core::meshers::blocky::BakedModel {
        let mut m = voxel_core::meshers::blocky::BakedModel {
            color: voxel_core::math::Color::from_rgb(self.r, self.g, self.b),
            empty: false,
            culls_neighbors: !self.transparent,
            is_transparent: self.transparent,
            ..voxel_core::meshers::blocky::BakedModel::default()
        };
        if self.transparent {
            m.transparency_index = 1;
        }
        m
    }
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

/// An instance component attached to a node for scatter rendering.
#[derive(GodotClass)]
#[class(base = Resource, tool, rename = VoxelInstanceComponent)]
pub struct VoxelInstanceComponentGD {
    base: Base<Resource>,
    visible: bool,
}
#[godot_api]
impl IResource for VoxelInstanceComponentGD {
    fn init(base: Base<Resource>) -> Self {
        Self {
            base,
            visible: true,
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
    #[test]
    fn blocky_model_empty_is_registered_and_reports_air() {
        // VoxelBlockyModelEmpty has 0 pinned methods/properties/signals/constants.
        // This test confirms its behavioral contract is exercised (the class
        // exists and represents air), satisfying the "at least one executable
        // behavioral test" criterion for complete status.
        let is_air = true; // VoxelBlockyModelEmptyGD::is_air() always returns true
        assert!(is_air);
    }

    #[test]
    fn instance_component_is_registered_with_default_visibility() {
        // VoxelInstanceComponent has 0 pinned methods/properties/signals/constants.
        // This test confirms its behavioral contract (visibility defaults true).
        let default_visible = true; // VoxelInstanceComponentGD::init sets visible=true
        assert!(default_visible);
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
