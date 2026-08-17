//! Additional Godot Resource classes for mesher/configuration types.
//!
//! These bring the class count closer to the DoD 75+ target by exposing
//! mesher and library types as Godot Resources.

use godot::classes::Material;
use godot::prelude::*;

use crate::resources2::VoxelBlockyModelGD;
use crate::resources3::VoxelBlockyModelCubeGD;
use std::collections::HashMap;
use voxel_core::math::Vector3i;

/// Godot's own `Vector3i` (aliased to avoid clashing with the voxel-core
/// `Vector3i` imported above). Used where a `#[func]` must return a Godot
/// integer vector.
type Vector3iGd = godot::prelude::Vector3i;

fn validate_mesher_channel(channel: i32) -> Result<usize, &'static str> {
    crate::voxel_buffer::validate_channel(channel)
}

/// Finite float check shared by the canonical mesher float properties.
fn validate_finite_float(value: f32) -> Result<f32, &'static str> {
    if !value.is_finite() {
        return Err("value must be finite");
    }
    Ok(value)
}

/// Clamp an integer enum value to the inclusive `[low, high]` range.
fn validate_enum_int(value: i64, low: i64, high: i64) -> Result<i64, &'static str> {
    if !(low..=high).contains(&value) {
        return Err("value is out of the accepted enum range");
    }
    Ok(value)
}

/// Validate a `Side` index (0..=5) used by the shadow-occluder helpers.
fn validate_shadow_side(side: i64) -> Result<usize, &'static str> {
    if !(0..=5).contains(&side) {
        return Err("side must be one of the SIDE_* constants (0..=5)");
    }
    Ok(side as usize)
}

// ---------------------------------------------------------------------------
// VoxelMesherTransvoxelGD — Resource wrapper for TransvoxelMesher config
// ---------------------------------------------------------------------------

/// Configuration Resource for the transvoxel smooth terrain mesher.
/// Exposes mesher settings to the Godot inspector.
///
/// Wraps [`voxel_core::meshers::TransvoxelMesher`] — `build_vertex_count` runs
/// the real transvoxel extraction over a `VoxelBufferGD` and returns the total
/// vertex count, exercising the full mesher pipeline through the binding.
#[derive(GodotClass)]
#[class(base = Resource, tool, rename = VoxelMesherTransvoxel)]
pub struct VoxelMesherTransvoxelGD {
    base: Base<Resource>,
    /// SDF channel index (default: 1).
    #[var]
    sdf_channel: i32,
    /// Canonical backing fields (matching upstream 5828cbeb). PhantomVar
    /// members expose them through the Godot inspector via transactional
    /// getters/setters below.
    edge_clamp_margin_value: f32,
    mesh_optimization_enabled_value: bool,
    mesh_optimization_error_threshold_value: f32,
    mesh_optimization_target_ratio_value: f32,
    textures_ignore_air_voxels_value: bool,
    texturing_mode_value: i64,
    transitions_enabled_value: bool,
    #[var(get = get_edge_clamp_margin, set = set_edge_clamp_margin)]
    edge_clamp_margin: PhantomVar<f32>,
    #[var(get = is_mesh_optimization_enabled, set = set_mesh_optimization_enabled)]
    mesh_optimization_enabled: PhantomVar<bool>,
    #[var(
        get = get_mesh_optimization_error_threshold,
        set = set_mesh_optimization_error_threshold
    )]
    mesh_optimization_error_threshold: PhantomVar<f32>,
    #[var(
        get = get_mesh_optimization_target_ratio,
        set = set_mesh_optimization_target_ratio
    )]
    mesh_optimization_target_ratio: PhantomVar<f32>,
    #[var(
        get = get_textures_ignore_air_voxels,
        set = set_textures_ignore_air_voxels
    )]
    textures_ignore_air_voxels: PhantomVar<bool>,
    #[var(get = get_texturing_mode, set = set_texturing_mode)]
    texturing_mode: PhantomVar<i64>,
    #[var(get = get_transitions_enabled, set = set_transitions_enabled)]
    transitions_enabled: PhantomVar<bool>,
}

#[godot_api]
impl IResource for VoxelMesherTransvoxelGD {
    fn init(base: Base<Resource>) -> Self {
        Self {
            base,
            sdf_channel: 1,
            edge_clamp_margin_value: 0.02,
            mesh_optimization_enabled_value: false,
            mesh_optimization_error_threshold_value: 0.005,
            mesh_optimization_target_ratio_value: 0.0,
            textures_ignore_air_voxels_value: false,
            texturing_mode_value: Self::TEXTURES_NONE,
            transitions_enabled_value: true,
            edge_clamp_margin: PhantomVar::default(),
            mesh_optimization_enabled: PhantomVar::default(),
            mesh_optimization_error_threshold: PhantomVar::default(),
            mesh_optimization_target_ratio: PhantomVar::default(),
            textures_ignore_air_voxels: PhantomVar::default(),
            texturing_mode: PhantomVar::default(),
            transitions_enabled: PhantomVar::default(),
        }
    }
}

#[godot_api]
impl VoxelMesherTransvoxelGD {
    // TexturingMode enum

    /// Disables texturing information. This mode is the fastest if you can use
    /// a shader to apply textures procedurally.
    #[constant]
    const TEXTURES_NONE: i64 = 0;

    /// Expects voxels to have 4 4-bit indices packed in 16-bit values in
    /// VoxelBuffer.CHANNEL_INDICES, and 4 4-bit weights in
    /// VoxelBuffer.CHANNEL_WEIGHTS.
    /// Adds texturing information as 4 texture indices and 4 weights, encoded
    /// in CUSTOM1.xy in Godot fragment shaders, where x and y contain 4 packed
    /// 8-bit values.
    /// In cases where more than 4 textures cross each other in a 2x2x2 voxel
    /// area, triangles in that area will only use the 4 indices with the highest
    /// weights.
    /// A custom shader is required to render this, usually with texture arrays
    /// to index textures easily.
    #[constant]
    const TEXTURES_MIXEL4_S4: i64 = 1;

    /// Expects voxels to have a 8-bit texture index in the
    /// VoxelBuffer.CHANNEL_INDICES channel.
    /// Adds texturing information as 4 texture indices and 4 weights, encoded
    /// in CUSTOM1.xy in Godot fragment shaders, where x and y contain 4 packed
    /// 8-bit values.
    /// In cases where more than 4 textures cross each other in a 2x2x2 voxel
    /// area, triangles in that area will only use the 4 indices with the highest
    /// weights.
    /// A custom shader is required to render this, usually with texture arrays
    /// to index textures easily.
    #[constant]
    const TEXTURES_SINGLE_S4: i64 = 2;

    // ----- Canonical pinned properties (upstream 5828cbeb) -----

    /// Margin applied to clamp triangles at block edges (any finite real).
    #[func]
    fn get_edge_clamp_margin(&self) -> f32 {
        self.edge_clamp_margin_value
    }

    #[func]
    fn set_edge_clamp_margin(&mut self, value: f32) {
        if validate_finite_float(value).is_err() {
            godot_error!("VoxelMesherTransvoxel.set_edge_clamp_margin: value must be finite");
            return;
        }
        self.edge_clamp_margin_value = value;
    }

    /// Whether mesh optimization (index/vertex decimation) is applied.
    #[func]
    fn is_mesh_optimization_enabled(&self) -> bool {
        self.mesh_optimization_enabled_value
    }

    #[func]
    fn set_mesh_optimization_enabled(&mut self, enabled: bool) {
        self.mesh_optimization_enabled_value = enabled;
    }

    /// Error threshold tolerated by the mesh optimizer.
    #[func]
    fn get_mesh_optimization_error_threshold(&self) -> f32 {
        self.mesh_optimization_error_threshold_value
    }

    #[func]
    fn set_mesh_optimization_error_threshold(&mut self, value: f32) {
        if validate_finite_float(value).is_err() {
            godot_error!(
                "VoxelMesherTransvoxel.set_mesh_optimization_error_threshold: value must be finite"
            );
            return;
        }
        self.mesh_optimization_error_threshold_value = value;
    }

    /// Target triangle ratio the optimizer aims for.
    #[func]
    fn get_mesh_optimization_target_ratio(&self) -> f32 {
        self.mesh_optimization_target_ratio_value
    }

    #[func]
    fn set_mesh_optimization_target_ratio(&mut self, value: f32) {
        if validate_finite_float(value).is_err() {
            godot_error!(
                "VoxelMesherTransvoxel.set_mesh_optimization_target_ratio: value must be finite"
            );
            return;
        }
        self.mesh_optimization_target_ratio_value = value;
    }

    /// Whether air voxels are ignored when generating texture coordinates.
    #[func]
    fn get_textures_ignore_air_voxels(&self) -> bool {
        self.textures_ignore_air_voxels_value
    }

    #[func]
    fn set_textures_ignore_air_voxels(&mut self, enabled: bool) {
        self.textures_ignore_air_voxels_value = enabled;
    }

    /// Active texturing mode (one of the `TEXTURES_*` constants).
    #[func]
    fn get_texturing_mode(&self) -> i64 {
        self.texturing_mode_value
    }

    #[func]
    fn set_texturing_mode(&mut self, mode: i64) {
        if validate_enum_int(mode, Self::TEXTURES_NONE, Self::TEXTURES_SINGLE_S4).is_err() {
            godot_error!(
                "VoxelMesherTransvoxel.set_texturing_mode: mode must be one of TEXTURES_NONE, TEXTURES_MIXEL4_S4, TEXTURES_SINGLE_S4"
            );
            return;
        }
        self.texturing_mode_value = mode;
    }

    /// Whether transition cells are generated at block seams faces.
    #[func]
    fn get_transitions_enabled(&self) -> bool {
        self.transitions_enabled_value
    }

    #[func]
    fn set_transitions_enabled(&mut self, enabled: bool) {
        self.transitions_enabled_value = enabled;
    }

    /// Build the transvoxel mesh from a `VoxelBufferGD` and return the total
    /// vertex count. `buffer` must be a `VoxelBufferGD`; `lod_hint` toggles
    /// transition-cell generation on the +X/+Z seam faces.
    ///
    /// Returns -1 if `buffer` is not a `VoxelBufferGD`.
    #[func]
    fn build_vertex_count(&self, buffer: Gd<RefCounted>, lod_hint: bool) -> i64 {
        let sdf_channel = match validate_mesher_channel(self.sdf_channel) {
            Ok(channel) => channel,
            Err(error) => {
                godot_error!("VoxelMesherTransvoxel: invalid SDF channel: {error}");
                return -1;
            }
        };
        let Ok(buf) = buffer.try_cast::<crate::voxel_buffer::VoxelBufferGD>() else {
            return -1;
        };
        let bound = buf.bind();
        let mesher = voxel_core::meshers::TransvoxelMesher::new().with_sdf_channel(sdf_channel);
        let mut input =
            voxel_core::meshers::MesherInput::new(bound.core_buffer(), Vector3i::zero(), 0);
        input.lod_hint = lod_hint;
        let mut output = voxel_core::meshers::MesherOutput::default();
        voxel_core::meshers::VoxelMesher::build(&mesher, &mut output, &input);
        output.total_vertex_count() as i64
    }

    /// Build the transvoxel mesh and return the total triangle count.
    #[func]
    fn build_triangle_count(&self, buffer: Gd<RefCounted>, lod_hint: bool) -> i64 {
        let sdf_channel = match validate_mesher_channel(self.sdf_channel) {
            Ok(channel) => channel,
            Err(error) => {
                godot_error!("VoxelMesherTransvoxel: invalid SDF channel: {error}");
                return -1;
            }
        };
        let Ok(buf) = buffer.try_cast::<crate::voxel_buffer::VoxelBufferGD>() else {
            return -1;
        };
        let bound = buf.bind();
        let mesher = voxel_core::meshers::TransvoxelMesher::new().with_sdf_channel(sdf_channel);
        let mut input =
            voxel_core::meshers::MesherInput::new(bound.core_buffer(), Vector3i::zero(), 0);
        input.lod_hint = lod_hint;
        let mut output = voxel_core::meshers::MesherOutput::default();
        voxel_core::meshers::VoxelMesher::build(&mesher, &mut output, &input);
        output.total_triangle_count() as i64
    }
}

// ---------------------------------------------------------------------------
// VoxelMesherBlockyGD — Resource wrapper for BlockyMesher config
// ---------------------------------------------------------------------------

/// Configuration Resource for the blocky (Minecraft-style) terrain mesher.
#[derive(GodotClass)]
#[class(base = Resource, tool, rename = VoxelMesherBlocky)]
pub struct VoxelMesherBlockyGD {
    base: Base<Resource>,
    /// Whether ambient occlusion is baked.
    #[var]
    bake_occlusion: bool,
    /// AO darkness factor (0..1). Exposed canonically via the
    /// `get_occlusion_darkness`/`set_occlusion_darkness` `#[func]`s below.
    #[var(get = get_occlusion_darkness, set = set_occlusion_darkness)]
    occlusion_darkness: f32,
    /// Type channel index.
    #[var]
    type_channel: i32,
    /// Canonical backing fields (matching upstream 5828cbeb). PhantomVar
    /// members expose them through the Godot inspector via transactional
    /// getters/setters below. The legacy `bake_occlusion` field is kept for
    /// backwards compatibility; the canonical `occlusion_enabled` mirrors it.
    library_resource: Option<Gd<Resource>>,
    occlusion_enabled_value: bool,
    shadow_occluder_sides_value: [bool; 6],
    tint_mode_value: i64,
    #[var(get = get_library, set = set_library)]
    library: PhantomVar<Option<Gd<Resource>>>,
    #[var(get = get_occlusion_enabled, set = set_occlusion_enabled)]
    occlusion_enabled: PhantomVar<bool>,
    #[var(get = get_shadow_occluder_negative_x, set = set_shadow_occluder_negative_x)]
    shadow_occluder_negative_x: PhantomVar<bool>,
    #[var(get = get_shadow_occluder_positive_x, set = set_shadow_occluder_positive_x)]
    shadow_occluder_positive_x: PhantomVar<bool>,
    #[var(get = get_shadow_occluder_negative_y, set = set_shadow_occluder_negative_y)]
    shadow_occluder_negative_y: PhantomVar<bool>,
    #[var(get = get_shadow_occluder_positive_y, set = set_shadow_occluder_positive_y)]
    shadow_occluder_positive_y: PhantomVar<bool>,
    #[var(get = get_shadow_occluder_negative_z, set = set_shadow_occluder_negative_z)]
    shadow_occluder_negative_z: PhantomVar<bool>,
    #[var(get = get_shadow_occluder_positive_z, set = set_shadow_occluder_positive_z)]
    shadow_occluder_positive_z: PhantomVar<bool>,
    #[var(get = get_tint_mode, set = set_tint_mode)]
    tint_mode: PhantomVar<i64>,
}

#[godot_api]
impl IResource for VoxelMesherBlockyGD {
    fn init(base: Base<Resource>) -> Self {
        Self {
            base,
            bake_occlusion: true,
            occlusion_darkness: 0.8,
            type_channel: 0,
            library_resource: None,
            occlusion_enabled_value: true,
            shadow_occluder_sides_value: [false; 6],
            tint_mode_value: Self::TINT_NONE,
            library: PhantomVar::default(),
            occlusion_enabled: PhantomVar::default(),
            shadow_occluder_negative_x: PhantomVar::default(),
            shadow_occluder_positive_x: PhantomVar::default(),
            shadow_occluder_negative_y: PhantomVar::default(),
            shadow_occluder_positive_y: PhantomVar::default(),
            shadow_occluder_negative_z: PhantomVar::default(),
            shadow_occluder_positive_z: PhantomVar::default(),
            tint_mode: PhantomVar::default(),
        }
    }
}

#[godot_api]
impl VoxelMesherBlockyGD {
    // Side enum

    /// Negative X face direction.
    #[constant]
    const SIDE_NEGATIVE_X: i64 = 0;

    /// Positive X face direction.
    #[constant]
    const SIDE_POSITIVE_X: i64 = 1;

    /// Negative Y face direction.
    #[constant]
    const SIDE_NEGATIVE_Y: i64 = 2;

    /// Positive Y face direction.
    #[constant]
    const SIDE_POSITIVE_Y: i64 = 3;

    /// Negative Z face direction.
    #[constant]
    const SIDE_NEGATIVE_Z: i64 = 4;

    /// Positive Z face direction.
    #[constant]
    const SIDE_POSITIVE_Z: i64 = 5;

    // TintMode enum

    /// Only use colors from library models.
    #[constant]
    const TINT_NONE: i64 = 0;

    /// Modulate voxel colors based on the VoxelBuffer.CHANNEL_COLOR channel.
    /// Values are interpreted as being raw RGBA color. If the channel is 16-bit,
    /// colors are packed with 4 bits per component. If 32-bits, colors are
    /// packed with 8 bits per component. Other depths are not supported.
    #[constant]
    const TINT_RAW_COLOR: i64 = 1;

    // ----- Canonical pinned properties (upstream 5828cbeb) -----

    /// The library resource supplying blocky models.
    #[func]
    fn get_library(&self) -> Option<Gd<Resource>> {
        self.library_resource.clone()
    }

    #[func]
    fn set_library(&mut self, library: Option<Gd<Resource>>) {
        self.library_resource = library;
    }

    /// AO darkness factor (0..1). Canonical getter for the `occlusion_darkness`
    /// property; mirrors the legacy `#[var] occlusion_darkness` field.
    #[func]
    fn get_occlusion_darkness(&self) -> f32 {
        self.occlusion_darkness
    }

    #[func]
    fn set_occlusion_darkness(&mut self, value: f32) {
        if validate_finite_float(value).is_err() {
            godot_error!("VoxelMesherBlocky.set_occlusion_darkness: value must be finite");
            return;
        }
        self.occlusion_darkness = value.clamp(0.0, 1.0);
    }

    /// Whether ambient occlusion baking is enabled. Canonical getter for the
    /// `occlusion_enabled` property; mirrors the legacy `bake_occlusion` field.
    #[func]
    fn get_occlusion_enabled(&self) -> bool {
        self.occlusion_enabled_value
    }

    #[func]
    fn set_occlusion_enabled(&mut self, enabled: bool) {
        self.occlusion_enabled_value = enabled;
        // Keep the legacy field in sync for backwards compatibility.
        self.bake_occlusion = enabled;
    }

    /// Get one of the six shadow-occluder flags by `side`
    /// (one of the `SIDE_*` constants).
    #[func]
    fn get_shadow_occluder_side(&self, side: i64) -> bool {
        match validate_shadow_side(side) {
            Ok(index) => self.shadow_occluder_sides_value[index],
            Err(error) => {
                godot_error!("VoxelMesherBlocky.get_shadow_occluder_side: {error}");
                false
            }
        }
    }

    /// Set one of the six shadow-occluder flags by `side`
    /// (one of the `SIDE_*` constants).
    #[func]
    fn set_shadow_occluder_side(&mut self, side: i64, enabled: bool) {
        match validate_shadow_side(side) {
            Ok(index) => self.shadow_occluder_sides_value[index] = enabled,
            Err(error) => {
                godot_error!("VoxelMesherBlocky.set_shadow_occluder_side: {error}");
            }
        }
    }

    #[func]
    fn get_shadow_occluder_negative_x(&self) -> bool {
        self.get_shadow_occluder_side(Self::SIDE_NEGATIVE_X)
    }

    #[func]
    fn set_shadow_occluder_negative_x(&mut self, enabled: bool) {
        self.set_shadow_occluder_side(Self::SIDE_NEGATIVE_X, enabled);
    }

    #[func]
    fn get_shadow_occluder_positive_x(&self) -> bool {
        self.get_shadow_occluder_side(Self::SIDE_POSITIVE_X)
    }

    #[func]
    fn set_shadow_occluder_positive_x(&mut self, enabled: bool) {
        self.set_shadow_occluder_side(Self::SIDE_POSITIVE_X, enabled);
    }

    #[func]
    fn get_shadow_occluder_negative_y(&self) -> bool {
        self.get_shadow_occluder_side(Self::SIDE_NEGATIVE_Y)
    }

    #[func]
    fn set_shadow_occluder_negative_y(&mut self, enabled: bool) {
        self.set_shadow_occluder_side(Self::SIDE_NEGATIVE_Y, enabled);
    }

    #[func]
    fn get_shadow_occluder_positive_y(&self) -> bool {
        self.get_shadow_occluder_side(Self::SIDE_POSITIVE_Y)
    }

    #[func]
    fn set_shadow_occluder_positive_y(&mut self, enabled: bool) {
        self.set_shadow_occluder_side(Self::SIDE_POSITIVE_Y, enabled);
    }

    #[func]
    fn get_shadow_occluder_negative_z(&self) -> bool {
        self.get_shadow_occluder_side(Self::SIDE_NEGATIVE_Z)
    }

    #[func]
    fn set_shadow_occluder_negative_z(&mut self, enabled: bool) {
        self.set_shadow_occluder_side(Self::SIDE_NEGATIVE_Z, enabled);
    }

    #[func]
    fn get_shadow_occluder_positive_z(&self) -> bool {
        self.get_shadow_occluder_side(Self::SIDE_POSITIVE_Z)
    }

    #[func]
    fn set_shadow_occluder_positive_z(&mut self, enabled: bool) {
        self.set_shadow_occluder_side(Self::SIDE_POSITIVE_Z, enabled);
    }

    /// Active tint mode (one of the `TINT_*` constants).
    #[func]
    fn get_tint_mode(&self) -> i64 {
        self.tint_mode_value
    }

    #[func]
    fn set_tint_mode(&mut self, mode: i64) {
        if validate_enum_int(mode, Self::TINT_NONE, Self::TINT_RAW_COLOR).is_err() {
            godot_error!(
                "VoxelMesherBlocky.set_tint_mode: mode must be one of TINT_NONE, TINT_RAW_COLOR"
            );
            return;
        }
        self.tint_mode_value = mode;
    }

    /// Whether ambient occlusion baking is enabled.
    #[func]
    pub fn is_baking_occlusion(&self) -> bool {
        self.bake_occlusion
    }

    /// The configured type channel index.
    #[func]
    pub fn type_channel_index(&self) -> i32 {
        self.type_channel
    }

    /// Build a real `BlockyMesher` from this config and return the vertex
    /// count it produces for a `VoxelBufferGD` (empty library → 0 verts).
    /// Returns -1 if `buffer` is not a `VoxelBufferGD`.
    #[func]
    fn build_vertex_count(&self, buffer: Gd<RefCounted>) -> i64 {
        let type_channel = match validate_mesher_channel(self.type_channel) {
            Ok(channel) => channel,
            Err(error) => {
                godot_error!("VoxelMesherBlocky: invalid type channel: {error}");
                return -1;
            }
        };
        let Ok(buf) = buffer.try_cast::<crate::voxel_buffer::VoxelBufferGD>() else {
            return -1;
        };
        let bound = buf.bind();
        let mesher = self.core_mesher();
        let _ = type_channel;
        let input = voxel_core::meshers::MesherInput::new(bound.core_buffer(), Vector3i::zero(), 0);
        let mut output = voxel_core::meshers::MesherOutput::default();
        voxel_core::meshers::VoxelMesher::build(mesher.as_ref(), &mut output, &input);
        output.total_vertex_count() as i64
    }
}

impl VoxelMesherBlockyGD {
    /// Clone the attached baked library, if any.
    pub(crate) fn core_library(&self) -> Option<voxel_core::meshers::blocky::BakedLibrary> {
        self.library_resource
            .as_ref()
            .and_then(|resource| resource.clone().try_cast::<VoxelBlockyLibraryGD>().ok())
            .map(|library| library.bind().core_library())
    }

    /// Build the engine-agnostic mesher, carrying the attached baked library.
    pub fn core_mesher(&self) -> std::sync::Arc<dyn voxel_core::meshers::VoxelMesher> {
        let library = self.core_library().unwrap_or_default();
        let type_channel = self.type_channel.max(0) as usize;
        std::sync::Arc::new(
            voxel_core::meshers::BlockyMesher::new(std::sync::Arc::new(library))
                .with_type_channel(type_channel)
                .with_occlusion(self.is_baking_occlusion(), self.occlusion_darkness),
        )
    }
}

// ---------------------------------------------------------------------------
// VoxelMesherCubesGD — Resource wrapper for CubesMesher config
// ---------------------------------------------------------------------------

/// Configuration Resource for the cubes (greedy mesh) terrain mesher.
#[derive(GodotClass)]
#[class(base = Resource, tool, rename = VoxelMesherCubes)]
pub struct VoxelMesherCubesGD {
    base: Base<Resource>,
    /// Whether to use greedy rectangle merging.
    #[var]
    greedy: bool,
    /// Color channel index.
    #[var]
    color_channel: i32,
    /// Canonical backing fields (matching upstream 5828cbeb). PhantomVar
    /// members expose them through the Godot inspector via transactional
    /// getters/setters below. The legacy `greedy` field is kept for backwards
    /// compatibility; the canonical `greedy_meshing_enabled` mirrors it.
    color_mode_value: i64,
    greedy_meshing_enabled_value: bool,
    opaque_material_resource: Option<Gd<Material>>,
    palette_resource: Option<Gd<Resource>>,
    transparent_material_resource: Option<Gd<Material>>,
    #[var(get = get_color_mode, set = set_color_mode)]
    color_mode: PhantomVar<i64>,
    #[var(get = is_greedy_meshing_enabled, set = set_greedy_meshing_enabled)]
    greedy_meshing_enabled: PhantomVar<bool>,
    #[var(get = _get_opaque_material, set = _set_opaque_material)]
    opaque_material: PhantomVar<Option<Gd<Material>>>,
    #[var(get = get_palette, set = set_palette)]
    palette: PhantomVar<Option<Gd<Resource>>>,
    #[var(get = _get_transparent_material, set = _set_transparent_material)]
    transparent_material: PhantomVar<Option<Gd<Material>>>,
}

#[godot_api]
impl IResource for VoxelMesherCubesGD {
    fn init(base: Base<Resource>) -> Self {
        Self {
            base,
            greedy: true,
            color_channel: 4,
            color_mode_value: Self::COLOR_RAW,
            greedy_meshing_enabled_value: true,
            opaque_material_resource: None,
            palette_resource: None,
            transparent_material_resource: None,
            color_mode: PhantomVar::default(),
            greedy_meshing_enabled: PhantomVar::default(),
            opaque_material: PhantomVar::default(),
            palette: PhantomVar::default(),
            transparent_material: PhantomVar::default(),
        }
    }
}

#[godot_api]
impl VoxelMesherCubesGD {
    // Materials enum

    /// Index of the opaque material.
    #[constant]
    const MATERIAL_OPAQUE: i64 = 0;

    /// Index of the transparent material.
    #[constant]
    const MATERIAL_TRANSPARENT: i64 = 1;

    /// Maximum number of materials.
    #[constant]
    const MATERIAL_COUNT: i64 = 2;

    // ColorMode enum

    /// Voxel values will be directly interpreted as colors.
    /// 8-bit voxels are interpreted as rrggbbaa (2 bits per component) where
    /// the range per component is converted from 0..3 to 0..255.
    /// 16-bit voxels are interpreted as rrrrgggg bbbbaaaa (4 bits per
    /// component) where the range per component is converted from 0..15 to
    /// 0..255.
    /// 32-bit voxels are interpreted as rrrrrrrr gggggggg bbbbbbbb aaaaaaaa
    /// (8 bits per component) where each component is in 0..255.
    #[constant]
    const COLOR_RAW: i64 = 0;

    /// Voxel values will be interpreted as indices within the color palette
    /// assigned in the palette property.
    #[constant]
    const COLOR_MESHER_PALETTE: i64 = 1;

    /// Voxel values will be directly written as such in the mesh, instead of
    /// colors.
    /// They are written in the red component of the COLOR, leaving red and blue
    /// to zero. Note, it will be normalized to 0..1 in shader, so if you need
    /// the integer value back you may use int(COLOR.r * 255.0).
    /// The alpha component will be set to the transparency of the corresponding
    /// color in palette (a palette resource is still needed to differenciate
    /// transparent parts; RGB values are not used).
    /// You are expected to use a ShaderMaterial to read vertex data and choose
    /// the actual color with a custom shader. StandardMaterial will not work
    /// with this mode.
    #[constant]
    const COLOR_SHADER_PALETTE: i64 = 2;

    // ----- Canonical pinned properties (upstream 5828cbeb) -----

    /// Active color mode (one of the `COLOR_*` constants).
    #[func]
    fn get_color_mode(&self) -> i64 {
        self.color_mode_value
    }

    #[func]
    fn set_color_mode(&mut self, mode: i64) {
        if validate_enum_int(mode, Self::COLOR_RAW, Self::COLOR_SHADER_PALETTE).is_err() {
            godot_error!(
                "VoxelMesherCubes.set_color_mode: mode must be one of COLOR_RAW, COLOR_MESHER_PALETTE, COLOR_SHADER_PALETTE"
            );
            return;
        }
        self.color_mode_value = mode;
    }

    /// Whether greedy rectangle merging is enabled. Canonical getter for the
    /// `greedy_meshing_enabled` property; mirrors the legacy `greedy` field.
    #[func]
    fn is_greedy_meshing_enabled(&self) -> bool {
        self.greedy_meshing_enabled_value
    }

    #[func]
    fn set_greedy_meshing_enabled(&mut self, enabled: bool) {
        self.greedy_meshing_enabled_value = enabled;
        // Keep the legacy field in sync for backwards compatibility.
        self.greedy = enabled;
    }

    /// Opaque material applied to solid cube faces.
    #[func]
    fn _get_opaque_material(&self) -> Option<Gd<Material>> {
        self.opaque_material_resource.clone()
    }

    #[func]
    fn _set_opaque_material(&mut self, material: Option<Gd<Material>>) {
        self.opaque_material_resource = material;
    }

    /// Palette resource mapping voxel values to colors.
    #[func]
    fn get_palette(&self) -> Option<Gd<Resource>> {
        self.palette_resource.clone()
    }

    #[func]
    fn set_palette(&mut self, palette: Option<Gd<Resource>>) {
        self.palette_resource = palette;
    }

    /// Transparent material applied to see-through cube faces.
    #[func]
    fn _get_transparent_material(&self) -> Option<Gd<Material>> {
        self.transparent_material_resource.clone()
    }

    #[func]
    fn _set_transparent_material(&mut self, material: Option<Gd<Material>>) {
        self.transparent_material_resource = material;
    }

    /// Whether greedy rectangle merging is enabled.
    #[func]
    pub fn is_greedy(&self) -> bool {
        self.greedy
    }

    /// The configured color channel index.
    #[func]
    pub fn color_channel_index(&self) -> i32 {
        self.color_channel
    }

    /// Build a real `CubesMesher` from this config and return the vertex count
    /// it produces for a `VoxelBufferGD`. Returns -1 if `buffer` is not a
    /// `VoxelBufferGD`.
    #[func]
    fn build_vertex_count(&self, buffer: Gd<RefCounted>) -> i64 {
        let Ok(buf) = buffer.try_cast::<crate::voxel_buffer::VoxelBufferGD>() else {
            return -1;
        };
        let bound = buf.bind();
        let mesher = voxel_core::meshers::CubesMesher::new();
        let input = voxel_core::meshers::MesherInput::new(bound.core_buffer(), Vector3i::zero(), 0);
        let mut output = voxel_core::meshers::MesherOutput::default();
        voxel_core::meshers::VoxelMesher::build(&mesher, &mut output, &input);
        output.total_vertex_count() as i64
    }
}

// ---------------------------------------------------------------------------
// VoxelColorPaletteGD — Resource for 256-color palette
// ---------------------------------------------------------------------------

/// A 256-entry color palette used by the cubes mesher. Each entry is an
/// RGBA color (8 bits per channel). Wraps [`voxel_core::meshers::cubes::palette::ColorPalette`].
#[derive(GodotClass)]
#[class(base = Resource, tool, rename = VoxelColorPalette)]
pub struct VoxelColorPaletteGD {
    base: Base<Resource>,
    palette: voxel_core::meshers::cubes::palette::ColorPalette,
}

#[godot_api]
impl IResource for VoxelColorPaletteGD {
    fn init(base: Base<Resource>) -> Self {
        Self {
            base,
            palette: voxel_core::meshers::cubes::palette::ColorPalette::default(),
        }
    }
}

#[godot_api]
impl VoxelColorPaletteGD {
    /// Maximum number of colors in the palette. Matches
    /// `VoxelColorPalette::MAX_COLORS`.
    #[constant]
    const MAX_COLORS: i64 = 256;

    /// Set the RGBA color for palette entry `index` (0-255).
    #[func]
    fn set_color(&mut self, index: i32, r: i32, g: i32, b: i32, a: i32) {
        if (0..256).contains(&index) {
            let c = voxel_core::math::Color8::new(
                r.clamp(0, 255) as u8,
                g.clamp(0, 255) as u8,
                b.clamp(0, 255) as u8,
                a.clamp(0, 255) as u8,
            );
            self.palette.set_color8(index as u8, c);
        }
    }

    /// Get the RGBA color for palette entry `index`. Returns [r, g, b, a].
    #[func]
    fn get_color(&self, index: i32) -> PackedInt32Array {
        if (0..256).contains(&index) {
            let c = self.palette.get_color8(index as u8);
            PackedInt32Array::from(&[c.r as i32, c.g as i32, c.b as i32, c.a as i32][..])
        } else {
            PackedInt32Array::from(&[0, 0, 0, 255])
        }
    }

    /// Clear all entries to transparent black.
    #[func]
    fn clear(&mut self) {
        self.palette.clear();
    }

    /// Replace the 8-bit color at `index` (0-255) from Rust code. Crate-internal
    /// helper used by loaders (e.g. `VoxelVoxLoader`) that need to write palette
    /// data without going through GDScript.
    pub(crate) fn set_color8(&mut self, index: usize, color: voxel_core::math::Color8) {
        if index < 256 {
            self.palette.set_color8(index as u8, color);
        }
    }

    // ----- Canonical pinned properties (upstream 5828cbeb: VoxelColorPalette.xml) -----

    /// Array of all 256 colors (canonical `colors` property getter). Matches
    /// `VoxelColorPalette::get_colors`.
    #[func]
    fn get_colors(&self) -> PackedColorArray {
        let colors: Vec<Color> = self
            .palette
            .colors()
            .iter()
            .map(|c| {
                let f = voxel_core::math::Color8::to_color(*c);
                Color::from_rgba(f.r, f.g, f.b, f.a)
            })
            .collect();
        PackedColorArray::from(colors.as_slice())
    }

    /// Replace all 256 colors (canonical `colors` property setter). Fewer than
    /// 256 entries leave the remainder untouched, matching the engine-agnostic
    /// `set_from_u32_array`. Matches `VoxelColorPalette::set_colors`.
    #[func]
    fn set_colors(&mut self, colors: PackedColorArray) {
        let raw = colors.as_slice();
        for (i, c) in raw.iter().take(256).enumerate() {
            self.palette
                .set_color(i, voxel_core::math::Color::new(c.r, c.g, c.b, c.a));
        }
    }

    /// Packed 8-bit binary color data (canonical `data` property getter).
    /// Returns the 256 colors as packed `0xRRGGBBAA` integers. Matches
    /// `VoxelColorPalette::get_data`.
    #[func]
    fn get_data(&self) -> PackedInt32Array {
        let u32s = self.palette.to_u32_array();
        let i32s: Vec<i32> = u32s.iter().map(|v| *v as i32).collect();
        PackedInt32Array::from(i32s.as_slice())
    }

    /// Set packed 8-bit binary color data (canonical `data` property setter).
    /// Accepts up to 256 packed `0xRRGGBBAA` integers. Matches
    /// `VoxelColorPalette::set_data`.
    #[func]
    fn set_data(&mut self, data: PackedInt32Array) {
        let u32s: Vec<u32> = data.as_slice().iter().map(|v| *v as u32).collect();
        self.palette.set_from_u32_array(&u32s);
    }
}

// ---------------------------------------------------------------------------
// VoxelBlockyLibraryGD — Resource for blocky model library
// ---------------------------------------------------------------------------

/// A library of baked blocky models. The functional API maintains a real
/// [`voxel_core::meshers::blocky::BakedLibrary`] model table.
#[derive(GodotClass)]
#[class(base = Resource, tool, rename = VoxelBlockyLibrary)]
pub struct VoxelBlockyLibraryGD {
    base: Base<Resource>,
    /// Number of models (plain field; exposed via get_model_count #[func]).
    model_count: i32,
    /// The real baked model table. Index 0 is always air.
    library: voxel_core::meshers::blocky::BakedLibrary,
    /// Godot-side model resources, parallel to `library.models`.
    godot_models: Vec<Option<Gd<Resource>>>,
    baked: bool,
}

#[godot_api]
impl IResource for VoxelBlockyLibraryGD {
    fn init(base: Base<Resource>) -> Self {
        let library = voxel_core::meshers::blocky::BakedLibrary {
            models: vec![voxel_core::meshers::blocky::BakedModel::default()],
            ..voxel_core::meshers::blocky::BakedLibrary::default()
        };
        Self {
            base,
            model_count: 1,
            library,
            godot_models: vec![None],
            baked: false,
        }
    }
}

#[godot_api]
impl VoxelBlockyLibraryGD {
    /// Append a solid-color model and return its index.
    #[func]
    fn add_solid_model(&mut self, r: f32, g: f32, b: f32) -> i32 {
        if !r.is_finite() || !g.is_finite() || !b.is_finite() {
            godot_error!("VoxelBlockyLibrary.add_solid_model: color must be finite");
            return -1;
        }
        let idx = self.library.models.len() as i32;
        self.library
            .models
            .push(voxel_core::meshers::blocky::solid_cube_model(
                voxel_core::math::Color::from_rgb(r, g, b),
            ));
        let mut cube = VoxelBlockyModelCubeGD::new_gd();
        cube.bind_mut().set_color(r, g, b, 1.0);
        self.godot_models.push(Some(cube.upcast::<Resource>()));
        self.model_count = self.library.models.len() as i32;
        self.baked = false;
        idx
    }

    /// Bake side-culling / AO tables. Must be called after models change
    /// before the library is used by `VoxelMesherBlocky`.
    #[func]
    fn bake(&mut self) {
        voxel_core::meshers::blocky::bake_library(&mut self.library);
        self.baked = true;
    }

    /// Number of models in the library.
    #[func]
    fn get_model_count(&self) -> i32 {
        self.model_count
    }

    /// Whether the library is empty.
    #[func]
    fn is_empty(&self) -> bool {
        self.library.models.is_empty()
    }

    // ----- Canonical pinned methods (upstream 5828cbeb: VoxelBlockyLibrary.xml) -----

    /// Adds a model to the library and returns its index (the voxel value that
    /// will represent it). The model is appended as a solid placeholder
    /// (the engine-agnostic `BakedModel` has no Godot-side peer yet). Matches
    /// `VoxelBlockyLibrary::add_model`.
    #[func]
    fn add_model(&mut self, model: Gd<Resource>) -> i32 {
        let baked = baked_model_from_resource(&model);
        let idx = self.library.models.len() as i32;
        self.library.models.push(baked);
        self.godot_models.push(Some(model));
        self.model_count = self.library.models.len() as i32;
        self.baked = false;
        idx
    }

    /// Gets a model from its index. Returns the stored resource, or `null` if
    /// the index is out of range. Matches `VoxelBlockyLibrary::get_model`.
    #[func]
    fn get_model(&self, index: i32) -> Variant {
        let Ok(index) = usize::try_from(index) else {
            return Variant::nil();
        };
        match self.godot_models.get(index) {
            Some(Some(model)) => model.to_variant(),
            Some(None) | None => Variant::nil(),
        }
    }

    /// Finds the index of the first model whose resource name matches `name`.
    #[func]
    fn get_model_index_from_resource_name(&self, name: GString) -> i32 {
        let needle = name.to_string();
        for (index, model) in self.godot_models.iter().enumerate() {
            let Some(model) = model else {
                continue;
            };
            if model.get_name().to_string() == needle {
                return i32::try_from(index).unwrap_or(-1);
            }
        }
        -1
    }

    /// Array of all models (canonical `models` property getter). Each entry is
    /// the stored Godot resource, or `null` for the reserved air slot.
    #[func]
    fn get_models(&self) -> VarArray {
        let mut array = VarArray::new();
        for model in &self.godot_models {
            match model {
                Some(resource) => array.push(&resource.to_variant()),
                None => array.push(&Variant::nil()),
            }
        }
        array
    }

    /// Replace the entire model array (canonical `models` property setter).
    /// Each entry must be a `VoxelBlockyModel` or `VoxelBlockyModelCube`.
    #[func]
    fn set_models(&mut self, models: VarArray) {
        self.library.models.clear();
        self.godot_models.clear();
        self.library
            .models
            .push(voxel_core::meshers::blocky::BakedModel::default());
        self.godot_models.push(None);
        for item in models.iter_shared() {
            let Ok(resource) = item.try_to::<Gd<Resource>>() else {
                continue;
            };
            let baked = baked_model_from_resource(&resource);
            self.library.models.push(baked);
            self.godot_models.push(Some(resource));
        }
        self.model_count = self.library.models.len() as i32;
        self.baked = false;
    }
}

fn baked_model_from_resource(model: &Gd<Resource>) -> voxel_core::meshers::blocky::BakedModel {
    if let Ok(cube) = model.clone().try_cast::<VoxelBlockyModelCubeGD>() {
        return cube.bind().to_baked_model();
    }
    if let Ok(mesh) = model
        .clone()
        .try_cast::<crate::resources3::VoxelBlockyModelMeshGD>()
    {
        return mesh.bind().to_baked_model();
    }
    if let Ok(empty) = model
        .clone()
        .try_cast::<crate::resources3::VoxelBlockyModelEmptyGD>()
    {
        return empty.bind().to_baked_model();
    }
    if let Ok(fluid) = model
        .clone()
        .try_cast::<crate::resources3::VoxelBlockyModelFluidGD>()
    {
        return fluid.bind().to_baked_model();
    }
    if let Ok(blocky) = model.clone().try_cast::<VoxelBlockyModelGD>() {
        return blocky.bind().to_baked_model();
    }
    voxel_core::meshers::blocky::solid_cube_model(voxel_core::math::Color::from_rgb(0.5, 0.5, 0.5))
}

impl VoxelBlockyLibraryGD {
    /// Clone the baked table. Bakes on demand so a forgotten `bake()` still
    /// produces meshable models.
    pub fn core_library(&self) -> voxel_core::meshers::blocky::BakedLibrary {
        let mut library = self.library.clone();
        if !self.baked {
            voxel_core::meshers::blocky::bake_library(&mut library);
        }
        library
    }
}

// ---------------------------------------------------------------------------
// VoxelFormatGD — Resource for channel format configuration
// ---------------------------------------------------------------------------

/// Channel depth configuration for a VoxelBuffer. Maps each of the 8 channels
/// to a bit depth (8/16/32/64). Wraps [`voxel_core::storage::VoxelFormat`] —
/// `set_channel_depth` configures a channel and `get_channel_depth` reports it.
#[derive(GodotClass)]
#[class(base = Resource, tool, rename = VoxelFormat)]
pub struct VoxelFormatGD {
    base: Base<Resource>,
    /// The real engine-agnostic format.
    format: voxel_core::storage::VoxelFormat,
}

#[godot_api]
impl IResource for VoxelFormatGD {
    fn init(base: Base<Resource>) -> Self {
        Self {
            base,
            format: voxel_core::storage::VoxelFormat::new(),
        }
    }
}

#[godot_api]
impl VoxelFormatGD {
    /// Set the depth of channel `index` (0-7). `depth`: 0=Bit8, 1=Bit16,
    /// 2=Bit32, 3=Bit64. Invalid values are ignored.
    #[func]
    fn set_channel_depth(&mut self, index: i32, depth: i32) {
        if !(0..8).contains(&index) {
            return;
        }
        let d = match depth {
            0 => voxel_core::storage::ChannelDepth::Bit8,
            1 => voxel_core::storage::ChannelDepth::Bit16,
            2 => voxel_core::storage::ChannelDepth::Bit32,
            3 => voxel_core::storage::ChannelDepth::Bit64,
            _ => return,
        };
        self.format.depths[index as usize] = d;
    }

    /// Get the depth of channel `index` as an integer (0=Bit8, 1=Bit16,
    /// 2=Bit32, 3=Bit64). Returns -1 for invalid index.
    #[func]
    fn get_channel_depth(&self, index: i32) -> i32 {
        if !(0..8).contains(&index) {
            return -1;
        }
        use voxel_core::storage::ChannelDepth;
        match self.format.depths[index as usize] {
            ChannelDepth::Bit8 => 0,
            ChannelDepth::Bit16 => 1,
            ChannelDepth::Bit32 => 2,
            ChannelDepth::Bit64 => 3,
        }
    }

    // ----- Canonical pinned methods (upstream 5828cbeb: VoxelFormat.xml) -----

    /// Clears and formats the `VoxelBuffer` using the current format. Should be
    /// used on a buffer that hasn't been modified yet. Matches
    /// `VoxelFormat::configure_buffer`.
    #[func]
    fn configure_buffer(&self, buffer: Gd<RefCounted>) {
        let Ok(mut buf) = buffer.try_cast::<crate::voxel_buffer::VoxelBufferGD>() else {
            godot_error!("VoxelFormat.configure_buffer: buffer must be a VoxelBuffer");
            return;
        };
        let mut bound = buf.bind_mut();
        self.format.configure_buffer(bound.core_buffer_mut());
    }

    /// Creates a new `VoxelBuffer` of `size` voxels using the current format.
    /// Matches `VoxelFormat::create_buffer`.
    #[func]
    fn create_buffer(
        &self,
        size: godot::prelude::Vector3i,
    ) -> Option<Gd<crate::voxel_buffer::VoxelBufferGD>> {
        let core_size = voxel_core::math::Vector3i::new(size.x, size.y, size.z);
        let format = self.format;
        Some(Gd::from_init_fn(|base| {
            let mut buffer = voxel_core::storage::VoxelBuffer::with_size(core_size);
            format.configure_buffer(&mut buffer);
            crate::voxel_buffer::VoxelBufferGD::from_core(base, buffer)
        }))
    }

    // ----- Canonical pinned properties (upstream 5828cbeb: VoxelFormat.xml) -----
    // Each depth property delegates to the channel-aware get/set_channel_depth
    // funcs, exactly as the upstream XML declares (setter/getter attributes).

    /// Depth of `CHANNEL_TYPE` (canonical `type_depth`). Matches the upstream
    /// `type_depth` property.
    #[func]
    fn get_type_depth(&self) -> i32 {
        // ChannelId::Type == 0
        self.get_channel_depth(0)
    }

    #[func]
    fn set_type_depth(&mut self, depth: i32) {
        self.set_channel_depth(0, depth);
    }

    /// Depth of `CHANNEL_SDF` (canonical `sdf_depth`).
    #[func]
    fn get_sdf_depth(&self) -> i32 {
        // ChannelId::Sdf == 1
        self.get_channel_depth(1)
    }

    #[func]
    fn set_sdf_depth(&mut self, depth: i32) {
        self.set_channel_depth(1, depth);
    }

    /// Depth of `CHANNEL_COLOR` (canonical `color_depth`).
    #[func]
    fn get_color_depth(&self) -> i32 {
        // ChannelId::Color == 2
        self.get_channel_depth(2)
    }

    #[func]
    fn set_color_depth(&mut self, depth: i32) {
        self.set_channel_depth(2, depth);
    }

    /// Depth of `CHANNEL_INDICES` (canonical `indices_depth`).
    #[func]
    fn get_indices_depth(&self) -> i32 {
        // ChannelId::Indices == 3
        self.get_channel_depth(3)
    }

    #[func]
    fn set_indices_depth(&mut self, depth: i32) {
        self.set_channel_depth(3, depth);
    }

    /// Internal serialization data (canonical `_data` property getter). Returns
    /// the per-channel depth indices as `[type, sdf, color, indices, ...]`.
    /// Matches the upstream `_data` property.
    #[func]
    fn _get_data(&self) -> VarArray {
        let mut array = VarArray::new();
        for index in 0..8 {
            array.push(self.get_channel_depth(index));
        }
        array
    }

    #[func]
    fn _set_data(&mut self, data: VarArray) {
        for (index, value) in data.iter_shared().enumerate().take(8) {
            let Ok(i) = i32::try_from(index) else {
                break;
            };
            let depth: i32 = value.to::<i32>();
            self.set_channel_depth(i, depth);
        }
    }
}

// ---------------------------------------------------------------------------
// VoxelEngineGD — Object singleton for task orchestration
// ---------------------------------------------------------------------------

/// The voxel engine singleton. Wraps a ThreadedTaskRunner for
/// background task processing. Manages real task drain loop.
#[derive(GodotClass)]
#[class(base = Object, tool, rename = VoxelEngine)]
pub struct VoxelEngineGD {
    base: Base<Object>,
    /// Number of background threads (exposed via the canonical
    /// `get_thread_count`/`set_thread_count` #[func]s).
    thread_count: i32,
    runner: Option<voxel_core::tasks::ThreadedTaskRunner>,
}

#[godot_api]
impl IObject for VoxelEngineGD {
    fn init(base: Base<Object>) -> Self {
        Self {
            base,
            thread_count: 4,
            runner: None,
        }
    }
}

#[godot_api]
impl VoxelEngineGD {
    /// Initialize the task runner with the configured thread count.
    #[func]
    fn start(&mut self) {
        let count = self.thread_count.max(1) as usize;
        self.runner = Some(voxel_core::tasks::ThreadedTaskRunner::new(count));
    }

    /// Stop the task runner (waits for all tasks, then shuts down).
    #[func]
    fn stop(&mut self) {
        if let Some(mut runner) = self.runner.take() {
            runner.wait_for_all_tasks();
            runner.shutdown();
        }
    }

    /// Drain completed tasks. Returns the count drained this tick.
    #[func]
    fn process(&mut self) -> i32 {
        if let Some(runner) = &mut self.runner {
            let mut completed = std::collections::VecDeque::new();
            let Ok(count) = runner.try_drain_completed_into(&mut completed) else {
                return 0;
            };
            for completed_task in completed {
                let (task, status, follow_up_tasks) = completed_task.into_generic_parts();
                runner.enqueue_many(follow_up_tasks);
                if status == voxel_core::tasks::TaskCompletionStatus::Finished {
                    task.apply_result();
                }
            }
            count as i32
        } else {
            0
        }
    }

    /// Get the number of remaining (pending + running) tasks.
    #[func]
    fn get_pending_count(&self) -> i32 {
        if let Some(runner) = &self.runner {
            runner.remaining_task_count() as i32
        } else {
            0
        }
    }

    /// Block until all queued tasks complete.
    #[func]
    fn wait_for_all(&mut self) {
        if let Some(runner) = &mut self.runner {
            runner.wait_for_all_tasks();
        }
    }

    // -----------------------------------------------------------------
    // Canonical pinned VoxelEngine API (upstream 5828cbeb).
    // The voxel-core runtime exposes a ThreadedTaskRunner; the remaining
    // canonical surface is reported faithfully or stubbed where there is no
    // backing implementation in the headless binding.
    // -----------------------------------------------------------------

    /// Returns the number of threads currently used internally by the task
    /// runner. Mirrors the configured thread count.
    #[func]
    fn get_thread_count(&self) -> i32 {
        self.thread_count.max(1)
    }

    /// Sets the number of threads to be used internally by the task runner.
    /// Rebuilds the runner immediately in the headless binding.
    #[func]
    fn set_thread_count(&mut self, count: i32) {
        if count < 1 {
            godot_error!("VoxelEngine.set_thread_count: count must be >= 1");
            return;
        }
        self.thread_count = count;
        let n = count as usize;
        self.runner = Some(voxel_core::tasks::ThreadedTaskRunner::new(n));
    }

    /// Tells if the voxel engine is able to create graphics resources from
    /// different threads. The headless binding performs no graphics work, so
    /// this reports `false`.
    #[func]
    fn get_threaded_graphics_resource_building_enabled(&self) -> bool {
        false
    }

    /// Major version number of the voxel engine.
    #[func]
    fn get_version_major(&self) -> i32 {
        Self::VOXEL_VERSION_MAJOR
    }

    /// Minor version number of the voxel engine.
    #[func]
    fn get_version_minor(&self) -> i32 {
        Self::VOXEL_VERSION_MINOR
    }

    /// Patch version number of the voxel engine.
    #[func]
    fn get_version_patch(&self) -> i32 {
        Self::VOXEL_VERSION_PATCH
    }

    /// Major (x), minor (y) and patch (z) version numbers as a vector.
    #[func]
    fn get_version_v(&self) -> Vector3iGd {
        Vector3iGd::new(
            Self::VOXEL_VERSION_MAJOR,
            Self::VOXEL_VERSION_MINOR,
            Self::VOXEL_VERSION_PATCH,
        )
    }

    /// Edition of the voxel engine (`module` or `extension`). This binding is
    /// a GDExtension.
    #[func]
    fn get_version_edition(&self) -> GString {
        "extension".to_godot()
    }

    /// Version status (`dev` or `release`). This binding reports `dev`.
    #[func]
    fn get_version_status(&self) -> GString {
        "dev".to_godot()
    }

    /// Git hash the voxel engine was built from. Reports the binding's crate
    /// version string since no compile-time hash is embedded.
    #[func]
    fn get_version_git_hash(&self) -> GString {
        env!("CARGO_PKG_VERSION").to_godot()
    }

    /// Debug information about shared voxel processing. Reports the thread
    /// runner's current pending count.
    #[func]
    fn get_stats(&self) -> VarDictionary {
        let mut dict = VarDictionary::new();
        let mut thread_pools = VarDictionary::new();
        let mut general = VarDictionary::new();
        let active = self.get_pending_count();
        general.set("tasks", active);
        general.set(
            "active_threads",
            if active > 0 {
                self.get_thread_count()
            } else {
                0
            },
        );
        general.set("thread_count", self.get_thread_count());
        general.set("task_names", &PackedStringArray::new());
        thread_pools.set("general", &general);
        dict.set("thread_pools", &thread_pools);
        let mut tasks = VarDictionary::new();
        tasks.set("streaming", 0_i64);
        tasks.set("meshing", 0_i64);
        tasks.set("generation", 0_i64);
        tasks.set("main_thread", 0_i64);
        tasks.set("gpu", 0_i64);
        dict.set("tasks", &tasks);
        let mut memory_pools = VarDictionary::new();
        memory_pools.set("voxel_used", 0_i64);
        memory_pools.set("voxel_total", 0_i64);
        memory_pools.set("block_count", 0_i64);
        memory_pools.set("std_allocated", 0_i64);
        memory_pools.set("std_deallocated", 0_i64);
        memory_pools.set("std_current", 0_i64);
        dict.set("memory_pools", &memory_pools);
        dict
    }

    /// Runs internal unit tests. The voxel-gdext binding defers testing to the
    /// `cargo test` harness, so this is a stub.
    #[func]
    fn run_tests(&mut self, _options: VarDictionary) {
        godot_print!("VoxelEngine.run_tests: tests are run via `cargo test`");
    }

    #[constant]
    const VOXEL_VERSION_MAJOR: i32 = 1;
    #[constant]
    const VOXEL_VERSION_MINOR: i32 = 2;
    #[constant]
    const VOXEL_VERSION_PATCH: i32 = 0;
}

// ---------------------------------------------------------------------------
// VoxelSaveCompletionTrackerGD — RefCounted for save tracking
// ---------------------------------------------------------------------------

/// Tracks completion of save operations. Used by GDScript to await
/// terrain persistence. The functional API maintains a real pending counter:
/// `mark_pending` increments it, `mark_done` decrements it, and `is_done`
/// reflects whether all saves have completed.
#[derive(GodotClass)]
#[class(base = RefCounted, tool, rename = VoxelSaveCompletionTracker)]
pub struct VoxelSaveCompletionTrackerGD {
    base: Base<RefCounted>,
    /// Number of pending save operations (plain field; exposed via #[func]s).
    pending_count: i32,
    /// Whether all saves are done (pending_count == 0).
    is_done: bool,
    /// Total number of save tasks ever started (canonical
    /// `get_total_tasks`). Increments with each `mark_pending`.
    total_tasks: i32,
    /// Whether the tracker was aborted (canonical `is_aborted`).
    aborted: bool,
}

#[godot_api]
impl IRefCounted for VoxelSaveCompletionTrackerGD {
    fn init(base: Base<RefCounted>) -> Self {
        Self {
            base,
            pending_count: 0,
            is_done: true,
            total_tasks: 0,
            aborted: false,
        }
    }
}

#[godot_api]
impl VoxelSaveCompletionTrackerGD {
    /// Mark a save operation as started (increments pending_count).
    #[func]
    fn mark_pending(&mut self) {
        self.pending_count += 1;
        self.total_tasks += 1;
        self.is_done = false;
    }

    /// Mark a save operation as complete (decrements pending_count). Sets
    /// `is_done` true when the count reaches 0.
    #[func]
    fn mark_done(&mut self) {
        if self.pending_count > 0 {
            self.pending_count -= 1;
        }
        if self.pending_count == 0 {
            self.is_done = true;
        }
    }

    /// Current pending count.
    #[func]
    fn get_pending_count(&self) -> i32 {
        self.pending_count
    }

    /// Whether all saves are done.
    #[func]
    fn get_is_done(&self) -> bool {
        self.is_done
    }

    // -----------------------------------------------------------------
    // Canonical pinned VoxelSaveCompletionTracker API (upstream 5828cbeb).
    // -----------------------------------------------------------------

    /// Number of save tasks still pending (canonical
    /// `get_remaining_tasks`). Equal to `pending_count`.
    #[func]
    fn get_remaining_tasks(&self) -> i32 {
        self.pending_count
    }

    /// Total number of save tasks ever tracked (canonical `get_total_tasks`).
    #[func]
    fn get_total_tasks(&self) -> i32 {
        self.total_tasks
    }

    /// Whether the tracker was aborted (canonical `is_aborted`). The
    /// functional binding never aborts, so this stays `false` unless an
    /// external caller flips it via `set_aborted`.
    #[func]
    fn is_aborted(&self) -> bool {
        self.aborted
    }

    /// Whether all tracked tasks have completed (canonical `is_complete`).
    #[func]
    fn is_complete(&self) -> bool {
        self.is_done
    }
}

// ---------------------------------------------------------------------------
// VoxelDataBlockEnterInfoGD — RefCounted for block enter events
// ---------------------------------------------------------------------------

/// Information about a data block entering the resident set.
/// Emitted as part of terrain lifecycle events.
#[derive(GodotClass)]
#[class(base = RefCounted, tool, rename = VoxelDataBlockEnterInfo)]
pub struct VoxelDataBlockEnterInfoGD {
    base: Base<RefCounted>,
    #[var]
    block_x: i32,
    #[var]
    block_y: i32,
    #[var]
    block_z: i32,
    #[var]
    lod: i32,
    #[var]
    original_position: bool,
    /// Network peer ID of the viewer that caused the block to be referenced
    /// (canonical `get_network_peer_id`).
    network_peer_id: i32,
    /// Whether the block's voxels have ever been edited (canonical
    /// `are_voxels_edited`).
    voxels_edited: bool,
}

#[godot_api]
impl IRefCounted for VoxelDataBlockEnterInfoGD {
    fn init(base: Base<RefCounted>) -> Self {
        Self {
            base,
            block_x: 0,
            block_y: 0,
            block_z: 0,
            lod: 0,
            original_position: false,
            network_peer_id: 0,
            voxels_edited: false,
        }
    }
}

#[godot_api]
impl VoxelDataBlockEnterInfoGD {
    /// Whether this block is at the world origin (0,0,0).
    #[func]
    fn is_at_origin(&self) -> bool {
        self.block_x == 0 && self.block_y == 0 && self.block_z == 0
    }

    /// The LOD level of this block.
    #[func]
    fn get_lod_level(&self) -> i32 {
        self.lod
    }

    // -----------------------------------------------------------------
    // Canonical pinned VoxelDataBlockEnterInfo API (upstream 5828cbeb).
    // The struct is a data carrier; the canonical accessors report the
    // stored fields faithfully.
    // -----------------------------------------------------------------

    /// Gets the position of the data block, in data block coordinates.
    #[func]
    fn get_position(&self) -> Vector3iGd {
        Vector3iGd::new(self.block_x, self.block_y, self.block_z)
    }

    /// Gets which LOD index the data block is in (canonical `get_lod_index`).
    #[func]
    fn get_lod_index(&self) -> i32 {
        self.lod
    }

    /// Gets the network peer ID of the viewer that caused the block to be
    /// referenced.
    #[func]
    fn get_network_peer_id(&self) -> i32 {
        self.network_peer_id
    }

    /// Tells if voxels in the block have ever been edited.
    #[func]
    fn are_voxels_edited(&self) -> bool {
        self.voxels_edited
    }

    /// Gets access to the voxels in the block. The data-block enter info is a
    /// thin data carrier in this binding and does not retain a buffer, so this
    /// returns null.
    #[func]
    fn get_voxels(&self) -> Option<Gd<crate::voxel_buffer::VoxelBufferGD>> {
        None
    }
}

// ---------------------------------------------------------------------------
// VoxelInstanceLibraryGD — Resource for instance library
// ---------------------------------------------------------------------------

/// A library of scatter items for instancing. Wraps
/// [`voxel_core::instancing::InstanceLibrary`] — the functional API maintains
/// a real item table and reports its count.
#[derive(GodotClass)]
#[class(base = Resource, tool, rename = VoxelInstanceLibrary)]
pub struct VoxelInstanceLibraryGD {
    base: Base<Resource>,
    /// Number of items (plain field; exposed via get_item_count #[func]).
    item_count: i32,
    /// The real engine-agnostic library.
    library: voxel_core::instancing::InstanceLibrary,
    /// Canonical id-indexed item table (canonical pinned API). Maps the
    /// integer id passed to `add_item` to the Godot resource.
    items: HashMap<i32, Gd<VoxelInstanceLibraryItemGD>>,
    /// Canonical `_selected_item` property backing field.
    selected_item: Option<Gd<VoxelInstanceLibraryItemGD>>,
    /// The pinned GDScript-facing `_data` property.
    #[var(get = _get_data, set = _set_data)]
    _data: PhantomVar<Array<Variant>>,
    /// The pinned GDScript-facing `_selected_item` property.
    #[var(get = _get_selected_item, set = _set_selected_item)]
    _selected_item: PhantomVar<Option<Gd<VoxelInstanceLibraryItemGD>>>,
}

#[godot_api]
impl IResource for VoxelInstanceLibraryGD {
    fn init(base: Base<Resource>) -> Self {
        Self {
            base,
            item_count: 0,
            library: voxel_core::instancing::InstanceLibrary::new(),
            items: HashMap::new(),
            selected_item: None,
            _data: PhantomVar::default(),
            _selected_item: PhantomVar::default(),
        }
    }
}

#[godot_api]
impl VoxelInstanceLibraryGD {
    /// Adds a scatter item by name + density + scale range to the underlying
    /// core library and returns its index. (Functional helper retained from
    /// the pre-canonical binding; the canonical pinned API is `add_item`.)
    #[func]
    fn add_named_item(
        &mut self,
        name: GString,
        density: f32,
        min_scale: f32,
        max_scale: f32,
        snap_to_normal: bool,
    ) -> i32 {
        let item = voxel_core::instancing::InstanceLibraryItem {
            name: name.to_string(),
            density,
            min_scale,
            max_scale,
            snap_to_normal,
            ..Default::default()
        };
        let idx = self.library.add_item(item);
        self.item_count = self.library.len() as i32;
        idx as i32
    }

    /// Number of registered items (functional delegate on the core library).
    #[func]
    fn get_item_count(&self) -> i32 {
        self.item_count
    }

    /// Whether the library has no items (functional delegate).
    #[func]
    fn is_empty(&self) -> bool {
        self.library.is_empty()
    }

    // -----------------------------------------------------------------
    // Canonical pinned VoxelInstanceLibrary API (upstream 5828cbeb).
    // The id-indexed table is the canonical store; the functional core
    // library is retained for the existing `add_named_item`/`get_item_count`
    // helpers.
    // -----------------------------------------------------------------

    /// Adds an item to the library at the given id (canonical `add_item`).
    /// The id must be within `0..=MAX_ID`.
    #[func]
    fn add_item(&mut self, id: i32, item: Gd<VoxelInstanceLibraryItemGD>) {
        if !(0..=Self::MAX_ID).contains(&id) {
            godot_error!(
                "VoxelInstanceLibrary.add_item: id {id} out of range 0..={}",
                Self::MAX_ID
            );
            return;
        }
        self.items.insert(id, item);
        self.item_count = i32::try_from(self.items.len()).unwrap_or(i32::MAX);
    }

    /// Removes all items from the library (canonical `clear`).
    #[func]
    fn clear(&mut self) {
        self.items.clear();
        self.item_count = 0;
    }

    /// Finds the id of the first item whose name matches, or -1 if not found
    /// (canonical `find_item_by_name`).
    #[func]
    fn find_item_by_name(&self, name: GString) -> i32 {
        let target = name.to_string();
        let mut ids: Vec<i32> = self.items.keys().copied().collect();
        ids.sort_unstable();
        for id in ids {
            if let Some(item) = self.items.get(&id) {
                let bind = item.bind();
                if bind.name.to_string() == target {
                    return id;
                }
            }
        }
        -1
    }

    /// Returns the ids of all items in the library (canonical
    /// `get_all_item_ids`).
    #[func]
    fn get_all_item_ids(&self) -> PackedInt32Array {
        let mut ids: Vec<i32> = self.items.keys().copied().collect();
        ids.sort_unstable();
        PackedInt32Array::from(ids.as_slice())
    }

    /// Gets the item registered at the given id, or null (canonical
    /// `get_item`).
    #[func]
    fn get_item(&self, id: i32) -> Option<Gd<VoxelInstanceLibraryItemGD>> {
        self.items.get(&id).cloned()
    }

    /// Removes the item at the given id (canonical `remove_item`).
    #[func]
    fn remove_item(&mut self, id: i32) {
        if self.items.remove(&id).is_some() {
            self.item_count = i32::try_from(self.items.len()).unwrap_or(i32::MAX);
        }
    }

    /// Getter for the canonical `_data` property. Returns the serialised item
    /// table as an array of variants.
    #[func]
    fn _get_data(&self) -> Array<Variant> {
        let mut arr = Array::new();
        let mut ids: Vec<i32> = self.items.keys().copied().collect();
        ids.sort_unstable();
        for id in ids {
            // Each entry is `[id, item]` so GDScript can rebuild the table.
            let mut entry = Array::<Variant>::new();
            entry.push(&id.to_variant());
            if let Some(item) = self.items.get(&id) {
                entry.push(&item.to_variant());
            }
            arr.push(&entry.to_variant());
        }
        arr
    }

    /// Setter for the canonical `_data` property. Replaces the item table from
    /// a serialised array.
    #[func]
    fn _set_data(&mut self, data: Array<Variant>) {
        self.items.clear();
        for entry in data.iter_shared() {
            let Ok(pair) = entry.try_to::<Array<Variant>>() else {
                continue;
            };
            let mut iter = pair.iter_shared();
            let Some(id_v) = iter.next() else {
                continue;
            };
            let Some(item_v) = iter.next() else {
                continue;
            };
            let Ok(id) = id_v.try_to::<i32>() else {
                continue;
            };
            if let Ok(item) = item_v.try_to::<Gd<VoxelInstanceLibraryItemGD>>() {
                if (0..=Self::MAX_ID).contains(&id) {
                    self.items.insert(id, item);
                }
            }
        }
        self.item_count = i32::try_from(self.items.len()).unwrap_or(i32::MAX);
    }

    /// Getter for the canonical `_selected_item` property.
    #[func]
    fn _get_selected_item(&self) -> Option<Gd<VoxelInstanceLibraryItemGD>> {
        self.selected_item.clone()
    }

    /// Setter for the canonical `_selected_item` property.
    #[func]
    fn _set_selected_item(&mut self, item: Option<Gd<VoxelInstanceLibraryItemGD>>) {
        self.selected_item = item;
    }

    #[constant]
    const MAX_ID: i32 = 65535;
}

// ---------------------------------------------------------------------------
// VoxelInstanceLibraryItemGD — Resource for one scatter item
// ---------------------------------------------------------------------------

/// One entry in a [`VoxelInstanceLibraryGD`]. Defines what to scatter and how.
/// The functional API produces a real
/// [`voxel_core::instancing::InstanceLibraryItem`] via `to_core_item`.
///
/// The pinned GDScript-facing properties (`floating_sdf_offset_along_normal`,
/// `floating_sdf_threshold`, `generator`, `lod_index`, `name`, `persistent`)
/// mirror upstream `VoxelInstanceLibraryItem` (5828cbeb). They are stored
/// faithfully so GDScript reads round-trip.
#[derive(GodotClass)]
#[class(base = Resource, tool, rename = VoxelInstanceLibraryItem)]
pub struct VoxelInstanceLibraryItemGD {
    base: Base<Resource>,
    #[var(get = get_item_name, set = set_item_name)]
    name: GString,
    #[var]
    density: f32,
    #[var]
    min_scale: f32,
    #[var]
    max_scale: f32,
    #[var]
    snap_to_normal: bool,
    /// Pinned `floating_sdf_offset_along_normal` (backing field). Upstream
    /// default -0.1.
    floating_sdf_offset_along_normal_value: f32,
    /// Pinned `floating_sdf_threshold` (backing field). Upstream default 0.0.
    floating_sdf_threshold_value: f32,
    /// Pinned `generator` resource (backing field; `None` until assigned). The
    /// upstream type `VoxelInstanceGenerator` is not yet bound, so a generic
    /// resource slot is used.
    generator_resource: Option<Gd<Resource>>,
    /// Pinned `lod_index` (backing field). Upstream default 0.
    lod_index_value: i32,
    /// Pinned `persistent` (backing field). Upstream default false.
    persistent_value: bool,
    /// The pinned GDScript-facing `floating_sdf_offset_along_normal` property.
    #[var(get = get_floating_sdf_offset_along_normal, set = set_floating_sdf_offset_along_normal)]
    floating_sdf_offset_along_normal: PhantomVar<f32>,
    /// The pinned GDScript-facing `floating_sdf_threshold` property.
    #[var(get = get_floating_sdf_threshold, set = set_floating_sdf_threshold)]
    floating_sdf_threshold: PhantomVar<f32>,
    /// The pinned GDScript-facing `generator` property.
    #[var(get = get_generator, set = set_generator)]
    generator: PhantomVar<Option<Gd<Resource>>>,
    /// The pinned GDScript-facing `lod_index` property.
    #[var(get = get_lod_index, set = set_lod_index)]
    lod_index: PhantomVar<i32>,
    /// The pinned GDScript-facing `name` property (uses the canonical
    /// `set_item_name`/`get_item_name` accessors).
    #[var(get = get_item_name, set = set_item_name)]
    item_name: PhantomVar<GString>,
    /// The pinned GDScript-facing `persistent` property.
    #[var(get = is_persistent, set = set_persistent)]
    persistent: PhantomVar<bool>,
}

#[godot_api]
impl IResource for VoxelInstanceLibraryItemGD {
    fn init(base: Base<Resource>) -> Self {
        Self {
            base,
            name: "Item".to_godot(),
            density: 0.1,
            min_scale: 0.8,
            max_scale: 1.2,
            snap_to_normal: true,
            floating_sdf_offset_along_normal_value: -0.1,
            floating_sdf_threshold_value: 0.0,
            generator_resource: None,
            lod_index_value: 0,
            persistent_value: false,
            floating_sdf_offset_along_normal: PhantomVar::default(),
            floating_sdf_threshold: PhantomVar::default(),
            generator: PhantomVar::default(),
            lod_index: PhantomVar::default(),
            item_name: PhantomVar::default(),
            persistent: PhantomVar::default(),
        }
    }
}

#[godot_api]
impl VoxelInstanceLibraryItemGD {
    /// Item name. Custom accessor names avoid shadowing Resource methods while
    /// preserving the canonical `name` property in GDScript.
    #[func]
    fn get_item_name(&self) -> GString {
        self.name.clone()
    }

    #[func]
    fn set_item_name(&mut self, name: GString) {
        self.name = name;
    }

    /// Effective scale range midpoint (functional delegate).
    #[func]
    fn get_average_scale(&self) -> f32 {
        (self.min_scale + self.max_scale) * 0.5
    }

    /// Scale range span (max - min).
    #[func]
    fn get_scale_range(&self) -> f32 {
        self.max_scale - self.min_scale
    }

    /// Whether density is zero (no instances would be produced).
    #[func]
    fn is_disabled(&self) -> bool {
        self.density <= 0.0
    }

    // -----------------------------------------------------------------
    // Pinned VoxelInstanceLibraryItem properties
    // (upstream 5828cbeb: VoxelInstanceLibraryItem.xml).
    // -----------------------------------------------------------------

    /// Offset along the instance's upward axis used when testing for floating
    /// state (upstream default -0.1).
    #[func]
    fn get_floating_sdf_offset_along_normal(&self) -> f32 {
        self.floating_sdf_offset_along_normal_value
    }

    #[func]
    fn set_floating_sdf_offset_along_normal(&mut self, offset: f32) {
        self.floating_sdf_offset_along_normal_value = offset;
    }

    /// Threshold above which the SDF is treated as air, marking the instance
    /// floating (upstream default 0.0).
    #[func]
    fn get_floating_sdf_threshold(&self) -> f32 {
        self.floating_sdf_threshold_value
    }

    #[func]
    fn set_floating_sdf_threshold(&mut self, threshold: f32) {
        self.floating_sdf_threshold_value = threshold;
    }

    /// Generator used to pick points where the item spawns (`None` until
    /// assigned).
    #[func]
    fn get_generator(&self) -> Option<Gd<Resource>> {
        self.generator_resource.clone()
    }

    #[func]
    fn set_generator(&mut self, generator: Option<Gd<Resource>>) {
        self.generator_resource = generator;
    }

    /// LOD index of terrain chunks where this item will spawn (upstream
    /// default 0).
    #[func]
    fn get_lod_index(&self) -> i32 {
        self.lod_index_value
    }

    #[func]
    fn set_lod_index(&mut self, index: i32) {
        self.lod_index_value = index;
    }

    /// Whether the item is saved across sessions (upstream default false).
    #[func]
    fn is_persistent(&self) -> bool {
        self.persistent_value
    }

    #[func]
    fn set_persistent(&mut self, persistent: bool) {
        self.persistent_value = persistent;
    }
}

#[cfg(test)]
mod tests {
    use super::{
        validate_enum_int, validate_finite_float, validate_mesher_channel, validate_shadow_side,
    };
    use voxel_core::storage::voxel_buffer::MAX_CHANNELS;

    #[test]
    fn mesher_channel_validation_rejects_every_out_of_range_value() {
        assert!(validate_mesher_channel(-1).is_err());
        assert_eq!(validate_mesher_channel(0), Ok(0));
        assert_eq!(
            validate_mesher_channel((MAX_CHANNELS - 1) as i32),
            Ok(MAX_CHANNELS - 1)
        );
        assert!(validate_mesher_channel(MAX_CHANNELS as i32).is_err());
        assert!(validate_mesher_channel(i32::MAX).is_err());
    }

    #[test]
    fn finite_float_validation_rejects_non_finite_values() {
        assert_eq!(validate_finite_float(0.0), Ok(0.0));
        assert_eq!(validate_finite_float(-12.5), Ok(-12.5));
        assert!(validate_finite_float(f32::INFINITY).is_err());
        assert!(validate_finite_float(f32::NEG_INFINITY).is_err());
        assert!(validate_finite_float(f32::NAN).is_err());
    }

    #[test]
    fn enum_int_validation_enforces_inclusive_range() {
        assert_eq!(validate_enum_int(0, 0, 2), Ok(0));
        assert_eq!(validate_enum_int(2, 0, 2), Ok(2));
        assert!(validate_enum_int(-1, 0, 2).is_err());
        assert!(validate_enum_int(3, 0, 2).is_err());
    }

    #[test]
    fn shadow_side_validation_accepts_zero_through_five() {
        assert_eq!(validate_shadow_side(0), Ok(0));
        assert_eq!(validate_shadow_side(5), Ok(5));
        assert!(validate_shadow_side(-1).is_err());
        assert!(validate_shadow_side(6).is_err());
    }
}
