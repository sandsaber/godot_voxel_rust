//! Builtin [`VoxelMesher`] implementations wrapping the per-algorithm free
//! functions (`meshers::transvoxel`, `meshers::cubes`, `meshers::blocky`).
//!
//! These adapters let the terrain pipeline drive real meshers through the
//! trait object without rewriting the algorithm code — each mesher is a thin
//! shim that pulls voxels out of [`VoxelBuffer`] and forwards them to the
//! existing free function. All three adapters (transvoxel, cubes, blocky)
//! are implemented below.

use crate::math::Vector3i;
use crate::meshers::transvoxel::{
    build_regular_mesh, build_transition_mesh, BuildRegularMeshParams, Cache, MeshArrays,
    RegularMesherInput, MAX_PADDING, MIN_PADDING, SIDE_COUNT,
};
use crate::meshers::{MesherInput, MesherOutput, Surface, SurfaceArrays, VoxelMesher};
use crate::storage::funcs;
use crate::storage::voxel_buffer::{
    QUANTIZED_SDF_16_BITS_SCALE_INV, QUANTIZED_SDF_8_BITS_SCALE_INV,
};
use crate::storage::{ChannelData, ChannelDepth, ChannelId, Compression, VoxelBuffer};
#[cfg(test)]
use std::cell::Cell;
use std::cell::RefCell;

thread_local! {
    static TRANSVOXEL_CACHE: RefCell<Cache> = RefCell::new(Cache::default());
}

#[cfg(test)]
thread_local! {
    static TRANSVOXEL_SAMPLE_COUNT: Cell<usize> = const { Cell::new(0) };
}

/// Resolved, depth-dispatched SDF sampler over a materialised [`ChannelData`].
///
/// This is the **fallback adapter** used when a monomorphized typed input
/// ([`TypedSdfInput`]) is not available (Bit64 channels, typed-cast misses):
/// the variant + depth-specific decode are resolved **once** in
/// [`TransvoxelMesher::build`] before the core loop, so
/// [`RegularMesherInput::sample_f32`] still collapses to a single typed slice
/// index (`slice[data_index]`) plus one decode — no per-voxel `match` on
/// depth, no `(x,y,z)` div/mod reconstruction, and no second pass through
/// `raw_voxel_to_real` (mirroring how the C++ mesher settles data types
/// up-front to rid the hot loop of abstraction layers).
///
/// The channel must be `Compression::None` (callers guard with `is_uniform`),
/// so the typed slice is guaranteed non-empty.
enum TypedSdfSampler<'a> {
    Bit8(&'a [u8]),
    Bit16(&'a [u16]),
    Bit32(&'a [u32]),
    Bit64(&'a [u64]),
}

impl<'a> TypedSdfSampler<'a> {
    /// Resolve the SDF channel of `buffer` into a typed sampler. The channel
    /// must already be decompressed (`Compression::None`); a uniform channel
    /// yields an empty slice and is the caller's responsibility to short-circuit.
    fn new(buffer: &'a VoxelBuffer, sdf_channel: usize, depth: ChannelDepth) -> Self {
        // `channel_data` returns the live typed storage; for Compression::None
        // it is the fully-materialised ZXY array the core indexes directly.
        let data = buffer.channel_data(sdf_channel);
        match (depth, data) {
            (ChannelDepth::Bit8, ChannelData::U8(v)) => Self::Bit8(v.as_slice()),
            (ChannelDepth::Bit16, ChannelData::U16(v)) => Self::Bit16(v.as_slice()),
            (ChannelDepth::Bit32, ChannelData::U32(v)) => Self::Bit32(v.as_slice()),
            (ChannelDepth::Bit64, ChannelData::U64(v)) => Self::Bit64(v.as_slice()),
            // Variant/depth mismatch should never happen for a materialised
            // channel — fall back to an empty slice so the core sees no samples.
            _ => Self::Bit8(&[]),
        }
    }

    /// Decode one SDF sample to `sdf_as_float` semantics (negated, matching
    /// C++ `sdf_as_float(float)`): 8/16-bit expand from snorm × SDF scale,
    /// 32-bit store float bits directly, 64-bit f64→f32.
    #[inline]
    fn sample(&self, data_index: usize) -> f32 {
        match self {
            // snorm decode matches `raw_voxel_to_real` for Bit8/Bit16.
            Self::Bit8(v) => {
                -(funcs::s8_to_snorm(v[data_index] as i8) * QUANTIZED_SDF_8_BITS_SCALE_INV)
            }
            Self::Bit16(v) => {
                -(funcs::s16_to_snorm(v[data_index] as i16) * QUANTIZED_SDF_16_BITS_SCALE_INV)
            }
            Self::Bit32(v) => -f32::from_bits(v[data_index]),
            Self::Bit64(v) => -(f64::from_bits(v[data_index]) as f32),
        }
    }

    #[inline]
    fn len(&self) -> usize {
        match self {
            Self::Bit8(v) => v.len(),
            Self::Bit16(v) => v.len(),
            Self::Bit32(v) => v.len(),
            Self::Bit64(v) => v.len(),
        }
    }
}

/// `RegularMesherInput` adapter carrying a resolved [`TypedSdfSampler`] plus the
/// padded block size. `sample_f32` indexes the typed slice directly by the flat
/// ZXY `data_index` the core already computed.
struct VoxelBufferTransvoxelInput<'a> {
    sampler: TypedSdfSampler<'a>,
    size: Vector3i,
}

impl<'a> VoxelBufferTransvoxelInput<'a> {
    fn new(buffer: &'a VoxelBuffer, sdf_channel: usize) -> Self {
        let depth = buffer.channel_depth(sdf_channel);
        Self {
            sampler: TypedSdfSampler::new(buffer, sdf_channel, depth),
            size: buffer.size(),
        }
    }
}

impl<'a> RegularMesherInput for VoxelBufferTransvoxelInput<'a> {
    fn len(&self) -> usize {
        self.sampler.len()
    }

    fn block_size(&self) -> Vector3i {
        self.size
    }

    fn sample_f32(&self, data_index: usize) -> f32 {
        #[cfg(test)]
        TRANSVOXEL_SAMPLE_COUNT.with(|samples| samples.set(samples.get() + 1));

        // No div/mod back to (x,y,z): the core passes the flat ZXY index the
        // slice already uses. Depth dispatch happened once in `TypedSdfSampler::new`.
        self.sampler.sample(data_index)
    }
}

fn checked_regular_collision_prefix_ends(
    vertex_count: usize,
    index_count: usize,
) -> Option<(i32, i32)> {
    if index_count == 0 {
        return None;
    }
    Some((
        i32::try_from(vertex_count).ok()?,
        i32::try_from(index_count).ok()?,
    ))
}

/// Per-type SDF sample → float conversion. Mirrors C++ `sdf_as_float` overloads
/// (`transvoxel.cpp:133-147`), but uses the **clamp** snorm forms to match
/// `VoxelBuffer::get_voxel_f` → `raw_voxel_to_real` (which clamps) exactly, so
/// the typed path produces byte-identical output to the dyn-dispatch path.
///
/// The conversion is applied once per sample inside [`TypedSdfInput`], with the
/// depth branch hoisted out of the per-voxel loop by the monomorphized
/// `build_regular_mesh<TypedSdfInput<T>>`.
trait SdfSample {
    /// `f32` equivalent of this raw sample, negated (engine SDF sign convention).
    fn to_sdf_f32(raw: Self) -> f32;
}

impl SdfSample for i8 {
    #[inline]
    fn to_sdf_f32(raw: Self) -> f32 {
        -(funcs::s8_to_snorm(raw) * QUANTIZED_SDF_8_BITS_SCALE_INV)
    }
}

impl SdfSample for i16 {
    #[inline]
    fn to_sdf_f32(raw: Self) -> f32 {
        -(funcs::s16_to_snorm(raw) * QUANTIZED_SDF_16_BITS_SCALE_INV)
    }
}

impl SdfSample for f32 {
    #[inline]
    fn to_sdf_f32(raw: Self) -> f32 {
        -raw
    }
}

/// Zero-copy typed SDF input over a flat channel slice. This is the
/// monomorphization target for [`build_regular_mesh`]: the per-type conversion
/// and the `block_size` strides are known to the compiler, so the inner cell
/// loop indexes the slice directly with no vtable call, no div/mod index
/// re-derivation, and no per-sample depth branch.
///
/// Obtained from [`VoxelBuffer::channel_typed_slice`] once per mesh build (the
/// adapter's depth dispatch). Falls back to [`VoxelBufferTransvoxelInput`] when
/// a typed slice is unavailable (uniform channel or alignment mismatch).
struct TypedSdfInput<'a, T: SdfSample> {
    slice: &'a [T],
    block_size: Vector3i,
}

impl<'a, T: SdfSample + bytemuck::Pod> RegularMesherInput for TypedSdfInput<'a, T> {
    #[inline]
    fn len(&self) -> usize {
        self.slice.len()
    }

    #[inline]
    fn block_size(&self) -> Vector3i {
        self.block_size
    }

    #[inline]
    fn sample_f32(&self, data_index: usize) -> f32 {
        #[cfg(test)]
        TRANSVOXEL_SAMPLE_COUNT.with(|samples| samples.set(samples.get() + 1));

        T::to_sdf_f32(self.slice[data_index])
    }
}

/// Smooth (SDF) terrain mesher wrapping the transvoxel regular-cell path.
///
/// Produces one [`Surface`] per `build` call (single material, index 0).
/// Vertices/normals/indices are stored in a [`MeshArrays`] wrapped by
/// [`SurfaceArrays::Transvoxel`]. Padding is fixed at the transvoxel
/// algorithm's `MIN_PADDING=1` / `MAX_PADDING=2` requirement.
pub struct TransvoxelMesher {
    sdf_channel: usize,
    /// Minimum distance from cell corners below which interpolated vertices
    /// are clamped (0.0 = no clamping, 0.5 = always mid-edge). Mirrors upstream
    /// `VoxelMesherTransvoxel::edge_clamp_margin` (pinned default 0.02 in the
    /// Godot class; the **core** default stays 0.0 so the C++ parity goldens,
    /// which were generated with 0.0, remain byte-identical).
    edge_clamp_margin: f32,
    /// When `false`, the `MesherInput::lod_hint` transition passes are
    /// suppressed. Mirrors upstream `VoxelMesherTransvoxel::transitions_enabled`
    /// (default enabled).
    transitions_enabled: bool,
}

impl Default for TransvoxelMesher {
    fn default() -> Self {
        Self::new()
    }
}

impl TransvoxelMesher {
    pub fn new() -> Self {
        Self {
            sdf_channel: ChannelId::Sdf.index(),
            edge_clamp_margin: 0.0,
            transitions_enabled: true,
        }
    }

    /// Use a channel other than the default SDF channel.
    pub fn with_sdf_channel(mut self, channel: usize) -> Self {
        self.sdf_channel = channel;
        self
    }

    /// Set the edge-clamp margin. Values are clamped to `[0.0, 0.5]`, matching
    /// the upstream property contract (0 = no clamping, 0.5 = mid-edge).
    pub fn with_edge_clamp_margin(mut self, margin: f32) -> Self {
        self.edge_clamp_margin = if margin.is_finite() {
            margin.clamp(0.0, 0.5)
        } else {
            0.0
        };
        self
    }

    /// Enable or disable LOD transition-mesh generation (upstream
    /// `transitions_enabled`).
    pub fn with_transitions_enabled(mut self, enabled: bool) -> Self {
        self.transitions_enabled = enabled;
        self
    }

    /// The configured edge-clamp margin (already clamped to 0..=0.5).
    pub fn edge_clamp_margin(&self) -> f32 {
        self.edge_clamp_margin
    }

    /// Whether LOD transition meshes are generated.
    pub fn transitions_enabled(&self) -> bool {
        self.transitions_enabled
    }

    /// Build **only** the transition mesh for one `direction` (0..=5, see the
    /// `SIDE_*` constants), without the regular cells. Mirrors upstream
    /// `VoxelMesherTransvoxel::build_transition_mesh` (mainly for testing).
    /// Returns the mesh arrays containing just that transition pass; a uniform
    /// SDF channel yields empty arrays (matching the regular path's fast-path).
    pub fn build_transition_mesh_for_direction(
        &self,
        input: &MesherInput<'_>,
        direction: u8,
    ) -> MeshArrays {
        let mut arrays = MeshArrays::default();
        if direction >= SIDE_COUNT || input.voxels.is_uniform(self.sdf_channel) {
            return arrays;
        }
        let params = BuildRegularMeshParams {
            lod_index: u32::from(input.lod_index),
            edge_clamp_margin: self.edge_clamp_margin,
        };
        let transvoxel_input = VoxelBufferTransvoxelInput::new(input.voxels, self.sdf_channel);
        TRANSVOXEL_CACHE.with(|cache| {
            build_transition_mesh(
                &transvoxel_input,
                &params,
                direction,
                &mut cache.borrow_mut(),
                &mut arrays,
            );
        });
        arrays
    }
}

/// Resolve a typed SDF slice and run the monomorphized transvoxel mesher on it.
/// Returns `false` (without touching `arrays`) when a zero-copy typed cast is
/// not possible, so the caller can fall back to the dyn-dispatch adapter. This
/// is the Rust analogue of C++ `build_regular_mesh_dispatch_sd<T>` +
/// `Span::reinterpret_cast_to<T>()`: one depth branch + one cast before the
/// hot loop, no per-sample abstraction.
fn run_typed_mesher<T: SdfSample + bytemuck::Pod>(
    voxels: &VoxelBuffer,
    sdf_channel: usize,
    params: &BuildRegularMeshParams,
    arrays: &mut MeshArrays,
) -> bool {
    let Some(slice) = voxels.channel_typed_slice::<T>(sdf_channel) else {
        return false;
    };
    let typed_input = TypedSdfInput {
        slice,
        block_size: voxels.size(),
    };
    TRANSVOXEL_CACHE.with(|cache| {
        build_regular_mesh(&typed_input, params, &mut cache.borrow_mut(), arrays);
    });
    true
}

impl VoxelMesher for TransvoxelMesher {
    fn build(&self, output: &mut MesherOutput, input: &MesherInput<'_>) {
        if input.voxels.is_uniform(self.sdf_channel) {
            output.surfaces.push(Surface::new(
                SurfaceArrays::Transvoxel(MeshArrays::default()),
                0,
            ));
            return;
        }

        let params = BuildRegularMeshParams {
            lod_index: u32::from(input.lod_index),
            edge_clamp_margin: self.edge_clamp_margin,
        };
        let transvoxel_input = VoxelBufferTransvoxelInput::new(input.voxels, self.sdf_channel);
        // B3 (audit §9.6-B3): reuse a pooled `MeshArrays` when the terrain core
        // supplies a free-list. The pool returns a cleared buffer, and
        // `build_regular_mesh` appends into it, so reuse is safe. When no pool
        // is attached, fall back to a fresh allocation.
        let mut arrays = match input.mesh_arrays_pool {
            Some(pool) => pool.acquire(),
            None => MeshArrays::default(),
        };
        // B1: depth-dispatch up-front — resolve a typed slice once (mirroring
        // C++ `build_regular_mesh_dispatch_sd` /
        // `Span::reinterpret_cast_to<T>()`) and feed it to a monomorphized
        // `build_regular_mesh`. Falls back to the enum-dispatched typed-slice
        // adapter (Bit64 / cast miss) when a typed cast isn't available.
        // Transition meshes stay on that adapter via `&dyn`: they are six
        // small passes appended after the regular mesh, not the per-voxel hot
        // loop.
        let depth = input.voxels.channel_depth(self.sdf_channel);
        let compression = input.voxels.channel_compression(self.sdf_channel);
        let typed_ok = compression == Compression::None
            && match depth {
                ChannelDepth::Bit8 => {
                    run_typed_mesher::<i8>(input.voxels, self.sdf_channel, &params, &mut arrays)
                }
                ChannelDepth::Bit16 => {
                    run_typed_mesher::<i16>(input.voxels, self.sdf_channel, &params, &mut arrays)
                }
                ChannelDepth::Bit32 => {
                    run_typed_mesher::<f32>(input.voxels, self.sdf_channel, &params, &mut arrays)
                }
                ChannelDepth::Bit64 => false,
            };
        TRANSVOXEL_CACHE.with(|cache| {
            if !typed_ok {
                build_regular_mesh(
                    &transvoxel_input,
                    &params,
                    &mut cache.borrow_mut(),
                    &mut arrays,
                );
            }
            // Collision uses only the regular mesh prefix. Transition geometry
            // is appended below for rendering, but must never enter physics.
            // Keep the default -1 sentinels if the prefix is empty or cannot be
            // represented by the C++-compatible i32 range contract.
            if input.collision_hint {
                if let Some((vertex_end, index_end)) = checked_regular_collision_prefix_ends(
                    arrays.vertices.len(),
                    arrays.indices.len(),
                ) {
                    output.collision_surface.submesh_vertex_end = vertex_end;
                    output.collision_surface.submesh_index_end = index_end;
                }
            }
            // M2.2: build transition meshes on all 6 faces when LOD transitions
            // are needed (lod_hint = true) and not disabled via the upstream
            // `transitions_enabled` knob. Transition verts are appended to the
            // same MeshArrays, producing a watertight surface across LOD seams.
            if input.lod_hint && self.transitions_enabled {
                for dir in 0..SIDE_COUNT {
                    build_transition_mesh(
                        &transvoxel_input,
                        &params,
                        dir,
                        &mut cache.borrow_mut(),
                        &mut arrays,
                    );
                }
            }
        });
        // Even an empty transvoxel run produces a surface (zero triangles);
        // match C++ which always emits the surface and lets the caller drop
        // empty ones.
        output
            .surfaces
            .push(Surface::new(SurfaceArrays::Transvoxel(arrays), 0));
    }

    fn minimum_padding(&self) -> u32 {
        MIN_PADDING as u32
    }

    fn maximum_padding(&self) -> u32 {
        MAX_PADDING as u32
    }

    fn used_channels_mask(&self) -> u32 {
        1u32 << self.sdf_channel
    }

    fn is_generating_collision_surface(&self) -> bool {
        true
    }
}

/// Color mode for [`CubesMesher`]. Matches C++ `VoxelMesherCubes::ColorMode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CubesColorMode {
    /// Interpret each voxel value as packed RGBA at the channel's native width.
    /// This is the C++ default (`COLOR_RAW`).
    #[default]
    Raw,
    /// Look up each voxel value (low 8 bits) in a [`ColorPalette`].
    Palette,
    /// Write the raw voxel value into the red component and the palette
    /// entry's alpha into alpha, leaving green and blue at zero (upstream
    /// `COLOR_SHADER_PALETTE`). A custom shader reads the palette index back
    /// with `int(COLOR.r * 255.0)`; the palette is still required so
    /// transparent entries (alpha < 255) are sorted into the transparent
    /// material. Its RGB values are unused.
    ShaderPalette,
}

/// Blocky colored-cube mesher wrapping the existing greedy-cubes free
/// function. Reads the `Color` channel as 32-bit voxel ids, looks up each
/// id in a [`ColorPalette`], and emits two surfaces (opaque + transparent)
/// matching the C++ `VoxelMesherCubes` output.
pub struct CubesMesher {
    type_channel: usize,
    palette: crate::meshers::cubes::palette::ColorPalette,
    color_mode: CubesColorMode,
    /// When `true`, uses the greedy rectangle-merging path. When `false`,
    /// emits one quad per face (the simpler reference path). Greedy is the
    /// C++ default.
    greedy: bool,
}

impl Default for CubesMesher {
    fn default() -> Self {
        Self::new()
    }
}

impl CubesMesher {
    pub fn new() -> Self {
        Self {
            type_channel: ChannelId::Color.index(),
            palette: crate::meshers::cubes::palette::ColorPalette::default(),
            color_mode: CubesColorMode::default(),
            greedy: true,
        }
    }

    pub fn with_palette(mut self, palette: crate::meshers::cubes::palette::ColorPalette) -> Self {
        self.palette = palette;
        self
    }

    pub fn with_color_mode(mut self, mode: CubesColorMode) -> Self {
        self.color_mode = mode;
        self
    }

    pub fn with_greedy(mut self, greedy: bool) -> Self {
        self.greedy = greedy;
        self
    }

    pub fn with_type_channel(mut self, channel: usize) -> Self {
        self.type_channel = channel;
        self
    }

    /// The configured color mode (upstream `COLOR_*` constant).
    pub fn color_mode(&self) -> CubesColorMode {
        self.color_mode
    }

    /// The configured palette (used by the [`CubesColorMode::Palette`] and
    /// [`CubesColorMode::ShaderPalette`] modes).
    pub fn palette(&self) -> &crate::meshers::cubes::palette::ColorPalette {
        &self.palette
    }

    /// Whether the greedy rectangle-merging path is active.
    pub fn is_greedy(&self) -> bool {
        self.greedy
    }

    /// Extract the typed-channel slice the cubes mesher expects (`&[u32]`). The
    /// C++ runtime packs voxels as `u32` regardless of the on-disk depth.
    ///
    /// B5 (audit §9.6-B5): dispatch the channel depth **once** and widen the
    /// typed slice into `Vec<u32>` in a single pass (`Vec::iter().map().collect()`)
    /// instead of calling `get_voxel` (per-voxel depth dispatch) per voxel.
    fn extract_voxel_slice(buffer: &VoxelBuffer, channel: usize) -> Vec<u32> {
        use crate::storage::{ChannelData, Compression};
        let size = buffer.size();
        let cap = (size.x as usize) * (size.y as usize) * (size.z as usize);
        // Uniform channel: materialise a full Vec of the default value (the
        // greedy/simple mesher indexes the slice directly, so it must be sized).
        if buffer.channel_compression(channel) == Compression::Uniform {
            return vec![buffer.channel_default(channel) as u32; cap];
        }
        match buffer.channel_data(channel) {
            ChannelData::U8(v) => v.iter().map(|&x| x as u32).collect(),
            ChannelData::U16(v) => v.iter().map(|&x| x as u32).collect(),
            ChannelData::U32(v) => {
                let mut out = Vec::with_capacity(cap.max(v.len()));
                out.extend_from_slice(v);
                out
            }
            ChannelData::U64(v) => v.iter().map(|&x| x as u32).collect(),
        }
    }

    /// Resolve the voxel slice the cubes free function consumes (`&[u32]`).
    /// When the channel is materialized 32-bit, returns a zero-copy borrow of
    /// the backing store (mirrors C++ `raw_channel.reinterpret_cast_to<const
    /// uint32_t>()`, `voxel_mesher_cubes.cpp:844`); otherwise falls back to a
    /// single-pass widen/copy over the `ChannelData` variants
    /// ([`Self::extract_voxel_slice`]).
    fn voxel_slice<'a>(buffer: &'a VoxelBuffer, channel: usize) -> std::borrow::Cow<'a, [u32]> {
        if let Some(slice) = buffer.channel_typed_slice::<u32>(channel) {
            std::borrow::Cow::Borrowed(slice)
        } else {
            std::borrow::Cow::Owned(Self::extract_voxel_slice(buffer, channel))
        }
    }
}

impl VoxelMesher for CubesMesher {
    fn build(&self, output: &mut MesherOutput, input: &MesherInput<'_>) {
        use crate::meshers::cubes::greedy::MATERIAL_COUNT;
        let voxels = Self::voxel_slice(input.voxels, self.type_channel);
        let size = input.voxels.size();
        let block_size = [size.x, size.y, size.z];

        // CUBES-1 parity: dispatch color function by color_mode. C++ defaults
        // to COLOR_RAW which interprets the voxel value as packed RGBA.
        let depth = input.voxels.channel_depth(self.type_channel);
        let palette = self.palette.clone();
        let color_mode = self.color_mode;
        let color_func = move |raw: u32| match color_mode {
            CubesColorMode::Raw => match depth {
                crate::storage::ChannelDepth::Bit8 => crate::math::Color8::from_u8(raw as u8),
                crate::storage::ChannelDepth::Bit16 => crate::math::Color8::from_u16(raw as u16),
                _ => crate::math::Color8::from_u32(raw),
            },
            CubesColorMode::Palette => palette.get_color8(raw as u8),
            // Raw index in red, palette alpha in alpha; equal indices (with
            // equal palette alpha) still greedy-merge, matching upstream.
            CubesColorMode::ShaderPalette => {
                let entry = palette.get_color8(raw as u8);
                crate::math::Color8::new(raw as u8, 0, 0, entry.a)
            }
        };

        let mut arrays: [crate::meshers::cubes::arrays::CubesArrays; MATERIAL_COUNT] =
            Default::default();
        if self.greedy {
            crate::meshers::cubes::greedy::build_greedy_cubes(
                &mut arrays,
                &voxels,
                block_size,
                color_func,
            );
        } else {
            crate::meshers::cubes::simple::build_simple_cubes(
                &mut arrays,
                &voxels,
                block_size,
                color_func,
            );
        }

        // Two surfaces: opaque (index 0) and transparent (index 1). Both are
        // always emitted to match the C++ `Output::surfaces` shape; the
        // terrain layer can drop empty ones.
        output.surfaces.push(Surface::new(
            SurfaceArrays::Cubes(std::mem::take(&mut arrays[0])),
            0,
        ));
        output.surfaces.push(Surface::new(
            SurfaceArrays::Cubes(std::mem::take(&mut arrays[1])),
            1,
        ));
    }

    fn minimum_padding(&self) -> u32 {
        crate::meshers::cubes::greedy::PADDING as u32
    }

    fn maximum_padding(&self) -> u32 {
        crate::meshers::cubes::greedy::PADDING as u32
    }

    fn used_channels_mask(&self) -> u32 {
        1u32 << self.type_channel
    }

    fn supports_lod(&self) -> bool {
        false
    }
}

/// Voxel-model blocky mesher wrapping the existing `blocky::mesher::generate_mesh`
/// free function. Reads the `Type` channel as 16-bit voxel ids, looks up each
/// id in a [`BakedLibrary`] (voxel model library + side-culling matrix), and
/// emits one surface per material the library uses.
///
/// The library is shared via `Arc` so multiple terrain instances can use the
/// same baked data without re-baking. A library built with
/// [`BakedLibrary::default`] is empty (no models); callers must populate it
/// and run [`blocky::bake_library`] before passing it in.
pub struct BlockyMesher {
    type_channel: usize,
    library: std::sync::Arc<crate::meshers::blocky::baked_library::BakedLibrary>,
    /// 0fps-style corner ambient occlusion toggle (C++ default `true`).
    bake_occlusion: bool,
    /// AO strength (C++ default `0.8`).
    baked_occlusion_darkness: f32,
}

impl BlockyMesher {
    /// Build a mesher around a pre-baked library. The library must already
    /// have run `blocky::bake_library` so its side-culling matrix is valid.
    pub fn new(
        library: std::sync::Arc<crate::meshers::blocky::baked_library::BakedLibrary>,
    ) -> Self {
        Self {
            type_channel: ChannelId::Type.index(),
            library,
            bake_occlusion: true,
            baked_occlusion_darkness: 0.8,
        }
    }

    pub fn with_type_channel(mut self, channel: usize) -> Self {
        self.type_channel = channel;
        self
    }

    pub fn with_occlusion(mut self, enabled: bool, darkness: f32) -> Self {
        self.bake_occlusion = enabled;
        // BLOCKY-1 parity: clamp to [0,1] like C++ setter.
        self.baked_occlusion_darkness = darkness.clamp(0.0, 1.0);
        self
    }

    /// Fallback per-voxel narrowing (`u64 → u16`) used for the rare Bit32 and
    /// Bit64 Type channels: `build_blocky_into`'s main path takes zero-copy
    /// `ChannelData::U8`/`U16` borrows, while these depths narrow through one
    /// `get_voxel` pass (the blocky mesher only reads the low 16 bits,
    /// matching C++).
    fn extract_voxel_slice(buffer: &VoxelBuffer, channel: usize) -> Vec<u16> {
        let size = buffer.size();
        let mut out = Vec::with_capacity((size.x as usize) * (size.y as usize) * (size.z as usize));
        for z in 0..size.z {
            for x in 0..size.x {
                for y in 0..size.y {
                    out.push(buffer.get_voxel(x, y, z, channel) as u16);
                }
            }
        }
        out
    }
}

impl VoxelMesher for BlockyMesher {
    fn build(&self, output: &mut MesherOutput, input: &MesherInput<'_>) {
        let size = input.voxels.size();
        let material_count = self.library.indexed_materials_count.max(1) as usize;
        let mut arrays: Vec<crate::meshers::blocky::mesher::BlockyArrays> =
            (0..material_count).map(|_| Default::default()).collect();
        if input.collision_hint {
            let mut collision_arrays = crate::meshers::blocky::mesher::BlockyArrays::default();
            build_blocky_into(
                &mut arrays,
                Some(&mut collision_arrays),
                input.voxels,
                self.type_channel,
                size,
                &self.library,
                self.bake_occlusion,
                self.baked_occlusion_darkness,
            );
            output.collision_surface.positions = collision_arrays.positions;
            output.collision_surface.indices = collision_arrays.indices;
        } else {
            build_blocky_into(
                &mut arrays,
                None,
                input.voxels,
                self.type_channel,
                size,
                &self.library,
                self.bake_occlusion,
                self.baked_occlusion_darkness,
            );
        }
        for (material_index, arrays) in arrays.into_iter().enumerate() {
            output.surfaces.push(Surface::new(
                SurfaceArrays::Blocky(arrays),
                material_index as u16,
            ));
        }
    }

    fn minimum_padding(&self) -> u32 {
        crate::meshers::blocky::mesher::PADDING as u32
    }

    fn maximum_padding(&self) -> u32 {
        crate::meshers::blocky::mesher::PADDING as u32
    }

    fn used_channels_mask(&self) -> u32 {
        1u32 << self.type_channel
    }

    fn supports_lod(&self) -> bool {
        false
    }

    fn is_generating_collision_surface(&self) -> bool {
        true
    }
}

/// B5 (audit §9.6-B5): dispatch the blocky Type channel depth **once** and feed
/// `generate_mesh` a typed slice directly (`generate_mesh<T: Copy + Into<u16>>`
/// is generic), avoiding the per-voxel `get_voxel` copy into a fresh `Vec`.
/// For the common depths `Bit8`/`Bit16` this is zero-copy; for the rarer
/// `Bit32`/`Bit64` channels the values are narrowed into a temporary `Vec<u16>`
/// (the blocky mesher only reads the low 16 bits anyway, matching C++).
#[allow(clippy::too_many_arguments)] // mirrors the mesher + library config surface
fn build_blocky_into(
    arrays: &mut [crate::meshers::blocky::mesher::BlockyArrays],
    collision_arrays: Option<&mut crate::meshers::blocky::mesher::BlockyArrays>,
    buffer: &VoxelBuffer,
    type_channel: usize,
    size: Vector3i,
    library: &crate::meshers::blocky::baked_library::BakedLibrary,
    bake_occlusion: bool,
    baked_occlusion_darkness: f32,
) {
    use crate::meshers::blocky::mesher::{generate_mesh, generate_mesh_with_collision};
    use crate::storage::{ChannelData, Compression};
    let cap = (size.x as usize) * (size.y as usize) * (size.z as usize);
    // Uniform channel: `generate_mesh` indexes the slice directly (no bounds
    // check), so materialise a full Vec of the default value — matches the
    // Cubes fallback and the pre-B5 per-voxel `get_voxel` semantics.
    if buffer.channel_compression(type_channel) == Compression::Uniform {
        let voxels: Vec<u16> = vec![buffer.channel_default(type_channel) as u16; cap];
        match collision_arrays {
            Some(col) => generate_mesh_with_collision(
                arrays,
                col,
                &voxels,
                size,
                library,
                bake_occlusion,
                baked_occlusion_darkness,
            ),
            None => generate_mesh(
                arrays,
                &voxels,
                size,
                library,
                bake_occlusion,
                baked_occlusion_darkness,
            ),
        }
        return;
    }
    let data = buffer.channel_data(type_channel);
    match data {
        // Zero-copy fast paths: the typed storage variant already holds the
        // widths `generate_mesh` consumes via `T: Into<u16>`.
        ChannelData::U8(v) => match collision_arrays {
            Some(col) => generate_mesh_with_collision(
                arrays,
                col,
                v,
                size,
                library,
                bake_occlusion,
                baked_occlusion_darkness,
            ),
            None => generate_mesh(
                arrays,
                v,
                size,
                library,
                bake_occlusion,
                baked_occlusion_darkness,
            ),
        },
        ChannelData::U16(v) => match collision_arrays {
            Some(col) => generate_mesh_with_collision(
                arrays,
                col,
                v,
                size,
                library,
                bake_occlusion,
                baked_occlusion_darkness,
            ),
            None => generate_mesh(
                arrays,
                v,
                size,
                library,
                bake_occlusion,
                baked_occlusion_darkness,
            ),
        },
        // Rare Bit32/Bit64 channels: narrow to u16 in one pass (still a single
        // allocation, depth dispatched once — not per voxel).
        ChannelData::U32(_) | ChannelData::U64(_) => {
            let voxels = BlockyMesher::extract_voxel_slice(buffer, type_channel);
            match collision_arrays {
                Some(col) => generate_mesh_with_collision(
                    arrays,
                    col,
                    &voxels,
                    size,
                    library,
                    bake_occlusion,
                    baked_occlusion_darkness,
                ),
                None => generate_mesh(
                    arrays,
                    &voxels,
                    size,
                    library,
                    bake_occlusion,
                    baked_occlusion_darkness,
                ),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        checked_regular_collision_prefix_ends, BlockyMesher, CubesColorMode, CubesMesher,
        MeshArrays, TransvoxelMesher, TRANSVOXEL_SAMPLE_COUNT,
    };
    use crate::constants::cube_tables::{Side, CORNER_POSITION, SIDE_CORNERS, SIDE_QUAD_TRIANGLES};
    use crate::math::{Vector2f, Vector3f, Vector3i};
    use crate::meshers::blocky::baked_library::{BakedLibrary, BakedModel};
    use crate::meshers::blocky::SideSurface;
    use crate::meshers::transvoxel::{MAX_PADDING, MIN_PADDING};
    use crate::meshers::{MesherInput, MesherOutput, SurfaceArrays, VoxelMesher};
    use crate::storage::{ChannelDepth, ChannelId, VoxelBuffer, VoxelFormat};

    /// Build a `VoxelBuffer` of `inner³` voxels (padded with the transvoxel
    /// halo) containing an SDF sphere of `radius`, centred in the inner
    /// region. SDF convention: positive = outside (matches `SDF_FAR_OUTSIDE`).
    fn sphere_buffer(inner: i32, radius: f32) -> VoxelBuffer {
        let padded = inner + MIN_PADDING + MAX_PADDING;
        let mut buf = VoxelBuffer::with_size(Vector3i::splat(padded));
        let mut format = VoxelFormat::new();
        format.depths[ChannelId::Sdf.index()] = ChannelDepth::Bit32;
        format.configure_buffer(&mut buf);

        let centre = (inner as f32) * 0.5;
        for z in 0..padded {
            for x in 0..padded {
                for y in 0..padded {
                    let ix = x as f32 - MIN_PADDING as f32;
                    let iy = y as f32 - MIN_PADDING as f32;
                    let iz = z as f32 - MIN_PADDING as f32;
                    let distance =
                        ((ix - centre).powi(2) + (iy - centre).powi(2) + (iz - centre).powi(2))
                            .sqrt()
                            - radius;
                    buf.set_voxel_f(distance, x, y, z, ChannelId::Sdf.index());
                }
            }
        }
        buf
    }

    #[test]
    fn transvoxel_mesher_produces_substantial_geometry_for_sphere() {
        let mesher = TransvoxelMesher::new();
        let voxels = sphere_buffer(16, 6.0);
        let input = MesherInput::new(&voxels, Vector3i::zero(), 0);

        let mut output = MesherOutput::default();
        mesher.build(&mut output, &input);

        assert_eq!(output.surfaces.len(), 1);
        // The transvoxel sphere test in tests/transvoxel_sphere.rs asserts
        // > 100 vertices for the same configuration. Mirror that floor.
        let total_vertices = output.total_vertex_count();
        assert!(
            total_vertices > 100,
            "expected substantial mesh for an r=6 sphere in 16³ block, got {total_vertices}"
        );
        assert!(output.total_triangle_count() > 0);
    }

    #[test]
    fn transvoxel_mesher_emits_empty_surface_for_uniform_outside_volume() {
        // A buffer filled entirely with SDF_FAR_OUTSIDE (all air) has no
        // surface-crossing cells and produces an empty mesh.
        let mesher = TransvoxelMesher::new();
        let mut voxels = VoxelBuffer::with_size(Vector3i::splat(16));
        let mut format = VoxelFormat::new();
        format.depths[ChannelId::Sdf.index()] = ChannelDepth::Bit32;
        format.configure_buffer(&mut voxels);
        let input = MesherInput::new(&voxels, Vector3i::zero(), 0);

        let mut output = MesherOutput::default();
        mesher.build(&mut output, &input);

        assert_eq!(output.surfaces.len(), 1);
        assert_eq!(output.total_triangle_count(), 0);
    }

    #[test]
    fn transvoxel_mesher_fast_paths_uniform_sdf_without_sampling() {
        TRANSVOXEL_SAMPLE_COUNT.with(|samples| samples.set(0));

        let mesher = TransvoxelMesher::new();
        let mut voxels = VoxelBuffer::with_size(Vector3i::splat(16));
        let mut format = VoxelFormat::new();
        format.depths[ChannelId::Sdf.index()] = ChannelDepth::Bit32;
        format.configure_buffer(&mut voxels);
        assert!(voxels.is_uniform(ChannelId::Sdf.index()));

        let input = MesherInput::new(&voxels, Vector3i::zero(), 0);
        let mut output = MesherOutput::default();
        mesher.build(&mut output, &input);

        let sample_count = TRANSVOXEL_SAMPLE_COUNT.with(|samples| samples.get());
        assert_eq!(sample_count, 0);
        assert_eq!(output.surfaces.len(), 1);
        assert_eq!(output.total_triangle_count(), 0);
    }

    #[test]
    fn transvoxel_mesher_padding_matches_algorithm_constants() {
        let mesher = TransvoxelMesher::new();
        assert_eq!(mesher.minimum_padding(), MIN_PADDING as u32);
        assert_eq!(mesher.maximum_padding(), MAX_PADDING as u32);
        assert_eq!(mesher.used_channels_mask(), 1u32 << ChannelId::Sdf.index());
    }

    /// Verify the mesher is `Send + Sync` (required by `VoxelMesher`) so it
    /// can live behind `Arc<dyn VoxelMesher>` in MeshingDependency.
    #[test]
    fn transvoxel_mesher_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<TransvoxelMesher>();
    }

    /// The typed SDF-input path (B1) must produce byte-identical mesh arrays to
    /// the legacy dyn-dispatch path, for every supported SDF depth. This guards
    /// against sign-conversion / clamp regressions when the per-type
    /// `SdfSample::to_sdf_f32` is edited.
    #[test]
    fn transvoxel_typed_input_matches_dyn_input_per_depth() {
        use super::{SdfSample, TypedSdfInput, VoxelBufferTransvoxelInput};
        use crate::meshers::transvoxel::{build_regular_mesh, BuildRegularMeshParams};

        fn mesh_both<T>(voxels: &VoxelBuffer, channel: usize) -> (MeshArrays, MeshArrays)
        where
            T: SdfSample + bytemuck::Pod,
        {
            let params = BuildRegularMeshParams::default();
            let slice = voxels
                .channel_typed_slice::<T>(channel)
                .expect("typed slice must be available for a materialized channel");
            let typed_input = TypedSdfInput {
                slice,
                block_size: voxels.size(),
            };
            let mut cache_a = crate::meshers::transvoxel::Cache::default();
            let mut a = MeshArrays::default();
            build_regular_mesh(&typed_input, &params, &mut cache_a, &mut a);

            let dyn_input = VoxelBufferTransvoxelInput::new(voxels, channel);
            let mut cache_b = crate::meshers::transvoxel::Cache::default();
            let mut b = MeshArrays::default();
            build_regular_mesh(&dyn_input, &params, &mut cache_b, &mut b);
            (a, b)
        }

        let channel = ChannelId::Sdf.index();
        for depth in [ChannelDepth::Bit8, ChannelDepth::Bit16, ChannelDepth::Bit32] {
            // Build a sphere in a 16³ block (padded) at this depth.
            let padded = 16 + MIN_PADDING + MAX_PADDING;
            let mut voxels = VoxelBuffer::with_size(Vector3i::splat(padded));
            let mut format = VoxelFormat::new();
            format.depths[channel] = depth;
            format.configure_buffer(&mut voxels);
            let centre = 8.0f32;
            for z in 0..padded {
                for x in 0..padded {
                    for y in 0..padded {
                        let ix = x as f32 - MIN_PADDING as f32;
                        let iy = y as f32 - MIN_PADDING as f32;
                        let iz = z as f32 - MIN_PADDING as f32;
                        let d =
                            ((ix - centre).powi(2) + (iy - centre).powi(2) + (iz - centre).powi(2))
                                .sqrt()
                                - 6.0;
                        voxels.set_voxel_f(d, x, y, z, channel);
                    }
                }
            }
            let (typed, dyn_) = match depth {
                ChannelDepth::Bit8 => mesh_both::<i8>(&voxels, channel),
                ChannelDepth::Bit16 => mesh_both::<i16>(&voxels, channel),
                ChannelDepth::Bit32 => mesh_both::<f32>(&voxels, channel),
                ChannelDepth::Bit64 => unreachable!(),
            };
            assert_eq!(
                typed.vertices, dyn_.vertices,
                "vertex mismatch at {depth:?}"
            );
            assert_eq!(typed.normals, dyn_.normals, "normal mismatch at {depth:?}");
            assert_eq!(
                typed.lod_data, dyn_.lod_data,
                "lod data mismatch at {depth:?}"
            );
            assert_eq!(typed.indices, dyn_.indices, "index mismatch at {depth:?}");
            // And both must produce real geometry (sanity, not empty).
            assert!(!typed.vertices.is_empty(), "no vertices at {depth:?}");
        }
    }

    #[test]
    fn transvoxel_lod_hint_produces_transition_geometry() {
        // When lod_hint=true, TransvoxelMesher should build transition meshes
        // on the 6 faces, producing more vertices than regular-only. Use a
        // large sphere that intersects the block boundary so transition cells
        // actually cross the isolevel.
        let mesher = TransvoxelMesher::new();
        let voxels = sphere_buffer(16, 12.0);

        let mut input_no_lod = MesherInput::new(&voxels, Vector3i::zero(), 0);
        input_no_lod.lod_hint = false;
        let mut out_no_lod = MesherOutput::default();
        mesher.build(&mut out_no_lod, &input_no_lod);
        let verts_no_lod = out_no_lod.total_vertex_count();

        let mut input_lod = MesherInput::new(&voxels, Vector3i::zero(), 0);
        input_lod.lod_hint = true;
        let mut out_lod = MesherOutput::default();
        mesher.build(&mut out_lod, &input_lod);
        let verts_lod = out_lod.total_vertex_count();

        // Transition meshes add vertices on the LOD seam faces.
        assert!(
            verts_lod > verts_no_lod,
            "lod_hint should produce more vertices (transition geometry): lod={verts_lod} vs no_lod={verts_no_lod}"
        );
    }

    /// `with_edge_clamp_margin` must reach `BuildRegularMeshParams`: clamping
    /// moves interpolated vertices away from cell corners, so a clamped build
    /// produces different vertex positions (same topology) than an unclamped
    /// one. Out-of-range margins are clamped to the [0, 0.5] contract.
    #[test]
    fn transvoxel_edge_clamp_margin_builder_reaches_build_params() {
        let voxels = sphere_buffer(16, 6.0);

        let build = |margin: f32| -> (f32, Vec<crate::math::Vector3f>) {
            let mesher = TransvoxelMesher::new().with_edge_clamp_margin(margin);
            assert_eq!(mesher.edge_clamp_margin(), margin.clamp(0.0, 0.5));
            let input = MesherInput::new(&voxels, Vector3i::zero(), 0);
            let mut output = MesherOutput::default();
            mesher.build(&mut output, &input);
            match &output.surfaces[0].arrays {
                SurfaceArrays::Transvoxel(arrays) => {
                    (mesher.edge_clamp_margin(), arrays.vertices.clone())
                }
                _ => unreachable!(),
            }
        };

        let unclamped = build(0.0);
        let clamped = build(0.5);
        assert!(!unclamped.1.is_empty() && !clamped.1.is_empty());
        assert_ne!(
            unclamped.1, clamped.1,
            "edge clamping must change the mesh (positions and/or reuse-driven count)"
        );
        // Values outside [0, 0.5] clamp to the boundaries.
        let over = build(2.0);
        assert_eq!(over.0, 0.5);
        assert_eq!(over.1, clamped.1);
        let negative = build(-1.0);
        assert_eq!(negative.0, 0.0);
        assert_eq!(negative.1, unclamped.1);
        // Core default stays 0.0 (the parity goldens were generated with it).
        assert_eq!(TransvoxelMesher::new().edge_clamp_margin(), 0.0);
    }

    /// `with_transitions_enabled(false)` suppresses the lod_hint transition
    /// passes: the output must equal a plain regular-only build.
    #[test]
    fn transvoxel_transitions_disabled_suppresses_transition_geometry() {
        let voxels = sphere_buffer(16, 12.0);

        let mut input_regular = MesherInput::new(&voxels, Vector3i::zero(), 0);
        input_regular.lod_hint = false;
        let mut out_regular = MesherOutput::default();
        TransvoxelMesher::new().build(&mut out_regular, &input_regular);

        let mesher = TransvoxelMesher::new().with_transitions_enabled(false);
        assert!(!mesher.transitions_enabled());
        let mut input_lod = MesherInput::new(&voxels, Vector3i::zero(), 0);
        input_lod.lod_hint = true;
        let mut out_lod = MesherOutput::default();
        mesher.build(&mut out_lod, &input_lod);

        assert_eq!(
            out_lod.total_vertex_count(),
            out_regular.total_vertex_count(),
            "disabled transitions must reduce lod_hint output to the regular mesh"
        );
        // Default stays enabled (parity goldens and the terrain LOD runtime
        // rely on it).
        assert!(TransvoxelMesher::new().transitions_enabled());
    }

    /// `build_transition_mesh_for_direction` produces only the transition
    /// geometry for one side: fewer vertices than the full lod_hint build, and
    /// nothing for a uniform (all-air) buffer or an invalid direction.
    #[test]
    fn transvoxel_transition_only_build_covers_single_direction() {
        let mesher = TransvoxelMesher::new();
        let voxels = sphere_buffer(16, 12.0);

        let input = MesherInput::new(&voxels, Vector3i::zero(), 0);
        let arrays = mesher.build_transition_mesh_for_direction(&input, 0);
        assert!(
            !arrays.vertices.is_empty(),
            "a boundary-crossing sphere should produce transition vertices on side 0"
        );

        let mut air = VoxelBuffer::with_size(Vector3i::splat(16));
        let mut format = VoxelFormat::new();
        format.depths[ChannelId::Sdf.index()] = ChannelDepth::Bit32;
        format.configure_buffer(&mut air);
        let air_input = MesherInput::new(&air, Vector3i::zero(), 0);
        assert!(mesher
            .build_transition_mesh_for_direction(&air_input, 0)
            .vertices
            .is_empty());
        // Out-of-range directions yield empty arrays instead of panicking.
        let bad = MesherInput::new(&voxels, Vector3i::zero(), 0);
        assert!(mesher
            .build_transition_mesh_for_direction(&bad, 6)
            .vertices
            .is_empty());
    }

    #[test]
    fn transvoxel_collision_hint_populates_collision_submesh_range() {
        let mesher = TransvoxelMesher::new();
        let voxels = sphere_buffer(16, 6.0);
        let mut input = MesherInput::new(&voxels, Vector3i::zero(), 0);
        input.collision_hint = true;

        let mut output = MesherOutput::default();
        mesher.build(&mut output, &input);

        assert!(mesher.is_generating_collision_surface());
        assert!(output.total_triangle_count() > 0);
        assert!(output.collision_surface.submesh_vertex_end > 0);
        assert!(output.collision_surface.submesh_index_end > 0);
    }

    #[test]
    fn checked_collision_prefix_preserves_absent_sentinels_for_empty_or_overflow() {
        assert_eq!(checked_regular_collision_prefix_ends(8, 0), None);
        assert_eq!(
            checked_regular_collision_prefix_ends(12, 18),
            Some((12, 18))
        );

        let too_large = (i32::MAX as usize) + 1;
        assert_eq!(checked_regular_collision_prefix_ends(too_large, 18), None);
        assert_eq!(checked_regular_collision_prefix_ends(12, too_large), None);
    }

    #[test]
    fn collision_prefix_excludes_transition_indices() {
        let mesher = TransvoxelMesher::new();
        let voxels = sphere_buffer(16, 12.0);
        let mut input = MesherInput::new(&voxels, Vector3i::zero(), 0);
        input.collision_hint = true;
        input.lod_hint = true;

        let mut output = MesherOutput::default();
        mesher.build(&mut output, &input);

        let arrays = match &output.surfaces[0].arrays {
            SurfaceArrays::Transvoxel(arrays) => arrays,
            _ => unreachable!(),
        };
        let vertex_end = usize::try_from(output.collision_surface.submesh_vertex_end).unwrap();
        let index_end = usize::try_from(output.collision_surface.submesh_index_end).unwrap();
        assert!(arrays.vertices.len() > vertex_end);
        assert!(arrays.indices.len() > index_end);
        assert!(arrays.indices[..index_end]
            .iter()
            .all(|&index| usize::try_from(index).is_ok_and(|index| index < vertex_end)));
    }

    #[test]
    fn every_transvoxel_vertex_has_one_custom0_rgba() {
        let mesher = TransvoxelMesher::new();
        let voxels = sphere_buffer(16, 12.0);
        let mut input = MesherInput::new(&voxels, Vector3i::zero(), 0);
        input.lod_hint = true;

        let mut output = MesherOutput::default();
        mesher.build(&mut output, &input);

        let arrays = match &output.surfaces[0].arrays {
            SurfaceArrays::Transvoxel(arrays) => arrays,
            _ => unreachable!(),
        };
        let custom0 = arrays
            .lod_data
            .iter()
            .copied()
            .map(|attrib| attrib.custom0_rgba())
            .collect::<Vec<_>>();
        assert_eq!(custom0.len(), arrays.vertices.len());
    }

    #[test]
    fn empty_transvoxel_collision_prefix_keeps_absent_sentinels() {
        let mesher = TransvoxelMesher::new();
        let mut voxels = VoxelBuffer::with_size(Vector3i::splat(16));
        let mut format = VoxelFormat::new();
        format.depths[ChannelId::Sdf.index()] = ChannelDepth::Bit32;
        format.configure_buffer(&mut voxels);
        let mut input = MesherInput::new(&voxels, Vector3i::zero(), 0);
        input.collision_hint = true;
        input.lod_hint = true;

        let mut output = MesherOutput::default();
        mesher.build(&mut output, &input);

        assert!(output.is_empty());
        assert_eq!(output.collision_surface.submesh_vertex_end, -1);
        assert_eq!(output.collision_surface.submesh_index_end, -1);
    }

    /// Vertex positions should land in world space (origin_in_voxels offset
    /// applied). The transvoxel free function uses scaled block-local coords;
    /// the wrapper currently forwards them as-is — so for `lod_index==0` and
    /// a zero origin, positions should still be non-negative within the block.
    #[test]
    fn transvoxel_mesher_vertex_positions_are_within_block_for_zero_origin() {
        let mesher = TransvoxelMesher::new();
        let voxels = sphere_buffer(16, 6.0);
        let input = MesherInput::new(&voxels, Vector3i::zero(), 0);

        let mut output = MesherOutput::default();
        mesher.build(&mut output, &input);

        let arrays = match &output.surfaces[0].arrays {
            SurfaceArrays::Transvoxel(a) => a,
            _ => unreachable!(),
        };
        let padded_extent = (16 + MIN_PADDING + MAX_PADDING) as f32;
        assert!(arrays.vertices.iter().all(|p| {
            let Vector3f { x, y, z } = *p;
            x >= 0.0
                && y >= 0.0
                && z >= 0.0
                && x < padded_extent
                && y < padded_extent
                && z < padded_extent
        }));
    }

    /// Build a small VoxelBuffer filled with a single solid voxel type in the
    /// interior and air (0) on the padding halo — the typical input
    /// `CubesMesher` sees after the gather step.
    fn cubes_input_buffer() -> VoxelBuffer {
        // Padded block: PADDING(1) interior 2³ + PADDING(1) on each side.
        let mut voxels = VoxelBuffer::with_size(Vector3i::splat(2 + 2));
        let channel = ChannelId::Color.index();
        // Fill the interior (1..3)³ with voxel id 1.
        for z in 1..3 {
            for x in 1..3 {
                for y in 1..3 {
                    voxels.set_voxel(1, x, y, z, channel);
                }
            }
        }
        voxels
    }

    #[test]
    fn cubes_mesher_produces_two_surfaces_for_a_solid_block() {
        // Use Palette mode (the pre-CUBES-1 behavior) so the test voxel value
        // maps to opaque white via the default palette.
        let mesher = CubesMesher::new().with_color_mode(CubesColorMode::Palette);
        let voxels = cubes_input_buffer();
        let input = MesherInput::new(&voxels, Vector3i::zero(), 0);

        let mut output = MesherOutput::default();
        mesher.build(&mut output, &input);

        // Two surfaces emitted (opaque material 0, transparent material 1).
        assert_eq!(output.surfaces.len(), 2);
        // The opaque surface should have geometry for a solid 2³ block.
        let opaque_vertices = output.surfaces[0].arrays.vertex_count();
        assert!(
            opaque_vertices > 0,
            "expected opaque geometry for a solid block, got {opaque_vertices}"
        );
        // Material indices are 0 and 1 in order.
        assert_eq!(output.surfaces[0].material_index, 0);
        assert_eq!(output.surfaces[1].material_index, 1);
    }

    #[test]
    fn cubes_mesher_emits_empty_surfaces_for_air_block() {
        let mesher = CubesMesher::new();
        // All-zero Type channel → no solid voxels → no faces.
        let voxels = VoxelBuffer::with_size(Vector3i::splat(4));
        let input = MesherInput::new(&voxels, Vector3i::zero(), 0);

        let mut output = MesherOutput::default();
        mesher.build(&mut output, &input);

        assert_eq!(output.total_triangle_count(), 0);
    }

    #[test]
    fn cubes_mesher_padding_and_channels_match_constants() {
        let mesher = CubesMesher::new();
        assert_eq!(mesher.minimum_padding(), 1);
        assert_eq!(mesher.maximum_padding(), 1);
        assert_eq!(
            mesher.used_channels_mask(),
            1u32 << ChannelId::Color.index()
        );
    }

    #[test]
    fn cubes_mesher_reports_lod_unsupported_until_lod_inputs_are_used() {
        let mesher = CubesMesher::new();

        assert!(!mesher.supports_lod());
    }

    #[test]
    fn cubes_mesher_supports_non_greedy_simple_path() {
        let mesher = CubesMesher::new().with_greedy(false);
        let voxels = cubes_input_buffer();
        let input = MesherInput::new(&voxels, Vector3i::zero(), 0);

        let mut output = MesherOutput::default();
        mesher.build(&mut output, &input);

        // Simple path emits one quad per face — more vertices than greedy,
        // but still non-empty for a solid block.
        assert!(output.total_triangle_count() > 0);
    }

    #[test]
    fn cubes_mesher_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<CubesMesher>();
    }

    /// Upstream `COLOR_SHADER_PALETTE`: voxel values are written raw into the
    /// red component (green/blue zero) and the palette entry's alpha is copied
    /// into alpha, so transparent palette entries sort into the transparent
    /// material surface.
    #[test]
    fn cubes_shader_palette_writes_index_in_red_and_palette_alpha() {
        let mut palette = crate::meshers::cubes::palette::ColorPalette::default();
        palette.set_color8(1, crate::math::Color8::new(9, 9, 9, 255));
        palette.set_color8(2, crate::math::Color8::new(9, 9, 9, 128));

        let build = |voxel_id: u64| -> MesherOutput {
            let mut voxels = VoxelBuffer::with_size(Vector3i::splat(2 + 2));
            let channel = ChannelId::Color.index();
            for z in 1..3 {
                for x in 1..3 {
                    for y in 1..3 {
                        voxels.set_voxel(voxel_id, x, y, z, channel);
                    }
                }
            }
            let mesher = CubesMesher::new()
                .with_color_mode(CubesColorMode::ShaderPalette)
                .with_palette(palette.clone());
            let input = MesherInput::new(&voxels, Vector3i::zero(), 0);
            let mut output = MesherOutput::default();
            mesher.build(&mut output, &input);
            output
        };

        let colors_of = |output: &MesherOutput, surface: usize| -> Vec<crate::math::Color> {
            match &output.surfaces[surface].arrays {
                SurfaceArrays::Cubes(arrays) => arrays.colors.clone(),
                _ => unreachable!(),
            }
        };

        // Opaque palette entry (alpha 255) → opaque surface, index in red.
        let opaque = build(1);
        let opaque_colors = colors_of(&opaque, 0);
        assert!(!opaque_colors.is_empty());
        for c in &opaque_colors {
            assert!((c.r - 1.0 / 255.0).abs() < 1e-4, "red carries the index");
            assert_eq!(c.g, 0.0);
            assert_eq!(c.b, 0.0);
            assert!((c.a - 1.0).abs() < 1e-4, "alpha carries the palette alpha");
        }
        assert!(colors_of(&opaque, 1).is_empty());

        // Partial-alpha palette entry → transparent surface.
        let transparent = build(2);
        assert!(colors_of(&transparent, 0).is_empty());
        let transparent_colors = colors_of(&transparent, 1);
        assert!(!transparent_colors.is_empty());
        for c in &transparent_colors {
            assert!((c.r - 2.0 / 255.0).abs() < 1e-4);
            assert!((c.a - 128.0 / 255.0).abs() < 1e-4);
        }
    }

    fn empty_blocky_library() -> std::sync::Arc<crate::meshers::blocky::baked_library::BakedLibrary>
    {
        // Default-constructed library is empty (no models), so the mesher
        // emits no geometry regardless of input. Useful for testing the
        // adapter wiring without pulling in the full bake pass.
        std::sync::Arc::new(crate::meshers::blocky::baked_library::BakedLibrary::default())
    }

    fn full_cube_side_surface(side: usize) -> SideSurface {
        let positions: Vec<Vector3f> = SIDE_CORNERS[side]
            .iter()
            .map(|&corner| CORNER_POSITION[corner])
            .collect();
        let indices = SIDE_QUAD_TRIANGLES[side].to_vec();
        let uvs = vec![
            Vector2f::new(0.0, 0.0),
            Vector2f::new(1.0, 0.0),
            Vector2f::new(1.0, 1.0),
            Vector2f::new(0.0, 1.0),
        ];
        SideSurface {
            positions,
            uvs,
            indices,
            tangents: Vec::new(),
        }
    }

    fn full_cube_blocky_library(collision_enabled: bool) -> std::sync::Arc<BakedLibrary> {
        let air = BakedModel::default();
        let mut cube = BakedModel {
            empty: false,
            culls_neighbors: true,
            contributes_to_ao: true,
            ..Default::default()
        };
        cube.model.surface_count = 1;
        cube.model.surfaces[0].material_id = 0;
        cube.model.surfaces[0].collision_enabled = collision_enabled;
        for side in 0..Side::COUNT {
            cube.model.sides_surfaces[side][0] = full_cube_side_surface(side);
        }
        let mut library = BakedLibrary {
            models: vec![air, cube],
            indexed_materials_count: 1,
            ..Default::default()
        };
        crate::meshers::blocky::bake_library(&mut library);
        std::sync::Arc::new(library)
    }

    fn blocky_input_buffer() -> VoxelBuffer {
        let mut voxels = VoxelBuffer::with_size(Vector3i::splat(3));
        voxels.set_voxel(1, 1, 1, 1, ChannelId::Type.index());
        voxels
    }

    #[test]
    fn blocky_mesher_with_empty_library_emits_no_geometry() {
        let mesher = BlockyMesher::new(empty_blocky_library());
        // Solid block of voxel id 1 — but the library has no model for it,
        // so nothing gets emitted.
        let mut voxels = VoxelBuffer::with_size(Vector3i::splat(4));
        for z in 1..3 {
            for x in 1..3 {
                for y in 1..3 {
                    voxels.set_voxel(1, x, y, z, ChannelId::Type.index());
                }
            }
        }
        let input = MesherInput::new(&voxels, Vector3i::zero(), 0);

        let mut output = MesherOutput::default();
        mesher.build(&mut output, &input);

        assert_eq!(output.total_triangle_count(), 0);
    }

    #[test]
    fn blocky_mesher_handles_uniform_air_block_without_panicking() {
        // Regression for B5: a never-written-to buffer has a `Compression::Uniform`
        // Type channel (empty typed Vec). Before the uniform fallback in
        // `build_blocky_into` this panicked in `generate_mesh` (index out of
        // bounds on an empty slice). Mirrors `cubes_mesher_emits_empty_surfaces_for_air_block`.
        let mesher = BlockyMesher::new(empty_blocky_library());
        let voxels = VoxelBuffer::with_size(Vector3i::splat(4));
        let input = MesherInput::new(&voxels, Vector3i::zero(), 0);

        let mut output = MesherOutput::default();
        mesher.build(&mut output, &input);

        assert_eq!(output.total_triangle_count(), 0);
    }

    #[test]
    fn blocky_mesher_padding_and_channels_match_constants() {
        let mesher = BlockyMesher::new(empty_blocky_library());
        assert_eq!(mesher.minimum_padding(), 1);
        assert_eq!(mesher.maximum_padding(), 1);
        assert_eq!(mesher.used_channels_mask(), 1u32 << ChannelId::Type.index());
    }

    /// The blocky mesher's two configuration knobs are genuinely honored:
    /// geometry only appears for models present in the attached library, and
    /// enabling occlusion darkens corner vertices relative to AO-disabled
    /// output (same vertex count, different colors).
    #[test]
    fn blocky_core_mesher_honors_occlusion_and_library() {
        let library = full_cube_blocky_library(true);
        // An L-shaped cluster: the top face of (1,1,1) stays visible (air
        // above) while (2,2,1) laterally occludes its +x edge, so ambient
        // occlusion actually darkens something. A lone voxel or a full block
        // would expose only faces whose occluder ring is entirely air.
        let mut voxels = VoxelBuffer::with_size(Vector3i::splat(4));
        for (x, y, z) in [(1, 1, 1), (2, 1, 1), (2, 2, 1), (1, 1, 2)] {
            voxels.set_voxel(1, x, y, z, ChannelId::Type.index());
        }
        let mut input = MesherInput::new(&voxels, Vector3i::zero(), 0);
        input.collision_hint = false;

        // Library honored: the full-cube library meshes the lone voxel...
        let mut full = MesherOutput::default();
        BlockyMesher::new(library.clone()).build(&mut full, &input);
        assert!(full.total_triangle_count() > 0);

        // ...while an empty (default) library emits nothing.
        let mut empty = MesherOutput::default();
        BlockyMesher::new(empty_blocky_library()).build(&mut empty, &input);
        assert_eq!(empty.total_triangle_count(), 0);

        // Occlusion honored: same topology, but AO darkens some vertices.
        let colors_with = |occlusion: bool| -> Vec<crate::math::Color> {
            let mesher = BlockyMesher::new(library.clone()).with_occlusion(occlusion, 0.8);
            let mut output = MesherOutput::default();
            mesher.build(&mut output, &input);
            match &output.surfaces[0].arrays {
                SurfaceArrays::Blocky(arrays) => arrays.colors.clone(),
                _ => unreachable!(),
            }
        };
        let ao_off = colors_with(false);
        let ao_on = colors_with(true);
        assert_eq!(ao_off.len(), ao_on.len());
        assert!(
            ao_on
                .iter()
                .zip(&ao_off)
                .any(|(with, without)| with != without),
            "enabled occlusion must darken at least one vertex"
        );
    }

    #[test]
    fn blocky_collision_hint_emits_enabled_surface_geometry() {
        let mesher = BlockyMesher::new(full_cube_blocky_library(true));
        let voxels = blocky_input_buffer();
        let mut input = MesherInput::new(&voxels, Vector3i::zero(), 0);
        input.collision_hint = true;

        let mut output = MesherOutput::default();
        mesher.build(&mut output, &input);

        assert!(mesher.is_generating_collision_surface());
        assert!(output.total_triangle_count() > 0);
        assert_eq!(output.collision_surface.positions.len(), 24);
        assert_eq!(output.collision_surface.indices.len(), 36);
    }

    #[test]
    fn blocky_collision_hint_skips_surfaces_with_collision_disabled() {
        let mesher = BlockyMesher::new(full_cube_blocky_library(false));
        let voxels = blocky_input_buffer();
        let mut input = MesherInput::new(&voxels, Vector3i::zero(), 0);
        input.collision_hint = true;

        let mut output = MesherOutput::default();
        mesher.build(&mut output, &input);

        assert!(mesher.is_generating_collision_surface());
        assert!(output.total_triangle_count() > 0);
        assert!(output.collision_surface.positions.is_empty());
        assert!(output.collision_surface.indices.is_empty());
    }

    #[test]
    fn blocky_mesher_reports_lod_unsupported_until_lod_inputs_are_used() {
        let mesher = BlockyMesher::new(empty_blocky_library());

        assert!(!mesher.supports_lod());
    }

    #[test]
    fn blocky_mesher_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<BlockyMesher>();
    }

    /// The zero-copy Type path (B5) must produce identical mesh output to the
    /// per-voxel fallback. Bit16 dispatches `ChannelData::U16` directly
    /// (zero-copy); Bit64 narrows through the `extract_voxel_slice` per-voxel
    /// fallback. Same logical voxel values must mesh identically.
    #[test]
    fn blocky_mesher_zero_copy_path_matches_fallback() {
        let build = |depth: ChannelDepth| -> MesherOutput {
            let library = full_cube_blocky_library(true);
            let channel = ChannelId::Type.index();
            let mut voxels = VoxelBuffer::with_size(Vector3i::splat(4));
            let mut format = VoxelFormat::new();
            format.depths[channel] = depth;
            format.configure_buffer(&mut voxels);
            for z in 1..3 {
                for x in 1..3 {
                    for y in 1..3 {
                        voxels.set_voxel(1, x, y, z, channel);
                    }
                }
            }
            let mesher = BlockyMesher::new(library);
            let mut input = MesherInput::new(&voxels, Vector3i::zero(), 0);
            input.collision_hint = true;
            let mut output = MesherOutput::default();
            mesher.build(&mut output, &input);
            output
        };

        let zero_copy = build(ChannelDepth::Bit16);
        let fallback = build(ChannelDepth::Bit64);
        assert!(zero_copy.total_triangle_count() > 0);
        assert_eq!(zero_copy.surfaces.len(), fallback.surfaces.len());
        for (a, b) in zero_copy.surfaces.iter().zip(fallback.surfaces.iter()) {
            match (&a.arrays, &b.arrays) {
                (SurfaceArrays::Blocky(va), SurfaceArrays::Blocky(vb)) => {
                    assert_eq!(va.positions, vb.positions);
                    assert_eq!(va.normals, vb.normals);
                    assert_eq!(va.colors, vb.colors);
                    assert_eq!(va.indices, vb.indices);
                }
                _ => panic!("expected blocky surfaces"),
            }
        }
        assert_eq!(
            zero_copy.collision_surface.positions,
            fallback.collision_surface.positions
        );
        assert_eq!(
            zero_copy.collision_surface.indices,
            fallback.collision_surface.indices
        );
    }

    /// Same guard for the cubes path: a materialized 32-bit channel takes the
    /// zero-copy `Cow::Borrowed` route via `channel_typed_slice::<u32>`, while
    /// an 8-bit channel widens through the `extract_voxel_slice` fallback.
    /// Identical logical voxels must produce identical output (B5).
    #[test]
    fn cubes_mesher_zero_copy_path_matches_per_voxel_fallback() {
        let build = |depth: ChannelDepth| -> MesherOutput {
            let channel = ChannelId::Color.index();
            let mut voxels = VoxelBuffer::with_size(Vector3i::splat(4));
            let mut format = VoxelFormat::new();
            format.depths[channel] = depth;
            format.configure_buffer(&mut voxels);
            for z in 1..3 {
                for x in 1..3 {
                    for y in 1..3 {
                        voxels.set_voxel(1, x, y, z, channel);
                    }
                }
            }
            let mesher = CubesMesher::new().with_color_mode(CubesColorMode::Palette);
            let input = MesherInput::new(&voxels, Vector3i::zero(), 0);
            let mut output = MesherOutput::default();
            mesher.build(&mut output, &input);
            output
        };

        let zero_copy = build(ChannelDepth::Bit32);
        let fallback = build(ChannelDepth::Bit8);
        assert!(zero_copy.total_triangle_count() > 0);
        assert_eq!(zero_copy.surfaces.len(), fallback.surfaces.len());
        for (a, b) in zero_copy.surfaces.iter().zip(fallback.surfaces.iter()) {
            match (&a.arrays, &b.arrays) {
                (SurfaceArrays::Cubes(va), SurfaceArrays::Cubes(vb)) => {
                    assert_eq!(va.positions, vb.positions);
                    assert_eq!(va.normals, vb.normals);
                    assert_eq!(va.colors, vb.colors);
                    assert_eq!(va.indices, vb.indices);
                }
                _ => panic!("expected cubes surfaces"),
            }
        }
    }
}
