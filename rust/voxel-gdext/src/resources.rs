//! Additional Godot Resource classes for mesher/configuration types.
//!
//! These bring the class count closer to the DoD 75+ target by exposing
//! mesher and library types as Godot Resources.

use godot::prelude::*;

use voxel_core::math::Vector3i;

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
}

#[godot_api]
impl IResource for VoxelMesherTransvoxelGD {
    fn init(base: Base<Resource>) -> Self {
        Self {
            base,
            sdf_channel: 1,
        }
    }
}

#[godot_api]
impl VoxelMesherTransvoxelGD {
    /// Build the transvoxel mesh from a `VoxelBufferGD` and return the total
    /// vertex count. `buffer` must be a `VoxelBufferGD`; `lod_hint` toggles
    /// transition-cell generation on the +X/+Z seam faces.
    ///
    /// Returns -1 if `buffer` is not a `VoxelBufferGD`.
    #[func]
    fn build_vertex_count(&self, buffer: Gd<RefCounted>, lod_hint: bool) -> i64 {
        let Ok(buf) = buffer.try_cast::<crate::voxel_buffer::VoxelBufferGD>() else {
            return -1;
        };
        let bound = buf.bind();
        let mesher = voxel_core::meshers::TransvoxelMesher::new()
            .with_sdf_channel(self.sdf_channel.max(0) as usize);
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
        let Ok(buf) = buffer.try_cast::<crate::voxel_buffer::VoxelBufferGD>() else {
            return -1;
        };
        let bound = buf.bind();
        let mesher = voxel_core::meshers::TransvoxelMesher::new()
            .with_sdf_channel(self.sdf_channel.max(0) as usize);
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
    /// AO darkness factor (0..1).
    #[var]
    occlusion_darkness: f32,
    /// Type channel index.
    #[var]
    type_channel: i32,
}

#[godot_api]
impl IResource for VoxelMesherBlockyGD {
    fn init(base: Base<Resource>) -> Self {
        Self {
            base,
            bake_occlusion: true,
            occlusion_darkness: 0.8,
            type_channel: 0,
        }
    }
}

#[godot_api]
impl VoxelMesherBlockyGD {
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
        let Ok(buf) = buffer.try_cast::<crate::voxel_buffer::VoxelBufferGD>() else {
            return -1;
        };
        let bound = buf.bind();
        let lib = std::sync::Arc::new(voxel_core::meshers::blocky::BakedLibrary::default());
        let mesher = voxel_core::meshers::BlockyMesher::new(lib)
            .with_type_channel(self.type_channel.max(0) as usize);
        let input = voxel_core::meshers::MesherInput::new(bound.core_buffer(), Vector3i::zero(), 0);
        let mut output = voxel_core::meshers::MesherOutput::default();
        voxel_core::meshers::VoxelMesher::build(&mesher, &mut output, &input);
        output.total_vertex_count() as i64
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
}

#[godot_api]
impl IResource for VoxelMesherCubesGD {
    fn init(base: Base<Resource>) -> Self {
        Self {
            base,
            greedy: true,
            color_channel: 4,
        }
    }
}

#[godot_api]
impl VoxelMesherCubesGD {
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
            PackedInt32Array::from(&[c.r as i32, c.g as i32, c.b as i32, c.a as i32])
        } else {
            PackedInt32Array::from(&[0, 0, 0, 255])
        }
    }

    /// Clear all entries to transparent black.
    #[func]
    fn clear(&mut self) {
        self.palette.clear();
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
    /// The real baked model table.
    library: voxel_core::meshers::blocky::BakedLibrary,
}

#[godot_api]
impl IResource for VoxelBlockyLibraryGD {
    fn init(base: Base<Resource>) -> Self {
        Self {
            base,
            model_count: 0,
            library: voxel_core::meshers::blocky::BakedLibrary::default(),
        }
    }
}

#[godot_api]
impl VoxelBlockyLibraryGD {
    /// Append a solid-color model and return its index.
    #[func]
    fn add_solid_model(&mut self, r: f32, g: f32, b: f32) -> i32 {
        let model = voxel_core::meshers::blocky::BakedModel {
            color: voxel_core::math::Color::from_rgb(r, g, b),
            empty: false,
            culls_neighbors: true,
            ..voxel_core::meshers::blocky::BakedModel::default()
        };
        let idx = self.library.models.len() as i32;
        self.library.models.push(model);
        self.model_count = self.library.models.len() as i32;
        idx
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
    /// Number of background threads.
    #[var]
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
            let completed = runner.drain_completed_tasks();
            completed.len() as i32
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
}

#[godot_api]
impl IRefCounted for VoxelSaveCompletionTrackerGD {
    fn init(base: Base<RefCounted>) -> Self {
        Self {
            base,
            pending_count: 0,
            is_done: true,
        }
    }
}

#[godot_api]
impl VoxelSaveCompletionTrackerGD {
    /// Mark a save operation as started (increments pending_count).
    #[func]
    fn mark_pending(&mut self) {
        self.pending_count += 1;
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
}

#[godot_api]
impl IResource for VoxelInstanceLibraryGD {
    fn init(base: Base<Resource>) -> Self {
        Self {
            base,
            item_count: 0,
            library: voxel_core::instancing::InstanceLibrary::new(),
        }
    }
}

#[godot_api]
impl VoxelInstanceLibraryGD {
    /// Add a scatter item (name + density + scale range) and return its index.
    #[func]
    fn add_item(
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

    /// Number of registered items.
    #[func]
    fn get_item_count(&self) -> i32 {
        self.item_count
    }

    /// Whether the library has no items.
    #[func]
    fn is_empty(&self) -> bool {
        self.library.is_empty()
    }
}

// ---------------------------------------------------------------------------
// VoxelInstanceLibraryItemGD — Resource for one scatter item
// ---------------------------------------------------------------------------

/// One entry in a [`VoxelInstanceLibraryGD`]. Defines what to scatter and how.
/// The functional API produces a real
/// [`voxel_core::instancing::InstanceLibraryItem`] via `to_core_item`.
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
}
