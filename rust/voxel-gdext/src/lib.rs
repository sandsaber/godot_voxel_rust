//! `voxel-gdext` — Godot 4 GDExtension bindings for the Rust voxel engine.
//!
//! This crate is the **thin binding layer**: the only place that depends on the
//! `godot` crate and exposes `#[func]`/`#[base]`/`#[signal]` symbols to GDScript.
//! All engine-agnostic logic lives in [`voxel_core`]; this crate wraps it into
//! Godot classes.
//!
//! ## Loading in Godot
//! Build the `.so`/`.dylib`/`.dll`, then add a `.gdextension` file pointing at
//! it (see `rust/voxel-gdext/voxel_gdext.gdextension.in`). Restart the editor.

mod debug_draw;
mod editor;
mod generators;
mod resources;
mod resources2;
mod resources3;
mod streams;
mod terrain;
mod voxel_buffer;
mod voxel_tool;

use godot::init::{gdextension, ExtensionLibrary, InitStage};
use godot::prelude::*;

/// The GDExtension entry point. Exactly one `ExtensionLibrary` impl per library;
/// `#[gdextension]` generates the four FFI symbols Godot looks for.
struct VoxelGdExt;

#[gdextension]
unsafe impl ExtensionLibrary for VoxelGdExt {
    fn on_stage_init(stage: InitStage) {
        if stage == InitStage::Scene {
            godot_print!(
                "voxel-gdext: Scene stage initialized (voxel-core v{})",
                voxel_core::VERSION
            );
        }
    }
}
