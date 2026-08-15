use std::path::PathBuf;
use std::sync::Arc;

use godot::classes::ProjectSettings;
use godot::prelude::*;
use voxel_core::streams::{MemoryStream, RegionFilesStream, VoxelStream};

#[derive(Clone, Default)]
pub(crate) struct MemoryStreamHandle {
    stream: Arc<MemoryStream>,
}

impl MemoryStreamHandle {
    #[cfg(test)]
    pub(crate) fn typed_stream(&self) -> Arc<MemoryStream> {
        self.stream.clone()
    }

    pub(crate) fn core_stream(&self) -> Arc<dyn VoxelStream> {
        self.stream.clone()
    }

    fn block_count(&self) -> usize {
        self.stream.len()
    }

    fn clear(&self) {
        self.stream.clear();
    }
}

#[derive(GodotClass)]
#[class(base = Resource, tool)]
pub struct VoxelStreamMemory {
    base: Base<Resource>,
    handle: MemoryStreamHandle,
    /// Pinned `artificial_save_latency_usec` (backing field). Upstream default 0.
    artificial_save_latency_usec_value: i32,
    /// The pinned GDScript-facing `artificial_save_latency_usec` property.
    #[var(get = get_artificial_save_latency_usec, set = set_artificial_save_latency_usec)]
    artificial_save_latency_usec: PhantomVar<i32>,
}

#[godot_api]
impl IResource for VoxelStreamMemory {
    fn init(base: Base<Resource>) -> Self {
        Self {
            base,
            handle: MemoryStreamHandle::default(),
            artificial_save_latency_usec_value: 0,
            artificial_save_latency_usec: PhantomVar::default(),
        }
    }
}

#[godot_api]
impl VoxelStreamMemory {
    #[func]
    fn get_block_count(&self) -> i32 {
        i32::try_from(self.handle.block_count()).unwrap_or(i32::MAX)
    }

    #[func]
    fn clear(&self) {
        self.handle.clear();
    }

    pub(crate) fn core_stream(&self) -> Arc<dyn VoxelStream> {
        self.handle.core_stream()
    }

    // -----------------------------------------------------------------
    // Pinned VoxelStreamMemory properties
    // (upstream 5828cbeb: VoxelStreamMemory.xml).
    // -----------------------------------------------------------------

    /// Fakes long saving by making the calling thread sleep for some amount of
    /// time, in microseconds (upstream default 0).
    #[func]
    fn get_artificial_save_latency_usec(&self) -> i32 {
        self.artificial_save_latency_usec_value
    }

    #[func]
    fn set_artificial_save_latency_usec(&mut self, latency_usec: i32) {
        self.artificial_save_latency_usec_value = latency_usec;
    }
}

pub(crate) fn resolve_core_stream(resource: Gd<Resource>) -> Option<Arc<dyn VoxelStream>> {
    if let Ok(stream) = resource.clone().try_cast::<VoxelStreamMemory>() {
        return Some(stream.bind().core_stream());
    }
    if let Ok(stream) = resource.clone().try_cast::<VoxelStreamRegionFiles>() {
        return Some(stream.bind().core_stream());
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use voxel_core::math::Vector3i;
    use voxel_core::storage::VoxelBuffer;

    #[test]
    fn memory_handle_exposes_one_shared_stream() {
        let handle = MemoryStreamHandle::default();
        let core = handle.typed_stream();
        core.save_block(
            Vector3i::new(2, -1, 4),
            0,
            &VoxelBuffer::with_size(Vector3i::splat(1)),
        );

        assert_eq!(handle.block_count(), 1);
        handle.clear();
        assert_eq!(core.len(), 0);
    }
}

// ---------------------------------------------------------------------------
// VoxelStreamRegionFiles — disk persistence via .vxr region files
// ---------------------------------------------------------------------------

/// A Godot `Resource` that saves/loads voxel data to region files on disk.
/// Set the `directory` property to a writable folder, then assign this stream
/// to a [`VoxelTerrain`](crate::terrain::VoxelTerrain) to enable persistence.
///
/// The pinned GDScript-facing properties (`block_size_po2`, `directory`,
/// `region_size_po2`, `sector_size`) and methods (`convert_files`,
/// `get_region_size`) mirror upstream `VoxelStreamRegionFiles` (5828cbeb).
/// They are stored faithfully so GDScript reads round-trip.
#[derive(GodotClass)]
#[class(base = Resource, tool)]
pub struct VoxelStreamRegionFiles {
    base: Base<Resource>,
    /// Directory where `.vxr` region files are stored.
    #[var]
    directory: GString,
    /// Pinned `block_size_po2` (backing field). Upstream default 4.
    block_size_po2_value: i32,
    /// Pinned `region_size_po2` (backing field). Upstream default 4.
    region_size_po2_value: i32,
    /// Pinned `sector_size` (backing field). Upstream default 512.
    sector_size_value: i32,
    /// The pinned GDScript-facing `block_size_po2` property.
    #[var(get = get_block_size_po2, set = set_block_size_po2)]
    block_size_po2: PhantomVar<i32>,
    /// The pinned GDScript-facing `region_size_po2` property.
    #[var(get = get_region_size_po2, set = set_region_size_po2)]
    region_size_po2: PhantomVar<i32>,
    /// The pinned GDScript-facing `sector_size` property.
    #[var(get = get_sector_size, set = set_sector_size)]
    sector_size: PhantomVar<i32>,
}

#[godot_api]
impl IResource for VoxelStreamRegionFiles {
    fn init(base: Base<Resource>) -> Self {
        Self {
            base,
            directory: "user://voxel_data".to_godot(),
            block_size_po2_value: 4,
            region_size_po2_value: 4,
            sector_size_value: 512,
            block_size_po2: PhantomVar::default(),
            region_size_po2: PhantomVar::default(),
            sector_size: PhantomVar::default(),
        }
    }
}

#[godot_api]
impl VoxelStreamRegionFiles {
    /// Build a voxel-core `Arc<dyn VoxelStream>` from this resource.
    /// Creates region files lazily in the configured directory.
    pub(crate) fn core_stream(&self) -> Arc<dyn VoxelStream> {
        let globalized = ProjectSettings::singleton().globalize_path(&self.directory);
        Arc::new(RegionFilesStream::new(PathBuf::from(
            globalized.to_string(),
        )))
    }

    // -----------------------------------------------------------------
    // Pinned VoxelStreamRegionFiles methods
    // (upstream 5828cbeb: VoxelStreamRegionFiles.xml).
    // -----------------------------------------------------------------

    /// Converts existing region files to a new settings profile
    /// (canonical `convert_files`). `new_settings` carries the target
    /// parameters. Faithful stub: the Rust binding does not yet rewrite on-disk
    /// region files, so the call is a bounded no-op.
    #[func]
    fn convert_files(&self, _new_settings: VarDictionary) {
        // TODO(port): implement region-file conversion when the disk format
        // is fully wired.
    }

    /// Size of a region in blocks, as a `Vector3` (canonical `get_region_size`).
    /// Derived from `region_size_po2` and `block_size_po2`:
    /// `(1 << po2)` blocks per axis, scaled by the block size. Matches the
    /// upstream formula `region_size_in_blocks = (1 << region_size_po2) *
    /// (1 << block_size_po2)`.
    #[func]
    fn get_region_size(&self) -> Vector3 {
        let blocks_per_axis = (1i32 << self.region_size_po2_value.max(0))
            * (1i32 << self.block_size_po2_value.max(0));
        let s = blocks_per_axis.max(0) as f32;
        Vector3::new(s, s, s)
    }

    // -----------------------------------------------------------------
    // Pinned VoxelStreamRegionFiles properties
    // (upstream 5828cbeb: VoxelStreamRegionFiles.xml).
    // -----------------------------------------------------------------

    /// Power-of-two exponent of the block size (upstream default 4 ⇒ 16³).
    #[func]
    fn get_block_size_po2(&self) -> i32 {
        self.block_size_po2_value
    }

    #[func]
    fn set_block_size_po2(&mut self, po2: i32) {
        self.block_size_po2_value = po2;
    }

    /// Power-of-two exponent of the region size (upstream default 4).
    #[func]
    fn get_region_size_po2(&self) -> i32 {
        self.region_size_po2_value
    }

    #[func]
    fn set_region_size_po2(&mut self, po2: i32) {
        self.region_size_po2_value = po2;
    }

    /// Sector size in bytes used by region files (upstream default 512).
    #[func]
    fn get_sector_size(&self) -> i32 {
        self.sector_size_value
    }

    #[func]
    fn set_sector_size(&mut self, size: i32) {
        self.sector_size_value = size;
    }
}
