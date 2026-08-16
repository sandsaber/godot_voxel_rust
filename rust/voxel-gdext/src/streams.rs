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
        let region_blocks = 1i32 << self.region_size_po2_value.clamp(0, 8);
        let sector_size = u32::try_from(self.sector_size_value.max(1)).unwrap_or(512);
        Arc::new(RegionFilesStream::with_settings(
            PathBuf::from(globalized.to_string()),
            region_blocks,
            sector_size,
        ))
    }

    // -----------------------------------------------------------------
    // Pinned VoxelStreamRegionFiles methods
    // (upstream 5828cbeb: VoxelStreamRegionFiles.xml).
    // -----------------------------------------------------------------

    /// Rewrite on-disk `.vxr` files to a new region/sector size. Keys:
    /// `region_size_po2`, `sector_size`. Writes into a sibling temp folder,
    /// then replaces `lod*` trees on success.
    #[func]
    fn convert_files(&mut self, new_settings: VarDictionary) {
        let region_po2 = new_settings
            .get("region_size_po2")
            .and_then(|value| value.try_to::<i32>().ok())
            .unwrap_or(self.region_size_po2_value)
            .clamp(0, 8);
        let sector_size = new_settings
            .get("sector_size")
            .and_then(|value| value.try_to::<i32>().ok())
            .unwrap_or(self.sector_size_value)
            .max(1);
        let globalized = ProjectSettings::singleton().globalize_path(&self.directory);
        let source = PathBuf::from(globalized.to_string());
        if !source.is_dir() {
            godot_error!(
                "VoxelStreamRegionFiles.convert_files: directory {} does not exist",
                source.display()
            );
            return;
        }
        let dest = source.with_file_name(format!(
            "{}.convert-{}",
            source
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("voxel_data"),
            std::process::id()
        ));
        let region_blocks = 1i32 << region_po2;
        match RegionFilesStream::convert_directory(
            source.clone(),
            dest.clone(),
            region_blocks,
            u32::try_from(sector_size).unwrap_or(512),
        ) {
            Ok(copied) => {
                if let Err(error) = replace_region_directory(&source, &dest) {
                    godot_error!("VoxelStreamRegionFiles.convert_files: {error}");
                    let _ = std::fs::remove_dir_all(&dest);
                    return;
                }
                self.region_size_po2_value = region_po2;
                self.sector_size_value = sector_size;
                godot_print!("VoxelStreamRegionFiles.convert_files: rewrote {copied} blocks");
            }
            Err(error) => {
                godot_error!("VoxelStreamRegionFiles.convert_files: {error}");
                let _ = std::fs::remove_dir_all(&dest);
            }
        }
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
        self.block_size_po2_value = po2.clamp(0, 8);
    }

    /// Power-of-two exponent of the region size (upstream default 4).
    #[func]
    fn get_region_size_po2(&self) -> i32 {
        self.region_size_po2_value
    }

    #[func]
    fn set_region_size_po2(&mut self, po2: i32) {
        self.region_size_po2_value = po2.clamp(0, 8);
    }

    /// Sector size in bytes used by region files (upstream default 512).
    #[func]
    fn get_sector_size(&self) -> i32 {
        self.sector_size_value
    }

    #[func]
    fn set_sector_size(&mut self, size: i32) {
        self.sector_size_value = size.max(1);
    }
}

fn replace_region_directory(
    source: &std::path::Path,
    converted: &std::path::Path,
) -> Result<(), String> {
    let entries =
        std::fs::read_dir(source).map_err(|e| format!("read {}: {e}", source.display()))?;
    for entry in entries {
        let entry = entry.map_err(|e| format!("read entry: {e}"))?;
        let path = entry.path();
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        let is_region =
            name.starts_with("lod") || (name.starts_with("r.") && name.ends_with(".vxr"));
        if is_region {
            if path.is_dir() {
                std::fs::remove_dir_all(&path)
                    .map_err(|e| format!("remove {}: {e}", path.display()))?;
            } else {
                std::fs::remove_file(&path)
                    .map_err(|e| format!("remove {}: {e}", path.display()))?;
            }
        }
    }
    let converted_entries =
        std::fs::read_dir(converted).map_err(|e| format!("read {}: {e}", converted.display()))?;
    for entry in converted_entries {
        let entry = entry.map_err(|e| format!("read converted entry: {e}"))?;
        let from = entry.path();
        let to = source.join(entry.file_name());
        std::fs::rename(&from, &to)
            .map_err(|e| format!("move {} -> {}: {e}", from.display(), to.display()))?;
    }
    let _ = std::fs::remove_dir_all(converted);
    Ok(())
}
