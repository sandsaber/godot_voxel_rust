use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use voxel_core::math::Vector3i;
use voxel_core::storage::{MetadataValue, VoxelBuffer};
use voxel_core::streams::compressed_data::Compression;
use voxel_core::streams::region::{RegionFile, RegionFormat};
use voxel_core::streams::{RegionFilesStream, VoxelSaveQuery, VoxelStream};

static NEXT_DIR: AtomicU64 = AtomicU64::new(0);

struct TestDir(PathBuf);

impl TestDir {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "voxel-region-write-safety-{}-{}",
            std::process::id(),
            NEXT_DIR.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn block(value: u64) -> VoxelBuffer {
    let mut block = VoxelBuffer::with_size(Vector3i::splat(2));
    block.set_voxel(value, 0, 0, 0, 0);
    block
}

fn small_format(sector_size: u32) -> RegionFormat {
    let mut format = RegionFormat {
        block_size_po2: 1,
        region_size: Vector3i::splat(2),
        sector_size,
        ..RegionFormat::default()
    };
    let sample = block(0);
    for (index, depth) in format.channel_depths.iter_mut().enumerate() {
        *depth = sample.channel_depth(index);
    }
    format
}

#[test]
fn rejected_replacement_preserves_existing_and_following_blocks_on_disk() {
    let dir = TestDir::new();
    let path = dir.path().join("region.vxr");
    let format = small_format(64);
    let a = Vector3i::new(0, 0, 0);
    let b = Vector3i::new(1, 0, 0);

    {
        let mut region = RegionFile::open_with_format(&path, true, format).unwrap();
        region.save_block(a, &block(11), Compression::None).unwrap();
        region.save_block(b, &block(22), Compression::None).unwrap();
        region.flush().unwrap();
        let before = std::fs::read(&path).unwrap();

        let mut oversized = block(99);
        let mut bytes = vec![0u8; 200_000];
        let mut state = 0x1234_5678u32;
        for byte in &mut bytes {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            *byte = state as u8;
        }
        oversized.set_block_metadata(MetadataValue::Bytes(bytes));
        assert!(region.save_block(a, &oversized, Compression::None).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), before);
    }

    let mut reopened = RegionFile::open(&path, false).unwrap();
    let mut loaded_a = block(0);
    let mut loaded_b = block(0);
    reopened.load_block(a, &mut loaded_a).unwrap();
    reopened.load_block(b, &mut loaded_b).unwrap();
    assert_eq!(loaded_a.get_voxel(0, 0, 0, 0), 11);
    assert_eq!(loaded_b.get_voxel(0, 0, 0, 0), 22);
}

#[test]
fn invalid_sector_settings_fail_before_forest_metadata_is_published() {
    for sector_size in [0, u16::MAX as u32 + 1] {
        let dir = TestDir::new();
        let region_path = dir.path().join("invalid.vxr");
        assert!(
            RegionFile::open_with_format(&region_path, true, small_format(sector_size)).is_err()
        );
        assert!(!region_path.exists());
        let stream = RegionFilesStream::with_settings(dir.path().to_path_buf(), 16, sector_size);
        let buffer = VoxelBuffer::with_size(Vector3i::splat(16));
        assert!(stream
            .save_voxel_block(VoxelSaveQuery::new(&buffer, Vector3i::zero(), 0))
            .is_err());
        assert!(!dir.path().join("meta.vxrm").exists());
        assert!(!dir.path().join("lod0/r.0.0.0.vxr").exists());
    }
}

#[test]
fn largest_serializable_sector_size_round_trips() {
    let dir = TestDir::new();
    let path = dir.path().join("region.vxr");
    let format = small_format(u16::MAX as u32);
    let pos = Vector3i::zero();
    {
        let mut region = RegionFile::open_with_format(&path, true, format).unwrap();
        region
            .save_block(pos, &block(77), Compression::None)
            .unwrap();
        region.flush().unwrap();
    }
    let mut reopened = RegionFile::open(&path, false).unwrap();
    assert_eq!(reopened.format().sector_size, u16::MAX as u32);
    let mut loaded = block(0);
    reopened.load_block(pos, &mut loaded).unwrap();
    assert_eq!(loaded.get_voxel(0, 0, 0, 0), 77);
}

#[test]
fn conversion_rejects_invalid_sector_size_without_creating_destination() {
    let source = TestDir::new();
    let destination = TestDir::new();
    let target = destination.path().join("converted");
    assert!(RegionFilesStream::convert_directory(
        source.path().to_path_buf(),
        target.clone(),
        16,
        u16::MAX as u32 + 1,
    )
    .is_err());
    assert!(!target.exists());
}
