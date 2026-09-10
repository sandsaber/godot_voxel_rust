#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use voxel_core::engine::MeshingDependency;
    use voxel_core::math::{Box3i, Vector3f, Vector3i};
    use voxel_core::meshers::{MesherInput, MesherOutput, VoxelMesher};
    use voxel_core::storage::{ChannelDepth, ChannelId, VoxelBuffer, VoxelData};
    use voxel_core::streams::{
        LoadResult, RegionFilesStream, VoxelLoadQuery, VoxelSaveQuery, VoxelStream,
    };
    use voxel_core::terrain::VoxelTerrainCore;

    struct EmptyMesher;
    impl VoxelMesher for EmptyMesher {
        fn build(&self, _: &mut MesherOutput, _: &MesherInput<'_>) {}
    }

    struct TestDir(std::path::PathBuf);
    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    fn dir(name: &str) -> TestDir {
        let p = std::env::temp_dir().join(format!("voxel-review-{}-{name}", std::process::id()));
        std::fs::create_dir_all(&p).unwrap();
        TestDir(p)
    }
    fn block(value: f32) -> VoxelBuffer {
        let mut b = VoxelBuffer::with_size(Vector3i::splat(16));
        b.set_channel_depth(ChannelId::Sdf.index(), ChannelDepth::Bit32);
        b.fill(u64::from(value.to_bits()), ChannelId::Sdf.index());
        b
    }
    fn save(s: &RegionFilesStream, x: i32, value: f32) {
        s.save_voxel_block(VoxelSaveQuery::new(
            &block(value),
            Vector3i::new(x, 0, 0),
            0,
        ))
        .unwrap();
        s.flush().unwrap();
    }
    fn load(s: &RegionFilesStream, x: i32) -> LoadResult {
        s.load_voxel_block(VoxelLoadQuery::new(
            &mut block(0.0),
            Vector3i::new(x, 0, 0),
            0,
        ))
        .unwrap()
    }

    #[test]
    fn conversion_preserves_legacy_regions_when_lod0_also_exists() {
        let source = dir("mixed-source");
        let dest = dir("mixed-dest");
        {
            let s = RegionFilesStream::with_settings(source.0.clone(), 16, 512);
            save(&s, 0, -1.0);
            save(&s, 16, 1.0);
        }
        std::fs::rename(
            source.0.join("lod0/r.0.0.0.vxr"),
            source.0.join("r.0.0.0.vxr"),
        )
        .unwrap();
        {
            let s = RegionFilesStream::with_settings(source.0.clone(), 16, 512);
            assert_eq!(load(&s, 0), LoadResult::Found);
            assert_eq!(load(&s, 16), LoadResult::Found);
        }
        let copied =
            RegionFilesStream::convert_directory(source.0.clone(), dest.0.clone(), 16, 1024)
                .unwrap();
        let s = RegionFilesStream::with_settings(dest.0.clone(), 16, 1024);
        assert_eq!(copied, 2, "conversion must preserve both readable regions");
        assert_eq!(load(&s, 0), LoadResult::Found);
    }

    #[test]
    fn independent_streams_cannot_rewrite_locked_forest_layout() {
        let p = dir("shared-meta");
        let a = RegionFilesStream::with_settings(p.0.clone(), 32, 512);
        let b = RegionFilesStream::with_settings(p.0.clone(), 16, 512);
        assert_eq!(load(&a, 0), LoadResult::NotFound);
        assert_eq!(load(&b, 0), LoadResult::NotFound);
        save(&a, 32, -1.0);
        save(&b, 0, 1.0);
        drop(a);
        drop(b);
        let reopened = RegionFilesStream::new(p.0.clone());
        assert_eq!(
            load(&reopened, 32),
            LoadResult::Found,
            "a previously saved block must survive the second stream's first save"
        );
    }

    #[test]
    fn terrain_smoothing_reads_neighbors_across_data_blocks() {
        let mut data = VoxelData::new();
        data.set_bounds(Box3i::new(Vector3i::splat(-64), Vector3i::splat(128)));
        data.set_streaming_enabled(false);
        data.set_full_load_completed(true);
        let mut core = VoxelTerrainCore::new_generator_only(
            data,
            MeshingDependency::new(Arc::new(EmptyMesher), None),
        );
        assert!(core.try_install_remote_block(Vector3i::zero(), 0, block(-1.0)));
        assert!(core.try_install_remote_block(Vector3i::new(1, 0, 0), 0, block(1.0)));
        let count = core
            .try_edit_smooth(
                Vector3f::new(15.5, 8.0, 8.0),
                3.0,
                1,
                ChannelId::Sdf.index(),
            )
            .unwrap();
        assert_eq!(count, 2);
        let left = core
            .data()
            .block_snapshot(Vector3i::zero(), 0)
            .unwrap()
            .into_voxels()
            .unwrap()
            .get_voxel_f(15, 8, 8, ChannelId::Sdf.index());
        let right = core
            .data()
            .block_snapshot(Vector3i::new(1, 0, 0), 0)
            .unwrap()
            .into_voxels()
            .unwrap()
            .get_voxel_f(0, 8, 8, ChannelId::Sdf.index());
        assert!(
            (left + 1.0 / 3.0).abs() < 0.001 && (right - 1.0 / 3.0).abs() < 0.001,
            "expected boundary (-1/3,+1/3), got ({left},{right})"
        );
    }

    #[test]
    fn smooth_union_keeps_existing_solid_far_from_modifier() {
        use voxel_core::modifiers::{sdf_blend, SdfOperation};
        let actual = sdf_blend(-10.0, 10.0, SdfOperation::Add, 1.0);
        let expected = voxel_core::math::sdf::sdf_smooth_union(-10.0, 10.0, 1.0);
        assert_eq!(
            actual, expected,
            "an additive modifier must preserve existing solid outside its blend band"
        );
        for (a, b, op, expected) in [
            (10.0, -10.0, SdfOperation::Add, -10.0),
            (0.0, 0.0, SdfOperation::Add, -0.25),
            (10.0, 10.0, SdfOperation::Subtract, 10.0),
            (-10.0, -10.0, SdfOperation::Subtract, 10.0),
            (0.0, 0.0, SdfOperation::Subtract, 0.25),
        ] {
            assert_eq!(sdf_blend(a, b, op, 1.0), expected);
            let hard = match op {
                SdfOperation::Add => a.min(b),
                SdfOperation::Subtract => a.max(-b),
            };
            assert_eq!(sdf_blend(a, b, op, 0.0), hard);
        }
    }
}
