//! Generates starter corpora for the `cargo fuzz` targets in `rust/fuzz/`.
//!
//! The fuzz crate git-ignores its working corpus (`fuzz/corpus/`), so valid
//! seed inputs live under `rust/fuzz/seed_corpus/` and are produced here:
//!
//! ```sh
//! cd rust
//! cargo run -p voxel-core --example gen_fuzz_seeds
//! cargo +nightly fuzz run vox_parser fuzz/seed_corpus/vox_parser
//! ```
//!
//! Seeds are small VALID inputs — one per parser path — which libFuzzer
//! mutates from. They shrink the time the fuzzer spends rediscovering the
//! basic format structure.

use std::fs;
use std::path::PathBuf;

use voxel_core::math::Vector3i;
use voxel_core::storage::voxel_buffer::ChannelId;
use voxel_core::storage::{VoxelBuffer, VoxelFormat};
use voxel_core::streams::block_serializer;
use voxel_core::streams::compressed_data::Compression;

fn seed_dir(target: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../fuzz/seed_corpus")
        .join(target);
    fs::create_dir_all(&dir).expect("create seed_corpus dir");
    dir
}

fn write_seed(target: &str, name: &str, bytes: &[u8]) {
    let path = seed_dir(target).join(name);
    fs::write(&path, bytes).expect("write seed");
    println!("wrote {} ({} bytes)", path.display(), bytes.len());
}

fn main() {
    // ------------------------------------------------------------------
    // vox_parser: a minimal valid MagicaVoxel file
    // (magic + version + MAIN{SIZE, XYZI}).
    // ------------------------------------------------------------------
    let mut vox: Vec<u8> = Vec::new();
    vox.extend_from_slice(b"VOX ");
    vox.extend_from_slice(&150u32.to_le_bytes());
    // MAIN chunk: no content, children = SIZE + XYZI.
    vox.extend_from_slice(b"MAIN");
    vox.extend_from_slice(&0i32.to_le_bytes());
    vox.extend_from_slice(&48i32.to_le_bytes());
    // SIZE chunk: 4 x 4 x 4.
    vox.extend_from_slice(b"SIZE");
    vox.extend_from_slice(&12i32.to_le_bytes());
    vox.extend_from_slice(&0i32.to_le_bytes());
    vox.extend_from_slice(&4u32.to_le_bytes());
    vox.extend_from_slice(&4u32.to_le_bytes());
    vox.extend_from_slice(&4u32.to_le_bytes());
    // XYZI chunk: two voxels (x, y, z, palette index).
    vox.extend_from_slice(b"XYZI");
    vox.extend_from_slice(&12i32.to_le_bytes());
    vox.extend_from_slice(&0i32.to_le_bytes());
    vox.extend_from_slice(&2u32.to_le_bytes());
    vox.extend_from_slice(&[0, 0, 0, 1]);
    vox.extend_from_slice(&[1, 0, 0, 5]);
    // Self-check: the seed must actually parse.
    voxel_core::format::vox::parse(&vox).expect("vox seed must parse");
    write_seed("vox_parser", "minimal_valid.vox", &vox);

    // ------------------------------------------------------------------
    // block_serializer / region_file: a valid LZ4-compressed block payload
    // (the on-disk envelope both targets feed into decompress_and_deserialize).
    // ------------------------------------------------------------------
    let mut buffer = VoxelBuffer::with_size(Vector3i::splat(4));
    VoxelFormat::new().configure_buffer(&mut buffer);
    buffer.set_voxel(7, 0, 0, 0, ChannelId::Type.index());
    buffer.set_voxel(3, 1, 2, 0, ChannelId::Type.index());
    buffer.set_voxel_f(-0.25, 2, 2, 2, ChannelId::Sdf.index());

    let mut payload = Vec::new();
    block_serializer::serialize_and_compress(&buffer, &mut payload, Compression::Lz4)
        .expect("serialize_and_compress seed block");
    // Self-check: the seeds must round-trip through the fuzzed entry point.
    // (`decompress_and_deserialize` requires a compression envelope, so the
    // seeds always go through `serialize_and_compress`.)
    let mut check = VoxelBuffer::with_size(Vector3i::zero());
    block_serializer::decompress_and_deserialize(&payload, &mut check)
        .expect("lz4 seed must deserialize");
    write_seed("block_serializer", "valid_4x4x4_lz4.bin", &payload);
    write_seed("region_file", "valid_4x4x4_lz4.bin", &payload);

    // ------------------------------------------------------------------
    // Metadata section (R7): a valid payload carrying every MetadataValue
    // kind, plus two hostile-section anchors (foreign C++ custom tag 32 and a
    // truncated entry) for the skip-not-fail contract. Hostile sections are
    // spliced in front of the trailing magic with the same u32 envelope the
    // serializer uses.
    // ------------------------------------------------------------------
    use voxel_core::storage::MetadataValue;
    let mut meta_buffer = VoxelBuffer::with_size(Vector3i::splat(2));
    VoxelFormat::new().configure_buffer(&mut meta_buffer);
    meta_buffer.set_voxel(5, 1, 0, 0, ChannelId::Type.index());
    meta_buffer.set_block_metadata(MetadataValue::Text("seed".into()));
    meta_buffer.set_voxel_metadata(Vector3i::new(0, 1, 0), MetadataValue::Int(64));
    meta_buffer.set_voxel_metadata(Vector3i::new(1, 1, 1), MetadataValue::Float(0.25));
    meta_buffer.set_voxel_metadata(Vector3i::new(1, 0, 0), MetadataValue::Bytes(vec![1, 2, 3]));

    let mut meta_payload = Vec::new();
    block_serializer::serialize_and_compress(&meta_buffer, &mut meta_payload, Compression::Lz4)
        .expect("serialize_and_compress metadata seed");
    let mut meta_check = VoxelBuffer::with_size(Vector3i::zero());
    block_serializer::decompress_and_deserialize(&meta_payload, &mut meta_check)
        .expect("metadata seed must deserialize");
    assert_eq!(
        meta_check.voxel_metadata(Vector3i::new(0, 1, 0)),
        Some(&MetadataValue::Int(64))
    );
    write_seed(
        "block_serializer",
        "metadata-section-lz4.bin",
        &meta_payload,
    );
    write_seed("region_file", "metadata-section-lz4.bin", &meta_payload);

    let hostile = |section: &[u8]| -> Vec<u8> {
        // Start from the metadata-free block so the spliced section is the
        // only one, serialize with a verbatim (None) compression envelope,
        // then insert the section in front of the trailing magic.
        let mut raw = Vec::new();
        block_serializer::serialize_and_compress(&buffer, &mut raw, Compression::None)
            .expect("serialize seed base");
        let magic = raw.split_off(raw.len() - 4);
        raw.extend_from_slice(&(section.len() as u32).to_le_bytes());
        raw.extend_from_slice(section);
        raw.extend_from_slice(&magic);
        raw
    };

    // Foreign C++ custom tag (32 = Godot Variant): must decode as lossy, not
    // panic and not fail the voxel load.
    let foreign = hostile(&[32, 0xde, 0xad]);
    let mut foreign_check = VoxelBuffer::with_size(Vector3i::zero());
    block_serializer::decompress_and_deserialize(&foreign, &mut foreign_check)
        .expect("foreign-tag seed must load voxels");
    write_seed("block_serializer", "metadata-foreign-tag.bin", &foreign);
    write_seed("region_file", "metadata-foreign-tag.bin", &foreign);

    // Truncated u64 entry: skip-not-fail again.
    let truncated = hostile(&[1u8, 0x00, 0x01]);
    let mut truncated_check = VoxelBuffer::with_size(Vector3i::zero());
    block_serializer::decompress_and_deserialize(&truncated, &mut truncated_check)
        .expect("truncated-metadata seed must load voxels");
    write_seed("block_serializer", "metadata-truncated.bin", &truncated);
    write_seed("region_file", "metadata-truncated.bin", &truncated);
}
