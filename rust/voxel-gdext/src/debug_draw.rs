//! Wireframe AABB overlay for terrain debug-draw flags.
//!
//! Clipbox streaming has no legacy octree. "Octree nodes" are drawn as the
//! resident mesh-block leaves. Viewer clipboxes and volume bounds come from
//! [`voxel_core::terrain::TerrainDebugSnapshot`].

use godot::classes::mesh::PrimitiveType;
use godot::classes::{ArrayMesh, MeshInstance3D, StandardMaterial3D};
use godot::prelude::*;
use voxel_core::math::Box3i;
use voxel_core::terrain::TerrainDebugSnapshot;

const DEBUG_NODE_NAME: &str = "__voxel_debug_draw";
const MAX_DEBUG_BOXES: usize = 4_096;

/// World-space AABB in voxel units.
#[derive(Debug, Clone, Copy)]
pub(crate) struct DebugBox {
    pub origin: [f32; 3],
    pub size: [f32; 3],
    pub color: [f32; 4],
}

/// 12 edges × 2 endpoints.
pub(crate) fn aabb_line_vertices(origin: [f32; 3], size: [f32; 3]) -> [[f32; 3]; 24] {
    let x0 = origin[0];
    let y0 = origin[1];
    let z0 = origin[2];
    let x1 = origin[0] + size[0];
    let y1 = origin[1] + size[1];
    let z1 = origin[2] + size[2];
    let c = [
        [x0, y0, z0],
        [x1, y0, z0],
        [x1, y1, z0],
        [x0, y1, z0],
        [x0, y0, z1],
        [x1, y0, z1],
        [x1, y1, z1],
        [x0, y1, z1],
    ];
    let edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ];
    let mut out = [[0.0; 3]; 24];
    for (i, (a, b)) in edges.iter().enumerate() {
        out[i * 2] = c[*a];
        out[i * 2 + 1] = c[*b];
    }
    out
}

pub(crate) fn lod_wire_color(lod: u8, lod_count: u8) -> [f32; 4] {
    let denom = f32::from(lod_count.max(1));
    let g = (1.0 - f32::from(lod) / denom).max(0.0);
    let g = g * g;
    [1.0, g, 0.0, 1.0]
}

pub(crate) fn clipbox_wire_color(lod: u8, lod_count: u8) -> [f32; 4] {
    let denom = f32::from(lod_count.max(1));
    let g = (1.0 - f32::from(lod) / denom).max(0.0);
    let g = g * g;
    [g, 32.0 / 255.0, 1.0, 1.0]
}

fn box3i_to_debug(box3: Box3i, color: [f32; 4]) -> Option<DebugBox> {
    if box3.size.x <= 0 || box3.size.y <= 0 || box3.size.z <= 0 {
        return None;
    }
    Some(DebugBox {
        origin: [
            box3.position.x as f32,
            box3.position.y as f32,
            box3.position.z as f32,
        ],
        size: [box3.size.x as f32, box3.size.y as f32, box3.size.z as f32],
        color,
    })
}

fn block_box(position: voxel_core::math::Vector3i, lod: u8, mesh_block_size: i32) -> Box3i {
    let size = mesh_block_size
        .checked_shl(u32::from(lod))
        .unwrap_or(i32::MAX)
        .max(1);
    Box3i::new(
        voxel_core::math::Vector3i::new(
            position.x.saturating_mul(size),
            position.y.saturating_mul(size),
            position.z.saturating_mul(size),
        ),
        voxel_core::math::Vector3i::splat(size),
    )
}

/// Collect the boxes that the current flag set wants to show.
pub(crate) fn collect_debug_boxes(
    snapshot: &TerrainDebugSnapshot,
    flags: DebugDrawFlags,
) -> Vec<DebugBox> {
    let mut boxes = Vec::new();
    let push = |boxes: &mut Vec<DebugBox>, debug_box: Option<DebugBox>| {
        if boxes.len() >= MAX_DEBUG_BOXES {
            return;
        }
        if let Some(debug_box) = debug_box {
            boxes.push(debug_box);
        }
    };

    if flags.volume_bounds || flags.octree_bounds {
        push(
            &mut boxes,
            box3i_to_debug(snapshot.volume_bounds, [1.0, 1.0, 1.0, 1.0]),
        );
    }
    if flags.viewer_clipboxes {
        for &(lod, mesh_box) in &snapshot.viewer_mesh_boxes {
            let size = snapshot
                .mesh_block_size
                .checked_shl(u32::from(lod))
                .unwrap_or(i32::MAX)
                .max(1);
            let world = Box3i::new(
                voxel_core::math::Vector3i::new(
                    mesh_box.position.x.saturating_mul(size),
                    mesh_box.position.y.saturating_mul(size),
                    mesh_box.position.z.saturating_mul(size),
                ),
                voxel_core::math::Vector3i::new(
                    mesh_box.size.x.saturating_mul(size),
                    mesh_box.size.y.saturating_mul(size),
                    mesh_box.size.z.saturating_mul(size),
                ),
            );
            push(
                &mut boxes,
                box3i_to_debug(world, clipbox_wire_color(lod, snapshot.lod_count)),
            );
        }
    }
    let draw_mesh_leaves = flags.octree_nodes || flags.active_mesh_blocks;
    if draw_mesh_leaves || flags.loaded_visual_collision || flags.active_visual_collision {
        for block in &snapshot.mesh_blocks {
            if flags.active_visual_collision && !block.visual_active && !block.collision_active {
                continue;
            }
            if flags.active_mesh_blocks
                && !block.visual_active
                && !flags.octree_nodes
                && !flags.loaded_visual_collision
            {
                continue;
            }
            let color = if flags.loaded_visual_collision || flags.active_visual_collision {
                if block.visual_active && block.collision_active {
                    [0.0, 1.0, 0.0, 1.0]
                } else if block.visual_active {
                    [0.0, 0.6, 1.0, 1.0]
                } else if block.collision_active {
                    [1.0, 0.0, 0.0, 1.0]
                } else {
                    [0.1, 0.1, 0.1, 1.0]
                }
            } else {
                lod_wire_color(block.lod, snapshot.lod_count)
            };
            push(
                &mut boxes,
                box3i_to_debug(
                    block_box(block.position, block.lod, snapshot.mesh_block_size),
                    color,
                ),
            );
        }
    }
    if flags.edited_blocks {
        for block in &snapshot.edited_blocks {
            let color = if block.modified {
                [1.0, 1.0, 0.0, 1.0]
            } else {
                [0.0, 1.0, 0.0, 1.0]
            };
            push(
                &mut boxes,
                box3i_to_debug(
                    block_box(block.position, block.lod, snapshot.mesh_block_size),
                    color,
                ),
            );
        }
        for pos in &snapshot.metadata_voxels {
            push(
                &mut boxes,
                box3i_to_debug(
                    Box3i::new(*pos, voxel_core::math::Vector3i::splat(1)),
                    [1.0, 1.0, 0.0, 1.0],
                ),
            );
        }
    }
    boxes
}

fn build_line_mesh(boxes: &[DebugBox]) -> Option<Gd<ArrayMesh>> {
    if boxes.is_empty() {
        return None;
    }
    let mut positions = Vec::new();
    let mut colors = Vec::new();
    for debug_box in boxes {
        let color = Color::from_rgba(
            debug_box.color[0],
            debug_box.color[1],
            debug_box.color[2],
            debug_box.color[3],
        );
        for vertex in aabb_line_vertices(debug_box.origin, debug_box.size) {
            positions.push(Vector3::new(vertex[0], vertex[1], vertex[2]));
            colors.push(color);
        }
    }
    let mut arrays = Array::new();
    arrays.push(&PackedVector3Array::from(positions.as_slice()).to_variant());
    arrays.push(&Variant::nil()); // NORMAL
    arrays.push(&Variant::nil()); // TANGENT
    arrays.push(&PackedColorArray::from(colors.as_slice()).to_variant());
    for _ in 4..13 {
        arrays.push(&Variant::nil());
    }
    let mut mesh = ArrayMesh::new_gd();
    mesh.add_surface_from_arrays(PrimitiveType::LINES, &arrays);
    if mesh.get_surface_count() == 0 {
        return None;
    }
    let mut material = StandardMaterial3D::new_gd();
    material.set_shading_mode(godot::classes::base_material_3d::ShadingMode::UNSHADED);
    material.set_flag(
        godot::classes::base_material_3d::Flags::ALBEDO_FROM_VERTEX_COLOR,
        true,
    );
    material.set_flag(godot::classes::base_material_3d::Flags::DISABLE_FOG, true);
    // Overlay edges sit exactly on block-boundary planes where terrain
    // triangles are coplanar — classic z-fighting. Debug wireframes draw
    // through geometry instead.
    material.set_flag(
        godot::classes::base_material_3d::Flags::DISABLE_DEPTH_TEST,
        true,
    );
    material.set_render_priority(1);
    mesh.surface_set_material(0, &material);
    Some(mesh)
}

/// Rebuild or hide the internal overlay child of `host`.
pub(crate) fn refresh_debug_overlay(
    host: &mut Gd<Node3D>,
    snapshot: Option<&TerrainDebugSnapshot>,
    enabled: bool,
    flags: DebugDrawFlags,
) {
    let boxes = match (enabled, snapshot) {
        (true, Some(snapshot)) => collect_debug_boxes(snapshot, flags),
        _ => Vec::new(),
    };
    let existing = host
        .get_node_or_null(DEBUG_NODE_NAME)
        .and_then(|node| node.try_cast::<MeshInstance3D>().ok());
    // Detach before queue_free: a same-frame disable→enable must not find
    // and reuse the dying node through `get_node_or_null`.
    let retire = |host: &mut Gd<Node3D>, node: &mut Gd<MeshInstance3D>| {
        host.remove_child(&node.clone().upcast::<godot::prelude::Node>());
        node.queue_free();
    };
    if boxes.is_empty() {
        if let Some(mut node) = existing {
            retire(host, &mut node);
        }
        return;
    }
    let Some(mesh) = build_line_mesh(&boxes) else {
        if let Some(mut node) = existing {
            retire(host, &mut node);
        }
        return;
    };
    if let Some(mut node) = existing {
        node.set_mesh(&mesh);
        node.set_cast_shadows_setting(
            godot::classes::geometry_instance_3d::ShadowCastingSetting::OFF,
        );
        return;
    }
    let mut node = MeshInstance3D::new_alloc();
    node.set_name(DEBUG_NODE_NAME);
    node.set_mesh(&mesh);
    node.set_cast_shadows_setting(godot::classes::geometry_instance_3d::ShadowCastingSetting::OFF);
    host.add_child(&node);
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct DebugDrawFlags {
    pub octree_nodes: bool,
    pub octree_bounds: bool,
    pub volume_bounds: bool,
    pub active_mesh_blocks: bool,
    pub viewer_clipboxes: bool,
    pub loaded_visual_collision: bool,
    pub active_visual_collision: bool,
    pub edited_blocks: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use voxel_core::math::Vector3i;
    use voxel_core::terrain::{DebugMeshBlock, TerrainDebugSnapshot};

    #[test]
    fn aabb_line_mesh_has_twelve_edges() {
        let verts = aabb_line_vertices([0.0, 0.0, 0.0], [16.0, 16.0, 16.0]);
        assert_eq!(verts.len(), 24);
        assert_eq!(verts[0], [0.0, 0.0, 0.0]);
        assert_eq!(verts[1], [16.0, 0.0, 0.0]);
    }

    #[test]
    fn collect_boxes_honors_flags() {
        let snapshot = TerrainDebugSnapshot {
            volume_bounds: Box3i::new(Vector3i::zero(), Vector3i::splat(64)),
            mesh_block_size: 16,
            lod_count: 2,
            data_block_count: 1,
            mesh_blocks: vec![DebugMeshBlock {
                position: Vector3i::zero(),
                lod: 0,
                visual_active: true,
                collision_active: false,
                is_loaded: true,
            }],
            viewer_mesh_boxes: vec![(0, Box3i::new(Vector3i::zero(), Vector3i::splat(2)))],
            edited_blocks: Vec::new(),
            metadata_voxels: Vec::new(),
        };
        assert!(collect_debug_boxes(&snapshot, DebugDrawFlags::default()).is_empty());
        assert_eq!(
            collect_debug_boxes(
                &snapshot,
                DebugDrawFlags {
                    volume_bounds: true,
                    ..DebugDrawFlags::default()
                }
            )
            .len(),
            1
        );
        assert_eq!(
            collect_debug_boxes(
                &snapshot,
                DebugDrawFlags {
                    octree_nodes: true,
                    ..DebugDrawFlags::default()
                }
            )
            .len(),
            1
        );
        assert_eq!(
            collect_debug_boxes(
                &snapshot,
                DebugDrawFlags {
                    viewer_clipboxes: true,
                    ..DebugDrawFlags::default()
                }
            )
            .len(),
            1
        );
    }
}
