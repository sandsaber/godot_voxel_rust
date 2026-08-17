extends Node3D
## Behavioral integration gate for terrain rendering, invalid script inputs,
## and RegionFiles persistence. All assertions use public Godot-visible state.

const WAIT_TIMEOUT_MSEC := 20_000
const MESH_STABLE_MSEC := 250

var failures := 0


func _ready() -> void:
	call_deferred("_run")


func _ok(condition: bool, message: String) -> void:
	if condition:
		print("  PASS: ", message)
	else:
		print("  FAIL: ", message)
		failures += 1


func _new_terrain(stream_directory := "") -> Node3D:
	var terrain := ClassDB.instantiate("VoxelTerrain") as Node3D
	if terrain == null:
		return null

	var generator := ClassDB.instantiate("VoxelGeneratorWaves") as Resource
	if generator == null:
		terrain.free()
		return null
	terrain.set_generator(generator)

	if not stream_directory.is_empty():
		var stream := ClassDB.instantiate("VoxelStreamRegionFiles") as Resource
		if stream == null:
			terrain.free()
			return null
		stream.set("directory", stream_directory)
		terrain.set_stream(stream)

	var viewer := ClassDB.instantiate("VoxelViewer") as Node3D
	if viewer == null:
		terrain.free()
		return null
	viewer.set("view_distance", 48)
	terrain.add_child(viewer)
	add_child(terrain)
	return terrain


func _viewer(terrain: Node3D) -> Node3D:
	for child in terrain.get_children():
		if child.is_class("VoxelViewer"):
			return child as Node3D
	return null


func _mesh_children(terrain: Node3D) -> Array[MeshInstance3D]:
	var meshes: Array[MeshInstance3D] = []
	for child in terrain.get_children():
		if child is MeshInstance3D:
			meshes.append(child)
	return meshes


func _first_valid_mesh(terrain: Node3D) -> MeshInstance3D:
	if int(terrain.get_mesh_block_count()) <= 0:
		return null
	for mesh_instance in _mesh_children(terrain):
		if mesh_instance.mesh != null and mesh_instance.mesh.get_surface_count() > 0:
			return mesh_instance
	return null


func _mesh_has_transvoxel_custom0(mesh_instance: MeshInstance3D) -> bool:
	if mesh_instance == null or mesh_instance.mesh == null:
		return false
	for surface_index in mesh_instance.mesh.get_surface_count():
		var arrays: Array = mesh_instance.mesh.surface_get_arrays(surface_index)
		if arrays.size() != Mesh.ARRAY_MAX:
			continue
		var vertices: PackedVector3Array = arrays[Mesh.ARRAY_VERTEX]
		var custom0: PackedFloat32Array = arrays[Mesh.ARRAY_CUSTOM0]
		if not vertices.is_empty() and custom0.size() == vertices.size() * 4:
			var format := int(mesh_instance.mesh.surface_get_format(surface_index))
			var custom0_format := (
				format >> int(Mesh.ARRAY_FORMAT_CUSTOM0_SHIFT)
			) & int(Mesh.ARRAY_FORMAT_CUSTOM_MASK)
			if custom0_format == int(Mesh.ARRAY_CUSTOM_RGBA_FLOAT):
				return true
	return false


func _wait_for_mesh(terrain: Node3D) -> MeshInstance3D:
	var deadline := Time.get_ticks_msec() + WAIT_TIMEOUT_MSEC
	while Time.get_ticks_msec() < deadline:
		await get_tree().process_frame
		var mesh_instance := _first_valid_mesh(terrain)
		if mesh_instance != null:
			return mesh_instance
	return null


func _wait_for_stable_mesh(terrain: Node3D) -> MeshInstance3D:
	var deadline := Time.get_ticks_msec() + WAIT_TIMEOUT_MSEC
	var candidate_id := 0
	var candidate_position := Vector3.ZERO
	var stable_since := 0
	while Time.get_ticks_msec() < deadline:
		await get_tree().process_frame
		var current := _first_valid_mesh(terrain)
		if current == null:
			candidate_id = 0
			stable_since = 0
			continue
		var now := Time.get_ticks_msec()
		if (
			current.get_instance_id() == candidate_id
			and current.position.is_equal_approx(candidate_position)
		):
			if now - stable_since >= MESH_STABLE_MSEC:
				return current
		else:
			candidate_id = current.get_instance_id()
			candidate_position = current.position
			stable_since = now
	return null


func _edit_voxel_for_mesh(mesh_instance: MeshInstance3D) -> Vector3i:
	# Renderer names are part of the normal scene-tree representation, not a
	# debug API: mesh_lod{lod}_{block_x}_{block_y}_{block_z}. Pick an interior
	# voxel so the edit unambiguously invalidates this exact rendered block.
	var parts := String(mesh_instance.name).split("_")
	if parts.size() == 5 and parts[1] == "lod0":
		return Vector3i(
			int(parts[2]) * 16 + 8,
			int(parts[3]) * 16 + 8,
			int(parts[4]) * 16 + 8
		)
	return Vector3i(
		floori(mesh_instance.position.x) + 8,
		floori(mesh_instance.position.y) + 8,
		floori(mesh_instance.position.z) + 8
	)


func _wait_for_remesh(
	terrain: Node3D,
	instance_id: int,
	old_mesh_id: int
) -> bool:
	# Remesh uploads a fresh ArrayMesh onto the SAME MeshInstance3D; the node
	# is intentionally reused (upstream parity). Wait for the mesh resource to
	# change identity while the node stays alive.
	var deadline := Time.get_ticks_msec() + WAIT_TIMEOUT_MSEC
	while Time.get_ticks_msec() < deadline:
		await get_tree().process_frame
		for current in _mesh_children(terrain):
			if (
				current.get_instance_id() == instance_id
				and current.mesh != null
				and current.mesh.get_surface_count() > 0
				and current.mesh.get_instance_id() != old_mesh_id
			):
				return true
	return false


func _wait_for_replacement(
	terrain: Node3D,
	mesh_position: Vector3,
	old_instance_id: int
) -> MeshInstance3D:
	var deadline := Time.get_ticks_msec() + WAIT_TIMEOUT_MSEC
	while Time.get_ticks_msec() < deadline:
		await get_tree().process_frame
		for current in _mesh_children(terrain):
			if (
				current.position.is_equal_approx(mesh_position)
				and current.get_instance_id() != old_instance_id
			):
				return current
	return null


func _wait_for_removal(
	terrain: Node3D,
	mesh_position: Vector3,
	instance_weakref: WeakRef
) -> bool:
	var deadline := Time.get_ticks_msec() + WAIT_TIMEOUT_MSEC
	while Time.get_ticks_msec() < deadline:
		await get_tree().process_frame
		var position_still_rendered := false
		for current in _mesh_children(terrain):
			if current.position.is_equal_approx(mesh_position):
				position_still_rendered = true
				break
		if instance_weakref.get_ref() == null and not position_still_rendered:
			return true
	return false


func _queue_free_and_wait(node: Node) -> bool:
	var node_ref: WeakRef = weakref(node)
	node.queue_free()
	var deadline := Time.get_ticks_msec() + WAIT_TIMEOUT_MSEC
	while node_ref.get_ref() != null and Time.get_ticks_msec() < deadline:
		await get_tree().process_frame
	return node_ref.get_ref() == null


func _remove_directory_tree(path: String, validated_root: String) -> bool:
	if path != validated_root and not path.begins_with(validated_root + "/"):
		return false
	for file_name in DirAccess.get_files_at(path):
		var file_path := path.path_join(file_name)
		if not file_path.begins_with(validated_root + "/"):
			return false
		if DirAccess.remove_absolute(file_path) != OK:
			return false
	for directory_name in DirAccess.get_directories_at(path):
		var child_path := path.path_join(directory_name)
		if not _remove_directory_tree(child_path, validated_root):
			return false
	return DirAccess.remove_absolute(path) == OK


func _remove_generated_directory(directory: String) -> bool:
	var absolute_path := ProjectSettings.globalize_path(directory).simplify_path()
	var user_root := ProjectSettings.globalize_path("user://").simplify_path().trim_suffix("/")
	if (
		absolute_path.get_base_dir() != user_root
		or not absolute_path.get_file().begins_with("runtime_correctness_")
	):
		return false
	if not DirAccess.dir_exists_absolute(absolute_path):
		return true
	return _remove_directory_tree(absolute_path, absolute_path)


func _snapshot_buffer_values(buffer: RefCounted) -> Array[int]:
	var values: Array[int] = []
	for channel in range(8):
		for z in range(4):
			for y in range(4):
				for x in range(4):
					values.append(int(buffer.get_voxel(x, y, z, channel)))
	return values


func _exercise_invalid_calls(terrain: Node3D, edited_voxel: Vector3i) -> void:
	var buffer: RefCounted = ClassDB.instantiate("VoxelBuffer")
	_ok(buffer != null, "VoxelBuffer exists for invalid-input checks")
	if buffer == null:
		return

	buffer.create(4, 4, 4)
	for channel in range(8):
		for z in range(4):
			for y in range(4):
				for x in range(4):
					var cell_index := x + y * 4 + z * 16
					var seeded_value := 1 + ((cell_index + channel * 17) % 127)
					buffer.set_voxel(x, y, z, channel, seeded_value)
	var values_before := _snapshot_buffer_values(buffer)

	buffer.create(-1, 4, 4)
	buffer.create(2147483647, 4, 4)
	buffer.set_voxel(1, 1, 1, -1, 255)
	buffer.set_voxel(1, 1, 1, 8, 255)
	buffer.set_voxel(-1, 0, 0, 0, 255)
	buffer.set_voxel(4, 3, 3, 0, 255)
	buffer.set_voxel(0, -1, 1, 7, 255)
	buffer.set_voxel(3, 4, 2, 7, 255)
	buffer.fill_channel(8, 255)
	buffer.clear_channel(-1, 255)
	var values_after := _snapshot_buffer_values(buffer)
	_ok(
		buffer.get_size_x() == 4
		and buffer.get_size_y() == 4
		and buffer.get_size_z() == 4
		and values_before.size() == 4 * 4 * 4 * 8
		and values_after == values_before,
		"invalid calls preserve dimensions and all 512 seeded voxel values"
	)
	_ok(
		int(buffer.get_voxel(0, 0, 0, -1)) == 0
		and int(buffer.get_voxel(0, 0, 0, 8)) == 0
		and int(buffer.get_voxel(-1, 0, 0, 0)) == 0
		and int(buffer.get_voxel(4, 3, 3, 0)) == 0
		and int(buffer.get_voxel(0, -1, 1, 7)) == 0
		and int(buffer.get_voxel(3, 4, 2, 7)) == 0,
		"invalid channel and position getters return the neutral sentinel"
	)

	var multipass: Resource = ClassDB.instantiate("VoxelGeneratorMultipass")
	_ok(multipass != null, "VoxelGeneratorMultipass exists for invalid-count checks")
	if multipass != null:
		multipass.set_pass_count(2)
		multipass.set_pass_count(-1)
		multipass.set_pass_count(257)
		_ok(
			int(multipass.get_pass_count()) == 2,
			"invalid negative and oversized counts preserve generator state"
		)

	var mesh_count_before := int(terrain.get_mesh_block_count())
	var baseline_value := float(terrain.get_voxel_sdf(
		edited_voxel.x, edited_voxel.y, edited_voxel.z
	))
	var invalid_negative = terrain.raycast(0.0, 100.0, 0.0, 0.0, -1.0, 0.0, -1.0)
	var invalid_infinite = terrain.raycast(
		0.0, 100.0, 0.0, 0.0, -1.0, 0.0, INF
	)
	var invalid_oversized = terrain.raycast(
		0.0, 100.0, 0.0, 0.0, -1.0, 0.0, 65537.0
	)
	var invalid_edit_ok := bool(terrain.set_voxel_sdf(
		edited_voxel.x, edited_voxel.y, edited_voxel.z, NAN
	))
	var value_after := float(terrain.get_voxel_sdf(
		edited_voxel.x, edited_voxel.y, edited_voxel.z
	))
	_ok(
		invalid_negative.is_empty()
		and invalid_infinite.is_empty()
		and invalid_oversized.is_empty()
		and not invalid_edit_ok
		and int(terrain.get_mesh_block_count()) == mesh_count_before
		and is_equal_approx(value_after, baseline_value),
		"invalid raycast and SDF calls return safely without changing terrain state"
	)


func _run_lifecycle_checks() -> void:
	print("=== runtime correctness: mesh lifecycle and invalid inputs ===")
	var terrain := _new_terrain()
	_ok(terrain != null, "terrain, generator, and viewer instantiated")
	if terrain == null:
		return

	var target := await _wait_for_stable_mesh(terrain)
	_ok(
		target != null and int(terrain.get_mesh_block_count()) > 0,
		"at least one nonzero terrain mesh uploads and remains stable before edit"
	)
	_ok(
		_mesh_has_transvoxel_custom0(target),
		"Transvoxel mesh uploads four CUSTOM0 floats per vertex with RGBA_FLOAT format"
	)
	if target == null:
		_ok(
			await _queue_free_and_wait(terrain),
			"terrain fully frees after a mesh-upload timeout"
		)
		return

	var target_id := target.get_instance_id()
	var target_position := target.position
	var old_target_ref: WeakRef = weakref(target)
	var target_mesh_id := (
		target.mesh.get_instance_id() if target.mesh != null else 0
	)
	var edit_voxel := _edit_voxel_for_mesh(target)
	var old_value := float(terrain.get_voxel_sdf(edit_voxel.x, edit_voxel.y, edit_voxel.z))
	var edited_value := -1.0 if old_value >= 0.0 else 1.0
	# NOTE: do_sphere below may rewrite the same voxel; edited_value is
	# re-snapshotted afterwards so later assertions compare against reality.
	var edit_ok := bool(terrain.set_voxel_sdf(
		edit_voxel.x, edit_voxel.y, edit_voxel.z, edited_value
	))
	_ok(
		edit_ok
		and is_equal_approx(
			float(terrain.get_voxel_sdf(edit_voxel.x, edit_voxel.y, edit_voxel.z)),
			edited_value
		),
		"SDF edit succeeds and reads back while the target block is viewed"
	)
	var tool = terrain.get_voxel_tool()
	_ok(tool != null, "get_voxel_tool returns a live VoxelToolTerrain")
	if tool != null:
		# Force the strong branch: an Add sphere centered on an air voxel
		# (SDF +1) must solidify it, regardless of which mesh uploaded first
		# (the old `after != before or after < 0.0` form passed vacuously
		# whenever the pre-edit voxel was already solid).
		assert(bool(terrain.set_voxel_sdf(edit_voxel.x, edit_voxel.y, edit_voxel.z, 1.0)))
		tool.do_sphere(Vector3(edit_voxel.x, edit_voxel.y, edit_voxel.z), 1.5, 0)
		var after := float(terrain.get_voxel_sdf(edit_voxel.x, edit_voxel.y, edit_voxel.z))
		_ok(after < 0.0, "VoxelToolTerrain.do_sphere solidifies a +1 SDF voxel (after=%f)" % after)
		edited_value = after
	_ok(
		bool(terrain.flush_pending_saves())
		and terrain.is_inside_tree()
		and is_equal_approx(
			float(terrain.get_voxel_sdf(edit_voxel.x, edit_voxel.y, edit_voxel.z)),
			edited_value
		),
		"explicit save flush succeeds without deactivating the viewed terrain"
	)

	var remeshed := await _wait_for_remesh(terrain, target_id, target_mesh_id)
	_ok(
		remeshed
		and old_target_ref.get_ref() != null,
		"SDF edit swaps the ArrayMesh on the same MeshInstance3D (node identity preserved)"
	)

	_exercise_invalid_calls(terrain, edit_voxel)

	if remeshed:
		var replacement_ref: WeakRef = weakref(target)
		var viewer := _viewer(terrain)
		_ok(viewer != null, "terrain viewer remains available for unload check")
		if viewer != null:
			viewer.position = Vector3(4096.0, 4096.0, 4096.0)
			var removed := await _wait_for_removal(
				terrain, target_position, replacement_ref
			)
			_ok(
				removed,
				"moving the viewer removes the old object and every mesh at its block position"
			)
			viewer.position = Vector3.ZERO
			remove_child(terrain)
			_ok(
				not terrain.is_inside_tree(),
				"successfully flushed terrain can leave the scene tree"
			)
			add_child(terrain)
			var reentered_mesh := await _wait_for_mesh(terrain)
			_ok(
				reentered_mesh != null and int(terrain.get_mesh_block_count()) > 0,
				"successfully torn-down terrain reinitializes after scene-tree re-entry"
			)

	_ok(await _queue_free_and_wait(terrain), "lifecycle terrain fully frees")


func _run_persistence_checks() -> void:
	print("=== runtime correctness: RegionFiles persistence ===")
	var directory := "user://runtime_correctness_%d_%d" % [
		OS.get_process_id(), Time.get_ticks_usec()
	]
	var terrain := _new_terrain(directory)
	_ok(terrain != null, "RegionFiles terrain and viewer instantiated")
	if terrain == null:
		_ok(
			_remove_generated_directory(directory),
			"temporary RegionFiles directory is safely removed"
		)
		return

	var target := await _wait_for_mesh(terrain)
	_ok(target != null, "RegionFiles terrain uploads a mesh")
	if target == null:
		var terrain_freed := await _queue_free_and_wait(terrain)
		_ok(terrain_freed, "RegionFiles terrain fully frees after a mesh timeout")
		if terrain_freed:
			_ok(
				_remove_generated_directory(directory),
				"temporary RegionFiles directory is safely removed"
			)
		return

	var edit_voxel := _edit_voxel_for_mesh(target)
	var old_value := float(terrain.get_voxel_sdf(edit_voxel.x, edit_voxel.y, edit_voxel.z))
	var persisted_value := -0.375 if old_value >= 0.0 else 0.375
	var edit_ok := bool(terrain.set_voxel_sdf(
		edit_voxel.x, edit_voxel.y, edit_voxel.z, persisted_value
	))
	_ok(
		edit_ok
		and is_equal_approx(
			float(terrain.get_voxel_sdf(edit_voxel.x, edit_voxel.y, edit_voxel.z)),
			persisted_value
		),
		"RegionFiles edit succeeds while its block remains viewed"
	)

	# Free without moving the viewer. VoxelTerrain._exit_tree must flush the
	# still-resident edit before the terrain core is destroyed.
	var terrain_freed := await _queue_free_and_wait(terrain)
	_ok(
		terrain_freed,
		"viewed RegionFiles terrain and stream fully free before recreation"
	)
	if not terrain_freed:
		return

	var recreated := _new_terrain(directory)
	_ok(recreated != null, "terrain recreates against the same RegionFiles directory")
	if recreated == null:
		_ok(
			_remove_generated_directory(directory),
			"temporary RegionFiles directory is safely removed"
		)
		return

	var persisted := false
	var deadline := Time.get_ticks_msec() + WAIT_TIMEOUT_MSEC
	while Time.get_ticks_msec() < deadline:
		await get_tree().process_frame
		var loaded_value := float(recreated.get_voxel_sdf(
			edit_voxel.x, edit_voxel.y, edit_voxel.z
		))
		if is_equal_approx(loaded_value, persisted_value):
			persisted = true
			break
	_ok(
		persisted,
		"edited voxel persists across terrain free and recreation while still viewed"
	)

	var recreated_freed := await _queue_free_and_wait(recreated)
	_ok(
		recreated_freed,
		"recreated RegionFiles terrain and stream fully free before cleanup"
	)
	if recreated_freed:
		_ok(
			_remove_generated_directory(directory),
			"temporary RegionFiles directory is safely removed"
		)


func _run() -> void:
	await _run_lifecycle_checks()
	await _run_persistence_checks()
	print("=== runtime correctness result: %d failure(s) ===" % failures)
	get_tree().quit(1 if failures > 0 else 0)
