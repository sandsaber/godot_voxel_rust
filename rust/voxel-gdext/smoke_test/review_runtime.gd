extends Node3D
## Behavioral regressions for live terrain settings, random ticks and image sizes.

var failures := 0
const TIMEOUT_MSEC := 10_000


func _ready() -> void:
	get_tree().create_timer(30.0).timeout.connect(func():
		push_error("review runtime checks did not finish")
		get_tree().quit(1)
	)
	call_deferred("_run")


func _ok(condition: bool, message: String) -> void:
	if condition:
		print("  PASS: ", message)
	else:
		print("  FAIL: ", message)
		failures += 1


func _new_terrain() -> Node3D:
	var terrain: Node3D = ClassDB.instantiate("VoxelLodTerrain")
	terrain.set_lod_count(3)
	terrain.set_lod_distance(16)
	terrain.set_secondary_lod_distance(16)
	terrain.set_voxel_bounds(AABB(Vector3.ONE * -128.0, Vector3.ONE * 256.0))
	terrain.set_generator(ClassDB.instantiate("VoxelGeneratorWaves"))
	var viewer: Node3D = ClassDB.instantiate("VoxelViewer")
	viewer.name = "Viewer"
	viewer.set_view_distance(48)
	terrain.add_child(viewer)
	add_child(terrain)
	var deadline := Time.get_ticks_msec() + TIMEOUT_MSEC
	while terrain.get_mesh_block_count() == 0 and Time.get_ticks_msec() < deadline:
		await get_tree().process_frame
	_ok(terrain.get_mesh_block_count() > 0, "LOD terrain loads mesh blocks")
	return terrain


func _ticks(tool: RefCounted) -> Array:
	var positions: Array = []
	tool.run_blocky_random_tick(
		AABB(Vector3.ONE, Vector3.ONE * 4.0), 4,
		func(pos, _value): positions.append(pos), 4, 0
	)
	return positions


func _bounds_and_random_tick() -> void:
	var terrain := await _new_terrain()
	var tool: RefCounted = terrain.get_voxel_tool()
	tool.set_channel(0)
	tool.set_voxel(Vector3i.ONE, 77)
	_ok(tool.get_voxel(Vector3i.ONE) == 77, "edit is resident before bounds change")
	var original_bounds: AABB = terrain.get_voxel_bounds()
	terrain.set_voxel_bounds(AABB(Vector3.ONE * -128.0, Vector3.ONE * 384.0))
	await get_tree().process_frame
	_ok(terrain.get_voxel_bounds() == original_bounds, "live bounds change preserves original bounds")
	_ok(terrain.get_statistics() != null, "live bounds change preserves the running core")
	_ok(tool.get_voxel(Vector3i.ONE) == 77, "live bounds change preserves unsaved edits")
	tool.set_voxel(Vector3i.ONE, 88)
	_ok(tool.get_voxel(Vector3i.ONE) == 88, "terrain remains editable after rejected bounds change")
	tool.set_value(7)
	tool.do_box(Vector3i.ONE, Vector3i.ONE * 4, 2)
	tool.set_seed(123)
	var first := _ticks(tool)
	var second := _ticks(tool)
	_ok(first.size() == 4 and second.size() == 4, "random ticks return the requested draws")
	_ok(first != second, "successive random ticks advance the random sequence")
	tool.set_seed(123)
	_ok(_ticks(tool) == first and _ticks(tool) == second, "resetting the seed replays the sequence")
	# Paired tools must have independent, reproducible random state.
	var other: RefCounted = terrain.get_voxel_tool()
	other.set_channel(0)
	other.set_seed(123)
	_ok(_ticks(other) == first, "each tool owns its random sequence")
	terrain.queue_free()
	await get_tree().process_frame


func _live_lod_distances() -> void:
	var terrain := await _new_terrain()
	var tool: RefCounted = terrain.get_voxel_tool()
	tool.set_channel(0)
	tool.set_voxel(Vector3i.ONE, 99)
	for i in range(2):
		var secondary := 32.0 if i == 0 else 16.0
		terrain.set_secondary_lod_distance(secondary)
		terrain.set_lod_distance(secondary)
		var before: int = terrain.get_mesh_revision()
		# A real edit forces new mesh work even when both distance settings
		# happen to select the same frontier inside these finite bounds.
		tool.set_channel(1)
		tool.do_sphere(Vector3.ZERO, 4.0, 1 if i == 0 else 0)
		var deadline := Time.get_ticks_msec() + TIMEOUT_MSEC
		while terrain.get_mesh_revision() == before and Time.get_ticks_msec() < deadline:
			await get_tree().process_frame
		_ok(terrain.get_mesh_revision() > before, "meshing continues after changing live LOD distances")
		tool.set_channel(0)
		_ok(tool.get_voxel(Vector3i.ONE) == 99, "LOD distance change preserves voxel edits")
	terrain.queue_free()
	await get_tree().process_frame


func _height_dimensions() -> void:
	var generator: Resource = ClassDB.instantiate("VoxelGeneratorImage")
	_ok(generator.set_heights(PackedByteArray([0, 255, 128, 64]), 2, 2), "valid heightmap is accepted")
	for size in [Vector2i(50_000, 50_000), Vector2i(65_536, 65_536), Vector2i(0, 2), Vector2i(-2, 2)]:
		_ok(not generator.set_heights(PackedByteArray(), size.x, size.y), "invalid heightmap dimensions are rejected: %s" % size)
		_ok(generator.has_image(), "invalid heightmap dimensions preserve the previous image")
	_ok(not generator.set_heights(PackedByteArray([1]), 2, 2), "mismatched heightmap length is rejected")


func _stream_sector_sizes() -> void:
	var stream: Resource = ClassDB.instantiate("VoxelStreamRegionFiles")
	stream.set_sector_size(1024)
	for size in [-1, 0, 65536, 2147483647]:
		stream.set_sector_size(size)
		_ok(stream.get_sector_size() == 1024, "invalid sector size preserves the current setting: %d" % size)
	stream.set_sector_size(65535)
	_ok(stream.get_sector_size() == 65535, "largest serializable sector size is accepted")


func _run() -> void:
	await _bounds_and_random_tick()
	await _live_lod_distances()
	_height_dimensions()
	_stream_sector_sizes()
	print("=== review runtime result: %d failure(s) ===" % failures)
	get_tree().quit(1 if failures > 0 else 0)
