extends SceneTree
## Standalone API test: exercises the voxel-gdext #[func] surface from GDScript.
## Runs headless:  godot --headless --script api_test.gd
## Exits with code 0 on success, 1 on any failure.

var failures := 0

func _ok(cond: bool, msg: String) -> void:
	if cond:
		print("  PASS: ", msg)
	else:
		print("  FAIL: ", msg)
		failures += 1

func _init() -> void:
	print("=== voxel-gdext API test ===")

	# 1. Classes are registered under canonical names (matching upstream
	#    godot_voxel C++ names, no GD suffix): VoxelBuffer, VoxelMesherBlocky, etc.
	var classes := [
		"VoxelTerrain", "VoxelViewer",
		"VoxelGeneratorWaves", "VoxelGeneratorFlat",
		"VoxelBuffer", "VoxelMesherBlocky", "VoxelMesherTransvoxel",
		"VoxelStreamMemory", "VoxelColorPalette", "VoxelBoxMover",
		"VoxelGraphFunction", "VoxelInstanceLibraryItem", "VoxelBlockyType",
	]
	var missing := 0
	for c in classes:
		var exists: bool = ClassDB.class_exists(c)
		if not exists:
			missing += 1
			print("  FAIL: ", c, " class NOT registered")
	_ok(missing == 0, "%d/%d expected classes registered" % [classes.size() - missing, classes.size()])

	# 2. Instantiate a terrain and inspect its #[func] API.
	var terrain := ClassDB.instantiate("VoxelTerrain") as Node
	_ok(terrain != null, "VoxelTerrain instantiated")
	if terrain:
		var v: String = terrain.get_version()
		_ok(v.length() > 0, "get_version() returns '%s'" % v)
		_ok(int(terrain.get_lod_count()) >= 1, "get_lod_count() >= 1 (got %d)" % int(terrain.get_lod_count()))

		# 3. Assign a generator resource (wires up the generator property).
		var gen := ClassDB.instantiate("VoxelGeneratorWaves") as Resource
		_ok(gen != null, "VoxelGeneratorWaves instantiated")
		if gen:
			terrain.set_generator(gen)
			var got = terrain.get_generator()
			_ok(got != null, "set_generator/get_generator round-trips resource")

		# 4. Voxel read/write API (edition surface). NOTE: set_voxel_sdf returns
		#    false and get_voxel_sdf returns 0.0 until the terrain's _ready() has
		#    run (which initialises the core). _ready() does NOT fire
		#    synchronously from add_child in a SceneTree --script run (it runs on
		#    the next idle frame, but this script exits in _init). So here we can
		#    only assert the *honest* not-ready behaviour; the real edit path is
		#    exercised in runtime_scene.tscn, which pumps real frames.
		var set_ok = bool(terrain.set_voxel_sdf(0, 0, 0, -1.0))
		var sdf = float(terrain.get_voxel_sdf(0, 0, 0))
		_ok(set_ok == false and sdf == 0.0,
			"set/get_voxel_sdf correctly report not-ready before _ready (set=%s sdf=%f)" % [set_ok, sdf])

		# 5. Bounds querying. NOTE: a node created outside the tree never runs
		#    _ready(), so the terrain core may report empty bounds until it is
		#    added to the scene and paged. Both empty and 6-int results are valid.
		var bounds = terrain.get_bounds()
		var bsize = int(bounds.size())
		_ok(bsize == 0 or bsize == 6, "get_bounds() returns 0 (un-ready) or 6 ints (got %d)" % bsize)

		# 6. Collision/material properties round-trip.
		terrain.set_generate_collision(true)
		_ok(bool(terrain.get_generate_collision()) == true, "generate_collision round-trips true")
		var mat := StandardMaterial3D.new()
		terrain.set_material_override(mat)
		_ok(terrain.get_material_override() == mat, "material_override round-trips resource")

		terrain.queue_free()

	# 7. A pure-data resource: VoxelBuffer allocate + write/read a voxel.
	var buf: RefCounted = ClassDB.instantiate("VoxelBuffer")
	_ok(buf != null, "VoxelBuffer instantiated")
	if buf:
		buf.create(16, 16, 16)
		buf.set_voxel(0, 0, 0, 0, 7)
		var iv = buf.get_voxel(0, 0, 0, 0)
		_ok(int(iv) == 7, "VoxelBuffer set_voxel/get_voxel round-trips (got %d)" % int(iv))

	# 8. Name-like properties use class-specific accessors. Generic
	#    get_name/set_name methods shadow Resource methods and become errors in
	#    godot-rust 0.6.
	var graph_function: Resource = ClassDB.instantiate("VoxelGraphFunction")
	graph_function.name = "smoke_function"
	_ok(graph_function.get_function_name() == "smoke_function", "VoxelGraphFunction.name round-trips")
	var instance_item: Resource = ClassDB.instantiate("VoxelInstanceLibraryItem")
	instance_item.name = "smoke_item"
	_ok(instance_item.get_item_name() == "smoke_item", "VoxelInstanceLibraryItem.name round-trips")
	var blocky_type: Resource = ClassDB.instantiate("VoxelBlockyType")
	blocky_type.name = "smoke_type"
	_ok(blocky_type.get_type_name() == "smoke_type", "VoxelBlockyType.name round-trips")

	print("=== result: %d failure(s) ===" % failures)
	quit(1 if failures > 0 else 0)
