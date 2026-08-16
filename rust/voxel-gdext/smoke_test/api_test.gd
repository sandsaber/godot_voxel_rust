extends Node
## API test: exercises the voxel-gdext #[func] surface from GDScript.
## Run as a scene so Godot loads the project GDExtension first.
## Exits with code 0 on success, 1 on any failure.

var failures := 0

func _ok(cond: bool, msg: String) -> void:
	if cond:
		print("  PASS: ", msg)
	else:
		print("  FAIL: ", msg)
		failures += 1

func _ready() -> void:
	print("=== voxel-gdext API test ===")

	# 1. Classes are registered under canonical names (matching upstream
	#    godot_voxel C++ names, no GD suffix): VoxelBuffer, VoxelMesherBlocky, etc.
	var classes := [
		"VoxelTerrain", "VoxelViewer",
		"VoxelGeneratorWaves", "VoxelGeneratorFlat",
		"VoxelBuffer", "VoxelMesherBlocky", "VoxelMesherTransvoxel",
		"VoxelStreamMemory", "VoxelColorPalette", "VoxelBoxMover",
		"VoxelGraphFunction", "VoxelInstanceLibraryItem", "VoxelBlockyType",
		"VoxelInstancer",
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
		#    the next idle frame, but this script exits in _ready). So here we can
		#    only assert the *honest* not-ready behaviour; the real edit path is
		#    exercised in runtime_scene.tscn, which pumps real frames.
		var set_ok = bool(terrain.set_voxel_sdf(0, 0, 0, -1.0))
		var sdf = float(terrain.get_voxel_sdf(0, 0, 0))
		_ok(set_ok == false and sdf == 0.0,
			"set/get_voxel_sdf correctly report not-ready before _ready (set=%s sdf=%f)" % [set_ok, sdf])
		_ok(
			terrain.has_method("flush_pending_saves")
			and bool(terrain.flush_pending_saves()) == false,
			"flush_pending_saves() is registered and safely reports not-ready"
		)

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
		buf.set_block_metadata("chunk")
		_ok(str(buf.get_block_metadata()) == "chunk", "VoxelBuffer block metadata round-trips")
		buf.set_voxel_metadata(Vector3i(1, 2, 3), 42)
		_ok(int(buf.get_voxel_metadata(Vector3i(1, 2, 3))) == 42, "VoxelBuffer voxel metadata round-trips")
		buf.clear_voxel_metadata(Vector3i(1, 2, 3))
		_ok(buf.get_voxel_metadata(Vector3i(1, 2, 3)) == null, "VoxelBuffer clear_voxel_metadata drops the entry")

	# 8. Name-like properties use class-specific accessors: generic `name`
	# cannot be assigned on these Resource subclasses in current Godot, so
	# each class exposes its own getter/setter pair.
	var graph_function: Resource = ClassDB.instantiate("VoxelGraphFunction")
	_ok(graph_function != null, "VoxelGraphFunction instantiated")
	if graph_function:
		graph_function.set_name("smoke_function")
		_ok(graph_function.get_name() == "smoke_function", "VoxelGraphFunction name round-trips via class accessors")
	var instance_item: Resource = ClassDB.instantiate("VoxelInstanceLibraryItem")
	_ok(instance_item != null, "VoxelInstanceLibraryItem instantiated")
	if instance_item:
		instance_item.set_item_name("smoke_item")
		_ok(instance_item.get_item_name() == "smoke_item", "VoxelInstanceLibraryItem name round-trips via class accessors")
	var blocky_type: Resource = ClassDB.instantiate("VoxelBlockyType")
	_ok(blocky_type != null, "VoxelBlockyType instantiated")
	if blocky_type:
		blocky_type.set_unique_name(&"smoke_type")
		_ok(blocky_type.get_unique_name() == &"smoke_type", "VoxelBlockyType unique name round-trips via class accessors")

	# 9. Scene-item instancing: a scene-typed item spawns real Node3D children
	#    per instance (R5), with type flipping and root validation.
	var instancer := ClassDB.instantiate("VoxelInstancer") as Node
	_ok(instancer != null, "VoxelInstancer instantiated")
	if instancer:
		add_child(instancer)
		var item_index := int(instancer.add_item("trees", 1.0, 1.0, 1.0))
		_ok(item_index == 0, "VoxelInstancer.add_item returns index 0")

		# Build a PackedScene whose root is a Node3D, and one whose root is a
		# plain Node (must be rejected).
		var good_root := Node3D.new()
		var good_scene := PackedScene.new()
		var pack_ok := good_scene.pack(good_root)
		good_root.free()
		_ok(pack_ok == OK, "Node3D-rooted helper scene packs")
		var bad_root := Node.new()
		var bad_scene := PackedScene.new()
		bad_scene.pack(bad_root)
		bad_root.free()

		instancer.set_item_scene(item_index, bad_scene)
		_ok(instancer.get_item_scene(item_index) == null,
			"set_item_scene rejects a non-Node3D scene root")
		instancer.set_item_scene(item_index, good_scene)
		_ok(instancer.get_item_scene(item_index) == good_scene,
			"set_item_scene/get_item_scene round-trip")

		var count := int(instancer.scatter_test(4))
		_ok(count == 4, "scatter_test returns 4 (got %d)" % count)
		# Assert by class and transform, not node names (naming is an
		# implementation detail): scatter_test places instances at x = 0..3.
		var scene_children := 0
		var positions := {}
		for child in instancer.get_children():
			if child is Node3D:
				scene_children += 1
				positions[int(roundf(child.position.x))] = true
		_ok(scene_children == 4,
			"scene item spawns one real Node3D per instance (got %d)" % scene_children)
		_ok(
			positions.has(0) and positions.has(1) and positions.has(2) and positions.has(3),
			"scene instances are placed at their scatter positions (x set: %s)" % str(positions.keys())
		)

		# Assigning a mesh flips the item back to MultiMesh and clears the scene.
		var mesh := BoxMesh.new()
		instancer.set_item_mesh(item_index, mesh)
		_ok(instancer.get_item_scene(item_index) == null,
			"set_item_mesh clears the scene slot")
		_ok(int(instancer.scatter_test(2)) == 2, "MultiMesh path still scatters after the flip")

		instancer.queue_free()

	print("=== result: %d failure(s) ===" % failures)
	get_tree().quit(1 if failures > 0 else 0)
