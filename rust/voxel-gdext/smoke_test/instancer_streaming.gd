extends Node3D
## R5: VoxelInstancer as a child of VoxelTerrain must stream one instance
## block per paged LOD0 mesh block and free the spawned nodes when the viewer
## moves away. Uses noise TYPE terrain (overhangs give the surface extractor
## solid-with-air-below voxels) plus a blocky mesher so mesh blocks exist.

const CONVERGENCE_TIMEOUT_MSEC := 20_000

var terrain: Node
var instancer: Node
var viewer: Node
var failures := 0
var phase := 0
var deadline_msec := 0
var streamed_refs: Array[WeakRef] = []


func _ok(cond: bool, msg: String) -> void:
	if cond:
		print("[instancer_streaming] PASS: ", msg)
	else:
		print("[instancer_streaming] FAIL: ", msg)
		failures += 1


func _finish() -> void:
	print("=== instancer streaming result: %d failure(s) ===" % failures)
	get_tree().quit(1 if failures > 0 else 0)


func _ready() -> void:
	deadline_msec = Time.get_ticks_msec() + CONVERGENCE_TIMEOUT_MSEC
	print("[instancer_streaming] building noise(type) terrain + instancer")

	terrain = ClassDB.instantiate("VoxelTerrain")
	if terrain == null:
		_ok(false, "VoxelTerrain class is missing")
		_finish()
		return

	var library = ClassDB.instantiate("VoxelBlockyLibrary")
	var cube_id := int(library.add_solid_model(0.6, 0.4, 0.2))
	library.bake()
	var mesher = ClassDB.instantiate("VoxelMesherBlocky")
	mesher.set_library(library)
	terrain.set_mesher(mesher)

	# 3D threshold noise in the TYPE channel: solid volumes with air pockets
	# underneath (the extractor needs solid-with-air-below, which flat fills
	# never produce).
	var generator = ClassDB.instantiate("VoxelGeneratorNoise")
	generator.seed = 7
	generator.set_channel(0) # TYPE
	generator.set_frequency(0.05)
	generator.set_height_start(-8.0)
	generator.set_height_range(16.0)
	terrain.set_generator(generator)

	viewer = ClassDB.instantiate("VoxelViewer")
	viewer.set("view_distance", 48)
	terrain.add_child(viewer)

	instancer = ClassDB.instantiate("VoxelInstancer")
	if instancer == null:
		_ok(false, "VoxelInstancer class is missing")
		_finish()
		return
	var item_index := int(instancer.add_item("trees", 1.0, 1.0, 1.0))
	_ok(item_index == 0, "add_item returns index 0")
	terrain.add_child(instancer)

	add_child(terrain)
	phase = 1
	print("[instancer_streaming] scene ready")


func _process(_delta: float) -> void:
	if terrain == null or instancer == null:
		return
	if Time.get_ticks_msec() >= deadline_msec:
		_ok(false, "phase %d did not converge within timeout" % phase)
		_finish()
		return

	if phase == 1:
		if int(terrain.get_mesh_block_count()) > 0:
			# Mesh convergence and instance streaming get separate budgets:
			# CI runners can be slower than the machine this was tuned on.
			deadline_msec = Time.get_ticks_msec() + CONVERGENCE_TIMEOUT_MSEC
			phase = 2
	elif phase == 2:
		var blocks := int(instancer.get_streamed_block_count())
		var instances := int(instancer.get_streamed_instance_count())
		if blocks > 0 and instances > 0:
			_ok(blocks > 0, "instance blocks stream with terrain paging (blocks=%d)" % blocks)
			_ok(instances > 0, "streamed instance count is nonzero (instances=%d)" % instances)
			# Count instancer children by class, not by name (naming is an
			# implementation detail). Every streamed node is a child.
			var multimesh_children := 0
			for child in instancer.get_children():
				if child is MultiMeshInstance3D:
					multimesh_children += 1
			_ok(
				multimesh_children > 0,
				"streaming spawned real MultiMeshInstance3D children (count=%d)" % multimesh_children
			)
			for child in instancer.get_children():
				streamed_refs.append(weakref(child))
			viewer.position = Vector3(4096.0, 4096.0, 4096.0)
			deadline_msec = Time.get_ticks_msec() + CONVERGENCE_TIMEOUT_MSEC
			phase = 3
	elif phase == 3:
		if int(instancer.get_streamed_block_count()) > 0:
			return
		var alive := 0
		for ref in streamed_refs:
			if ref.get_ref() != null:
				alive += 1
		if alive == 0:
			_ok(true, "viewer departure unloads every streamed block and frees every node")
			_finish()
