extends Node3D
## R1: type-channel Flat + baked VoxelBlockyLibrary + VoxelMesherBlocky
## must produce visible mesh blocks on VoxelTerrain.

const CONVERGENCE_TIMEOUT_MSEC := 20_000

var terrain: Node
var failures := 0
var deadline_msec := 0


func _ready() -> void:
	deadline_msec = Time.get_ticks_msec() + CONVERGENCE_TIMEOUT_MSEC
	print("[blocky_terrain] building VoxelTerrain + Flat(type) + baked cube library")
	terrain = ClassDB.instantiate("VoxelTerrain")
	if terrain == null:
		_fail("VoxelTerrain class is missing")
		_finish()
		return

	var library = ClassDB.instantiate("VoxelBlockyLibrary")
	if library == null:
		_fail("VoxelBlockyLibrary class is missing")
		_finish()
		return
	var cube_id := int(library.add_solid_model(0.6, 0.4, 0.2))
	if cube_id < 1:
		_fail("add_solid_model should return id >= 1 (0 is air), got %d" % cube_id)
	library.bake()

	var mesher = ClassDB.instantiate("VoxelMesherBlocky")
	if mesher == null:
		_fail("VoxelMesherBlocky class is missing")
		_finish()
		return
	mesher.set_library(library)
	terrain.set_mesher(mesher)

	var generator = ClassDB.instantiate("VoxelGeneratorFlat")
	if generator == null:
		_fail("VoxelGeneratorFlat class is missing")
		_finish()
		return
	generator.set_channel(0) # TYPE
	generator.set_voxel_type(cube_id)
	generator.set_height(8.0)
	terrain.set_generator(generator)

	var viewer = ClassDB.instantiate("VoxelViewer")
	if viewer == null:
		_fail("VoxelViewer class is missing")
		_finish()
		return
	viewer.set("view_distance", 48)
	terrain.add_child(viewer)
	add_child(terrain)
	print("[blocky_terrain] scene ready")


func _process(_delta: float) -> void:
	if terrain == null:
		return
	var bc := int(terrain.get_mesh_block_count())
	if bc > 0:
		print("[blocky_terrain] PASS mesh_block_count=%d" % bc)
		_finish()
		return
	if Time.get_ticks_msec() >= deadline_msec:
		_fail("nonzero mesh upload within timeout (last count=%d)" % bc)
		_finish()


func _fail(message: String) -> void:
	print("[blocky_terrain] FAIL %s" % message)
	failures += 1


func _finish() -> void:
	print("[blocky_terrain] DONE with %d failure(s)" % failures)
	get_tree().quit(1 if failures > 0 else 0)
