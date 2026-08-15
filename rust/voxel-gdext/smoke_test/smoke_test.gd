extends Node3D

var failures := 0

func _ready() -> void:
	var terrain = get_node_or_null("VoxelTerrain")
	if terrain:
		print("Smoke test: VoxelTerrain found, type=", terrain.get_class())
		if terrain.has_method("get_version"):
			print("Version: ", terrain.get_version())
		if terrain.has_method("get_lod_count"):
			print("LOD count: ", terrain.get_lod_count())
		if terrain.has_method("get_mesh_block_count"):
			print("Mesh block count: ", terrain.get_mesh_block_count())
	else:
		print("Smoke test: VoxelTerrain NOT found")
		failures += 1
	call_deferred("_finish")


func _finish() -> void:
	print("Smoke scene result: %d failure(s)" % failures)
	get_tree().quit(1 if failures > 0 else 0)
