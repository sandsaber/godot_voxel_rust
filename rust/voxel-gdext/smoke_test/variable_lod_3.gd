extends Node3D
## Exercise actual viewer motion and the active LOD frontier, including a
## translated/rotated terrain. Payload residency alone is not visual coverage.

const CONVERGENCE_TIMEOUT_MSEC := 20_000
const STABLE_MSEC := 250
const DESTINATION := Vector3(-96.0, 0.0, -96.0)

var terrain: Node3D
var viewer: Node3D
var failures := 0
var deadline_msec := 0
var stable_since := 0
var moved_at := 0
var previous_locations: Dictionary = {}
var before_move: Dictionary = {}
var finished := false


func _ready() -> void:
	terrain = ClassDB.instantiate("VoxelLodTerrain") as Node3D
	viewer = ClassDB.instantiate("VoxelViewer") as Node3D
	var generator := ClassDB.instantiate("VoxelGeneratorWaves") as Resource
	if terrain == null or viewer == null or generator == null:
		_fail("required classes are missing")
		_finish()
		return
	terrain.set_lod_count(3)
	terrain.set_generate_collision(true)
	terrain.set_generator(generator)
	terrain.position = Vector3(300.0, 0.0, 200.0)
	terrain.rotation.y = PI / 2.0
	viewer.set_view_distance(64)
	terrain.add_child(viewer)
	add_child(terrain)
	viewer.global_position = terrain.to_global(Vector3.ZERO)
	if terrain.get_lod_count() != 3:
		_fail("lod_count must be 3")
	deadline_msec = Time.get_ticks_msec() + CONVERGENCE_TIMEOUT_MSEC
	stable_since = Time.get_ticks_msec()


func _locations() -> Dictionary:
	var packed: PackedInt32Array = terrain.get_mesh_block_locations()
	var result: Dictionary = {}
	for i in range(0, packed.size(), 4):
		result[Vector4i(packed[i], packed[i + 1], packed[i + 2], packed[i + 3])] = true
	return result


func _same_locations(a: Dictionary, b: Dictionary) -> bool:
	if a.size() != b.size():
		return false
	for key in a:
		if not b.has(key):
			return false
	return true


func _has_difference(a: Dictionary, b: Dictionary) -> bool:
	for key in a:
		if not b.has(key):
			return true
	return false


func _check_active_frontier() -> void:
	var active: Array[AABB] = []
	var colliders := 0
	for child in terrain.get_children():
		if not child is MeshInstance3D:
			continue
		var mesh := child as MeshInstance3D
		if mesh.visible and mesh.mesh != null:
			var parts := String(mesh.name).split("_")
			var lod := int(parts[1].trim_prefix("lod"))
			var bounds := AABB(mesh.position, Vector3.ONE * float(16 << lod))
			for other in active:
				if bounds.intersects(other):
					_fail("active parent/child meshes overlap")
			active.append(bounds)
		for body in mesh.get_children():
			if body is StaticBody3D and body.collision_layer != 0:
				colliders += 1
	if active.is_empty():
		_fail("active visual coverage is empty")
	if colliders == 0:
		_fail("active collision coverage is empty")


func _process(_delta: float) -> void:
	if finished or terrain == null or viewer == null:
		return
	var now := Time.get_ticks_msec()
	var locations := _locations()
	if not _same_locations(locations, previous_locations):
		previous_locations = locations
		stable_since = now
	if not locations.is_empty() and now - stable_since >= STABLE_MSEC:
		if moved_at == 0:
			_check_active_frontier()
			# The transformed terrain must still page around its local origin.
			var near_origin := false
			for key in locations:
				if key.w == 0 and abs(key.x) <= 2 and abs(key.z) <= 2:
					near_origin = true
			if not near_origin:
				_fail("viewer was not converted to terrain-local coordinates")
			before_move = locations.duplicate()
			viewer.global_position = terrain.to_global(DESTINATION)
			if not viewer.position.is_equal_approx(DESTINATION):
				_fail("viewer movement readback failed")
			moved_at = now
			stable_since = now
			print("[variable_lod_3] viewer moved; checking entered AND exited blocks")
		elif now - moved_at >= STABLE_MSEC and _has_difference(locations, before_move) and _has_difference(before_move, locations):
			_check_active_frontier()
			print("[variable_lod_3] PASS paging changed the resident block set after actual motion")
			_finish()
	if now >= deadline_msec:
		_fail("paging/movement did not converge before deadline")
		_finish()


func _fail(message: String) -> void:
	print("[variable_lod_3] FAIL ", message)
	failures += 1


func _finish() -> void:
	finished = true
	print("[variable_lod_3] DONE with %d failure(s)" % failures)
	get_tree().quit(1 if failures > 0 else 0)
