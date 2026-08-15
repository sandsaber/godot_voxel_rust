extends Node3D
## Attached to the runtime paging scene. Builds terrain+viewer+generator at
## runtime, lets paging/meshing run for several real frames, then reports.
const CONVERGENCE_TIMEOUT_MSEC := 20_000
const REPORT_INTERVAL_MSEC := 1_000

var terrain: Node
var frames := 0
var failures := 0
var deadline_msec := 0
var next_report_msec := 0

func _ready() -> void:
	var now := Time.get_ticks_msec()
	deadline_msec = now + CONVERGENCE_TIMEOUT_MSEC
	next_report_msec = now + REPORT_INTERVAL_MSEC
	print("[runtime] building terrain + viewer + generator")
	terrain = ClassDB.instantiate("VoxelTerrain")
	if terrain == null:
		print("[runtime] FAIL VoxelTerrain class is missing")
		failures += 1
		get_tree().quit(1)
		return
	add_child(terrain)
	var gen: Resource = ClassDB.instantiate("VoxelGeneratorWaves")
	if gen:
		terrain.set_generator(gen)
	else:
		print("[runtime] FAIL VoxelGeneratorWaves class is missing")
		failures += 1
	var viewer: Node = ClassDB.instantiate("VoxelViewer")
	if viewer:
		terrain.add_child(viewer)  # viewer must be a child of terrain
	else:
		print("[runtime] FAIL VoxelViewer class is missing")
		failures += 1
	print("[runtime] scene ready, generator + viewer assigned")

func _process(_delta: float) -> void:
	frames += 1
	if frames == 1:
		print("[runtime] frame 1 reached — paging pipeline active")
		# Now that _ready() has run (core is live), exercise the edition API
		# strictly: set_voxel_sdf must report success and the value must stick.
		var set_ok = bool(terrain.set_voxel_sdf(0, 0, 0, -1.0))
		var sdf = float(terrain.get_voxel_sdf(0, 0, 0))
		if set_ok and sdf == -1.0:
			print("[runtime] PASS set_voxel_sdf/get_voxel_sdf (set=true sdf=%f)" % sdf)
		else:
			print("[runtime] FAIL set_voxel_sdf/get_voxel_sdf (set=%s sdf=%f, expected true/-1.0)" % [set_ok, sdf])
			failures += 1
	var now := Time.get_ticks_msec()
	var bc := int(terrain.get_mesh_block_count())
	if now >= next_report_msec:
		print("[runtime] elapsed_ms=%d — mesh_block_count=%d" % [
			CONVERGENCE_TIMEOUT_MSEC - (deadline_msec - now), bc
		])
		next_report_msec = now + REPORT_INTERVAL_MSEC
	if frames > 1 and bc > 0:
		print("[runtime] PASS nonzero mesh upload after %d ms — mesh_block_count=%d" % [
			CONVERGENCE_TIMEOUT_MSEC - (deadline_msec - now), bc
		])
		print("[runtime] DONE with %d failure(s)" % failures)
		get_tree().quit(1 if failures > 0 else 0)
	elif now >= deadline_msec:
		print("[runtime] FAIL zero mesh upload after %d ms" % CONVERGENCE_TIMEOUT_MSEC)
		failures += 1
		print("[runtime] DONE with %d failure(s)" % failures)
		get_tree().quit(1 if failures > 0 else 0)
