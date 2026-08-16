extends Node3D
## 3-LOD Variable LOD integration check (Phase C step C1).
##
## Builds a `VoxelLodTerrain` with `lod_count = 3` (the production Variable LOD
## planner path), a Waves generator, and a viewer, then verifies that the
## multi-LOD paging pipeline produces mesh blocks and reacts to viewer
## movement across negative coordinates. The acceptance criteria:
##
##  - the terrain reports `lod_count == 3`;
##  - paging converges to a nonzero mesh block count (visual demand paged in);
##  - moving the viewer to fresh coordinates (including negatives) keeps the
##    route reactive (block count stays nonzero / refreshes);
##  - the route never panics or emits a fatal Rust diagnostic during paging.
##
## This is a headless integration check (no renderer needed); it relies on the
## production `try_process` multi-LOD path being the planner cutover.

const CONVERGENCE_TIMEOUT_MSEC := 20_000
const REPORT_INTERVAL_MSEC := 2_000
const MOVE_AT_FRAME := 30

var terrain: Node
var viewer: Node
var frames := 0
var failures := 0
var deadline_msec := 0
var next_report_msec := 0
var converged_once := false
var pre_move_count := 0


func _ready() -> void:
	var now := Time.get_ticks_msec()
	deadline_msec = now + CONVERGENCE_TIMEOUT_MSEC
	next_report_msec = now + REPORT_INTERVAL_MSEC
	print("[variable_lod_3] building 3-LOD VoxelLodTerrain + viewer + generator")
	terrain = ClassDB.instantiate("VoxelLodTerrain")
	if terrain == null:
		_fail("VoxelLodTerrain class is missing")
		_finish()
		return
	# Configure the LOD count BEFORE adding the node to the tree: _ready()
	# constructs the Variable LOD core with this count and rejects later changes.
	terrain.set_lod_count(3)
	if terrain.has_method("set_generate_collisions"):
		terrain.set_generate_collisions(true)
	elif terrain.has_method("set_generate_collision"):
		terrain.set_generate_collision(true)
	add_child(terrain)
	var lod_count := int(terrain.get_lod_count())
	if lod_count == 3:
		print("[variable_lod_3] PASS lod_count == 3")
	else:
		_fail("lod_count == 3 (got %d)" % lod_count)
	var gen: Resource = ClassDB.instantiate("VoxelGeneratorWaves")
	if gen:
		terrain.set_generator(gen)
	else:
		_fail("VoxelGeneratorWaves class is missing")
	viewer = ClassDB.instantiate("VoxelViewer")
	if viewer:
		terrain.add_child(viewer)
		if viewer.has_method("set_world_position"):
			viewer.set_world_position(Vector3(0.0, 0.0, 0.0))
	else:
		_fail("VoxelViewer class is missing")
	print("[variable_lod_3] scene ready, generator + viewer assigned")


func _process(_delta: float) -> void:
	frames += 1
	var now := Time.get_ticks_msec()
	var bc := int(terrain.get_mesh_block_count())
	if not converged_once and bc > 0:
		converged_once = true
		pre_move_count = bc
		print("[variable_lod_3] PASS nonzero mesh upload — mesh_block_count=%d" % bc)
	if frames == MOVE_AT_FRAME and viewer != null:
		# Move the viewer to a fresh position crossing negative coordinates so
		# the multi-LOD route must page out the old demand and page in new blocks
		# (exercises split/join ordering across LODs and negative-coordinate
		# canonicalization).
		if viewer.has_method("set_world_position"):
			viewer.set_world_position(Vector3(-96.0, 0.0, -96.0))
		print("[variable_lod_3] viewer moved to (-96, 0, -96) at frame %d" % frames)
	if now >= next_report_msec:
		print("[variable_lod_3] elapsed_ms=%d — mesh_block_count=%d" % [
			CONVERGENCE_TIMEOUT_MSEC - (deadline_msec - now), bc
		])
		next_report_msec = now + REPORT_INTERVAL_MSEC
	# Finish once we converged AND have observed the post-move state for a while.
	if converged_once and frames > MOVE_AT_FRAME + 15:
		if bc > 0:
			print("[variable_lod_3] PASS route reactive after move — mesh_block_count=%d (pre-move=%d)" % [bc, pre_move_count])
		else:
			_fail("route reactive after move (mesh_block_count dropped to 0)")
		_finish()
	elif now >= deadline_msec:
		if not converged_once:
			_fail("nonzero mesh upload within %d ms (last count=%d)" % [CONVERGENCE_TIMEOUT_MSEC, bc])
		_finish()


func _fail(message: String) -> void:
	print("[variable_lod_3] FAIL %s" % message)
	failures += 1


func _finish() -> void:
	print("[variable_lod_3] DONE with %d failure(s)" % failures)
	get_tree().quit(1 if failures > 0 else 0)
