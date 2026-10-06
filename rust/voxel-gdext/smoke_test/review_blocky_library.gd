extends Node
## Blocky library regressions: rebaking edited resources and preserving model IDs.

var failures := 0


func _ok(condition: bool, message: String) -> void:
	if condition:
		print("  PASS: ", message)
	else:
		print("  FAIL: ", message)
		failures += 1


func _mesh_for(library: RefCounted, model_id: int) -> Mesh:
	var mesher: RefCounted = ClassDB.instantiate("VoxelMesherBlocky")
	mesher.set_library(library)
	mesher.set_occlusion_enabled(false)
	var buffer: RefCounted = ClassDB.instantiate("VoxelBuffer")
	buffer.create(8, 8, 8)
	buffer.set_voxel(3, 3, 3, 0, model_id)
	return mesher.build_mesh(buffer, [], {})


func _vertex_count(mesh: Mesh) -> int:
	if mesh == null:
		return 0
	var count := 0
	for surface in range(mesh.get_surface_count()):
		count += mesh.surface_get_array_len(surface)
	return count


func _first_color(mesh: Mesh) -> Color:
	if mesh == null:
		return Color.BLACK
	for surface in range(mesh.get_surface_count()):
		var arrays: Array = mesh.surface_get_arrays(surface)
		var colors: PackedColorArray = arrays[Mesh.ARRAY_COLOR]
		if colors.size() > 0:
			return colors[0]
	return Color.BLACK


func _is_color(actual: Color, expected: Color) -> bool:
	return actual.is_equal_approx(expected)


func _ready() -> void:
	print("=== blocky library rebake and sparse model IDs ===")
	var cube: RefCounted = ClassDB.instantiate("VoxelBlockyModelCube")
	cube.set_color(1.0, 0.0, 0.0, 1.0)
	var library: RefCounted = ClassDB.instantiate("VoxelBlockyLibrary")
	var cube_id := int(library.add_model(cube))
	library.bake()
	_ok(cube_id == 1, "first model keeps ID 1 after reserved air")
	_ok(_is_color(_first_color(_mesh_for(library, cube_id)), Color.RED),
		"initial bake uses the model's red color")

	cube.set_color(0.0, 1.0, 0.0, 1.0)
	library.bake()
	_ok(library.get_model(cube_id) == cube, "library retains the edited resource")
	_ok(_is_color(_first_color(_mesh_for(library, cube_id)), Color.GREEN),
		"rebake uses the edited green color")
	library.bake()
	_ok(_is_color(_first_color(_mesh_for(library, cube_id)), Color.GREEN),
		"repeated bake keeps the edited model")

	var blue_id := int(library.add_solid_model(0.0, 0.0, 1.0))
	library.bake()
	_ok(blue_id == 2 and int(library.get_model_count()) == 3,
		"adding a model after baking retains sequential IDs")
	_ok(_is_color(_first_color(_mesh_for(library, cube_id)), Color.GREEN),
		"adding another model does not revert the edited model")
	_ok(_is_color(_first_color(_mesh_for(library, blue_id)), Color.BLUE),
		"new solid model is baked at its returned ID")

	var roundtrip: RefCounted = ClassDB.instantiate("VoxelBlockyLibrary")
	roundtrip.set_models(library.get_models())
	roundtrip.bake()
	_ok(int(roundtrip.get_model_count()) == 3,
		"get_models/set_models roundtrip keeps the model count")
	_ok(_is_color(_first_color(_mesh_for(roundtrip, cube_id)), Color.GREEN),
		"get_models/set_models roundtrip keeps edited model ID 1")
	_ok(_is_color(_first_color(_mesh_for(roundtrip, blue_id)), Color.BLUE),
		"get_models/set_models roundtrip keeps solid model ID 2")

	var cube3: RefCounted = ClassDB.instantiate("VoxelBlockyModelCube")
	var sparse: RefCounted = ClassDB.instantiate("VoxelBlockyLibrary")
	sparse.set_models([null, cube, null, cube3])
	sparse.bake()
	_ok(int(sparse.get_model_count()) == 4 and sparse.get_models().size() == 4,
		"null model slot retains its index")
	_ok(sparse.get_model(2) == null and sparse.get_model(3) == cube3,
		"get_model reports the empty slot and model at ID 3")
	_ok(_vertex_count(_mesh_for(sparse, 2)) == 0 and
		_vertex_count(_mesh_for(sparse, 3)) > 0,
		"mesher uses the model at ID 3 and leaves ID 2 empty")
	var sparse_roundtrip: RefCounted = ClassDB.instantiate("VoxelBlockyLibrary")
	sparse_roundtrip.set_models(sparse.get_models())
	sparse_roundtrip.bake()
	_ok(int(sparse_roundtrip.get_model_count()) == 4 and
		_vertex_count(_mesh_for(sparse_roundtrip, 2)) == 0 and
		_vertex_count(_mesh_for(sparse_roundtrip, 3)) > 0,
		"sparse get_models/set_models roundtrip keeps voxel IDs")

	print("=== result: %d failure(s) ===" % failures)
	get_tree().quit(1 if failures > 0 else 0)
