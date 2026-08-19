extends Node
## R9 Stage 2 mesher API smoke test: behavioral anchors for VoxelMesher base
## padding defaults, VoxelMesherTransvoxel build_mesh/build_transition_mesh,
## VoxelMesherCubes set_material_by_index/generate_mesh_from_image,
## VoxelMesherBlocky build_mesh with a baked library, and the
## VoxelColorPalette colors/data properties.
## Exits with code 0 on success, 1 on any failure.

var failures := 0


func _ok(cond: bool, msg: String) -> void:
	if cond:
		print("  PASS: ", msg)
	else:
		print("  FAIL: ", msg)
		failures += 1


func _ready() -> void:
	print("=== voxel-gdext mesher API test ===")
	base_mesher_padding_defaults()
	transvoxel_build_mesh_and_transition_mesh()
	cubes_materials_and_generate_mesh_from_image()
	blocky_build_mesh_with_library()
	palette_properties_and_color_accessors()
	print("=== result: %d failure(s) ===" % failures)
	get_tree().quit(1 if failures > 0 else 0)


## The abstract VoxelMesher base reports 0/0 paddings (concrete meshers
## override these with their algorithm requirements).
func base_mesher_padding_defaults() -> void:
	var mesher: RefCounted = ClassDB.instantiate("VoxelMesher")
	_ok(mesher != null, "VoxelMesher base instantiated")
	if mesher == null:
		return
	_ok(int(mesher.get_minimum_padding()) == 0, "base get_minimum_padding() == 0")
	_ok(int(mesher.get_maximum_padding()) == 0, "base get_maximum_padding() == 0")


## VoxelMesherTransvoxel.build_mesh produces a non-empty ArrayMesh from an SDF
## buffer, and build_transition_mesh returns the seam geometry for one
## direction. The sphere (radius 12) crosses the block boundary so transition
## cells actually contain triangles.
func transvoxel_build_mesh_and_transition_mesh() -> void:
	var mesher: RefCounted = ClassDB.instantiate("VoxelMesherTransvoxel")
	_ok(mesher != null, "VoxelMesherTransvoxel instantiated")
	if mesher == null:
		return
	_ok(int(mesher.get_minimum_padding()) == 1, "transvoxel minimum padding == 1")
	_ok(int(mesher.get_maximum_padding()) == 2, "transvoxel maximum padding == 2")

	var buffer: RefCounted = ClassDB.instantiate("VoxelBuffer")
	buffer.create(18, 18, 18)
	buffer.set_channel_depth(1, 2) # SDF, 32-bit floats
	for z in 18:
		for x in 18:
			for y in 18:
				var d := Vector3(x - 9.0, y - 9.0, z - 9.0).length() - 12.0
				buffer.set_voxel_f(d, x, y, z, 1)

	var mesh: Mesh = mesher.build_mesh(buffer, [], {})
	_ok(mesh != null, "transvoxel build_mesh returned a Mesh")
	if mesh:
		_ok(mesh.get_surface_count() >= 1, "transvoxel build_mesh has a surface")
		_ok(mesh.surface_get_array_len(0) > 0, "transvoxel build_mesh surface has vertices")

	var transition: Mesh = mesher.build_transition_mesh(buffer, 0)
	_ok(transition != null, "transvoxel build_transition_mesh(0) returned a Mesh")
	if transition:
		_ok(transition.get_surface_count() >= 1, "transition mesh has a surface")
		_ok(transition.surface_get_array_len(0) > 0, "transition mesh surface has vertices")
	# Out-of-range directions are rejected with null.
	_ok(mesher.build_transition_mesh(buffer, 6) == null, "transition direction 6 rejected")
	_ok(mesher.build_transition_mesh(buffer, -1) == null, "transition direction -1 rejected")


## VoxelMesherCubes.set_material_by_index routes into the pinned
## opaque/transparent material properties, and the static
## generate_mesh_from_image turns a 16x16 image into a centered, voxel-size
## scaled mesh (opaque + transparent surfaces).
func cubes_materials_and_generate_mesh_from_image() -> void:
	var mesher: RefCounted = ClassDB.instantiate("VoxelMesherCubes")
	_ok(mesher != null, "VoxelMesherCubes instantiated")
	if mesher == null:
		return
	_ok(int(mesher.get_minimum_padding()) == 1, "cubes minimum padding == 1")
	_ok(int(mesher.get_maximum_padding()) == 1, "cubes maximum padding == 1")

	var opaque := StandardMaterial3D.new()
	var transparent := StandardMaterial3D.new()
	mesher.set_material_by_index(0, opaque)
	mesher.set_material_by_index(1, transparent)
	_ok(mesher.get("opaque_material") == opaque, "set_material_by_index(0) sets opaque_material")
	_ok(
		mesher.get("transparent_material") == transparent,
		"set_material_by_index(1) sets transparent_material"
	)

	var image := Image.create_empty(16, 16, false, Image.FORMAT_RGBA8)
	image.fill(Color(1.0, 0.5, 0.25, 1.0))
	var mesh: Mesh = VoxelMesherCubes.generate_mesh_from_image(image, 1.0)
	_ok(mesh != null, "generate_mesh_from_image returned a Mesh")
	if mesh:
		_ok(mesh.get_surface_count() >= 1, "image mesh has a surface")
		var aabb := mesh.get_aabb()
		_ok(
			absf(aabb.size.x - 16.0) < 0.01 and absf(aabb.size.y - 16.0) < 0.01,
			"voxel_size 1.0 mesh spans the 16x16 image (aabb %s)" % str(aabb.size)
		)
	# voxel_size scales the whole mesh.
	var half: Mesh = VoxelMesherCubes.generate_mesh_from_image(image, 0.5)
	_ok(half != null, "generate_mesh_from_image(0.5) returned a Mesh")
	if half:
		var half_aabb := half.get_aabb()
		_ok(
			absf(half_aabb.size.x - 8.0) < 0.01 and absf(half_aabb.size.y - 8.0) < 0.01,
			"voxel_size 0.5 halves the mesh extent (aabb %s)" % str(half_aabb.size)
		)
	# Invalid inputs are rejected with null.
	_ok(
		VoxelMesherCubes.generate_mesh_from_image(image, 0.0) == null,
		"generate_mesh_from_image rejects voxel_size 0"
	)
	var empty := Image.create_empty(0, 0, false, Image.FORMAT_RGBA8)
	_ok(
		VoxelMesherCubes.generate_mesh_from_image(empty, 1.0) == null,
		"generate_mesh_from_image rejects empty images"
	)


## VoxelMesherBlocky.build_mesh meshes a buffer against a baked cube library
## (same setup pattern as blocky_terrain.gd).
func blocky_build_mesh_with_library() -> void:
	var library: RefCounted = ClassDB.instantiate("VoxelBlockyLibrary")
	var cube_id := int(library.add_solid_model(0.6, 0.4, 0.2))
	library.bake()
	var mesher: RefCounted = ClassDB.instantiate("VoxelMesherBlocky")
	_ok(mesher != null, "VoxelMesherBlocky instantiated")
	if mesher == null:
		return
	mesher.set_library(library)
	_ok(int(mesher.get_minimum_padding()) == 1, "blocky minimum padding == 1")
	_ok(int(mesher.get_maximum_padding()) == 1, "blocky maximum padding == 1")

	var buffer: RefCounted = ClassDB.instantiate("VoxelBuffer")
	buffer.create(8, 8, 8)
	for z in range(2, 6):
		for x in range(2, 6):
			for y in range(2, 6):
				buffer.set_voxel(x, y, z, 0, cube_id) # CHANNEL_TYPE

	var mesh: Mesh = mesher.build_mesh(buffer, [], {})
	_ok(mesh != null, "blocky build_mesh returned a Mesh")
	if mesh:
		_ok(mesh.get_surface_count() >= 1, "blocky build_mesh has a surface")
		_ok(mesh.surface_get_array_len(0) > 0, "blocky build_mesh surface has vertices")


## VoxelColorPalette exposes the pinned `colors`/`data` properties (256
## entries) and the Color-based set_color/get_color accessors.
func palette_properties_and_color_accessors() -> void:
	var palette: RefCounted = ClassDB.instantiate("VoxelColorPalette")
	_ok(palette != null, "VoxelColorPalette instantiated")
	if palette == null:
		return
	_ok(int(palette.get("colors").size()) == 256, "colors property has 256 entries")
	_ok(int(palette.get("data").size()) == 256, "data property has 256 entries")

	palette.set_color(3, Color(1.0, 0.0, 0.0, 1.0))
	var red: Color = palette.get_color(3)
	_ok(red is Color and red.r > 0.99 and red.g < 0.01, "get_color returns the set Color")
	# Out-of-range indices report transparent black.
	var bad: Color = palette.get_color(300)
	_ok(bad is Color and bad.r == 0.0 and bad.a == 0.0, "get_color(300) returns transparent black")
	# data round-trips through set_data.
	var data: PackedInt32Array = palette.get("data")
	data[3] = 0x00FF00FF
	palette.set("data", data)
	var check: Color = palette.get_color(3)
	_ok(
		check.g > 0.99 and check.r < 0.01 and check.a > 0.99,
		"data property round-trips packed 0xRRGGBBAA colors"
	)
