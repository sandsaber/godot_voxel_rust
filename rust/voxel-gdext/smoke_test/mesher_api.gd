extends Node
## R9 Stage 2 mesher API smoke test: behavioral anchors for VoxelMesher base
## padding defaults, VoxelMesherTransvoxel build_mesh/build_transition_mesh,
## VoxelMesherCubes set_material_by_index/generate_mesh_from_image/palette
## color modes/material-slot fallbacks, VoxelMesherBlocky build_mesh with a
## baked library, the VoxelColorPalette colors/data properties, and the
## VoxelRaycastResult pinned members.
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
	cubes_palette_mode_build_mesh()
	cubes_generate_mesh_from_image_orientation()
	blocky_build_mesh_with_library()
	palette_properties_and_color_accessors()
	raycast_result_members()
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


## Shared setup for the cubes palette/material checks: a 6x6x6 buffer on
## CHANNEL_COLOR with three distinct voxel values (1 and 2 opaque via the
## palette, 3 half-transparent).
func _make_cubes_palette_buffer() -> RefCounted:
	var buffer: RefCounted = ClassDB.instantiate("VoxelBuffer")
	buffer.create(6, 6, 6)
	buffer.set_voxel(1, 1, 1, 2, 1) # value 1 — palette red (opaque)
	buffer.set_voxel(4, 1, 1, 2, 2) # value 2 — palette green (opaque)
	buffer.set_voxel(1, 4, 1, 2, 3) # value 3 — palette blue (half-transparent)
	return buffer


## Palette matching _make_cubes_palette_buffer's voxel values: red, green and
## a half-alpha blue so the mesh splits into opaque + transparent surfaces.
func _make_cubes_palette() -> RefCounted:
	var palette: RefCounted = ClassDB.instantiate("VoxelColorPalette")
	palette.set_color(1, Color(1.0, 0.0, 0.0, 1.0))
	palette.set_color(2, Color(0.0, 1.0, 0.0, 1.0))
	palette.set_color(3, Color(0.0, 0.0, 1.0, 0.5))
	return palette


## Whether a packed color array contains a color within per-channel tolerance.
func _colors_contain(colors: PackedColorArray, expected: Color, tolerance := 0.01) -> bool:
	for color in colors:
		if (
			absf(color.r - expected.r) < tolerance
			and absf(color.g - expected.g) < tolerance
			and absf(color.b - expected.b) < tolerance
			and absf(color.a - expected.a) < tolerance
		):
			return true
	return false


## VoxelMesherCubes.set_material_by_index routes into the pinned
## opaque/transparent material properties, the stored materials genuinely
## apply when build_mesh is called without a materials array (upstream
## get_material_by_index fallback; a caller array wins for the slots it
## covers, and null entries keep their position instead of shifting), and the
## static generate_mesh_from_image turns a 16x16 image into a voxel-size
## scaled mesh.
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

	# Palette mode so the buffer produces both an opaque and a transparent
	# surface (slots 0 and 1) for the fallback assertions below.
	mesher.set("palette", _make_cubes_palette())
	mesher.set_color_mode(VoxelMesherCubes.COLOR_MESHER_PALETTE)
	var buffer: RefCounted = _make_cubes_palette_buffer()

	# No materials argument: each surface falls back to the mesher's stored
	# material for its slot.
	var built: Mesh = mesher.build_mesh(buffer, [], {})
	_ok(built != null, "cubes build_mesh (no materials) returned a Mesh")
	if built:
		_ok(built.get_surface_count() >= 2, "palette mesh has opaque + transparent surfaces")
		_ok(
			built.surface_get_material(0) == opaque,
			"no-materials build_mesh applies opaque_material to surface 0"
		)
		_ok(
			built.surface_get_material(1) == transparent,
			"no-materials build_mesh applies transparent_material to surface 1"
		)

	# Explicit array wins for the slots it covers; uncovered slots still fall
	# back to the stored materials (upstream VoxelMesher::build_mesh).
	var slot0_override := StandardMaterial3D.new()
	var built_override: Mesh = mesher.build_mesh(buffer, [slot0_override], {})
	_ok(built_override != null, "cubes build_mesh (explicit array) returned a Mesh")
	if built_override:
		_ok(
			built_override.surface_get_material(0) == slot0_override,
			"caller material wins on slot 0"
		)
		_ok(
			built_override.surface_get_material(1) == transparent,
			"uncovered slot 1 still falls back to transparent_material"
		)

	# A null entry keeps its slot: slot 1's material must NOT shift into
	# surface 0 (positional slots).
	var slot1_only := StandardMaterial3D.new()
	var built_null_first: Mesh = mesher.build_mesh(buffer, [null, slot1_only], {})
	_ok(built_null_first != null, "cubes build_mesh ([null, material]) returned a Mesh")
	if built_null_first:
		_ok(
			built_null_first.surface_get_material(0) == opaque,
			"null slot 0 entry does not shift slot 1's material into surface 0"
		)
		_ok(
			built_null_first.surface_get_material(1) == slot1_only,
			"slot 1 keeps the caller's material"
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


## COLOR_MESHER_PALETTE resolves voxel values through the assigned
## VoxelColorPalette resource (surface colors carry the palette entries;
## alpha < 1 sorts into the transparent surface). COLOR_SHADER_PALETTE writes
## the raw index into the red channel and the palette alpha into alpha,
## leaving green/blue at zero. Both palette modes without an assigned palette
## are rejected, mirroring upstream ERR_FAIL_COND_MSG("Palette mode is used
## but no palette was specified").
func cubes_palette_mode_build_mesh() -> void:
	var mesher: RefCounted = ClassDB.instantiate("VoxelMesherCubes")
	mesher.set("palette", _make_cubes_palette())
	var buffer: RefCounted = _make_cubes_palette_buffer()

	mesher.set_color_mode(VoxelMesherCubes.COLOR_MESHER_PALETTE)
	var mesh: Mesh = mesher.build_mesh(buffer, [], {})
	_ok(mesh != null, "COLOR_MESHER_PALETTE build_mesh returned a Mesh")
	if mesh:
		_ok(mesh.get_surface_count() >= 2, "palette mesh has opaque + transparent surfaces")
		var opaque_colors: PackedColorArray = mesh.surface_get_arrays(0)[Mesh.ARRAY_COLOR]
		_ok(
			_colors_contain(opaque_colors, Color(1.0, 0.0, 0.0, 1.0)),
			"COLOR_MESHER_PALETTE surface 0 carries the red palette entry"
		)
		_ok(
			_colors_contain(opaque_colors, Color(0.0, 1.0, 0.0, 1.0)),
			"COLOR_MESHER_PALETTE surface 0 carries the green palette entry"
		)
		var transparent_colors: PackedColorArray = mesh.surface_get_arrays(1)[Mesh.ARRAY_COLOR]
		_ok(
			_colors_contain(transparent_colors, Color(0.0, 0.0, 1.0, 0.5)),
			"COLOR_MESHER_PALETTE surface 1 carries the half-alpha blue entry"
		)

	mesher.set_color_mode(VoxelMesherCubes.COLOR_SHADER_PALETTE)
	var shader_mesh: Mesh = mesher.build_mesh(buffer, [], {})
	_ok(shader_mesh != null, "COLOR_SHADER_PALETTE build_mesh returned a Mesh")
	if shader_mesh:
		var opaque_indices: PackedColorArray = shader_mesh.surface_get_arrays(0)[Mesh.ARRAY_COLOR]
		_ok(
			_colors_contain(opaque_indices, Color(1.0 / 255.0, 0.0, 0.0, 1.0)),
			"COLOR_SHADER_PALETTE writes index 1 in red with the palette alpha"
		)
		_ok(
			_colors_contain(opaque_indices, Color(2.0 / 255.0, 0.0, 0.0, 1.0)),
			"COLOR_SHADER_PALETTE writes index 2 in red with the palette alpha"
		)
		var transparent_indices: PackedColorArray = shader_mesh.surface_get_arrays(1)[
			Mesh.ARRAY_COLOR
		]
		_ok(
			_colors_contain(transparent_indices, Color(3.0 / 255.0, 0.0, 0.0, 0.5)),
			"COLOR_SHADER_PALETTE transparent entry keeps the palette alpha"
		)

	# Palette modes without a palette abort the build (upstream ERR_FAIL
	# parity: no mesh is produced).
	mesher.set("palette", null)
	_ok(
		mesher.build_mesh(buffer, [], {}) == null,
		"COLOR_SHADER_PALETTE without a palette returns null"
	)
	mesher.set_color_mode(VoxelMesherCubes.COLOR_MESHER_PALETTE)
	_ok(
		mesher.build_mesh(buffer, [], {}) == null,
		"COLOR_MESHER_PALETTE without a palette returns null"
	)


## generate_mesh_from_image from a non-uniform image: the centering offset
## -(w, h, 1)/2 (the greedy mesher emits padding-relative coordinates, so the
## offset lands on the exact half extents) and the Y flip (image rows grow
## down, world Y grows up) must both be observable — a single white pixel in
## the image's top-right corner produces white vertices in the world's
## top-right corner of the mesh.
func cubes_generate_mesh_from_image_orientation() -> void:
	var image := Image.create_empty(8, 8, false, Image.FORMAT_RGBA8)
	image.fill(Color(0.0, 0.0, 0.0, 1.0))
	image.set_pixel(7, 0, Color(1.0, 1.0, 1.0, 1.0)) # top-right in image coords
	var mesh: Mesh = VoxelMesherCubes.generate_mesh_from_image(image, 1.0)
	_ok(mesh != null, "orientation image returned a Mesh")
	if mesh == null:
		return

	# The greedy mesher emits padding-relative coordinates, so the 8x8x1
	# plane spans [0, 8) before the -(w, h, 1)/2 centering offset is applied:
	# the mesh lands exactly on [-4, 4) x [-4, 4) x [-0.5, 0.5).
	var aabb := mesh.get_aabb()
	_ok(
		aabb.position.distance_to(Vector3(-4.0, -4.0, -0.5)) < 0.01,
		"centering offset places the plane at -(w, h, 1)/2 (aabb %s)" % str(aabb.position)
	)
	_ok(
		aabb.size.distance_to(Vector3(8.0, 8.0, 1.0)) < 0.01,
		"mesh extent matches the 8x8x1 plane (aabb size %s)" % str(aabb.size)
	)

	# Y flip + centering: every white vertex must sit in the top-right
	# (max X, max Y) unit cell of the mesh, while the black background
	# reaches the bottom-left corner.
	var arrays: Array = mesh.surface_get_arrays(0)
	var vertices: PackedVector3Array = arrays[Mesh.ARRAY_VERTEX]
	var colors: PackedColorArray = arrays[Mesh.ARRAY_COLOR]
	var white_found := false
	var white_in_top_right := true
	var black_in_bottom_left := false
	for i in vertices.size():
		if colors[i].r > 0.9 and colors[i].g > 0.9 and colors[i].b > 0.9:
			white_found = true
			if vertices[i].x < 2.9 or vertices[i].y < 2.9:
				white_in_top_right = false
		elif colors[i].r < 0.1 and colors[i].g < 0.1 and colors[i].b < 0.1:
			if vertices[i].x < -3.9 and vertices[i].y < -3.9:
				black_in_bottom_left = true
	_ok(white_found, "the single white pixel produced white vertices")
	_ok(white_in_top_right, "white vertices are in the top-right corner (Y flip)")
	_ok(black_in_bottom_left, "black background reaches the bottom-left corner")


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
	# Out-of-range indices report Godot's default-constructed opaque black.
	var bad: Color = palette.get_color(300)
	_ok(bad is Color and bad.r == 0.0 and bad.a == 1.0, "get_color(300) returns opaque black")
	# data round-trips through set_data.
	var data: PackedInt32Array = palette.get("data")
	data[3] = 0x00FF00FF
	palette.set("data", data)
	var check: Color = palette.get_color(3)
	_ok(
		check.g > 0.99 and check.r < 0.01 and check.a > 0.99,
		"data property round-trips packed 0xRRGGBBAA colors"
	)


## VoxelRaycastResult exposes the pinned read-only members `distance`,
## `normal`, `position` and `previous_position` under their canonical Godot
## property names, with the upstream defaults (0.0 / zero vectors). The
## getters compose the writable scalar backing fields, and scripted sets on
## the read-only members must not change the values.
func raycast_result_members() -> void:
	var result: RefCounted = ClassDB.instantiate("VoxelRaycastResult")
	_ok(result != null, "VoxelRaycastResult instantiated")
	if result == null:
		return
	# Defaults: upstream VoxelRaycastResult starts at distance 0 and zero
	# vectors, which also means "no hit".
	_ok(absf(result.get_distance()) < 0.0001, "default get_distance() == 0.0")
	_ok(result.get_position() == Vector3i(0, 0, 0), "default get_position() == (0, 0, 0)")
	_ok(result.get_normal() == Vector3(0.0, 0.0, 0.0), "default get_normal() == (0, 0, 0)")
	_ok(
		result.get_previous_position() == Vector3i(0, 0, 0),
		"default get_previous_position() == (0, 0, 0)"
	)
	# The pinned member names must exist as Godot properties.
	_ok(absf(float(result.get("distance"))) < 0.0001, "default `distance` property == 0.0")
	_ok(result.get("position") == Vector3i(0, 0, 0), "default `position` property == (0, 0, 0)")
	_ok(result.get("normal") == Vector3(0.0, 0.0, 0.0), "default `normal` property == (0, 0, 0)")
	_ok(
		result.get("previous_position") == Vector3i(0, 0, 0),
		"default `previous_position` property == (0, 0, 0)"
	)
	_ok(!bool(result.did_hit()), "default result reports no hit")

	# The writable scalar backing fields compose the pinned getters.
	result.hit_x = 3
	result.hit_y = -2
	result.hit_z = 5
	result.prev_x = 2
	result.prev_y = -2
	result.prev_z = 5
	result.normal_x = -1
	_ok(result.get_position() == Vector3i(3, -2, 5), "get_position() composes hit_x/y/z")
	_ok(
		result.get_previous_position() == Vector3i(2, -2, 5),
		"get_previous_position() composes prev_x/y/z"
	)
	_ok(result.get_normal() == Vector3(-1.0, 0.0, 0.0), "get_normal() composes normal_x/y/z")
	_ok(
		result.get("position") == Vector3i(3, -2, 5),
		"`position` property tracks the backing fields"
	)

	# Read-only members: a scripted set must not change the pinned values.
	# (The engine prints an error for the missing setter; the point here is
	# that the member does not silently accept the write.)
	result.set("distance", 5.0)
	_ok(absf(float(result.get("distance"))) < 0.0001, "`distance` stays read-only after set()")
	result.set("position", Vector3i(9, 9, 9))
	_ok(
		result.get("position") == Vector3i(3, -2, 5),
		"`position` stays read-only after set()"
	)
