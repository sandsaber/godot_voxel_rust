@tool
extends Control

## Visual GraphEdit editor for VoxelGeneratorGraph.
## The canvas is the working copy; Apply / Compile rebuild the Rust graph
## via clear_graph + add_node in topological order.

const NODE_KINDS := [
	"InputX", "InputY", "InputZ", "Constant",
	"Add", "Subtract", "Multiply", "Divide",
	"Min", "Max", "Sin", "Cos", "Abs",
	"SdfPlane", "SdfSphere", "SdfBox",
	"SdfUnion", "SdfSubtract", "SdfSmoothUnion",
	"OutputSdf", "Expression",
]

const INPUT_COUNTS := {
	"InputX": 0, "InputY": 0, "InputZ": 0, "Constant": 0,
	"Add": 2, "Subtract": 2, "Multiply": 2, "Divide": 2,
	"Min": 2, "Max": 2, "Sin": 1, "Cos": 1, "Abs": 1,
	"SdfPlane": 2, "SdfSphere": 4, "SdfBox": 3,
	"SdfUnion": 2, "SdfSubtract": 2, "SdfSmoothUnion": 2,
	"OutputSdf": 1, "Expression": 3,
}

const PORT_NAMES := {
	"SdfSphere": ["x", "y", "z", "r"],
	"SdfPlane": ["y", "height"],
	"SdfBox": ["x", "y", "z"],
	"Expression": ["x", "y", "z"],
	"OutputSdf": ["sdf"],
}

var _graph_edit: GraphEdit
var _current_graph: VoxelGeneratorGraph
var _status_label: Label
var _sample_x: SpinBox
var _sample_y: SpinBox
var _sample_z: SpinBox
var _next_visual_id := 0

func _ready() -> void:
	_build_ui()

func _build_ui() -> void:
	var vbox := VBoxContainer.new()
	vbox.set_anchors_preset(PRESET_FULL_RECT)
	add_child(vbox)

	var toolbar := HBoxContainer.new()
	vbox.add_child(toolbar)

	var add_menu := MenuButton.new()
	add_menu.text = "Add Node"
	var popup := add_menu.get_popup()
	for i in NODE_KINDS.size():
		popup.add_item(NODE_KINDS[i], i)
	popup.id_pressed.connect(_on_add_node)
	toolbar.add_child(add_menu)

	var apply_btn := Button.new()
	apply_btn.text = "Apply to Resource"
	apply_btn.pressed.connect(_on_apply)
	toolbar.add_child(apply_btn)

	var compile_btn := Button.new()
	compile_btn.text = "Compile & Sample"
	compile_btn.pressed.connect(_on_compile)
	toolbar.add_child(compile_btn)

	var clear_btn := Button.new()
	clear_btn.text = "Clear"
	clear_btn.pressed.connect(_on_clear)
	toolbar.add_child(clear_btn)

	_sample_x = _make_coord_spin("x", toolbar)
	_sample_y = _make_coord_spin("y", toolbar)
	_sample_z = _make_coord_spin("z", toolbar)

	_status_label = Label.new()
	_status_label.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_status_label.text = "No graph selected"
	toolbar.add_child(_status_label)

	_graph_edit = GraphEdit.new()
	_graph_edit.size_flags_vertical = Control.SIZE_EXPAND_FILL
	_graph_edit.connection_request.connect(_on_connection_request)
	_graph_edit.disconnection_request.connect(_on_disconnection_request)
	vbox.add_child(_graph_edit)


func _make_coord_spin(label_text: String, parent: Control) -> SpinBox:
	var label := Label.new()
	label.text = label_text
	parent.add_child(label)
	var spin := SpinBox.new()
	spin.min_value = -4096
	spin.max_value = 4096
	spin.step = 0.5
	spin.custom_minimum_size.x = 80
	parent.add_child(spin)
	return spin


func edit_graph(graph: VoxelGeneratorGraph) -> void:
	_current_graph = graph
	_refresh_from_graph()


func _refresh_from_graph() -> void:
	_clear_visual()
	if _current_graph == null:
		_status_label.text = "No graph selected"
		return
	var json_text := _current_graph.get_graph_json()
	var parsed = JSON.parse_string(json_text)
	if typeof(parsed) != TYPE_DICTIONARY:
		_status_label.text = "Editing empty graph"
		return
	var nodes: Array = parsed.get("nodes", [])
	var created: Dictionary = {}
	for i in nodes.size():
		var spec: Dictionary = nodes[i]
		var kind := str(spec.get("kind", "Constant"))
		var gn := _create_visual_node(kind, Vector2(40 + (i % 5) * 180, 40 + int(i / 5) * 140))
		if spec.has("value") and gn.has_meta("value_spin"):
			(gn.get_meta("value_spin") as SpinBox).value = float(spec["value"])
		if spec.has("expr") and gn.has_meta("expr_edit"):
			(gn.get_meta("expr_edit") as LineEdit).text = str(spec["expr"])
		created[int(spec.get("id", i))] = gn.name
	for spec in nodes:
		var dest_name: StringName = created.get(int(spec.get("id", -1)), StringName())
		if String(dest_name).is_empty():
			continue
		_try_connect_port(created, dest_name, spec, "a", 0)
		_try_connect_port(created, dest_name, spec, "b", 1)
		_try_connect_port(created, dest_name, spec, "c", 2)
		_try_connect_port(created, dest_name, spec, "d", 3)
	_status_label.text = "Editing graph (%d nodes)" % nodes.size()


func _try_connect_port(created: Dictionary, dest_name: StringName, spec: Dictionary, key: String, port: int) -> void:
	if not spec.has(key):
		return
	var src_id := int(spec[key])
	if src_id < 0 or not created.has(src_id):
		return
	_graph_edit.connect_node(created[src_id], 0, dest_name, port)


func _on_add_node(id: int) -> void:
	if _current_graph == null:
		_status_label.text = "No graph selected"
		return
	if id < 0 or id >= NODE_KINDS.size():
		return
	var kind: String = NODE_KINDS[id]
	_create_visual_node(kind, Vector2(80 + randf() * 240, 80 + randf() * 160))
	_status_label.text = "Added %s (apply to write into the resource)" % kind


func _create_visual_node(kind: String, offset: Vector2) -> GraphNode:
	var gn := GraphNode.new()
	gn.title = kind
	gn.name = "n%d" % _next_visual_id
	_next_visual_id += 1
	gn.position_offset = offset
	gn.set_meta("kind", kind)
	var in_count: int = INPUT_COUNTS.get(kind, 1)
	var names: Array = PORT_NAMES.get(kind, [])
	var row_count := maxi(in_count, 1)
	for i in row_count:
		if kind == "Constant" and i == 0:
			var spin := SpinBox.new()
			spin.min_value = -4096
			spin.max_value = 4096
			spin.step = 0.1
			spin.value = 1.0
			gn.add_child(spin)
			gn.set_meta("value_spin", spin)
		elif kind == "Expression" and i == 0:
			var edit := LineEdit.new()
			edit.placeholder_text = "x + y"
			edit.custom_minimum_size.x = 120
			gn.add_child(edit)
			gn.set_meta("expr_edit", edit)
		elif kind == "SdfSmoothUnion" and i == 0:
			var spin := SpinBox.new()
			spin.min_value = 0
			spin.max_value = 64
			spin.step = 0.1
			spin.value = 2.0
			gn.add_child(spin)
			gn.set_meta("value_spin", spin)
		else:
			var label := Label.new()
			if i < names.size():
				label.text = names[i]
			elif in_count == 0:
				label.text = "out"
			else:
				label.text = "in%d" % i
			gn.add_child(label)
		var enable_in := i < in_count
		var enable_out := i == 0 and kind != "OutputSdf"
		gn.set_slot(i, enable_in, 0, Color(0.4, 0.8, 0.5), enable_out, 0, Color(0.9, 0.6, 0.3))
	_graph_edit.add_child(gn)
	return gn


func _on_connection_request(from_node: StringName, from_port: int, to_node: StringName, to_port: int) -> void:
	_graph_edit.connect_node(from_node, from_port, to_node, to_port)


func _on_disconnection_request(from_node: StringName, from_port: int, to_node: StringName, to_port: int) -> void:
	_graph_edit.disconnect_node(from_node, from_port, to_node, to_port)


func _visual_nodes() -> Array[GraphNode]:
	var nodes: Array[GraphNode] = []
	for child in _graph_edit.get_children():
		if child is GraphNode:
			nodes.append(child)
	return nodes


func _apply_to_resource() -> bool:
	if _current_graph == null:
		_status_label.text = "No graph"
		return false
	var visual := _visual_nodes()
	var order := _topo_order(visual)
	if order.is_empty() and not visual.is_empty():
		_status_label.text = "Cycle in connections — cannot apply"
		return false
	_current_graph.clear_graph()
	var rust_ids: Dictionary = {}
	for gn in order:
		var kind := str(gn.get_meta("kind"))
		var ports := [-1, -1, -1, -1]
		for conn in _graph_edit.get_connection_list():
			if conn["to_node"] != gn.name:
				continue
			var src: GraphNode = _graph_edit.get_node_or_null(NodePath(str(conn["from_node"])))
			if src == null:
				continue
			var port := int(conn["to_port"])
			if port >= 0 and port < 4 and rust_ids.has(src.name):
				ports[port] = int(rust_ids[src.name])
		var value := 0.0
		if gn.has_meta("value_spin"):
			value = float((gn.get_meta("value_spin") as SpinBox).value)
		var new_id := -1
		if kind == "Expression":
			var expr := "x"
			if gn.has_meta("expr_edit"):
				expr = (gn.get_meta("expr_edit") as LineEdit).text
			new_id = _current_graph.add_expression_node(expr, ports[0], ports[1], ports[2])
		else:
			new_id = _current_graph.add_node(kind, ports[0], ports[1], ports[2], ports[3], value)
		if new_id < 0:
			_status_label.text = "Failed to add %s" % kind
			return false
		rust_ids[gn.name] = new_id
	return true


func _topo_order(visual: Array[GraphNode]) -> Array[GraphNode]:
	var incoming: Dictionary = {}
	var by_name: Dictionary = {}
	for gn in visual:
		incoming[gn.name] = 0
		by_name[gn.name] = gn
	for conn in _graph_edit.get_connection_list():
		var dest: StringName = conn["to_node"]
		if incoming.has(dest):
			incoming[dest] = int(incoming[dest]) + 1
	var queue: Array[GraphNode] = []
	for gn in visual:
		if int(incoming[gn.name]) == 0:
			queue.append(gn)
	var order: Array[GraphNode] = []
	while not queue.is_empty():
		var node: GraphNode = queue.pop_front()
		order.append(node)
		for conn in _graph_edit.get_connection_list():
			if conn["from_node"] != node.name:
				continue
			var dest: StringName = conn["to_node"]
			if not incoming.has(dest):
				continue
			incoming[dest] = int(incoming[dest]) - 1
			if int(incoming[dest]) == 0:
				queue.append(by_name[dest])
	if order.size() != visual.size():
		return []
	return order


func _on_apply() -> void:
	if _apply_to_resource():
		_status_label.text = "Applied %d nodes" % _current_graph.get_graph_node_count()


func _on_compile() -> void:
	if not _apply_to_resource():
		return
	if not _current_graph.compile_graph():
		_status_label.text = "Compile failed (cycle or dangling port)"
		return
	var sdf := _current_graph.compile_and_sample(_sample_x.value, _sample_y.value, _sample_z.value)
	_status_label.text = "SDF(%.1f, %.1f, %.1f) = %.4f  nodes=%d" % [
		_sample_x.value, _sample_y.value, _sample_z.value, sdf, _current_graph.get_graph_node_count()
	]


func _on_clear() -> void:
	_clear_visual()
	if _current_graph:
		_current_graph.clear_graph()
	_status_label.text = "Cleared"


func _clear_visual() -> void:
	_graph_edit.clear_connections()
	for child in _graph_edit.get_children():
		if child is GraphNode:
			child.queue_free()
