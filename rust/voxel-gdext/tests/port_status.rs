use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::{Map, Value};

const PINNED_UPSTREAM_COMMIT: &str = "5828cbeba19050033f550485abc5f8c3586b1bf5";
const TOP_LEVEL_KEYS: [&str; 6] = [
    "schema_version",
    "upstream",
    "status_semantics",
    "status_counts",
    "classes",
    "extension_only_classes",
];
const UPSTREAM_KEYS: [&str; 4] = [
    "repository",
    "commit",
    "class_list_source",
    "public_class_count",
];
const STATUS_KEYS: [&str; 3] = ["complete", "partial", "deferred"];
const CLASS_KEYS: [&str; 7] = [
    "name",
    "status",
    "binding_state",
    "registered_name",
    "evidence",
    "source_paths",
    "behavioral_tests",
];
const HELPER_KEYS: [&str; 3] = ["name", "source_path", "reason"];
const EXPECTED_UPSTREAM_CLASSES: [&str; 73] = [
    "FastNoise2",
    "VoxelAStarGrid3D",
    "VoxelBlockSerializer",
    "VoxelBlockyAttribute",
    "VoxelBlockyAttributeAxis",
    "VoxelBlockyAttributeCustom",
    "VoxelBlockyAttributeDirection",
    "VoxelBlockyAttributeRotation",
    "VoxelBlockyFluid",
    "VoxelBlockyLibrary",
    "VoxelBlockyLibraryBase",
    "VoxelBlockyModel",
    "VoxelBlockyModelCube",
    "VoxelBlockyModelEmpty",
    "VoxelBlockyModelFluid",
    "VoxelBlockyModelMesh",
    "VoxelBlockyType",
    "VoxelBlockyTypeLibrary",
    "VoxelBoxMover",
    "VoxelBuffer",
    "VoxelColorPalette",
    "VoxelDataBlockEnterInfo",
    "VoxelEngine",
    "VoxelFormat",
    "VoxelGenerator",
    "VoxelGeneratorFlat",
    "VoxelGeneratorGraph",
    "VoxelGeneratorHeightmap",
    "VoxelGeneratorImage",
    "VoxelGeneratorMultipassCB",
    "VoxelGeneratorNoise",
    "VoxelGeneratorNoise2D",
    "VoxelGeneratorScript",
    "VoxelGeneratorWaves",
    "VoxelGraphFunction",
    "VoxelInstanceComponent",
    "VoxelInstanceGenerator",
    "VoxelInstanceLibrary",
    "VoxelInstanceLibraryItem",
    "VoxelInstanceLibraryMultiMeshItem",
    "VoxelInstanceLibrarySceneItem",
    "VoxelInstancer",
    "VoxelInstancerRigidBody",
    "VoxelLodTerrain",
    "VoxelMeshSDF",
    "VoxelMesher",
    "VoxelMesherBlocky",
    "VoxelMesherCubes",
    "VoxelMesherTransvoxel",
    "VoxelModifier",
    "VoxelModifierMesh",
    "VoxelModifierSphere",
    "VoxelNode",
    "VoxelRaycastResult",
    "VoxelSaveCompletionTracker",
    "VoxelStream",
    "VoxelStreamMemory",
    "VoxelStreamRegionFiles",
    "VoxelStreamSQLite",
    "VoxelStreamScript",
    "VoxelTerrain",
    "VoxelTerrainMultiplayerSynchronizer",
    "VoxelTool",
    "VoxelToolBuffer",
    "VoxelToolLodTerrain",
    "VoxelToolMultipassGenerator",
    "VoxelToolTerrain",
    "VoxelViewer",
    "VoxelVoxLoader",
    "ZN_FastNoiseLite",
    "ZN_FastNoiseLiteGradient",
    "ZN_SpotNoise",
    "ZN_ThreadedTask",
];
fn manifest_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("api")
        .join("port_status.json")
}

fn repository_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("voxel-gdext must be nested under rust/")
        .to_path_buf()
}

fn load_manifest() -> Value {
    let source = std::fs::read_to_string(manifest_path()).expect("port_status.json must exist");
    serde_json::from_str(&source).expect("port_status.json must be strict JSON")
}

fn object(value: &Value) -> &Map<String, Value> {
    value
        .as_object()
        .unwrap_or_else(|| panic!("expected JSON object, got {value:?}"))
}

fn object_mut(value: &mut Value) -> &mut Map<String, Value> {
    value
        .as_object_mut()
        .unwrap_or_else(|| panic!("expected JSON object"))
}

fn array(value: &Value) -> &[Value] {
    value
        .as_array()
        .unwrap_or_else(|| panic!("expected JSON array, got {value:?}"))
}

fn string(value: &Value) -> &str {
    value
        .as_str()
        .unwrap_or_else(|| panic!("expected JSON string, got {value:?}"))
}

fn field<'a>(value: &'a Value, name: &str) -> &'a Value {
    object(value)
        .get(name)
        .unwrap_or_else(|| panic!("missing required field {name:?}"))
}

fn require_exact_keys(value: &Value, expected: &[&str], context: &str) -> Result<(), String> {
    let actual: BTreeSet<_> = object(value).keys().map(String::as_str).collect();
    let expected: BTreeSet<_> = expected.iter().copied().collect();
    if actual == expected {
        Ok(())
    } else {
        Err(format!(
            "{context} keys differ: expected {expected:?}, got {actual:?}"
        ))
    }
}

fn validate_exact_schema(manifest: &Value) -> Result<(), String> {
    require_exact_keys(manifest, &TOP_LEVEL_KEYS, "top level")?;
    require_exact_keys(field(manifest, "upstream"), &UPSTREAM_KEYS, "upstream")?;
    require_exact_keys(
        field(manifest, "status_semantics"),
        &STATUS_KEYS,
        "status_semantics",
    )?;
    require_exact_keys(
        field(manifest, "status_counts"),
        &STATUS_KEYS,
        "status_counts",
    )?;
    for class in array(field(manifest, "classes")) {
        let name = object(class)
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or("<unknown>");
        require_exact_keys(class, &CLASS_KEYS, &format!("class {name}"))?;
    }
    for helper in array(field(manifest, "extension_only_classes")) {
        let name = object(helper)
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or("<unknown>");
        require_exact_keys(helper, &HELPER_KEYS, &format!("helper {name}"))?;
    }
    Ok(())
}

fn rust_source_files() -> Vec<PathBuf> {
    let mut files: Vec<_> = std::fs::read_dir(Path::new(env!("CARGO_MANIFEST_DIR")).join("src"))
        .expect("voxel-gdext/src must be readable")
        .map(|entry| {
            entry
                .expect("source directory entry must be readable")
                .path()
        })
        .filter(|path| path.extension().is_some_and(|extension| extension == "rs"))
        .collect();
    files.sort();
    files
}

fn repository_relative(path: &Path) -> String {
    path.strip_prefix(repository_root())
        .expect("source must be inside repository")
        .to_string_lossy()
        .replace('\\', "/")
}

fn scan_registered_classes() -> Result<BTreeMap<String, String>, String> {
    let mut registrations = BTreeMap::new();
    for path in rust_source_files() {
        let source = std::fs::read_to_string(&path)
            .map_err(|error| format!("could not read {}: {error}", path.display()))?;
        let mut remainder = source.as_str();
        while let Some(attribute_offset) = remainder.find("#[class(") {
            remainder = &remainder[attribute_offset + "#[class(".len()..];
            let attribute_end = remainder
                .find(")]")
                .ok_or_else(|| format!("unterminated #[class] attribute in {}", path.display()))?;
            let attribute = &remainder[..attribute_end];
            if attribute.contains('\n') {
                return Err(format!(
                    "multiline #[class] attribute is not covered by the port-status scanner in {}",
                    path.display()
                ));
            }
            remainder = &remainder[attribute_end + 2..];
            let struct_offset = remainder.find("pub struct ").ok_or_else(|| {
                format!(
                    "#[class] is not followed by pub struct in {}",
                    path.display()
                )
            })?;
            if !remainder[..struct_offset].trim().is_empty() {
                return Err(format!(
                    "#[class] is not immediately followed by pub struct in {}",
                    path.display()
                ));
            }
            let struct_tail = &remainder[struct_offset + "pub struct ".len()..];
            let struct_name: String = struct_tail
                .chars()
                .take_while(|character| character.is_ascii_alphanumeric() || *character == '_')
                .collect();
            if struct_name.is_empty() {
                return Err(format!(
                    "missing struct name after #[class] in {}",
                    path.display()
                ));
            }
            let registered_name = attribute
                .split(',')
                .find_map(|part| part.trim().strip_prefix("rename = "))
                .map(str::to_string)
                .unwrap_or(struct_name);
            let source_path = repository_relative(&path);
            if let Some(previous) = registrations.insert(registered_name.clone(), source_path) {
                return Err(format!(
                    "duplicate registered class {registered_name}: also declared in {previous}"
                ));
            }
            remainder = struct_tail;
        }
    }
    Ok(registrations)
}

fn manifest_registered_classes(manifest: &Value) -> Result<BTreeMap<String, String>, String> {
    let mut registrations = BTreeMap::new();
    for class in array(field(manifest, "classes")) {
        let public_name = string(field(class, "name"));
        let binding_state = string(field(class, "binding_state"));
        match binding_state {
            "registered" => {
                let registered_name =
                    field(class, "registered_name").as_str().ok_or_else(|| {
                        format!("{public_name} is registered without registered_name")
                    })?;
                if registered_name != public_name {
                    return Err(format!(
                        "canonical class {public_name} registers unexpected name {registered_name}"
                    ));
                }
                let local_sources: Vec<_> = array(field(class, "source_paths"))
                    .iter()
                    .map(string)
                    .filter(|source| source.starts_with("rust/voxel-gdext/src/"))
                    .collect();
                if local_sources.len() != 1 {
                    return Err(format!(
                        "registered class {public_name} must cite exactly one binding source"
                    ));
                }
                registrations.insert(registered_name.to_string(), local_sources[0].to_string());
            }
            "unregistered" => {
                if !field(class, "registered_name").is_null() {
                    return Err(format!(
                        "unregistered class {public_name} must have null registered_name"
                    ));
                }
            }
            other => return Err(format!("{public_name} has invalid binding_state {other:?}")),
        }
    }
    for helper in array(field(manifest, "extension_only_classes")) {
        registrations.insert(
            string(field(helper, "name")).to_string(),
            string(field(helper, "source_path")).to_string(),
        );
    }
    Ok(registrations)
}

fn invoked_smoke_files() -> Result<BTreeSet<String>, String> {
    let smoke_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("smoke_test");
    let driver_path = smoke_root.join("run_smoke_test.sh");
    let driver = std::fs::read_to_string(&driver_path)
        .map_err(|error| format!("could not read smoke driver: {error}"))?;
    let mut candidates = Vec::new();
    for entry in std::fs::read_dir(&smoke_root)
        .map_err(|error| format!("could not read smoke directory: {error}"))?
    {
        let path = entry
            .map_err(|error| format!("could not read smoke entry: {error}"))?
            .path();
        if path.is_file() {
            candidates.push(path);
        }
    }
    let mut invoked = BTreeSet::new();
    for path in &candidates {
        let file_name = path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| format!("non-UTF8 smoke path {}", path.display()))?;
        if driver.contains(file_name)
            && matches!(
                path.extension().and_then(|x| x.to_str()),
                Some("gd" | "tscn")
            )
        {
            invoked.insert(repository_relative(path));
        }
    }
    let invoked_scenes: Vec<_> = invoked
        .iter()
        .filter(|path| path.ends_with(".tscn"))
        .cloned()
        .collect();
    for scene in invoked_scenes {
        let scene_source = std::fs::read_to_string(repository_root().join(&scene))
            .map_err(|error| format!("could not read {scene}: {error}"))?;
        for path in &candidates {
            if path.extension().and_then(|extension| extension.to_str()) != Some("gd") {
                continue;
            }
            let file_name = path
                .file_name()
                .and_then(|name| name.to_str())
                .ok_or_else(|| format!("non-UTF8 smoke script path {}", path.display()))?;
            if scene_source.contains(file_name) {
                invoked.insert(repository_relative(path));
            }
        }
    }
    Ok(invoked)
}

fn validate_behavioral_reference(reference: &str) -> Result<(), String> {
    let (path, anchor) = reference
        .split_once('#')
        .ok_or_else(|| format!("behavioral test reference must contain #anchor: {reference}"))?;
    if path.is_empty() || anchor.is_empty() || anchor.contains('#') {
        return Err(format!("invalid behavioral test reference {reference:?}"));
    }
    let absolute = repository_root().join(path);
    let source = std::fs::read_to_string(&absolute)
        .map_err(|_| format!("behavioral test path does not exist: {path}"))?;
    match absolute
        .extension()
        .and_then(|extension| extension.to_str())
    {
        Some("rs") => {
            if !path.starts_with("rust/voxel-gdext/tests/")
                && !path.starts_with("rust/voxel-gdext/src/")
            {
                return Err(format!(
                    "Rust behavioral test is outside voxel-gdext: {path}"
                ));
            }
            let needle = format!("fn {anchor}");
            let offset = source.find(&needle).ok_or_else(|| {
                format!("behavioral test anchor {anchor:?} is absent from {path}")
            })?;
            let prefix = &source[..offset];
            let window_start = prefix
                .char_indices()
                .rev()
                .nth(256)
                .map_or(0, |(index, _)| index);
            let prefix = &prefix[window_start..];
            if !prefix.contains("#[test]") {
                return Err(format!("Rust anchor {anchor:?} is not a #[test] in {path}"));
            }
        }
        Some("gd" | "tscn") => {
            if !invoked_smoke_files()?.contains(path) {
                return Err(format!(
                    "smoke test is not invoked by run_smoke_test.sh: {path}"
                ));
            }
            if absolute
                .extension()
                .and_then(|extension| extension.to_str())
                == Some("gd")
            {
                let declares_anchor = source.lines().any(|line| {
                    let line = line.trim_start();
                    let declaration = line
                        .strip_prefix("func ")
                        .or_else(|| line.strip_prefix("static func "));
                    declaration.is_some_and(|declaration| {
                        declaration
                            .split_once('(')
                            .is_some_and(|(name, _)| name.trim() == anchor)
                    })
                });
                if !declares_anchor {
                    return Err(format!(
                        "GDScript function anchor {anchor:?} is absent from {path}"
                    ));
                }
            } else if !source.lines().any(|line| line.trim() == anchor) {
                return Err(format!(
                    "scene anchor line {anchor:?} is absent from {path}"
                ));
            }
        }
        _ => return Err(format!("unsupported behavioral test target: {path}")),
    }
    Ok(())
}

fn validate_complete_behavioral_tests(manifest: &Value) -> Result<(), String> {
    for class in array(field(manifest, "classes")) {
        if string(field(class, "status")) != "complete" {
            continue;
        }
        let tests = array(field(class, "behavioral_tests"));
        if tests.is_empty() {
            return Err(format!(
                "{} is complete but has no executable behavioral test",
                string(field(class, "name"))
            ));
        }
        for test in tests {
            let reference = string(test);
            validate_behavioral_reference(reference)?;
        }
    }
    Ok(())
}

#[test]
fn manifest_is_strict_json_with_the_pinned_schema() {
    let manifest = load_manifest();
    validate_exact_schema(&manifest).expect("manifest schema must use exact keys");
    assert_eq!(field(&manifest, "schema_version").as_u64(), Some(1));

    let upstream = field(&manifest, "upstream");
    assert_eq!(
        string(field(upstream, "repository")),
        "https://github.com/Zylann/godot_voxel"
    );
    assert_eq!(string(field(upstream, "commit")), PINNED_UPSTREAM_COMMIT);
    assert_eq!(field(upstream, "public_class_count").as_u64(), Some(73));
    assert_eq!(
        string(field(upstream, "class_list_source")),
        "doc/classes/*.xml"
    );
}

#[test]
fn exact_schema_rejects_unknown_fields() {
    let mut manifest = load_manifest();
    object_mut(&mut manifest).insert("future_guess".to_string(), Value::Bool(true));
    let error = validate_exact_schema(&manifest).expect_err("unknown field must be rejected");
    assert!(error.contains("top level keys differ"), "{error}");

    let mut manifest = load_manifest();
    object_mut(
        object_mut(&mut manifest)
            .get_mut("upstream")
            .expect("upstream must exist"),
    )
    .remove("commit");
    let error = validate_exact_schema(&manifest).expect_err("missing field must be rejected");
    assert!(error.contains("upstream keys differ"), "{error}");
}

#[test]
fn manifest_has_each_pinned_upstream_class_exactly_once() {
    let manifest = load_manifest();
    let classes = array(field(&manifest, "classes"));
    let actual: BTreeSet<_> = classes
        .iter()
        .map(|class| string(field(class, "name")))
        .collect();
    let expected: BTreeSet<_> = EXPECTED_UPSTREAM_CLASSES.into_iter().collect();

    assert_eq!(classes.len(), expected.len(), "class names must be unique");
    assert_eq!(
        actual, expected,
        "manifest must match the pinned XML class set"
    );
}

#[test]
fn available_pinned_git_object_matches_the_checked_in_class_snapshot() {
    let expected: BTreeSet<_> = EXPECTED_UPSTREAM_CLASSES
        .into_iter()
        .map(str::to_string)
        .collect();
    let output = Command::new("git")
        .current_dir(repository_root())
        .args([
            "ls-tree",
            "-r",
            "--name-only",
            PINNED_UPSTREAM_COMMIT,
            "--",
            "doc/classes",
        ])
        .output();
    let Ok(output) = output else {
        eprintln!("git unavailable; checked-in 73-class snapshot remains authoritative");
        return;
    };
    if !output.status.success() {
        eprintln!(
            "pinned git object unavailable; checked-in 73-class snapshot remains authoritative"
        );
        return;
    }
    let stdout = String::from_utf8(output.stdout).expect("git paths must be UTF-8");
    let actual: BTreeSet<_> = stdout
        .lines()
        .filter_map(|path| {
            path.strip_prefix("doc/classes/")
                .and_then(|path| path.strip_suffix(".xml"))
                .map(str::to_string)
        })
        .collect();
    assert_eq!(actual, expected, "pinned upstream XML class set drifted");
}

#[test]
fn every_class_has_an_allowed_status_and_traceable_evidence() {
    let manifest = load_manifest();
    let mut actual_counts = BTreeMap::new();
    for class in array(field(&manifest, "classes")) {
        let name = string(field(class, "name"));
        let status = string(field(class, "status"));
        assert!(
            matches!(status, "complete" | "partial" | "deferred"),
            "{name} has unsupported status {status:?}"
        );
        assert!(
            !string(field(class, "evidence")).trim().is_empty(),
            "{name} must explain its status"
        );
        let sources = array(field(class, "source_paths"));
        assert!(!sources.is_empty(), "{name} must cite at least one source");
        let upstream_doc = format!("upstream:doc/classes/{name}.xml");
        assert!(
            sources.iter().any(|source| string(source) == upstream_doc),
            "{name} must cite its pinned upstream class document"
        );
        let local_sources: Vec<_> = sources
            .iter()
            .map(string)
            .filter(|source| source.starts_with("rust/voxel-gdext/"))
            .collect();
        if status != "deferred" {
            assert!(
                !local_sources.is_empty(),
                "{name} cannot be {status} without a local implementation source"
            );
        }
        for source in local_sources {
            assert!(repository_root().join(source).is_file(), "missing {source}");
        }
        assert!(field(class, "behavioral_tests").is_array());
        *actual_counts.entry(status).or_insert(0_u64) += 1;
    }

    let summary = field(&manifest, "status_counts");
    for status in ["complete", "partial", "deferred"] {
        assert_eq!(
            field(summary, status).as_u64(),
            Some(*actual_counts.get(status).unwrap_or(&0)),
            "status count for {status} must be derived from the class entries"
        );
    }
    validate_complete_behavioral_tests(&manifest)
        .expect("every complete claim must cite an existing behavioral test");
}

#[test]
fn intentionally_omitted_subsystems_remain_deferred() {
    let manifest = load_manifest();
    let statuses: BTreeMap<_, _> = array(field(&manifest, "classes"))
        .iter()
        .map(|class| (string(field(class, "name")), string(field(class, "status"))))
        .collect();

    for name in [
        "VoxelGeneratorMultipassCB",
        "VoxelToolMultipassGenerator",
        "VoxelStreamSQLite",
        "VoxelInstancerRigidBody",
    ] {
        assert_eq!(statuses.get(name), Some(&"deferred"), "{name}");
    }
}

#[test]
fn complete_status_requires_an_existing_behavioral_test_reference() {
    let mut manifest = load_manifest();
    let classes = object_mut(&mut manifest)
        .get_mut("classes")
        .and_then(Value::as_array_mut)
        .expect("classes must be an array");
    let candidate = classes.first_mut().expect("manifest must not be empty");
    object_mut(candidate).insert("status".to_string(), Value::String("complete".to_string()));
    object_mut(candidate).insert("behavioral_tests".to_string(), Value::Array(Vec::new()));

    let error = validate_complete_behavioral_tests(&manifest)
        .expect_err("an untested complete claim must be rejected");
    assert!(error.contains("no executable behavioral test"), "{error}");
    {
        let classes = object_mut(&mut manifest)
            .get_mut("classes")
            .and_then(Value::as_array_mut)
            .expect("classes must be an array");
        object_mut(&mut classes[0])
            .insert("status".to_string(), Value::String("partial".to_string()));
    }

    let candidate = array(field(&manifest, "classes"))
        .iter()
        .position(|class| string(field(class, "name")) == "VoxelBuffer")
        .expect("VoxelBuffer status must exist");
    {
        let classes = object_mut(&mut manifest)
            .get_mut("classes")
            .and_then(Value::as_array_mut)
            .expect("classes must be an array");
        object_mut(&mut classes[candidate])
            .insert("status".to_string(), Value::String("complete".to_string()));
        object_mut(&mut classes[candidate]).insert(
            "behavioral_tests".to_string(),
            Value::Array(vec![Value::String(
                "rust/voxel-gdext/README.md#Status".to_string(),
            )]),
        );
    }
    let error = validate_complete_behavioral_tests(&manifest)
        .expect_err("README must not qualify as an executable behavioral test");
    assert!(
        error.contains("unsupported behavioral test target"),
        "{error}"
    );

    {
        let classes = object_mut(&mut manifest)
            .get_mut("classes")
            .and_then(Value::as_array_mut)
            .expect("classes must be an array");
        object_mut(&mut classes[candidate]).insert(
            "behavioral_tests".to_string(),
            Value::Array(vec![Value::String(
                "rust/voxel-gdext/smoke_test/api_test.gd#missing_anchor".to_string(),
            )]),
        );
    }
    let error = validate_complete_behavioral_tests(&manifest)
        .expect_err("missing anchor must not qualify as behavioral evidence");
    assert!(error.contains("anchor"), "{error}");

    {
        let classes = object_mut(&mut manifest)
            .get_mut("classes")
            .and_then(Value::as_array_mut)
            .expect("classes must be an array");
        object_mut(&mut classes[candidate]).insert(
            "behavioral_tests".to_string(),
            Value::Array(vec![Value::String(
                "rust/voxel-gdext/smoke_test/runtime_correctness.gd#call_deferred".to_string(),
            )]),
        );
    }
    let error = validate_complete_behavioral_tests(&manifest)
        .expect_err("a raw GDScript token must not qualify as a function anchor");
    assert!(error.contains("GDScript function anchor"), "{error}");

    {
        let classes = object_mut(&mut manifest)
            .get_mut("classes")
            .and_then(Value::as_array_mut)
            .expect("classes must be an array");
        object_mut(&mut classes[candidate]).insert(
            "behavioral_tests".to_string(),
            Value::Array(vec![Value::String(
                "rust/voxel-gdext/smoke_test/api_test.gd#_init".to_string(),
            )]),
        );
    }
    validate_complete_behavioral_tests(&manifest)
        .expect("driver-invoked smoke test with a real anchor must qualify");

    {
        let classes = object_mut(&mut manifest)
            .get_mut("classes")
            .and_then(Value::as_array_mut)
            .expect("classes must be an array");
        object_mut(&mut classes[candidate]).insert(
            "behavioral_tests".to_string(),
            Value::Array(vec![Value::String(
                "rust/voxel-gdext/tests/port_status.rs#exact_schema_rejects_unknown_fields"
                    .to_string(),
            )]),
        );
    }
    validate_complete_behavioral_tests(&manifest)
        .expect("Cargo integration test target with a real #[test] anchor must qualify");
}

#[test]
fn manifest_registration_snapshot_matches_current_godot_classes() {
    let manifest = load_manifest();
    let expected = manifest_registered_classes(&manifest)
        .expect("manifest registration metadata must be valid");
    let actual = scan_registered_classes().expect("Rust registrations must be scanable");
    assert_eq!(actual, expected, "registered Godot class set drifted");
}

#[test]
fn extension_only_registered_classes_are_separate_from_upstream_api() {
    let manifest = load_manifest();
    let upstream_names: BTreeSet<_> = array(field(&manifest, "classes"))
        .iter()
        .map(|class| string(field(class, "name")))
        .collect();
    let helpers = array(field(&manifest, "extension_only_classes"));
    let helper_names: BTreeSet<_> = helpers
        .iter()
        .map(|helper| string(field(helper, "name")))
        .collect();
    assert_eq!(
        helpers.len(),
        helper_names.len(),
        "helper names must be unique"
    );
    assert!(
        upstream_names.is_disjoint(&helper_names),
        "extension-only helpers must not inflate upstream API coverage"
    );
    for helper in helpers {
        let source = string(field(helper, "source_path"));
        assert!(
            source.starts_with("rust/voxel-gdext/src/"),
            "helper source must be local: {source}"
        );
        assert!(repository_root().join(source).is_file(), "missing {source}");
        assert!(!string(field(helper, "reason")).trim().is_empty());
    }
}
