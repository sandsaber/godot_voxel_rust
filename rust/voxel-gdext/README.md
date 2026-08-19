# `voxel-gdext` — Godot 4 GDExtension bindings

The thin binding layer: the only crate in the workspace that depends on the
[`godot`](https://godot-rust.github.io) crate (gdext) and exposes Rust symbols
to GDScript via `#[func]`/`#[base]`/`#[signal]`. All engine-agnostic logic lives
in [`voxel-core`](../voxel-core); this crate wraps it into Godot classes.

## Status

The **fixed-LOD and Variable LOD runtimes are functional**: the extension loads
in Godot 4.7+ and the paging/generation/meshing, editing, remeshing, RegionFiles
persistence, transition masks, topology events, and collision_surface paths run
end-to-end. All 73 canonical upstream classes have their pinned API surface
(methods, properties, constants, signals) implemented (stub or delegated);
canonical API compatibility is still partial because behavioral completeness per
class is ongoing.

At upstream commit `5828cbeba19050033f550485abc5f8c3586b1bf5`, the documented
API contains 73 public classes. The machine-readable
[`api/port_status.json`](api/port_status.json) manifest, enforced by
[`tests/port_status.rs`](tests/port_status.rs), tracks each class's status.
The extension also registers Rust-port/editor helpers which do not count as
upstream API coverage. Intentional deferrals include SQLite streams, multipass
generation and rigid-body physics integration.

### Class names

Documented classes that are exposed use **canonical names** matching the
upstream godot_voxel C++ module (`VoxelBuffer`, `VoxelMesherBlocky`,
`VoxelTerrain`, …), via the `#[class(rename = ...)]` attribute — the Rust
structs may keep a `GD` suffix (`VoxelBufferGD`) internally while ClassDB and
GDScript see `VoxelBuffer`.

Canonical upstream exceptions use the `ZN_` prefix to avoid Godot builtin
clashes: `ZN_FastNoiseLite` and `ZN_SpotNoise`. `ZN_Curve` is a Rust-port helper
listed separately in the status manifest and does not count toward the pinned
upstream class set.

## Build

```sh
cd rust
cargo build -p voxel-gdext              # debug .so/.dylib/.dll
cargo build -p voxel-gdext --release    # optimized
```

This is a `cdylib`; the artifact is `target/<profile>/libvoxel_gdext.so` (Linux),
`libvoxel_gdext.dylib` (macOS), or `voxel_gdext.dll` (Windows).

## Load in Godot 4.7

1. Copy `voxel_gdext.gdextension.in` → `voxel_gdext.gdextension` somewhere under
   `res://` (e.g. the crate dir). The template defaults to debug desktop
   artifacts (`cargo build -p voxel-gdext`) and release Android artifacts
   (`./scripts/android-build.sh`); switch the paths if you load a release desktop
   build.
2. (Re)open the Godot project — the editor scans for `.gdextension` files and
   loads the library on startup.
3. The classes are now available in GDScript:

```gdscript
var terrain = VoxelTerrain.new()
terrain.set_generator(VoxelGeneratorWaves.new())
add_child(terrain)
var viewer = VoxelViewer.new()
terrain.add_child(viewer)   # viewer must be a child of the terrain
```

### Verified

Tested headless against Godot 4.7.1.stable on Linux x86_64 (2026-07-30).
The `smoke_test/` Godot project ships eight runnable checks plus a driver script.

**Reproducing on a clean checkout** — the compiled library is a git-ignored build
artifact, so build it first. The driver does everything:

```sh
cd rust
./voxel-gdext/smoke_test/run_smoke_test.sh          # debug build + all 8 checks
./voxel-gdext/smoke_test/run_smoke_test.sh --release
```

It (1) `cargo build`s `voxel-gdext`, (2) copies the `.so`/`.dylib` next to the
`.gdextension`, then runs:

- **`api_test.tscn`** — class
  registration, `VoxelTerrain` instantiate, `set_generator`, property round-trip,
  `VoxelBuffer` voxel read/write. The SDF edit is asserted **honestly**: before
  `_ready()` runs it reports the not-ready state (set=false, sdf=0.0) rather than
  a false positive.
- **`runtime_scene.tscn`** — builds a `VoxelTerrain` + `VoxelGeneratorWaves` +
  `VoxelViewer` in the tree, waits for a nonzero mesh upload and verifies the
  live SDF edition path.
- **`smoke_test.tscn`** — loads a scene containing a canonical `VoxelTerrain`
  node and verifies that the extension-backed scene starts successfully.
- **`runtime_correctness.tscn`** — verifies mesh replacement after edits,
  viewer-driven unload, invalid-input safety and RegionFiles persistence using
  public Godot-visible behavior.
- **`variable_lod_3.tscn`** — 3-LOD `VoxelLodTerrain` paging, split/join and
  negative coordinates.
- **`blocky_terrain.tscn`** — Type-channel Flat + baked blocky library
  rendering.
- **`instancer_streaming.tscn`** — `VoxelInstancer` streaming under a noise
  Type terrain: instance blocks follow mesh-block paging, spawn real nodes,
  and free them when the viewer leaves.
- **`mesher_api.tscn`** — canonical mesher API surface: base padding
  defaults, Transvoxel `build_mesh`/`build_transition_mesh`, Cubes
  materials/palette modes/`generate_mesh_from_image` (orientation pinned),
  Blocky `build_mesh` with a library, palette properties, and
  `VoxelRaycastResult` member composition.

## Android

`voxel-gdext` cross-compiles for Android arm64 (device) and x86_64 (emulator)
with the NDK. The build script works around the rustc↔NDK LLVM skew (NDK r29's
`lld` can't parse rustc/LLVM-22 objects; the script forces rust's `lld` via
`-fuse-ld=lld`) and exports `CC`/`CXX` so the `godot` crate builds `godot-cpp`
for the same target.

```sh
cd rust
./scripts/android-build.sh                                  # aarch64 .so (device)
./scripts/android-build.sh --target x86_64-linux-android    # x86_64 .so (emulator)
./scripts/android-build.sh --strip                          # strip debug symbols
```

Verified with NDK r29 (14206865), `ANDROID_API=21`, Godot `api-4-7`: produces
`target/<triple>/release/libvoxel_gdext.so` exporting `gdext_rust_init`
(~3.2 MB unstripped). The `.gdextension.in` carries matching `android.arm64`
and `android.x86_64` entries.

Loading the `.so` inside a Godot Android export template still requires a
custom template compiled with `platform=android` (the stock template does not
load GDExtensions at runtime on device) — that packaging step is tracked as
the remaining Phase 2 mobile-half item (needs an SDK + device/emulator).
