# Integrating the Rust voxel GDExtension into a Godot project

This is a **native GDExtension** (a compiled `.so` / `.dll` / `.dylib`), not pure
GDScript. So yes — you must **compile** the library for your platform, then point
a `.gdextension` file at it. This guide covers desktop (Linux/Windows/macOS),
mobile (Android), and iOS.

> Godot **4.7+** is required (`compatibility_minimum = 4.7`). Verified on
> **Godot 4.7.1**. The binding uses the `godot` crate (godot-rust 0.5.x,
> `api-4-7`).

---

## TL;DR (Linux desktop, 60 seconds)

```sh
cd rust
cargo build -p voxel-gdext --release          # → rust/target/release/libvoxel_gdext.so
cp rust/voxel-gdext/voxel_gdext.gdextension.in ../your_godot_project/voxel_gdext.gdextension
# edit the .gdextension: set linux.x86_64 path to the built .so
```

Then open the Godot project — the editor loads the library on startup, and the
`VoxelTerrain`, `VoxelBuffer`, `VoxelGeneratorWaves`, … classes appear.

---

## Prerequisites

- **Rust toolchain** matching `rust/rust-toolchain.toml` (Rust 1.96.1). `rustup`
  auto-installs it on first `cargo` invocation in `rust/`.
- **Godot 4.7+** to run the project.
- For non-Linux desktop/mobile targets: the platform's C/C++ toolchain (see
  below). This is because `voxel-gdext` depends on **`godot-cpp`** (a C++ library
  the `godot` crate builds from source), and that C++ must be cross-compiled for
  the target.

> **Why a C++ toolchain?** `voxel-core` (the engine) is pure Rust and
> cross-compiles to every target with only `rustc`. But `voxel-gdext` (the
> binding) links `godot-cpp`, which needs a C++ compiler for the target.
> Desktop-native builds use your host compiler; Android/iOS need the NDK/Xcode.

---

## Step 1 — build the library

### Debug vs release

| Profile | Command | Size (Linux x86_64) | When to use |
|---|---|---|---|
| debug | `cargo build -p voxel-gdext` | ~225 MB | Development (fast compile, has debug symbols, asserts on) |
| **release** | `cargo build -p voxel-gdext --release` | ~5.5 MB | **Production / actual use** (LTO=fat, opt-level 3, panic=abort) |

For distribution or real gameplay, **always use `--release`** — debug builds are
huge and slow.

### Linux (x86_64)

```sh
cd rust
cargo build -p voxel-gdext --release
# → rust/target/release/libvoxel_gdext.so
```

No extra setup — uses the host GCC/Clang for `godot-cpp`.

### Windows (x86_64)

Build **on Windows** with the `x86_64-pc-windows-msvc` (or `-gnu`) target:

```powershell
cd rust
cargo build -p voxel-gdext --release
# → rust\target\release\voxel_gdext.dll
```

Requires the **MSVC build tools** (Visual Studio Build Tools / `cl.exe`) for
`godot-cpp`. Cross-compiling *from* Linux is possible by installing the
`x86_64-pc-windows-gnu` target and a mingw-w64 toolchain, but building natively
on Windows is simplest.

### macOS (arm64 / Apple Silicon)

```sh
cd rust
cargo build -p voxel-gdext --release
# → rust/target/release/libvoxel_gdext.dylib
```

For an Intel (x86_64) build, add the `x86_64-apple-darwin` target. For a
universal binary, build both slices and `lipo` them together, then point the
`macos` key (below) at the result. Requires Xcode command-line tools for
`godot-cpp`.

### Android (arm64 device / x86_64 emulator)

This needs the **Android NDK** and the bundled helper script, which works around
a known rustc/NDK LLVM-version skew:

```sh
cd rust
# NDK r29+ required; set ANDROID_NDK_HOME before running.
export ANDROID_NDK_HOME="$ANDROID_SDK/ndk/29.0.14206865"
./scripts/android-build.sh                       # aarch64 .so (device)
./scripts/android-build.sh --target x86_64-linux-android   # x86_64 .so (emulator)
# → rust/target/aarch64-linux-android/release/libvoxel_gdext.so
./scripts/android-build.sh --strip               # strip debug symbols (~3.2 MB)
```

The script forces rust's bundled `lld` (NDK r29's linker can't parse rustc/LLVM-22
objects) and exports `CC`/`CXX` at the NDK clang so `godot-cpp` cross-compiles.

> Loading the `.so` **on a device** still requires a custom Godot Android export
> template compiled with `platform=android` (the stock template does not load
> GDExtensions at runtime). That packaging step is the remaining mobile item.

### iOS (arm64)

Requires **Xcode** (macOS host). Add the target and build:

```sh
rustup target add aarch64-apple-ios
cd rust
cargo build -p voxel-gdext --target aarch64-apple-ios --release
# → rust/target/aarch64-apple-ios/release/libvoxel_gdext.a (static) or .dylib
```

Packaging into an iOS app needs Xcode + a Godot iOS export template that loads
the extension. `voxel-core` itself is verified to cross-compile to
`aarch64-apple-ios` (pure-Rust, no SDK needed); the `godot-cpp` link step needs
Xcode.

---

## Step 2 — create the `.gdextension` file

Copy the template into your Godot project (under `res://`) and rename it:

```sh
cp rust/voxel-gdext/voxel_gdext.gdextension.in  your_project/addons/voxel/voxel_gdext.gdextension
```

Then edit it so the `[libraries]` paths point at your built artifact. Paths are
**relative to the `.gdextension` file** (`res://`-prefixed):

```ini
[configuration]
entry_symbol = "gdext_rust_init"
compatibility_minimum = 4.7
reloadable = true              # hot-reload class changes on Linux/macOS without editor restart

[libraries]
# Point each platform key at the artifact you built for it. Use one or more.
linux.x86_64   = "res://addons/voxel/libvoxel_gdext.so"
windows.x86_64 = "res://addons/voxel/voxel_gdext.dll"
macos          = "res://addons/voxel/libvoxel_gdext.dylib"
android.arm64  = "res://addons/voxel/libvoxel_gdext.android.arm64.so"
ios            = "res://addons/voxel/libvoxel_gdext.ios.dylib"
```

> The `entry_symbol` must be exactly `gdext_rust_init` — it is generated by
> the `#[gdextension]` macro in `voxel-gdext/src/lib.rs`. Don't change it.

You only need the key(s) for the platform(s) you actually ship; Godot ignores the
others. Either copy the built binary into your project (recommended for
distribution) **or** point the path straight at `rust/target/...` (handy during
development).

---

## Step 3 — use it in Godot

1. (Re)open the Godot project. The editor scans for `.gdextension` files on
   startup and loads the library. Watch the **Output** log for:
   ```
   Initialize godot-rust (API v4.7.stable.official, runtime v4.7.1.stable...)
   voxel-gdext: Scene stage initialized (voxel-core v0.1.0)
   ```
2. The classes are now available. Add a `VoxelTerrain` node, give it a
   `VoxelGeneratorWaves` (or other generator) resource, and add a `VoxelViewer`
   child:

```gdscript
var terrain = VoxelTerrain.new()
terrain.set_generator(VoxelGeneratorWaves.new())
add_child(terrain)
var viewer = VoxelViewer.new()
terrain.add_child(viewer)   # viewer MUST be a child of the terrain
```

> **Class names:** classes are registered under **canonical upstream names**
> (`VoxelBuffer`, `VoxelMesherBlocky`, `VoxelTerrain`, …). Three noise/curve
> classes use a `ZN_` prefix to avoid clashing with Godot builtins:
> `ZN_FastNoiseLite`, `ZN_SpotNoise`, `ZN_Curve`.

---

## Verifying it works

### Automated smoke tests (shipped)

The repo includes a runnable Godot project that verifies the binding:

```sh
cd rust
./voxel-gdext/smoke_test/run_smoke_test.sh    # builds .so + runs all 3 checks
```

It builds the library, copies it next to the `.gdextension`, and runs
`api_test.gd` (class registration + `#[func]` surface), `runtime_scene.tscn`
(paging generates 210 mesh blocks), and `smoke_test.tscn`. Requires `godot`
(4.7+) on `PATH`.

### Manual check

If Godot fails to load the extension, common causes:
- **Path wrong** in `.gdextension` — the `res://` path must resolve to an actual
  built file. Check the editor log for "Could not load GDExtension library".
- **`entry_symbol` typo** — must be `gdext_rust_init`.
- **Wrong architecture/profile** — e.g. a debug `.so` where release was expected,
  or an arm64 binary on x86_64. Rebuild for your platform.
- **Missing C++ runtime** (Windows) — ensure the MSVC redistributable matches the
  build.

---

## Distribution checklist

For shipping a game:
1. Build **release** for every target platform you ship.
2. Place each binary under your project (e.g. `addons/voxel/`).
3. Keep only the matching `[libraries]` key(s) per platform.
4. Do **not** commit the binaries to version control — they are build artifacts
   (already covered by `.gitignore`).

---

## See also

- [`AGENTS.md`](../../AGENTS.md) — architecture, crate layout, the
  voxel-core↔voxel-gdext boundary, build/test commands.
- [`voxel-gdext/README.md`](../voxel-gdext/README.md) — binding-specific build
  notes and the class-naming scheme.
- [`voxel-gdext/smoke_test/run_smoke_test.sh`](../voxel-gdext/smoke_test/run_smoke_test.sh)
  — a working reference for the desktop build+load+verify flow.
