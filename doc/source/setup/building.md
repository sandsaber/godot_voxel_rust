# Building the extension

The engine ships as a **native GDExtension**: a compiled shared library
(`.so` / `.dll` / `.dylib`). You compile it with Cargo, then point a
`.gdextension` file at it (next page).

!!! note
    The full, platform-by-platform version of this guide — including Android
    NDK and iOS details — lives in the repository at
    [`rust/docs/INTEGRATION.md`](https://github.com/sandsaber/godot_voxel_rust/blob/master/rust/docs/INTEGRATION.md).

## Prerequisites

- **Rust toolchain** — the workspace pins Rust 1.96.1 in
  `rust/rust-toolchain.toml`; `rustup` installs it automatically on first
  `cargo` invocation.
- A **C++ compiler** for the target platform (GCC/Clang on Linux, MSVC on
  Windows, Xcode CLT on macOS). The engine core is pure Rust, but the Godot
  binding links `godot-cpp`, which is C++.
- **Godot 4.7+** to run the result (the binding is built against API 4.7).

## Desktop builds

All commands run from the `rust/` directory.

```sh
cargo build -p voxel-gdext --release
```

| Platform | Artifact |
|---|---|
| Linux x86_64 | `rust/target/release/libvoxel_gdext.so` |
| Windows x86_64 | `rust\target\release\voxel_gdext.dll` |
| macOS arm64 | `rust/target/release/libvoxel_gdext.dylib` |

Always use `--release` for real use: the release profile enables
`opt-level = 3`, fat LTO and `panic = "unwind"` (≈5 MB). Debug builds are
≈200 MB and much slower.

## Mobile builds

- **Android**: requires the NDK (r29+). Use the bundled helper, which works
  around a rustc↔NDK LLVM-version skew:

  ```sh
  export ANDROID_NDK_HOME="$ANDROID_SDK/ndk/29.0.14206865"
  ./scripts/android-build.sh                         # aarch64 device .so
  ./scripts/android-build.sh --target x86_64-linux-android   # emulator
  ```

- **iOS**: `rustup target add aarch64-apple-ios`, then
  `cargo build -p voxel-gdext --target aarch64-apple-ios --release`
  (Xcode required for the `godot-cpp` link step).

The engine core itself (`voxel-core`) is pure Rust and cross-compiles to
Android/iOS/WASM targets with `rustc` alone — no SDK needed.

## Verifying the build

```sh
cargo test --workspace        # 1494 tests (unit + parity + integration + stress)
cargo clippy --workspace --all-targets
cargo fmt --check
```

To verify the extension actually loads in Godot, run the shipped smoke tests
(requires `godot` 4.7+ on `PATH`):

```sh
cd rust
./voxel-gdext/smoke_test/run_smoke_test.sh
```

This builds the library, then runs a real Godot project: class-registration
checks, and a terrain paging scene that generates mesh blocks over real
frames.
