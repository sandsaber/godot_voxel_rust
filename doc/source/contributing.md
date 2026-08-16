# Contributing

## Ground rules

1. **Logic goes in `voxel-core`.** The binding crate (`voxel-gdext`) must stay
   thin: type conversion + delegation. If you catch yourself writing loops or
   math in a `#[func]`, move it into the core and forward to it.
2. **No `unsafe` in production code.** The convention is enforced by review;
   the only `unsafe` allowed is in test helpers.
3. **Never panic across the FFI boundary.** The release profile uses
   `panic = "unwind"` so unit tests can `catch_unwind`, but unwinding through
   the Godot C ABI is undefined behaviour. Validate GDScript-supplied
   indices/coordinates and return an error or a default; use `debug_assert!`
   so misuse is caught in debug builds.
4. Prefer `Result` / `Option` over `unwrap` / `expect` on paths reachable from
   user input.

## Workflow

```sh
cd rust
cargo build --workspace
cargo test --workspace                  # 1494 tests — all must pass
cargo clippy --workspace --all-targets  # must be warning-clean
cargo fmt --check                       # must pass (cargo fmt to fix)
```

The toolchain is pinned in `rust/rust-toolchain.toml`; `rustup` picks it up
automatically.

If you touch the Godot surface, also run the smoke tests (needs `godot` 4.7+
on `PATH`):

```sh
cd rust
./voxel-gdext/smoke_test/run_smoke_test.sh
```

## Adding a Godot class

- Register it under the **canonical upstream name** with
  `#[class(rename = …)]`; keep the `GD` suffix on the Rust struct.
- Only the three builtin-clashing classes keep the `ZN_` prefix.
- Update the [class reference](api/class_reference.md).

## Tests

- Unit tests live in `#[cfg(test)]` modules next to the code; integration and
  parity tests live in `voxel-core/tests/`.
- Tests that pin numeric output should say where the expected value comes
  from (C++ golden, hand-computed, or measured from the implementation).
- Parity goldens must stay C++-generated (`cpp-baseline`); a Rust-regenerated
  golden fails the parity guard on purpose.

## Fuzzing

Seed corpora are committed under `rust/fuzz/seed_corpus/`. Regenerate them
after changing formats:

```sh
cargo run -p voxel-core --example gen_fuzz_seeds
```

Run campaigns on nightly: `cargo +nightly fuzz run <target> <seed_dir>` (see
`rust/fuzz/README.md`).

## Documentation

These pages are built with MkDocs from `doc/source/`:

```sh
pip install -r doc/requirements.txt
mkdocs serve -f doc/mkdocs.yml
```

Keep docs honest about stubs: mark partial features with a status note rather
than documenting aspirational behavior.
