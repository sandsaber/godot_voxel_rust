//! Empty library root for the `tsan` workspace member.
//!
//! The actual ThreadSanitizer targets live in `tests/` as integration tests so
//! they exercise `voxel-core` purely through its public API. Run them with:
//!
//! ```text
//! CARGO_INCREMENTAL=0 RUSTFLAGS="-Zsanitizer=thread" \
//!   cargo +nightly test -Zbuild-std -p tsan --target x86_64-unknown-linux-gnu \
//!   --tests -- --test-threads=1
//! ```
//!
//! Use `--tests` (not a bare `cargo test`) so rustdoc doctests are skipped.
//! With `-Zbuild-std` + `-Zsanitizer=thread`, doctest crates are compiled
//! *without* the sanitizer flag and fail with an ABI-mismatch error.
