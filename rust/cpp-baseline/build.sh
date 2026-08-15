#!/usr/bin/env bash
# Build & run the transvoxel table dumper, regenerating the C++ golden dump
# consumed by voxel-core's Rust parity test.
#
# The dumper compiles the REAL upstream `transvoxel_tables.cpp` (copied here at
# build time) against an empty stub for its only dependency (`util/errors.h`),
# so the printed data is upstream's actual table contents — not a hand copy.
#
# Usage:
#   ./build.sh                 # default: g++, output to voxel-core golden dir
#   CXX=clang++ ./build.sh     # override compiler
#
# Requirements: a C++17 compiler (g++ or clang++).

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"          # godot_voxel repo root
BUILD="$HERE/build"
CXX="${CXX:-g++}"

rm -rf "$BUILD"
mkdir -p "$BUILD/util" "$BUILD/meshers/transvoxel"

# Empty stub for the only header the tables file includes. Its ZN_ASSERT uses
# are inside #ifdef DEBUG_ENABLED (which we leave undefined), so an empty stub
# compiles; we still define the macro defensively.
cat > "$BUILD/util/errors.h" <<'EOF'
#ifndef ZN_ERRORS_STUB_H
#define ZN_ERRORS_STUB_H
#define ZN_ASSERT(cond) ((void)0)
#define ZN_ASSERT_MSG(cond, msg) ((void)0)
#endif
EOF

# Copy the REAL upstream tables so we dump current data, not a stale hand-copy.
# The C++ runtime sources are gone from the working tree (the Rust port is the
# source of truth), so extract the pinned-upstream tables from the git object
# `5828cbeb` (recorded in rust/voxel-gdext/api/port_status.json) instead of a
# working-tree path. Requires the object to be present in the local clone.
PINNED_UPSTREAM="5828cbeba19050033f550485abc5f8c3586b1bf5"
UPSTREAM_TABLES_PATH="meshers/transvoxel/transvoxel_tables.cpp"
if ! git cat-file -t "$PINNED_UPSTREAM" >/dev/null 2>&1; then
	echo "ERROR: pinned upstream commit $PINNED_UPSTREAM is not present in the local clone." >&2
	echo "       Run: git fetch origin $PINNED_UPSTREAM (or clone with full history)." >&2
	exit 1
fi
git cat-file -p "$PINNED_UPSTREAM:$UPSTREAM_TABLES_PATH" > "$BUILD/meshers/transvoxel/transvoxel_tables.cpp"

# -I$BUILD lets dump_tables.cpp find meshers/transvoxel/transvoxel_tables.cpp;
# the copied file's `../../util/errors.h` then resolves to build/util/errors.h.
"$CXX" -std=c++17 -O0 -Wno-pedantic -I"$BUILD" "$HERE/dump_tables.cpp" -o "$BUILD/dump_tables"

OUT="$REPO/rust/voxel-core/tests/golden/transvoxel_tables_cpp.txt"
"$BUILD/dump_tables" > "$OUT"
echo "wrote $OUT"
