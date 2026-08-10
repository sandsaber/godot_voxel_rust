#!/usr/bin/env bash
# Builds the voxel-gdext library and runs the headless Godot smoke tests.
#
# The compiled `.so`/`.dylib`/`.dll` is a build artifact (git-ignored), so on a
# clean checkout it must be produced before Godot can load the GDExtension.
# This script does that, copies the artifact next to the .gdextension, registers
# it in the generated Godot extension list, then runs all checks. Requires
# `cargo` and `godot` on PATH.
#
# Usage:  ./voxel-gdext/smoke_test/run_smoke_test.sh [--release]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_RUST="$(cd "$SCRIPT_DIR/../.." && pwd)"
PROFILE="debug"
GODOT="${GODOT:-godot}"

while [[ $# -gt 0 ]]; do
	case "$1" in
		--release) PROFILE="release"; shift;;
		*) echo "unknown arg: $1" >&2; exit 2;;
	esac
done

cd "$REPO_RUST"
echo ">> building voxel-gdext ($PROFILE)..."
CARGO_ARGS=(build -p voxel-gdext)
if [[ "$PROFILE" == "release" ]]; then
	CARGO_ARGS+=(--release)
fi
cargo "${CARGO_ARGS[@]}"

# Copy the host artifact next to the .gdextension. Rust uses a different
# library prefix on Windows, so select the complete filename rather than only
# the extension.
case "$(uname -s)" in
	Linux*) LIB_NAME="libvoxel_gdext.so" ;;
	Darwin*) LIB_NAME="libvoxel_gdext.dylib" ;;
	MINGW*|MSYS*|CYGWIN*) LIB_NAME="voxel_gdext.dll" ;;
	*) echo "unsupported smoke-test host: $(uname -s)" >&2; exit 2 ;;
esac
SRC="$REPO_RUST/target/$PROFILE/$LIB_NAME"
DST="$SCRIPT_DIR/$LIB_NAME"
cp -f "$SRC" "$DST"
echo ">> copied $SRC -> $DST"

# A clean checkout has no `.godot/extension_list.cfg` because it is generated
# editor state. Script-only Godot runs do not scan for new `.gdextension`
# descriptors, so register this dedicated smoke extension explicitly.
mkdir -p "$SCRIPT_DIR/.godot"
printf '%s\n' 'res://voxel_gdext.gdextension' > "$SCRIPT_DIR/.godot/extension_list.cfg"

echo
echo ">> [1/3] API test (class registration + func surface)..."
"$GODOT" --headless --path "$SCRIPT_DIR" --script api_test.gd

echo
echo ">> [2/3] runtime paging test (terrain + generator + viewer, real frames)..."
"$GODOT" --headless --path "$SCRIPT_DIR" runtime_scene.tscn --quit-after 120

echo
echo ">> [3/3] smoke scene (VoxelTerrain node in a scene)..."
"$GODOT" --headless --path "$SCRIPT_DIR" smoke_test.tscn --quit-after 30

echo
echo ">> all smoke tests complete"
