#!/usr/bin/env bash
# Builds the voxel-gdext library and runs the headless Godot smoke tests.
#
# The compiled `.so`/`.dylib`/`.dll` is a build artifact (git-ignored), so on a
# clean checkout it must be produced before Godot can load the GDExtension.
# This script does that, copies the artifact next to the .gdextension, then runs
# all checks. Requires `cargo` and `godot` on PATH.
#
# Usage:  ./voxel-gdext/smoke_test/run_smoke_test.sh [--release]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_RUST="$(cd "$SCRIPT_DIR/../.." && pwd)"
PROFILE="debug"
GODOT="${GODOT:-godot}"
TIMEOUT_SECONDS="${GODOT_SMOKE_TIMEOUT_SECONDS:-180}"

if [[ ! "$TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
	echo "GODOT_SMOKE_TIMEOUT_SECONDS must be a positive integer" >&2
	exit 2
fi

LOG_DIR="$(mktemp -d /tmp/voxel-gdext-smoke.XXXXXX)"

cleanup_logs() {
	if [[ "$LOG_DIR" == /tmp/voxel-gdext-smoke.* ]]; then
		rm -rf -- "$LOG_DIR"
	else
		echo "refusing to remove unexpected smoke log directory: $LOG_DIR" >&2
		return 1
	fi
}
trap cleanup_logs EXIT

run_with_deadline() {
	if [[ "${GODOT_SMOKE_FORCE_WATCHDOG:-0}" != "1" ]]; then
		if command -v timeout >/dev/null 2>&1; then
			timeout "$TIMEOUT_SECONDS" "$@"
			return
		fi
		if command -v gtimeout >/dev/null 2>&1; then
			gtimeout "$TIMEOUT_SECONDS" "$@"
			return
		fi
	fi

	# Portable fallback for macOS and minimal systems without GNU coreutils.
	local deadline_marker
	deadline_marker="$(mktemp "$LOG_DIR/deadline.XXXXXX")"
	"$@" &
	local command_pid=$!
	(
		sleep "$TIMEOUT_SECONDS"
		if kill -0 "$command_pid" 2>/dev/null; then
			printf 'timed_out\n' > "$deadline_marker"
			if command -v pkill >/dev/null 2>&1; then
				pkill -TERM -P "$command_pid" 2>/dev/null || true
			fi
			kill -TERM "$command_pid" 2>/dev/null || true
			sleep 5
			if command -v pkill >/dev/null 2>&1; then
				pkill -KILL -P "$command_pid" 2>/dev/null || true
			fi
			kill -KILL "$command_pid" 2>/dev/null || true
		fi
	) >/dev/null 2>&1 &
	local watchdog_pid=$!
	local command_status
	if wait "$command_pid"; then
		command_status=0
	else
		command_status=$?
	fi
	kill "$watchdog_pid" 2>/dev/null || true
	wait "$watchdog_pid" 2>/dev/null || true
	if [[ -s "$deadline_marker" ]]; then
		return 124
	fi
	return "$command_status"
}

run_godot_check() {
	local label="$1"
	shift
	local log_file
	log_file="$(mktemp "$LOG_DIR/check.XXXXXX")"
	local -a pipeline_status

	set +e
	run_with_deadline "$@" 2>&1 | tee "$log_file"
	pipeline_status=("${PIPESTATUS[@]}")
	set -e

	if (( pipeline_status[1] != 0 )); then
		echo "smoke check '$label' could not write its diagnostic log" >&2
		return "${pipeline_status[1]}"
	fi
	if (( pipeline_status[0] != 0 )); then
		echo "smoke check '$label' exited ${pipeline_status[0]}" >&2
		return "${pipeline_status[0]}"
	fi
	if grep -E -q '\[panic |panicked at|fatal runtime error' "$log_file"; then
		echo "smoke check '$label' emitted a fatal Rust runtime diagnostic" >&2
		return 1
	fi
}

while [[ $# -gt 0 ]]; do
	case "$1" in
		--release) PROFILE="release"; shift;;
		*) echo "unknown arg: $1" >&2; exit 2;;
	esac
done

cd "$REPO_RUST"
echo ">> building voxel-gdext ($PROFILE)..."
BUILD_ARGS=(-p voxel-gdext)
if [[ "$PROFILE" == "release" ]]; then
	BUILD_ARGS+=(--release)
fi
cargo build "${BUILD_ARGS[@]}"

# Copy the artifact next to the .gdextension (which points at res://libvoxel_gdext.so).
EXT="so"; [[ "$(uname -s)" == "Darwin" ]] && EXT="dylib"
SRC="$REPO_RUST/target/$PROFILE/libvoxel_gdext.$EXT"
DST="$SCRIPT_DIR/libvoxel_gdext.$EXT"
cp -f "$SRC" "$DST"
echo ">> copied $SRC -> $DST ($(wc -c < "$DST") bytes)"
ls -l "$DST" "$SCRIPT_DIR/voxel_gdext.gdextension"

echo
echo ">> [1/5] API test (class registration + func surface)..."
run_godot_check "API" \
	"$GODOT" --headless --path "$SCRIPT_DIR" --script api_test.gd

echo
echo ">> [2/5] runtime paging test (terrain + generator + viewer, real frames)..."
run_godot_check "runtime paging" \
	"$GODOT" --headless --path "$SCRIPT_DIR" runtime_scene.tscn

echo
echo ">> [3/5] smoke scene (VoxelTerrain node in a scene)..."
run_godot_check "smoke scene" \
	"$GODOT" --headless --path "$SCRIPT_DIR" smoke_test.tscn

echo
echo ">> [4/5] runtime correctness (remesh + unload + safety + persistence)..."
run_godot_check "runtime correctness" \
	"$GODOT" --headless --path "$SCRIPT_DIR" runtime_correctness.tscn

echo
echo ">> [5/6] 3-LOD Variable LOD integration (multi-LOD paging, split/join, negatives)..."
run_godot_check "variable lod 3" \
	"$GODOT" --headless --path "$SCRIPT_DIR" variable_lod_3.tscn

echo
echo ">> [6/6] blocky library on terrain (type channel + baked cube)..."
run_godot_check "blocky terrain" \
	"$GODOT" --headless --path "$SCRIPT_DIR" blocky_terrain.tscn

echo
echo ">> all smoke tests complete"
