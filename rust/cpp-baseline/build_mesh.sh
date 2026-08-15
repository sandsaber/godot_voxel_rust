#!/usr/bin/env bash
# Build & run the transvoxel mesh-dumping harness — invokes the REAL upstream
# `build_regular_mesh<float, NullProcessor>` on an SDF sphere identical to the
# Rust golden, emitting a GoldenMesh JSON (H1 parity) + timing (H2 baseline).
#
# Strategy: copy the REAL transvoxel.{h,cpp}, tables, and null-material into a
# BUILD dir, then create minimal STUB headers alongside them so the relative
# `#include "../../storage/voxel_buffer.h"` and `../../util/...` paths resolve
# to stubs (empty or minimal) rather than the heavy Godot-dependent originals.
# Only the portable `util/math/` headers (vector3t, vector3f, funcs, span,
# fixed_array) are pulled from the real repo root via `-I$REPO` AFTER the stubs,
# so they shadow correctly.
#
# Usage:
#   ./build_mesh.sh                # 16-sphere: JSON to stdout, timing to stderr
#   ./build_mesh.sh --regenerate   # rewrite both golden JSON files from C++
#   CXX=clang++ ./build_mesh.sh

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
BUILD="$HERE/build"
CXX="${CXX:-g++}"
REGENERATE=0

for arg in "$@"; do
    case "$arg" in
        --regenerate) REGENERATE=1;;
        *) echo "unknown arg: $arg" >&2; exit 2;;
    esac
done

# Wipe and rebuild the stub tree from scratch each run (deterministic).
rm -rf "$BUILD"
mkdir -p "$BUILD/meshers/transvoxel" "$BUILD/storage" "$BUILD/util/godot/core" \
         "$BUILD/util/math" "$BUILD/util/memory" "$BUILD/util/containers" \
         "$BUILD/constants" "$BUILD/util"

# The C++ runtime sources are gone from the working tree (Rust port is
# the source of truth), so extract ALL needed sources from the pinned
# upstream commit 5828cbeb (recorded in port_status.json).
PINNED_UPSTREAM="5828cbeba19050033f550485abc5f8c3586b1bf5"
if ! git cat-file -t "$PINNED_UPSTREAM" >/dev/null 2>&1; then
	echo "ERROR: pinned upstream commit $PINNED_UPSTREAM is not present in the local clone." >&2
	echo "       Run: git fetch origin $PINNED_UPSTREAM (or clone with full history)." >&2
	exit 1
fi
extract_upstream() {  # extract_upstream <repo-path> <dest>
	git cat-file -p "$PINNED_UPSTREAM:$1" > "$2"
}
extract_tree() {  # extract_tree <git-path-prefix> <dest-dir>
	local prefix="$1" dest="$2"
	mkdir -p "$dest"
	git cat-file -p "$PINNED_UPSTREAM:$prefix" | while read mode type sha name; do
		if [ "$type" = "blob" ]; then
			git cat-file -p "$sha" > "$dest/$name"
		elif [ "$type" = "tree" ]; then
			extract_tree "$prefix/$name" "$dest/$name"
		fi
	done
}

# Extract the PORTABLE real headers from the pinned git object.
# util/containers: real headers from git object.
extract_tree "util/containers" "$BUILD/util/containers"
# util/math: real headers from git object.
extract_tree "util/math" "$BUILD/util/math"
# util/godot: real headers + classes tree from git object (we stub core/ below).
for f in $(git cat-file -p "$PINNED_UPSTREAM:util/godot/" | awk '$2=="blob"{print $4}'); do
	git cat-file -p "$PINNED_UPSTREAM:util/godot/$f" > "$BUILD/util/godot/$f"
done
extract_tree "util/godot/classes" "$BUILD/util/godot/classes"

# -----------------------------------------------------------------------
# Copy REAL upstream sources. transvoxel.cpp is TRIMMED to the inner regular
# template only (lines 1-602): the transition mesh + VoxelBuffer dispatcher
# (lines 604+) pull in heavy Godot APIs we don't exercise, so they're cut to
# avoid stubbing the engine. A closing namespace brace is re-appended.
# The C++ runtime sources are gone from the working tree (the Rust port is
# the source of truth), so extract the pinned-upstream transvoxel sources
# from the git object 5828cbeb (recorded in port_status.json). The
# PINNED_UPSTREAM and extract_upstream helpers were defined above.
# -----------------------------------------------------------------------
extract_upstream "meshers/transvoxel/transvoxel.h"                "$BUILD/meshers/transvoxel/transvoxel.h"
extract_upstream "meshers/transvoxel/transvoxel_tables.cpp"       "$BUILD/meshers/transvoxel/transvoxel_tables.cpp"
extract_upstream "meshers/transvoxel/transvoxel_materials_null.h" "$BUILD/meshers/transvoxel/transvoxel_materials_null.h"
# Trim transvoxel.cpp: keep includes + inner template (1-602), drop transition/
# dispatcher (604-1553), keep the namespace close (re-added manually).
TRANSVOXEL_CPP="$BUILD/meshers/transvoxel/transvoxel.cpp"
extract_upstream "meshers/transvoxel/transvoxel.cpp" "$TRANSVOXEL_CPP.full"
head -n 602 "$TRANSVOXEL_CPP.full" > "$TRANSVOXEL_CPP"
rm -f "$TRANSVOXEL_CPP.full"
# Drop the unused mixel4/single_s4 material includes (lines 8-9) — they pull in
# Godot APIs and we only use NullProcessor.
sed '/transvoxel_materials_mixel4.h/d; /transvoxel_materials_single_s4.h/d' "$TRANSVOXEL_CPP" > "$TRANSVOXEL_CPP.tmp"
mv "$TRANSVOXEL_CPP.tmp" "$TRANSVOXEL_CPP"
# Re-close the namespace (the original close was at the file's end we trimmed).
printf '\n} // namespace zylann::voxel::transvoxel\n' >> "$TRANSVOXEL_CPP"

# -----------------------------------------------------------------------
# STUB headers. These shadow the Godot-dependent originals by sitting at the
# same relative path the real transvoxel.{h,cpp} reach via `../../`.
# -----------------------------------------------------------------------

# util/errors.h — no-op asserts (we leave DEBUG_ENABLED undefined).
cat > "$BUILD/util/errors.h" <<'EOF'
#ifndef ZN_ERRORS_STUB_H
#define ZN_ERRORS_STUB_H
#define ZN_ASSERT(cond) ((void)0)
#define ZN_ASSERT_MSG(cond, msg) ((void)0)
#define ZN_ASSERT_RETURN(cond) ((void)0)
#define ZN_ASSERT_RETURN_V(cond, retval) (retval)
#define ZN_ASSERT_RETURN_V_MSG(cond, retval, msg) (retval)
#define ZN_CRASH()
#define ZN_CRASH_MSG(msg)
#define ZN_PRINT_ERROR(msg)
#define CRASH_NOW() ((void)0)
#define CRASH_COND(cond) ((void)0)
#endif
EOF

# util/macros.h
cat > "$BUILD/util/macros.h" <<'EOF'
#ifndef ZN_MACROS_STUB_H
#define ZN_MACROS_STUB_H
#define ZN_GODOT_NAMESPACE_BEGIN
#define ZN_GODOT_NAMESPACE_END
#endif
EOF

# util/profiling.h — no-op (TRACY_ENABLE undefined).
cat > "$BUILD/util/profiling.h" <<'EOF'
#ifndef ZN_PROFILING_STUB_H
#define ZN_PROFILING_STUB_H
#define ZN_PROFILE_SCOPE()
#define ZN_PROFILE_SCOPE_NAMED(name)
#endif
EOF

# util/memory/memory.h + std_allocator.h — map allocators to std (no Godot).
cat > "$BUILD/util/memory/memory.h" <<'EOF'
#ifndef ZN_MEMORY_STUB_H
#define ZN_MEMORY_STUB_H
#include <cstdlib>
#include <memory>
namespace zylann {
template <typename T> struct DefaultObjectDeleter { void operator()(T *o){ delete o; } };
template <typename T> using UniquePtr = std::unique_ptr<T, DefaultObjectDeleter<T>>;
template <class T, class... A> UniquePtr<T> make_unique_instance(A&&... a){
    return UniquePtr<T>(new T(std::forward<A>(a)...)); }
}
#define ZN_NEW(T) new T
#define ZN_DELETE(p) delete p
#define ZN_ALLOC(n) std::malloc(n)
#define ZN_REALLOC(p, n) std::realloc(p, n)
#define ZN_FREE(p) std::free(p)
#endif
EOF
cat > "$BUILD/util/memory/std_allocator.h" <<'EOF'
#ifndef ZN_STD_ALLOCATOR_STUB_H
#define ZN_STD_ALLOCATOR_STUB_H
#include <memory>
namespace zylann {
template <class T> struct StdDefaultAllocator : public std::allocator<T> {};
}
#endif
EOF

# util/hash_funcs.h — minimal (vector3i.h includes it; mesher doesn't hash).
cat > "$BUILD/util/hash_funcs.h" <<'EOF'
#ifndef ZN_HASH_FUNCS_STUB_H
#define ZN_HASH_FUNCS_STUB_H
#include <cstdint>
namespace zylann {
inline uint32_t hash_djb2_one_32(uint32_t p_in, uint32_t p_prev = 5381){
    return ((p_prev << 5) + p_prev) ^ p_in; }
inline uint64_t hash_djb2_one_64(uint64_t p_in, uint64_t p_prev = 5381){
    return ((p_prev << 5) + p_prev) ^ p_in; }
inline uint32_t hash_murmur3_one_32(uint32_t, uint32_t = 0x7F07C65){ return 0; }
inline uint32_t hash_fmix32(uint32_t h){ return h; }
}
#endif
EOF

# util/godot/core/vector3i.h — define Godot's Vector3i in the GLOBAL namespace.
# The mesher's util/math/vector3i.h needs `Vector3i` to be a real type.
cat > "$BUILD/util/godot/core/vector3i.h" <<'EOF'
#ifndef ZN_GODOT_VECTOR3I_STUB_H
#define ZN_GODOT_VECTOR3I_STUB_H
struct Vector3i {
    static const int AXIS_X = 0;
    static const int AXIS_Y = 1;
    static const int AXIS_Z = 2;
    static const unsigned int AXIS_COUNT = 3;
    enum Axis { AXIS_X_, AXIS_Y_, AXIS_Z_ };
    int x = 0, y = 0, z = 0;
    inline Vector3i() {}
    inline Vector3i(int px, int py, int pz) : x(px), y(py), z(pz) {}
    inline Vector3i operator+(const Vector3i &o) const { return Vector3i(x+o.x, y+o.y, z+o.z); }
    inline Vector3i operator-(const Vector3i &o) const { return Vector3i(x-o.x, y-o.y, z-o.z); }
    inline Vector3i operator*(int s) const { return Vector3i(x*s, y*s, z*s); }
    inline int &operator[](unsigned int i){ return (i==0)?x:(i==1)?y:z; }
    inline const int &operator[](unsigned int i) const { return (i==0)?x:(i==1)?y:z; }
    inline bool operator==(const Vector3i &o) const { return x==o.x && y==o.y && z==o.z; }
};
inline Vector3i operator*(int s, const Vector3i &v){ return v*s; }
#endif
EOF

# util/godot/core/color.h must define a minimal `Color` type — the real
# util/math/color.h references it in lerp.
cat > "$BUILD/util/godot/core/color.h" <<'EOF'
#ifndef ZN_GODOT_COLOR_STUB_H
#define ZN_GODOT_COLOR_STUB_H
struct Color {
    float r = 0, g = 0, b = 0, a = 1;
    inline Color() {}
    inline Color(float pr, float pg, float pb, float pa = 1.f) : r(pr), g(pg), b(pb), a(pa) {}
    inline Color operator*(float s) const { return Color(r*s, g*s, b*s, a*s); }
    inline Color operator+(const Color &o) const { return Color(r+o.r, g+o.g, b+o.b, a+o.a); }
};
inline Color lerp(const Color a, const Color b, float t){ return Color(
    a.r + (b.r-a.r)*t, a.g + (b.g-a.g)*t, a.b + (b.b-a.b)*t, a.a + (b.a-a.a)*t); }
#endif
EOF

# util/godot/core/vector2.h, vector2i.h, vector3.h, transform_3d.h, sort_array.h
# — define minimal placeholder types so conv.h's (uncalled) signatures compile.
for h in vector3 Vector3 vector2 Vector2 vector2i Vector2i; do :; done
cat > "$BUILD/util/godot/core/vector3.h" <<'EOF'
#ifndef ZN_GODOT_VECTOR3_STUB_H
#define ZN_GODOT_VECTOR3_STUB_H
struct Vector3 {
    float x=0, y=0, z=0;
    inline Vector3(){}
    inline Vector3(float px, float py, float pz): x(px), y(py), z(pz){}
    inline Vector3 operator-(const Vector3&o) const { return Vector3(x-o.x, y-o.y, z-o.z); }
};
#endif
EOF
cat > "$BUILD/util/godot/core/vector2.h" <<'EOF'
#ifndef ZN_GODOT_VECTOR2_STUB_H
#define ZN_GODOT_VECTOR2_STUB_H
struct Vector2 {
    float x=0, y=0;
    inline Vector2(){}
    inline Vector2(float px, float py): x(px), y(py){}
};
#endif
EOF
cat > "$BUILD/util/godot/core/vector2i.h" <<'EOF'
#ifndef ZN_GODOT_VECTOR2I_STUB_H
#define ZN_GODOT_VECTOR2I_STUB_H
struct Vector2i {
    int x=0, y=0;
    inline Vector2i(){}
    inline Vector2i(int px, int py): x(px), y(py){}
};
#endif
EOF
# util/godot/core/basis.h — minimal Basis (conv.h mentions it in to_basis3f).
cat > "$BUILD/util/godot/core/basis.h" <<'EOF'
#ifndef ZN_GODOT_BASIS_STUB_H
#define ZN_GODOT_BASIS_STUB_H
struct Basis {
    Vector3 rows[3];
    inline Basis(){}
    inline Basis(const Vector3&a,const Vector3&b,const Vector3&c){ rows[0]=a; rows[1]=b; rows[2]=c; }
};
#endif
EOF
# util/godot/core/transform_3d.h — minimal Transform3D (conv.h mentions it).
cat > "$BUILD/util/godot/core/transform_3d.h" <<'EOF'
#ifndef ZN_GODOT_TRANSFORM3D_STUB_H
#define ZN_GODOT_TRANSFORM3D_STUB_H
#include "basis.h"
#include "vector3.h"
struct Transform3D {
    Basis basis; Vector3 origin;
    inline Transform3D(){}
    inline Transform3D(const Basis&b, const Vector3&o): basis(b), origin(o){}
};
#endif
EOF
echo "#ifndef STUB_sort_array" > "$BUILD/util/godot/core/sort_array.h"
echo "#define STUB_sort_array" >> "$BUILD/util/godot/core/sort_array.h"
echo "#endif" >> "$BUILD/util/godot/core/sort_array.h"

# util/math/funcs.h needs `Math::`. We force-inject a Math namespace covering
# every function the real funcs.h calls (audit: abs atan cos floor
# is_equal_approx is_zero_approx lerp pow sin smoothstep snapped sqrt wrapf wrapi).
cat > "$BUILD/util/math/godot_math_funcs.h" <<'EOF'
#ifndef ZN_GODOT_MATH_STUB_H
#define ZN_GODOT_MATH_STUB_H
#include <cmath>
namespace Math {
    inline float abs(float x){ return std::fabs(x); }
    inline int abs(int x){ return x<0?-x:x; }
    inline float atan(float x){ return std::atan(x); }
    inline float atan2(float y, float x){ return std::atan2(y, x); }
    inline float cos(float x){ return std::cos(x); }
    inline float sin(float x){ return std::sin(x); }
    inline float floor(float x){ return std::floor(x); }
    inline float ceil(float x){ return std::ceil(x); }
    inline float round(float x){ return std::round(x); }
    inline float sqrt(float x){ return std::sqrt(x); }
    inline double sqrt(double x){ return std::sqrt(x); }
    inline float pow(float a, float b){ return std::pow(a, b); }
    inline float min(float a, float b){ return a<b?a:b; }
    inline float max(float a, float b){ return a>b?a:b; }
    inline int min(int a, int b){ return a<b?a:b; }
    inline int max(int a, int b){ return a>b?a:b; }
    template <typename T> inline T lerp(T a, T b, T t){ return a + (b-a)*t; }
    inline bool is_zero_approx(float x){ return std::fabs(x) < 1e-6f; }
    inline bool is_equal_approx(float a, float b){ return std::fabs(a-b) < 1e-6f*std::fmax(std::fabs(a),std::fabs(b)); }
    inline bool is_equal_approx(float a, float b, float tol){ if (a==b) return true; return std::fabs(a-b) <= tol; }
    inline float wrapf(float x, float d){ return Math::is_zero_approx(d) ? 0.f : x - (d * Math::floor(x/d)); }
    inline int wrapi(int x, int d){ return ((x % d) + d) % d; }
    inline float snapped(float v, float step){ return step!=0.f ? std::floor(v/step + 0.5f)*step : v; }
    inline float smoothstep(float from, float to, float w){
        if (is_equal_approx(from, to)) return from;
        float x = min(max((w-from)/(to-from), 0.f), 1.f);
        return x*x*(3.f-2.f*x);
    }
}
#endif
EOF

# storage/voxel_buffer.h — minimal VoxelBuffer with the enums/per-method-sigs the
# dispatcher (compiled but not called) references as nested name specifiers.
cat > "$BUILD/storage/voxel_buffer.h" <<'EOF'
#ifndef ZN_VOXEL_BUFFER_STUB_H
#define ZN_VOXEL_BUFFER_STUB_H
#include <cstdint>
namespace zylann {
class VoxelBuffer {
public:
    enum Compression { COMPRESSION_NONE, COMPRESSION_UNIFORM };
    enum Depth { DEPTH_8_BIT, DEPTH_16_BIT, DEPTH_32_BIT, DEPTH_64_BIT };
    Compression get_channel_compression(unsigned int) const { return COMPRESSION_NONE; }
    Depth get_channel_depth(unsigned int) const { return DEPTH_32_BIT; }
    unsigned int get_size_in_bytes_for_volume(unsigned int) const { return 0; }
};
}
#endif
EOF

# constants/cube_tables.h — provides the Cube::SIDE_* enum used by the
# transition-mesh helpers (compiled but not called by our regular path).
cat > "$BUILD/constants/cube_tables.h" <<'EOF'
#ifndef ZN_CUBE_TABLES_STUB_H
#define ZN_CUBE_TABLES_STUB_H
namespace Cube {
    enum Side {
        SIDE_NEGATIVE_X, SIDE_POSITIVE_X,
        SIDE_NEGATIVE_Y, SIDE_POSITIVE_Y,
        SIDE_NEGATIVE_Z, SIDE_POSITIVE_Z,
    };
}
#endif
EOF
echo "#ifndef STUB_MIXEL4" > "$BUILD/storage/mixel4.h"
echo "#define STUB_MIXEL4" >> "$BUILD/storage/mixel4.h"
echo "#endif" >> "$BUILD/storage/mixel4.h"

# meshers/transvoxel/transvoxel_materials_mixel4.h & _single_s4.h — only
# triggered by template instantiation we don't perform; empty stubs compile.
echo "#ifndef STUB_MIXEL4_MAT" > "$BUILD/meshers/transvoxel/transvoxel_materials_mixel4.h"
echo "#define STUB_MIXEL4_MAT" >> "$BUILD/meshers/transvoxel/transvoxel_materials_mixel4.h"
echo "#endif" >> "$BUILD/meshers/transvoxel/transvoxel_materials_mixel4.h"
echo "#ifndef STUB_S4_MAT" > "$BUILD/meshers/transvoxel/transvoxel_materials_single_s4.h"
echo "#define STUB_S4_MAT" >> "$BUILD/meshers/transvoxel/transvoxel_materials_single_s4.h"
echo "#endif" >> "$BUILD/meshers/transvoxel/transvoxel_materials_single_s4.h"

# -----------------------------------------------------------------------
# Compile. Include order:
#   1. -I$BUILD   — stubs win (same relative paths resolve here first)
#   2. -I$REPO    — real portable util/ headers (span, fixed_array, funcs,
#                   vector3t, vector3f, vector2t, vector2f, conv, vector3i,
#                   vector3i16, transform3f, box3i, box2i...)
# Math:: injected via -include so it's defined before funcs.h runs.
# -----------------------------------------------------------------------
echo "compiling dump_mesh harness..."
# Force-include: <cstdint> for uint64_t (funcs.h), the Math:: stub, UNIT_EPSILON
# (vector3f.h via constants.h), a plain StdVector alias (transvoxel.h uses it but
# expects voxel_buffer.h to provide it), and a forward-declared VoxelBuffer so
# the dispatcher signatures (not called) compile.
cat > "$BUILD/_force_include.h" <<'EOF'
#include <cstdint>
#include <vector>
#include "util/math/godot_math_funcs.h"
#ifndef UNIT_EPSILON
#define UNIT_EPSILON 0.00001f
#endif
namespace zylann {
template <typename T> using StdVector = std::vector<T>;
class VoxelBuffer;  // forward decl: only the dispatcher (not called) needs the full type.
// snorm conversions used by sdf_as_float(int8/int16) — not on the float path
// we exercise, but they compile as free functions and must resolve.
inline float s8_to_snorm_noclamp(int8_t v) { return float(v) / 127.0f; }
inline float s16_to_snorm_noclamp(int16_t v) { return float(v) / 32767.0f; }
}
EOF
"$CXX" -std=c++17 -O2 -DNDEBUG -w \
    -I"$BUILD" -I"$REPO" \
    -include "$BUILD/_force_include.h" \
    "$HERE/dump_mesh.cpp" -o "$BUILD/dump_mesh"
echo "built $BUILD/dump_mesh"
echo "------------------------------------------------------------------------"

GOLDEN_DIR="$REPO/rust/voxel-core/tests/golden"

if [ "$REGENERATE" = "1" ]; then
    for spec in "16 6.0 transvoxel_sphere_16.json" "32 13.0 transvoxel_sphere_32.json"; do
        set -- $spec
        inner=$1; radius=$2; name=$3
        out="$GOLDEN_DIR/$3"
        echo "regenerating $out (inner=$inner, radius=$radius)..."
        "$BUILD/dump_mesh" "$inner" "$radius" "$out"
    done
    echo "done. Inspect the C++ golden diff, then commit."
else
    # Default: run the 16-sphere — timing to stderr, JSON to stdout.
    "$BUILD/dump_mesh" 16 6.0
fi
