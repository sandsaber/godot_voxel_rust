//! `streams::variant_wire` — pure-Rust codec for Godot's Variant binary wire
//! format (ROADMAP R7 wide).
//!
//! Upstream godot_voxel stores "wide" block metadata (Dictionaries, Arrays,
//! vectors…) as a custom metadata entry whose payload is exactly Godot's
//! `encode_variant(variant, allow_objects = false)` byte stream
//! (`VoxelMetadataVariant`, custom tag 32). Decoding those saves without a
//! Godot runtime requires reimplementing that wire format here — this module
//! is that reimplementation, so `voxel-core` stays engine-free.
//!
//! Supported types cover everything a metadata payload realistically holds:
//! scalars, strings, vectors/rects/plane/quaternion/AABB/color, Dictionary,
//! Array, and the packed arrays. Node paths, RIDs, objects, callables and
//! transforms are rejected on decode (objects are already excluded by
//! upstream's `allow_objects = false`; the rest has no meaningful metadata
//! use) — a section containing them is skipped by the block serializer, same
//! as before the wide codec existed.
//!
//! Format reference (Godot 4 `core/io/marshalls.cpp`, `encode_variant`):
//! every value starts with a `u32` little-endian header whose low 16 bits
//! are the `Variant::Type` and whose high bits are flags
//! (`ENCODE_FLAG_OBJECT_AS_ID = 1`). `FLOAT` is a 64-bit double, integers
//! are 64-bit, `STRING` is `u32` byte length + UTF-8 padded to 4 bytes,
//! containers are `u32` element counts followed by their elements, and
//! packed byte/string arrays pad their payload to 4-byte multiples.

use crate::io::serialization::{MemoryReader, MemoryWriter};

/// Variant type ids as written into the wire header (Godot 4 `Variant::Type`).
mod types {
    pub const NIL: u32 = 0;
    pub const BOOL: u32 = 1;
    pub const INT: u32 = 2;
    pub const FLOAT: u32 = 3;
    pub const STRING: u32 = 4;
    pub const VECTOR2: u32 = 5;
    pub const VECTOR2I: u32 = 6;
    pub const RECT2: u32 = 7;
    pub const RECT2I: u32 = 8;
    pub const VECTOR3: u32 = 9;
    pub const VECTOR3I: u32 = 10;
    pub const TRANSFORM2D: u32 = 11;
    pub const VECTOR4: u32 = 12;
    pub const VECTOR4I: u32 = 13;
    pub const PLANE: u32 = 14;
    pub const QUATERNION: u32 = 15;
    pub const AABB: u32 = 16;
    pub const BASIS: u32 = 17;
    pub const TRANSFORM3D: u32 = 18;
    pub const PROJECTION: u32 = 19;
    pub const COLOR: u32 = 20;
    pub const STRING_NAME: u32 = 21;
    pub const NODE_PATH: u32 = 22;
    pub const RID: u32 = 23;
    pub const OBJECT: u32 = 24;
    pub const CALLABLE: u32 = 25;
    pub const SIGNAL: u32 = 26;
    pub const DICTIONARY: u32 = 27;
    pub const ARRAY: u32 = 28;
    pub const PACKED_BYTE_ARRAY: u32 = 29;
    pub const PACKED_INT32_ARRAY: u32 = 30;
    pub const PACKED_INT64_ARRAY: u32 = 31;
    pub const PACKED_FLOAT32_ARRAY: u32 = 32;
    pub const PACKED_FLOAT64_ARRAY: u32 = 33;
    pub const PACKED_STRING_ARRAY: u32 = 34;
    pub const PACKED_VECTOR2_ARRAY: u32 = 35;
    pub const PACKED_VECTOR3_ARRAY: u32 = 36;
    pub const PACKED_COLOR_ARRAY: u32 = 37;
}

/// Flag in the wire header's high bits: object encoded as an id.
const ENCODE_FLAG_OBJECT_AS_ID: u32 = 1;

/// A decoded/encodable Godot Variant for metadata purposes. Floats are the
/// wire-width types (f64 for `FLOAT` and vector components, f32 for colors).
///
/// `Dictionary` preserves wire order as a pair list; two dictionaries with
/// the same pairs in different orders compare unequal (`PartialEq` is
/// derived). Godot dictionaries are unordered, so treat equality of
/// dictionary-carrying metadata as an implementation detail, not a contract.
#[derive(Debug, Clone, PartialEq)]
pub enum VariantWireValue {
    Nil,
    Bool(bool),
    Int(i64),
    Float(f64),
    Text(String),
    Vector2([f64; 2]),
    Vector2i([i32; 2]),
    Rect2([f64; 4]),
    Rect2i([i32; 4]),
    Vector3([f64; 3]),
    Vector3i([i32; 3]),
    Vector4([f64; 4]),
    Vector4i([i32; 4]),
    Plane([f64; 4]),
    Quaternion([f64; 4]),
    Aabb([f64; 6]),
    Color([f32; 4]),
    Array(Vec<VariantWireValue>),
    Dictionary(Vec<(VariantWireValue, VariantWireValue)>),
    ByteArray(Vec<u8>),
    Int32Array(Vec<i32>),
    Int64Array(Vec<i64>),
    Float32Array(Vec<f32>),
    Float64Array(Vec<f64>),
    StringArray(Vec<String>),
    Vector2Array(Vec<[f64; 2]>),
    Vector3Array(Vec<[f64; 3]>),
    ColorArray(Vec<[f32; 4]>),
}

/// Why a Variant payload could not be decoded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VariantWireError {
    /// Ran out of bytes mid-value.
    UnexpectedEof,
    /// Hostile (or corrupt) nesting exceeded the depth budget.
    TooDeep,
    /// A count/length prefix exceeds the remaining payload.
    LengthOverflow,
    /// The value uses a type this codec intentionally does not support
    /// (objects, callables, node paths, transforms…). The caller treats the
    /// containing metadata section as foreign rather than corrupt.
    UnsupportedType(u32),
    /// Structurally invalid (e.g. invalid UTF-8, bad bool).
    Invalid,
}

/// Encode `value` into `dst` in Godot's Variant wire format. Self-delimiting:
/// exactly the bytes of this value are appended, so a stream of values (the
/// C++ custom-metadata payload) can continue right after it.
pub fn encode_variant(value: &VariantWireValue, dst: &mut Vec<u8>) {
    let mut w = MemoryWriter::little(dst);
    use VariantWireValue as V;
    match value {
        V::Nil => w.store_32(types::NIL),
        V::Bool(b) => {
            w.store_32(types::BOOL);
            w.store_32(u32::from(*b));
        }
        V::Int(i) => {
            w.store_32(types::INT);
            w.store_64(*i as u64);
        }
        V::Float(f) => {
            w.store_32(types::FLOAT);
            w.store_64(f.to_bits());
        }
        V::Text(s) => {
            w.store_32(types::STRING);
            store_string(&mut w, s);
        }
        V::Vector2(v) => {
            w.store_32(types::VECTOR2);
            store_f64s(&mut w, v);
        }
        V::Vector2i(v) => {
            w.store_32(types::VECTOR2I);
            store_i32s(&mut w, v);
        }
        V::Rect2(v) => {
            w.store_32(types::RECT2);
            store_f64s(&mut w, v);
        }
        V::Rect2i(v) => {
            w.store_32(types::RECT2I);
            store_i32s(&mut w, v);
        }
        V::Vector3(v) => {
            w.store_32(types::VECTOR3);
            store_f64s(&mut w, v);
        }
        V::Vector3i(v) => {
            w.store_32(types::VECTOR3I);
            store_i32s(&mut w, v);
        }
        V::Vector4(v) => {
            w.store_32(types::VECTOR4);
            store_f64s(&mut w, v);
        }
        V::Vector4i(v) => {
            w.store_32(types::VECTOR4I);
            store_i32s(&mut w, v);
        }
        V::Plane(v) => {
            w.store_32(types::PLANE);
            store_f64s(&mut w, v);
        }
        V::Quaternion(v) => {
            w.store_32(types::QUATERNION);
            store_f64s(&mut w, v);
        }
        V::Aabb(v) => {
            w.store_32(types::AABB);
            store_f64s(&mut w, v);
        }
        V::Color(v) => {
            w.store_32(types::COLOR);
            for c in v {
                w.store_32(c.to_bits());
            }
        }
        V::Array(items) => {
            w.store_32(types::ARRAY);
            store_count(&mut w, items.len());
            for item in items {
                // Re-borrow: each element appends to the same `dst`.
                encode_variant(item, dst);
            }
        }
        V::Dictionary(pairs) => {
            w.store_32(types::DICTIONARY);
            store_count(&mut w, pairs.len());
            for (key, value) in pairs {
                encode_variant(key, dst);
                encode_variant(value, dst);
            }
        }
        V::ByteArray(bytes) => {
            w.store_32(types::PACKED_BYTE_ARRAY);
            store_padded_bytes(&mut w, bytes);
        }
        V::Int32Array(items) => {
            w.store_32(types::PACKED_INT32_ARRAY);
            store_count(&mut w, items.len());
            for i in items {
                w.store_32(*i as u32);
            }
        }
        V::Int64Array(items) => {
            w.store_32(types::PACKED_INT64_ARRAY);
            store_count(&mut w, items.len());
            for i in items {
                w.store_64(*i as u64);
            }
        }
        V::Float32Array(items) => {
            w.store_32(types::PACKED_FLOAT32_ARRAY);
            store_count(&mut w, items.len());
            for f in items {
                w.store_32(f.to_bits());
            }
        }
        V::Float64Array(items) => {
            w.store_32(types::PACKED_FLOAT64_ARRAY);
            store_count(&mut w, items.len());
            for f in items {
                w.store_64(f.to_bits());
            }
        }
        V::StringArray(items) => {
            w.store_32(types::PACKED_STRING_ARRAY);
            store_count(&mut w, items.len());
            for s in items {
                store_string(&mut w, s);
            }
        }
        V::Vector2Array(items) => {
            w.store_32(types::PACKED_VECTOR2_ARRAY);
            store_count(&mut w, items.len());
            for v in items {
                store_f64s(&mut w, v);
            }
        }
        V::Vector3Array(items) => {
            w.store_32(types::PACKED_VECTOR3_ARRAY);
            store_count(&mut w, items.len());
            for v in items {
                store_f64s(&mut w, v);
            }
        }
        V::ColorArray(items) => {
            w.store_32(types::PACKED_COLOR_ARRAY);
            store_count(&mut w, items.len());
            for c in items {
                for f in c {
                    w.store_32(f.to_bits());
                }
            }
        }
    }
}

fn store_f64s(w: &mut MemoryWriter<'_, Vec<u8>>, values: &[f64]) {
    for v in values {
        w.store_64(v.to_bits());
    }
}

fn store_i32s(w: &mut MemoryWriter<'_, Vec<u8>>, values: &[i32]) {
    for v in values {
        w.store_32(*v as u32);
    }
}

fn store_count(w: &mut MemoryWriter<'_, Vec<u8>>, count: usize) {
    w.store_32(count as u32);
}

fn store_string(w: &mut MemoryWriter<'_, Vec<u8>>, s: &str) {
    w.store_32(s.len() as u32);
    w.store_buffer(s.as_bytes());
    let pad = s.len() % 4;
    if pad != 0 {
        for _ in 0..(4 - pad) {
            w.store_8(0);
        }
    }
}

fn store_padded_bytes(w: &mut MemoryWriter<'_, Vec<u8>>, bytes: &[u8]) {
    w.store_32(bytes.len() as u32);
    w.store_buffer(bytes);
    let pad = bytes.len() % 4;
    if pad != 0 {
        for _ in 0..(4 - pad) {
            w.store_8(0);
        }
    }
}

/// Decode one Variant from `src` starting at the reader's position. On
/// success the reader has consumed exactly this value's bytes.
pub fn decode_variant(
    r: &mut MemoryReader<'_>,
    max_depth: u32,
) -> Result<VariantWireValue, VariantWireError> {
    decode_at_depth(r, max_depth, 0)
}

fn decode_at_depth(
    r: &mut MemoryReader<'_>,
    max_depth: u32,
    depth: u32,
) -> Result<VariantWireValue, VariantWireError> {
    if depth > max_depth {
        return Err(VariantWireError::TooDeep);
    }
    let header = r.try_get_32().ok_or(VariantWireError::UnexpectedEof)?;
    let flags = header >> 16;
    let type_id = header & 0xffff;
    if flags != 0 {
        // Objects-as-id and any future flags: foreign, not corrupt.
        return Err(VariantWireError::UnsupportedType(type_id));
    }
    match type_id {
        types::NIL => Ok(VariantWireValue::Nil),
        types::BOOL => {
            let raw = r.try_get_32().ok_or(VariantWireError::UnexpectedEof)?;
            if raw > 1 {
                return Err(VariantWireError::Invalid);
            }
            Ok(VariantWireValue::Bool(raw != 0))
        }
        types::INT => Ok(VariantWireValue::Int(
            r.try_get_64().ok_or(VariantWireError::UnexpectedEof)? as i64,
        )),
        types::FLOAT => Ok(VariantWireValue::Float(f64::from_bits(
            r.try_get_64().ok_or(VariantWireError::UnexpectedEof)?,
        ))),
        types::STRING | types::STRING_NAME => {
            let s = read_string(r)?;
            Ok(VariantWireValue::Text(s))
        }
        types::VECTOR2 => Ok(VariantWireValue::Vector2([read_f64(r)?, read_f64(r)?])),
        types::VECTOR2I => Ok(VariantWireValue::Vector2i([read_i32(r)?, read_i32(r)?])),
        types::RECT2 => Ok(VariantWireValue::Rect2([
            read_f64(r)?,
            read_f64(r)?,
            read_f64(r)?,
            read_f64(r)?,
        ])),
        types::RECT2I => Ok(VariantWireValue::Rect2i([
            read_i32(r)?,
            read_i32(r)?,
            read_i32(r)?,
            read_i32(r)?,
        ])),
        types::VECTOR3 => Ok(VariantWireValue::Vector3([
            read_f64(r)?,
            read_f64(r)?,
            read_f64(r)?,
        ])),
        types::VECTOR3I => Ok(VariantWireValue::Vector3i([
            read_i32(r)?,
            read_i32(r)?,
            read_i32(r)?,
        ])),
        types::VECTOR4 => Ok(VariantWireValue::Vector4([
            read_f64(r)?,
            read_f64(r)?,
            read_f64(r)?,
            read_f64(r)?,
        ])),
        types::VECTOR4I => Ok(VariantWireValue::Vector4i([
            read_i32(r)?,
            read_i32(r)?,
            read_i32(r)?,
            read_i32(r)?,
        ])),
        types::PLANE => Ok(VariantWireValue::Plane([
            read_f64(r)?,
            read_f64(r)?,
            read_f64(r)?,
            read_f64(r)?,
        ])),
        types::QUATERNION => Ok(VariantWireValue::Quaternion([
            read_f64(r)?,
            read_f64(r)?,
            read_f64(r)?,
            read_f64(r)?,
        ])),
        types::AABB => Ok(VariantWireValue::Aabb([
            read_f64(r)?,
            read_f64(r)?,
            read_f64(r)?,
            read_f64(r)?,
            read_f64(r)?,
            read_f64(r)?,
        ])),
        types::COLOR => Ok(VariantWireValue::Color([
            read_f32(r)?,
            read_f32(r)?,
            read_f32(r)?,
            read_f32(r)?,
        ])),
        types::DICTIONARY => {
            let count = read_count(r)?;
            let mut pairs = Vec::new();
            for _ in 0..count {
                let key = decode_at_depth(r, max_depth, depth + 1)?;
                let value = decode_at_depth(r, max_depth, depth + 1)?;
                pairs.push((key, value));
            }
            Ok(VariantWireValue::Dictionary(pairs))
        }
        types::ARRAY => {
            let count = read_count(r)?;
            let mut items = Vec::new();
            for _ in 0..count {
                items.push(decode_at_depth(r, max_depth, depth + 1)?);
            }
            Ok(VariantWireValue::Array(items))
        }
        types::PACKED_BYTE_ARRAY => {
            let bytes = read_padded_bytes(r)?;
            Ok(VariantWireValue::ByteArray(bytes))
        }
        types::PACKED_INT32_ARRAY => {
            let count = read_count(r)?;
            let mut items = Vec::with_capacity(count.min(4096));
            for _ in 0..count {
                items.push(read_i32(r)?);
            }
            Ok(VariantWireValue::Int32Array(items))
        }
        types::PACKED_INT64_ARRAY => {
            let count = read_count(r)?;
            let mut items = Vec::with_capacity(count.min(4096));
            for _ in 0..count {
                items.push(r.try_get_64().ok_or(VariantWireError::UnexpectedEof)? as i64);
            }
            Ok(VariantWireValue::Int64Array(items))
        }
        types::PACKED_FLOAT32_ARRAY => {
            let count = read_count(r)?;
            let mut items = Vec::with_capacity(count.min(4096));
            for _ in 0..count {
                items.push(read_f32(r)?);
            }
            Ok(VariantWireValue::Float32Array(items))
        }
        types::PACKED_FLOAT64_ARRAY => {
            let count = read_count(r)?;
            let mut items = Vec::with_capacity(count.min(4096));
            for _ in 0..count {
                items.push(read_f64(r)?);
            }
            Ok(VariantWireValue::Float64Array(items))
        }
        types::PACKED_STRING_ARRAY => {
            let count = read_count(r)?;
            let mut items = Vec::new();
            for _ in 0..count {
                items.push(read_string(r)?);
            }
            Ok(VariantWireValue::StringArray(items))
        }
        types::PACKED_VECTOR2_ARRAY => {
            let count = read_count(r)?;
            let mut items = Vec::new();
            for _ in 0..count {
                items.push([read_f64(r)?, read_f64(r)?]);
            }
            Ok(VariantWireValue::Vector2Array(items))
        }
        types::PACKED_VECTOR3_ARRAY => {
            let count = read_count(r)?;
            let mut items = Vec::new();
            for _ in 0..count {
                items.push([read_f64(r)?, read_f64(r)?, read_f64(r)?]);
            }
            Ok(VariantWireValue::Vector3Array(items))
        }
        types::PACKED_COLOR_ARRAY => {
            let count = read_count(r)?;
            let mut items = Vec::new();
            for _ in 0..count {
                items.push([read_f32(r)?, read_f32(r)?, read_f32(r)?, read_f32(r)?]);
            }
            Ok(VariantWireValue::ColorArray(items))
        }
        other => Err(VariantWireError::UnsupportedType(other)),
    }
}

fn read_f64(r: &mut MemoryReader<'_>) -> Result<f64, VariantWireError> {
    Ok(f64::from_bits(
        r.try_get_64().ok_or(VariantWireError::UnexpectedEof)?,
    ))
}

fn read_f32(r: &mut MemoryReader<'_>) -> Result<f32, VariantWireError> {
    Ok(f32::from_bits(
        r.try_get_32().ok_or(VariantWireError::UnexpectedEof)?,
    ))
}

fn read_i32(r: &mut MemoryReader<'_>) -> Result<i32, VariantWireError> {
    Ok(r.try_get_32().ok_or(VariantWireError::UnexpectedEof)? as i32)
}

fn read_count(r: &mut MemoryReader<'_>) -> Result<usize, VariantWireError> {
    let count = r.try_get_32().ok_or(VariantWireError::UnexpectedEof)? as usize;
    // A count can never exceed the remaining bytes (every element costs at
    // least 4 bytes); reject early so `Vec::with_capacity` stays sane.
    let remaining = r.remaining();
    if count > remaining / 4 {
        return Err(VariantWireError::LengthOverflow);
    }
    Ok(count)
}

fn read_string(r: &mut MemoryReader<'_>) -> Result<String, VariantWireError> {
    let len = r.try_get_32().ok_or(VariantWireError::UnexpectedEof)? as usize;
    let remaining = r.remaining();
    if len > remaining {
        return Err(VariantWireError::LengthOverflow);
    }
    let bytes = r.try_take(len).ok_or(VariantWireError::UnexpectedEof)?;
    let pad = len % 4;
    if pad != 0 {
        r.try_take(4 - pad).ok_or(VariantWireError::UnexpectedEof)?;
    }
    String::from_utf8(bytes.to_vec()).map_err(|_| VariantWireError::Invalid)
}

fn read_padded_bytes(r: &mut MemoryReader<'_>) -> Result<Vec<u8>, VariantWireError> {
    let len = r.try_get_32().ok_or(VariantWireError::UnexpectedEof)? as usize;
    let remaining = r.remaining();
    if len > remaining {
        return Err(VariantWireError::LengthOverflow);
    }
    let bytes = r
        .try_take(len)
        .map(<[u8]>::to_vec)
        .ok_or(VariantWireError::UnexpectedEof)?;
    let pad = len % 4;
    if pad != 0 {
        r.try_take(4 - pad).ok_or(VariantWireError::UnexpectedEof)?;
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(value: VariantWireValue) -> VariantWireValue {
        let mut bytes = Vec::new();
        encode_variant(&value, &mut bytes);
        let mut r = MemoryReader::little(&bytes);
        let decoded = decode_variant(&mut r, 64).expect("decode");
        assert_eq!(r.position(), bytes.len(), "must consume exactly its bytes");
        decoded
    }

    #[test]
    fn scalars_round_trip() {
        assert_eq!(round_trip(VariantWireValue::Nil), VariantWireValue::Nil);
        assert_eq!(
            round_trip(VariantWireValue::Bool(true)),
            VariantWireValue::Bool(true)
        );
        assert_eq!(
            round_trip(VariantWireValue::Int(-42)),
            VariantWireValue::Int(-42)
        );
        assert_eq!(
            round_trip(VariantWireValue::Float(2.5)),
            VariantWireValue::Float(2.5)
        );
        assert_eq!(
            round_trip(VariantWireValue::Text("héllo".into())),
            VariantWireValue::Text("héllo".into())
        );
        // 5 bytes -> padded to 8: encoding length check.
        let mut bytes = Vec::new();
        encode_variant(&VariantWireValue::Text("12345".into()), &mut bytes);
        assert_eq!(bytes.len(), 4 + 4 + 8);
    }

    #[test]
    fn vectors_and_containers_round_trip() {
        assert_eq!(
            round_trip(VariantWireValue::Vector3([1.0, -2.0, 3.5])),
            VariantWireValue::Vector3([1.0, -2.0, 3.5])
        );
        assert_eq!(
            round_trip(VariantWireValue::Color([0.5, 0.25, 0.75, 1.0])),
            VariantWireValue::Color([0.5, 0.25, 0.75, 1.0])
        );
        let dict = VariantWireValue::Dictionary(vec![
            (VariantWireValue::Text("a".into()), VariantWireValue::Int(1)),
            (
                VariantWireValue::Text("nested".into()),
                VariantWireValue::Array(vec![VariantWireValue::Bool(false)]),
            ),
        ]);
        assert_eq!(round_trip(dict.clone()), dict);
        assert_eq!(
            round_trip(VariantWireValue::ByteArray(vec![1, 2, 3, 4, 5])),
            VariantWireValue::ByteArray(vec![1, 2, 3, 4, 5])
        );
        assert_eq!(
            round_trip(VariantWireValue::StringArray(vec!["x".into(), "yy".into()])),
            VariantWireValue::StringArray(vec!["x".into(), "yy".into()])
        );
    }

    #[test]
    fn godot_encoded_bytes_decode() {
        // Hand-built Godot wire bytes: int 7 (header 2, i64 7) followed by a
        // string "ok" (header 4, len 2, "ok", 2 pad bytes).
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&7i64.to_le_bytes());
        bytes.extend_from_slice(&4u32.to_le_bytes());
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(b"ok");
        bytes.extend_from_slice(&[0, 0]);
        let mut r = MemoryReader::little(&bytes);
        assert_eq!(
            decode_variant(&mut r, 64).unwrap(),
            VariantWireValue::Int(7)
        );
        assert_eq!(
            decode_variant(&mut r, 64).unwrap(),
            VariantWireValue::Text("ok".into())
        );
        assert_eq!(r.position(), bytes.len());
    }

    #[test]
    fn depth_bomb_is_rejected() {
        // 100 nested arrays.
        let mut bytes = Vec::new();
        for _ in 0..100 {
            bytes.extend_from_slice(&28u32.to_le_bytes());
            bytes.extend_from_slice(&1u32.to_le_bytes());
        }
        bytes.extend_from_slice(&0u32.to_le_bytes()); // nil leaf
        let mut r = MemoryReader::little(&bytes);
        assert_eq!(decode_variant(&mut r, 64), Err(VariantWireError::TooDeep));
    }

    #[test]
    fn count_bomb_is_rejected() {
        // Dictionary claiming 1 billion pairs with no payload.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&27u32.to_le_bytes());
        bytes.extend_from_slice(&1_000_000_000u32.to_le_bytes());
        let mut r = MemoryReader::little(&bytes);
        assert_eq!(
            decode_variant(&mut r, 64),
            Err(VariantWireError::LengthOverflow)
        );
    }

    #[test]
    fn object_flag_is_foreign_not_corrupt() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(types::OBJECT | (ENCODE_FLAG_OBJECT_AS_ID << 16)).to_le_bytes());
        bytes.extend_from_slice(&42u64.to_le_bytes());
        let mut r = MemoryReader::little(&bytes);
        assert_eq!(
            decode_variant(&mut r, 64),
            Err(VariantWireError::UnsupportedType(types::OBJECT))
        );
    }

    #[test]
    fn unsupported_types_report_their_id() {
        for type_id in [
            types::TRANSFORM2D,
            types::NODE_PATH,
            types::RID,
            types::OBJECT,
            types::CALLABLE,
        ] {
            let bytes = type_id.to_le_bytes().to_vec();
            let mut r = MemoryReader::little(&bytes);
            assert_eq!(
                decode_variant(&mut r, 64),
                Err(VariantWireError::UnsupportedType(type_id))
            );
        }
    }
}
