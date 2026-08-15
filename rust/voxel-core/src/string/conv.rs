//! String conversion helpers (number ↔ bytes).
//!
//! Ported from `util/string/conv.{h,cpp}`. The C++ versions write into a
//! caller-provided `Span<char>` buffer (mirroring `std::to_chars`/`snprintf`
//! semantics) and return the number of bytes written. The Rust port keeps the
//! same buffer-writing API for parity. Float output follows the relevant
//! `%.*g` rules used by the C++ `snprintf` path.
//!
//! Not ported: the `#ifndef USE_STD` hand-rolled digit loops (the C++ build
//! always defines `USE_STD`, so the `std::to_chars` path is the reference).

/// Compile-time `ceil(log10(num))` for positive integers. Matches
/// `conv_detail::log10ceil`. Used to size float-formatting buffers.
#[doc(hidden)]
pub const fn log10ceil(num: u64) -> u32 {
    if num < 10 {
        1
    } else {
        1 + log10ceil(num / 10)
    }
}

/// Max chars a `%g`-formatted `f32` can produce (incl. sign/exponent/dot).
/// Matches `max_float_chars_general<f32>`:
/// `4 + max_digits10 + max(2, log10ceil(max_exponent10))`.
/// `f32`: max_digits10 = 9, max_exponent10 = 38 → log10ceil(38) = 2.
pub const MAX_FLOAT32_CHARS_GENERAL: usize = 4 + 9 + max_u32(2, log10ceil(38));

/// Max chars a `%g`-formatted `f64` can produce. Matches `max_float_chars_general<f64>`.
/// `f64`: max_digits10 = 17, max_exponent10 = 308 → log10ceil(308) = 3.
pub const MAX_FLOAT64_CHARS_GENERAL: usize = 4 + 17 + max_u32(2, log10ceil(308));

/// Max base-10 chars for an `i32` (incl. `-`): `"-2147483648"` = 11.
/// Matches `MAX_INT32_CHAR_COUNT_BASE10`.
pub const MAX_INT32_CHAR_COUNT_BASE10: usize = 11;

/// Max base-10 chars for an `i64` (incl. `-`): `"-9223372036854775808"` = 20.
/// Matches `MAX_INT64_CHAR_COUNT_BASE10`.
pub const MAX_INT64_CHAR_COUNT_BASE10: usize = 20;

const fn max_u32(a: u32, b: u32) -> usize {
    if a > b {
        a as usize
    } else {
        b as usize
    }
}

/// Write `x` as base-10 ASCII into `out` (no null terminator). Returns the
/// number of bytes written. `out` must be at least large enough to hold the
/// result; if too small, writes nothing and returns 0 (and panics in debug).
/// Matches `int32_to_string_base10`.
pub fn int32_to_string_base10(x: i32, out: &mut [u8]) -> usize {
    let s = format!("{x}");
    let n = s.len();
    debug_assert!(
        n <= out.len(),
        "int32_to_string_base10: buffer too small ({}, need {n})",
        out.len()
    );
    if n > out.len() {
        return 0;
    }
    out[..n].copy_from_slice(s.as_bytes());
    n
}

/// Write `x` as base-10 ASCII into `out`. Same contract as
/// [`int32_to_string_base10`]. Matches `int64_to_string_base10`.
pub fn int64_to_string_base10(x: i64, out: &mut [u8]) -> usize {
    let s = format!("{x}");
    let n = s.len();
    debug_assert!(
        n <= out.len(),
        "int64_to_string_base10: buffer too small ({}, need {n})",
        out.len()
    );
    if n > out.len() {
        return 0;
    }
    out[..n].copy_from_slice(s.as_bytes());
    n
}

/// Write `x` using `%g`-style formatting (precision 8 significant digits) into
/// `out`. Returns bytes written (no null terminator). Matches `float32_to_string`.
pub fn float32_to_string(x: f32, out: &mut [u8]) -> usize {
    float_to_string_general(x, out, 8)
}

/// Write `x` using `%g`-style formatting (precision 16 significant digits) into
/// `out`. Matches `float64_to_string`.
pub fn float64_to_string(x: f64, out: &mut [u8]) -> usize {
    float_to_string_general(x, out, 16)
}

/// `%g`-style general formatting into a byte buffer.
fn float_to_string_general(x: impl Into<f64> + Copy, out: &mut [u8], precision: usize) -> usize {
    let x = x.into();
    let s = format_general_float(x, precision);
    let n = s.len();
    if n > out.len() {
        let written = out.len();
        out.copy_from_slice(&s.as_bytes()[..written]);
        debug_assert!(
            n <= out.len(),
            "float_to_string: buffer too small ({}, need {n})",
            out.len()
        );
        return written; // mirror C++ clamping on overflow
    }
    out[..n].copy_from_slice(s.as_bytes());
    n
}

fn format_general_float(x: f64, precision: usize) -> String {
    debug_assert!(precision > 0);
    if x.is_nan() {
        return "nan".to_string();
    }
    if x.is_infinite() {
        return if x.is_sign_negative() { "-inf" } else { "inf" }.to_string();
    }
    if x == 0.0 {
        return if x.is_sign_negative() {
            "-0".to_string()
        } else {
            "0".to_string()
        };
    }

    let sign = if x.is_sign_negative() { "-" } else { "" };
    let abs = x.abs();
    let exponent = abs.log10().floor() as i32;
    if exponent < -4 || exponent >= precision as i32 {
        let body = format!("{:.*e}", precision - 1, abs);
        let (mantissa, exp) = body
            .split_once('e')
            .expect("scientific formatting contains exponent");
        let mantissa = trim_float_suffix(mantissa);
        let exp_value: i32 = exp.parse().expect("scientific exponent parses");
        format!("{sign}{mantissa}e{exp_value:+03}")
    } else {
        let decimals = (precision as i32 - exponent - 1).max(0) as usize;
        let body = format!("{abs:.decimals$}");
        format!("{sign}{}", trim_float_suffix(&body))
    }
}

fn trim_float_suffix(s: &str) -> &str {
    let s = s.trim_end_matches('0');
    s.strip_suffix('.').unwrap_or(s)
}

/// Parse a base-10 integer prefix of `s`. Returns the number of bytes consumed
/// (0 if nothing parsed, or `None` on overflow/invalid). Matches
/// `string_base10_to_int32` (which returns -1 on failure).
///
/// Stops at the first non-digit (like C++ `from_chars`). A leading `-` is
/// accepted.
pub fn string_base10_to_int32(s: &str) -> Option<(usize, i32)> {
    let bytes = s.as_bytes();
    let mut pos = 0usize;
    let negative = bytes.first() == Some(&b'-');
    if negative {
        pos += 1;
    }
    let start = pos;
    let mut acc: i64 = 0;
    while pos < bytes.len() {
        let c = bytes[pos];
        if c.is_ascii_digit() {
            acc = acc * 10 + (c - b'0') as i64;
            if negative && -acc < i32::MIN as i64 {
                return None;
            }
            if !negative && acc > i32::MAX as i64 {
                return None;
            }
            pos += 1;
        } else {
            break;
        }
    }
    if pos == start {
        // No digits consumed (empty or just a sign).
        return None;
    }
    let v = if negative { -acc as i32 } else { acc as i32 };
    Some((pos, v))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log10ceil_values() {
        assert_eq!(log10ceil(0), 1);
        assert_eq!(log10ceil(9), 1);
        assert_eq!(log10ceil(10), 2);
        assert_eq!(log10ceil(99), 2);
        assert_eq!(log10ceil(100), 3);
        assert_eq!(log10ceil(308), 3);
    }

    #[test]
    fn buffer_size_constants() {
        // f32: 4 + 9 + max(2, log10ceil(38)=2) = 15
        assert_eq!(MAX_FLOAT32_CHARS_GENERAL, 15);
        // f64: 4 + 17 + max(2, log10ceil(308)=3) = 24
        assert_eq!(MAX_FLOAT64_CHARS_GENERAL, 24);
        assert_eq!(MAX_INT32_CHAR_COUNT_BASE10, 11);
        assert_eq!(MAX_INT64_CHAR_COUNT_BASE10, 20);
    }

    #[test]
    fn int32_to_string_roundtrip() {
        let mut buf = [0u8; MAX_INT32_CHAR_COUNT_BASE10];
        let n = int32_to_string_base10(-2147483647, &mut buf);
        assert_eq!(&buf[..n], b"-2147483647");

        let n = int32_to_string_base10(0, &mut buf);
        assert_eq!(&buf[..n], b"0");

        let n = int32_to_string_base10(42, &mut buf);
        assert_eq!(&buf[..n], b"42");
    }

    #[test]
    fn int64_to_string_roundtrip() {
        let mut buf = [0u8; MAX_INT64_CHAR_COUNT_BASE10];
        let n = int64_to_string_base10(-9223372036854775807i64, &mut buf);
        assert_eq!(&buf[..n], b"-9223372036854775807");
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "buffer too small")]
    fn int_to_string_buffer_too_small_panics_in_debug() {
        // Mirrors C++ `ZN_ASSERT(s.size() >= ...)`: an undersized buffer is a
        // programmer error, caught by the debug assert rather than silently
        // truncating. In release builds the assert is compiled out and the write
        // is skipped (returns 0), but tests run in debug.
        let mut tiny = [0u8; 2];
        let _ = int32_to_string_base10(123456, &mut tiny);
    }

    #[cfg(not(debug_assertions))]
    #[test]
    fn int_to_string_buffer_too_small_returns_zero_in_release() {
        let mut tiny = [0u8; 2];
        assert_eq!(int32_to_string_base10(123456, &mut tiny), 0);
    }

    #[test]
    fn float32_to_string_basic() {
        let mut buf = [0u8; MAX_FLOAT32_CHARS_GENERAL];
        let n = float32_to_string(1.0, &mut buf);
        assert_eq!(&buf[..n], b"1");
        let n = float32_to_string(1.5, &mut buf);
        assert_eq!(&buf[..n], b"1.5");
        let n = float32_to_string(-2.5, &mut buf);
        assert_eq!(&buf[..n], b"-2.5");
        let n = float32_to_string(123456789.0, &mut buf);
        assert_eq!(&buf[..n], b"1.2345679e+08");
        let n = float32_to_string(0.000012345678, &mut buf);
        assert_eq!(&buf[..n], b"1.2345678e-05");
    }

    #[test]
    fn float_to_string_tiny_buffer_gets_truncated_prefix() {
        let mut buf = *b"xxxx";
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            float32_to_string(12345.0, &mut buf)
        }));

        #[cfg(debug_assertions)]
        assert!(result.is_err());
        #[cfg(not(debug_assertions))]
        assert_eq!(result.unwrap(), 4);

        assert_eq!(&buf, b"1234");
    }

    #[test]
    fn float64_to_string_uses_16_significant_digits() {
        let mut buf = [0u8; MAX_FLOAT64_CHARS_GENERAL];
        let n = float64_to_string(1.2345678901234567, &mut buf);
        assert_eq!(&buf[..n], b"1.234567890123457");
    }

    #[test]
    fn string_to_int32_parses_prefix() {
        // Full parse.
        assert_eq!(string_base10_to_int32("42"), Some((2, 42)));
        // Stops at non-digit, returns consumed count.
        assert_eq!(string_base10_to_int32("123abc"), Some((3, 123)));
        // Negative.
        assert_eq!(string_base10_to_int32("-7"), Some((2, -7)));
        // No digit at the start is a parse failure (like from_chars).
        assert_eq!(string_base10_to_int32(" 5"), None);
        assert_eq!(string_base10_to_int32("abc"), None);
        assert_eq!(string_base10_to_int32(""), None);
        assert_eq!(string_base10_to_int32("-"), None);
    }

    #[test]
    fn string_to_int32_overflow_is_none() {
        // Value exceeding i32 range.
        assert_eq!(string_base10_to_int32("9999999999"), None);
        // Just the boundary is fine.
        assert_eq!(string_base10_to_int32("2147483647"), Some((10, 2147483647)));
    }
}
