//! Shared bounds for decoding untrusted voxel formats.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeLimits {
    pub max_bytes: usize,
    pub max_block_voxels: u64,
    pub max_region_blocks: usize,
    pub max_vox_models: usize,
    pub max_vox_total_voxels: u64,
    pub max_vox_nodes: usize,
    pub max_string_bytes: usize,
    /// Maximum nesting depth of a decoded Godot Variant (R7 wide). Blocks
    /// recursion bombs; 64 comfortably exceeds any real metadata payload.
    pub max_variant_depth: u32,
}

impl Default for DecodeLimits {
    fn default() -> Self {
        Self {
            max_bytes: 64 * 1024 * 1024,
            max_block_voxels: 256 * 256 * 256,
            max_region_blocks: 255 * 255 * 255,
            max_vox_models: 4096,
            max_vox_total_voxels: 64 * 1024 * 1024,
            max_vox_nodes: 65_536,
            max_string_bytes: 4096,
            max_variant_depth: 64,
        }
    }
}

impl DecodeLimits {
    pub const fn trusted() -> Self {
        Self {
            max_bytes: usize::MAX,
            max_block_voxels: u64::MAX,
            max_region_blocks: usize::MAX,
            max_vox_models: usize::MAX,
            max_vox_total_voxels: u64::MAX,
            max_vox_nodes: usize::MAX,
            max_string_bytes: usize::MAX,
            max_variant_depth: u32::MAX,
        }
    }

    pub fn check_bytes(self, label: &'static str, requested: usize) -> DecodeLimitResult {
        check_u128(label, requested as u128, self.max_bytes as u128)
    }

    pub fn check_block_voxels(self, requested: u64) -> DecodeLimitResult {
        check_u128(
            "block voxels",
            requested as u128,
            self.max_block_voxels as u128,
        )
    }

    pub fn check_region_blocks(self, requested: usize) -> DecodeLimitResult {
        check_u128(
            "region blocks",
            requested as u128,
            self.max_region_blocks as u128,
        )
    }

    pub fn check_vox_models(self, requested: usize) -> DecodeLimitResult {
        check_u128("vox models", requested as u128, self.max_vox_models as u128)
    }

    pub fn check_vox_total_voxels(self, requested: u64) -> DecodeLimitResult {
        check_u128(
            "vox total voxels",
            requested as u128,
            self.max_vox_total_voxels as u128,
        )
    }

    pub fn check_vox_nodes(self, requested: usize) -> DecodeLimitResult {
        check_u128("vox nodes", requested as u128, self.max_vox_nodes as u128)
    }

    pub fn check_string_bytes(self, requested: usize) -> DecodeLimitResult {
        check_u128(
            "string bytes",
            requested as u128,
            self.max_string_bytes as u128,
        )
    }
}

type DecodeLimitResult = Result<(), DecodeLimitError>;

fn check_u128(label: &'static str, requested: u128, limit: u128) -> DecodeLimitResult {
    if requested > limit {
        return Err(DecodeLimitError::LimitExceeded {
            label,
            requested,
            limit,
        });
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodeLimitError {
    LimitExceeded {
        label: &'static str,
        requested: u128,
        limit: u128,
    },
    AllocationFailed {
        label: &'static str,
        requested: usize,
    },
}

impl std::fmt::Display for DecodeLimitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LimitExceeded {
                label,
                requested,
                limit,
            } => write!(
                f,
                "{label} limit exceeded: requested {requested}, limit {limit}"
            ),
            Self::AllocationFailed { label, requested } => {
                write!(f, "{label} allocation failed for {requested} bytes")
            }
        }
    }
}

impl std::error::Error for DecodeLimitError {}

pub fn reserve_vec<T>(
    vec: &mut Vec<T>,
    label: &'static str,
    additional: usize,
) -> Result<(), DecodeLimitError> {
    vec.try_reserve(additional)
        .map_err(|_| DecodeLimitError::AllocationFailed {
            label,
            requested: additional.saturating_mul(std::mem::size_of::<T>()),
        })
}

#[cfg(test)]
mod tests {
    use super::{DecodeLimitError, DecodeLimits};

    #[test]
    fn default_limits_allow_reasonable_small_inputs() {
        let limits = DecodeLimits::default();

        assert!(limits.check_bytes("raw", 1024).is_ok());
        assert!(limits.check_block_voxels(16 * 16 * 16).is_ok());
        assert!(limits.check_vox_models(1).is_ok());
    }

    #[test]
    fn byte_limit_reports_requested_and_limit() {
        let limits = DecodeLimits {
            max_bytes: 4,
            ..DecodeLimits::default()
        };

        assert_eq!(
            limits.check_bytes("payload", 5),
            Err(DecodeLimitError::LimitExceeded {
                label: "payload",
                requested: 5,
                limit: 4,
            })
        );
    }
}
