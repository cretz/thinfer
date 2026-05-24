//! Execution policy types. Today: `ResidencyBudget`. `ExecutionPolicy`
//! presets (orig-plan) land here when scheduler shape settles.

/// Total memory caps the engine must stay under. Both fields are
/// *full-picture* totals: VRAM covers weights + workspace + staging, RAM
/// covers all engine-controlled host buffers (mmap page cache is OS-managed
/// and not counted). The residency manager evicts weights to keep the total
/// under `vram_bytes`, using *current* live workspace + staging as the
/// dynamic floor (no static reserve carve-out).
#[derive(Clone, Copy, Debug)]
pub struct ResidencyBudget {
    pub ram_bytes: u64,
    pub vram_bytes: u64,
}

#[derive(Debug, PartialEq, Eq)]
pub enum ByteSizeError {
    Empty,
    InvalidNumber,
    UnknownSuffix,
    Overflow,
}

impl core::fmt::Display for ByteSizeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Empty => f.write_str("empty size string"),
            Self::InvalidNumber => f.write_str("invalid number"),
            Self::UnknownSuffix => {
                f.write_str("unknown size suffix (expected K/M/G/T or KiB/MiB/...)")
            }
            Self::Overflow => f.write_str("size overflows u64"),
        }
    }
}

/// Parse byte sizes: raw bytes, or with suffix `K`/`M`/`G`/`T` (binary, 1024-based)
/// or `KiB`/`MiB`/`GiB`/`TiB`. Case-insensitive. Whitespace between the number
/// and suffix is allowed. Decimal points are not supported - this is a budget
/// knob, not a UX-formatter.
pub fn parse_bytes(s: &str) -> Result<u64, ByteSizeError> {
    let s = s.trim();
    if s.is_empty() {
        return Err(ByteSizeError::Empty);
    }
    let split = s.find(|c: char| !c.is_ascii_digit()).unwrap_or(s.len());
    let (num, suffix) = s.split_at(split);
    let num: u64 = num.parse().map_err(|_| ByteSizeError::InvalidNumber)?;
    let suffix = suffix.trim().to_ascii_lowercase();
    let mult: u64 = match suffix.as_str() {
        "" | "b" => 1,
        "k" | "kib" | "kb" => 1024,
        "m" | "mib" | "mb" => 1024 * 1024,
        "g" | "gib" | "gb" => 1024 * 1024 * 1024,
        "t" | "tib" | "tb" => 1024u64 * 1024 * 1024 * 1024,
        _ => return Err(ByteSizeError::UnknownSuffix),
    };
    num.checked_mul(mult).ok_or(ByteSizeError::Overflow)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_suffixes() {
        assert_eq!(parse_bytes("0"), Ok(0));
        assert_eq!(parse_bytes("1024"), Ok(1024));
        assert_eq!(parse_bytes("1K"), Ok(1024));
        assert_eq!(parse_bytes("8G"), Ok(8 * 1024 * 1024 * 1024));
        assert_eq!(parse_bytes("2 GiB"), Ok(2 * 1024 * 1024 * 1024));
        assert_eq!(parse_bytes("4mb"), Ok(4 * 1024 * 1024));
    }

    #[test]
    fn rejects_bad_input() {
        assert_eq!(parse_bytes(""), Err(ByteSizeError::Empty));
        assert_eq!(parse_bytes("abc"), Err(ByteSizeError::InvalidNumber));
        assert_eq!(parse_bytes("1X"), Err(ByteSizeError::UnknownSuffix));
        assert!(parse_bytes("99999999999999999999G").is_err());
    }
}
