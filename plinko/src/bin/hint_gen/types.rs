use clap::Parser;
use std::path::PathBuf;

/// On-disk v3 record size (`database.bin` stride).
pub use plinko::schema40::ENTRY_SIZE as DB_ENTRY_SIZE;

/// Bytes XOR'd into hint parity (first 32 bytes of each v3 entry).
pub const PARITY_WORD_SIZE: usize = 32;

/// Alias kept for older call sites; prefer `PARITY_WORD_SIZE` / `DB_ENTRY_SIZE`.
#[allow(dead_code)]
pub const WORD_SIZE: usize = PARITY_WORD_SIZE;

#[derive(Parser, Debug)]
#[command(author, version, about = "Plinko PIR Hint Generator (Paper-compliant)", long_about = None)]
/// Command-line arguments for the hint generator binary.
pub struct Args {
    /// Path to the flat binary database file produced by the extraction pipeline.
    #[arg(short, long, default_value = "/mnt/plinko/data/database.bin")]
    pub db_path: PathBuf,

    /// Security parameter (number of regular hints per block).
    #[arg(short, long, default_value = "128")]
    pub lambda: usize,

    /// Number of backup hints; defaults to `lambda * w` if omitted.
    #[arg(short, long)]
    pub backup_hints: Option<usize>,

    /// Entries per block (`w`), a power of two; defaults to the smallest power of two >= `sqrt(N)`.
    #[arg(short, long)]
    pub entries_per_block: Option<usize>,

    /// Debug flag: drop tail entries instead of padding (violates security assumptions).
    #[arg(long, default_value = "false", hide = true)]
    pub allow_truncation: bool,

    /// Deterministic master seed as a 64-char hex string; random if omitted.
    #[arg(long)]
    pub seed: Option<String>,

    /// Print the master seed to stdout after generation/parsing.
    #[arg(long)]
    pub print_seed: bool,

    /// Enable constant-time processing path (data-independent memory access).
    #[arg(long)]
    pub constant_time: bool,
}

/// Load the 32-byte parity word for logical entry `index` in a v3 `database.bin` mmap.
#[inline]
pub fn parity_word_at(db_bytes: &[u8], index: usize) -> [u8; PARITY_WORD_SIZE] {
    let offset = index * DB_ENTRY_SIZE;
    db_bytes[offset..offset + PARITY_WORD_SIZE]
        .try_into()
        .expect("v3 entry must be at least 32 bytes")
}

/// A regular PIR hint: the XOR parity of database entries in a pseudorandom
/// subset of `c/2 + 1` blocks.
pub struct RegularHint {
    /// Cutoff that defines the block subset (see `plinko::subsets`).
    /// `INVALID_CUTOFF` marks a hint with no valid subset; its parity stays zero.
    pub cutoff: u32,
    /// XOR accumulator over all entries in the subset.
    pub parity: [u8; 32],
}

/// A backup hint used for puncturing: splits blocks into an "in" subset of
/// `c/2` blocks and its complement "out", maintaining separate XOR parities
/// for each half.
pub struct BackupHint {
    /// Cutoff that defines the in/out partition (see `plinko::subsets`).
    /// `INVALID_CUTOFF` marks a hint with no valid partition; its parities stay zero.
    pub cutoff: u32,
    /// XOR parity over entries whose block is *in* the subset.
    pub parity_in: [u8; 32],
    /// XOR parity over entries whose block is *outside* the subset.
    pub parity_out: [u8; 32],
}
