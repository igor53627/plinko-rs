use clap::Parser;
use std::path::PathBuf;

/// Database word size in bytes (one 256-bit entry).
pub const WORD_SIZE: usize = 32;

/// Domain-separation label used when deriving seeds for regular hint subsets.
pub const SEED_LABEL_REGULAR: &[u8] = b"plinko_regular_subset";
/// Domain-separation label used when deriving seeds for backup hint subsets.
pub const SEED_LABEL_BACKUP: &[u8] = b"plinko_backup_subset";

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

    /// Entries per block (`w`); defaults to `sqrt(N)` if omitted.
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

/// A regular PIR hint: the XOR parity of database entries in a pseudorandom
/// subset of blocks, keyed by `subset_seed`.
pub struct RegularHint {
    /// Seed that deterministically defines which blocks belong to this hint's subset.
    #[allow(dead_code)]
    pub subset_seed: [u8; 32],
    /// XOR accumulator over all entries in the subset.
    pub parity: [u8; 32],
}

/// A backup hint used for puncturing: splits blocks into an "in" subset and
/// its complement "out", maintaining separate XOR parities for each half.
pub struct BackupHint {
    /// Seed that deterministically defines the in/out partition.
    #[allow(dead_code)]
    pub subset_seed: [u8; 32],
    /// XOR parity over entries whose block is *in* the subset.
    pub parity_in: [u8; 32],
    /// XOR parity over entries whose block is *outside* the subset.
    pub parity_out: [u8; 32],
}
