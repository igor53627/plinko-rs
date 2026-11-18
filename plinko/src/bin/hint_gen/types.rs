use clap::Parser;
use std::path::PathBuf;

/// Record size of a v3 `database.bin` (`plinko::schema40`), the default of `--record-size`.
pub const DEFAULT_RECORD_SIZE: usize = plinko::schema40::ENTRY_SIZE;

#[derive(Parser, Debug)]
#[command(author, version, about = "Plinko PIR Hint Generator (Paper-compliant)", long_about = None)]
/// Command-line arguments for the hint generator binary.
pub struct Args {
    /// Path to the flat binary database file produced by the extraction pipeline.
    #[arg(short, long, default_value = "/mnt/plinko/data/database.bin")]
    pub db_path: PathBuf,

    /// Bytes of one database record. The file is an array of records of this
    /// size, and a hint parity is the XOR of whole records.
    #[arg(short, long, default_value_t = DEFAULT_RECORD_SIZE)]
    pub record_size: usize,

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
    /// The process list shows this value; prefer `--seed-file`.
    #[arg(long)]
    pub seed: Option<String>,

    /// File that holds the master seed as 64 hex characters.
    #[arg(long, conflicts_with = "seed")]
    pub seed_file: Option<PathBuf>,

    /// Write the hint set to this file (format: `plinko::hints`).
    /// Requires `--seed` or `--seed-file`; the file does not contain the seed.
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Print the master seed to stdout after generation/parsing.
    #[arg(long)]
    pub print_seed: bool,

    /// Enable the constant-time path (experimental, `tee` feature). On one
    /// thread, the sequence of operations depends only on public values. The
    /// memory accesses at the secret hint index, and the time that follows
    /// from them, are not hidden (docs/constant_time_mode.md).
    #[cfg(feature = "tee")]
    #[arg(long)]
    pub constant_time: bool,

    /// Worker threads of HintInit. The default is the number of available cores,
    /// and 1 with `--constant-time`. The hints do not depend on this value.
    #[arg(long)]
    pub threads: Option<usize>,
}

impl Args {
    /// `true` if the run uses the constant-time path. A build without the
    /// `tee` feature has no such path.
    pub fn constant_time(&self) -> bool {
        #[cfg(feature = "tee")]
        {
            self.constant_time
        }
        #[cfg(not(feature = "tee"))]
        {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_record_size_is_the_v3_record() {
        let args: Args = Args::parse_from(["plinko_hints"]);
        assert_eq!(args.record_size, 40);
        let args: Args = Args::parse_from(["plinko_hints", "--record-size", "256"]);
        assert_eq!(args.record_size, 256);
    }
}
