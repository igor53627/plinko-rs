//! Driver helpers for plinko_hints binary - geometry, validation, and initialization.

use crate::hint_gen::{BackupHint, RegularHint, DB_ENTRY_SIZE};
use plinko::hints::HintSet;
use plinko::iprf::MAX_PREIMAGES;
use plinko::subsets::{backup_subset_size, regular_subset_size, SubsetPrf, INVALID_CUTOFF};
use rand::RngCore;
use std::fs::{File, OpenOptions};
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::time::Duration;
use tracing::{info, warn};

use super::Args;

/// Database geometry parameters computed from args and DB size.
pub struct Geometry {
    /// Number of real entries in the database file.
    pub n_entries: usize,
    /// Effective entry count after padding (always a multiple of `w`).
    pub n_effective: usize,
    /// Entries per block.
    pub w: usize,
    /// Number of blocks (`n_effective / w`), guaranteed even.
    pub c: usize,
    /// Number of zero-padding entries appended to reach `n_effective`.
    pub pad_entries: usize,
}

/// Hint count parameters.
pub struct HintParams {
    /// Number of regular hints (`lambda * w`).
    pub num_regular: usize,
    /// Number of backup hints (defaults to `num_regular`).
    pub num_backup: usize,
    /// `num_regular + num_backup`.
    pub total_hints: usize,
}

/// Return type for [`init_hints`]: `(regular_hints, backup_hints)`.
pub type HintInitOutput = (Vec<RegularHint>, Vec<BackupHint>);

impl HintParams {
    pub fn from_args(args: &Args, w: usize) -> Self {
        let num_regular = args.lambda * w;
        let num_backup = args.backup_hints.unwrap_or(num_regular);
        let total_hints = num_regular + num_backup;
        Self {
            num_regular,
            num_backup,
            total_hints,
        }
    }
}

/// Validate command-line arguments.
pub fn validate_args(args: &Args) -> eyre::Result<()> {
    if args.lambda == 0 {
        eyre::bail!("lambda must be >= 1");
    }
    if let Some(w) = args.entries_per_block {
        if !w.is_power_of_two() {
            eyre::bail!(
                "entries_per_block (w) must be a power of two, got {w}. \
                 The iPRF range is w, and the PMNS sampler requires a power of two."
            );
        }
    }
    if args.backup_hints == Some(0) {
        eyre::bail!("num_backup must be > 0 (backup hints are required for correctness)");
    }
    if let Some(output) = &args.output {
        if args.seed.is_none() && args.seed_file.is_none() {
            eyre::bail!(
                "--output requires --seed or --seed-file: \
                 a client cannot use a hint file without its master seed"
            );
        }
        if args.allow_truncation {
            eyre::bail!("--output cannot be used with --allow-truncation");
        }
        let parent: &Path = output.parent().unwrap_or(Path::new(""));
        if !parent.as_os_str().is_empty() && !parent.is_dir() {
            eyre::bail!("output directory {} does not exist", parent.display());
        }
    }
    Ok(())
}

/// Validate that hint parameters are within bounds for CT mode.
///
/// For Plinko with (lambda, w, q), expected preimages per offset = (lambda*w + q) / w.
/// With default q = lambda*w: expected = 2*lambda = 256 for lambda=128.
///
/// We require expected * 2 <= MAX_PREIMAGES to ensure truncation probability
/// is negligible (< 2^{-100}) via Chernoff bounds.
pub fn validate_hint_params(params: &HintParams, w: usize) -> eyre::Result<()> {
    let expected_preimages = params.total_hints.div_ceil(w);
    if expected_preimages * 2 > MAX_PREIMAGES {
        eyre::bail!(
            "Parameter configuration too dense for constant-time mode.\n\
             Expected preimages per offset ({}) exceeds MAX_PREIMAGES/2 ({}).\n\
             Reduce total_hints or increase w.",
            expected_preimages,
            MAX_PREIMAGES / 2
        );
    }
    Ok(())
}

/// Compute database geometry from file length and arguments.
pub fn compute_geometry(db_len_bytes: usize, args: &Args) -> eyre::Result<Geometry> {
    #[allow(clippy::manual_is_multiple_of)]
    if db_len_bytes % DB_ENTRY_SIZE != 0 {
        eyre::bail!(
            "DB size must be a multiple of {} bytes (v3 entry size)",
            DB_ENTRY_SIZE
        );
    }

    let n_entries = db_len_bytes / DB_ENTRY_SIZE;
    if n_entries == 0 {
        eyre::bail!("Database must contain at least one entry");
    }

    let w = args
        .entries_per_block
        .unwrap_or_else(|| plinko::protocol::default_block_size(n_entries));

    let remainder = n_entries % w;
    let (logical_n_entries, pad_entries) = if remainder == 0 {
        (n_entries, 0usize)
    } else {
        let pad = w - remainder;
        (n_entries + pad, pad)
    };

    if pad_entries > 0 {
        if args.allow_truncation {
            warn!(
                n_entries,
                w,
                tail_entries_ignored = remainder,
                "N not divisible by w, tail entries will be ignored"
            );
        } else {
            info!(
                n_entries,
                w, pad_entries, "N not divisible by w, padding with dummy entries"
            );
        }
    }

    let (mut n_effective, final_pad) = if args.allow_truncation && remainder != 0 {
        warn!("--allow-truncation is a debug flag that violates security assumptions");
        (n_entries - remainder, 0usize)
    } else {
        (logical_n_entries, pad_entries)
    };
    let mut c = n_effective / w;

    let mut pad_entries = final_pad;
    #[allow(clippy::manual_is_multiple_of)]
    if !args.allow_truncation && c % 2 != 0 {
        c += 1;
        n_effective = c * w;
        pad_entries = n_effective - n_entries;
        info!(
            old_c = c - 1,
            new_c = c,
            pad_entries,
            "Bumped c to even value"
        );
    }

    if c < 2 {
        eyre::bail!("Number of blocks (c = {}) must be at least 2.", c);
    }

    Ok(Geometry {
        n_entries,
        n_effective,
        w,
        c,
        pad_entries,
    })
}

/// Parses a master seed of 64 hex characters, with an optional `0x` prefix.
fn parse_seed_hex(text: &str, source: &str) -> eyre::Result<[u8; 32]> {
    let hex_clean: &str = text.strip_prefix("0x").unwrap_or(text);
    if hex_clean.len() != 64 || !hex_clean.is_ascii() {
        eyre::bail!("{source} must be exactly 32 bytes (64 hex chars)");
    }
    let mut seed = [0u8; 32];
    for (i, byte) in seed.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&hex_clean[2 * i..2 * i + 2], 16)
            .map_err(|_| eyre::eyre!("invalid hex in {source} at position {}", i * 2))?;
    }
    Ok(seed)
}

/// Takes the master seed from `--seed` or `--seed-file`, or generates a random seed.
pub fn parse_or_generate_seed(args: &Args) -> eyre::Result<[u8; 32]> {
    let provided: Option<[u8; 32]> = if let Some(hex_seed) = &args.seed {
        Some(parse_seed_hex(hex_seed, "--seed")?)
    } else if let Some(path) = &args.seed_file {
        let text: String = std::fs::read_to_string(path)
            .map_err(|error| eyre::eyre!("cannot read --seed-file {}: {error}", path.display()))?;
        Some(parse_seed_hex(text.trim(), "--seed-file")?)
    } else {
        None
    };

    let seed: [u8; 32] = match provided {
        Some(seed) => seed,
        None => {
            let mut seed = [0u8; 32];
            rand::thread_rng().fill_bytes(&mut seed);
            seed
        }
    };
    if args.print_seed {
        let hex: String = seed.iter().map(|b| format!("{:02x}", b)).collect();
        let origin: &str = if provided.is_some() {
            "Using provided seed"
        } else {
            "Generated seed"
        };
        eprintln!("{origin}: 0x{hex}");
    }
    Ok(seed)
}

/// Writes the hint set to `path` through a temporary file in the same directory.
///
/// The temporary file is `<path>.<pid>.tmp`. It is created exclusively, so a
/// run never writes into a file that another run or another user made. The
/// file is client-secret; on Unix it gets mode 0600.
pub fn write_hint_file(path: &Path, hints: &HintSet) -> eyre::Result<()> {
    let mut tmp_name = path.as_os_str().to_owned();
    tmp_name.push(format!(".{}.tmp", std::process::id()));
    let tmp_path: PathBuf = PathBuf::from(tmp_name);

    let mut options: OpenOptions = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let file: File = options
        .open(&tmp_path)
        .map_err(|error| eyre::eyre!("cannot create {}: {error}", tmp_path.display()))?;

    let written: eyre::Result<()> = write_and_publish(file, &tmp_path, path, hints);
    if written.is_err() {
        // Best effort: the temporary file is ours and is incomplete.
        let _ = std::fs::remove_file(&tmp_path);
    }
    written
}

fn write_and_publish(
    file: File,
    tmp_path: &Path,
    path: &Path,
    hints: &HintSet,
) -> eyre::Result<()> {
    let mut writer = BufWriter::new(file);
    hints.write_to(&mut writer)?;
    let file: File = writer
        .into_inner()
        .map_err(|error| eyre::eyre!("cannot flush {}: {error}", tmp_path.display()))?;
    file.sync_all()?;
    drop(file);

    std::fs::rename(tmp_path, path)
        .map_err(|error| eyre::eyre!("cannot move hint file to {}: {error}", path.display()))?;
    Ok(())
}

/// Initialize hint structures and compute the cutoff of each block subset.
///
/// Regular hint `j` has hint ID `j` and a subset of `c/2 + 1` blocks. Backup
/// hint `b` has hint ID `num_regular + b` and a subset of `c/2` blocks.
pub fn init_hints(subset_prf: &SubsetPrf, c: usize, params: &HintParams) -> HintInitOutput {
    let mut scratch: Vec<u32> = Vec::with_capacity(c);

    let regular_hints: Vec<RegularHint> = (0..params.num_regular)
        .map(|j| RegularHint {
            cutoff: subset_prf.cutoff(j as u64, c, regular_subset_size(c), &mut scratch),
            parity: [0u8; 32],
        })
        .collect();

    let backup_hints: Vec<BackupHint> = (0..params.num_backup)
        .map(|b| BackupHint {
            cutoff: subset_prf.cutoff(
                (params.num_regular + b) as u64,
                c,
                backup_subset_size(c),
                &mut scratch,
            ),
            parity_in: [0u8; 32],
            parity_out: [0u8; 32],
        })
        .collect();

    (regular_hints, backup_hints)
}

/// Print final results summary.
pub fn print_results(
    duration: Duration,
    file_len: usize,
    regular_hints: &[RegularHint],
    backup_hints: &[BackupHint],
    params: &HintParams,
) {
    let throughput_mb = (file_len as f64 / 1024.0 / 1024.0) / duration.as_secs_f64();

    let non_zero_regular = regular_hints
        .iter()
        .filter(|h| h.parity.iter().any(|&b| b != 0))
        .count();
    let invalid_regular = regular_hints
        .iter()
        .filter(|h| h.cutoff == INVALID_CUTOFF)
        .count();

    info!(
        duration_secs = format_args!("{:.2}", duration.as_secs_f64()),
        throughput_mb_s = format_args!("{:.2}", throughput_mb),
        non_zero_regular,
        total_regular = params.num_regular,
        invalid_regular,
        "Results"
    );

    if params.num_backup > 0 {
        let non_zero_backup_in = backup_hints
            .iter()
            .filter(|h| h.parity_in.iter().any(|&b| b != 0))
            .count();
        let non_zero_backup_out = backup_hints
            .iter()
            .filter(|h| h.parity_out.iter().any(|&b| b != 0))
            .count();
        let invalid_backup = backup_hints
            .iter()
            .filter(|h| h.cutoff == INVALID_CUTOFF)
            .count();
        info!(
            non_zero_parity_in = non_zero_backup_in,
            non_zero_parity_out = non_zero_backup_out,
            total_backup = params.num_backup,
            invalid_backup,
            "Backup hint results"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn rejects_entries_per_block_that_is_not_a_power_of_two() {
        for w in ["0", "3", "42826"] {
            let args: Args = Args::parse_from(["plinko_hints", "--entries-per-block", w]);
            let error: String = validate_args(&args)
                .expect_err("w must be rejected")
                .to_string();
            assert!(error.contains("must be a power of two"), "w={w}: {error}");
        }

        let args: Args = Args::parse_from(["plinko_hints", "--entries-per-block", "65536"]);
        assert!(validate_args(&args).is_ok());
    }

    #[test]
    fn output_requires_a_reproducible_seed() {
        let seed_hex: String = "ab".repeat(32);

        let args: Args = Args::parse_from(["plinko_hints", "--output", "hints.bin"]);
        let error: String = validate_args(&args).expect_err("no seed").to_string();
        assert!(error.contains("--seed or --seed-file"), "{error}");

        let args: Args =
            Args::parse_from(["plinko_hints", "--output", "hints.bin", "--seed", &seed_hex]);
        assert!(validate_args(&args).is_ok());

        let args: Args = Args::parse_from([
            "plinko_hints",
            "--output",
            "hints.bin",
            "--seed-file",
            "seed.hex",
        ]);
        assert!(validate_args(&args).is_ok());

        let args: Args = Args::parse_from([
            "plinko_hints",
            "--output",
            "hints.bin",
            "--seed",
            &seed_hex,
            "--allow-truncation",
        ]);
        assert!(validate_args(&args).is_err());

        let conflict = Args::try_parse_from([
            "plinko_hints",
            "--seed",
            &seed_hex,
            "--seed-file",
            "seed.hex",
        ]);
        assert!(conflict.is_err(), "--seed and --seed-file must conflict");
    }

    #[test]
    fn seed_hex_is_parsed_strictly() {
        let mut expected = [0u8; 32];
        expected[0] = 0x01;
        expected[31] = 0xFE;
        let hex: String = format!("01{}fe", "00".repeat(30));

        assert_eq!(parse_seed_hex(&hex, "--seed").expect("plain"), expected);
        assert_eq!(
            parse_seed_hex(&format!("0x{hex}"), "--seed").expect("prefixed"),
            expected
        );
        assert!(parse_seed_hex(&hex[2..], "--seed").is_err(), "too short");
        assert!(
            parse_seed_hex(&format!("{hex}00"), "--seed").is_err(),
            "too long"
        );
        assert!(
            parse_seed_hex(&format!("zz{}", &hex[2..]), "--seed").is_err(),
            "not hex"
        );
    }

    #[test]
    fn default_geometry_matches_the_protocol_client() {
        let args: Args = Args::parse_from(["plinko_hints"]);
        let geometry_of = |n_entries: usize| -> (usize, usize) {
            let geometry: Geometry =
                compute_geometry(n_entries * DB_ENTRY_SIZE, &args).expect("geometry");
            (geometry.w, geometry.c)
        };

        assert_eq!(geometry_of(1_834_095_877), (65_536, 27_988));

        // Includes databases that fill less than two blocks, and odd block counts.
        for n_entries in (1..=70).chain([1_000, 65_537, 1_834_095_877]) {
            let client = plinko::protocol::ProtocolParams::new(n_entries, None, 128, None)
                .expect("client params");
            assert_eq!(
                geometry_of(n_entries),
                (client.block_size, client.num_blocks),
                "n_entries={n_entries}"
            );
        }
    }
}
