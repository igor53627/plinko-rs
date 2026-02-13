//! Driver helpers for plinko_hints binary - geometry, validation, and initialization.

use crate::hint_gen::{
    compute_backup_blocks, compute_regular_blocks, derive_subset_seed, BackupHint, RegularHint,
    SEED_LABEL_BACKUP, SEED_LABEL_REGULAR, WORD_SIZE,
};
use plinko::iprf::{PrfKey128, MAX_PREIMAGES};
use rand::RngCore;
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

/// Return type for [`init_hints`]: `(regular_hints, regular_hint_blocks, backup_hints, backup_hint_blocks)`.
pub type HintInitOutput = (
    Vec<RegularHint>,
    Vec<Vec<usize>>,
    Vec<BackupHint>,
    Vec<Vec<usize>>,
);

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
    if args.entries_per_block == Some(0) {
        eyre::bail!("entries_per_block (w) must be > 0");
    }
    if args.backup_hints == Some(0) {
        eyre::bail!("num_backup must be > 0 (backup hints are required for correctness)");
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
    if db_len_bytes % WORD_SIZE != 0 {
        eyre::bail!("DB size must be multiple of 32 bytes");
    }

    let n_entries = db_len_bytes / WORD_SIZE;
    if n_entries == 0 {
        eyre::bail!("Database must contain at least one entry");
    }

    let default_w = (n_entries as f64).sqrt().round() as usize;
    let w = args.entries_per_block.unwrap_or(default_w);

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

    if c < 2 {
        eyre::bail!("Number of blocks (c = {}) must be at least 2.", c);
    }

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

    Ok(Geometry {
        n_entries,
        n_effective,
        w,
        c,
        pad_entries,
    })
}

/// Parse seed from hex string or generate random seed.
pub fn parse_or_generate_seed(args: &Args) -> eyre::Result<[u8; 32]> {
    if let Some(ref hex_seed) = args.seed {
        let hex_clean = hex_seed.strip_prefix("0x").unwrap_or(hex_seed);
        if hex_clean.len() != 64 {
            eyre::bail!("--seed must be exactly 32 bytes (64 hex chars)");
        }
        let mut seed = [0u8; 32];
        for (i, chunk) in hex_clean.as_bytes().chunks(2).enumerate() {
            let hex_str = std::str::from_utf8(chunk)?;
            seed[i] = u8::from_str_radix(hex_str, 16)
                .map_err(|_| eyre::eyre!("invalid hex in --seed at position {}", i * 2))?;
        }
        if args.print_seed {
            eprintln!("Using provided seed: 0x{}", hex_clean);
        }
        Ok(seed)
    } else {
        let mut seed = [0u8; 32];
        rand::thread_rng().fill_bytes(&mut seed);
        if args.print_seed {
            let hex: String = seed.iter().map(|b| format!("{:02x}", b)).collect();
            eprintln!("Generated seed: 0x{}", hex);
        }
        Ok(seed)
    }
}

/// Initialize hint structures and compute block memberships.
pub fn init_hints(master_seed: &[u8; 32], c: usize, params: &HintParams) -> HintInitOutput {
    let mut regular_hints: Vec<RegularHint> = Vec::with_capacity(params.num_regular);
    let mut regular_hint_blocks: Vec<Vec<usize>> = Vec::with_capacity(params.num_regular);

    for j in 0..params.num_regular {
        let subset_seed = derive_subset_seed(master_seed, SEED_LABEL_REGULAR, j as u64);
        let blocks = compute_regular_blocks(&subset_seed, c);
        regular_hints.push(RegularHint {
            subset_seed,
            parity: [0u8; 32],
        });
        regular_hint_blocks.push(blocks);
    }

    let mut backup_hints: Vec<BackupHint> = Vec::with_capacity(params.num_backup);
    let mut backup_hint_blocks: Vec<Vec<usize>> = Vec::with_capacity(params.num_backup);

    for j in 0..params.num_backup {
        let subset_seed = derive_subset_seed(master_seed, SEED_LABEL_BACKUP, j as u64);
        let blocks = compute_backup_blocks(&subset_seed, c);
        backup_hints.push(BackupHint {
            subset_seed,
            parity_in: [0u8; 32],
            parity_out: [0u8; 32],
        });
        backup_hint_blocks.push(blocks);
    }

    (
        regular_hints,
        regular_hint_blocks,
        backup_hints,
        backup_hint_blocks,
    )
}

/// Print final results summary.
#[allow(clippy::too_many_arguments)]
pub fn print_results(
    duration: Duration,
    file_len: usize,
    regular_hints: &[RegularHint],
    backup_hints: &[BackupHint],
    params: &HintParams,
    _block_keys: &[PrfKey128],
    _w: usize,
    _c: usize,
) {
    let throughput_mb = (file_len as f64 / 1024.0 / 1024.0) / duration.as_secs_f64();

    let non_zero_regular = regular_hints
        .iter()
        .filter(|h| h.parity.iter().any(|&b| b != 0))
        .count();
    let zero_regular_seeds = regular_hints
        .iter()
        .filter(|h| h.subset_seed.iter().all(|&b| b == 0))
        .count();

    info!(
        duration_secs = format_args!("{:.2}", duration.as_secs_f64()),
        throughput_mb_s = format_args!("{:.2}", throughput_mb),
        non_zero_regular,
        total_regular = params.num_regular,
        zero_regular_seeds,
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
        let zero_backup_seeds = backup_hints
            .iter()
            .filter(|h| h.subset_seed.iter().all(|&b| b == 0))
            .count();
        info!(
            non_zero_parity_in = non_zero_backup_in,
            non_zero_parity_out = non_zero_backup_out,
            total_backup = params.num_backup,
            zero_backup_seeds,
            "Backup hint results"
        );
    }
}
