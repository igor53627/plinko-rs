//! Experimental multi-level PIR prototype (`ePrint 2026/1273`, Section 4)
//! extended with dynamic `UpdateHint` delta updates.
//!
//! This module is a standalone research prototype separate from the Fig. 7
//! Plinko protocol in [`crate::protocol`]. It implements the failure-private
//! waterfall construction of Shi, Peng, Li, Hong, and Zhou (2026) with two
//! per-level index/color backends so their `HintInit`, `UpdateHint`, and
//! `Query` costs can be compared directly:
//!
//! - [`LevelBackend::Iprf`]: uses [`crate::iprf::Iprf`] per block (Variant A
//!   on early levels) for `O(kappa)` asymptotic hint updates.
//! - [`LevelBackend::FlatAes`]: uses batched AES-128-CTR per block (Variant A
//!   on small-domain upper levels, or Variant B on all levels) with linear
//!   scan over the level's hints during `UpdateHint` and `O(1)` single-block
//!   AES lookup during `Query`.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::io::{self, Read, Write};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use aes::cipher::{generic_array::GenericArray, BlockEncrypt, KeyInit};
use aes::{Aes128, Block};
use rand::Rng;

use crate::checked_io::{u32_at, u64_at, HashingReader, HashingWriter, TrailerError};
use crate::iprf::{Iprf, IprfTable, PrfKey128};

/// Magic bytes for serialized multi-level hint sets.
const MAGIC: &[u8; 8] = b"PLNKMLH1";

/// Binary format version for serialized multi-level hint sets.
const FORMAT_VERSION: u32 = 1;

/// Maximum levels accepted when decoding a serialized hint set.
const MAX_LEVELS: usize = 16;

/// Default maximum decoded byte budget accepted by [`MultiLevelClient::read_hints`]
/// (512 MiB) to bound memory allocation on untrusted checkpoints.
const MAX_CHECKPOINT_BYTES: usize = 512 * 1024 * 1024;

/// Maximum supported block size `w_i` (`2^31`, since `FlatAes` uses bit 31 as the color bit
/// for `u32` words and query offsets are stored as `u32`).
const MAX_BLOCK_SIZE: usize = 1usize << 31;

/// Process-unique monotonic session ID counter so pending queries cannot cross clients
/// or survive reloads even when two clients share a seed and parameters.
static NEXT_SESSION_ID: AtomicU64 = AtomicU64::new(1);

fn next_session_id() -> u64 {
    NEXT_SESSION_ID.fetch_add(1, Ordering::Relaxed)
}

/// Sentinel value in `hint_id_to_slot` indicating a hint is not in the active primary table.
const NO_SLOT: u32 = u32::MAX;

/// AES batch size (number of 128-bit blocks encrypted together per `encrypt_blocks` call).
const AES_BATCH_BLOCKS: usize = 8;

/// Maximum per-level parity buffer size (in bytes) for which `FlatAes` `HintInit`
/// uses thread-local non-atomic accumulators to avoid cache-line contention.
const LOCAL_ACCUM_THRESHOLD_BYTES: usize = 8 * 1024 * 1024;

/// Maximum combined thread-local accumulator memory across all levels and workers
/// before remaining levels fall back to shared atomic accumulators.
const LOCAL_ACCUM_MAX_WORKERS_BYTES: usize = 128 * 1024 * 1024;

/// Default database streaming chunk buffer size (64 MiB on native targets) used by
/// [`MultiLevelHintBuilder::new`], [`MultiLevelClient::hint_init_from_reader`], and
/// [`MultiLevelClient::hint_init_from_file`].
#[cfg(not(target_arch = "wasm32"))]
pub const DEFAULT_STREAM_CHUNK_BYTES: usize = 64 * 1024 * 1024;

/// Default database streaming chunk buffer size (4 MiB on `wasm32`) used by
/// [`MultiLevelHintBuilder::new`] and [`MultiLevelClient::hint_init_from_reader`].
#[cfg(target_arch = "wasm32")]
pub const DEFAULT_STREAM_CHUNK_BYTES: usize = 4 * 1024 * 1024;

/// Backend that generates `(color, offset)` pairs for one level's blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LevelBackend {
    /// Per-block [`Iprf`] (`SwapOrNotSr` + PMNS) for offsets plus AES-128 color bits.
    Iprf,
    /// Per-block batched AES-128-CTR producing both color bit and offset directly.
    FlatAes,
}

/// Configuration for one level of the multi-level waterfall.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LevelSpec {
    /// Number of records per block (`w_i`, must be a power of two >= 2).
    pub block_size: usize,
    /// Number of blocks (`c_i = ceil(num_entries / block_size)`).
    pub num_blocks: usize,
    /// Number of primary hints (`M_i`).
    pub num_primary: usize,
    /// Number of backup hints (`Q_i`).
    pub num_backup: usize,
    /// Index/color generation backend for this level.
    pub backend: LevelBackend,
}

impl LevelSpec {
    /// Total hints (`num_primary + num_backup`) generated for this level.
    pub fn total_hints(&self) -> usize {
        self.num_primary + self.num_backup
    }

    /// Theoretical single-query failure probability `(1 - 1 / (2 * w_i))^{M_i}`.
    pub fn failure_probability(&self) -> f64 {
        let p_in: f64 = 1.0 / (2.0 * self.block_size as f64);
        (self.num_primary as f64 * (-p_in).ln_1p()).exp()
    }

    /// Hint storage bytes for this level (parities + 4-byte metadata per primary,
    /// 2 parities per backup).
    pub fn hint_bytes(&self, record_size: usize) -> usize {
        self.num_primary * (4 + record_size) + self.num_backup * (2 * record_size)
    }

    /// Approximate query upload bytes for this level (1 color bit + `log2(w_i)` offset bits per block).
    pub fn query_upload_bytes(&self) -> usize {
        let bits_per_block: usize = 1 + self.block_size.trailing_zeros() as usize;
        (self.num_blocks * bits_per_block).div_ceil(8)
    }
}

/// Strategy for assigning [`LevelBackend`] across levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendVariant {
    /// Variant A (hybrid): [`LevelBackend::Iprf`] on the first `iprf_levels` levels,
    /// [`LevelBackend::FlatAes`] on all upper (small-domain) levels.
    HybridA {
        /// Number of early levels using [`LevelBackend::Iprf`].
        iprf_levels: usize,
    },
    /// Variant B: [`LevelBackend::FlatAes`] on every level.
    FlatB,
}

/// Parameters for a multi-level PIR instance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultiLevelParams {
    /// Total number of database records.
    pub num_entries: usize,
    /// Size of each fixed-size record in bytes.
    pub record_size: usize,
    /// Levels ordered from lightest (level 0) to heaviest (last level).
    pub levels: Vec<LevelSpec>,
}

/// Errors returned by multi-level PIR parameter construction and operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MultiLevelError {
    /// Invalid parameter configuration.
    InvalidParams(&'static str),
    /// Database input length does not match `num_entries * record_size`.
    DatabaseLengthMismatch {
        /// Expected byte length.
        expected: usize,
        /// Actual byte length, or a lower bound (`expected.saturating_add(1)`) when a streaming
        /// [`Read`] source has trailing bytes past `expected` (detected via a single-byte probe
        /// without draining the stream).
        actual: usize,
    },
    /// Query or delta index is out of bounds.
    IndexOutOfRange {
        /// Out-of-bounds index.
        index: usize,
        /// Database record count.
        num_entries: usize,
    },
    /// Delta byte length does not match `record_size`.
    DeltaLengthMismatch {
        /// Expected record size.
        expected: usize,
        /// Actual delta length.
        actual: usize,
    },
    /// All levels failed to cover the queried index.
    AllLevelsFailed {
        /// Queried index.
        index: usize,
    },
    /// Backup pool on a level was exhausted during refresh.
    ///
    /// When this occurs inside [`MultiLevelClient::finish_query`], `recovered` holds the
    /// correctly decoded database record while the consumed slot remains retired; the client
    /// must re-run `HintInit` before serving more queries.
    BackupExhausted {
        /// Level index whose backup pool ran out.
        level: usize,
        /// Record recovered before refresh aborted, if any.
        recovered: Option<Vec<u8>>,
    },
    /// Server answer parity lengths did not match `record_size`.
    AnswerLengthMismatch {
        /// Expected record size in bytes.
        expected: usize,
        /// Actual `(parity_0.len(), parity_1.len())`.
        actual: (usize, usize),
    },
    /// Query passed to [`MultiLevelServer::answer`] had invalid level, length, color, or offset.
    InvalidQuery(&'static str),
    /// Pending query ID was unknown or already completed.
    UnknownPendingQuery {
        /// Unrecognized query reservation ID.
        query_id: u64,
    },
    /// Database deltas were applied between `prepare_query` and `finish_query`, so the
    /// server answer and client parities come from different snapshots.
    SnapshotMismatch {
        /// Client snapshot epoch when `prepare_query` ran.
        query_epoch: u64,
        /// Current client snapshot epoch.
        current_epoch: u64,
    },
    /// Operation cannot run while one or more prepared queries are still in flight.
    QueryInFlight,
    /// Serialized hint set header or body failed structural validation.
    InvalidFormat(&'static str),
    /// Stored BLAKE3 checksum does not match the serialized bytes.
    ChecksumMismatch,
    /// Unexpected bytes follow the trailing BLAKE3 checksum.
    TrailingData,
    /// Underlying I/O error during serialization or deserialization.
    Io(io::ErrorKind),
}

impl fmt::Display for MultiLevelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParams(msg) => write!(f, "invalid multi-level params: {msg}"),
            Self::DatabaseLengthMismatch { expected, actual } => {
                write!(
                    f,
                    "database length mismatch: expected {expected} bytes, got {actual}"
                )
            }
            Self::IndexOutOfRange { index, num_entries } => {
                write!(f, "index {index} out of range (num_entries={num_entries})")
            }
            Self::DeltaLengthMismatch { expected, actual } => {
                write!(
                    f,
                    "delta length mismatch: expected {expected} bytes, got {actual}"
                )
            }
            Self::AllLevelsFailed { index } => {
                write!(f, "all levels failed to cover index {index}")
            }
            Self::BackupExhausted { level, .. } => {
                write!(f, "backup pool exhausted on level {level}")
            }
            Self::AnswerLengthMismatch { expected, actual } => {
                write!(
                    f,
                    "answer parity length mismatch: expected ({expected}, {expected}), got {actual:?}"
                )
            }
            Self::InvalidQuery(msg) => write!(f, "invalid query: {msg}"),
            Self::UnknownPendingQuery { query_id } => {
                write!(f, "unknown or already completed pending query {query_id}")
            }
            Self::SnapshotMismatch {
                query_epoch,
                current_epoch,
            } => write!(
                f,
                "database snapshot mismatch: query prepared at epoch {query_epoch}, client is at epoch {current_epoch}"
            ),
            Self::QueryInFlight => {
                write!(f, "cannot serialize hints while a query is in flight")
            }
            Self::InvalidFormat(msg) => write!(f, "invalid serialized hint set: {msg}"),
            Self::ChecksumMismatch => write!(f, "hint set BLAKE3 checksum mismatch"),
            Self::TrailingData => write!(f, "trailing data after hint set checksum"),
            Self::Io(kind) => write!(f, "I/O error: {kind:?}"),
        }
    }
}

impl From<io::Error> for MultiLevelError {
    fn from(err: io::Error) -> Self {
        Self::Io(err.kind())
    }
}

impl std::error::Error for MultiLevelError {}

impl MultiLevelParams {
    /// Validates and builds custom multi-level parameters.
    pub fn custom(
        num_entries: usize,
        record_size: usize,
        levels: Vec<LevelSpec>,
    ) -> Result<Self, MultiLevelError> {
        if num_entries == 0 {
            return Err(MultiLevelError::InvalidParams("num_entries must be > 0"));
        }
        if record_size == 0 || record_size > u32::MAX as usize {
            return Err(MultiLevelError::InvalidParams(
                "record_size must be in 1..=u32::MAX",
            ));
        }
        if levels.is_empty() || levels.len() > MAX_LEVELS {
            return Err(MultiLevelError::InvalidParams(
                "levels.len() must be in 1..=MAX_LEVELS",
            ));
        }
        for spec in &levels {
            if spec.block_size < 2
                || !spec.block_size.is_power_of_two()
                || spec.block_size > MAX_BLOCK_SIZE
            {
                return Err(MultiLevelError::InvalidParams(
                    "block_size must be a power of two in 2..=2^31",
                ));
            }
            let expected_blocks: usize = num_entries.div_ceil(spec.block_size);
            if spec.num_blocks != expected_blocks {
                return Err(MultiLevelError::InvalidParams(
                    "num_blocks must equal ceil(num_entries / block_size)",
                ));
            }
            if spec.num_blocks > u32::MAX as usize {
                return Err(MultiLevelError::InvalidParams("num_blocks must fit in u32"));
            }
            if spec.num_primary == 0 || spec.num_backup == 0 {
                return Err(MultiLevelError::InvalidParams(
                    "num_primary and num_backup must be > 0",
                ));
            }
            let total: usize = spec
                .num_primary
                .checked_add(spec.num_backup)
                .ok_or(MultiLevelError::InvalidParams("total_hints overflow"))?;
            if total >= u32::MAX as usize {
                return Err(MultiLevelError::InvalidParams(
                    "total_hints per level must fit in u32",
                ));
            }
            let max_vec_bytes: usize = isize::MAX as usize;
            let primary_bytes: usize = spec
                .num_primary
                .checked_mul(record_size)
                .filter(|&n| n <= max_vec_bytes)
                .ok_or(MultiLevelError::InvalidParams(
                    "primary parity storage overflow",
                ))?;
            let backup_bytes: usize = spec
                .num_backup
                .checked_mul(record_size)
                .filter(|&n| n <= max_vec_bytes)
                .ok_or(MultiLevelError::InvalidParams(
                    "backup parity storage overflow",
                ))?;
            let _ = primary_bytes
                .checked_add(backup_bytes)
                .and_then(|n| n.checked_add(backup_bytes))
                .ok_or(MultiLevelError::InvalidParams(
                    "hint parity storage overflow",
                ))?;
            let words: usize = record_size.div_ceil(8);
            let acc_word_bytes: usize = words.checked_mul(8).ok_or(
                MultiLevelError::InvalidParams("accumulator word size overflow"),
            )?;
            let acc_hints: usize = spec
                .num_backup
                .checked_mul(2)
                .and_then(|n| n.checked_add(spec.num_primary))
                .ok_or(MultiLevelError::InvalidParams(
                    "accumulator hint count overflow",
                ))?;
            let _ = acc_hints
                .checked_mul(acc_word_bytes)
                .filter(|&n| n <= max_vec_bytes)
                .ok_or(MultiLevelError::InvalidParams(
                    "accumulator storage overflow",
                ))?;
        }
        Ok(Self {
            num_entries,
            record_size,
            levels,
        })
    }

    /// Constructs 2-level parameters following Table 2 of `ePrint 2026/1273`.
    ///
    /// Level 0 has failure probability `~1 / kappa` (`M_0 = ceil(2 * w_0 * ln(kappa))`),
    /// and Level 1 has failure probability `~exp(-kappa)` (`M_1 = ceil(2 * w_1 * kappa)`),
    /// with `w_1 = max(2, w_0 / next_power_of_two(kappa))`.
    pub fn two_level(
        num_entries: usize,
        record_size: usize,
        kappa: usize,
        query_budget: usize,
        variant: BackendVariant,
    ) -> Result<Self, MultiLevelError> {
        if num_entries == 0 || kappa < 4 || query_budget == 0 {
            return Err(MultiLevelError::InvalidParams(
                "require num_entries > 0, kappa >= 4, query_budget > 0",
            ));
        }
        let sqrt_n: usize = (num_entries as f64).sqrt().ceil() as usize;
        let w0: usize = (sqrt_n / 2).max(2).next_power_of_two();
        let ratio: usize =
            kappa
                .checked_next_power_of_two()
                .ok_or(MultiLevelError::InvalidParams(
                    "kappa power-of-two overflow",
                ))?;
        if ratio > w0 / 2 {
            return Err(MultiLevelError::InvalidParams(
                "database too small for 2-level kappa (need ratio <= w0 / 2)",
            ));
        }
        let w1: usize = w0 / ratio;
        let c0: usize = num_entries.div_ceil(w0);
        let c1: usize = num_entries.div_ceil(w1);

        let ln_k: f64 = (kappa as f64).ln();
        let m0: usize = ((2.0 * w0 as f64 * ln_k).ceil() as usize).max(1);
        let m1: usize = w1
            .checked_mul(2)
            .and_then(|v| v.checked_mul(kappa))
            .ok_or(MultiLevelError::InvalidParams(
                "level-1 primary count overflow",
            ))?
            .max(1);

        let p0_fail: f64 = (-(m0 as f64) / (2.0 * w0 as f64)).exp();
        // Level 0 runs on every query; Lemma 10 guarantees 3*Q backups suffice.
        let q0: usize = query_budget
            .checked_mul(3)
            .ok_or(MultiLevelError::InvalidParams(
                "level-0 backup budget overflow",
            ))?
            .max(64);
        // Level 1 runs only on level 0 failure (~Q * p0_fail times), with generous safety margin.
        let expected_l1_calls: f64 = query_budget as f64 * p0_fail;
        let scaled_l1: f64 = (3.0 * expected_l1_calls).ceil();
        if !scaled_l1.is_finite() || scaled_l1 > usize::MAX as f64 {
            return Err(MultiLevelError::InvalidParams(
                "level-1 backup budget overflow",
            ));
        }
        let margin_l1: usize = kappa.checked_mul(4).ok_or(MultiLevelError::InvalidParams(
            "level-1 backup margin overflow",
        ))?;
        let q1: usize = (scaled_l1 as usize)
            .checked_add(margin_l1)
            .ok_or(MultiLevelError::InvalidParams(
                "level-1 backup budget overflow",
            ))?
            .max(64);

        let b0: LevelBackend = match variant {
            BackendVariant::HybridA { iprf_levels } if iprf_levels >= 1 => LevelBackend::Iprf,
            _ => LevelBackend::FlatAes,
        };
        let b1: LevelBackend = match variant {
            BackendVariant::HybridA { iprf_levels } if iprf_levels >= 2 => LevelBackend::Iprf,
            _ => LevelBackend::FlatAes,
        };

        Self::custom(
            num_entries,
            record_size,
            vec![
                LevelSpec {
                    block_size: w0,
                    num_blocks: c0,
                    num_primary: m0,
                    num_backup: q0,
                    backend: b0,
                },
                LevelSpec {
                    block_size: w1,
                    num_blocks: c1,
                    num_primary: m1,
                    num_backup: q1,
                    backend: b1,
                },
            ],
        )
    }

    /// Constructs a `num_levels`-level geometric waterfall following Table 3 of `ePrint 2026/1273`.
    ///
    /// Level `i` (`0..num_levels`) has `w_i = w_0 / 4^i` (power of two `>= 2`) and
    /// `M_i = 2 * w_i * 2^i`, giving per-level failure probability `exp(-2^i)` and
    /// cumulative failure probability `exp(-(2^{num_levels} - 1))`.
    pub fn waterfall(
        num_entries: usize,
        record_size: usize,
        num_levels: usize,
        query_budget: usize,
        variant: BackendVariant,
    ) -> Result<Self, MultiLevelError> {
        if num_entries == 0 || num_levels == 0 || num_levels > 10 || query_budget == 0 {
            return Err(MultiLevelError::InvalidParams(
                "require num_entries > 0, 1 <= num_levels <= 10, query_budget > 0",
            ));
        }
        let sqrt_n: usize = (num_entries as f64).sqrt().ceil() as usize;
        // Pick w0 as a power of two near sqrt(n)/2, ensuring w_{d-1} = w0 / 4^{d-1} >= 2.
        let min_w0: usize = 2usize
            .checked_shl(2 * (num_levels as u32 - 1))
            .ok_or(MultiLevelError::InvalidParams("num_levels overflow"))?;
        let candidate_w0: usize = (sqrt_n / 2).max(2).next_power_of_two();
        // Prefer the power of two <= sqrt(n) if it still keeps w_{d-1} >= 2.
        let w0: usize = if candidate_w0 >= min_w0 {
            candidate_w0
        } else {
            return Err(MultiLevelError::InvalidParams(
                "database too small for requested waterfall depth (need sqrt(n)/2 >= 2 * 4^{d-1})",
            ));
        };

        let mut levels: Vec<LevelSpec> = Vec::with_capacity(num_levels);
        let mut cum_exec_prob: f64 = 1.0;

        for i in 0..num_levels {
            let w_i: usize = w0 >> (2 * i);
            let c_i: usize = num_entries.div_ceil(w_i);
            let cov_i: usize = 1usize << i; // target M_i / (2 * w_i) = 2^i
            let m_i: usize = 2 * w_i * cov_i;
            let fail_i: f64 = (-(cov_i as f64)).exp();

            // Conservative backup budget for this level: ~3x expected calls + tail safety margin.
            let expected_calls: f64 = query_budget as f64 * cum_exec_prob;
            let expected_backups_per_call: f64 = 1.0 + (cov_i as f64) * fail_i;
            let q_i: usize = if i == 0 {
                query_budget
                    .checked_mul(3)
                    .ok_or(MultiLevelError::InvalidParams(
                        "level-0 backup budget overflow",
                    ))?
                    .max(128)
            } else {
                let scaled: f64 = (3.0 * expected_calls * expected_backups_per_call).ceil();
                if !scaled.is_finite() || scaled > usize::MAX as f64 {
                    return Err(MultiLevelError::InvalidParams("backup budget overflow"));
                }
                (scaled as usize)
                    .checked_add(128)
                    .ok_or(MultiLevelError::InvalidParams("backup budget overflow"))?
                    .max(128)
            };

            let backend: LevelBackend = match variant {
                BackendVariant::HybridA { iprf_levels } if i < iprf_levels => LevelBackend::Iprf,
                _ => LevelBackend::FlatAes,
            };

            levels.push(LevelSpec {
                block_size: w_i,
                num_blocks: c_i,
                num_primary: m_i,
                num_backup: q_i,
                backend,
            });
            cum_exec_prob *= fail_i;
        }

        Self::custom(num_entries, record_size, levels)
    }

    /// Total hint storage bytes across all levels.
    pub fn total_hint_bytes(&self) -> usize {
        self.levels
            .iter()
            .map(|spec| spec.hint_bytes(self.record_size))
            .sum()
    }

    /// Total hints (primary + backup) summed across all levels.
    pub fn total_hints_all_levels(&self) -> usize {
        self.levels.iter().map(LevelSpec::total_hints).sum()
    }

    /// Cumulative theoretical failure probability across all levels.
    pub fn total_failure_probability(&self) -> f64 {
        self.levels
            .iter()
            .map(LevelSpec::failure_probability)
            .product()
    }
}

/// Single-chunk override attached to a primary hint promoted from a backup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SlotOverride {
    block: u32,
    offset: u32,
    color: u8,
}

/// One primary hint slot in a level's primary table.
#[derive(Debug, Clone)]
struct PrimarySlot {
    /// Original hint ID (`0..num_primary` if original, `num_primary..total_hints` if promoted).
    hint_id: u32,
    /// Active color bit `c in {0, 1}`.
    selector: u8,
    /// Optional single-chunk overwrite from `PromoteIn` or `PromoteOut`.
    override_entry: Option<SlotOverride>,
    /// True while reserved by an in-flight query.
    reserved: bool,
    /// True once permanently retired (abandoned/mismatched query); never reused for queries.
    retired: bool,
}

/// Per-level runtime state held by the client.
struct LevelState {
    spec: LevelSpec,
    record_size: usize,
    /// AES cipher deriving per-block keys `K_{level, block}`.
    block_key_cipher: Aes128,
    /// AES cipher deriving color bits when `backend == LevelBackend::Iprf`.
    color_cipher: Aes128,
    /// Primary slot metadata (length `spec.num_primary`).
    slots: Vec<PrimarySlot>,
    /// Flat primary parities (length `spec.num_primary * record_size`).
    primary_parities: Vec<u8>,
    /// Flat backup parities for color 0, starting at backup index `backup_offset`
    /// (length `(spec.num_backup - backup_offset) * record_size`).
    backup_parities_0: Vec<u8>,
    /// Flat backup parities for color 1, starting at backup index `backup_offset`
    /// (length `(spec.num_backup - backup_offset) * record_size`).
    backup_parities_1: Vec<u8>,
    /// First backup index stored in `backup_parities_{0,1}` (`0` after HintInit,
    /// `next_backup` after loading a checkpoint so consumed backups take zero memory).
    backup_offset: usize,
    /// Next unused backup index in `backup_offset..=spec.num_backup`.
    next_backup: usize,
    /// Maps `hint_id -> active slot index` in `0..spec.num_primary`, or [`NO_SLOT`].
    hint_id_to_slot: Vec<u32>,
    /// Reverse index `(block, offset) -> list of primary slot indices` having an override there.
    override_index: HashMap<(u32, u32), Vec<u32>>,
}

impl LevelState {
    fn new_zeroed(spec: LevelSpec, record_size: usize, seed: &[u8; 32], level_idx: usize) -> Self {
        let block_key_cipher: Aes128 = derive_level_cipher(seed, b"ml-block-key", level_idx);
        let color_cipher: Aes128 = derive_level_cipher(seed, b"ml-color-bit", level_idx);
        let selector_cipher: Aes128 = derive_level_cipher(seed, b"ml-init-sel", level_idx);

        let mut slots: Vec<PrimarySlot> = Vec::with_capacity(spec.num_primary);
        let mut hint_id_to_slot: Vec<u32> = vec![NO_SLOT; spec.total_hints()];

        for (j, slot_ref) in hint_id_to_slot
            .iter_mut()
            .enumerate()
            .take(spec.num_primary)
        {
            let selector: u8 = initial_selector(&selector_cipher, j);
            slots.push(PrimarySlot {
                hint_id: j as u32,
                selector,
                override_entry: None,
                reserved: false,
                retired: false,
            });
            *slot_ref = j as u32;
        }

        Self {
            spec,
            record_size,
            block_key_cipher,
            color_cipher,
            slots,
            primary_parities: vec![0u8; spec.num_primary * record_size],
            backup_parities_0: vec![0u8; spec.num_backup * record_size],
            backup_parities_1: vec![0u8; spec.num_backup * record_size],
            backup_offset: 0,
            next_backup: 0,
            hint_id_to_slot,
            override_index: HashMap::new(),
        }
    }

    #[inline]
    fn derive_block_key(&self, block: usize) -> PrfKey128 {
        let mut blk: Block = Block::default();
        blk[0..8].copy_from_slice(&(block as u64).to_le_bytes());
        self.block_key_cipher.encrypt_block(&mut blk);
        blk.into()
    }

    #[inline]
    fn iprf_color_bit(&self, block: usize, hint_id: u32) -> u8 {
        let mut blk: Block = Block::default();
        blk[0..8].copy_from_slice(&(block as u64).to_le_bytes());
        blk[8..16].copy_from_slice(&((hint_id >> 7) as u64).to_le_bytes());
        self.color_cipher.encrypt_block(&mut blk);
        let bit_pos: usize = (hint_id & 127) as usize;
        (blk[bit_pos >> 3] >> (bit_pos & 7)) & 1
    }

    /// Evaluates `(color, offset)` for one `(block, hint_id)` pair.
    fn forward(&self, block: usize, hint_id: u32) -> (u8, u32) {
        let key: PrfKey128 = self.derive_block_key(block);
        match self.spec.backend {
            LevelBackend::Iprf => {
                let iprf: Iprf = Iprf::new(
                    key,
                    self.spec.total_hints() as u64,
                    self.spec.block_size as u64,
                );
                let offset: u32 = iprf.forward(hint_id as u64) as u32;
                let color: u8 = self.iprf_color_bit(block, hint_id);
                (color, offset)
            }
            LevelBackend::FlatAes => {
                let cipher: Aes128 = Aes128::new(&GenericArray::from(key));
                flat_aes_forward(&cipher, self.spec.block_size, hint_id)
            }
        }
    }

    /// Evaluates `inverse(block, target_offset)` returning all `(hint_id, color)` pairs
    /// whose generated offset in `block` equals `target_offset`.
    fn inverse(&self, block: usize, target_offset: u32, out: &mut Vec<(u32, u8)>) {
        out.clear();
        let key: PrfKey128 = self.derive_block_key(block);
        match self.spec.backend {
            LevelBackend::Iprf => {
                let iprf: Iprf = Iprf::new(
                    key,
                    self.spec.total_hints() as u64,
                    self.spec.block_size as u64,
                );
                for h in iprf.inverse(target_offset as u64) {
                    let hint_id: u32 = h as u32;
                    let color: u8 = self.iprf_color_bit(block, hint_id);
                    out.push((hint_id, color));
                }
            }
            LevelBackend::FlatAes => {
                let cipher: Aes128 = Aes128::new(&GenericArray::from(key));
                flat_aes_inverse(
                    &cipher,
                    self.spec.block_size,
                    self.spec.total_hints(),
                    target_offset,
                    out,
                );
            }
        }
    }

    /// Finds all active, non-reserved primary slot indices currently containing `entry_idx`.
    fn find_matching_slots(&self, entry_idx: usize, scratch: &mut Vec<(u32, u8)>) -> Vec<u32> {
        let block: usize = entry_idx / self.spec.block_size;
        let offset: u32 = (entry_idx % self.spec.block_size) as u32;
        self.inverse(block, offset, scratch);

        let mut matches: Vec<u32> = Vec::new();
        for &(hint_id, color_old) in scratch.iter() {
            let slot_idx: u32 = self.hint_id_to_slot[hint_id as usize];
            if slot_idx == NO_SLOT {
                continue;
            }
            let slot: &PrimarySlot = &self.slots[slot_idx as usize];
            if slot.reserved || slot.retired {
                continue;
            }
            if let Some(ov) = slot.override_entry {
                if ov.block as usize == block {
                    continue;
                }
            }
            if color_old == slot.selector {
                matches.push(slot_idx);
            }
        }

        if let Some(ov_slots) = self.override_index.get(&(block as u32, offset)) {
            for &slot_idx in ov_slots {
                let slot: &PrimarySlot = &self.slots[slot_idx as usize];
                if slot.reserved || slot.retired {
                    continue;
                }
                if let Some(ov) = slot.override_entry {
                    if ov.color == slot.selector {
                        matches.push(slot_idx);
                    }
                }
            }
        }
        matches
    }

    /// Replaces primary slot `slot_idx` with a new slot and parity, updating both lookup indices.
    fn install_slot(&mut self, slot_idx: u32, new_slot: PrimarySlot, new_parity: &[u8]) {
        let old: PrimarySlot = self.slots[slot_idx as usize].clone();
        self.hint_id_to_slot[old.hint_id as usize] = NO_SLOT;
        if let Some(ov) = old.override_entry {
            let key: (u32, u32) = (ov.block, ov.offset);
            if let Some(list) = self.override_index.get_mut(&key) {
                if let Some(pos) = list.iter().position(|&s| s == slot_idx) {
                    list.swap_remove(pos);
                }
                if list.is_empty() {
                    self.override_index.remove(&key);
                }
            }
        }

        self.hint_id_to_slot[new_slot.hint_id as usize] = slot_idx;
        if let Some(ov) = new_slot.override_entry {
            self.override_index
                .entry((ov.block, ov.offset))
                .or_default()
                .push(slot_idx);
        }
        self.slots[slot_idx as usize] = new_slot;

        let r: usize = self.record_size;
        let dst: &mut [u8] =
            &mut self.primary_parities[slot_idx as usize * r..(slot_idx as usize + 1) * r];
        dst.copy_from_slice(new_parity);
    }

    /// Promotes the next unused backup hint into a fresh primary hint conditioned on containing `x`.
    fn promote_in(
        &mut self,
        level_idx: usize,
        x_block: u32,
        x_offset: u32,
        value: &[u8],
    ) -> Result<(PrimarySlot, Vec<u8>), MultiLevelError> {
        if self.next_backup >= self.spec.num_backup {
            return Err(MultiLevelError::BackupExhausted {
                level: level_idx,
                recovered: None,
            });
        }
        let b: usize = self.next_backup;
        self.next_backup += 1;
        let local_b: usize = b - self.backup_offset;
        let hint_id: u32 = (self.spec.num_primary + b) as u32;
        let (s_old, _delta_old) = self.forward(x_block as usize, hint_id);
        let new_color: u8 = 1 - s_old;
        let r: usize = self.record_size;
        let base_parity: &[u8] = if new_color == 0 {
            &self.backup_parities_0[local_b * r..(local_b + 1) * r]
        } else {
            &self.backup_parities_1[local_b * r..(local_b + 1) * r]
        };
        let mut parity: Vec<u8> = base_parity.to_vec();
        xor_bytes(&mut parity, value);
        let slot: PrimarySlot = PrimarySlot {
            hint_id,
            selector: new_color,
            override_entry: Some(SlotOverride {
                block: x_block,
                offset: x_offset,
                color: new_color,
            }),
            reserved: false,
            retired: false,
        };
        Ok((slot, parity))
    }

    /// Promotes the next unused backup hint into a fresh primary hint conditioned on excluding `x`
    /// (Lemma 8 of `ePrint 2026/1273`).
    fn promote_out<R: Rng + ?Sized>(
        &mut self,
        level_idx: usize,
        x_block: u32,
        x_offset: u32,
        rng: &mut R,
    ) -> Result<(PrimarySlot, Vec<u8>), MultiLevelError> {
        if self.next_backup >= self.spec.num_backup {
            return Err(MultiLevelError::BackupExhausted {
                level: level_idx,
                recovered: None,
            });
        }
        let b: usize = self.next_backup;
        self.next_backup += 1;
        let local_b: usize = b - self.backup_offset;
        let hint_id: u32 = (self.spec.num_primary + b) as u32;
        let (s_old, delta_old) = self.forward(x_block as usize, hint_id);

        let w: u64 = self.spec.block_size as u64;
        let denom: u64 = 2 * w - 1;
        let draw: u64 = rng.gen_range(0..denom);

        let (selector, override_entry): (u8, Option<SlotOverride>) = if delta_old == x_offset {
            let sel: u8 = 1 - s_old;
            if draw < w {
                (sel, None)
            } else {
                let mut new_off: u32 = rng.gen_range(0..(self.spec.block_size as u32 - 1));
                if new_off >= x_offset {
                    new_off += 1;
                }
                (
                    sel,
                    Some(SlotOverride {
                        block: x_block,
                        offset: new_off,
                        color: s_old,
                    }),
                )
            }
        } else {
            let sel: u8 = if draw < w { s_old } else { 1 - s_old };
            (sel, None)
        };

        let r: usize = self.record_size;
        let base_parity: &[u8] = if selector == 0 {
            &self.backup_parities_0[local_b * r..(local_b + 1) * r]
        } else {
            &self.backup_parities_1[local_b * r..(local_b + 1) * r]
        };
        let slot: PrimarySlot = PrimarySlot {
            hint_id,
            selector,
            override_entry,
            reserved: false,
            retired: false,
        };
        Ok((slot, base_parity.to_vec()))
    }

    /// Applies a single database record delta to all affected primary and unused backup hints.
    fn apply_delta(&mut self, entry_idx: usize, delta: &[u8], scratch: &mut Vec<(u32, u8)>) {
        let block: usize = entry_idx / self.spec.block_size;
        let offset: u32 = (entry_idx % self.spec.block_size) as u32;
        let r: usize = self.record_size;

        self.inverse(block, offset, scratch);

        for &(hint_id, color_old) in scratch.iter() {
            let h: usize = hint_id as usize;
            let slot_idx: u32 = self.hint_id_to_slot[h];
            if slot_idx != NO_SLOT {
                let slot: &PrimarySlot = &self.slots[slot_idx as usize];
                if let Some(ov) = slot.override_entry {
                    if ov.block as usize == block {
                        continue;
                    }
                }
                if color_old == slot.selector {
                    let s: usize = slot_idx as usize;
                    xor_bytes(&mut self.primary_parities[s * r..(s + 1) * r], delta);
                }
            } else if h >= self.spec.num_primary {
                let b: usize = h - self.spec.num_primary;
                if b >= self.next_backup {
                    let local_b: usize = b - self.backup_offset;
                    let target: &mut [u8] = if color_old == 0 {
                        &mut self.backup_parities_0[local_b * r..(local_b + 1) * r]
                    } else {
                        &mut self.backup_parities_1[local_b * r..(local_b + 1) * r]
                    };
                    xor_bytes(target, delta);
                }
            }
        }

        if let Some(ov_slots) = self.override_index.get(&(block as u32, offset)) {
            for &slot_idx in ov_slots {
                let slot: &PrimarySlot = &self.slots[slot_idx as usize];
                if let Some(ov) = slot.override_entry {
                    if ov.color == slot.selector {
                        let s: usize = slot_idx as usize;
                        xor_bytes(&mut self.primary_parities[s * r..(s + 1) * r], delta);
                    }
                }
            }
        }
    }
}

struct LocalParityBuf {
    reg: Vec<u64>,
    b0: Vec<u64>,
    b1: Vec<u64>,
}

enum LevelParityAccum {
    Local(Vec<LocalParityBuf>),
    Atomic {
        reg: Vec<AtomicU64>,
        b0: Vec<AtomicU64>,
        b1: Vec<AtomicU64>,
    },
}

struct LevelAccumulator {
    spec: LevelSpec,
    record_size: usize,
    words: usize,
    block_key_cipher: Aes128,
    color_cipher: Aes128,
    selectors: Vec<u8>,
    workers: usize,
    accum: LevelParityAccum,
}

impl LevelAccumulator {
    fn new(
        spec: LevelSpec,
        record_size: usize,
        seed: &[u8; 32],
        level_idx: usize,
        threads: usize,
        remaining_local_budget: &mut usize,
    ) -> Self {
        let block_key_cipher: Aes128 = derive_level_cipher(seed, b"ml-block-key", level_idx);
        let color_cipher: Aes128 = derive_level_cipher(seed, b"ml-color-bit", level_idx);
        let selector_cipher: Aes128 = derive_level_cipher(seed, b"ml-init-sel", level_idx);
        let selectors: Vec<u8> = (0..spec.num_primary)
            .map(|j| initial_selector(&selector_cipher, j))
            .collect();

        let words: usize = record_size.div_ceil(8);
        let workers: usize = effective_threads(threads, spec.num_blocks);
        let level_bytes: usize = (spec.num_primary + 2 * spec.num_backup) * words * 8;
        let use_local: bool = if spec.backend != LevelBackend::FlatAes {
            false
        } else if workers == 1 {
            // Single-worker non-atomic buffer uses the same memory as AtomicU64 (no extra copies).
            true
        } else if level_bytes <= LOCAL_ACCUM_THRESHOLD_BYTES {
            let total_worker_bytes: usize = workers.saturating_mul(level_bytes);
            if total_worker_bytes <= *remaining_local_budget {
                *remaining_local_budget -= total_worker_bytes;
                true
            } else {
                false
            }
        } else {
            false
        };

        let accum: LevelParityAccum = if use_local {
            let bufs: Vec<LocalParityBuf> = (0..workers)
                .map(|_| LocalParityBuf {
                    reg: vec![0u64; spec.num_primary * words],
                    b0: vec![0u64; spec.num_backup * words],
                    b1: vec![0u64; spec.num_backup * words],
                })
                .collect();
            LevelParityAccum::Local(bufs)
        } else {
            LevelParityAccum::Atomic {
                reg: (0..spec.num_primary * words)
                    .map(|_| AtomicU64::new(0))
                    .collect(),
                b0: (0..spec.num_backup * words)
                    .map(|_| AtomicU64::new(0))
                    .collect(),
                b1: (0..spec.num_backup * words)
                    .map(|_| AtomicU64::new(0))
                    .collect(),
            }
        };

        Self {
            spec,
            record_size,
            words,
            block_key_cipher,
            color_cipher,
            selectors,
            workers,
            accum,
        }
    }

    /// Accumulates all complete (and final partial) blocks intersecting `entry_start..entry_end`.
    ///
    /// `entry_start` must be a multiple of `self.spec.block_size`, and `chunk` must contain
    /// `(entry_end - entry_start) * self.record_size` bytes.
    fn process_chunk(
        &mut self,
        chunk: &[u8],
        entry_start: usize,
        entry_end: usize,
        num_entries: usize,
    ) {
        debug_assert_eq!(entry_start % self.spec.block_size, 0);
        debug_assert_eq!(
            chunk.len(),
            entry_end.saturating_sub(entry_start) * self.record_size
        );
        if entry_start >= entry_end {
            return;
        }
        let block_start: usize = entry_start / self.spec.block_size;
        let block_end: usize = entry_end.div_ceil(self.spec.block_size);
        let chunk_workers: usize = effective_threads(self.workers, block_end - block_start);

        let spec: LevelSpec = self.spec;
        let r: usize = self.record_size;
        let words: usize = self.words;
        let block_key_cipher: &Aes128 = &self.block_key_cipher;
        let color_cipher: &Aes128 = &self.color_cipher;
        let selectors: &[u8] = &self.selectors;
        let next_block: AtomicUsize = AtomicUsize::new(block_start);

        match &mut self.accum {
            LevelParityAccum::Local(bufs) => {
                let active_bufs: &mut [LocalParityBuf] = &mut bufs[..chunk_workers];
                run_scoped_workers_mut(active_bufs, |buf| {
                    let mut entry: Vec<u64> = vec![0u64; words];
                    loop {
                        let block: usize = next_block.fetch_add(1, Ordering::Relaxed);
                        if block >= block_end {
                            break;
                        }
                        let key: PrfKey128 = derive_block_key_with(block_key_cipher, block);
                        let cipher: Aes128 = Aes128::new(&GenericArray::from(key));
                        let base_entry: usize = block * spec.block_size;

                        flat_aes_scan_all(
                            &cipher,
                            spec.block_size,
                            spec.total_hints(),
                            |h, color, off| {
                                let idx: usize = base_entry + off as usize;
                                if idx >= entry_end || idx >= num_entries {
                                    return;
                                }
                                let local_idx: usize = idx - entry_start;
                                load_record_words(
                                    &chunk[local_idx * r..(local_idx + 1) * r],
                                    &mut entry,
                                );
                                let j: usize = h as usize;
                                if j < spec.num_primary {
                                    if color == selectors[j] {
                                        xor_words(&mut buf.reg[j * words..(j + 1) * words], &entry);
                                    }
                                } else {
                                    let b: usize = j - spec.num_primary;
                                    let dst: &mut [u64] = if color == 0 {
                                        &mut buf.b0[b * words..(b + 1) * words]
                                    } else {
                                        &mut buf.b1[b * words..(b + 1) * words]
                                    };
                                    xor_words(dst, &entry);
                                }
                            },
                        );
                    }
                });
            }
            LevelParityAccum::Atomic { reg, b0, b1 } => {
                let reg_ref: &[AtomicU64] = reg;
                let b0_ref: &[AtomicU64] = b0;
                let b1_ref: &[AtomicU64] = b1;
                run_scoped_workers(chunk_workers, || {
                    let mut entry: Vec<u64> = vec![0u64; words];
                    loop {
                        let block: usize = next_block.fetch_add(1, Ordering::Relaxed);
                        if block >= block_end {
                            break;
                        }
                        let key: PrfKey128 = derive_block_key_with(block_key_cipher, block);
                        let base_entry: usize = block * spec.block_size;

                        match spec.backend {
                            LevelBackend::Iprf => {
                                let iprf: Iprf = Iprf::new(
                                    key,
                                    spec.total_hints() as u64,
                                    spec.block_size as u64,
                                );
                                let table: IprfTable<'_> = iprf.table();
                                let max_off: usize = spec
                                    .block_size
                                    .min(entry_end.saturating_sub(base_entry))
                                    .min(num_entries.saturating_sub(base_entry));
                                for offset in 0..max_off {
                                    let idx: usize = base_entry + offset;
                                    let local_idx: usize = idx - entry_start;
                                    load_record_words(
                                        &chunk[local_idx * r..(local_idx + 1) * r],
                                        &mut entry,
                                    );
                                    for h in table.inverse(offset as u64) {
                                        let j: usize = h as usize;
                                        let color: u8 =
                                            iprf_color_bit_with(color_cipher, block, h as u32);
                                        if j < spec.num_primary {
                                            if color == selectors[j] {
                                                xor_atomic(
                                                    &reg_ref[j * words..(j + 1) * words],
                                                    &entry,
                                                );
                                            }
                                        } else {
                                            let b: usize = j - spec.num_primary;
                                            let dst: &[AtomicU64] = if color == 0 {
                                                &b0_ref[b * words..(b + 1) * words]
                                            } else {
                                                &b1_ref[b * words..(b + 1) * words]
                                            };
                                            xor_atomic(dst, &entry);
                                        }
                                    }
                                }
                            }
                            LevelBackend::FlatAes => {
                                let cipher: Aes128 = Aes128::new(&GenericArray::from(key));
                                flat_aes_scan_all(
                                    &cipher,
                                    spec.block_size,
                                    spec.total_hints(),
                                    |h, color, off| {
                                        let idx: usize = base_entry + off as usize;
                                        if idx >= entry_end || idx >= num_entries {
                                            return;
                                        }
                                        let local_idx: usize = idx - entry_start;
                                        load_record_words(
                                            &chunk[local_idx * r..(local_idx + 1) * r],
                                            &mut entry,
                                        );
                                        let j: usize = h as usize;
                                        if j < spec.num_primary {
                                            if color == selectors[j] {
                                                xor_atomic(
                                                    &reg_ref[j * words..(j + 1) * words],
                                                    &entry,
                                                );
                                            }
                                        } else {
                                            let b: usize = j - spec.num_primary;
                                            let dst: &[AtomicU64] = if color == 0 {
                                                &b0_ref[b * words..(b + 1) * words]
                                            } else {
                                                &b1_ref[b * words..(b + 1) * words]
                                            };
                                            xor_atomic(dst, &entry);
                                        }
                                    },
                                );
                            }
                        }
                    }
                });
            }
        }
    }

    fn into_level_state(self) -> LevelState {
        let spec: LevelSpec = self.spec;
        let r: usize = self.record_size;
        let words: usize = self.words;

        let mut slots: Vec<PrimarySlot> = Vec::with_capacity(spec.num_primary);
        let mut hint_id_to_slot: Vec<u32> = vec![NO_SLOT; spec.total_hints()];
        for (j, (&selector, slot_ref)) in self
            .selectors
            .iter()
            .zip(hint_id_to_slot.iter_mut())
            .enumerate()
        {
            slots.push(PrimarySlot {
                hint_id: j as u32,
                selector,
                override_entry: None,
                reserved: false,
                retired: false,
            });
            *slot_ref = j as u32;
        }

        let mut primary_parities: Vec<u8> = vec![0u8; spec.num_primary * r];
        let mut backup_parities_0: Vec<u8> = vec![0u8; spec.num_backup * r];
        let mut backup_parities_1: Vec<u8> = vec![0u8; spec.num_backup * r];

        match self.accum {
            LevelParityAccum::Local(mut bufs) => {
                if let Some((first, rest)) = bufs.split_first_mut() {
                    for other in rest {
                        xor_words(&mut first.reg, &other.reg);
                        xor_words(&mut first.b0, &other.b0);
                        xor_words(&mut first.b1, &other.b1);
                    }
                    store_words_into_bytes(&first.reg, words, r, &mut primary_parities);
                    store_words_into_bytes(&first.b0, words, r, &mut backup_parities_0);
                    store_words_into_bytes(&first.b1, words, r, &mut backup_parities_1);
                }
            }
            LevelParityAccum::Atomic { reg, b0, b1 } => {
                store_atomic_into_bytes(&reg, words, r, &mut primary_parities);
                store_atomic_into_bytes(&b0, words, r, &mut backup_parities_0);
                store_atomic_into_bytes(&b1, words, r, &mut backup_parities_1);
            }
        }

        LevelState {
            spec,
            record_size: r,
            block_key_cipher: self.block_key_cipher,
            color_cipher: self.color_cipher,
            slots,
            primary_parities,
            backup_parities_0,
            backup_parities_1,
            backup_offset: 0,
            next_backup: 0,
            hint_id_to_slot,
            override_index: HashMap::new(),
        }
    }
}

/// Incremental, single-pass streaming `HintInit` builder for [`MultiLevelClient`].
///
/// Feeds raw database bytes across all levels in a single pass with bounded RAM. Because every
/// level's `block_size` is a power of two, their least common multiple equals `max(block_size)`;
/// aligned chunks are processed zero-copy across all levels simultaneously, and unaligned inputs
/// are buffered up to `chunk_bytes` rounded down to a whole superblock (`max(block_size) *
/// record_size`, minimum one superblock), or the exact database size when it fits in one
/// superblock.
///
/// Works on both native targets and `wasm32-unknown-unknown` (`--no-default-features`) with zero
/// host imports.
///
/// # Example
/// ```
/// use plinko::multilevel::{BackendVariant, MultiLevelHintBuilder, MultiLevelParams};
///
/// let params = MultiLevelParams::two_level(256, 8, 4, 16, BackendVariant::FlatB).unwrap();
/// let db = vec![0x5Au8; 256 * 8];
/// let seed = [0x11u8; 32];
/// let mut builder = MultiLevelHintBuilder::new(params, &seed, 1).unwrap();
/// for part in db.chunks(137) {
///     builder.push_bytes(part).unwrap();
/// }
/// let client = builder.finish().unwrap();
/// assert_eq!(client.epoch(), 0);
/// ```
pub struct MultiLevelHintBuilder {
    params: MultiLevelParams,
    levels: Vec<LevelAccumulator>,
    expected_bytes: usize,
    max_block_size: usize,
    superblock_bytes: usize,
    flush_bytes: usize,
    entries_processed: usize,
    pending_buf: Vec<u8>,
}

impl MultiLevelHintBuilder {
    /// Creates a streaming `HintInit` builder using [`DEFAULT_STREAM_CHUNK_BYTES`].
    pub fn new(
        params: MultiLevelParams,
        seed: &[u8; 32],
        threads: usize,
    ) -> Result<Self, MultiLevelError> {
        Self::with_chunk_bytes(params, seed, threads, DEFAULT_STREAM_CHUNK_BYTES)
    }

    /// Creates a streaming `HintInit` builder with a custom target chunk buffer size in bytes.
    ///
    /// `chunk_bytes` is rounded down to a positive multiple of `max(block_size) * record_size`
    /// (at least one full superblock of `max(block_size)` records), or the exact database size
    /// when `num_entries <= max(block_size)`.
    pub fn with_chunk_bytes(
        params: MultiLevelParams,
        seed: &[u8; 32],
        threads: usize,
        chunk_bytes: usize,
    ) -> Result<Self, MultiLevelError> {
        if chunk_bytes == 0 {
            return Err(MultiLevelError::InvalidParams("chunk_bytes must be > 0"));
        }
        let expected_bytes: usize = params
            .num_entries
            .checked_mul(params.record_size)
            .ok_or(MultiLevelError::InvalidParams("database size overflow"))?;
        let params: MultiLevelParams =
            MultiLevelParams::custom(params.num_entries, params.record_size, params.levels)?;

        let max_block_size: usize = params.levels.iter().map(|s| s.block_size).max().ok_or(
            MultiLevelError::InvalidParams("levels.len() must be in 1..=MAX_LEVELS"),
        )?;
        // If the entire database fits in a single superblock, use its exact byte length so valid
        // small databases with large nominal `block_size` still work on 32-bit targets (`wasm32`).
        let (superblock_bytes, flush_bytes): (usize, usize) =
            if params.num_entries <= max_block_size {
                (expected_bytes, expected_bytes)
            } else {
                let sb: usize = max_block_size
                    .checked_mul(params.record_size)
                    .ok_or(MultiLevelError::InvalidParams("superblock size overflow"))?;
                let superblocks_per_chunk: usize = (chunk_bytes / sb).max(1);
                let fb: usize = superblocks_per_chunk.checked_mul(sb).unwrap_or(sb);
                (sb, fb)
            };

        let mut remaining_local_budget: usize = LOCAL_ACCUM_MAX_WORKERS_BYTES;
        let levels: Vec<LevelAccumulator> = params
            .levels
            .iter()
            .enumerate()
            .map(|(i, &spec)| {
                LevelAccumulator::new(
                    spec,
                    params.record_size,
                    seed,
                    i,
                    threads,
                    &mut remaining_local_budget,
                )
            })
            .collect();

        Ok(Self {
            params,
            levels,
            expected_bytes,
            max_block_size,
            superblock_bytes,
            flush_bytes,
            entries_processed: 0,
            pending_buf: Vec::new(),
        })
    }

    /// Returns the total database byte length expected across all [`Self::push_bytes`] calls.
    pub fn expected_bytes(&self) -> usize {
        self.expected_bytes
    }

    /// Returns the number of database bytes accepted so far.
    pub fn bytes_fed(&self) -> usize {
        self.entries_processed * self.params.record_size + self.pending_buf.len()
    }

    /// Returns the aligned chunk size in bytes used for buffered flushes and reader pulls.
    fn flush_bytes(&self) -> usize {
        self.flush_bytes
    }

    /// Feeds the next contiguous slice of database bytes into the streaming `HintInit` pass.
    ///
    /// Slices may have arbitrary length (even unaligned to `record_size`). Pushing past
    /// [`Self::expected_bytes`] fails immediately with [`MultiLevelError::DatabaseLengthMismatch`].
    pub fn push_bytes(&mut self, mut data: &[u8]) -> Result<(), MultiLevelError> {
        let r: usize = self.params.record_size;
        let already: usize = self.entries_processed * r + self.pending_buf.len();
        let new_total: usize =
            already
                .checked_add(data.len())
                .ok_or(MultiLevelError::DatabaseLengthMismatch {
                    expected: self.expected_bytes,
                    actual: usize::MAX,
                })?;
        if new_total > self.expected_bytes {
            return Err(MultiLevelError::DatabaseLengthMismatch {
                expected: self.expected_bytes,
                actual: new_total,
            });
        }

        while !data.is_empty() {
            let remaining_unprocessed: usize = self.expected_bytes - self.entries_processed * r;
            if self.pending_buf.is_empty() {
                if data.len() == remaining_unprocessed {
                    // Exact final chunk (or full in-memory database): zero-copy flush.
                    let chunk_entries: usize = data.len() / r;
                    self.process_aligned_chunk(data, chunk_entries);
                    data = &[];
                } else if data.len() >= self.superblock_bytes {
                    // Zero-copy fast path for complete superblocks.
                    let full_sbs: usize = data.len() / self.superblock_bytes;
                    let take_bytes: usize = full_sbs * self.superblock_bytes;
                    self.process_aligned_chunk(&data[..take_bytes], full_sbs * self.max_block_size);
                    data = &data[take_bytes..];
                } else {
                    self.pending_buf.extend_from_slice(data);
                    data = &[];
                }
            } else {
                let target: usize = self.flush_bytes.min(remaining_unprocessed);
                let need: usize = target - self.pending_buf.len();
                let take: usize = data.len().min(need);
                self.pending_buf.extend_from_slice(&data[..take]);
                data = &data[take..];
                if self.pending_buf.len() == target {
                    let buf: Vec<u8> = std::mem::take(&mut self.pending_buf);
                    let chunk_entries: usize = target / r;
                    self.process_aligned_chunk(&buf, chunk_entries);
                    self.pending_buf = buf;
                    self.pending_buf.clear();
                }
            }
        }

        Ok(())
    }

    fn process_aligned_chunk(&mut self, chunk: &[u8], chunk_entries: usize) {
        let entry_start: usize = self.entries_processed;
        let entry_end: usize = entry_start + chunk_entries;
        let num_entries: usize = self.params.num_entries;
        for level in &mut self.levels {
            level.process_chunk(chunk, entry_start, entry_end, num_entries);
        }
        self.entries_processed = entry_end;
    }

    /// Finalizes streaming `HintInit` and returns the initialized [`MultiLevelClient`].
    ///
    /// Fails with [`MultiLevelError::DatabaseLengthMismatch`] if fewer than
    /// [`Self::expected_bytes`] bytes were pushed.
    pub fn finish(self) -> Result<MultiLevelClient, MultiLevelError> {
        let actual: usize = self.bytes_fed();
        if actual != self.expected_bytes || !self.pending_buf.is_empty() {
            return Err(MultiLevelError::DatabaseLengthMismatch {
                expected: self.expected_bytes,
                actual,
            });
        }
        let levels: Vec<LevelState> = self
            .levels
            .into_iter()
            .map(LevelAccumulator::into_level_state)
            .collect();
        Ok(MultiLevelClient {
            params: self.params,
            levels,
            session_id: next_session_id(),
            epoch: 0,
            next_query_id: 1,
            in_flight: HashMap::new(),
        })
    }
}

/// Wire query sent from client to server for one level.
#[derive(Debug, Clone)]
pub struct MultiLevelQuery {
    /// Zero-based index of the level that succeeded locally.
    pub level: usize,
    /// Edited color bit per chunk (0 or 1, length `num_blocks`).
    pub colors: Vec<u8>,
    /// Edited offset per chunk in `0..block_size` (length `num_blocks`).
    pub offsets: Vec<u32>,
}

/// Wire response returned by the server for a [`MultiLevelQuery`].
#[derive(Debug, Clone)]
pub struct MultiLevelAnswer {
    /// Server database snapshot epoch at which this answer was computed.
    /// Must match [`MultiLevelClient::epoch`] when passed to [`MultiLevelClient::finish_query`].
    pub snapshot_epoch: u64,
    /// XOR parity of all queried entries with `colors[a] == 0`.
    pub parity_0: Vec<u8>,
    /// XOR parity of all queried entries with `colors[a] == 1`.
    pub parity_1: Vec<u8>,
}

/// Client-side state for an in-flight query before [`MultiLevelClient::finish_query`].
#[derive(Debug, Clone)]
pub struct PendingMultiLevelQuery {
    /// Session identity of the [`MultiLevelClient`] that created this handle.
    pub session_id: u64,
    /// Unique reservation ID tying this query to its reserved primary slot.
    pub query_id: u64,
    /// Queried database index.
    pub index: usize,
    /// Level that matched `index`.
    pub level: usize,
    /// Primary slot index reserved on `level`.
    pub slot_idx: u32,
    /// Immutable hint ID (`0..total_hints`) of the reserved slot, logged to a WAL
    /// before transmission and passed to [`MultiLevelClient::retire_hints`] on recovery.
    pub hint_id: u32,
    /// Selector bit of the reserved slot.
    pub selector: u8,
    /// Wire query to send to the server.
    pub query: MultiLevelQuery,
}

impl PendingMultiLevelQuery {
    /// Returns the immutable hint identifier `(level, hint_id)` exposed by this query.
    ///
    /// Record this identifier durably in a write-ahead log before transmitting `query`
    /// so [`MultiLevelClient::retire_hints`] can prevent hint reuse after a crash.
    pub fn hint_id(&self) -> (usize, u32) {
        (self.level, self.hint_id)
    }
}

/// Active query reservation tracked by [`MultiLevelClient`].
#[derive(Debug, Clone, Copy)]
struct InFlightReservation {
    index: usize,
    level: usize,
    slot_idx: u32,
    hint_id: u32,
    selector: u8,
    epoch: u64,
}

/// Client for the multi-level PIR prototype.
pub struct MultiLevelClient {
    params: MultiLevelParams,
    levels: Vec<LevelState>,
    /// Unique per-instance session identity (renewed on reload) binding pending queries.
    session_id: u64,
    /// Monotonically incremented whenever deltas mutate the client's parities.
    epoch: u64,
    /// Next unique ID for `PendingMultiLevelQuery` reservations.
    next_query_id: u64,
    /// Active in-flight query reservations keyed by `query_id`.
    in_flight: HashMap<u64, InFlightReservation>,
}

impl MultiLevelClient {
    /// Allocates a client with zeroed parities (useful for `UpdateHint` and `Query`
    /// microbenchmarks at very large `num_entries` without materializing the full database).
    pub fn new_zeroed(params: MultiLevelParams, seed: &[u8; 32]) -> Self {
        let levels: Vec<LevelState> = params
            .levels
            .iter()
            .enumerate()
            .map(|(i, &spec)| LevelState::new_zeroed(spec, params.record_size, seed, i))
            .collect();
        Self {
            params,
            levels,
            session_id: next_session_id(),
            epoch: 0,
            next_query_id: 1,
            in_flight: HashMap::new(),
        }
    }

    /// Runs full `HintInit` across all levels on `threads` worker threads from an in-memory slice.
    ///
    /// Processes one level at a time so scratch parity accumulators for each level are released
    /// before allocating the next level's accumulators.
    pub fn hint_init(
        params: MultiLevelParams,
        seed: &[u8; 32],
        database: &[u8],
        threads: usize,
    ) -> Result<Self, MultiLevelError> {
        let expected: usize = params
            .num_entries
            .checked_mul(params.record_size)
            .ok_or(MultiLevelError::InvalidParams("database size overflow"))?;
        if database.len() != expected {
            return Err(MultiLevelError::DatabaseLengthMismatch {
                expected,
                actual: database.len(),
            });
        }
        let params: MultiLevelParams =
            MultiLevelParams::custom(params.num_entries, params.record_size, params.levels)?;
        let num_entries: usize = params.num_entries;
        let record_size: usize = params.record_size;
        let mut levels: Vec<LevelState> = Vec::with_capacity(params.levels.len());
        for (i, &spec) in params.levels.iter().enumerate() {
            let mut local_budget: usize = LOCAL_ACCUM_MAX_WORKERS_BYTES;
            let mut acc: LevelAccumulator =
                LevelAccumulator::new(spec, record_size, seed, i, threads, &mut local_budget);
            acc.process_chunk(database, 0, num_entries, num_entries);
            levels.push(acc.into_level_state());
        }
        Ok(Self {
            params,
            levels,
            session_id: next_session_id(),
            epoch: 0,
            next_query_id: 1,
            in_flight: HashMap::new(),
        })
    }

    /// Runs single-pass streaming `HintInit` from any EOF-terminated [`Read`] source using
    /// [`DEFAULT_STREAM_CHUNK_BYTES`] of database read buffer.
    ///
    /// Fails with [`MultiLevelError::DatabaseLengthMismatch`] if the stream terminates before
    /// `num_entries * record_size` bytes or contains trailing bytes beyond the expected length
    /// (on trailing bytes, `actual` is reported as a lower bound `expected.saturating_add(1)`
    /// without draining the stream).
    ///
    /// Because this method probes one byte past `num_entries * record_size` to detect trailing
    /// data, the reader must signal EOF after the database bytes. For length-delimited messages
    /// on a persistent connection, validate that the declared frame length equals
    /// `num_entries * record_size` first and then bound the read by that validated frame length
    /// (such as `reader.take(frame_len)` — noting that a bounded reader cannot detect trailing
    /// bytes beyond its bound), or feed chunks directly via [`MultiLevelHintBuilder::push_bytes`].
    pub fn hint_init_from_reader<R: Read>(
        params: MultiLevelParams,
        seed: &[u8; 32],
        reader: R,
        threads: usize,
    ) -> Result<Self, MultiLevelError> {
        Self::hint_init_from_reader_with_chunk_bytes(
            params,
            seed,
            reader,
            threads,
            DEFAULT_STREAM_CHUNK_BYTES,
        )
    }

    /// Runs single-pass streaming `HintInit` from any EOF-terminated [`Read`] source with a custom
    /// chunk buffer size in bytes (aligned to a positive multiple of `max(block_size) * record_size`
    /// and capped to the database size).
    ///
    /// Fails with [`MultiLevelError::DatabaseLengthMismatch`] if the stream ends early or has
    /// trailing bytes after `num_entries * record_size` (reporting `actual = expected.saturating_add(1)`
    /// as a lower bound without draining the stream). See [`Self::hint_init_from_reader`] for
    /// persistent-connection EOF guidance.
    pub fn hint_init_from_reader_with_chunk_bytes<R: Read>(
        params: MultiLevelParams,
        seed: &[u8; 32],
        mut reader: R,
        threads: usize,
        chunk_bytes: usize,
    ) -> Result<Self, MultiLevelError> {
        let mut builder: MultiLevelHintBuilder =
            MultiLevelHintBuilder::with_chunk_bytes(params, seed, threads, chunk_bytes)?;
        let expected: usize = builder.expected_bytes();
        let buf_len: usize = builder.flush_bytes().min(expected);
        let mut buf: Vec<u8> = Vec::new();
        buf.try_reserve_exact(buf_len)
            .map_err(|_| MultiLevelError::InvalidParams("chunk buffer allocation failed"))?;
        buf.resize(buf_len, 0);

        let mut remaining: usize = expected;
        while remaining > 0 {
            let want: usize = buf_len.min(remaining);
            let n: usize = read_fill(&mut reader, &mut buf[..want])?;
            if n < want {
                return Err(MultiLevelError::DatabaseLengthMismatch {
                    expected,
                    actual: expected - remaining + n,
                });
            }
            builder.push_bytes(&buf[..n])?;
            remaining -= n;
        }

        let mut extra: [u8; 1] = [0u8; 1];
        if read_fill(&mut reader, &mut extra)? > 0 {
            return Err(MultiLevelError::DatabaseLengthMismatch {
                expected,
                actual: expected.saturating_add(1),
            });
        }

        builder.finish()
    }

    /// Runs single-pass streaming `HintInit` directly from a database file on disk using
    /// [`DEFAULT_STREAM_CHUNK_BYTES`] of read buffer.
    ///
    /// For regular files, the file length is checked against `num_entries * record_size` before
    /// allocating hint accumulators. The file must remain unchanged for the entire call (the
    /// length check cannot detect concurrent same-length in-place writes), and the server must
    /// answer queries against that same snapshot.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn hint_init_from_file<P: AsRef<std::path::Path>>(
        params: MultiLevelParams,
        seed: &[u8; 32],
        path: P,
        threads: usize,
    ) -> Result<Self, MultiLevelError> {
        Self::hint_init_from_file_with_chunk_bytes(
            params,
            seed,
            path,
            threads,
            DEFAULT_STREAM_CHUNK_BYTES,
        )
    }

    /// Runs single-pass streaming `HintInit` directly from a database file on disk with a custom
    /// chunk buffer size in bytes.
    ///
    /// The input file must remain unchanged for the entire call (concurrent same-length edits are
    /// not detected by the length check), and the server must answer queries against that same
    /// snapshot.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn hint_init_from_file_with_chunk_bytes<P: AsRef<std::path::Path>>(
        params: MultiLevelParams,
        seed: &[u8; 32],
        path: P,
        threads: usize,
        chunk_bytes: usize,
    ) -> Result<Self, MultiLevelError> {
        if chunk_bytes == 0 {
            return Err(MultiLevelError::InvalidParams("chunk_bytes must be > 0"));
        }
        let expected: usize = params
            .num_entries
            .checked_mul(params.record_size)
            .ok_or(MultiLevelError::InvalidParams("database size overflow"))?;
        let file: std::fs::File = std::fs::File::open(path)?;
        let meta: std::fs::Metadata = file.metadata()?;
        if meta.is_file() {
            let file_len: u64 = meta.len();
            if file_len != expected as u64 {
                return Err(MultiLevelError::DatabaseLengthMismatch {
                    expected,
                    actual: usize::try_from(file_len).unwrap_or(usize::MAX),
                });
            }
        }
        Self::hint_init_from_reader_with_chunk_bytes(params, seed, file, threads, chunk_bytes)
    }

    /// Runs `HintInit` on the first `sample_blocks` blocks of level `level_idx` using an
    /// isolated scratch accumulator so no live client hint state is mutated, and returns
    /// `(fixed_duration, scan_duration)` separating one-time accumulator setup/writeback
    /// from per-block scanning work. Used to extrapolate `HintInit` throughput accurately.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn bench_level_blocks(
        params: &MultiLevelParams,
        seed: &[u8; 32],
        level_idx: usize,
        sample_db: &[u8],
        sample_blocks: usize,
        threads: usize,
    ) -> (std::time::Duration, std::time::Duration) {
        let spec: LevelSpec = params.levels[level_idx];
        let num_entries: usize = sample_blocks * spec.block_size;
        let sample_bytes: usize = num_entries * params.record_size;
        let sample_threads: usize = threads.min(sample_blocks.max(1));
        let t_total: std::time::Instant = std::time::Instant::now();
        let mut local_budget: usize = LOCAL_ACCUM_MAX_WORKERS_BYTES;
        let mut acc: LevelAccumulator = LevelAccumulator::new(
            spec,
            params.record_size,
            seed,
            level_idx,
            sample_threads,
            &mut local_budget,
        );
        let t_scan: std::time::Instant = std::time::Instant::now();
        acc.process_chunk(&sample_db[..sample_bytes], 0, num_entries, num_entries);
        let scan_ns: u64 = t_scan.elapsed().as_nanos() as u64;
        let _state: LevelState = acc.into_level_state();
        let total_ns: u64 = t_total.elapsed().as_nanos() as u64;
        let fixed_ns: u64 = total_ns.saturating_sub(scan_ns);
        (
            std::time::Duration::from_nanos(fixed_ns),
            std::time::Duration::from_nanos(scan_ns),
        )
    }

    /// Returns the protocol parameters.
    pub fn params(&self) -> &MultiLevelParams {
        &self.params
    }

    /// Returns the client's current database snapshot epoch (starts at `0` at `HintInit`,
    /// incremented on each `apply_delta` / `apply_deltas_batch`, and persisted across checkpoints).
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Returns remaining unused backups per level.
    pub fn remaining_backups(&self) -> Vec<usize> {
        self.levels
            .iter()
            .map(|l| l.spec.num_backup.saturating_sub(l.next_backup))
            .collect()
    }

    /// Returns the number of queries currently in flight.
    pub fn queries_in_flight(&self) -> usize {
        self.in_flight.len()
    }

    /// Estimated hint storage footprint in bytes (primary parities, 4B compact slot metadata,
    /// and backup parities). Excludes auxiliary runtime lookup indices such as `hint_id_to_slot`
    /// and `override_index`.
    pub fn hint_memory_bytes(&self) -> usize {
        self.params.total_hint_bytes()
    }

    /// Serializes this client's multi-level hint set (parameters, active primary slots,
    /// overrides, retired flags, primary parities, and remaining unused backup parities) in
    /// versioned binary format (`PLNKMLH1`) with a trailing 32-byte BLAKE3 checksum.
    ///
    /// Fails with [`MultiLevelError::QueryInFlight`] if any prepared query has not been
    /// completed via [`MultiLevelClient::finish_query`] or aborted via
    /// [`MultiLevelClient::abort_query`].
    ///
    /// The master seed is never written; callers must provide the same `seed` to
    /// [`MultiLevelClient::read_hints`] or [`MultiLevelClient::from_bytes`].
    pub fn write_hints<W: Write>(&self, writer: W) -> Result<(), MultiLevelError> {
        if !self.in_flight.is_empty() {
            return Err(MultiLevelError::QueryInFlight);
        }
        if self.levels.len() > MAX_LEVELS || self.params.record_size > u32::MAX as usize {
            return Err(MultiLevelError::InvalidParams(
                "hint set exceeds serialization format limits",
            ));
        }
        let mut out: HashingWriter<W> = HashingWriter::new(writer);

        let mut header: [u8; 36] = [0u8; 36];
        header[0..8].copy_from_slice(MAGIC);
        header[8..12].copy_from_slice(&FORMAT_VERSION.to_le_bytes());
        header[12..20].copy_from_slice(&(self.params.num_entries as u64).to_le_bytes());
        header[20..24].copy_from_slice(&(self.params.record_size as u32).to_le_bytes());
        header[24..28].copy_from_slice(&(self.levels.len() as u32).to_le_bytes());
        header[28..36].copy_from_slice(&self.epoch.to_le_bytes());
        out.write_hashed(&header)?;

        let r: usize = self.params.record_size;
        for level in &self.levels {
            let spec: LevelSpec = level.spec;
            let mut lvl_hdr: [u8; 36] = [0u8; 36];
            lvl_hdr[0..8].copy_from_slice(&(spec.block_size as u64).to_le_bytes());
            lvl_hdr[8..16].copy_from_slice(&(spec.num_blocks as u64).to_le_bytes());
            lvl_hdr[16..24].copy_from_slice(&(spec.num_primary as u64).to_le_bytes());
            lvl_hdr[24..32].copy_from_slice(&(spec.num_backup as u64).to_le_bytes());
            lvl_hdr[32..36].copy_from_slice(&(level.next_backup as u32).to_le_bytes());
            out.write_hashed(&lvl_hdr)?;

            let backend_byte: u8 = match spec.backend {
                LevelBackend::Iprf => 0,
                LevelBackend::FlatAes => 1,
            };
            out.write_hashed(&[backend_byte, 0, 0, 0])?;

            let mut slot_meta: Vec<u8> = Vec::with_capacity(spec.num_primary * 5);
            let mut overrides: Vec<(u32, u32, u32)> = Vec::new();
            for (slot_idx, slot) in level.slots.iter().enumerate() {
                slot_meta.extend_from_slice(&slot.hint_id.to_le_bytes());
                let mut flags: u8 = slot.selector & 1;
                if let Some(ov) = slot.override_entry {
                    flags |= 1 << 1;
                    flags |= (ov.color & 1) << 2;
                    overrides.push((slot_idx as u32, ov.block, ov.offset));
                }
                if slot.retired {
                    flags |= 1 << 3;
                }
                slot_meta.push(flags);
            }
            out.write_hashed(&slot_meta)?;

            out.write_hashed(&(overrides.len() as u32).to_le_bytes())?;
            if !overrides.is_empty() {
                let mut ov_buf: Vec<u8> = Vec::with_capacity(overrides.len() * 12);
                for (slot_idx, block, offset) in overrides {
                    ov_buf.extend_from_slice(&slot_idx.to_le_bytes());
                    ov_buf.extend_from_slice(&block.to_le_bytes());
                    ov_buf.extend_from_slice(&offset.to_le_bytes());
                }
                out.write_hashed(&ov_buf)?;
            }

            out.write_hashed(&level.primary_parities)?;
            let bak_start: usize = (level.next_backup - level.backup_offset) * r;
            out.write_hashed(&level.backup_parities_0[bak_start..])?;
            out.write_hashed(&level.backup_parities_1[bak_start..])?;
        }

        out.finish()?;
        Ok(())
    }

    /// Deserializes a multi-level hint set written by [`MultiLevelClient::write_hints`]
    /// using the default decoded memory budget (`512 MiB`), verifying structural invariants
    /// and the trailing BLAKE3 checksum.
    pub fn read_hints<R: Read>(reader: R, seed: &[u8; 32]) -> Result<Self, MultiLevelError> {
        Self::read_hints_with_limit(reader, seed, MAX_CHECKPOINT_BYTES)
    }

    /// Deserializes a multi-level hint set while enforcing `max_alloc_bytes` across all
    /// decoded structures and using fallible allocations so truncated or untrusted headers
    /// cannot exhaust memory.
    pub fn read_hints_with_limit<R: Read>(
        reader: R,
        seed: &[u8; 32],
        max_alloc_bytes: usize,
    ) -> Result<Self, MultiLevelError> {
        let mut inp: HashingReader<R> = HashingReader::new(reader);

        let mut header: [u8; 36] = [0u8; 36];
        inp.read_hashed(&mut header)?;
        if &header[0..8] != MAGIC {
            return Err(MultiLevelError::InvalidFormat("bad magic bytes"));
        }
        if u32_at(&header, 8) != FORMAT_VERSION {
            return Err(MultiLevelError::InvalidFormat("unsupported format version"));
        }
        let num_entries: usize = usize::try_from(u64_at(&header, 12))
            .map_err(|_| MultiLevelError::InvalidFormat("num_entries overflow"))?;
        let record_size: usize = u32_at(&header, 20) as usize;
        let num_levels: usize = u32_at(&header, 24) as usize;
        let epoch: u64 = u64_at(&header, 28);
        if num_levels == 0 || num_levels > MAX_LEVELS {
            return Err(MultiLevelError::InvalidFormat("num_levels out of range"));
        }

        let mut specs: Vec<LevelSpec> = Vec::with_capacity(num_levels);
        let mut levels: Vec<LevelState> = Vec::with_capacity(num_levels);
        let mut budget_used: usize = 0;

        let mut charge_budget = |bytes: usize| -> Result<(), MultiLevelError> {
            let next: usize =
                budget_used
                    .checked_add(bytes)
                    .ok_or(MultiLevelError::InvalidFormat(
                        "checkpoint size limit exceeded",
                    ))?;
            if next > max_alloc_bytes {
                return Err(MultiLevelError::InvalidFormat(
                    "checkpoint size limit exceeded",
                ));
            }
            budget_used = next;
            Ok(())
        };

        for level_idx in 0..num_levels {
            let mut lvl_hdr: [u8; 36] = [0u8; 36];
            inp.read_hashed(&mut lvl_hdr)?;
            let block_size: usize = usize::try_from(u64_at(&lvl_hdr, 0))
                .map_err(|_| MultiLevelError::InvalidFormat("block_size overflow"))?;
            let num_blocks: usize = usize::try_from(u64_at(&lvl_hdr, 8))
                .map_err(|_| MultiLevelError::InvalidFormat("num_blocks overflow"))?;
            let num_primary: usize = usize::try_from(u64_at(&lvl_hdr, 16))
                .map_err(|_| MultiLevelError::InvalidFormat("num_primary overflow"))?;
            let num_backup: usize = usize::try_from(u64_at(&lvl_hdr, 24))
                .map_err(|_| MultiLevelError::InvalidFormat("num_backup overflow"))?;
            let next_backup: usize = u32_at(&lvl_hdr, 32) as usize;

            let mut backend_buf: [u8; 4] = [0u8; 4];
            inp.read_hashed(&mut backend_buf)?;
            if backend_buf[1..4] != [0, 0, 0] {
                return Err(MultiLevelError::InvalidFormat("non-zero reserved bytes"));
            }
            let backend: LevelBackend = match backend_buf[0] {
                0 => LevelBackend::Iprf,
                1 => LevelBackend::FlatAes,
                _ => return Err(MultiLevelError::InvalidFormat("unknown backend tag")),
            };

            let spec: LevelSpec = LevelSpec {
                block_size,
                num_blocks,
                num_primary,
                num_backup,
                backend,
            };
            // Validate single-level geometry immediately before allocating buffers.
            MultiLevelParams::custom(num_entries, record_size, vec![spec])?;
            if next_backup > num_backup {
                return Err(MultiLevelError::InvalidFormat("next_backup > num_backup"));
            }
            let total_hints: usize = spec.total_hints();
            let remaining_backups: usize = num_backup - next_backup;

            let meta_len: usize =
                num_primary
                    .checked_mul(5)
                    .ok_or(MultiLevelError::InvalidFormat(
                        "slot metadata size overflow",
                    ))?;
            let id_map_bytes: usize = total_hints.checked_mul(std::mem::size_of::<u32>()).ok_or(
                MultiLevelError::InvalidFormat("hint_id_to_slot size overflow"),
            )?;
            let slot_vec_bytes: usize = num_primary
                .checked_mul(std::mem::size_of::<PrimarySlot>())
                .ok_or(MultiLevelError::InvalidFormat("slots size overflow"))?;
            let primary_bytes: usize = num_primary
                .checked_mul(record_size)
                .ok_or(MultiLevelError::InvalidFormat("primary parity overflow"))?;
            let rem_backup_bytes: usize = remaining_backups
                .checked_mul(record_size)
                .ok_or(MultiLevelError::InvalidFormat("backup parity overflow"))?;
            let level_alloc: usize = meta_len
                .checked_add(id_map_bytes)
                .and_then(|v| v.checked_add(slot_vec_bytes))
                .and_then(|v| v.checked_add(primary_bytes))
                .and_then(|v| v.checked_add(rem_backup_bytes))
                .and_then(|v| v.checked_add(rem_backup_bytes))
                .ok_or(MultiLevelError::InvalidFormat("level allocation overflow"))?;
            charge_budget(level_alloc)?;

            let mut slot_meta: Vec<u8> = try_alloc_zeroed(meta_len)?;
            inp.read_hashed(&mut slot_meta)?;

            let mut slots: Vec<PrimarySlot> = Vec::new();
            slots
                .try_reserve_exact(num_primary)
                .map_err(|_| MultiLevelError::InvalidFormat("allocation failed"))?;
            let mut hint_id_to_slot: Vec<u32> = try_alloc_filled(total_hints, NO_SLOT)?;
            let mut expected_overrides: Vec<(u32, u8)> = Vec::new();
            let mut override_flag_count: usize = 0;
            for slot_idx in 0..num_primary {
                let flags: u8 = slot_meta[slot_idx * 5 + 4];
                if (flags & (1 << 1)) != 0 {
                    override_flag_count += 1;
                }
            }
            if override_flag_count > 0 {
                let exp_ov_bytes: usize = override_flag_count
                    .checked_mul(std::mem::size_of::<(u32, u8)>())
                    .ok_or(MultiLevelError::InvalidFormat(
                        "override metadata size overflow",
                    ))?;
                charge_budget(exp_ov_bytes)?;
                expected_overrides
                    .try_reserve_exact(override_flag_count)
                    .map_err(|_| MultiLevelError::InvalidFormat("allocation failed"))?;
            }

            for slot_idx in 0..num_primary {
                let base: usize = slot_idx * 5;
                let hint_id: u32 = u32_at(&slot_meta, base);
                let flags: u8 = slot_meta[base + 4];
                if (flags & !0b1111) != 0 {
                    return Err(MultiLevelError::InvalidFormat("invalid slot flags"));
                }
                let h: usize = hint_id as usize;
                if h >= total_hints {
                    return Err(MultiLevelError::InvalidFormat("slot hint_id out of range"));
                }
                if h >= num_primary && (h - num_primary) >= next_backup {
                    return Err(MultiLevelError::InvalidFormat(
                        "promoted hint_id references unused backup",
                    ));
                }
                if hint_id_to_slot[h] != NO_SLOT {
                    return Err(MultiLevelError::InvalidFormat("duplicate active hint_id"));
                }
                hint_id_to_slot[h] = slot_idx as u32;

                let selector: u8 = flags & 1;
                let has_override: bool = (flags & (1 << 1)) != 0;
                let ov_color: u8 = (flags >> 2) & 1;
                let retired: bool = (flags & (1 << 3)) != 0;
                if !has_override && ov_color != 0 {
                    return Err(MultiLevelError::InvalidFormat(
                        "override color set without has_override flag",
                    ));
                }
                if has_override {
                    expected_overrides.push((slot_idx as u32, ov_color));
                }
                slots.push(PrimarySlot {
                    hint_id,
                    selector,
                    override_entry: None,
                    reserved: false,
                    retired,
                });
            }

            let mut ov_count_buf: [u8; 4] = [0u8; 4];
            inp.read_hashed(&mut ov_count_buf)?;
            let ov_count: usize = u32::from_le_bytes(ov_count_buf) as usize;
            if ov_count != expected_overrides.len() {
                return Err(MultiLevelError::InvalidFormat(
                    "override count does not match slot flags",
                ));
            }
            let mut override_index: HashMap<(u32, u32), Vec<u32>> = HashMap::new();
            if ov_count > 0 {
                let ov_bytes_len: usize = ov_count
                    .checked_mul(12)
                    .ok_or(MultiLevelError::InvalidFormat("override buffer overflow"))?;
                // Account for the HashMap bucket table, per-key Vec header, and per-override u32 payload.
                let per_ov_index_bytes: usize =
                    std::mem::size_of::<((u32, u32), Vec<u32>)>() + std::mem::size_of::<u32>() + 16;
                let ov_index_bytes: usize = ov_count
                    .checked_mul(per_ov_index_bytes)
                    .ok_or(MultiLevelError::InvalidFormat("override index overflow"))?;
                charge_budget(
                    ov_bytes_len
                        .checked_add(ov_index_bytes)
                        .ok_or(MultiLevelError::InvalidFormat("override budget overflow"))?,
                )?;
                override_index
                    .try_reserve(ov_count)
                    .map_err(|_| MultiLevelError::InvalidFormat("allocation failed"))?;
                let mut ov_buf: Vec<u8> = try_alloc_zeroed(ov_bytes_len)?;
                inp.read_hashed(&mut ov_buf)?;

                for (i, &(expected_slot_idx, ov_color)) in expected_overrides.iter().enumerate() {
                    let base: usize = i * 12;
                    let slot_idx: u32 = u32_at(&ov_buf, base);
                    let block: u32 = u32_at(&ov_buf, base + 4);
                    let offset: u32 = u32_at(&ov_buf, base + 8);
                    if slot_idx != expected_slot_idx {
                        return Err(MultiLevelError::InvalidFormat(
                            "override slot order mismatch",
                        ));
                    }
                    if block as usize >= num_blocks || offset as usize >= block_size {
                        return Err(MultiLevelError::InvalidFormat(
                            "override block or offset out of range",
                        ));
                    }
                    slots[slot_idx as usize].override_entry = Some(SlotOverride {
                        block,
                        offset,
                        color: ov_color,
                    });
                    let entry: &mut Vec<u32> = override_index.entry((block, offset)).or_default();
                    entry
                        .try_reserve(1)
                        .map_err(|_| MultiLevelError::InvalidFormat("allocation failed"))?;
                    entry.push(slot_idx);
                }
            }

            let mut primary_parities: Vec<u8> = try_alloc_zeroed(primary_bytes)?;
            inp.read_hashed(&mut primary_parities)?;

            let mut backup_parities_0: Vec<u8> = try_alloc_zeroed(rem_backup_bytes)?;
            inp.read_hashed(&mut backup_parities_0)?;

            let mut backup_parities_1: Vec<u8> = try_alloc_zeroed(rem_backup_bytes)?;
            inp.read_hashed(&mut backup_parities_1)?;

            specs.push(spec);
            levels.push(LevelState {
                spec,
                record_size,
                block_key_cipher: derive_level_cipher(seed, b"ml-block-key", level_idx),
                color_cipher: derive_level_cipher(seed, b"ml-color-bit", level_idx),
                slots,
                primary_parities,
                backup_parities_0,
                backup_parities_1,
                backup_offset: next_backup,
                next_backup,
                hint_id_to_slot,
                override_index,
            });
        }

        match inp.finish() {
            Ok(()) => {}
            Err(TrailerError::ChecksumMismatch) => return Err(MultiLevelError::ChecksumMismatch),
            Err(TrailerError::TrailingData) => return Err(MultiLevelError::TrailingData),
            Err(TrailerError::Io(err)) => return Err(MultiLevelError::Io(err.kind())),
        }

        let params: MultiLevelParams = MultiLevelParams::custom(num_entries, record_size, specs)?;
        Ok(Self {
            params,
            levels,
            session_id: next_session_id(),
            epoch,
            next_query_id: 1,
            in_flight: HashMap::new(),
        })
    }

    /// Computes the exact serialized byte length of this client's checkpoint, accounting
    /// only for active overrides and remaining unconsumed backups.
    fn serialized_byte_len(&self) -> Option<usize> {
        let r: usize = self.params.record_size;
        // 36-byte header + 32-byte BLAKE3 trailer.
        let mut total: usize = 36 + 32;
        for level in &self.levels {
            let spec: LevelSpec = level.spec;
            let remaining_backups: usize = spec.num_backup.saturating_sub(level.next_backup);
            let ov_count: usize = level
                .slots
                .iter()
                .filter(|s| s.override_entry.is_some())
                .count();
            // 36-byte level header + 4-byte backend/reserved + 4-byte ov_count.
            total = total.checked_add(44)?;
            total = total.checked_add(spec.num_primary.checked_mul(5)?)?;
            total = total.checked_add(ov_count.checked_mul(12)?)?;
            total = total.checked_add(spec.num_primary.checked_mul(r)?)?;
            let rem_backup_bytes: usize = remaining_backups.checked_mul(r)?;
            total = total.checked_add(rem_backup_bytes)?;
            total = total.checked_add(rem_backup_bytes)?;
        }
        Some(total)
    }

    /// Serializes this client's hint set into an in-memory byte vector.
    pub fn to_bytes(&self) -> Result<Vec<u8>, MultiLevelError> {
        let mut buf: Vec<u8> = Vec::new();
        if let Some(cap) = self.serialized_byte_len() {
            let _ = buf.try_reserve_exact(cap);
        }
        self.write_hints(&mut buf)?;
        Ok(buf)
    }

    /// Deserializes a client hint set from an in-memory byte slice using the default decoded
    /// memory budget (`512 MiB`).
    pub fn from_bytes(bytes: &[u8], seed: &[u8; 32]) -> Result<Self, MultiLevelError> {
        Self::read_hints(bytes, seed)
    }

    /// Deserializes a client hint set from an in-memory byte slice while enforcing
    /// `max_alloc_bytes` across decoded structures.
    pub fn from_bytes_with_limit(
        bytes: &[u8],
        seed: &[u8; 32],
        max_alloc_bytes: usize,
    ) -> Result<Self, MultiLevelError> {
        Self::read_hints_with_limit(bytes, seed, max_alloc_bytes)
    }

    /// Prepares a private query for `index` by probing levels `0..d` locally until one matches.
    ///
    /// The chosen primary slot is immediately reserved so a concurrent or retried call cannot
    /// emit a second query derived from the same hint. Complete the query via
    /// [`MultiLevelClient::finish_query`].
    pub fn prepare_query<R: Rng + ?Sized>(
        &mut self,
        index: usize,
        rng: &mut R,
    ) -> Result<PendingMultiLevelQuery, MultiLevelError> {
        if index >= self.params.num_entries {
            return Err(MultiLevelError::IndexOutOfRange {
                index,
                num_entries: self.params.num_entries,
            });
        }
        let mut scratch: Vec<(u32, u8)> = Vec::new();
        for level_idx in 0..self.levels.len() {
            let matches: Vec<u32> = self.levels[level_idx].find_matching_slots(index, &mut scratch);
            if matches.is_empty() {
                continue;
            }
            let slot_idx: u32 = matches[rng.gen_range(0..matches.len())];
            self.levels[level_idx].slots[slot_idx as usize].reserved = true;

            let level: &LevelState = &self.levels[level_idx];
            let slot: &PrimarySlot = &level.slots[slot_idx as usize];
            let selector: u8 = slot.selector;
            let hint_id: u32 = slot.hint_id;
            let override_entry: Option<SlotOverride> = slot.override_entry;
            let target_block: usize = index / level.spec.block_size;

            let mut colors: Vec<u8> = Vec::with_capacity(level.spec.num_blocks);
            let mut offsets: Vec<u32> = Vec::with_capacity(level.spec.num_blocks);

            for blk in 0..level.spec.num_blocks {
                if blk == target_block {
                    colors.push(1 - selector);
                    offsets.push(rng.gen_range(0..level.spec.block_size as u32));
                } else if let Some(ov) = override_entry {
                    if ov.block as usize == blk {
                        colors.push(ov.color);
                        offsets.push(ov.offset);
                    } else {
                        let (c, off) = level.forward(blk, hint_id);
                        colors.push(c);
                        offsets.push(off);
                    }
                } else {
                    let (c, off) = level.forward(blk, hint_id);
                    colors.push(c);
                    offsets.push(off);
                }
            }

            let query_id: u64 = self.next_query_id;
            self.next_query_id = self.next_query_id.wrapping_add(1);
            self.in_flight.insert(
                query_id,
                InFlightReservation {
                    index,
                    level: level_idx,
                    slot_idx,
                    hint_id,
                    selector,
                    epoch: self.epoch,
                },
            );

            return Ok(PendingMultiLevelQuery {
                session_id: self.session_id,
                query_id,
                index,
                level: level_idx,
                slot_idx,
                hint_id,
                selector,
                query: MultiLevelQuery {
                    level: level_idx,
                    colors,
                    offsets,
                },
            });
        }

        Err(MultiLevelError::AllLevelsFailed { index })
    }

    /// Aborts an abandoned in-flight query (for example after a network timeout) without
    /// releasing its exposed primary slot back into circulation.
    ///
    /// The slot is permanently marked `retired` so no future query can reuse the hint
    /// that may have reached the server, while removing the in-flight reservation so
    /// subsequent [`MultiLevelClient::write_hints`] checkpoints succeed.
    pub fn abort_query(&mut self, pending: &PendingMultiLevelQuery) -> Result<(), MultiLevelError> {
        if pending.session_id != self.session_id {
            return Err(MultiLevelError::UnknownPendingQuery {
                query_id: pending.query_id,
            });
        }
        let Some(&res) = self.in_flight.get(&pending.query_id) else {
            return Err(MultiLevelError::UnknownPendingQuery {
                query_id: pending.query_id,
            });
        };
        if res.index != pending.index
            || res.level != pending.level
            || res.slot_idx != pending.slot_idx
            || res.hint_id != pending.hint_id
            || res.selector != pending.selector
        {
            return Err(MultiLevelError::UnknownPendingQuery {
                query_id: pending.query_id,
            });
        }
        self.in_flight.remove(&pending.query_id);
        let slot: &mut PrimarySlot = &mut self.levels[res.level].slots[res.slot_idx as usize];
        slot.reserved = false;
        slot.retired = true;
        Ok(())
    }

    /// Cancels an unsent in-flight query without retiring its primary slot.
    ///
    /// # Security
    /// Call this **only** if `pending.query` has never left the client (for example when
    /// rolling back a multi-segment or batch preparation after a later segment fails locally).
    /// If the query may have reached the server, call [`MultiLevelClient::abort_query`]
    /// instead so the exposed hint is permanently retired.
    pub fn cancel_unsent_query(
        &mut self,
        pending: &PendingMultiLevelQuery,
    ) -> Result<(), MultiLevelError> {
        if pending.session_id != self.session_id {
            return Err(MultiLevelError::UnknownPendingQuery {
                query_id: pending.query_id,
            });
        }
        let Some(&res) = self.in_flight.get(&pending.query_id) else {
            return Err(MultiLevelError::UnknownPendingQuery {
                query_id: pending.query_id,
            });
        };
        if res.index != pending.index
            || res.level != pending.level
            || res.slot_idx != pending.slot_idx
            || res.hint_id != pending.hint_id
            || res.selector != pending.selector
        {
            return Err(MultiLevelError::UnknownPendingQuery {
                query_id: pending.query_id,
            });
        }
        self.in_flight.remove(&pending.query_id);
        self.levels[res.level].slots[res.slot_idx as usize].reserved = false;
        Ok(())
    }

    /// Validates that every `(level_idx, hint_id)` belongs to this client without mutating state.
    pub(crate) fn validate_retire_hints(
        &self,
        hints: &[(usize, u32)],
    ) -> Result<(), MultiLevelError> {
        for &(level_idx, hint_id) in hints {
            let Some(level) = self.levels.get(level_idx) else {
                return Err(MultiLevelError::InvalidParams(
                    "hint to retire references an out-of-range level",
                ));
            };
            if (hint_id as usize) >= level.spec.total_hints() {
                return Err(MultiLevelError::InvalidParams(
                    "hint_id to retire is out of range for its level",
                ));
            }
        }
        Ok(())
    }

    /// Retires WAL-logged `(level, hint_id)` identifiers after reloading a checkpoint so
    /// neither original primary hints nor promoted backup hints transmitted since that
    /// checkpoint can ever be reused.
    ///
    /// Validates every entry before mutating client state. Any active primary slot currently
    /// holding `hint_id` is marked `retired` (and any matching in-flight reservation is
    /// cleared); any backup `hint_id >= num_primary` advances that level's `next_backup`
    /// past it so the backup can never be promoted again.
    pub fn retire_hints(&mut self, hints: &[(usize, u32)]) -> Result<(), MultiLevelError> {
        self.validate_retire_hints(hints)?;
        for &(level_idx, hint_id) in hints {
            let level: &mut LevelState = &mut self.levels[level_idx];
            let h: usize = hint_id as usize;
            let slot_idx: u32 = level.hint_id_to_slot[h];
            if slot_idx != NO_SLOT {
                let slot: &mut PrimarySlot = &mut level.slots[slot_idx as usize];
                slot.reserved = false;
                slot.retired = true;
                self.in_flight
                    .retain(|_, res| !(res.level == level_idx && res.slot_idx == slot_idx));
            }
            if let Some(backup_idx) = h.checked_sub(level.spec.num_primary) {
                if backup_idx >= level.next_backup {
                    level.next_backup = backup_idx + 1;
                }
            }
        }
        Ok(())
    }

    /// Validates `answer` against `pending` and current client state without mutating hints.
    pub(crate) fn validate_answer(
        &self,
        pending: &PendingMultiLevelQuery,
        answer: &MultiLevelAnswer,
    ) -> Result<(), MultiLevelError> {
        let r: usize = self.params.record_size;
        if answer.parity_0.len() != r || answer.parity_1.len() != r {
            return Err(MultiLevelError::AnswerLengthMismatch {
                expected: r,
                actual: (answer.parity_0.len(), answer.parity_1.len()),
            });
        }
        if pending.session_id != self.session_id {
            return Err(MultiLevelError::UnknownPendingQuery {
                query_id: pending.query_id,
            });
        }
        let Some(&res) = self.in_flight.get(&pending.query_id) else {
            return Err(MultiLevelError::UnknownPendingQuery {
                query_id: pending.query_id,
            });
        };
        if res.index != pending.index
            || res.level != pending.level
            || res.slot_idx != pending.slot_idx
            || res.hint_id != pending.hint_id
            || res.selector != pending.selector
        {
            return Err(MultiLevelError::UnknownPendingQuery {
                query_id: pending.query_id,
            });
        }
        if res.epoch != self.epoch {
            return Err(MultiLevelError::SnapshotMismatch {
                query_epoch: res.epoch,
                current_epoch: self.epoch,
            });
        }
        if answer.snapshot_epoch != self.epoch {
            return Err(MultiLevelError::SnapshotMismatch {
                query_epoch: answer.snapshot_epoch,
                current_epoch: self.epoch,
            });
        }
        if self.levels[res.level].next_backup >= self.levels[res.level].spec.num_backup {
            return Err(MultiLevelError::BackupExhausted {
                level: res.level,
                recovered: None,
            });
        }
        Ok(())
    }

    /// Validates a batch of `(pending, answer)` pairs without mutating hints, including
    /// duplicate `query_id` rejection and minimum per-level backup demand checks.
    pub(crate) fn validate_batch_answers(
        &self,
        pendings: &[&PendingMultiLevelQuery],
        answers: &[MultiLevelAnswer],
    ) -> Result<(), MultiLevelError> {
        if pendings.len() != answers.len() {
            return Err(MultiLevelError::InvalidParams(
                "pending query count does not match answer count",
            ));
        }
        let mut seen_ids: std::collections::HashSet<u64> =
            std::collections::HashSet::with_capacity(pendings.len());
        let mut min_backup_demand: Vec<usize> = vec![0usize; self.levels.len()];
        for (&pending, answer) in pendings.iter().zip(answers.iter()) {
            if !seen_ids.insert(pending.query_id) {
                return Err(MultiLevelError::UnknownPendingQuery {
                    query_id: pending.query_id,
                });
            }
            self.validate_answer(pending, answer)?;
            min_backup_demand[pending.level] += 1;
        }
        for (lvl_idx, &need) in min_backup_demand.iter().enumerate() {
            let rem: usize = self.levels[lvl_idx]
                .spec
                .num_backup
                .saturating_sub(self.levels[lvl_idx].next_backup);
            if need > rem {
                return Err(MultiLevelError::BackupExhausted {
                    level: lvl_idx,
                    recovered: None,
                });
            }
        }
        Ok(())
    }

    /// Reconstructs `DB[index]` from the server's answer and runs failure-private `Refresh`
    /// (Fig. 3, Lemma 9) on all failed levels `0..pending.level` and on `pending.level`.
    ///
    /// Borrows `pending` so the caller retains ownership to retry or call
    /// [`MultiLevelClient::abort_query`] if `answer` has invalid parity lengths or a mismatched
    /// server snapshot epoch. Validates answer parity lengths, session/reservation binding,
    /// client and server snapshot epochs, and backup availability across all affected levels
    /// before mutating any hint slot.
    pub fn finish_query<R: Rng + ?Sized>(
        &mut self,
        pending: &PendingMultiLevelQuery,
        answer: &MultiLevelAnswer,
        rng: &mut R,
    ) -> Result<Vec<u8>, MultiLevelError> {
        let r: usize = self.params.record_size;
        if answer.parity_0.len() != r || answer.parity_1.len() != r {
            return Err(MultiLevelError::AnswerLengthMismatch {
                expected: r,
                actual: (answer.parity_0.len(), answer.parity_1.len()),
            });
        }

        if pending.session_id != self.session_id {
            return Err(MultiLevelError::UnknownPendingQuery {
                query_id: pending.query_id,
            });
        }
        let Some(&res) = self.in_flight.get(&pending.query_id) else {
            return Err(MultiLevelError::UnknownPendingQuery {
                query_id: pending.query_id,
            });
        };
        if res.index != pending.index
            || res.level != pending.level
            || res.slot_idx != pending.slot_idx
            || res.hint_id != pending.hint_id
            || res.selector != pending.selector
        {
            return Err(MultiLevelError::UnknownPendingQuery {
                query_id: pending.query_id,
            });
        }
        if res.epoch != self.epoch {
            // Retire the slot permanently: the wire query may have been sent.
            self.in_flight.remove(&pending.query_id);
            let slot: &mut PrimarySlot = &mut self.levels[res.level].slots[res.slot_idx as usize];
            slot.reserved = false;
            slot.retired = true;
            return Err(MultiLevelError::SnapshotMismatch {
                query_epoch: res.epoch,
                current_epoch: self.epoch,
            });
        }
        if answer.snapshot_epoch != self.epoch {
            // Reject stale/ahead server answer without consuming the reservation so caller
            // can retry against an up-to-date server snapshot or call `abort_query`.
            return Err(MultiLevelError::SnapshotMismatch {
                query_epoch: answer.snapshot_epoch,
                current_epoch: self.epoch,
            });
        }

        let s: usize = res.slot_idx as usize;
        let mut value: Vec<u8> =
            self.levels[res.level].primary_parities[s * r..(s + 1) * r].to_vec();
        let server_parity: &[u8] = if res.selector == 0 {
            &answer.parity_0
        } else {
            &answer.parity_1
        };
        xor_bytes(&mut value, server_parity);

        // Plan all refresh replacements and verify backup budgets BEFORE mutating any level.
        let mut failed_plans: Vec<Vec<u32>> = Vec::with_capacity(res.level);
        for lvl_idx in 0..res.level {
            let level: &LevelState = &self.levels[lvl_idx];
            let avail_slots: Vec<u32> = (0..level.spec.num_primary as u32)
                .filter(|&pos| {
                    let slot: &PrimarySlot = &level.slots[pos as usize];
                    !slot.reserved && !slot.retired
                })
                .collect();
            let p_in: f64 = 1.0 / (2.0 * level.spec.block_size as f64);
            // Sample over `avail_slots.len()` so each active, non-reserved slot keeps exact
            // per-slot probability `1 / (2 * w_i)` even when other slots are reserved or retired.
            let k: usize = sample_binomial_exact(avail_slots.len(), p_in, rng);
            let chosen: Vec<u32> = sample_distinct_from(&avail_slots, k, rng);
            let remaining: usize = level.spec.num_backup.saturating_sub(level.next_backup);
            if chosen.len() > remaining {
                self.in_flight.remove(&pending.query_id);
                let slot: &mut PrimarySlot =
                    &mut self.levels[res.level].slots[res.slot_idx as usize];
                slot.reserved = false;
                slot.retired = true;
                return Err(MultiLevelError::BackupExhausted {
                    level: lvl_idx,
                    recovered: Some(value),
                });
            }
            failed_plans.push(chosen);
        }

        // Plan the successful level's replacement + optional PromoteOut sweep.
        let (promote_out_slots, needed_succ_backups): (Vec<u32>, usize) = {
            let level: &LevelState = &self.levels[res.level];
            let active_slots: usize = level
                .slots
                .iter()
                .enumerate()
                .filter(|&(pos, slot)| !slot.retired && (!slot.reserved || pos == s))
                .count();
            let neg_log_p0: f64 =
                -(active_slots as f64) * (-1.0 / (2.0 * level.spec.block_size as f64)).ln_1p();
            if sample_bernoulli_neg_exp(neg_log_p0, rng) {
                let mut scratch: Vec<(u32, u8)> = Vec::new();
                // `find_matching_slots` skips `res.slot_idx` because it is reserved; after
                // PromoteIn replaces `res.slot_idx`, it will also contain `res.index`.
                let mut containing: Vec<u32> = level.find_matching_slots(res.index, &mut scratch);
                containing.push(res.slot_idx);
                let needed: usize = 1 + containing.len();
                (containing, needed)
            } else {
                (Vec::new(), 1)
            }
        };
        let succ_remaining: usize = self.levels[res.level]
            .spec
            .num_backup
            .saturating_sub(self.levels[res.level].next_backup);
        if needed_succ_backups > succ_remaining {
            self.in_flight.remove(&pending.query_id);
            let slot: &mut PrimarySlot = &mut self.levels[res.level].slots[res.slot_idx as usize];
            slot.reserved = false;
            slot.retired = true;
            return Err(MultiLevelError::BackupExhausted {
                level: res.level,
                recovered: Some(value),
            });
        }

        // All backup budgets verified; commit the reservation removal and atomic refresh.
        self.in_flight.remove(&pending.query_id);

        for (lvl_idx, chosen) in failed_plans.into_iter().enumerate() {
            let level: &mut LevelState = &mut self.levels[lvl_idx];
            let x_block: u32 = (res.index / level.spec.block_size) as u32;
            let x_offset: u32 = (res.index % level.spec.block_size) as u32;
            for slot_pos in chosen {
                let (new_slot, new_parity) =
                    level.promote_in(lvl_idx, x_block, x_offset, &value)?;
                level.install_slot(slot_pos, new_slot, &new_parity);
            }
        }

        {
            let lvl_idx: usize = res.level;
            let level: &mut LevelState = &mut self.levels[lvl_idx];
            let x_block: u32 = (res.index / level.spec.block_size) as u32;
            let x_offset: u32 = (res.index % level.spec.block_size) as u32;

            let (repl_slot, repl_parity) = level.promote_in(lvl_idx, x_block, x_offset, &value)?;
            level.install_slot(res.slot_idx, repl_slot, &repl_parity);

            for slot_pos in promote_out_slots {
                let (out_slot, out_parity) = level.promote_out(lvl_idx, x_block, x_offset, rng)?;
                level.install_slot(slot_pos, out_slot, &out_parity);
            }
        }

        Ok(value)
    }

    /// Applies one record update `DB[index] ^= delta` across all levels and advances
    /// the client snapshot epoch.
    pub fn apply_delta(&mut self, index: usize, delta: &[u8]) -> Result<(), MultiLevelError> {
        if index >= self.params.num_entries {
            return Err(MultiLevelError::IndexOutOfRange {
                index,
                num_entries: self.params.num_entries,
            });
        }
        if delta.len() != self.params.record_size {
            return Err(MultiLevelError::DeltaLengthMismatch {
                expected: self.params.record_size,
                actual: delta.len(),
            });
        }
        let mut scratch: Vec<(u32, u8)> = Vec::new();
        for level in &mut self.levels {
            level.apply_delta(index, delta, &mut scratch);
        }
        self.epoch = self.epoch.wrapping_add(1);
        Ok(())
    }

    /// Applies a batch of `(index, delta)` updates across all levels, computing per-level
    /// inverse matches in parallel on `threads` workers before XORing into parities.
    pub fn apply_deltas_batch(
        &mut self,
        updates: &[(usize, Vec<u8>)],
        threads: usize,
    ) -> Result<(), MultiLevelError> {
        for (idx, delta) in updates {
            if *idx >= self.params.num_entries {
                return Err(MultiLevelError::IndexOutOfRange {
                    index: *idx,
                    num_entries: self.params.num_entries,
                });
            }
            if delta.len() != self.params.record_size {
                return Err(MultiLevelError::DeltaLengthMismatch {
                    expected: self.params.record_size,
                    actual: delta.len(),
                });
            }
        }
        if updates.is_empty() {
            return Ok(());
        }

        let workers: usize = effective_threads(threads, updates.len());
        if workers <= 1 {
            let mut scratch: Vec<(u32, u8)> = Vec::new();
            for (idx, delta) in updates {
                for level in &mut self.levels {
                    level.apply_delta(*idx, delta, &mut scratch);
                }
            }
            self.epoch = self.epoch.wrapping_add(1);
            return Ok(());
        }

        // Parallelize per-level inverse discovery across updates, then apply XORs sequentially.
        #[cfg(not(target_arch = "wasm32"))]
        for level in &mut self.levels {
            type UpdateHits = (usize, Vec<(u32, u8)>);
            let chunk_size: usize = updates.len().div_ceil(workers);
            let spec: LevelSpec = level.spec;
            let block_key_cipher: &Aes128 = &level.block_key_cipher;
            let color_cipher: &Aes128 = &level.color_cipher;

            let compute_chunk = |base: usize, chunk: &[(usize, Vec<u8>)]| -> Vec<UpdateHits> {
                let mut results: Vec<UpdateHits> = Vec::with_capacity(chunk.len());
                for (local_i, &(entry_idx, _)) in chunk.iter().enumerate() {
                    let block: usize = entry_idx / spec.block_size;
                    let offset: u32 = (entry_idx % spec.block_size) as u32;
                    let key: PrfKey128 = derive_block_key_with(block_key_cipher, block);
                    let mut hits: Vec<(u32, u8)> = Vec::new();
                    match spec.backend {
                        LevelBackend::Iprf => {
                            let iprf: Iprf =
                                Iprf::new(key, spec.total_hints() as u64, spec.block_size as u64);
                            for h in iprf.inverse(offset as u64) {
                                let c: u8 = iprf_color_bit_with(color_cipher, block, h as u32);
                                hits.push((h as u32, c));
                            }
                        }
                        LevelBackend::FlatAes => {
                            let cipher: Aes128 = Aes128::new(&GenericArray::from(key));
                            flat_aes_inverse(
                                &cipher,
                                spec.block_size,
                                spec.total_hints(),
                                offset,
                                &mut hits,
                            );
                        }
                    }
                    results.push((base + local_i, hits));
                }
                results
            };
            let compute_chunk_ref = &compute_chunk;

            let per_chunk: Vec<Vec<UpdateHits>> = std::thread::scope(|s| {
                let mut handles = Vec::new();
                let mut inline_results: Vec<Vec<UpdateHits>> = Vec::new();
                for (chunk_idx, chunk) in updates.chunks(chunk_size).enumerate() {
                    let base: usize = chunk_idx * chunk_size;
                    match std::thread::Builder::new()
                        .spawn_scoped(s, move || compute_chunk_ref(base, chunk))
                    {
                        Ok(handle) => handles.push(handle),
                        Err(_) => inline_results.push(compute_chunk_ref(base, chunk)),
                    }
                }
                for h in handles {
                    inline_results.push(h.join().unwrap());
                }
                inline_results
            });

            let r: usize = level.record_size;
            for chunk_res in per_chunk {
                for (upd_i, hits) in chunk_res {
                    let (entry_idx, ref delta) = updates[upd_i];
                    let block: usize = entry_idx / spec.block_size;
                    let offset: u32 = (entry_idx % spec.block_size) as u32;

                    for &(hint_id, color_old) in &hits {
                        let h: usize = hint_id as usize;
                        let slot_idx: u32 = level.hint_id_to_slot[h];
                        if slot_idx != NO_SLOT {
                            let slot: &PrimarySlot = &level.slots[slot_idx as usize];
                            if let Some(ov) = slot.override_entry {
                                if ov.block as usize == block {
                                    continue;
                                }
                            }
                            if color_old == slot.selector {
                                let s: usize = slot_idx as usize;
                                xor_bytes(&mut level.primary_parities[s * r..(s + 1) * r], delta);
                            }
                        } else if h >= spec.num_primary {
                            let b: usize = h - spec.num_primary;
                            if b >= level.next_backup {
                                let local_b: usize = b - level.backup_offset;
                                let target: &mut [u8] = if color_old == 0 {
                                    &mut level.backup_parities_0[local_b * r..(local_b + 1) * r]
                                } else {
                                    &mut level.backup_parities_1[local_b * r..(local_b + 1) * r]
                                };
                                xor_bytes(target, delta);
                            }
                        }
                    }

                    if let Some(ov_slots) = level.override_index.get(&(block as u32, offset)) {
                        for &slot_idx in ov_slots {
                            let slot: &PrimarySlot = &level.slots[slot_idx as usize];
                            if let Some(ov) = slot.override_entry {
                                if ov.color == slot.selector {
                                    let s: usize = slot_idx as usize;
                                    xor_bytes(
                                        &mut level.primary_parities[s * r..(s + 1) * r],
                                        delta,
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
        self.epoch = self.epoch.wrapping_add(1);
        Ok(())
    }
}

/// Stateless server answer routine for [`MultiLevelQuery`].
pub struct MultiLevelServer;

impl MultiLevelServer {
    /// Validates `query` and `database` bounds, then computes both color parities
    /// `(parity_0, parity_1)` tagged with the server's `snapshot_epoch`.
    pub fn answer(
        database: &[u8],
        params: &MultiLevelParams,
        query: &MultiLevelQuery,
        snapshot_epoch: u64,
    ) -> Result<MultiLevelAnswer, MultiLevelError> {
        let expected_db: usize = params
            .num_entries
            .checked_mul(params.record_size)
            .ok_or(MultiLevelError::InvalidParams("database size overflow"))?;
        if database.len() != expected_db {
            return Err(MultiLevelError::DatabaseLengthMismatch {
                expected: expected_db,
                actual: database.len(),
            });
        }
        let spec: LevelSpec = *params
            .levels
            .get(query.level)
            .ok_or(MultiLevelError::InvalidQuery("query level out of range"))?;
        if query.colors.len() != spec.num_blocks || query.offsets.len() != spec.num_blocks {
            return Err(MultiLevelError::InvalidQuery(
                "query colors/offsets length does not match level num_blocks",
            ));
        }

        let r: usize = params.record_size;
        let mut parity_0: Vec<u8> = vec![0u8; r];
        let mut parity_1: Vec<u8> = vec![0u8; r];

        for blk in 0..spec.num_blocks {
            let color: u8 = query.colors[blk];
            let offset: usize = query.offsets[blk] as usize;
            if color > 1 {
                return Err(MultiLevelError::InvalidQuery("query color must be 0 or 1"));
            }
            if offset >= spec.block_size {
                return Err(MultiLevelError::InvalidQuery(
                    "query offset out of block_size bounds",
                ));
            }
            let idx: usize = blk * spec.block_size + offset;
            if idx >= params.num_entries {
                continue;
            }
            let rec: &[u8] = &database[idx * r..(idx + 1) * r];
            if color == 0 {
                xor_bytes(&mut parity_0, rec);
            } else {
                xor_bytes(&mut parity_1, rec);
            }
        }

        Ok(MultiLevelAnswer {
            snapshot_epoch,
            parity_0,
            parity_1,
        })
    }
}

fn derive_level_cipher(seed: &[u8; 32], domain_sep: &[u8], level_idx: usize) -> Aes128 {
    let mut hasher = blake3::Hasher::new_keyed(seed);
    hasher.update(domain_sep);
    hasher.update(&(level_idx as u64).to_le_bytes());
    let digest = hasher.finalize();
    let key: [u8; 16] = digest.as_bytes()[..16].try_into().unwrap();
    Aes128::new(&GenericArray::from(key))
}

#[inline]
fn derive_block_key_with(cipher: &Aes128, block: usize) -> PrfKey128 {
    let mut blk: Block = Block::default();
    blk[0..8].copy_from_slice(&(block as u64).to_le_bytes());
    cipher.encrypt_block(&mut blk);
    blk.into()
}

#[inline]
fn iprf_color_bit_with(cipher: &Aes128, block: usize, hint_id: u32) -> u8 {
    let mut blk: Block = Block::default();
    blk[0..8].copy_from_slice(&(block as u64).to_le_bytes());
    blk[8..16].copy_from_slice(&((hint_id >> 7) as u64).to_le_bytes());
    cipher.encrypt_block(&mut blk);
    let bit_pos: usize = (hint_id & 127) as usize;
    (blk[bit_pos >> 3] >> (bit_pos & 7)) & 1
}

#[inline]
fn initial_selector(cipher: &Aes128, hint_idx: usize) -> u8 {
    let mut blk: Block = Block::default();
    blk[0..8].copy_from_slice(&((hint_idx >> 7) as u64).to_le_bytes());
    cipher.encrypt_block(&mut blk);
    let bit_pos: usize = hint_idx & 127;
    (blk[bit_pos >> 3] >> (bit_pos & 7)) & 1
}

/// Evaluates one `(color, offset)` pair for `FlatAes` in a single AES-128 block encryption.
#[inline]
fn flat_aes_forward(cipher: &Aes128, block_size: usize, hint_id: u32) -> (u8, u32) {
    let mask: u32 = (block_size - 1) as u32;
    if block_size <= 32768 {
        let blk_idx: u64 = (hint_id >> 3) as u64;
        let lane: usize = (hint_id & 7) as usize;
        let mut blk: Block = Block::default();
        blk[0..8].copy_from_slice(&blk_idx.to_le_bytes());
        cipher.encrypt_block(&mut blk);
        let word: u16 = u16::from_le_bytes([blk[lane * 2], blk[lane * 2 + 1]]);
        let color: u8 = (word >> 15) as u8;
        let offset: u32 = (word as u32) & mask;
        (color, offset)
    } else {
        let blk_idx: u64 = (hint_id >> 2) as u64;
        let lane: usize = (hint_id & 3) as usize;
        let mut blk: Block = Block::default();
        blk[0..8].copy_from_slice(&blk_idx.to_le_bytes());
        cipher.encrypt_block(&mut blk);
        let base: usize = lane * 4;
        let word: u32 =
            u32::from_le_bytes([blk[base], blk[base + 1], blk[base + 2], blk[base + 3]]);
        let color: u8 = (word >> 31) as u8;
        let offset: u32 = word & mask;
        (color, offset)
    }
}

/// Scans all `0..total_hints` via batched AES-128-CTR and emits every `(hint_id, color)` whose
/// offset equals `target_offset`.
fn flat_aes_inverse(
    cipher: &Aes128,
    block_size: usize,
    total_hints: usize,
    target_offset: u32,
    out: &mut Vec<(u32, u8)>,
) {
    let mask: u32 = (block_size - 1) as u32;
    let mut blocks: [Block; AES_BATCH_BLOCKS] = [Block::default(); AES_BATCH_BLOCKS];

    if block_size <= 32768 {
        let total_aes_blocks: usize = total_hints.div_ceil(8);
        let target16: u16 = target_offset as u16;
        let mask16: u16 = mask as u16;
        let mut blk_idx: usize = 0;

        while blk_idx < total_aes_blocks {
            let count: usize = AES_BATCH_BLOCKS.min(total_aes_blocks - blk_idx);
            for (i, blk) in blocks.iter_mut().take(count).enumerate() {
                blk.fill(0);
                blk[0..8].copy_from_slice(&((blk_idx + i) as u64).to_le_bytes());
            }
            cipher.encrypt_blocks(&mut blocks[..count]);

            for (i, blk) in blocks.iter().take(count).enumerate() {
                let h_base: usize = (blk_idx + i) * 8;
                let lanes: usize = 8.min(total_hints - h_base);
                for lane in 0..lanes {
                    let word: u16 = u16::from_le_bytes([blk[lane * 2], blk[lane * 2 + 1]]);
                    if (word & mask16) == target16 {
                        out.push(((h_base + lane) as u32, (word >> 15) as u8));
                    }
                }
            }
            blk_idx += count;
        }
    } else {
        let total_aes_blocks: usize = total_hints.div_ceil(4);
        let mut blk_idx: usize = 0;

        while blk_idx < total_aes_blocks {
            let count: usize = AES_BATCH_BLOCKS.min(total_aes_blocks - blk_idx);
            for (i, blk) in blocks.iter_mut().take(count).enumerate() {
                blk.fill(0);
                blk[0..8].copy_from_slice(&((blk_idx + i) as u64).to_le_bytes());
            }
            cipher.encrypt_blocks(&mut blocks[..count]);

            for (i, blk) in blocks.iter().take(count).enumerate() {
                let h_base: usize = (blk_idx + i) * 4;
                let lanes: usize = 4.min(total_hints - h_base);
                for lane in 0..lanes {
                    let base: usize = lane * 4;
                    let word: u32 = u32::from_le_bytes([
                        blk[base],
                        blk[base + 1],
                        blk[base + 2],
                        blk[base + 3],
                    ]);
                    if (word & mask) == target_offset {
                        out.push(((h_base + lane) as u32, (word >> 31) as u8));
                    }
                }
            }
            blk_idx += count;
        }
    }
}

/// Scans all `0..total_hints` via batched AES-128-CTR and invokes `f(hint_id, color, offset)`.
fn flat_aes_scan_all(
    cipher: &Aes128,
    block_size: usize,
    total_hints: usize,
    mut f: impl FnMut(u32, u8, u32),
) {
    let mask: u32 = (block_size - 1) as u32;
    let mut blocks: [Block; AES_BATCH_BLOCKS] = [Block::default(); AES_BATCH_BLOCKS];

    if block_size <= 32768 {
        let total_aes_blocks: usize = total_hints.div_ceil(8);
        let mask16: u16 = mask as u16;
        let mut blk_idx: usize = 0;

        while blk_idx < total_aes_blocks {
            let count: usize = AES_BATCH_BLOCKS.min(total_aes_blocks - blk_idx);
            for (i, blk) in blocks.iter_mut().take(count).enumerate() {
                blk.fill(0);
                blk[0..8].copy_from_slice(&((blk_idx + i) as u64).to_le_bytes());
            }
            cipher.encrypt_blocks(&mut blocks[..count]);

            for (i, blk) in blocks.iter().take(count).enumerate() {
                let h_base: usize = (blk_idx + i) * 8;
                let lanes: usize = 8.min(total_hints - h_base);
                for lane in 0..lanes {
                    let word: u16 = u16::from_le_bytes([blk[lane * 2], blk[lane * 2 + 1]]);
                    f(
                        (h_base + lane) as u32,
                        (word >> 15) as u8,
                        (word & mask16) as u32,
                    );
                }
            }
            blk_idx += count;
        }
    } else {
        let total_aes_blocks: usize = total_hints.div_ceil(4);
        let mut blk_idx: usize = 0;

        while blk_idx < total_aes_blocks {
            let count: usize = AES_BATCH_BLOCKS.min(total_aes_blocks - blk_idx);
            for (i, blk) in blocks.iter_mut().take(count).enumerate() {
                blk.fill(0);
                blk[0..8].copy_from_slice(&((blk_idx + i) as u64).to_le_bytes());
            }
            cipher.encrypt_blocks(&mut blocks[..count]);

            for (i, blk) in blocks.iter().take(count).enumerate() {
                let h_base: usize = (blk_idx + i) * 4;
                let lanes: usize = 4.min(total_hints - h_base);
                for lane in 0..lanes {
                    let base: usize = lane * 4;
                    let word: u32 = u32::from_le_bytes([
                        blk[base],
                        blk[base + 1],
                        blk[base + 2],
                        blk[base + 3],
                    ]);
                    f((h_base + lane) as u32, (word >> 31) as u8, word & mask);
                }
            }
            blk_idx += count;
        }
    }
}

/// Exact `Binomial(n, p)` sampler via geometric inter-arrival gaps (`O(E[K])` steps).
fn sample_binomial_exact<R: Rng + ?Sized>(n: usize, p: f64, rng: &mut R) -> usize {
    if n == 0 || p <= 0.0 {
        return 0;
    }
    if p >= 1.0 {
        return n;
    }
    let ln_q: f64 = (-p).ln_1p();
    let mut pos: usize = 0;
    let mut count: usize = 0;
    loop {
        let u: f64 = rng.gen::<f64>().max(f64::MIN_POSITIVE);
        let gap_f: f64 = (u.ln() / ln_q).floor();
        if !gap_f.is_finite() || gap_f >= (n - pos) as f64 {
            break;
        }
        let gap: usize = gap_f as usize;
        pos += gap + 1;
        if pos > n {
            break;
        }
        count += 1;
    }
    count
}

/// Approximately samples `Bernoulli(exp(-neg_log_p))` with 32-bit tail-chunking so tiny
/// probabilities below `2^-53` (where a bare `f64` draw would floor to `0.0`) remain
/// accurately represented down to machine floating-point precision on the final chunk.
fn sample_bernoulli_neg_exp<R: Rng + ?Sized>(mut neg_log_p: f64, rng: &mut R) -> bool {
    if neg_log_p <= 0.0 {
        return true;
    }
    if !neg_log_p.is_finite() {
        return false;
    }
    let chunk_neg_log: f64 = 32.0 * std::f64::consts::LN_2;
    while neg_log_p > chunk_neg_log {
        if rng.gen::<u32>() != 0 {
            return false;
        }
        neg_log_p -= chunk_neg_log;
    }
    let p_rem: f64 = (-neg_log_p).exp();
    rng.gen::<f64>() < p_rem
}

/// Samples `k` distinct uniform elements from `pool` without replacement.
fn sample_distinct_from<R: Rng + ?Sized>(pool: &[u32], k: usize, rng: &mut R) -> Vec<u32> {
    let k: usize = k.min(pool.len());
    let mut seen: HashSet<usize> = HashSet::with_capacity(k);
    let mut out: Vec<u32> = Vec::with_capacity(k);
    while out.len() < k {
        let pos: usize = rng.gen_range(0..pool.len());
        if seen.insert(pos) {
            out.push(pool[pos]);
        }
    }
    out
}

#[inline]
fn xor_bytes(dst: &mut [u8], src: &[u8]) {
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        *d ^= s;
    }
}

#[inline]
fn xor_words(dst: &mut [u64], src: &[u64]) {
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        *d ^= s;
    }
}

#[inline]
fn xor_atomic(dst: &[AtomicU64], src: &[u64]) {
    for (d, &s) in dst.iter().zip(src.iter()) {
        d.fetch_xor(s, Ordering::Relaxed);
    }
}

#[inline]
fn load_record_words(rec: &[u8], out: &mut [u64]) {
    let full: usize = rec.len() / 8;
    for (i, w) in out.iter_mut().take(full).enumerate() {
        let base: usize = i * 8;
        *w = u64::from_le_bytes(rec[base..base + 8].try_into().unwrap());
    }
    let rem: usize = rec.len() % 8;
    if rem > 0 {
        let mut tail: [u8; 8] = [0u8; 8];
        tail[..rem].copy_from_slice(&rec[full * 8..]);
        out[full] = u64::from_le_bytes(tail);
    }
}

fn store_words_into_bytes(src: &[u64], words_per_rec: usize, record_size: usize, dst: &mut [u8]) {
    let count: usize = dst.len() / record_size;
    for i in 0..count {
        let w_slice: &[u64] = &src[i * words_per_rec..(i + 1) * words_per_rec];
        let out: &mut [u8] = &mut dst[i * record_size..(i + 1) * record_size];
        let full: usize = record_size / 8;
        for (k, &w) in w_slice.iter().take(full).enumerate() {
            out[k * 8..(k + 1) * 8].copy_from_slice(&w.to_le_bytes());
        }
        let rem: usize = record_size % 8;
        if rem > 0 {
            out[full * 8..].copy_from_slice(&w_slice[full].to_le_bytes()[..rem]);
        }
    }
}

fn store_atomic_into_bytes(
    src: &[AtomicU64],
    words_per_rec: usize,
    record_size: usize,
    dst: &mut [u8],
) {
    let count: usize = dst.len() / record_size;
    for i in 0..count {
        let w_slice: &[AtomicU64] = &src[i * words_per_rec..(i + 1) * words_per_rec];
        let out: &mut [u8] = &mut dst[i * record_size..(i + 1) * record_size];
        let full: usize = record_size / 8;
        for (k, w) in w_slice.iter().take(full).enumerate() {
            out[k * 8..(k + 1) * 8].copy_from_slice(&w.load(Ordering::Relaxed).to_le_bytes());
        }
        let rem: usize = record_size % 8;
        if rem > 0 {
            let tail: u64 = w_slice[full].load(Ordering::Relaxed);
            out[full * 8..].copy_from_slice(&tail.to_le_bytes()[..rem]);
        }
    }
}

fn effective_threads(threads: usize, work_items: usize) -> usize {
    #[cfg(target_arch = "wasm32")]
    {
        let _ = (threads, work_items);
        1
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        threads.max(1).min(work_items.max(1))
    }
}

fn try_alloc_zeroed(len: usize) -> Result<Vec<u8>, MultiLevelError> {
    let mut v: Vec<u8> = Vec::new();
    v.try_reserve_exact(len)
        .map_err(|_| MultiLevelError::InvalidFormat("allocation failed"))?;
    v.resize(len, 0);
    Ok(v)
}

fn try_alloc_filled(len: usize, val: u32) -> Result<Vec<u32>, MultiLevelError> {
    let mut v: Vec<u32> = Vec::new();
    v.try_reserve_exact(len)
        .map_err(|_| MultiLevelError::InvalidFormat("allocation failed"))?;
    v.resize(len, val);
    Ok(v)
}

fn read_fill<R: Read>(reader: &mut R, mut buf: &mut [u8]) -> io::Result<usize> {
    let mut total: usize = 0;
    while !buf.is_empty() {
        match reader.read(buf) {
            Ok(0) => break,
            Ok(n) => {
                total += n;
                buf = &mut buf[n..];
            }
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
    Ok(total)
}

fn run_scoped_workers<F: Fn() + Sync>(workers: usize, f: F) {
    #[cfg(target_arch = "wasm32")]
    {
        let _ = workers;
        f();
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        if workers <= 1 {
            f();
            return;
        }
        std::thread::scope(|s| {
            let f_ref: &F = &f;
            let mut handles = Vec::with_capacity(workers - 1);
            for _ in 1..workers {
                if let Ok(handle) = std::thread::Builder::new().spawn_scoped(s, f_ref) {
                    handles.push(handle);
                }
            }
            f_ref();
            for h in handles {
                h.join().unwrap();
            }
        });
    }
}

fn run_scoped_workers_mut<S: Send, F: Fn(&mut S) + Sync>(states: &mut [S], f: F) {
    let Some((first, rest)) = states.split_first_mut() else {
        return;
    };
    #[cfg(target_arch = "wasm32")]
    {
        let _ = rest;
        f(first);
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        if rest.is_empty() {
            f(first);
            return;
        }
        std::thread::scope(|s| {
            let f_ref: &F = &f;
            let mut handles = Vec::with_capacity(rest.len());
            for st in rest.iter_mut() {
                if let Ok(handle) = std::thread::Builder::new().spawn_scoped(s, move || f_ref(st)) {
                    handles.push(handle);
                }
            }
            f_ref(first);
            for h in handles {
                h.join().unwrap();
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    fn make_db(num_entries: usize, record_size: usize, seed: u64) -> Vec<u8> {
        let mut rng: ChaCha20Rng = ChaCha20Rng::seed_from_u64(seed);
        let mut db: Vec<u8> = vec![0u8; num_entries * record_size];
        rng.fill(&mut db[..]);
        db
    }

    fn verify_all_parities_exact(client: &MultiLevelClient, db: &[u8]) {
        let r: usize = client.params.record_size;
        let n: usize = client.params.num_entries;
        for (lvl_idx, level) in client.levels.iter().enumerate() {
            let mut exp_primary: Vec<Vec<u8>> = vec![vec![0u8; r]; level.spec.num_primary];
            let mut exp_bak0: Vec<Vec<u8>> = vec![vec![0u8; r]; level.spec.num_backup];
            let mut exp_bak1: Vec<Vec<u8>> = vec![vec![0u8; r]; level.spec.num_backup];

            for blk in 0..level.spec.num_blocks {
                let key: PrfKey128 = level.derive_block_key(blk);
                let iprf: Option<Iprf> = match level.spec.backend {
                    LevelBackend::Iprf => Some(Iprf::new(
                        key,
                        level.spec.total_hints() as u64,
                        level.spec.block_size as u64,
                    )),
                    LevelBackend::FlatAes => None,
                };
                let flat_cipher: Option<Aes128> = match level.spec.backend {
                    LevelBackend::FlatAes => Some(Aes128::new(&GenericArray::from(key))),
                    LevelBackend::Iprf => None,
                };

                let eval = |hint_id: u32| -> (u8, u32) {
                    if let Some(ref f) = iprf {
                        (
                            level.iprf_color_bit(blk, hint_id),
                            f.forward(hint_id as u64) as u32,
                        )
                    } else {
                        flat_aes_forward(
                            flat_cipher.as_ref().unwrap(),
                            level.spec.block_size,
                            hint_id,
                        )
                    }
                };

                for (s, slot) in level.slots.iter().enumerate() {
                    let (color, off) = match slot.override_entry {
                        Some(ov) if ov.block as usize == blk => (ov.color, ov.offset),
                        _ => eval(slot.hint_id),
                    };
                    if color == slot.selector {
                        let idx: usize = blk * level.spec.block_size + off as usize;
                        if idx < n {
                            xor_bytes(&mut exp_primary[s], &db[idx * r..(idx + 1) * r]);
                        }
                    }
                }
                for b in level.next_backup..level.spec.num_backup {
                    let hint_id: u32 = (level.spec.num_primary + b) as u32;
                    let (color, off) = eval(hint_id);
                    let idx: usize = blk * level.spec.block_size + off as usize;
                    if idx < n {
                        if color == 0 {
                            xor_bytes(&mut exp_bak0[b], &db[idx * r..(idx + 1) * r]);
                        } else {
                            xor_bytes(&mut exp_bak1[b], &db[idx * r..(idx + 1) * r]);
                        }
                    }
                }
            }

            for (s, expected) in exp_primary.iter().enumerate() {
                assert_eq!(
                    &level.primary_parities[s * r..(s + 1) * r],
                    &expected[..],
                    "primary parity mismatch at level {lvl_idx} slot {s} (hint_id={}, ov={:?})",
                    level.slots[s].hint_id,
                    level.slots[s].override_entry
                );
            }
            for b in level.next_backup..level.spec.num_backup {
                let local_b: usize = b - level.backup_offset;
                assert_eq!(
                    &level.backup_parities_0[local_b * r..(local_b + 1) * r],
                    &exp_bak0[b][..],
                    "backup0 parity mismatch at level {lvl_idx} backup {b}"
                );
                assert_eq!(
                    &level.backup_parities_1[local_b * r..(local_b + 1) * r],
                    &exp_bak1[b][..],
                    "backup1 parity mismatch at level {lvl_idx} backup {b}"
                );
            }
        }
    }

    fn small_three_level_params(
        num_entries: usize,
        record_size: usize,
        variant: BackendVariant,
    ) -> MultiLevelParams {
        let specs: [(usize, usize, usize); 3] = [(16, 32, 96), (4, 16, 48), (2, 16, 32)];
        let levels: Vec<LevelSpec> = specs
            .iter()
            .enumerate()
            .map(|(i, &(w, m, q))| LevelSpec {
                block_size: w,
                num_blocks: num_entries.div_ceil(w),
                num_primary: m,
                num_backup: q,
                backend: match variant {
                    BackendVariant::HybridA { iprf_levels } if i < iprf_levels => {
                        LevelBackend::Iprf
                    }
                    _ => LevelBackend::FlatAes,
                },
            })
            .collect();
        MultiLevelParams::custom(num_entries, record_size, levels).unwrap()
    }

    #[test]
    fn hint_init_matches_bruteforce_and_is_thread_deterministic() {
        let num_entries: usize = 256;
        let record_size: usize = 40;
        let db: Vec<u8> = make_db(num_entries, record_size, 11);
        let seed: [u8; 32] = [0x42; 32];

        for variant in [
            BackendVariant::HybridA { iprf_levels: 1 },
            BackendVariant::FlatB,
        ] {
            let params: MultiLevelParams =
                small_three_level_params(num_entries, record_size, variant);
            let c1: MultiLevelClient =
                MultiLevelClient::hint_init(params.clone(), &seed, &db, 1).unwrap();
            let c4: MultiLevelClient = MultiLevelClient::hint_init(params, &seed, &db, 4).unwrap();

            verify_all_parities_exact(&c1, &db);
            for (l1, l4) in c1.levels.iter().zip(c4.levels.iter()) {
                assert_eq!(l1.primary_parities, l4.primary_parities);
                assert_eq!(l1.backup_parities_0, l4.backup_parities_0);
                assert_eq!(l1.backup_parities_1, l4.backup_parities_1);
            }
        }
    }

    #[test]
    fn e2e_queries_refresh_and_interleaved_deltas_hybrid_and_flat() {
        let num_entries: usize = 256;
        let record_size: usize = 40;
        let seed: [u8; 32] = [0x99; 32];

        // Small custom 3-level waterfall where level 0 fails ~37% of the time so both
        // failure-path PromoteIn and success-path PromoteOut are heavily exercised.
        for variant in [
            BackendVariant::HybridA { iprf_levels: 1 },
            BackendVariant::FlatB,
        ] {
            let mut db: Vec<u8> = make_db(num_entries, record_size, 77);
            let params: MultiLevelParams =
                small_three_level_params(num_entries, record_size, variant);
            let mut client: MultiLevelClient =
                MultiLevelClient::hint_init(params.clone(), &seed, &db, 2).unwrap();
            let mut rng: ChaCha20Rng = ChaCha20Rng::seed_from_u64(12345);

            let mut level_hits: [usize; 3] = [0; 3];
            let mut queried: Vec<usize> = Vec::new();

            for step in 0..40 {
                // Mix fresh indices and repeat previously queried indices (hits promoted overrides).
                let idx: usize = if step % 3 == 2 && !queried.is_empty() {
                    queried[rng.gen_range(0..queried.len())]
                } else {
                    rng.gen_range(0..num_entries)
                };

                let pending: PendingMultiLevelQuery = client.prepare_query(idx, &mut rng).unwrap();
                level_hits[pending.level] += 1;
                let answer: MultiLevelAnswer =
                    MultiLevelServer::answer(&db, &params, &pending.query, client.epoch()).unwrap();
                let recovered: Vec<u8> = client.finish_query(&pending, &answer, &mut rng).unwrap();
                assert_eq!(
                    &recovered[..],
                    &db[idx * record_size..(idx + 1) * record_size],
                    "query mismatch at step {step}, idx {idx}, variant {variant:?}"
                );
                queried.push(idx);

                // Apply 3 deltas after each query, including one on the just-queried index
                // (stresses PromoteIn / PromoteOut override tracking) and 2 random indices.
                let mut batch: Vec<(usize, Vec<u8>)> = Vec::new();
                for upd_i in 0..3 {
                    let target: usize = if upd_i == 0 {
                        idx
                    } else {
                        rng.gen_range(0..num_entries)
                    };
                    let mut delta: Vec<u8> = vec![0u8; record_size];
                    rng.fill(&mut delta[..]);
                    xor_bytes(
                        &mut db[target * record_size..(target + 1) * record_size],
                        &delta,
                    );
                    if step % 2 == 0 {
                        client.apply_delta(target, &delta).unwrap();
                    } else {
                        batch.push((target, delta));
                    }
                }
                if !batch.is_empty() {
                    client.apply_deltas_batch(&batch, 2).unwrap();
                }

                // Every 10 steps verify every single primary and backup parity against brute force.
                if step % 10 == 9 {
                    verify_all_parities_exact(&client, &db);
                }
            }

            // Confirm level 0 both succeeded and failed (so waterfall repair actually ran).
            assert!(
                level_hits[0] > 0 && (level_hits[1] + level_hits[2]) > 0,
                "expected both level-0 hits and fallbacks, got {level_hits:?}"
            );
            verify_all_parities_exact(&client, &db);
        }
    }

    #[test]
    fn two_level_builder_and_e2e() {
        let num_entries: usize = 65_536;
        let record_size: usize = 40;
        let seed: [u8; 32] = [0x17; 32];
        let mut db: Vec<u8> = make_db(num_entries, record_size, 5);

        let params: MultiLevelParams =
            MultiLevelParams::two_level(num_entries, record_size, 32, 128, BackendVariant::FlatB)
                .unwrap();
        assert_eq!(params.levels.len(), 2);
        assert!(params.total_failure_probability() < 1e-12);

        let mut client: MultiLevelClient =
            MultiLevelClient::hint_init(params.clone(), &seed, &db, 4).unwrap();
        let mut rng: ChaCha20Rng = ChaCha20Rng::seed_from_u64(99);

        for _ in 0..25 {
            let idx: usize = rng.gen_range(0..num_entries);
            let pending: PendingMultiLevelQuery = client.prepare_query(idx, &mut rng).unwrap();
            let answer: MultiLevelAnswer =
                MultiLevelServer::answer(&db, &params, &pending.query, client.epoch()).unwrap();
            let got: Vec<u8> = client.finish_query(&pending, &answer, &mut rng).unwrap();
            assert_eq!(&got[..], &db[idx * record_size..(idx + 1) * record_size]);

            let mut delta: Vec<u8> = vec![0u8; record_size];
            rng.fill(&mut delta[..]);
            xor_bytes(&mut db[idx * record_size..(idx + 1) * record_size], &delta);
            client.apply_delta(idx, &delta).unwrap();
        }
    }

    #[test]
    fn binomial_sampler_mean_matches_expectation() {
        let mut rng: ChaCha20Rng = ChaCha20Rng::seed_from_u64(42);
        let trials: usize = 10_000;
        let n: usize = 4096;
        let p: f64 = 1.0 / 1024.0; // expected mean = 4.0
        let sum: usize = (0..trials)
            .map(|_| sample_binomial_exact(n, p, &mut rng))
            .sum();
        let mean: f64 = sum as f64 / trials as f64;
        assert!(
            (mean - 4.0).abs() < 0.15,
            "unexpected binomial mean {mean} (expected ~4.0)"
        );
    }

    #[test]
    fn serialization_roundtrip_fresh_and_mid_session() {
        let num_entries: usize = 256;
        let record_size: usize = 40;
        let seed: [u8; 32] = [0xAB; 32];

        for variant in [
            BackendVariant::HybridA { iprf_levels: 1 },
            BackendVariant::FlatB,
        ] {
            let mut db: Vec<u8> = make_db(num_entries, record_size, 555);
            let params: MultiLevelParams =
                small_three_level_params(num_entries, record_size, variant);
            let client: MultiLevelClient =
                MultiLevelClient::hint_init(params.clone(), &seed, &db, 2).unwrap();

            // 1. Fresh round-trip right after HintInit.
            let fresh_bytes: Vec<u8> = client.to_bytes().unwrap();
            let mut restored: MultiLevelClient =
                MultiLevelClient::from_bytes(&fresh_bytes, &seed).unwrap();
            assert_eq!(restored.to_bytes().unwrap(), fresh_bytes);
            verify_all_parities_exact(&restored, &db);

            // 2. Run queries (inducing PromoteIn & PromoteOut overrides) + deltas,
            //    then serialize mid-session, reload, and keep querying + updating.
            let mut rng: ChaCha20Rng = ChaCha20Rng::seed_from_u64(777);
            let mut queried: Vec<usize> = Vec::new();
            for _ in 0..20 {
                let idx: usize = rng.gen_range(0..num_entries);
                let pending: PendingMultiLevelQuery =
                    restored.prepare_query(idx, &mut rng).unwrap();
                let ans: MultiLevelAnswer =
                    MultiLevelServer::answer(&db, &params, &pending.query, restored.epoch())
                        .unwrap();
                let got: Vec<u8> = restored.finish_query(&pending, &ans, &mut rng).unwrap();
                assert_eq!(&got[..], &db[idx * record_size..(idx + 1) * record_size]);
                queried.push(idx);

                let mut delta: Vec<u8> = vec![0u8; record_size];
                rng.fill(&mut delta[..]);
                xor_bytes(&mut db[idx * record_size..(idx + 1) * record_size], &delta);
                restored.apply_delta(idx, &delta).unwrap();
            }

            // Active backups were consumed, so mid-session serialization must be strictly smaller.
            let mid_bytes: Vec<u8> = restored.to_bytes().unwrap();
            assert!(mid_bytes.len() < fresh_bytes.len());

            let mut reloaded: MultiLevelClient =
                MultiLevelClient::from_bytes(&mid_bytes, &seed).unwrap();
            assert_eq!(reloaded.epoch(), restored.epoch());
            assert_eq!(reloaded.to_bytes().unwrap(), mid_bytes);
            verify_all_parities_exact(&reloaded, &db);

            // Continue querying (including previously queried overridden indices) and applying deltas.
            for step in 0..15 {
                let idx: usize = if step % 2 == 0 {
                    queried[rng.gen_range(0..queried.len())]
                } else {
                    rng.gen_range(0..num_entries)
                };
                let pending: PendingMultiLevelQuery =
                    reloaded.prepare_query(idx, &mut rng).unwrap();
                let ans: MultiLevelAnswer =
                    MultiLevelServer::answer(&db, &params, &pending.query, reloaded.epoch())
                        .unwrap();
                let got: Vec<u8> = reloaded.finish_query(&pending, &ans, &mut rng).unwrap();
                assert_eq!(&got[..], &db[idx * record_size..(idx + 1) * record_size]);

                let mut delta: Vec<u8> = vec![0u8; record_size];
                rng.fill(&mut delta[..]);
                xor_bytes(&mut db[idx * record_size..(idx + 1) * record_size], &delta);
                reloaded.apply_delta(idx, &delta).unwrap();
            }
            verify_all_parities_exact(&reloaded, &db);
        }
    }

    #[test]
    fn serialization_rejects_corruption_and_invalid_structures() {
        let num_entries: usize = 512;
        let record_size: usize = 40;
        let seed: [u8; 32] = [0xCD; 32];
        let db: Vec<u8> = make_db(num_entries, record_size, 888);
        let params: MultiLevelParams =
            small_three_level_params(num_entries, record_size, BackendVariant::FlatB);
        let client: MultiLevelClient = MultiLevelClient::hint_init(params, &seed, &db, 1).unwrap();
        let valid: Vec<u8> = client.to_bytes().unwrap();

        // Bad magic.
        let mut bad_magic: Vec<u8> = valid.clone();
        bad_magic[0] ^= 0xFF;
        assert_eq!(
            MultiLevelClient::from_bytes(&bad_magic, &seed).err(),
            Some(MultiLevelError::InvalidFormat("bad magic bytes"))
        );

        // Bad version.
        let mut bad_ver: Vec<u8> = valid.clone();
        bad_ver[8] = 99;
        assert_eq!(
            MultiLevelClient::from_bytes(&bad_ver, &seed).err(),
            Some(MultiLevelError::InvalidFormat("unsupported format version"))
        );

        // Flipped parity byte -> checksum mismatch.
        let mut bad_parity: Vec<u8> = valid.clone();
        let mid: usize = bad_parity.len() / 2;
        bad_parity[mid] ^= 0x01;
        assert_eq!(
            MultiLevelClient::from_bytes(&bad_parity, &seed).err(),
            Some(MultiLevelError::ChecksumMismatch)
        );

        // Trailing garbage byte.
        let mut trailing: Vec<u8> = valid.clone();
        trailing.push(0x00);
        assert_eq!(
            MultiLevelClient::from_bytes(&trailing, &seed).err(),
            Some(MultiLevelError::TrailingData)
        );

        // Truncated buffer.
        let truncated: &[u8] = &valid[..valid.len() - 10];
        assert_eq!(
            MultiLevelClient::from_bytes(truncated, &seed).err(),
            Some(MultiLevelError::Io(io::ErrorKind::UnexpectedEof))
        );
    }

    #[test]
    fn safety_invariants_reservation_snapshot_answer_bounds_and_exhaustion() {
        let num_entries: usize = 256;
        let record_size: usize = 40;
        let seed: [u8; 32] = [0xEF; 32];
        let mut db: Vec<u8> = make_db(num_entries, record_size, 999);
        let params: MultiLevelParams =
            small_three_level_params(num_entries, record_size, BackendVariant::FlatB);
        let mut client: MultiLevelClient =
            MultiLevelClient::hint_init(params.clone(), &seed, &db, 2).unwrap();
        let mut rng: ChaCha20Rng = ChaCha20Rng::seed_from_u64(404);

        // 1. Slot reservation prevents reusing the same slot across concurrent prepare_query calls,
        //    and write_hints refuses serialization while a query is in flight.
        let p1: PendingMultiLevelQuery = client.prepare_query(7, &mut rng).unwrap();
        assert_eq!(
            client.to_bytes().err(),
            Some(MultiLevelError::QueryInFlight)
        );
        let p2: PendingMultiLevelQuery = client.prepare_query(7, &mut rng).unwrap();
        assert_ne!((p1.level, p1.slot_idx), (p2.level, p2.slot_idx));

        // 2. Truncated answer is rejected without consuming the handle or reservation,
        //    so the caller can retry or abort without cloning `p1`.
        let bad_ans: MultiLevelAnswer = MultiLevelAnswer {
            snapshot_epoch: client.epoch(),
            parity_0: vec![0u8; record_size - 1],
            parity_1: vec![0u8; record_size],
        };
        assert_eq!(
            client.finish_query(&p1, &bad_ans, &mut rng),
            Err(MultiLevelError::AnswerLengthMismatch {
                expected: record_size,
                actual: (record_size - 1, record_size),
            })
        );

        // Server answer from a mismatched snapshot epoch is also rejected without consuming p1.
        let wrong_epoch_ans: MultiLevelAnswer =
            MultiLevelServer::answer(&db, &params, &p1.query, client.epoch() + 1).unwrap();
        assert_eq!(
            client.finish_query(&p1, &wrong_epoch_ans, &mut rng),
            Err(MultiLevelError::SnapshotMismatch {
                query_epoch: 1,
                current_epoch: 0,
            })
        );

        // Completing p1 with the valid matching-epoch answer still succeeds, and a second
        // completion with the same handle is rejected.
        let good_ans1: MultiLevelAnswer =
            MultiLevelServer::answer(&db, &params, &p1.query, client.epoch()).unwrap();
        let got1: Vec<u8> = client.finish_query(&p1, &good_ans1, &mut rng).unwrap();
        assert_eq!(&got1[..], &db[7 * record_size..8 * record_size]);
        assert_eq!(
            client.finish_query(&p1, &good_ans1, &mut rng),
            Err(MultiLevelError::UnknownPendingQuery {
                query_id: p1.query_id,
            })
        );

        // 3. Interleaved delta advances snapshot epoch, so stale p2 is rejected without corrupting state.
        let ans2_stale: MultiLevelAnswer =
            MultiLevelServer::answer(&db, &params, &p2.query, client.epoch()).unwrap();
        let delta: Vec<u8> = vec![0x55; record_size];
        xor_bytes(&mut db[0..record_size], &delta);
        client.apply_delta(0, &delta).unwrap();
        assert_eq!(
            client.finish_query(&p2, &ans2_stale, &mut rng),
            Err(MultiLevelError::SnapshotMismatch {
                query_epoch: 0,
                current_epoch: 1,
            })
        );

        // 4. MultiLevelServer::answer rejects malformed queries cleanly without panicking.
        let mut bad_q: MultiLevelQuery = MultiLevelQuery {
            level: params.levels.len(),
            colors: vec![],
            offsets: vec![],
        };
        assert!(matches!(
            MultiLevelServer::answer(&db, &params, &bad_q, client.epoch()),
            Err(MultiLevelError::InvalidQuery(_))
        ));
        bad_q.level = 0;
        bad_q.colors = vec![0; params.levels[0].num_blocks];
        bad_q.offsets = vec![params.levels[0].block_size as u32; params.levels[0].num_blocks];
        assert!(matches!(
            MultiLevelServer::answer(&db, &params, &bad_q, client.epoch()),
            Err(MultiLevelError::InvalidQuery(_))
        ));

        // 5. Exhaustion returns the recovered record without partially refreshing other slots.
        let tiny_params: MultiLevelParams = MultiLevelParams::custom(
            num_entries,
            record_size,
            vec![LevelSpec {
                block_size: 2,
                num_blocks: num_entries / 2,
                num_primary: 16,
                num_backup: 1,
                backend: LevelBackend::FlatAes,
            }],
        )
        .unwrap();
        let mut tiny_client: MultiLevelClient =
            MultiLevelClient::hint_init(tiny_params.clone(), &seed, &db, 1).unwrap();
        // Consume the single backup on query 1, then trigger exhaustion on query 2.
        let q1: PendingMultiLevelQuery = tiny_client.prepare_query(3, &mut rng).unwrap();
        let a1: MultiLevelAnswer =
            MultiLevelServer::answer(&db, &tiny_params, &q1.query, tiny_client.epoch()).unwrap();
        let _ = tiny_client.finish_query(&q1, &a1, &mut rng);

        // Exhaust any remaining backups so next query is guaranteed to hit BackupExhausted.
        tiny_client.levels[0].next_backup = tiny_client.levels[0].spec.num_backup;
        let q2: PendingMultiLevelQuery = tiny_client.prepare_query(4, &mut rng).unwrap();
        let a2: MultiLevelAnswer =
            MultiLevelServer::answer(&db, &tiny_params, &q2.query, tiny_client.epoch()).unwrap();
        let expected_rec: Vec<u8> = db[4 * record_size..5 * record_size].to_vec();
        assert_eq!(
            tiny_client.finish_query(&q2, &a2, &mut rng),
            Err(MultiLevelError::BackupExhausted {
                level: 0,
                recovered: Some(expected_rec),
            })
        );
    }

    #[test]
    fn abort_query_retires_slot_survives_reload_and_rejects_cross_session_pending() {
        let num_entries: usize = 256;
        let record_size: usize = 40;
        let seed: [u8; 32] = [0xAB; 32];
        let db: Vec<u8> = make_db(num_entries, record_size, 1234);
        let params: MultiLevelParams =
            small_three_level_params(num_entries, record_size, BackendVariant::FlatB);
        let mut client: MultiLevelClient =
            MultiLevelClient::hint_init(params.clone(), &seed, &db, 1).unwrap();
        let mut rng: ChaCha20Rng = ChaCha20Rng::seed_from_u64(5678);

        // 1. Reserve a slot, abort it, verify checkpointing works again and the retired slot
        //    is never selected again (even after serialization/reload).
        let abandoned: PendingMultiLevelQuery = client.prepare_query(11, &mut rng).unwrap();
        assert_eq!(
            client.to_bytes().err(),
            Some(MultiLevelError::QueryInFlight)
        );
        let (ab_lvl, ab_slot) = (abandoned.level, abandoned.slot_idx);
        client.abort_query(&abandoned).unwrap();
        assert!(client.levels[ab_lvl].slots[ab_slot as usize].retired);
        assert!(!client.levels[ab_lvl].slots[ab_slot as usize].reserved);

        // Double abort or finishing an aborted query must fail cleanly.
        assert_eq!(
            client.abort_query(&abandoned),
            Err(MultiLevelError::UnknownPendingQuery {
                query_id: abandoned.query_id,
            })
        );
        let ans_ab: MultiLevelAnswer =
            MultiLevelServer::answer(&db, &params, &abandoned.query, client.epoch()).unwrap();
        assert_eq!(
            client.finish_query(&abandoned, &ans_ab, &mut rng),
            Err(MultiLevelError::UnknownPendingQuery {
                query_id: abandoned.query_id,
            })
        );

        // Serialization now succeeds and preserves the retired bit across reload.
        let bytes: Vec<u8> = client.to_bytes().unwrap();
        let mut reloaded: MultiLevelClient = MultiLevelClient::from_bytes(&bytes, &seed).unwrap();
        assert!(reloaded.levels[ab_lvl].slots[ab_slot as usize].retired);

        // Even when two sessions mint identical query_id/slot_idx for the same index,
        // `reloaded` rejects a pending query from another session (different session_id).
        let mut rng_a: ChaCha20Rng = ChaCha20Rng::seed_from_u64(9001);
        let mut rng_b: ChaCha20Rng = ChaCha20Rng::seed_from_u64(9001);
        let mut fresh_copy: MultiLevelClient = MultiLevelClient::from_bytes(&bytes, &seed).unwrap();
        let p_other: PendingMultiLevelQuery = fresh_copy.prepare_query(12, &mut rng_a).unwrap();
        let p_live: PendingMultiLevelQuery = reloaded.prepare_query(12, &mut rng_b).unwrap();
        assert_eq!(p_other.query_id, p_live.query_id);
        assert_eq!(p_other.slot_idx, p_live.slot_idx);
        let ans_other: MultiLevelAnswer =
            MultiLevelServer::answer(&db, &params, &p_other.query, fresh_copy.epoch()).unwrap();
        assert_eq!(
            reloaded.finish_query(&p_other, &ans_other, &mut rng),
            Err(MultiLevelError::UnknownPendingQuery {
                query_id: p_other.query_id,
            })
        );
        assert_eq!(
            reloaded.abort_query(&p_other),
            Err(MultiLevelError::UnknownPendingQuery {
                query_id: p_other.query_id,
            })
        );
        // `reloaded`'s own reservation `p_live` is still intact and finishes cleanly.
        let ans_live: MultiLevelAnswer =
            MultiLevelServer::answer(&db, &params, &p_live.query, reloaded.epoch()).unwrap();
        let got_live: Vec<u8> = reloaded.finish_query(&p_live, &ans_live, &mut rng).unwrap();
        assert_eq!(&got_live[..], &db[12 * record_size..13 * record_size]);

        // Reloaded client can still serve queries and never picks the retired slot.
        for _ in 0..10 {
            let p: PendingMultiLevelQuery = reloaded.prepare_query(11, &mut rng).unwrap();
            assert_ne!((p.level, p.slot_idx), (ab_lvl, ab_slot));
            let a: MultiLevelAnswer =
                MultiLevelServer::answer(&db, &params, &p.query, reloaded.epoch()).unwrap();
            let got: Vec<u8> = reloaded.finish_query(&p, &a, &mut rng).unwrap();
            assert_eq!(&got[..], &db[11 * record_size..12 * record_size]);
        }
    }

    #[test]
    fn read_hints_with_limit_rejects_oversized_checkpoint_and_param_overflows() {
        let num_entries: usize = 256;
        let record_size: usize = 40;
        let seed: [u8; 32] = [0x99; 32];
        let db: Vec<u8> = make_db(num_entries, record_size, 321);
        let params: MultiLevelParams =
            small_three_level_params(num_entries, record_size, BackendVariant::FlatB);
        let client: MultiLevelClient =
            MultiLevelClient::hint_init(params.clone(), &seed, &db, 1).unwrap();
        let valid: Vec<u8> = client.to_bytes().unwrap();
        assert_eq!(client.serialized_byte_len(), Some(valid.len()));

        assert_eq!(
            MultiLevelClient::from_bytes_with_limit(&valid, &seed, 1024).err(),
            Some(MultiLevelError::InvalidFormat(
                "checkpoint size limit exceeded"
            ))
        );

        // Unrepresentable parity/accumulator sizes in `custom` must return InvalidParams before allocation.
        assert!(matches!(
            MultiLevelParams::custom(
                2,
                u32::MAX as usize,
                vec![LevelSpec {
                    block_size: 2,
                    num_blocks: 1,
                    num_primary: (isize::MAX as usize) / (u32::MAX as usize) + 1,
                    num_backup: 1,
                    backend: LevelBackend::FlatAes,
                }],
            ),
            Err(MultiLevelError::InvalidParams(_))
        ));

        // Overflowing kappa or query_budget in constructors must return the specific InvalidParams error, not panic.
        assert_eq!(
            MultiLevelParams::two_level(256, 40, usize::MAX, 16, BackendVariant::FlatB).err(),
            Some(MultiLevelError::InvalidParams(
                "kappa power-of-two overflow"
            ))
        );
        assert_eq!(
            MultiLevelParams::two_level(65_536, 40, 32, usize::MAX, BackendVariant::FlatB).err(),
            Some(MultiLevelError::InvalidParams(
                "level-0 backup budget overflow"
            ))
        );
        assert_eq!(
            MultiLevelParams::waterfall(256, 40, 2, usize::MAX, BackendVariant::FlatB).err(),
            Some(MultiLevelError::InvalidParams(
                "level-0 backup budget overflow"
            ))
        );
        // Overflowing num_entries * record_size in hint_init must return InvalidParams, not panic.
        let huge_params: MultiLevelParams = MultiLevelParams {
            num_entries: usize::MAX,
            record_size: 40,
            levels: params.levels.clone(),
        };
        assert_eq!(
            MultiLevelClient::hint_init(huge_params, &seed, &[], 1).err(),
            Some(MultiLevelError::InvalidParams("database size overflow"))
        );
    }

    #[test]
    fn refresh_preserves_unbiased_probabilities_with_unavailable_slots() {
        let record_size: usize = 8;
        let seed: [u8; 32] = [0x42; 32];
        let mut rng: ChaCha20Rng = ChaCha20Rng::seed_from_u64(0x5EED_BEEF);

        // Part 1: Failed-level PromoteIn with M_0 = 2, w_0 = 2 (p = 0.25) and 1 retired slot (A_0 = 1).
        // Unbiased per-slot replacement probability is 0.25 (~500 / 2000), whereas sampling over
        // M_0 = 2 and clamping to A_0 = 1 would give 1 - (3/4)^2 = 7/16 = 0.4375 (~875 / 2000).
        {
            let num_entries: usize = 4;
            let db: Vec<u8> = make_db(num_entries, record_size, 11);
            let params: MultiLevelParams = MultiLevelParams::custom(
                num_entries,
                record_size,
                vec![
                    LevelSpec {
                        block_size: 2,
                        num_blocks: 2,
                        num_primary: 2,
                        num_backup: 2_500,
                        backend: LevelBackend::FlatAes,
                    },
                    LevelSpec {
                        block_size: 2,
                        num_blocks: 2,
                        num_primary: 48,
                        num_backup: 6_000,
                        backend: LevelBackend::FlatAes,
                    },
                ],
            )
            .unwrap();
            let mut client: MultiLevelClient =
                MultiLevelClient::hint_init(params.clone(), &seed, &db, 1).unwrap();
            client.levels[0].slots[0].retired = true;

            let mut scratch: Vec<(u32, u8)> = Vec::new();
            let trials: usize = 2_000;
            for _ in 0..trials {
                let miss_idx: usize = (0..num_entries)
                    .find(|&idx| {
                        client.levels[0]
                            .find_matching_slots(idx, &mut scratch)
                            .is_empty()
                    })
                    .expect("a single slot with w=2 cannot cover all entries");
                let pending: PendingMultiLevelQuery =
                    client.prepare_query(miss_idx, &mut rng).unwrap();
                assert_eq!(pending.level, 1);
                let answer: MultiLevelAnswer =
                    MultiLevelServer::answer(&db, &params, &pending.query, client.epoch()).unwrap();
                let value: Vec<u8> = client.finish_query(&pending, &answer, &mut rng).unwrap();
                assert_eq!(
                    value.as_slice(),
                    &db[miss_idx * record_size..(miss_idx + 1) * record_size]
                );
            }
            let l0_replacements: usize = client.levels[0].next_backup;
            assert!(
                (420..=580).contains(&l0_replacements),
                "expected ~500 unbiased failed-level replacements out of {trials}, got {l0_replacements}"
            );

            // Boundary: A_0 = 0 (all level-0 slots retired) replaces 0 slots without error.
            client.levels[0].slots[1].retired = true;
            let before_zero: usize = client.levels[0].next_backup;
            let pending: PendingMultiLevelQuery = client.prepare_query(0, &mut rng).unwrap();
            assert_eq!(pending.level, 1);
            let answer: MultiLevelAnswer =
                MultiLevelServer::answer(&db, &params, &pending.query, client.epoch()).unwrap();
            let value: Vec<u8> = client.finish_query(&pending, &answer, &mut rng).unwrap();
            assert_eq!(value.as_slice(), &db[0..record_size]);
            assert_eq!(client.levels[0].next_backup, before_zero);
        }

        // Part 2: Winning-level PromoteOut sweep with M_0 = 4, w_0 = 2 (p = 0.25) and 3 retired
        // slots (A_0 = 1). Exact sweep probability is (1 - 0.25)^1 = 0.75 (~1500 / 2000), whereas
        // using M_0 = 4 would give (3/4)^4 = 81/256 ≈ 0.3164 (~633 / 2000).
        {
            let num_entries: usize = 16;
            let db: Vec<u8> = make_db(num_entries, record_size, 22);
            let params: MultiLevelParams = MultiLevelParams::custom(
                num_entries,
                record_size,
                vec![
                    LevelSpec {
                        block_size: 2,
                        num_blocks: 8,
                        num_primary: 4,
                        num_backup: 5_000,
                        backend: LevelBackend::FlatAes,
                    },
                    LevelSpec {
                        block_size: 2,
                        num_blocks: 8,
                        num_primary: 48,
                        num_backup: 2_000,
                        backend: LevelBackend::FlatAes,
                    },
                ],
            )
            .unwrap();
            let mut client: MultiLevelClient =
                MultiLevelClient::hint_init(params.clone(), &seed, &db, 1).unwrap();
            for pos in 0..3 {
                client.levels[0].slots[pos].retired = true;
            }

            let mut scratch: Vec<(u32, u8)> = Vec::new();
            let trials: usize = 2_000;
            let mut sweeps: usize = 0;
            let mut l0_hits: usize = 0;
            while l0_hits < trials {
                if let Some(hit_idx) = (0..num_entries).find(|&idx| {
                    !client.levels[0]
                        .find_matching_slots(idx, &mut scratch)
                        .is_empty()
                }) {
                    let before: usize = client.levels[0].next_backup;
                    let pending: PendingMultiLevelQuery =
                        client.prepare_query(hit_idx, &mut rng).unwrap();
                    assert_eq!(pending.level, 0);
                    let answer: MultiLevelAnswer =
                        MultiLevelServer::answer(&db, &params, &pending.query, client.epoch())
                            .unwrap();
                    let value: Vec<u8> = client.finish_query(&pending, &answer, &mut rng).unwrap();
                    assert_eq!(
                        value.as_slice(),
                        &db[hit_idx * record_size..(hit_idx + 1) * record_size]
                    );
                    let used: usize = client.levels[0].next_backup - before;
                    assert!(used == 1 || used == 2);
                    if used == 2 {
                        sweeps += 1;
                    }
                    l0_hits += 1;
                } else {
                    // Rare case (~2^-8) where the single active slot has no matching block;
                    // issue a level-1 query so level-0 refresh can repopulate it.
                    let pending: PendingMultiLevelQuery =
                        client.prepare_query(0, &mut rng).unwrap();
                    assert_eq!(pending.level, 1);
                    let answer: MultiLevelAnswer =
                        MultiLevelServer::answer(&db, &params, &pending.query, client.epoch())
                            .unwrap();
                    let value: Vec<u8> = client.finish_query(&pending, &answer, &mut rng).unwrap();
                    assert_eq!(value.as_slice(), &db[0..record_size]);
                }
            }
            assert!(
                (1_420..=1_580).contains(&sweeps),
                "expected ~1500 unbiased winning-level sweeps out of {trials}, got {sweeps}"
            );
        }
    }

    /// Reader wrapper that yields short reads and periodic `ErrorKind::Interrupted` errors
    /// to verify `read_fill` and streaming `HintInit` resilience.
    struct FlakyReader<'a> {
        data: &'a [u8],
        pos: usize,
        max_chunk: usize,
        calls: usize,
    }

    impl<'a> Read for FlakyReader<'a> {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            self.calls += 1;
            if self.calls.is_multiple_of(3) {
                return Err(io::Error::new(
                    io::ErrorKind::Interrupted,
                    "simulated EINTR",
                ));
            }
            if self.pos >= self.data.len() {
                return Ok(0);
            }
            let n: usize = buf
                .len()
                .min(self.max_chunk)
                .min(self.data.len() - self.pos);
            buf[..n].copy_from_slice(&self.data[self.pos..self.pos + n]);
            self.pos += n;
            Ok(n)
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    struct PrivateTempDir(std::path::PathBuf);

    #[cfg(not(target_arch = "wasm32"))]
    impl PrivateTempDir {
        fn new(prefix: &str) -> Self {
            use std::fs::DirBuilder;
            #[cfg(unix)]
            use std::os::unix::fs::DirBuilderExt;

            let base: std::path::PathBuf = std::env::temp_dir();
            let pid: u32 = std::process::id();
            let mut builder: DirBuilder = DirBuilder::new();
            #[cfg(unix)]
            builder.mode(0o700);

            for _ in 0..64 {
                let tag: u64 = next_session_id();
                let nanos: u128 = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos())
                    .unwrap_or(0);
                let dir: std::path::PathBuf = base.join(format!("{prefix}_{pid}_{tag}_{nanos}"));
                match builder.create(&dir) {
                    Ok(()) => return Self(dir),
                    Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
                    Err(e) => panic!("failed to create private temp dir: {e}"),
                }
            }
            panic!("failed to allocate unique private temp dir");
        }

        fn path(&self) -> &std::path::Path {
            &self.0
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    impl Drop for PrivateTempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn streaming_hint_init_matches_in_memory_for_hybrid_and_flat_with_unaligned_and_odd_sizes() {
        // Odd entry count (not divisible by w=16, 4, or 2) and non-8-byte-aligned record size.
        let num_entries: usize = 101;
        let record_size: usize = 37;
        let seed: [u8; 32] = [0xA7; 32];
        let db: Vec<u8> = make_db(num_entries, record_size, 777);

        for variant in [
            BackendVariant::FlatB,
            BackendVariant::HybridA { iprf_levels: 1 },
        ] {
            let params: MultiLevelParams =
                small_three_level_params(num_entries, record_size, variant);
            let baseline: MultiLevelClient =
                MultiLevelClient::hint_init(params.clone(), &seed, &db, 4).unwrap();
            let baseline_bytes: Vec<u8> = baseline.to_bytes().unwrap();

            // 1. Byte-by-byte push into MultiLevelHintBuilder with minimum chunk size (1 superblock).
            let mut byte_builder: MultiLevelHintBuilder =
                MultiLevelHintBuilder::with_chunk_bytes(params.clone(), &seed, 4, 1).unwrap();
            assert_eq!(byte_builder.expected_bytes(), db.len());
            for b in &db {
                byte_builder.push_bytes(&[*b]).unwrap();
            }
            assert_eq!(byte_builder.bytes_fed(), db.len());
            let from_bytes_client: MultiLevelClient = byte_builder.finish().unwrap();
            assert_eq!(from_bytes_client.to_bytes().unwrap(), baseline_bytes);

            // 2. Irregular slices (empty, tiny, > 1 superblock, partial) across multiple superblocks.
            let mut irreg_builder: MultiLevelHintBuilder =
                MultiLevelHintBuilder::with_chunk_bytes(params.clone(), &seed, 4, 1_200).unwrap();
            let pattern: [usize; 7] = [0, 13, 700, 1, 1_300, 7, 500];
            let mut offset: usize = 0;
            let mut step: usize = 0;
            while offset < db.len() {
                let take: usize = pattern[step % pattern.len()].min(db.len() - offset);
                irreg_builder
                    .push_bytes(&db[offset..offset + take])
                    .unwrap();
                offset += take;
                step += 1;
            }
            let from_irreg: MultiLevelClient = irreg_builder.finish().unwrap();
            assert_eq!(from_irreg.to_bytes().unwrap(), baseline_bytes);

            // 3. Flaky short-read + EINTR reader.
            let flaky = FlakyReader {
                data: &db,
                pos: 0,
                max_chunk: 19,
                calls: 0,
            };
            let from_reader: MultiLevelClient =
                MultiLevelClient::hint_init_from_reader_with_chunk_bytes(
                    params.clone(),
                    &seed,
                    flaky,
                    4,
                    900,
                )
                .unwrap();
            assert_eq!(from_reader.to_bytes().unwrap(), baseline_bytes);

            // 4. Direct disk file streaming with both small (1 superblock) and multi-superblock buffers.
            #[cfg(not(target_arch = "wasm32"))]
            {
                let dir = PrivateTempDir::new("plinko_ml_stream");
                let path: std::path::PathBuf = dir.path().join("database.bin");
                std::fs::write(&path, &db).unwrap();

                let mut from_file: MultiLevelClient =
                    MultiLevelClient::hint_init_from_file_with_chunk_bytes(
                        params.clone(),
                        &seed,
                        &path,
                        4,
                        1_500,
                    )
                    .unwrap();
                assert_eq!(from_file.to_bytes().unwrap(), baseline_bytes);

                // Verify `bench_level_blocks` accepts a larger `sample_db` than `sample_blocks`.
                let _ = MultiLevelClient::bench_level_blocks(&params, &seed, 0, &db, 2, 1);

                let mut rng: ChaCha20Rng = ChaCha20Rng::seed_from_u64(909);
                for &idx in &[0usize, 15, 16, 50, 100] {
                    let pending: PendingMultiLevelQuery =
                        from_file.prepare_query(idx, &mut rng).unwrap();
                    let ans: MultiLevelAnswer =
                        MultiLevelServer::answer(&db, &params, &pending.query, from_file.epoch())
                            .unwrap();
                    let val: Vec<u8> = from_file.finish_query(&pending, &ans, &mut rng).unwrap();
                    assert_eq!(
                        val.as_slice(),
                        &db[idx * record_size..(idx + 1) * record_size]
                    );
                }
            }
        }

        // Sub-superblock database (num_entries < max(block_size)).
        let tiny_n: usize = 5;
        let tiny_r: usize = 13;
        let tiny_db: Vec<u8> = make_db(tiny_n, tiny_r, 888);
        let tiny_params: MultiLevelParams =
            small_three_level_params(tiny_n, tiny_r, BackendVariant::FlatB);
        let tiny_mem: MultiLevelClient =
            MultiLevelClient::hint_init(tiny_params.clone(), &seed, &tiny_db, 2).unwrap();
        let tiny_stream: MultiLevelClient =
            MultiLevelClient::hint_init_from_reader(tiny_params, &seed, &tiny_db[..], 2).unwrap();
        assert_eq!(
            tiny_stream.to_bytes().unwrap(),
            tiny_mem.to_bytes().unwrap()
        );
    }

    #[test]
    fn streaming_hint_init_error_handling() {
        let num_entries: usize = 64;
        let record_size: usize = 16;
        let seed: [u8; 32] = [0x3C; 32];
        let db: Vec<u8> = make_db(num_entries, record_size, 555);
        let params: MultiLevelParams =
            small_three_level_params(num_entries, record_size, BackendVariant::FlatB);

        // chunk_bytes == 0 rejected.
        assert_eq!(
            MultiLevelHintBuilder::with_chunk_bytes(params.clone(), &seed, 1, 0).err(),
            Some(MultiLevelError::InvalidParams("chunk_bytes must be > 0"))
        );

        // Finishing early returns DatabaseLengthMismatch.
        let mut short_builder: MultiLevelHintBuilder =
            MultiLevelHintBuilder::new(params.clone(), &seed, 1).unwrap();
        short_builder.push_bytes(&db[..100]).unwrap();
        assert_eq!(
            short_builder.finish().err(),
            Some(MultiLevelError::DatabaseLengthMismatch {
                expected: db.len(),
                actual: 100,
            })
        );

        // Overfeeding returns DatabaseLengthMismatch immediately.
        let mut over_builder: MultiLevelHintBuilder =
            MultiLevelHintBuilder::new(params.clone(), &seed, 1).unwrap();
        over_builder.push_bytes(&db).unwrap();
        assert_eq!(
            over_builder.push_bytes(&[0u8]).err(),
            Some(MultiLevelError::DatabaseLengthMismatch {
                expected: db.len(),
                actual: db.len() + 1,
            })
        );

        // Truncated reader returns DatabaseLengthMismatch.
        assert_eq!(
            MultiLevelClient::hint_init_from_reader(params.clone(), &seed, &db[..db.len() - 5], 1,)
                .err(),
            Some(MultiLevelError::DatabaseLengthMismatch {
                expected: db.len(),
                actual: db.len() - 5,
            })
        );

        // Reader that returns EOF after a partial chunk must fail at first EOF even if more bytes
        // would appear on a subsequent read (e.g. concurrent append).
        struct EofThenMore<'a> {
            first: &'a [u8],
            second: &'a [u8],
            stage: u8,
        }
        impl Read for EofThenMore<'_> {
            fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
                match self.stage {
                    0 => {
                        self.stage = 1;
                        let n: usize = buf.len().min(self.first.len());
                        buf[..n].copy_from_slice(&self.first[..n]);
                        Ok(n)
                    }
                    1 => {
                        self.stage = 2;
                        Ok(0)
                    }
                    _ => {
                        let n: usize = buf.len().min(self.second.len());
                        buf[..n].copy_from_slice(&self.second[..n]);
                        Ok(n)
                    }
                }
            }
        }
        let split: usize = 64;
        assert_eq!(
            MultiLevelClient::hint_init_from_reader(
                params.clone(),
                &seed,
                EofThenMore {
                    first: &db[..split],
                    second: &db[split..],
                    stage: 0,
                },
                1,
            )
            .err(),
            Some(MultiLevelError::DatabaseLengthMismatch {
                expected: db.len(),
                actual: split,
            })
        );

        // Trailing bytes in reader return DatabaseLengthMismatch.
        let mut extra_db: Vec<u8> = db.clone();
        extra_db.push(0xFF);
        assert_eq!(
            MultiLevelClient::hint_init_from_reader(params.clone(), &seed, &extra_db[..], 1).err(),
            Some(MultiLevelError::DatabaseLengthMismatch {
                expected: db.len(),
                actual: db.len() + 1,
            })
        );

        #[cfg(not(target_arch = "wasm32"))]
        {
            let dir = PrivateTempDir::new("plinko_ml_err");
            let missing_path: std::path::PathBuf = dir.path().join("missing.bin");
            assert_eq!(
                MultiLevelClient::hint_init_from_file(params.clone(), &seed, &missing_path, 1)
                    .err(),
                Some(MultiLevelError::Io(io::ErrorKind::NotFound))
            );

            let bad_path: std::path::PathBuf = dir.path().join("badlen.bin");
            std::fs::write(&bad_path, &db[..db.len() - 1]).unwrap();
            assert_eq!(
                MultiLevelClient::hint_init_from_file(params.clone(), &seed, &bad_path, 1).err(),
                Some(MultiLevelError::DatabaseLengthMismatch {
                    expected: db.len(),
                    actual: db.len() - 1,
                })
            );

            std::fs::write(&bad_path, &extra_db).unwrap();
            assert_eq!(
                MultiLevelClient::hint_init_from_file(params, &seed, &bad_path, 1).err(),
                Some(MultiLevelError::DatabaseLengthMismatch {
                    expected: db.len(),
                    actual: db.len() + 1,
                })
            );
        }
    }
}
