//! Lightweight end-to-end Plinko protocol model.
//!
//! This module provides a practical, in-memory protocol flow on top of the
//! existing iPRF primitive:
//! - offline hint initialization (regular + backup), or a hint set that
//!   `plinko_hints` wrote to a file (see [`crate::hints`])
//! - online query construction
//! - server answer computation
//! - client reconstruction + backup promotion
//! - hint updates from deltas between database versions (see [`crate::delta`])
//!
//! The database has a version number. An answer carries the version of the
//! database that it comes from, and the client accepts it only if its hints
//! belong to the same version.
//!
//! A client that stops and starts again must keep its state: the hints that
//! it sent, the promoted backup hints, the cache and the queries in flight
//! ([`Client::write_state`], [`Client::read_state`]).
//!
//! The rules of the online phase (decoy, query format, retry, query limit) and
//! the deviations from the paper are in `docs/online_protocol.md`.
//!
//! In-memory model: the database is a flat byte array of `num_entries`
//! records of `record_size` bytes each ([`ProtocolParams::record_size`]). A
//! parity, an answer and the change of a record in a delta have the record size.

use crate::delta::{Delta, DeltaHeader};
use crate::hint_init;
use crate::hints::{self, DatabaseSnapshot, HintGeometry, HintSet, MAX_RECORD_SIZE};
use crate::iprf::{Iprf, PrfKey128};
use crate::subsets::{SubsetPrf, INVALID_CUTOFF};
use rand::seq::SliceRandom;
use rand::{Rng, RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use sha2::{Digest, Sha256};
use std::collections::{BTreeSet, HashMap};
use std::error::Error;
use std::fmt::{Display, Formatter};

mod state;

pub use state::{ClientStateError, CLIENT_STATE_MAGIC, CLIENT_STATE_VERSION};

/// PMNS tree levels that the client keeps per block (see [`Iprf::with_cached_levels`]).
///
/// Eight levels take 1020 bytes per block: about 29 MB for the 27,988 blocks of mainnet.
pub const CLIENT_CACHED_PMNS_LEVELS: u32 = 8;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProtocolError {
    InvalidParams(&'static str),
    InvalidIndex {
        index: usize,
        num_entries: usize,
    },
    NoDecoyIndexAvailable,
    NoHintForIndex {
        index: usize,
    },
    /// No backup hint is free: each one is promoted, retired, or reserved by a
    /// query in flight.
    ///
    /// From [`Client::prepare_query`]: the queries in flight can still get
    /// their answers, so complete them before you replace the client. When no
    /// query is in flight, the client must run HintInit again with a new
    /// master seed. From [`Client::reconstruct_and_replenish`]: see that
    /// function.
    NoBackupHintsLeft,
    /// A query for this index was sent and has no answer yet.
    QueryInFlight {
        index: usize,
    },
    /// The prepared query is not a query in flight of this client.
    ///
    /// The check compares the queried index, the requested index, the hint
    /// slot and the order of the two subsets. It does not identify the
    /// client instance that built the query.
    UnknownPreparedQuery,
    QueryShapeMismatch {
        expected_offsets: usize,
        got: usize,
    },
    MaskLengthMismatch {
        expected_bytes: usize,
        got: usize,
    },
    OffsetOutOfRange {
        block: usize,
        offset: usize,
    },
    /// A hint set does not belong to these parameters or to this master seed.
    HintSetMismatch(&'static str),
    /// A record or a parity does not have the record size of the database.
    RecordSizeMismatch {
        expected: usize,
        got: usize,
    },
    /// An answer or a delta does not belong to the database version of the
    /// hints. `expected` is the version of the hints. `got` is the version of
    /// the answer, or the version that the delta starts from.
    DatabaseVersionMismatch {
        expected: u64,
        got: u64,
    },
    /// A delta is for a database with another number of records or another record size.
    DeltaMismatch(&'static str),
}

impl Display for ProtocolError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidParams(msg) => write!(f, "invalid params: {msg}"),
            Self::InvalidIndex { index, num_entries } => {
                write!(f, "index {index} out of range [0, {num_entries})")
            }
            Self::NoDecoyIndexAvailable => write!(f, "no decoy index available"),
            Self::NoHintForIndex { index } => write!(f, "no hint found for index {index}"),
            Self::NoBackupHintsLeft => write!(f, "no backup hints left for promotion"),
            Self::QueryInFlight { index } => {
                write!(f, "a query for index {index} has no answer yet")
            }
            Self::UnknownPreparedQuery => {
                write!(f, "prepared query is not in flight for this client")
            }
            Self::QueryShapeMismatch {
                expected_offsets,
                got,
            } => {
                write!(
                    f,
                    "query offset length mismatch: expected {expected_offsets}, got {got}"
                )
            }
            Self::MaskLengthMismatch {
                expected_bytes,
                got,
            } => {
                write!(
                    f,
                    "query mask length mismatch: expected {expected_bytes} bytes, got {got}"
                )
            }
            Self::OffsetOutOfRange { block, offset } => {
                write!(f, "offset {offset} out of range for block {block}")
            }
            Self::HintSetMismatch(what) => {
                write!(f, "hint set does not match the client: {what}")
            }
            Self::RecordSizeMismatch { expected, got } => {
                write!(
                    f,
                    "record size mismatch: expected {expected} bytes, got {got}"
                )
            }
            Self::DatabaseVersionMismatch { expected, got } => {
                write!(
                    f,
                    "database version mismatch: the hints have version {expected}, got {got}"
                )
            }
            Self::DeltaMismatch(what) => {
                write!(f, "delta does not match the client: {what}")
            }
        }
    }
}

impl Error for ProtocolError {}

/// Parameters for the in-memory Plinko protocol model.
#[derive(Clone, Debug)]
pub struct ProtocolParams {
    pub num_entries: usize,
    /// Bytes of one database record.
    pub record_size: usize,
    pub block_size: usize,
    pub num_blocks: usize,
    pub num_regular_hints: usize,
    pub num_backup_hints: usize,
    pub num_total_hints: usize,
}

/// Default entries per block: the smallest power of two that is at least `sqrt(num_entries)`.
///
/// The block size is the iPRF range, and the PMNS sampler requires a power of two.
///
/// # Examples
///
/// ```
/// use plinko::protocol::default_block_size;
///
/// assert_eq!(default_block_size(1_834_095_877), 65_536);
/// assert_eq!(default_block_size(64), 8);
/// assert_eq!(default_block_size(65), 16);
/// ```
pub fn default_block_size(num_entries: usize) -> usize {
    let raw = (num_entries as f64).sqrt().ceil() as usize;
    raw.max(1).next_power_of_two()
}

const PADDED_DATABASE_TOO_LARGE: &str = "the padded database does not fit in memory";

/// Largest block size: the largest range of an iPRF (see [`Iprf::new`]).
pub const MAX_BLOCK_SIZE: u64 = 1 << 32;

impl ProtocolParams {
    /// Create parameters with defaults similar to the Python runnable spec:
    /// - `record_size`: bytes of one database record, 1 to [`MAX_RECORD_SIZE`]
    /// - `block_size`: [`default_block_size`] if omitted; must be a power of two
    ///   of at most 2^32, because the iPRF range is the block size
    /// - `num_backup_hints`: `lambda * block_size` if omitted
    ///
    /// # Errors
    /// Returns [`ProtocolError::InvalidParams`] for a value outside its range,
    /// and for a database that does not fit in memory with its padding
    /// (`num_blocks * block_size * record_size` bytes).
    ///
    /// # Examples
    ///
    /// ```
    /// use plinko::protocol::ProtocolParams;
    ///
    /// let params = ProtocolParams::new(1000, 40, None, 128, None).unwrap();
    /// assert_eq!((params.record_size, params.block_size, params.num_blocks), (40, 32, 32));
    /// assert!(ProtocolParams::new(1000, 0, None, 128, None).is_err());
    /// ```
    pub fn new(
        num_entries: usize,
        record_size: usize,
        block_size: Option<usize>,
        lambda: usize,
        num_backup_hints: Option<usize>,
    ) -> Result<Self, ProtocolError> {
        if num_entries == 0 {
            return Err(ProtocolError::InvalidParams("num_entries must be > 0"));
        }
        if lambda == 0 {
            return Err(ProtocolError::InvalidParams("lambda must be > 0"));
        }
        if !(1..=MAX_RECORD_SIZE).contains(&record_size) {
            return Err(ProtocolError::InvalidParams(
                "record_size must be in the range 1 to MAX_RECORD_SIZE",
            ));
        }
        if num_entries.checked_mul(record_size).is_none() {
            return Err(ProtocolError::InvalidParams(
                "num_entries * record_size overflow",
            ));
        }

        let block_size = block_size.unwrap_or_else(|| default_block_size(num_entries));
        if !block_size.is_power_of_two() || block_size as u64 > MAX_BLOCK_SIZE {
            return Err(ProtocolError::InvalidParams(
                "block_size must be a power of two of at most 2^32",
            ));
        }

        let mut num_blocks = num_entries.div_ceil(block_size);
        if num_blocks < 2 {
            num_blocks = 2;
        }
        if num_blocks % 2 != 0 {
            num_blocks = num_blocks
                .checked_add(1)
                .ok_or(ProtocolError::InvalidParams(PADDED_DATABASE_TOO_LARGE))?;
        }
        // The server holds the padded database in one buffer.
        let padded_bytes: Option<usize> = num_blocks
            .checked_mul(block_size)
            .and_then(|entries| entries.checked_mul(record_size));
        if !padded_bytes.is_some_and(|bytes| bytes <= isize::MAX as usize) {
            return Err(ProtocolError::InvalidParams(PADDED_DATABASE_TOO_LARGE));
        }

        let num_regular_hints = lambda
            .checked_mul(block_size)
            .ok_or(ProtocolError::InvalidParams("lambda * block_size overflow"))?;

        let num_backup_hints = num_backup_hints.unwrap_or(num_regular_hints);
        if num_backup_hints == 0 {
            return Err(ProtocolError::InvalidParams("num_backup_hints must be > 0"));
        }

        let num_total_hints =
            num_regular_hints
                .checked_add(num_backup_hints)
                .ok_or(ProtocolError::InvalidParams(
                    "num_regular_hints + num_backup_hints overflow",
                ))?;

        Ok(Self {
            num_entries,
            record_size,
            block_size,
            num_blocks,
            num_regular_hints,
            num_backup_hints,
            num_total_hints,
        })
    }

    /// Derives the parameters that a hint set was generated for.
    ///
    /// `lambda` is the number of regular hints divided by the block size.
    pub fn for_hint_set(hints: &HintSet) -> Result<Self, ProtocolError> {
        let geometry: HintGeometry = hints.geometry();
        let num_entries: usize = usize::try_from(geometry.num_entries)
            .map_err(|_| ProtocolError::HintSetMismatch("num_entries is too large"))?;
        let block_size: usize = usize::try_from(geometry.block_size)
            .map_err(|_| ProtocolError::HintSetMismatch("block_size is too large"))?;
        if block_size == 0 {
            return Err(ProtocolError::HintSetMismatch("block_size is zero"));
        }

        let lambda: usize = hints.num_regular() / block_size;
        if lambda * block_size != hints.num_regular() {
            return Err(ProtocolError::HintSetMismatch(
                "the number of regular hints is not a multiple of block_size",
            ));
        }
        let params: Self = Self::new(
            num_entries,
            hints.record_size(),
            Some(block_size),
            lambda,
            Some(hints.num_backup()),
        )?;
        if params.hint_geometry() != geometry {
            return Err(ProtocolError::HintSetMismatch(
                "num_blocks is not the block count of this database size",
            ));
        }
        Ok(params)
    }

    /// Database geometry that hints for these parameters must have.
    pub fn hint_geometry(&self) -> HintGeometry {
        HintGeometry {
            num_entries: self.num_entries as u64,
            // `new` keeps the record size at or below `MAX_RECORD_SIZE`.
            record_size: self.record_size as u32,
            block_size: self.block_size as u64,
            num_blocks: self.num_blocks as u64,
        }
    }

    /// Checks that the fields are values that [`ProtocolParams::new`] derives.
    ///
    /// The fields are public, so a caller can change one of them. A block
    /// count or a block size that does not hold `num_entries` records gives
    /// hints and a server that ignore records.
    fn check_derived(&self) -> Result<(), ProtocolError> {
        const CHANGED: ProtocolError =
            ProtocolError::InvalidParams("the parameters are not those of ProtocolParams::new");
        #[allow(clippy::manual_is_multiple_of)]
        if self.block_size == 0 || self.num_regular_hints % self.block_size != 0 {
            return Err(CHANGED);
        }
        let derived: Self = Self::new(
            self.num_entries,
            self.record_size,
            Some(self.block_size),
            self.num_regular_hints / self.block_size,
            Some(self.num_backup_hints),
        )?;
        if derived.num_blocks != self.num_blocks || derived.num_total_hints != self.num_total_hints
        {
            return Err(CHANGED);
        }
        Ok(())
    }

    #[inline]
    pub fn n_effective_entries(&self) -> usize {
        self.num_blocks * self.block_size
    }

    #[inline]
    pub fn block_of(&self, index: usize) -> usize {
        index / self.block_size
    }

    #[inline]
    pub fn offset_in_block(&self, index: usize) -> usize {
        index % self.block_size
    }
}

/// Query format with shared offsets: one bitmask and `num_blocks/2` offsets.
///
/// Bit `k` of `mask` puts block `k` in subset 0 (set) or subset 1 (clear).
/// The `i`-th block of each subset uses `offsets[i]`. RMS24 and Plinko Fig. 7
/// send one offset per block; `docs/online_protocol.md` gives the argument
/// for the shared form.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Query {
    pub mask: Vec<u8>,
    pub offsets: Vec<u32>,
}

/// Answer of the server: the XOR of the selected records of each subset.
/// Each parity has `record_size` bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Response {
    /// Version of the database that the parities come from.
    pub database_version: u64,
    pub parity_0: Vec<u8>,
    pub parity_1: Vec<u8>,
}

/// Server for the in-memory protocol model.
pub struct Server {
    params: ProtocolParams,
    database_version: u64,
    /// `n_effective_entries() * record_size` bytes; the padding records are zero.
    database: Vec<u8>,
}

impl Server {
    /// Creates a server for `database`, the records of the database one
    /// after another. `database_version` is the version of these records.
    ///
    /// # Errors
    /// Returns [`ProtocolError::InvalidParams`] if `database` does not have
    /// `num_entries * record_size` bytes, or if a field of `params` is not a
    /// value that [`ProtocolParams::new`] derives (the fields are public): a
    /// record size outside the range 1 to [`MAX_RECORD_SIZE`], or blocks
    /// that do not hold `num_entries` records.
    pub fn new(
        params: ProtocolParams,
        database_version: u64,
        database: &[u8],
    ) -> Result<Self, ProtocolError> {
        check_database_len(&params, database)?;

        let mut database: Vec<u8> = database.to_vec();
        database.resize(params.n_effective_entries() * params.record_size, 0u8);

        Ok(Self {
            params,
            database_version,
            database,
        })
    }

    /// Version of the database that the server answers from.
    pub fn database_version(&self) -> u64 {
        self.database_version
    }

    /// Returns the record at `index`, or `None` if `index >= num_entries`.
    pub fn record(&self, index: usize) -> Option<&[u8]> {
        (index < self.params.num_entries).then(|| self.record_at(index))
    }

    fn record_at(&self, index: usize) -> &[u8] {
        let size: usize = self.params.record_size;
        &self.database[index * size..(index + 1) * size]
    }

    /// Computes the two parities of a query (Plinko Fig. 7, `Answer`).
    pub fn answer(&self, query: &Query) -> Result<Response, ProtocolError> {
        let half = self.params.num_blocks / 2;
        if query.offsets.len() != half {
            return Err(ProtocolError::QueryShapeMismatch {
                expected_offsets: half,
                got: query.offsets.len(),
            });
        }

        let mask_bytes: usize = self.params.num_blocks.div_ceil(8);
        if query.mask.len() != mask_bytes {
            return Err(ProtocolError::MaskLengthMismatch {
                expected_bytes: mask_bytes,
                got: query.mask.len(),
            });
        }

        let mut parity_0: Vec<u8> = vec![0u8; self.params.record_size];
        let mut parity_1: Vec<u8> = vec![0u8; self.params.record_size];
        let mut idx_0 = 0usize;
        let mut idx_1 = 0usize;

        for block in 0..self.params.num_blocks {
            let (parity, rank): (&mut [u8], &mut usize) = if bit_is_set(&query.mask, block) {
                (&mut parity_0, &mut idx_0)
            } else {
                (&mut parity_1, &mut idx_1)
            };
            let Some(&offset) = query.offsets.get(*rank) else {
                return Err(ProtocolError::QueryShapeMismatch {
                    expected_offsets: half,
                    got: query.offsets.len(),
                });
            };
            let offset: usize = offset as usize;
            if offset >= self.params.block_size {
                return Err(ProtocolError::OffsetOutOfRange { block, offset });
            }
            xor_entry(
                parity,
                self.record_at(block * self.params.block_size + offset),
            );
            *rank += 1;
        }

        if idx_0 != half || idx_1 != half {
            return Err(ProtocolError::QueryShapeMismatch {
                expected_offsets: half,
                got: query.offsets.len(),
            });
        }

        Ok(Response {
            database_version: self.database_version,
            parity_0,
            parity_1,
        })
    }

    /// Writes new values of records, moves the database to the next version
    /// (`database_version() + 1`), and returns the delta between the two
    /// versions.
    ///
    /// Each update is `(index, new value)`. A call with no update also makes
    /// a new version. On an error no record and no version is changed.
    ///
    /// # Errors
    /// Returns [`ProtocolError::InvalidIndex`] for an index outside the
    /// database, [`ProtocolError::RecordSizeMismatch`] for a value that
    /// does not have `record_size` bytes, and
    /// [`ProtocolError::InvalidParams`] if the version is `u64::MAX`.
    pub fn apply_updates(&mut self, updates: &[(usize, Vec<u8>)]) -> Result<Delta, ProtocolError> {
        let size: usize = self.params.record_size;
        for (index, new_value) in updates {
            check_index(&self.params, *index)?;
            check_record_len(&self.params, new_value)?;
        }
        let header = DeltaHeader {
            num_entries: self.params.num_entries as u64,
            // `Server::new` keeps the record size at or below `MAX_RECORD_SIZE`.
            record_size: size as u32,
            from_version: self.database_version,
            to_version: self
                .database_version
                .checked_add(1)
                .ok_or(ProtocolError::InvalidParams("no database version is left"))?,
        };

        let mut changes: Vec<(u64, Vec<u8>)> = Vec::with_capacity(updates.len());
        for (index, new_value) in updates {
            let record: &mut [u8] = &mut self.database[index * size..(index + 1) * size];
            let mut delta: Vec<u8> = record.to_vec();
            xor_entry(&mut delta, new_value);
            record.copy_from_slice(new_value);
            changes.push((*index as u64, delta));
        }

        self.database_version = header.to_version;
        Ok(Delta::new(header, changes).expect("the header and the changes were checked above"))
    }
}

/// State of one hint slot. Its parity is in the hint set of the client: the
/// parity of a regular hint, or the "in" parity of a promoted backup hint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct HintSlot {
    /// Cutoff of the original (pre-complement) subset; see [`crate::subsets`].
    cutoff: u32,
    complemented: bool,
    extra_index: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct CacheEntry {
    value: Vec<u8>,
    hint_idx: usize,
}

/// What the client keeps for a query in flight, under the queried index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct InFlight {
    hint_idx: usize,
    requested_index: usize,
    real_is_first: bool,
}

/// A query in flight: built for the server, with no answer yet.
///
/// Keep the value until [`Client::reconstruct_and_replenish`] accepts an
/// answer. To retry, send `query` again without a change. To stop, call
/// [`Client::abandon_query`]. After a restart, [`Client::read_state`] gives
/// the queries in flight again.
pub struct PreparedQuery {
    pub query: Query,
    requested_index: usize,
    queried_index: usize,
    hint_idx: usize,
    real_is_first: bool,
}

impl PreparedQuery {
    /// Returns the index of the hint slot that this query uses.
    pub fn hint_index(&self) -> usize {
        self.hint_idx
    }

    /// Returns the database index queried on the wire (equals the requested
    /// index on a cache miss, or the chosen decoy index on a cache hit).
    pub fn queried_index(&self) -> usize {
        self.queried_index
    }
}

/// Minimum number of real blocks in a query before [`Client::prepare_query`]
/// distributes iPRF forward calls across worker threads.
const PARALLEL_QUERY_MIN_BLOCKS: usize = 64;

/// Minimum blocks assigned to each worker thread in [`Client::prepare_query`].
const PARALLEL_QUERY_BLOCKS_PER_WORKER: usize = 32;

/// Minimum number of changed records in a delta before [`Client::apply_delta`]
/// distributes iPRF inverse calls across worker threads.
const PARALLEL_DELTA_MIN_CHANGES: usize = 8;

/// Maximum number of delta changes whose iPRF inverses are held in memory at
/// once during parallel [`Client::apply_delta`].
const PARALLEL_DELTA_BATCH_SIZE: usize = 256;

/// Client for the in-memory protocol model.
pub struct Client {
    params: ProtocolParams,
    iprfs: Vec<Iprf>,
    subset_prf: SubsetPrf,
    /// Cutoffs and parities. The parity of slot `j` is the parity of regular
    /// hint `j`, or the "in" parity of backup hint `j - num_regular_hints`
    /// after its promotion (the "out" parity is not used after that).
    hints: HintSet,
    slots: Vec<Option<HintSlot>>,
    /// `false` after promotion, and for a backup hint with no valid cutoff.
    backup_available: Vec<bool>,
    /// Number of `true` entries in `backup_available`.
    remaining_backups: usize,
    /// Lowest index in `backup_available` that is `true`, or `backup_available.len()`.
    next_backup_hint: usize,
    cache: HashMap<usize, CacheEntry>,
    rng: ChaCha20Rng,
    /// Each query in flight, under its queried index.
    in_flight: HashMap<usize, InFlight>,
    /// Backup hints that [`Client::retire_hints`] removed before a promotion.
    retired_backups: BTreeSet<usize>,
    /// `true` after the first query was built. The hint set then holds state
    /// that the hint file cannot store.
    queried: bool,
    /// Worker threads used by [`Client::prepare_query`] and [`Client::apply_delta`].
    threads: usize,
}

impl Client {
    /// Run offline HintInit from a flat database of `num_entries * record_size` bytes,
    /// which is version `database_version` of the database.
    ///
    /// Equal to [`compute_hints`] followed by [`Client::from_hints`]. HintInit uses all
    /// available cores.
    pub fn offline_init(
        params: ProtocolParams,
        master_seed: [u8; 32],
        query_rng_seed: [u8; 32],
        database_version: u64,
        database: &[u8],
    ) -> Result<Self, ProtocolError> {
        let hints: HintSet = compute_hints(&params, &master_seed, database_version, database)?;
        Self::from_hints(params, master_seed, query_rng_seed, hints)
    }

    /// Builds a client from a hint set, for example one that `plinko_hints` wrote to a file.
    ///
    /// Returns [`ProtocolError::HintSetMismatch`] if the record size, the
    /// geometry or the hint counts differ from `params`, or if the hints were
    /// made with another master seed. The function does not recompute cutoffs
    /// or parities.
    ///
    /// The client starts at the database version of the hint set
    /// ([`HintSet::database_version`]).
    ///
    /// `query_rng_seed` seeds the random choices of the queries (decoy, hint
    /// order, order of the two subsets). Use new random bytes for each
    /// client; a seed that an earlier client used with the same hints
    /// repeats choices of queries that the server saw.
    ///
    /// The hint set must be one that no client has sent a query with. A hint
    /// file has no field for the hints that queries used. A client that loads
    /// the same file a second time, after queries, uses those hints again,
    /// and the server then learns the queried indices (`docs/online_protocol.md`).
    /// The function cannot detect this. A client that sent a query starts
    /// again from its state file: [`Client::write_state`] and
    /// [`Client::read_state`].
    pub fn from_hints(
        params: ProtocolParams,
        master_seed: [u8; 32],
        query_rng_seed: [u8; 32],
        hints: HintSet,
    ) -> Result<Self, ProtocolError> {
        params.check_derived()?;
        if hints.record_size() != params.record_size {
            return Err(ProtocolError::HintSetMismatch("record size"));
        }
        if hints.geometry() != params.hint_geometry() {
            return Err(ProtocolError::HintSetMismatch("database geometry"));
        }
        if hints.num_regular() != params.num_regular_hints {
            return Err(ProtocolError::HintSetMismatch("number of regular hints"));
        }
        if hints.num_backup() != params.num_backup_hints {
            return Err(ProtocolError::HintSetMismatch("number of backup hints"));
        }
        if hints.seed_check() != hints::seed_check(&master_seed) {
            return Err(ProtocolError::HintSetMismatch("master seed"));
        }

        let mut slots: Vec<Option<HintSlot>> = Vec::with_capacity(params.num_total_hints);
        slots.extend(hints.regular_cutoffs().iter().map(|&cutoff| {
            (cutoff != INVALID_CUTOFF).then_some(HintSlot {
                cutoff,
                complemented: false,
                extra_index: None,
            })
        }));
        slots.resize(params.num_total_hints, None);

        let backup_available: Vec<bool> = hints
            .backup_cutoffs()
            .iter()
            .map(|&cutoff| cutoff != INVALID_CUTOFF)
            .collect();
        let remaining_backups: usize = backup_available.iter().filter(|&&free| free).count();
        let next_backup_hint: usize = backup_available
            .iter()
            .position(|&free| free)
            .unwrap_or(backup_available.len());

        Ok(Self {
            iprfs: block_iprfs(&params, &master_seed),
            subset_prf: SubsetPrf::new(&master_seed),
            params,
            hints,
            slots,
            backup_available,
            remaining_backups,
            next_backup_hint,
            cache: HashMap::new(),
            rng: ChaCha20Rng::from_seed(query_rng_seed),
            in_flight: HashMap::new(),
            retired_backups: BTreeSet::new(),
            queried: false,
            threads: hint_init::available_threads(),
        })
    }

    /// Returns the worker thread count used for [`Client::prepare_query`] and
    /// [`Client::apply_delta`].
    ///
    /// Defaults to [`crate::hint_init::available_threads`] (always 1 on `wasm32`).
    pub fn threads(&self) -> usize {
        self.threads
    }

    /// Sets the worker thread count used for [`Client::prepare_query`] and
    /// [`Client::apply_delta`].
    ///
    /// A value of `0` counts as `1`. On `wasm32` targets this is always clamped
    /// to `1`. Outputs do not depend on the thread count.
    pub fn set_threads(&mut self, threads: usize) {
        self.threads = if cfg!(target_arch = "wasm32") {
            1
        } else {
            threads.max(1)
        };
    }

    /// Returns the protocol parameters of this client.
    pub fn params(&self) -> &ProtocolParams {
        &self.params
    }

    /// Version of the database that the hints belong to.
    ///
    /// The client accepts an answer of this version only, and a delta that
    /// starts from this version only.
    pub fn database_version(&self) -> u64 {
        self.hints.database_version()
    }

    /// Returns the hint set if the client has built no query, and `None` after that.
    ///
    /// Write it with [`HintSet::write_to`] to keep the deltas that the client
    /// applied. After the first query the hints hold state that the hint file
    /// format cannot store; use [`Client::write_state`] then.
    pub fn unused_hints(&self) -> Option<&HintSet> {
        (!self.queried).then_some(&self.hints)
    }

    /// Builds the query for `requested_index` (Plinko Fig. 7, `Query`).
    ///
    /// For a cached index the query goes to a uniform index that is not cached
    /// and not in flight. The hint stays in the client, reserved, until
    /// [`Client::reconstruct_and_replenish`] or [`Client::abandon_query`].
    ///
    /// Each query in flight reserves one backup hint, so
    /// [`ProtocolError::NoBackupHintsLeft`] comes before a query is built. On
    /// an error no hint is reserved.
    pub fn prepare_query(
        &mut self,
        requested_index: usize,
    ) -> Result<PreparedQuery, ProtocolError> {
        check_index(&self.params, requested_index)?;
        if self.remaining_backup_hints() <= self.in_flight.len() {
            return Err(ProtocolError::NoBackupHintsLeft);
        }
        if self.in_flight.contains_key(&requested_index) {
            return Err(ProtocolError::QueryInFlight {
                index: requested_index,
            });
        }

        let queried_index = if self.cache.contains_key(&requested_index) {
            self.random_decoy()?
        } else {
            requested_index
        };

        let block = self.params.block_of(queried_index);
        let offset = self.params.offset_in_block(queried_index);
        let mut candidates: Vec<u64> = self.iprfs[block].inverse(offset as u64);
        candidates.shuffle(&mut self.rng);

        for candidate in candidates {
            let hint_idx = candidate as usize;
            if hint_idx >= self.params.num_total_hints || self.hint_is_reserved(hint_idx) {
                continue;
            }

            let Some(slot) = self.slots[hint_idx] else {
                continue;
            };

            if !self.slot_block_in_subset(hint_idx, &slot, block)
                && slot.extra_index != Some(queried_index)
            {
                continue;
            }

            let real_is_first = (self.rng.next_u32() & 1) == 0;
            let Some(query) = self.try_build_query(hint_idx, queried_index, &slot, real_is_first)
            else {
                continue;
            };

            self.in_flight.insert(
                queried_index,
                InFlight {
                    hint_idx,
                    requested_index,
                    real_is_first,
                },
            );
            self.queried = true;
            return Ok(PreparedQuery {
                query,
                requested_index,
                queried_index,
                hint_idx,
                real_is_first,
            });
        }

        Err(ProtocolError::NoHintForIndex {
            index: queried_index,
        })
    }

    /// Computes the answer, consumes the hint of the query, and promotes a
    /// backup hint in its place (Plinko Fig. 7, `Recon`).
    ///
    /// Returns the record at the requested index (`record_size` bytes). The
    /// query is then no longer in flight.
    ///
    /// # Errors
    /// On each of these errors the client does not change, and the query
    /// stays in flight:
    ///
    /// - [`ProtocolError::UnknownPreparedQuery`]: `prepared` is not a query in
    ///   flight of this client.
    /// - [`ProtocolError::DatabaseVersionMismatch`]: the answer comes from
    ///   another database version than the hints have. If the version of the
    ///   answer is above [`Client::database_version`], apply the deltas up to
    ///   that version and call the function again with the same answer. If
    ///   it is below, send `prepared.query` again.
    /// - [`ProtocolError::RecordSizeMismatch`]: a parity of `response` does
    ///   not have the record size. Such a response is not an answer.
    /// - [`ProtocolError::NoBackupHintsLeft`]: no backup hint can take the
    ///   place of the hint. This occurs only after [`Client::retire_hints`]
    ///   removed a backup hint that a query in flight had reserved. Call
    ///   [`Client::abandon_query`].
    pub fn reconstruct_and_replenish(
        &mut self,
        prepared: &PreparedQuery,
        response: &Response,
    ) -> Result<Vec<u8>, ProtocolError> {
        self.check_in_flight(prepared)?;
        let expected: u64 = self.database_version();
        if response.database_version != expected {
            return Err(ProtocolError::DatabaseVersionMismatch {
                expected,
                got: response.database_version,
            });
        }
        check_record_len(&self.params, &response.parity_0)?;
        check_record_len(&self.params, &response.parity_1)?;
        if self.remaining_backups == 0 {
            return Err(ProtocolError::NoBackupHintsLeft);
        }

        self.take_in_flight(prepared)?;
        let mut queried_entry: Vec<u8> = if prepared.real_is_first {
            response.parity_0.clone()
        } else {
            response.parity_1.clone()
        };
        xor_entry(&mut queried_entry, self.slot_parity(prepared.hint_idx));

        let promoted_hint_idx = self.promote_backup(prepared.queried_index, &queried_entry)?;
        self.cache.insert(
            prepared.queried_index,
            CacheEntry {
                value: queried_entry,
                hint_idx: promoted_hint_idx,
            },
        );

        // For a decoy query the requested index is another, older cache entry.
        let Some(entry) = self.cache.get(&prepared.requested_index) else {
            return Err(ProtocolError::NoHintForIndex {
                index: prepared.requested_index,
            });
        };
        Ok(entry.value.clone())
    }

    /// Stops a query in flight that gets no answer.
    ///
    /// The hint is removed, because the server saw its subset. The index is
    /// not cached and no backup hint is used. Prefer a retry: the proof of
    /// RMS24 does not cover a hint that is removed with no promotion (see
    /// `docs/online_protocol.md`).
    pub fn abandon_query(&mut self, prepared: PreparedQuery) -> Result<(), ProtocolError> {
        self.take_in_flight(&prepared).map(|_| ())
    }

    /// Cancels an unsent in-flight query without retiring its hint slot.
    ///
    /// # Security
    /// Call this **only** if `prepared.query` has never left the client (for
    /// example when rolling back a multi-segment or batch preparation after a
    /// later segment fails locally). If the query may have reached the server,
    /// call [`Client::abandon_query`] instead so the exposed hint is retired.
    pub fn cancel_unsent_query(&mut self, prepared: PreparedQuery) -> Result<(), ProtocolError> {
        self.check_in_flight(&prepared)?;
        self.in_flight.remove(&prepared.queried_index);
        Ok(())
    }

    /// Validates `response` against `prepared` and current client state without
    /// mutating any hints or reservations.
    pub(crate) fn validate_response(
        &self,
        prepared: &PreparedQuery,
        response: &Response,
    ) -> Result<(), ProtocolError> {
        self.check_in_flight(prepared)?;
        let expected: u64 = self.database_version();
        if response.database_version != expected {
            return Err(ProtocolError::DatabaseVersionMismatch {
                expected,
                got: response.database_version,
            });
        }
        check_record_len(&self.params, &response.parity_0)?;
        check_record_len(&self.params, &response.parity_1)?;
        if self.remaining_backups == 0 {
            return Err(ProtocolError::NoBackupHintsLeft);
        }
        Ok(())
    }

    /// Validates a batch of `(prepared, response)` pairs without mutating hints.
    pub(crate) fn validate_batch_responses(
        &self,
        prepareds: &[&PreparedQuery],
        responses: &[Response],
    ) -> Result<(), ProtocolError> {
        if prepareds.len() != responses.len() {
            return Err(ProtocolError::InvalidParams(
                "prepared query count does not match response count",
            ));
        }
        let mut seen_indices: std::collections::HashSet<usize> =
            std::collections::HashSet::with_capacity(prepareds.len());
        for (&prepared, response) in prepareds.iter().zip(responses.iter()) {
            if !seen_indices.insert(prepared.queried_index) {
                return Err(ProtocolError::UnknownPreparedQuery);
            }
            self.validate_response(prepared, response)?;
        }
        if self.remaining_backups < prepareds.len() {
            return Err(ProtocolError::NoBackupHintsLeft);
        }
        Ok(())
    }

    /// Validates that every hint index in `hint_indices` belongs to this client.
    pub(crate) fn validate_retire_hints(
        &self,
        hint_indices: &[usize],
    ) -> Result<(), ProtocolError> {
        if hint_indices
            .iter()
            .any(|&hint_idx| hint_idx >= self.params.num_total_hints)
        {
            return Err(ProtocolError::InvalidParams(
                "a hint to retire is not a hint of this client",
            ));
        }
        Ok(())
    }

    /// Applies a delta to the hints and moves the client to the version
    /// after it: one iPRF inverse for each changed record.
    ///
    /// The delta must start from [`Client::database_version`]. Apply the
    /// deltas of a database in the order of their versions.
    ///
    /// A version number identifies a state of one database, and a delta has
    /// no field that identifies its database. The caller must pass only
    /// deltas of the database that the hints belong to. A delta of another
    /// database with the same record size, number of records and version is
    /// accepted, and the hints are then wrong with no error.
    ///
    /// # Errors
    /// On an error no hint and no version is changed.
    ///
    /// - [`ProtocolError::DatabaseVersionMismatch`]: the delta starts from
    ///   another version. A delta that the client applied before starts
    ///   below the version of the client; a delta that comes too early
    ///   starts above it.
    /// - [`ProtocolError::DeltaMismatch`]: the delta is for a database with
    ///   another number of records or another record size.
    pub fn apply_delta(&mut self, delta: &Delta) -> Result<(), ProtocolError> {
        let header: DeltaHeader = delta.header();
        if header.record_size as usize != self.params.record_size {
            return Err(ProtocolError::DeltaMismatch("record size"));
        }
        if header.num_entries != self.params.num_entries as u64 {
            return Err(ProtocolError::DeltaMismatch("number of records"));
        }
        let expected: u64 = self.database_version();
        if header.from_version != expected {
            return Err(ProtocolError::DatabaseVersionMismatch {
                expected,
                got: header.from_version,
            });
        }

        let workers: usize = if cfg!(target_arch = "wasm32") {
            1
        } else {
            self.threads.max(1)
        };
        if workers > 1 && delta.len() >= PARALLEL_DELTA_MIN_CHANGES {
            let mut changes_iter = delta
                .changes()
                .map(|(index, change)| (index as usize, change));
            let mut batch: Vec<(usize, &[u8])> = Vec::with_capacity(PARALLEL_DELTA_BATCH_SIZE);
            loop {
                batch.clear();
                batch.extend(changes_iter.by_ref().take(PARALLEL_DELTA_BATCH_SIZE));
                if batch.is_empty() {
                    break;
                }
                let worker_count: usize = workers.min(batch.len());
                let chunk_len: usize = batch.len().div_ceil(worker_count);
                let mut batch_hints: Vec<Vec<u64>> = vec![Vec::new(); batch.len()];
                let iprfs: &[Iprf] = &self.iprfs;
                let params: &ProtocolParams = &self.params;

                std::thread::scope(|scope| {
                    for (change_chunk, out_chunk) in batch
                        .chunks(chunk_len)
                        .zip(batch_hints.chunks_mut(chunk_len))
                    {
                        scope.spawn(move || {
                            for (&(index, _), out) in change_chunk.iter().zip(out_chunk.iter_mut())
                            {
                                let block: usize = params.block_of(index);
                                let offset: usize = params.offset_in_block(index);
                                *out = iprfs[block].inverse(offset as u64);
                            }
                        });
                    }
                });

                for (&(index, change), hint_indices) in batch.iter().zip(batch_hints.iter()) {
                    self.apply_change_with_hints(index, change, hint_indices);
                }
            }
        } else {
            // An index of the delta is below `num_entries`, which is a `usize`.
            for (index, change) in delta.changes() {
                self.apply_change(index as usize, change);
            }
        }
        self.hints.set_database_version(header.to_version);
        Ok(())
    }

    /// XORs the change of the record at `index` into each parity that holds
    /// the record, and into its cached value.
    fn apply_change(&mut self, index: usize, change: &[u8]) {
        let block = self.params.block_of(index);
        let offset = self.params.offset_in_block(index);
        let hint_indices = self.iprfs[block].inverse(offset as u64);
        self.apply_change_with_hints(index, change, &hint_indices);
    }

    fn apply_change_with_hints(&mut self, index: usize, change: &[u8], hint_indices: &[u64]) {
        let block = self.params.block_of(index);

        for &hint_idx in hint_indices {
            let j = hint_idx as usize;
            if j >= self.params.num_total_hints {
                continue;
            }

            let slot_has_block: bool = self.slots[j]
                .as_ref()
                .is_some_and(|slot| self.slot_block_in_subset(j, slot, block));
            if slot_has_block {
                xor_entry(self.slot_parity_mut(j), change);
            } else if let Some(backup_idx) = j.checked_sub(self.params.num_regular_hints) {
                if self.backup_available[backup_idx] {
                    let cutoff: u32 = self.hints.backup_cutoffs()[backup_idx];
                    if self.subset_prf.contains(hint_idx, block, cutoff) {
                        xor_entry(self.hints.backup_parity_in_mut(backup_idx), change);
                    } else {
                        xor_entry(self.hints.backup_parity_out_mut(backup_idx), change);
                    }
                }
            }
        }

        let cached_slot: Option<usize> = self.cache.get_mut(&index).map(|cache| {
            xor_entry(&mut cache.value, change);
            cache.hint_idx
        });
        if let Some(slot_idx) = cached_slot {
            if self.slots[slot_idx].is_some() {
                xor_entry(self.slot_parity_mut(slot_idx), change);
            }
        }
    }

    /// Returns the number of backup hints that an answer can still promote:
    /// those that are not promoted, not retired, and have a valid cutoff.
    pub fn remaining_backup_hints(&self) -> usize {
        self.remaining_backups
    }

    /// Marks `backup_idx` as unavailable and updates the availability counters.
    ///
    /// Returns `true` if the backup hint was available before this call.
    fn consume_backup(&mut self, backup_idx: usize) -> bool {
        if !self.backup_available[backup_idx] {
            return false;
        }
        self.backup_available[backup_idx] = false;
        self.remaining_backups -= 1;
        if backup_idx == self.next_backup_hint {
            while self.next_backup_hint < self.backup_available.len()
                && !self.backup_available[self.next_backup_hint]
            {
                self.next_backup_hint += 1;
            }
        }
        true
    }

    /// Parity of hint slot `hint_idx`; see the `hints` field.
    fn slot_parity(&self, hint_idx: usize) -> &[u8] {
        match hint_idx.checked_sub(self.params.num_regular_hints) {
            None => self.hints.regular_parity(hint_idx),
            Some(backup_idx) => self.hints.backup_parity_in(backup_idx),
        }
    }

    fn slot_parity_mut(&mut self, hint_idx: usize) -> &mut [u8] {
        match hint_idx.checked_sub(self.params.num_regular_hints) {
            None => self.hints.regular_parity_mut(hint_idx),
            Some(backup_idx) => self.hints.backup_parity_in_mut(backup_idx),
        }
    }

    #[inline]
    pub fn num_cached_entries(&self) -> usize {
        self.cache.len()
    }

    /// Returns `true` if `index` is currently cached locally by this client.
    #[inline]
    pub fn is_cached(&self, index: usize) -> bool {
        self.cache.contains_key(&index)
    }

    /// Returns `true` if a query targeting wire index `index` is currently in flight.
    #[inline]
    pub fn is_in_flight(&self, index: usize) -> bool {
        self.in_flight.contains_key(&index)
    }

    /// Returns the number of queries that have no answer yet.
    #[inline]
    pub fn queries_in_flight(&self) -> usize {
        self.in_flight.len()
    }

    fn hint_is_reserved(&self, hint_idx: usize) -> bool {
        self.in_flight
            .values()
            .any(|reserved| reserved.hint_idx == hint_idx)
    }

    fn check_in_flight(&self, prepared: &PreparedQuery) -> Result<(), ProtocolError> {
        let expected = InFlight {
            hint_idx: prepared.hint_idx,
            requested_index: prepared.requested_index,
            real_is_first: prepared.real_is_first,
        };
        if self.in_flight.get(&prepared.queried_index) != Some(&expected) {
            return Err(ProtocolError::UnknownPreparedQuery);
        }
        Ok(())
    }

    fn take_in_flight(&mut self, prepared: &PreparedQuery) -> Result<HintSlot, ProtocolError> {
        self.check_in_flight(prepared)?;
        self.in_flight.remove(&prepared.queried_index);
        self.slots[prepared.hint_idx]
            .take()
            .ok_or(ProtocolError::UnknownPreparedQuery)
    }

    fn is_decoy_candidate(&self, index: usize) -> bool {
        !self.cache.contains_key(&index) && !self.in_flight.contains_key(&index)
    }

    /// Draws a uniform index among those that are not cached and not in flight
    /// (Plinko Fig. 7: `While Q[i] != ⊥: i <-$ [n]`).
    fn random_decoy(&mut self) -> Result<usize, ProtocolError> {
        let num_entries: usize = self.params.num_entries;
        let num_free: usize = num_entries.saturating_sub(self.cache.len() + self.in_flight.len());
        if num_free == 0 {
            return Err(ProtocolError::NoDecoyIndexAvailable);
        }

        if num_free >= num_entries / 2 {
            loop {
                let index: usize = self.rng.gen_range(0..num_entries);
                if self.is_decoy_candidate(index) {
                    return Ok(index);
                }
            }
        }

        let free: Vec<usize> = (0..num_entries)
            .filter(|&index| self.is_decoy_candidate(index))
            .collect();
        free.choose(&mut self.rng)
            .copied()
            .ok_or(ProtocolError::NoDecoyIndexAvailable)
    }

    fn try_build_query(
        &self,
        hint_idx: usize,
        queried_index: usize,
        slot: &HintSlot,
        real_is_first: bool,
    ) -> Option<Query> {
        let queried_block = self.params.block_of(queried_index);
        let half = self.params.num_blocks / 2;
        let mut mask = vec![0u8; self.params.num_blocks.div_ceil(8)];

        let extra_block = slot.extra_index.map(|idx| self.params.block_of(idx));
        let extra_offset = slot.extra_index.map(|idx| self.params.offset_in_block(idx));

        let mut select_values: Vec<u32> = Vec::with_capacity(self.params.num_blocks);
        self.subset_prf
            .select_values(hint_idx as u64, self.params.num_blocks, &mut select_values);

        let mut query_iprfs: Vec<&Iprf> = Vec::with_capacity(half);
        let mut extra_pos: Option<usize> = None;
        let mut real_count = 0usize;

        for (block, &select_value) in select_values.iter().enumerate() {
            let in_subset = (select_value < slot.cutoff) ^ slot.complemented;
            let is_real = block != queried_block && (in_subset || extra_block == Some(block));

            if is_real {
                if extra_block == Some(block) {
                    extra_pos = Some(real_count);
                } else {
                    query_iprfs.push(&self.iprfs[block]);
                }
                real_count += 1;
            }

            if is_real == real_is_first {
                set_bit(&mut mask, block);
            }
        }

        if real_count != half {
            return None;
        }

        let x: u64 = hint_idx as u64;
        let workers: usize =
            if cfg!(target_arch = "wasm32") || query_iprfs.len() < PARALLEL_QUERY_MIN_BLOCKS {
                1
            } else {
                self.threads.clamp(
                    1,
                    (query_iprfs.len() / PARALLEL_QUERY_BLOCKS_PER_WORKER).max(1),
                )
            };

        let forward_offsets: Vec<u64> = if workers == 1 {
            Iprf::forward_many(&query_iprfs, x)
        } else {
            let chunk_len: usize = query_iprfs.len().div_ceil(workers);
            let mut out: Vec<u64> = vec![0u64; query_iprfs.len()];
            std::thread::scope(|scope| {
                for (iprf_chunk, out_chunk) in
                    query_iprfs.chunks(chunk_len).zip(out.chunks_mut(chunk_len))
                {
                    scope.spawn(move || {
                        let chunk_offsets: Vec<u64> = Iprf::forward_many(iprf_chunk, x);
                        out_chunk.copy_from_slice(&chunk_offsets);
                    });
                }
            });
            out
        };

        let mut offsets = Vec::with_capacity(half);
        let mut forward_iter = forward_offsets.into_iter();
        for pos in 0..half {
            let offset: usize = if extra_pos == Some(pos) {
                extra_offset.expect("paired with extra_block")
            } else {
                forward_iter.next().expect("matches query_iprfs length") as usize
            };
            if offset >= self.params.block_size {
                return None;
            }
            offsets.push(offset as u32);
        }

        Some(Query { mask, offsets })
    }

    /// Promotes the first available backup hint into a regular slot.
    ///
    /// Any backup can serve any queried block because backup subsets have
    /// exactly c/2 blocks. If the queried block is in the subset, we
    /// complement it (c/2 blocks become c - c/2 = c/2, still valid).
    /// If it's outside, we use the subset as-is. Either way we get a
    /// c/2-block subset that excludes the queried block, which is then
    /// added via `extra_index`.
    fn promote_backup(
        &mut self,
        queried_index: usize,
        queried_entry: &[u8],
    ) -> Result<usize, ProtocolError> {
        if self.next_backup_hint >= self.backup_available.len() {
            return Err(ProtocolError::NoBackupHintsLeft);
        }
        let backup_idx: usize = self.next_backup_hint;
        debug_assert!(self.backup_available[backup_idx]);

        let queried_block = self.params.block_of(queried_index);
        let promoted_idx = self.params.num_regular_hints + backup_idx;
        let cutoff: u32 = self.hints.backup_cutoffs()[backup_idx];
        let queried_in_subset =
            self.subset_prf
                .contains(promoted_idx as u64, queried_block, cutoff);

        // The slot parity lives in the "in" parity of the backup hint.
        if queried_in_subset {
            let parity_out: Vec<u8> = self.hints.backup_parity_out(backup_idx).to_vec();
            self.hints
                .backup_parity_in_mut(backup_idx)
                .copy_from_slice(&parity_out);
        }
        xor_entry(self.hints.backup_parity_in_mut(backup_idx), queried_entry);
        self.consume_backup(backup_idx);

        self.slots[promoted_idx] = Some(HintSlot {
            cutoff,
            complemented: queried_in_subset,
            extra_index: Some(queried_index),
        });

        Ok(promoted_idx)
    }

    fn slot_block_in_subset(&self, hint_idx: usize, slot: &HintSlot, block: usize) -> bool {
        let in_original = self
            .subset_prf
            .contains(hint_idx as u64, block, slot.cutoff);
        in_original ^ slot.complemented
    }
}

/// Runs HintInit (Plinko Fig. 7) over an in-memory database on all available cores.
///
/// `database` holds the `num_entries` records of version `database_version`
/// one after another. The hint set gets this version and the hash of
/// `database` ([`DatabaseSnapshot`]). `plinko_hints` computes the same hint
/// set from a database file with the same bytes; both call
/// [`crate::hint_init`]. [`compute_hints_with_threads`] takes the number of
/// worker threads.
///
/// # Errors
/// Returns [`ProtocolError::InvalidParams`] if `database` does not have
/// `num_entries * record_size` bytes.
pub fn compute_hints(
    params: &ProtocolParams,
    master_seed: &[u8; 32],
    database_version: u64,
    database: &[u8],
) -> Result<HintSet, ProtocolError> {
    compute_hints_with_threads(
        params,
        master_seed,
        database_version,
        database,
        hint_init::available_threads(),
    )
}

/// Runs HintInit as [`compute_hints`] does, on `threads` worker threads.
///
/// A `threads` value of 0 counts as 1. The hint set does not depend on the
/// thread count.
///
/// # Errors
/// Returns [`ProtocolError::InvalidParams`] if `database` does not have
/// `num_entries * record_size` bytes.
pub fn compute_hints_with_threads(
    params: &ProtocolParams,
    master_seed: &[u8; 32],
    database_version: u64,
    database: &[u8],
    threads: usize,
) -> Result<HintSet, ProtocolError> {
    check_database_len(params, database)?;

    let iprfs: Vec<Iprf> = block_iprfs(params, master_seed);
    let subset_prf: SubsetPrf = SubsetPrf::new(master_seed);
    let (regular_cutoffs, backup_cutoffs) = hint_init::hint_cutoffs(
        &subset_prf,
        params.num_blocks,
        params.num_regular_hints,
        params.num_backup_hints,
        threads,
    );

    let mut hint_set: HintSet = HintSet::new(
        params.hint_geometry(),
        hints::seed_check(master_seed),
        DatabaseSnapshot::of(database_version, database),
        regular_cutoffs,
        backup_cutoffs,
    )
    .map_err(|_| ProtocolError::InvalidParams("the hint parities do not fit in memory"))?;

    hint_init::fill_parities(
        &mut hint_set,
        database,
        &iprfs,
        &subset_prf,
        threads,
        |_| {},
    );

    Ok(hint_set)
}

/// Checks the parameters, then the length of the database.
fn check_database_len(params: &ProtocolParams, database: &[u8]) -> Result<(), ProtocolError> {
    params.check_derived()?;
    if database.len() != params.num_entries * params.record_size {
        return Err(ProtocolError::InvalidParams(
            "database length must equal num_entries * record_size",
        ));
    }
    Ok(())
}

fn check_index(params: &ProtocolParams, index: usize) -> Result<(), ProtocolError> {
    if index >= params.num_entries {
        return Err(ProtocolError::InvalidIndex {
            index,
            num_entries: params.num_entries,
        });
    }
    Ok(())
}

fn check_record_len(params: &ProtocolParams, record: &[u8]) -> Result<(), ProtocolError> {
    if record.len() != params.record_size {
        return Err(ProtocolError::RecordSizeMismatch {
            expected: params.record_size,
            got: record.len(),
        });
    }
    Ok(())
}

fn block_iprfs(params: &ProtocolParams, master_seed: &[u8; 32]) -> Vec<Iprf> {
    derive_block_keys(master_seed, params.num_blocks)
        .iter()
        .map(|key| {
            Iprf::with_cached_levels(
                *key,
                params.num_total_hints as u64,
                params.block_size as u64,
                CLIENT_CACHED_PMNS_LEVELS,
            )
        })
        .collect()
}

/// Derives one 128-bit iPRF key per block from the master seed.
///
/// Each key is the first 16 bytes of `SHA-256(master_seed || "block_key" || block_idx_le)`.
pub fn derive_block_keys(master_seed: &[u8; 32], c: usize) -> Vec<PrfKey128> {
    let mut keys = Vec::with_capacity(c);
    for block_idx in 0..c {
        let mut hasher = Sha256::new();
        hasher.update(master_seed);
        hasher.update(b"block_key");
        hasher.update((block_idx as u64).to_le_bytes());
        let hash = hasher.finalize();
        let mut key = [0u8; 16];
        key.copy_from_slice(&hash[0..16]);
        keys.push(key);
    }
    keys
}

#[inline]
fn set_bit(mask: &mut [u8], bit: usize) {
    mask[bit / 8] |= 1u8 << (bit % 8);
}

#[inline]
fn bit_is_set(mask: &[u8], bit: usize) -> bool {
    ((mask[bit / 8] >> (bit % 8)) & 1) == 1
}

/// XORs `src` into `dst`. Both have the record size of the database.
#[inline]
fn xor_entry(dst: &mut [u8], src: &[u8]) {
    debug_assert_eq!(dst.len(), src.len());
    for (d, s) in dst.iter_mut().zip(src) {
        *d ^= s;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NUM_ENTRIES: usize = 64;
    const RECORD_SIZE: usize = 24;

    fn client_with_taken(cached: &[usize], in_flight: &[usize]) -> Client {
        let params: ProtocolParams =
            ProtocolParams::new(NUM_ENTRIES, RECORD_SIZE, Some(8), 2, None).expect("params");
        let database: Vec<u8> = vec![0u8; NUM_ENTRIES * RECORD_SIZE];
        let mut client: Client =
            Client::offline_init(params, [0x01u8; 32], [0x02u8; 32], 0, &database)
                .expect("offline init");
        for &index in cached {
            client.cache.insert(
                index,
                CacheEntry {
                    value: vec![0u8; RECORD_SIZE],
                    hint_idx: 0,
                },
            );
        }
        for &index in in_flight {
            client.in_flight.insert(
                index,
                InFlight {
                    hint_idx: 0,
                    requested_index: index,
                    real_is_first: false,
                },
            );
        }
        client
    }

    /// Draws 1000 decoys per free index and checks every count against the
    /// uniform mean with a bound of 6 standard deviations (about 31 each).
    fn assert_decoys_are_uniform(cached: &[usize], in_flight: &[usize]) {
        let mut client: Client = client_with_taken(cached, in_flight);
        let free: Vec<usize> = (0..NUM_ENTRIES)
            .filter(|index| !cached.contains(index) && !in_flight.contains(index))
            .collect();

        let mut counts: Vec<u32> = vec![0; NUM_ENTRIES];
        for _ in 0..(1000 * free.len()) {
            counts[client.random_decoy().expect("decoy")] += 1;
        }

        for (index, &count) in counts.iter().enumerate() {
            if free.contains(&index) {
                assert!(
                    (810..=1190).contains(&count),
                    "index {index}: count {count} is not uniform"
                );
            } else {
                assert_eq!(count, 0, "index {index} is cached or in flight");
            }
        }
    }

    #[test]
    fn decoy_is_uniform_among_indices_not_cached_and_not_in_flight() {
        let few_cached: Vec<usize> = (0..10).collect();
        assert_decoys_are_uniform(&few_cached, &[20]);

        // More than half of the indices are taken: the enumeration branch.
        let many_cached: Vec<usize> = (0..40).collect();
        assert_decoys_are_uniform(&many_cached, &[41, 63]);
    }

    #[test]
    fn decoy_needs_an_index_that_is_not_cached_and_not_in_flight() {
        let all_but_two: Vec<usize> = (0..NUM_ENTRIES - 2).collect();

        let mut one_free: Client = client_with_taken(&all_but_two, &[NUM_ENTRIES - 1]);
        for _ in 0..20 {
            assert_eq!(one_free.random_decoy(), Ok(NUM_ENTRIES - 2));
        }

        let mut none_free: Client =
            client_with_taken(&all_but_two, &[NUM_ENTRIES - 2, NUM_ENTRIES - 1]);
        assert_eq!(
            none_free.random_decoy(),
            Err(ProtocolError::NoDecoyIndexAvailable)
        );
    }
}
