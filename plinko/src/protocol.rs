//! Lightweight end-to-end Plinko protocol model.
//!
//! This module provides a practical, in-memory protocol flow on top of the
//! existing iPRF primitive:
//! - offline hint initialization (regular + backup), or a hint set that
//!   `plinko_hints` wrote to a file (see [`crate::hints`])
//! - online query construction
//! - server answer computation
//! - client reconstruction + backup promotion
//! - hint updates via XOR deltas
//!
//! The rules of the online phase (decoy, query format, retry, query limit) and
//! the deviations from the paper are in `docs/online_protocol.md`.
//!
//! Simplified in-memory model: logical entries are fixed 32-byte words.
//! Production `database.bin` uses the v3 40-byte on-disk layout (`schema40`);
//! see `hint_gen` and [`docs/data_format.md`](../../docs/data_format.md).

use crate::hints::{self, HintGeometry, HintSet};
use crate::iprf::{Iprf, IprfTable, PrfKey128};
use crate::subsets::{backup_subset_size, regular_subset_size, SubsetPrf, INVALID_CUTOFF};
use rand::seq::SliceRandom;
use rand::{Rng, RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::error::Error;
use std::fmt::{Display, Formatter};

/// PMNS tree levels that the client keeps per block (see [`Iprf::with_cached_levels`]).
///
/// Eight levels take 1020 bytes per block: about 29 MB for the 27,988 blocks of mainnet.
pub const CLIENT_CACHED_PMNS_LEVELS: u32 = 8;

/// Fixed entry size used by the protocol model.
pub const ENTRY_SIZE: usize = 32;
/// Database entry type.
pub type Entry = [u8; ENTRY_SIZE];

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
    /// Every backup hint is promoted or reserved by a query in flight.
    /// The client must run HintInit again.
    NoBackupHintsLeft,
    /// A query for this index was sent and has no answer yet.
    QueryInFlight {
        index: usize,
    },
    /// The prepared query is not a query in flight of this client.
    ///
    /// The check compares the queried index and the hint slot. It does not
    /// identify the client instance that built the query.
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
        }
    }
}

impl Error for ProtocolError {}

/// Parameters for the in-memory Plinko protocol model.
#[derive(Clone, Debug)]
pub struct ProtocolParams {
    pub num_entries: usize,
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

impl ProtocolParams {
    /// Create parameters with defaults similar to the Python runnable spec:
    /// - `block_size`: [`default_block_size`] if omitted; must be a power of two,
    ///   because the iPRF range is the block size
    /// - `num_backup_hints`: `lambda * block_size` if omitted
    pub fn new(
        num_entries: usize,
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

        let block_size = block_size.unwrap_or_else(|| default_block_size(num_entries));
        if !block_size.is_power_of_two() {
            return Err(ProtocolError::InvalidParams(
                "block_size must be a power of two",
            ));
        }

        let mut num_blocks = num_entries.div_ceil(block_size);
        if num_blocks < 2 {
            num_blocks = 2;
        }
        if num_blocks % 2 != 0 {
            num_blocks += 1;
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
        let geometry: HintGeometry = hints.geometry;
        let num_entries: usize = usize::try_from(geometry.num_entries)
            .map_err(|_| ProtocolError::HintSetMismatch("num_entries is too large"))?;
        let block_size: usize = usize::try_from(geometry.block_size)
            .map_err(|_| ProtocolError::HintSetMismatch("block_size is too large"))?;
        if block_size == 0 {
            return Err(ProtocolError::HintSetMismatch("block_size is zero"));
        }

        let lambda: usize = hints.regular.len() / block_size;
        if lambda * block_size != hints.regular.len() {
            return Err(ProtocolError::HintSetMismatch(
                "the number of regular hints is not a multiple of block_size",
            ));
        }
        let params: Self = Self::new(
            num_entries,
            Some(block_size),
            lambda,
            Some(hints.backup.len()),
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
            block_size: self.block_size as u64,
            num_blocks: self.num_blocks as u64,
        }
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EntryUpdate {
    pub index: usize,
    pub delta: Entry,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Response {
    pub parity_0: Entry,
    pub parity_1: Entry,
}

/// Server for the in-memory protocol model.
pub struct Server {
    params: ProtocolParams,
    database: Vec<Entry>,
}

impl Server {
    pub fn new(params: ProtocolParams, entries: &[Entry]) -> Result<Self, ProtocolError> {
        if entries.len() != params.num_entries {
            return Err(ProtocolError::InvalidParams(
                "entries length must equal num_entries",
            ));
        }

        let mut database = entries.to_vec();
        database.resize(params.n_effective_entries(), [0u8; ENTRY_SIZE]);

        Ok(Self { params, database })
    }

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

        let mut parity_0 = [0u8; ENTRY_SIZE];
        let mut parity_1 = [0u8; ENTRY_SIZE];
        let mut idx_0 = 0usize;
        let mut idx_1 = 0usize;

        for block in 0..self.params.num_blocks {
            let (parity, rank): (&mut Entry, &mut usize) = if bit_is_set(&query.mask, block) {
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
                &self.database[block * self.params.block_size + offset],
            );
            *rank += 1;
        }

        if idx_0 != half || idx_1 != half {
            return Err(ProtocolError::QueryShapeMismatch {
                expected_offsets: half,
                got: query.offsets.len(),
            });
        }

        Ok(Response { parity_0, parity_1 })
    }

    pub fn apply_updates(
        &mut self,
        updates: &[(usize, Entry)],
    ) -> Result<Vec<EntryUpdate>, ProtocolError> {
        let mut deltas = Vec::with_capacity(updates.len());

        for (index, new_value) in updates.iter().copied() {
            if index >= self.params.num_entries {
                return Err(ProtocolError::InvalidIndex {
                    index,
                    num_entries: self.params.num_entries,
                });
            }
            let old = self.database[index];
            let mut delta = old;
            xor_entry(&mut delta, &new_value);
            self.database[index] = new_value;
            deltas.push(EntryUpdate { index, delta });
        }

        Ok(deltas)
    }
}

#[derive(Clone)]
struct HintSlot {
    /// Cutoff of the original (pre-complement) subset; see [`crate::subsets`].
    cutoff: u32,
    complemented: bool,
    parity: Entry,
    extra_index: Option<usize>,
}

#[derive(Clone)]
struct BackupHint {
    /// Cutoff of the `c/2`-block subset; see [`crate::subsets`].
    cutoff: u32,
    parity_in: Entry,
    parity_out: Entry,
    /// `false` after promotion, and for a hint with no valid cutoff.
    available: bool,
}

#[derive(Clone, Copy)]
struct CacheEntry {
    value: Entry,
    hint_idx: usize,
}

/// A query in flight: built for the server, with no answer yet.
///
/// Keep the value until the answer arrives. To retry, send `query` again
/// without a change. To stop, call [`Client::abandon_query`].
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
}

/// Client for the in-memory protocol model.
pub struct Client {
    params: ProtocolParams,
    iprfs: Vec<Iprf>,
    subset_prf: SubsetPrf,
    slots: Vec<Option<HintSlot>>,
    backups: Vec<BackupHint>,
    cache: HashMap<usize, CacheEntry>,
    rng: ChaCha20Rng,
    /// Queried index and hint slot of each query in flight.
    in_flight: HashMap<usize, usize>,
}

impl Client {
    /// Run offline HintInit from a flat database.
    ///
    /// Equal to [`compute_hints`] followed by [`Client::from_hints`].
    pub fn offline_init(
        params: ProtocolParams,
        master_seed: [u8; 32],
        query_rng_seed: [u8; 32],
        entries: &[Entry],
    ) -> Result<Self, ProtocolError> {
        let hints: HintSet = compute_hints(&params, &master_seed, entries)?;
        Self::from_hints(params, master_seed, query_rng_seed, hints)
    }

    /// Builds a client from a hint set, for example one that `plinko_hints` wrote to a file.
    ///
    /// Returns [`ProtocolError::HintSetMismatch`] if the geometry or the hint
    /// counts differ from `params`, or if the hints were made with another
    /// master seed. The function does not recompute cutoffs or parities.
    pub fn from_hints(
        params: ProtocolParams,
        master_seed: [u8; 32],
        query_rng_seed: [u8; 32],
        hints: HintSet,
    ) -> Result<Self, ProtocolError> {
        if hints.geometry != params.hint_geometry() {
            return Err(ProtocolError::HintSetMismatch("database geometry"));
        }
        if hints.regular.len() != params.num_regular_hints {
            return Err(ProtocolError::HintSetMismatch("number of regular hints"));
        }
        if hints.backup.len() != params.num_backup_hints {
            return Err(ProtocolError::HintSetMismatch("number of backup hints"));
        }
        if hints.seed_check != hints::seed_check(&master_seed) {
            return Err(ProtocolError::HintSetMismatch("master seed"));
        }

        let mut slots: Vec<Option<HintSlot>> = Vec::with_capacity(params.num_total_hints);
        slots.extend(hints.regular.iter().map(|hint| {
            (hint.cutoff != INVALID_CUTOFF).then_some(HintSlot {
                cutoff: hint.cutoff,
                complemented: false,
                parity: hint.parity,
                extra_index: None,
            })
        }));
        slots.resize(params.num_total_hints, None);

        let backups: Vec<BackupHint> = hints
            .backup
            .iter()
            .map(|hint| BackupHint {
                cutoff: hint.cutoff,
                parity_in: hint.parity_in,
                parity_out: hint.parity_out,
                available: hint.cutoff != INVALID_CUTOFF,
            })
            .collect();

        Ok(Self {
            iprfs: block_iprfs(&params, &master_seed),
            subset_prf: SubsetPrf::new(&master_seed),
            params,
            slots,
            backups,
            cache: HashMap::new(),
            rng: ChaCha20Rng::from_seed(query_rng_seed),
            in_flight: HashMap::new(),
        })
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
        if requested_index >= self.params.num_entries {
            return Err(ProtocolError::InvalidIndex {
                index: requested_index,
                num_entries: self.params.num_entries,
            });
        }
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

            let Some(slot) = self.slots[hint_idx].as_ref() else {
                continue;
            };

            if !self.slot_block_in_subset(hint_idx, slot, block)
                && slot.extra_index != Some(queried_index)
            {
                continue;
            }

            let real_is_first = (self.rng.next_u32() & 1) == 0;
            let Some(query) = self.try_build_query(hint_idx, queried_index, slot, real_is_first)
            else {
                continue;
            };

            self.in_flight.insert(queried_index, hint_idx);
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
    /// Returns [`ProtocolError::UnknownPreparedQuery`] if `prepared` is not a
    /// query in flight of this client.
    pub fn reconstruct_and_replenish(
        &mut self,
        prepared: PreparedQuery,
        response: Response,
    ) -> Result<Entry, ProtocolError> {
        let hint: HintSlot = self.take_in_flight(&prepared)?;
        let mut queried_entry = if prepared.real_is_first {
            response.parity_0
        } else {
            response.parity_1
        };
        xor_entry(&mut queried_entry, &hint.parity);

        let promoted_hint_idx = self.promote_backup(prepared.queried_index, queried_entry)?;
        self.cache.insert(
            prepared.queried_index,
            CacheEntry {
                value: queried_entry,
                hint_idx: promoted_hint_idx,
            },
        );

        if prepared.requested_index == prepared.queried_index {
            return Ok(queried_entry);
        }

        let Some(entry) = self.cache.get(&prepared.requested_index) else {
            return Err(ProtocolError::NoHintForIndex {
                index: prepared.requested_index,
            });
        };
        Ok(entry.value)
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

    /// Apply server updates in O(expected preimages) per updated entry.
    pub fn apply_updates(&mut self, updates: &[EntryUpdate]) -> Result<(), ProtocolError> {
        for update in updates {
            if update.index >= self.params.num_entries {
                return Err(ProtocolError::InvalidIndex {
                    index: update.index,
                    num_entries: self.params.num_entries,
                });
            }

            let block = self.params.block_of(update.index);
            let offset = self.params.offset_in_block(update.index);
            let hint_indices = self.iprfs[block].inverse(offset as u64);

            for hint_idx in hint_indices {
                let j = hint_idx as usize;
                if j >= self.params.num_total_hints {
                    continue;
                }

                if j < self.params.num_regular_hints {
                    let should_update = self.slots[j]
                        .as_ref()
                        .is_some_and(|slot| self.slot_block_in_subset(j, slot, block));
                    if should_update {
                        xor_entry(&mut self.slots[j].as_mut().unwrap().parity, &update.delta);
                    }
                } else {
                    let backup_idx = j - self.params.num_regular_hints;

                    let should_update = self.slots[j]
                        .as_ref()
                        .is_some_and(|slot| self.slot_block_in_subset(j, slot, block));
                    if should_update {
                        xor_entry(&mut self.slots[j].as_mut().unwrap().parity, &update.delta);
                    } else if self.backups[backup_idx].available {
                        let cutoff: u32 = self.backups[backup_idx].cutoff;
                        if self.subset_prf.contains(hint_idx, block, cutoff) {
                            xor_entry(&mut self.backups[backup_idx].parity_in, &update.delta);
                        } else {
                            xor_entry(&mut self.backups[backup_idx].parity_out, &update.delta);
                        }
                    }
                }
            }

            if let Some(cache) = self.cache.get_mut(&update.index) {
                xor_entry(&mut cache.value, &update.delta);
                if let Some(slot) = self.slots[cache.hint_idx].as_mut() {
                    xor_entry(&mut slot.parity, &update.delta);
                }
            }
        }

        Ok(())
    }

    pub fn remaining_backup_hints(&self) -> usize {
        self.backups.iter().filter(|b| b.available).count()
    }

    #[inline]
    pub fn num_cached_entries(&self) -> usize {
        self.cache.len()
    }

    /// Returns the number of queries that have no answer yet.
    #[inline]
    pub fn queries_in_flight(&self) -> usize {
        self.in_flight.len()
    }

    fn hint_is_reserved(&self, hint_idx: usize) -> bool {
        self.in_flight
            .values()
            .any(|&reserved| reserved == hint_idx)
    }

    fn take_in_flight(&mut self, prepared: &PreparedQuery) -> Result<HintSlot, ProtocolError> {
        if self.in_flight.get(&prepared.queried_index) != Some(&prepared.hint_idx) {
            return Err(ProtocolError::UnknownPreparedQuery);
        }
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
        let mut offsets = Vec::with_capacity(half);
        let mut mask = vec![0u8; self.params.num_blocks.div_ceil(8)];
        let mut real_count = 0usize;

        let extra_block = slot.extra_index.map(|idx| self.params.block_of(idx));
        let extra_offset = slot.extra_index.map(|idx| self.params.offset_in_block(idx));

        let mut select_values: Vec<u32> = Vec::with_capacity(self.params.num_blocks);
        self.subset_prf
            .select_values(hint_idx as u64, self.params.num_blocks, &mut select_values);

        for (block, &select_value) in select_values.iter().enumerate() {
            let in_subset = (select_value < slot.cutoff) ^ slot.complemented;
            let is_real = block != queried_block && (in_subset || extra_block == Some(block));

            if is_real {
                real_count += 1;
                let offset = if extra_block == Some(block) {
                    extra_offset.expect("paired with extra_block")
                } else {
                    self.iprfs[block].forward(hint_idx as u64) as usize
                };
                if offset >= self.params.block_size {
                    return None;
                }
                offsets.push(offset as u32);
            }

            if is_real == real_is_first {
                set_bit(&mut mask, block);
            }
        }

        if real_count != half {
            return None;
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
        queried_entry: Entry,
    ) -> Result<usize, ProtocolError> {
        let Some(backup_idx) = self.backups.iter().position(|b| b.available) else {
            return Err(ProtocolError::NoBackupHintsLeft);
        };

        let queried_block = self.params.block_of(queried_index);
        let promoted_idx = self.params.num_regular_hints + backup_idx;
        let backup = &mut self.backups[backup_idx];
        let queried_in_subset =
            self.subset_prf
                .contains(promoted_idx as u64, queried_block, backup.cutoff);

        let mut parity = if queried_in_subset {
            backup.parity_out
        } else {
            backup.parity_in
        };
        xor_entry(&mut parity, &queried_entry);

        let cutoff: u32 = backup.cutoff;
        backup.available = false;

        self.slots[promoted_idx] = Some(HintSlot {
            cutoff,
            complemented: queried_in_subset,
            parity,
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

/// Runs HintInit (Plinko Fig. 7) over an in-memory database.
///
/// `plinko_hints` computes the same hint set from a `database.bin` file, where
/// an entry is the first 32 bytes of a record.
pub fn compute_hints(
    params: &ProtocolParams,
    master_seed: &[u8; 32],
    entries: &[Entry],
) -> Result<HintSet, ProtocolError> {
    if entries.len() != params.num_entries {
        return Err(ProtocolError::InvalidParams(
            "entries length must equal num_entries",
        ));
    }

    let iprfs: Vec<Iprf> = block_iprfs(params, master_seed);
    let subset_prf: SubsetPrf = SubsetPrf::new(master_seed);
    let mut scratch: Vec<u32> = Vec::with_capacity(params.num_blocks);

    let mut regular: Vec<hints::RegularHint> = (0..params.num_regular_hints)
        .map(|j| hints::RegularHint {
            cutoff: subset_prf.cutoff(
                j as u64,
                params.num_blocks,
                regular_subset_size(params.num_blocks),
                &mut scratch,
            ),
            parity: [0u8; ENTRY_SIZE],
        })
        .collect();

    let mut backup: Vec<hints::BackupHint> = (0..params.num_backup_hints)
        .map(|b| hints::BackupHint {
            cutoff: subset_prf.cutoff(
                (params.num_regular_hints + b) as u64,
                params.num_blocks,
                backup_subset_size(params.num_blocks),
                &mut scratch,
            ),
            parity_in: [0u8; ENTRY_SIZE],
            parity_out: [0u8; ENTRY_SIZE],
        })
        .collect();

    for (block, iprf) in iprfs.iter().enumerate() {
        let table: IprfTable<'_> = iprf.table();
        for offset in 0..params.block_size {
            let i = block * params.block_size + offset;
            let entry: Entry = entries.get(i).copied().unwrap_or([0u8; ENTRY_SIZE]);

            for hint_id in table.inverse(offset as u64) {
                let j = hint_id as usize;
                if j < params.num_regular_hints {
                    let hint: &mut hints::RegularHint = &mut regular[j];
                    if subset_prf.contains(hint_id, block, hint.cutoff) {
                        xor_entry(&mut hint.parity, &entry);
                    }
                } else if j < params.num_total_hints {
                    let hint: &mut hints::BackupHint = &mut backup[j - params.num_regular_hints];
                    if hint.cutoff == INVALID_CUTOFF {
                        continue;
                    }
                    if subset_prf.contains(hint_id, block, hint.cutoff) {
                        xor_entry(&mut hint.parity_in, &entry);
                    } else {
                        xor_entry(&mut hint.parity_out, &entry);
                    }
                }
            }
        }
    }

    Ok(HintSet {
        geometry: params.hint_geometry(),
        seed_check: hints::seed_check(master_seed),
        regular,
        backup,
    })
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

#[inline]
fn xor_entry(dst: &mut Entry, src: &Entry) {
    for i in 0..ENTRY_SIZE {
        dst[i] ^= src[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NUM_ENTRIES: usize = 64;

    fn client_with_taken(cached: &[usize], in_flight: &[usize]) -> Client {
        let params: ProtocolParams =
            ProtocolParams::new(NUM_ENTRIES, Some(8), 2, None).expect("params");
        let entries: Vec<Entry> = vec![[0u8; ENTRY_SIZE]; NUM_ENTRIES];
        let mut client: Client = Client::offline_init(params, [0x01u8; 32], [0x02u8; 32], &entries)
            .expect("offline init");
        for &index in cached {
            client.cache.insert(
                index,
                CacheEntry {
                    value: [0u8; ENTRY_SIZE],
                    hint_idx: 0,
                },
            );
        }
        for &index in in_flight {
            client.in_flight.insert(index, 0);
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
