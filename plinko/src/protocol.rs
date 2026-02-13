//! Lightweight end-to-end Plinko protocol model.
//!
//! This module provides a practical, in-memory protocol flow on top of the
//! existing iPRF primitive:
//! - offline hint initialization (regular + backup)
//! - online query construction
//! - server answer computation
//! - client reconstruction + backup promotion
//! - hint updates via XOR deltas
//!
//! Entries are fixed-size 32-byte words.

use crate::iprf::{Iprf, PrfKey128};
use rand::seq::index::sample;
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::error::Error;
use std::fmt::{Display, Formatter};

const SEED_LABEL_REGULAR: &[u8] = b"plinko_regular_subset";
const SEED_LABEL_BACKUP: &[u8] = b"plinko_backup_subset";

/// Fixed entry size used by the protocol model.
pub const ENTRY_SIZE: usize = 32;
/// Database entry type.
pub type Entry = [u8; ENTRY_SIZE];

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProtocolError {
    InvalidParams(&'static str),
    InvalidIndex { index: usize, num_entries: usize },
    NoDecoyIndexAvailable,
    NoHintForIndex { index: usize },
    NoBackupHintsLeft,
    QueryShapeMismatch { expected_offsets: usize, got: usize },
    OffsetOutOfRange { block: usize, offset: usize },
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
            Self::QueryShapeMismatch {
                expected_offsets,
                got,
            } => {
                write!(
                    f,
                    "query offset length mismatch: expected {expected_offsets}, got {got}"
                )
            }
            Self::OffsetOutOfRange { block, offset } => {
                write!(f, "offset {offset} out of range for block {block}")
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

impl ProtocolParams {
    /// Create parameters with defaults similar to the Python runnable spec:
    /// - `block_size`: `next_power_of_two(ceil(sqrt(num_entries)))` if omitted
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

        let block_size = match block_size {
            Some(v) => v,
            None => {
                let raw = (num_entries as f64).sqrt().ceil() as usize;
                raw.max(1).next_power_of_two()
            }
        };
        if block_size == 0 {
            return Err(ProtocolError::InvalidParams("block_size must be > 0"));
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

/// Query format using shared offsets optimization:
/// one bitmask + `num_blocks/2` offsets.
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

        let mut parity_0 = [0u8; ENTRY_SIZE];
        let mut parity_1 = [0u8; ENTRY_SIZE];
        let mut idx_0 = 0usize;
        let mut idx_1 = 0usize;

        for block in 0..self.params.num_blocks {
            if bit_is_set(&query.mask, block) {
                let offset = query.offsets[idx_0] as usize;
                if offset >= self.params.block_size {
                    return Err(ProtocolError::OffsetOutOfRange { block, offset });
                }
                xor_entry(
                    &mut parity_0,
                    &self.database[block * self.params.block_size + offset],
                );
                idx_0 += 1;
            } else {
                let offset = query.offsets[idx_1] as usize;
                if offset >= self.params.block_size {
                    return Err(ProtocolError::OffsetOutOfRange { block, offset });
                }
                xor_entry(
                    &mut parity_1,
                    &self.database[block * self.params.block_size + offset],
                );
                idx_1 += 1;
            }
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
    subset_blocks: Vec<usize>,
    parity: Entry,
    extra_index: Option<usize>,
}

#[derive(Clone)]
struct BackupHint {
    subset_blocks: Vec<usize>,
    parity_in: Entry,
    parity_out: Entry,
    available: bool,
}

#[derive(Clone, Copy)]
struct CacheEntry {
    value: Entry,
    hint_idx: usize,
}

pub struct PreparedQuery {
    pub query: Query,
    requested_index: usize,
    queried_index: usize,
    real_is_first: bool,
    consumed_parity: Entry,
}

/// Client for the in-memory protocol model.
pub struct Client {
    params: ProtocolParams,
    iprfs: Vec<Iprf>,
    slots: Vec<Option<HintSlot>>,
    backups: Vec<BackupHint>,
    cache: HashMap<usize, CacheEntry>,
    rng: ChaCha20Rng,
    next_decoy_index: usize,
}

impl Client {
    /// Run offline HintInit from a flat database.
    pub fn offline_init(
        params: ProtocolParams,
        master_seed: [u8; 32],
        query_rng_seed: [u8; 32],
        entries: &[Entry],
    ) -> Result<Self, ProtocolError> {
        if entries.len() != params.num_entries {
            return Err(ProtocolError::InvalidParams(
                "entries length must equal num_entries",
            ));
        }

        let block_keys = derive_block_keys(&master_seed, params.num_blocks);
        let iprfs: Vec<Iprf> = block_keys
            .iter()
            .map(|key| {
                Iprf::new(
                    *key,
                    params.num_total_hints as u64,
                    params.block_size as u64,
                )
            })
            .collect();

        let mut slots: Vec<Option<HintSlot>> = vec![None; params.num_total_hints];
        for j in 0..params.num_regular_hints {
            let seed = derive_subset_seed(&master_seed, SEED_LABEL_REGULAR, j as u64);
            let subset_blocks = compute_regular_blocks(&seed, params.num_blocks);
            slots[j] = Some(HintSlot {
                subset_blocks,
                parity: [0u8; ENTRY_SIZE],
                extra_index: None,
            });
        }

        let mut backups = Vec::with_capacity(params.num_backup_hints);
        for j in 0..params.num_backup_hints {
            let seed = derive_subset_seed(&master_seed, SEED_LABEL_BACKUP, j as u64);
            let subset_blocks = compute_backup_blocks(&seed, params.num_blocks);
            backups.push(BackupHint {
                subset_blocks,
                parity_in: [0u8; ENTRY_SIZE],
                parity_out: [0u8; ENTRY_SIZE],
                available: true,
            });
        }

        let mut padded = entries.to_vec();
        padded.resize(params.n_effective_entries(), [0u8; ENTRY_SIZE]);

        for (i, entry) in padded.iter().copied().enumerate() {
            let block = i / params.block_size;
            let offset = i % params.block_size;
            let hint_indices = iprfs[block].inverse(offset as u64);

            for hint_idx in hint_indices {
                let j = hint_idx as usize;
                if j < params.num_regular_hints {
                    if let Some(slot) = slots[j].as_mut() {
                        if block_in_subset(&slot.subset_blocks, block) {
                            xor_entry(&mut slot.parity, &entry);
                        }
                    }
                } else if j < params.num_total_hints {
                    let backup_idx = j - params.num_regular_hints;
                    if block_in_subset(&backups[backup_idx].subset_blocks, block) {
                        xor_entry(&mut backups[backup_idx].parity_in, &entry);
                    } else {
                        xor_entry(&mut backups[backup_idx].parity_out, &entry);
                    }
                }
            }
        }

        Ok(Self {
            params,
            iprfs,
            slots,
            backups,
            cache: HashMap::new(),
            rng: ChaCha20Rng::from_seed(query_rng_seed),
            next_decoy_index: 0,
        })
    }

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

        let queried_index = if self.cache.contains_key(&requested_index) {
            self.next_decoy()?
        } else {
            requested_index
        };

        let block = self.params.block_of(queried_index);
        let offset = self.params.offset_in_block(queried_index);
        let candidates = self.iprfs[block].inverse(offset as u64);

        for candidate in candidates {
            let hint_idx = candidate as usize;
            if hint_idx >= self.params.num_total_hints {
                continue;
            }

            let Some(slot) = self.slots[hint_idx].as_ref() else {
                continue;
            };

            if !block_in_subset(&slot.subset_blocks, block)
                && slot.extra_index != Some(queried_index)
            {
                continue;
            }

            let real_is_first = (self.rng.next_u32() & 1) == 0;
            let Some(query) = self.try_build_query(hint_idx, queried_index, slot, real_is_first)
            else {
                continue;
            };

            let consumed = self.slots[hint_idx]
                .take()
                .expect("slot existence checked above");
            return Ok(PreparedQuery {
                query,
                requested_index,
                queried_index,
                real_is_first,
                consumed_parity: consumed.parity,
            });
        }

        Err(ProtocolError::NoHintForIndex {
            index: queried_index,
        })
    }

    pub fn reconstruct_and_replenish(
        &mut self,
        prepared: PreparedQuery,
        response: Response,
    ) -> Result<Entry, ProtocolError> {
        let mut queried_entry = if prepared.real_is_first {
            response.parity_0
        } else {
            response.parity_1
        };
        xor_entry(&mut queried_entry, &prepared.consumed_parity);

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
                    if let Some(slot) = self.slots[j].as_mut() {
                        if block_in_subset(&slot.subset_blocks, block) {
                            xor_entry(&mut slot.parity, &update.delta);
                        }
                    }
                } else {
                    let backup_idx = j - self.params.num_regular_hints;

                    if let Some(slot) = self.slots[j].as_mut() {
                        if block_in_subset(&slot.subset_blocks, block)
                            || slot.extra_index == Some(update.index)
                        {
                            xor_entry(&mut slot.parity, &update.delta);
                        }
                    } else if self.backups[backup_idx].available {
                        if block_in_subset(&self.backups[backup_idx].subset_blocks, block) {
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

    fn next_decoy(&mut self) -> Result<usize, ProtocolError> {
        while self.next_decoy_index < self.params.num_entries {
            let idx = self.next_decoy_index;
            self.next_decoy_index += 1;
            if !self.cache.contains_key(&idx) {
                return Ok(idx);
            }
        }
        Err(ProtocolError::NoDecoyIndexAvailable)
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

        for block in 0..self.params.num_blocks {
            let is_real = if block == queried_block {
                false
            } else if block_in_subset(&slot.subset_blocks, block) {
                true
            } else if extra_block == Some(block) {
                true
            } else {
                false
            };

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

    fn promote_backup(
        &mut self,
        queried_index: usize,
        queried_entry: Entry,
    ) -> Result<usize, ProtocolError> {
        let Some(backup_idx) = self.backups.iter().position(|b| b.available) else {
            return Err(ProtocolError::NoBackupHintsLeft);
        };

        let queried_block = self.params.block_of(queried_index);
        let backup = &mut self.backups[backup_idx];
        let queried_in_subset = block_in_subset(&backup.subset_blocks, queried_block);

        let subset_blocks = if queried_in_subset {
            complement_subset(self.params.num_blocks, &backup.subset_blocks)
        } else {
            backup.subset_blocks.clone()
        };

        let mut parity = if queried_in_subset {
            backup.parity_out
        } else {
            backup.parity_in
        };
        xor_entry(&mut parity, &queried_entry);

        backup.available = false;

        let promoted_idx = self.params.num_regular_hints + backup_idx;
        self.slots[promoted_idx] = Some(HintSlot {
            subset_blocks,
            parity,
            extra_index: Some(queried_index),
        });

        Ok(promoted_idx)
    }
}

fn derive_subset_seed(master_seed: &[u8; 32], label: &[u8], idx: u64) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(master_seed);
    hasher.update(label);
    hasher.update(idx.to_le_bytes());
    let hash = hasher.finalize();
    let mut seed = [0u8; 32];
    seed.copy_from_slice(&hash[0..32]);
    seed
}

fn derive_block_keys(master_seed: &[u8; 32], c: usize) -> Vec<PrfKey128> {
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

fn compute_regular_blocks(seed: &[u8; 32], c: usize) -> Vec<usize> {
    let mut rng = ChaCha20Rng::from_seed(*seed);
    let mut blocks = sample(&mut rng, c, c / 2 + 1).into_vec();
    blocks.sort_unstable();
    blocks
}

fn compute_backup_blocks(seed: &[u8; 32], c: usize) -> Vec<usize> {
    let mut rng = ChaCha20Rng::from_seed(*seed);
    let mut blocks = sample(&mut rng, c, c / 2).into_vec();
    blocks.sort_unstable();
    blocks
}

#[inline]
fn block_in_subset(blocks: &[usize], block: usize) -> bool {
    blocks.binary_search(&block).is_ok()
}

fn complement_subset(total_blocks: usize, subset: &[usize]) -> Vec<usize> {
    let mut in_subset = vec![false; total_blocks];
    for &block in subset {
        in_subset[block] = true;
    }
    let mut complement = Vec::with_capacity(total_blocks - subset.len());
    for (block, present) in in_subset.into_iter().enumerate() {
        if !present {
            complement.push(block);
        }
    }
    complement
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
