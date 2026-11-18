//! HintInit (Plinko Fig. 7): the cutoffs and the parities of a hint set.
//!
//! HintInit has two phases, and both run on worker threads:
//!
//! 1. [`hint_cutoffs`] computes the cutoff of the block subset of each hint
//!    (see [`crate::subsets`]).
//! 2. [`fill_parities`] reads the database once. For the record at offset
//!    `beta` of block `alpha` it computes the iPRF inverse of `beta` with the
//!    key of block `alpha`, and XORs the record into the parities of the hints
//!    that the inverse gives.
//!
//! `plinko_hints` and [`crate::protocol::compute_hints`] call these functions,
//! so a hint file and a hint set of the protocol client come from one
//! implementation. The result does not depend on the thread count.
//!
//! With the `tee` feature, `fill_parities_ct` is the experimental
//! constant-time form of the second phase. It is not part of the trust model
//! (`docs/trust_model.md`).

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use crate::hints::{HintGeometry, HintSet};
use crate::iprf::{Iprf, IprfTable};
use crate::subsets::{backup_subset_size, regular_subset_size, SubsetPrf, INVALID_CUTOFF};

#[cfg(feature = "tee")]
mod tee;
#[cfg(test)]
mod tests;

#[cfg(feature = "tee")]
pub use tee::fill_parities_ct;

/// Entries between two progress reports of one worker.
const PROGRESS_STEP: usize = 1024;

/// Return type of [`hint_cutoffs`]: `(regular_cutoffs, backup_cutoffs)`.
pub type HintCutoffs = (Vec<u32>, Vec<u32>);

/// Returns the number of available cores, or 1 if the system does not tell.
pub fn available_threads() -> usize {
    std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(1)
}

/// Computes the cutoff of the block subset of each hint on `threads` workers.
///
/// Regular hint `j` has hint ID `j` and a subset of `c/2 + 1` of the
/// `num_blocks` blocks. Backup hint `b` has hint ID `num_regular + b` and a
/// subset of `c/2` blocks. A `threads` value of 0 counts as 1. The cutoffs do
/// not depend on the thread count.
///
/// # Examples
///
/// ```
/// use plinko::hint_init::hint_cutoffs;
/// use plinko::subsets::SubsetPrf;
///
/// let prf: SubsetPrf = SubsetPrf::new(&[7u8; 32]);
/// let (regular, backup) = hint_cutoffs(&prf, 10, 6, 3, 4);
/// assert_eq!((regular.len(), backup.len()), (6, 3));
/// assert_eq!((regular, backup), hint_cutoffs(&prf, 10, 6, 3, 1));
/// ```
pub fn hint_cutoffs(
    subset_prf: &SubsetPrf,
    num_blocks: usize,
    num_regular: usize,
    num_backup: usize,
    threads: usize,
) -> HintCutoffs {
    let mut regular: Vec<u32> = vec![INVALID_CUTOFF; num_regular];
    let mut backup: Vec<u32> = vec![INVALID_CUTOFF; num_backup];
    cutoffs_into(
        subset_prf,
        num_blocks,
        regular_subset_size(num_blocks),
        0,
        &mut regular,
        threads,
    );
    cutoffs_into(
        subset_prf,
        num_blocks,
        backup_subset_size(num_blocks),
        num_regular as u64,
        &mut backup,
        threads,
    );
    (regular, backup)
}

/// Writes the cutoffs of the hints `first_hint_id..` to `cutoffs`.
///
/// Each worker takes one contiguous part of `cutoffs`; the work for a hint
/// does not depend on the hint.
fn cutoffs_into(
    subset_prf: &SubsetPrf,
    num_blocks: usize,
    subset_size: usize,
    first_hint_id: u64,
    cutoffs: &mut [u32],
    threads: usize,
) {
    if cutoffs.is_empty() {
        return;
    }
    let workers: usize = threads.clamp(1, cutoffs.len());
    let part_len: usize = cutoffs.len().div_ceil(workers);
    std::thread::scope(|scope| {
        for (part_idx, part) in cutoffs.chunks_mut(part_len).enumerate() {
            scope.spawn(move || {
                let mut scratch: Vec<u32> = Vec::with_capacity(num_blocks);
                let part_first_id: u64 = first_hint_id + (part_idx * part_len) as u64;
                for (k, cutoff) in part.iter_mut().enumerate() {
                    *cutoff = subset_prf.cutoff(
                        part_first_id + k as u64,
                        num_blocks,
                        subset_size,
                        &mut scratch,
                    );
                }
            });
        }
    });
}

/// Geometry of one run, taken from the hint set.
struct Layout {
    record_size: usize,
    num_entries: usize,
    block_size: usize,
    num_blocks: usize,
    num_regular: usize,
    num_backup: usize,
}

impl Layout {
    /// # Panics
    /// Panics if `database` or `num_block_iprfs` does not fit the geometry of `hints`.
    fn new(hints: &HintSet, database: &[u8], num_block_iprfs: usize) -> Self {
        let geometry: HintGeometry = hints.geometry();
        let record_size: usize = hints.record_size();
        assert_eq!(
            database.len() as u64,
            geometry.num_entries.saturating_mul(record_size as u64),
            "the database must hold num_entries records of the record size of the hint set"
        );
        assert_eq!(
            num_block_iprfs as u64, geometry.num_blocks,
            "one iPRF is necessary for each block"
        );
        Self {
            record_size,
            // The database is in memory, so these values fit in `usize`.
            num_entries: database.len() / record_size,
            block_size: geometry.block_size as usize,
            num_blocks: num_block_iprfs,
            num_regular: hints.num_regular(),
            num_backup: hints.num_backup(),
        }
    }
}

/// Returns record `index` of a database of `record_size`-byte records.
#[inline]
fn record_at(database: &[u8], record_size: usize, index: usize) -> &[u8] {
    &database[index * record_size..(index + 1) * record_size]
}

/// 64-bit words that hold one record.
fn words_per_record(record_size: usize) -> usize {
    record_size.div_ceil(8)
}

/// Writes `entry` to `words` in little-endian order; see [`SharedParities`].
fn fill_entry_words(entry: &[u8], words: &mut [u64]) {
    debug_assert_eq!(words.len(), words_per_record(entry.len()));
    for (word, bytes) in words.iter_mut().zip(entry.chunks(8)) {
        let mut word_bytes = [0u8; 8];
        word_bytes[..bytes.len()].copy_from_slice(bytes);
        *word = u64::from_le_bytes(word_bytes);
    }
}

/// Parities that several threads update without a lock.
///
/// XOR is commutative and associative, so the final value of a parity does
/// not depend on the order of the updates.
///
/// A parity takes `words_per_record(record_size)` 64-bit words. The bytes of a
/// record fill the words in little-endian order; the unused bytes of the last
/// word stay zero.
struct SharedParities {
    words: Vec<AtomicU64>,
    words_per_parity: usize,
}

impl SharedParities {
    fn new(count: usize, record_size: usize) -> Self {
        let words_per_parity: usize = words_per_record(record_size);
        Self {
            words: (0..count * words_per_parity)
                .map(|_| AtomicU64::new(0))
                .collect(),
            words_per_parity,
        }
    }

    fn slot(&self, index: usize) -> &[AtomicU64] {
        &self.words[index * self.words_per_parity..(index + 1) * self.words_per_parity]
    }

    fn xor(&self, index: usize, entry: &[u64]) {
        debug_assert_eq!(entry.len(), self.words_per_parity);
        for (word, &value) in self.slot(index).iter().zip(entry) {
            word.fetch_xor(value, Ordering::Relaxed);
        }
    }

    /// XORs `entry` into parity `index` if `mask` is 1, and XORs zero if it is 0.
    ///
    /// The same words are read and written for both values of `mask`.
    #[cfg(feature = "tee")]
    fn xor_masked(&self, index: usize, entry: &[u64], mask: u64) {
        debug_assert_eq!(entry.len(), self.words_per_parity);
        debug_assert!(mask <= 1);
        let word_mask: u64 = mask.wrapping_neg();
        for (word, &value) in self.slot(index).iter().zip(entry) {
            word.fetch_xor(value & word_mask, Ordering::Relaxed);
        }
    }

    /// Writes parity `index` to `parity`, which has the record size.
    fn copy_to(&self, index: usize, parity: &mut [u8]) {
        debug_assert_eq!(words_per_record(parity.len()), self.words_per_parity);
        for (bytes, word) in parity.chunks_mut(8).zip(self.slot(index)) {
            let word_bytes: [u8; 8] = word.load(Ordering::Relaxed).to_le_bytes();
            bytes.copy_from_slice(&word_bytes[..bytes.len()]);
        }
    }
}

/// Accumulators of one run. They take the record size, rounded up to 8 bytes,
/// per regular hint and two times that per backup hint.
struct Accumulators {
    regular: SharedParities,
    backup_in: SharedParities,
    backup_out: SharedParities,
}

impl Accumulators {
    fn new(layout: &Layout) -> Self {
        Self {
            regular: SharedParities::new(layout.num_regular, layout.record_size),
            backup_in: SharedParities::new(layout.num_backup, layout.record_size),
            backup_out: SharedParities::new(layout.num_backup, layout.record_size),
        }
    }

    /// Overwrites the parities of `hints` with the accumulators.
    fn store(&self, hints: &mut HintSet) {
        for j in 0..hints.num_regular() {
            self.regular.copy_to(j, hints.regular_parity_mut(j));
        }
        for b in 0..hints.num_backup() {
            self.backup_in.copy_to(b, hints.backup_parity_in_mut(b));
            self.backup_out.copy_to(b, hints.backup_parity_out_mut(b));
        }
    }
}

/// Runs `process_block(block, report)` for each block on `threads` workers.
///
/// The workers take the blocks from a shared counter. `report(n)` tells that
/// `n` more entries are done; `progress_callback` then receives the total.
fn for_each_block<P, B>(num_blocks: usize, threads: usize, progress_callback: P, process_block: B)
where
    P: Fn(usize) + Sync,
    B: Fn(usize, &dyn Fn(usize)) + Sync,
{
    let workers: usize = threads.clamp(1, num_blocks.max(1));
    let next_block = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    let report = |entries: usize| {
        progress_callback(done.fetch_add(entries, Ordering::Relaxed) + entries);
    };

    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| loop {
                let block: usize = next_block.fetch_add(1, Ordering::Relaxed);
                if block >= num_blocks {
                    break;
                }
                process_block(block, &report);
            });
        }
    });
}

/// Computes the parities of `hints` from `database` (the streaming phase).
///
/// `hints` holds the geometry and the cutoffs; the function overwrites its
/// parities. `database` holds the `num_entries` records of the geometry one
/// after another. `block_iprfs[alpha]` is the iPRF of block `alpha`, with the
/// total hint count as its domain and the block size as its range.
/// `subset_prf` is the PRF that gave the cutoffs.
///
/// A record at an index of `num_entries` or above is a padding record. It is
/// zero and changes no parity, so the function does no iPRF inverse for it.
/// A hint with [`INVALID_CUTOFF`] gets no record.
///
/// `threads` workers process the blocks; 0 counts as 1, and more workers than
/// blocks are not started. The parities are equal for every thread count.
/// Until the function returns, the accumulators of the workers take the
/// record size, rounded up to 8 bytes, per regular hint and two times that
/// per backup hint, in addition to `hints`.
///
/// `progress_callback` receives the number of entries that are done; calls
/// from different workers can arrive out of order.
///
/// # Panics
/// Panics if `database` does not have `num_entries` records of the record
/// size of `hints`, or if `block_iprfs` does not have one iPRF for each block.
pub fn fill_parities(
    hints: &mut HintSet,
    database: &[u8],
    block_iprfs: &[Iprf],
    subset_prf: &SubsetPrf,
    threads: usize,
    progress_callback: impl Fn(usize) + Sync,
) {
    let layout: Layout = Layout::new(hints, database, block_iprfs.len());
    let accumulators: Accumulators = Accumulators::new(&layout);
    let regular_cutoffs: &[u32] = hints.regular_cutoffs();
    let backup_cutoffs: &[u32] = hints.backup_cutoffs();

    for_each_block(
        layout.num_blocks,
        threads,
        progress_callback,
        |block, report| {
            let table: IprfTable<'_> = block_iprfs[block].table();
            let mut entry: Vec<u64> = vec![0u64; words_per_record(layout.record_size)];
            let mut unreported: usize = 0;

            for offset in 0..layout.block_size {
                let i: usize = block * layout.block_size + offset;
                let preimages: Vec<u64> = if i < layout.num_entries {
                    fill_entry_words(record_at(database, layout.record_size, i), &mut entry);
                    table.inverse(offset as u64)
                } else {
                    Vec::new()
                };

                for hint_id in preimages {
                    let j = hint_id as usize;
                    if j < layout.num_regular {
                        if subset_prf.contains(hint_id, block, regular_cutoffs[j]) {
                            accumulators.regular.xor(j, &entry);
                        }
                    } else if j - layout.num_regular < layout.num_backup {
                        let b: usize = j - layout.num_regular;
                        let cutoff: u32 = backup_cutoffs[b];
                        if cutoff == INVALID_CUTOFF {
                            continue;
                        }
                        if subset_prf.contains(hint_id, block, cutoff) {
                            accumulators.backup_in.xor(b, &entry);
                        } else {
                            accumulators.backup_out.xor(b, &entry);
                        }
                    }
                }

                unreported += 1;
                if unreported == PROGRESS_STEP {
                    report(unreported);
                    unreported = 0;
                }
            }
            report(unreported);
        },
    );

    accumulators.store(hints);
}
