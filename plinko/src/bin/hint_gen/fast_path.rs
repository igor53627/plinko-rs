//! Non-constant-time (fast) hint processing path.
//!
//! Iterates over every database entry, resolves IPRF preimages to determine
//! which hints each entry contributes to, and XORs the entry into the
//! appropriate parity accumulators. This path leaks the access pattern
//! through data-dependent branches and is suitable only when timing
//! side-channels are not a concern.
//!
//! Worker threads take database blocks from a shared counter. The parity
//! accumulators are atomic words that the workers update with XOR, so the
//! result does not depend on the thread count or on the schedule.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use plinko::hints::HintSet;
use plinko::iprf::{Iprf, IprfTable};
use plinko::subsets::{SubsetPrf, INVALID_CUTOFF};

use crate::hint_gen::types::record_at;

/// Entries between two progress reports of one worker.
const PROGRESS_STEP: usize = 1024;

/// Immutable inputs for non-constant-time hint processing.
pub struct FastProcessInput<'a> {
    pub db_bytes: &'a [u8],
    /// Bytes of one database record; equal to the record size of the hint set.
    pub record_size: usize,
    pub n_entries: usize,
    pub n_effective: usize,
    pub w: usize,
    pub num_regular: usize,
    pub num_backup: usize,
    pub block_iprfs: &'a [Iprf],
    pub subset_prf: &'a SubsetPrf,
    /// Number of worker threads; at least 1. More threads than blocks are not started.
    pub threads: usize,
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

/// 64-bit words that hold one record.
fn words_per_record(record_size: usize) -> usize {
    record_size.div_ceil(8)
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

    /// Writes parity `index` to `parity`, which has the record size.
    fn copy_to(&self, index: usize, parity: &mut [u8]) {
        debug_assert_eq!(words_per_record(parity.len()), self.words_per_parity);
        for (bytes, word) in parity.chunks_mut(8).zip(self.slot(index)) {
            let word_bytes: [u8; 8] = word.load(Ordering::Relaxed).to_le_bytes();
            bytes.copy_from_slice(&word_bytes[..bytes.len()]);
        }
    }
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

/// Accumulators of one run and the cutoffs of the hints.
struct Accumulators<'a> {
    regular_cutoffs: &'a [u32],
    backup_cutoffs: &'a [u32],
    regular: SharedParities,
    backup_in: SharedParities,
    backup_out: SharedParities,
}

/// Processes all database entries through the fast (non-CT) path.
///
/// Computes the PMNS bin boundaries of each block once (`Iprf::table`).
/// For each entry at logical index `i`, computes the IPRF inverse to find
/// the set of hint indices, then XORs the record into the matching regular
/// or backup hint parity accumulator based on block membership. The cutoff
/// of each hint defines the membership; an invalid hint gets no record.
///
/// `hints` holds the cutoffs; the function overwrites its parities.
/// `input.threads` workers process the blocks. The parities are
/// equal for every thread count. The accumulators take the record size,
/// rounded up to 8 bytes, per regular hint and two times that per backup hint
/// in addition to the hints, until the function returns.
///
/// `progress_callback` receives the number of entries that are done; calls
/// from different workers can arrive out of order.
pub fn process_entries_fast(
    input: FastProcessInput<'_>,
    hints: &mut HintSet,
    progress_callback: impl Fn(usize) + Sync,
) {
    assert_eq!(input.record_size, hints.record_size());
    debug_assert_eq!(input.num_regular, hints.num_regular());
    debug_assert_eq!(input.num_backup, hints.num_backup());
    debug_assert_eq!(input.n_effective, input.block_iprfs.len() * input.w);

    let accumulators = Accumulators {
        regular_cutoffs: hints.regular_cutoffs(),
        backup_cutoffs: hints.backup_cutoffs(),
        regular: SharedParities::new(input.num_regular, input.record_size),
        backup_in: SharedParities::new(input.num_backup, input.record_size),
        backup_out: SharedParities::new(input.num_backup, input.record_size),
    };
    let num_blocks: usize = input.block_iprfs.len();
    let workers: usize = input.threads.clamp(1, num_blocks.max(1));
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
                process_block(&input, block, &accumulators, &report);
            });
        }
    });

    let Accumulators {
        regular,
        backup_in,
        backup_out,
        ..
    } = accumulators;
    for j in 0..input.num_regular {
        regular.copy_to(j, hints.regular_parity_mut(j));
    }
    for b in 0..input.num_backup {
        backup_in.copy_to(b, hints.backup_parity_in_mut(b));
        backup_out.copy_to(b, hints.backup_parity_out_mut(b));
    }
}

/// XORs the entries of one database block into the accumulators.
fn process_block(
    input: &FastProcessInput<'_>,
    block: usize,
    accumulators: &Accumulators<'_>,
    report: &impl Fn(usize),
) {
    let table: IprfTable<'_> = input.block_iprfs[block].table();
    let mut entry: Vec<u64> = vec![0u64; words_per_record(input.record_size)];
    let mut unreported: usize = 0;

    for offset in 0..input.w {
        let i: usize = block * input.w + offset;
        // A padding record is zero and changes no parity.
        let preimages: Vec<u64> = if i < input.n_entries {
            fill_entry_words(record_at(input.db_bytes, input.record_size, i), &mut entry);
            table.inverse(offset as u64)
        } else {
            Vec::new()
        };

        for hint_id in preimages {
            let j = hint_id as usize;
            if j < input.num_regular {
                let cutoff: u32 = accumulators.regular_cutoffs[j];
                if input.subset_prf.contains(hint_id, block, cutoff) {
                    accumulators.regular.xor(j, &entry);
                }
            } else {
                let backup_idx = j - input.num_regular;
                if backup_idx < input.num_backup {
                    let cutoff: u32 = accumulators.backup_cutoffs[backup_idx];
                    if cutoff == INVALID_CUTOFF {
                        continue;
                    }
                    if input.subset_prf.contains(hint_id, block, cutoff) {
                        accumulators.backup_in.xor(backup_idx, &entry);
                    } else {
                        accumulators.backup_out.xor(backup_idx, &entry);
                    }
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
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sizes below a word, at a word boundary, and with a partial last word.
    #[test]
    fn shared_parity_keeps_the_byte_order_of_the_entry() {
        for record_size in [1usize, 7, 8, 9, 32, 40, 45] {
            let entry: Vec<u8> = (0..record_size).map(|i| i as u8 + 1).collect();
            let mut words: Vec<u64> = vec![u64::MAX; words_per_record(record_size)];
            fill_entry_words(&entry, &mut words);

            let parities = SharedParities::new(3, record_size);
            let parity_of = |index: usize| -> Vec<u8> {
                let mut parity: Vec<u8> = vec![0xFFu8; record_size];
                parities.copy_to(index, &mut parity);
                parity
            };
            let zero: Vec<u8> = vec![0u8; record_size];

            parities.xor(1, &words);
            assert_eq!(parity_of(0), zero, "record size {record_size}");
            assert_eq!(parity_of(1), entry, "record size {record_size}");
            assert_eq!(parity_of(2), zero, "record size {record_size}");

            parities.xor(1, &words);
            assert_eq!(parity_of(1), zero, "record size {record_size}");
        }
    }
}
