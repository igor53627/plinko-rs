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

use plinko::iprf::{Iprf, IprfTable};
use plinko::subsets::{SubsetPrf, INVALID_CUTOFF};

use crate::hint_gen::subsets::xor_32;
use crate::hint_gen::types::{parity_word_at, BackupHint, RegularHint, PARITY_WORD_SIZE};

/// 64-bit words per parity.
const PARITY_WORDS: usize = PARITY_WORD_SIZE / 8;

/// Entries between two progress reports of one worker.
const PROGRESS_STEP: usize = 1024;

/// Immutable inputs for non-constant-time hint processing.
pub struct FastProcessInput<'a> {
    pub db_bytes: &'a [u8],
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
struct SharedParities {
    words: Vec<AtomicU64>,
}

impl SharedParities {
    fn new(count: usize) -> Self {
        Self {
            words: (0..count * PARITY_WORDS)
                .map(|_| AtomicU64::new(0))
                .collect(),
        }
    }

    fn xor(&self, index: usize, entry: &[u64; PARITY_WORDS]) {
        let slot: &[AtomicU64] = &self.words[index * PARITY_WORDS..(index + 1) * PARITY_WORDS];
        for (word, &value) in slot.iter().zip(entry) {
            word.fetch_xor(value, Ordering::Relaxed);
        }
    }

    fn parity(&self, index: usize) -> [u8; PARITY_WORD_SIZE] {
        let slot: &[AtomicU64] = &self.words[index * PARITY_WORDS..(index + 1) * PARITY_WORDS];
        let mut parity = [0u8; PARITY_WORD_SIZE];
        for (bytes, word) in parity.chunks_exact_mut(8).zip(slot) {
            bytes.copy_from_slice(&word.load(Ordering::Relaxed).to_le_bytes());
        }
        parity
    }
}

fn entry_words(entry: &[u8; PARITY_WORD_SIZE]) -> [u64; PARITY_WORDS] {
    let mut words = [0u64; PARITY_WORDS];
    for (word, bytes) in words.iter_mut().zip(entry.chunks_exact(8)) {
        *word = u64::from_le_bytes(bytes.try_into().expect("a chunk has 8 bytes"));
    }
    words
}

/// Accumulators of one run and the hints that give the cutoffs.
struct Accumulators<'a> {
    regular_hints: &'a [RegularHint],
    backup_hints: &'a [BackupHint],
    regular: SharedParities,
    backup_in: SharedParities,
    backup_out: SharedParities,
}

/// Processes all database entries through the fast (non-CT) path.
///
/// Computes the PMNS bin boundaries of each block once (`Iprf::table`).
/// For each entry at logical index `i`, computes the IPRF inverse to find
/// the set of hint indices, then XORs the entry into the matching regular
/// or backup hint parity accumulator based on block membership. The cutoff
/// of each hint defines the membership; an invalid hint gets no entry.
///
/// `input.threads` workers process the blocks. The parities are equal for
/// every thread count. The accumulators take 32 bytes per regular hint and
/// 64 bytes per backup hint in addition to the hints, until the function
/// returns.
///
/// `progress_callback` receives the number of entries that are done; calls
/// from different workers can arrive out of order.
pub fn process_entries_fast(
    input: FastProcessInput<'_>,
    regular_hints: &mut [RegularHint],
    backup_hints: &mut [BackupHint],
    progress_callback: impl Fn(usize) + Sync,
) {
    debug_assert_eq!(input.num_regular, regular_hints.len());
    debug_assert_eq!(input.num_backup, backup_hints.len());
    debug_assert_eq!(input.n_effective, input.block_iprfs.len() * input.w);

    let accumulators = Accumulators {
        regular_hints,
        backup_hints,
        regular: SharedParities::new(input.num_regular),
        backup_in: SharedParities::new(input.num_backup),
        backup_out: SharedParities::new(input.num_backup),
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
    for (j, hint) in regular_hints.iter_mut().enumerate() {
        xor_32(&mut hint.parity, &regular.parity(j));
    }
    for (b, hint) in backup_hints.iter_mut().enumerate() {
        xor_32(&mut hint.parity_in, &backup_in.parity(b));
        xor_32(&mut hint.parity_out, &backup_out.parity(b));
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
    let mut unreported: usize = 0;

    for offset in 0..input.w {
        let i: usize = block * input.w + offset;
        let entry: [u8; PARITY_WORD_SIZE] = if i < input.n_entries {
            parity_word_at(input.db_bytes, i)
        } else {
            [0u8; PARITY_WORD_SIZE]
        };
        let entry: [u64; PARITY_WORDS] = entry_words(&entry);

        for hint_id in table.inverse(offset as u64) {
            let j = hint_id as usize;
            if j < input.num_regular {
                let cutoff: u32 = accumulators.regular_hints[j].cutoff;
                if input.subset_prf.contains(hint_id, block, cutoff) {
                    accumulators.regular.xor(j, &entry);
                }
            } else {
                let backup_idx = j - input.num_regular;
                if backup_idx < input.num_backup {
                    let cutoff: u32 = accumulators.backup_hints[backup_idx].cutoff;
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

    #[test]
    fn shared_parity_keeps_the_byte_order_of_the_entry() {
        let mut entry = [0u8; PARITY_WORD_SIZE];
        for (i, byte) in entry.iter_mut().enumerate() {
            *byte = i as u8 + 1;
        }
        let parities = SharedParities::new(3);
        parities.xor(1, &entry_words(&entry));
        assert_eq!(parities.parity(0), [0u8; PARITY_WORD_SIZE]);
        assert_eq!(parities.parity(1), entry);
        assert_eq!(parities.parity(2), [0u8; PARITY_WORD_SIZE]);

        parities.xor(1, &entry_words(&entry));
        assert_eq!(parities.parity(1), [0u8; PARITY_WORD_SIZE]);
    }
}
