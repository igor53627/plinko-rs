//! Non-constant-time (fast) hint processing path.
//!
//! Iterates over every database entry, resolves IPRF preimages to determine
//! which hints each entry contributes to, and XORs the entry into the
//! appropriate parity accumulators. This path leaks the access pattern
//! through data-dependent branches and is suitable only when timing
//! side-channels are not a concern.

use plinko::iprf::{Iprf, IprfTable};
use plinko::subsets::{SubsetPrf, INVALID_CUTOFF};

use crate::hint_gen::subsets::xor_32;
use crate::hint_gen::types::{parity_word_at, BackupHint, RegularHint, PARITY_WORD_SIZE};

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
}

/// Processes all database entries through the fast (non-CT) path.
///
/// Computes the PMNS bin boundaries of each block once (`Iprf::table`).
/// For each entry at logical index `i`, computes the IPRF inverse to find
/// the set of hint indices, then XORs the entry into the matching regular
/// or backup hint parity accumulator based on block membership. The cutoff
/// of each hint defines the membership; an invalid hint gets no entry.
pub fn process_entries_fast(
    input: FastProcessInput<'_>,
    regular_hints: &mut [RegularHint],
    backup_hints: &mut [BackupHint],
    progress_callback: impl Fn(usize),
) {
    debug_assert_eq!(input.num_regular, regular_hints.len());
    debug_assert_eq!(input.num_backup, backup_hints.len());
    debug_assert_eq!(input.n_effective, input.block_iprfs.len() * input.w);
    let mut table: Option<IprfTable<'_>> = None;
    for i in 0..input.n_effective {
        let block = i / input.w;
        let offset = i % input.w;
        if offset == 0 {
            table = Some(input.block_iprfs[block].table());
        }

        let entry: [u8; PARITY_WORD_SIZE] = if i < input.n_entries {
            parity_word_at(input.db_bytes, i)
        } else {
            [0u8; PARITY_WORD_SIZE]
        };

        let hint_indices = table
            .as_ref()
            .expect("the table is built at offset 0 of each block")
            .inverse(offset as u64);

        for hint_id in hint_indices {
            let j = hint_id as usize;
            if j < input.num_regular {
                let hint: &mut RegularHint = &mut regular_hints[j];
                if input.subset_prf.contains(hint_id, block, hint.cutoff) {
                    xor_32(&mut hint.parity, &entry);
                }
            } else {
                let backup_idx = j - input.num_regular;
                if backup_idx < input.num_backup {
                    let hint: &mut BackupHint = &mut backup_hints[backup_idx];
                    if hint.cutoff == INVALID_CUTOFF {
                        continue;
                    }
                    if input.subset_prf.contains(hint_id, block, hint.cutoff) {
                        xor_32(&mut hint.parity_in, &entry);
                    } else {
                        xor_32(&mut hint.parity_out, &entry);
                    }
                }
            }
        }

        if i % 10000 == 0 {
            progress_callback(i);
        }
    }
}
