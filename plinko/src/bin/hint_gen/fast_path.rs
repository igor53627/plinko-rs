//! Non-constant-time (fast) hint processing path.
//!
//! Iterates over every database entry, resolves IPRF preimages to determine
//! which hints each entry contributes to, and XORs the entry into the
//! appropriate parity accumulators. This path leaks the access pattern
//! through data-dependent branches and is suitable only when timing
//! side-channels are not a concern.

use plinko::iprf::Iprf;

use crate::hint_gen::subsets::{block_in_subset, xor_32};
use crate::hint_gen::types::{BackupHint, RegularHint, WORD_SIZE};

/// Immutable inputs for non-constant-time hint processing.
pub struct FastProcessInput<'a> {
    pub db_bytes: &'a [u8],
    pub n_entries: usize,
    pub n_effective: usize,
    pub w: usize,
    pub num_regular: usize,
    pub num_backup: usize,
    pub block_iprfs: &'a [Iprf],
    pub regular_hint_blocks: &'a [Vec<usize>],
    pub backup_hint_blocks: &'a [Vec<usize>],
}

/// Processes all database entries through the fast (non-CT) path.
///
/// For each entry at logical index `i`, computes the IPRF inverse to find
/// the set of hint indices, then XORs the entry into the matching regular
/// or backup hint parity accumulator based on block membership.
pub fn process_entries_fast(
    input: FastProcessInput<'_>,
    regular_hints: &mut [RegularHint],
    backup_hints: &mut [BackupHint],
    progress_callback: impl Fn(usize),
) {
    for i in 0..input.n_effective {
        let block = i / input.w;
        let offset = i % input.w;

        let entry: [u8; 32] = if i < input.n_entries {
            let entry_offset = i * WORD_SIZE;
            input.db_bytes[entry_offset..entry_offset + WORD_SIZE]
                .try_into()
                .unwrap()
        } else {
            [0u8; 32]
        };

        let hint_indices = input.block_iprfs[block].inverse(offset as u64);

        for j in hint_indices {
            let j = j as usize;
            if j < input.num_regular {
                if block_in_subset(&input.regular_hint_blocks[j], block) {
                    xor_32(&mut regular_hints[j].parity, &entry);
                }
            } else {
                let backup_idx = j - input.num_regular;
                if backup_idx < input.num_backup {
                    if block_in_subset(&input.backup_hint_blocks[backup_idx], block) {
                        xor_32(&mut backup_hints[backup_idx].parity_in, &entry);
                    } else {
                        xor_32(&mut backup_hints[backup_idx].parity_out, &entry);
                    }
                }
            }
        }

        if i % 10000 == 0 {
            progress_callback(i);
        }
    }
}
