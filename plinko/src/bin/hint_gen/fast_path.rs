use plinko::iprf::Iprf;
#[cfg(feature = "batch_iprf")]
use plinko::iprf::SwapOrNotSrBatchScratch;

use crate::hint_gen::subsets::{block_in_subset, xor_32};
use crate::hint_gen::types::{BackupHint, RegularHint, WORD_SIZE};

#[allow(clippy::too_many_arguments)]
pub fn process_entries_fast(
    db_bytes: &[u8],
    n_entries: usize,
    n_effective: usize,
    w: usize,
    _c: usize,
    num_regular: usize,
    num_backup: usize,
    block_iprfs: &[Iprf],
    regular_hint_blocks: &[Vec<usize>],
    backup_hint_blocks: &[Vec<usize>],
    regular_hints: &mut [RegularHint],
    backup_hints: &mut [BackupHint],
    progress_callback: impl Fn(usize),
) {
    #[cfg(feature = "batch_iprf")]
    let mut preimages: Vec<u64> = Vec::new();
    #[cfg(feature = "batch_iprf")]
    let mut scratch = SwapOrNotSrBatchScratch::new();

    for i in 0..n_effective {
        let block = i / w;
        let offset = i % w;

        let entry: [u8; 32] = if i < n_entries {
            let entry_offset = i * WORD_SIZE;
            db_bytes[entry_offset..entry_offset + WORD_SIZE]
                .try_into()
                .unwrap()
        } else {
            [0u8; 32]
        };

        #[cfg(feature = "batch_iprf")]
        {
            block_iprfs[block].inverse_into(offset as u64, &mut preimages, &mut scratch);

            for &j in preimages.iter() {
                let j = j as usize;
                if j < num_regular {
                    if block_in_subset(&regular_hint_blocks[j], block) {
                        xor_32(&mut regular_hints[j].parity, &entry);
                    }
                } else {
                    let backup_idx = j - num_regular;
                    if backup_idx < num_backup {
                        if block_in_subset(&backup_hint_blocks[backup_idx], block) {
                            xor_32(&mut backup_hints[backup_idx].parity_in, &entry);
                        } else {
                            xor_32(&mut backup_hints[backup_idx].parity_out, &entry);
                        }
                    }
                }
            }
        }

        #[cfg(not(feature = "batch_iprf"))]
        {
            let hint_indices = block_iprfs[block].inverse(offset as u64);

            for j in hint_indices {
                let j = j as usize;
                if j < num_regular {
                    if block_in_subset(&regular_hint_blocks[j], block) {
                        xor_32(&mut regular_hints[j].parity, &entry);
                    }
                } else {
                    let backup_idx = j - num_regular;
                    if backup_idx < num_backup {
                        if block_in_subset(&backup_hint_blocks[backup_idx], block) {
                            xor_32(&mut backup_hints[backup_idx].parity_in, &entry);
                        } else {
                            xor_32(&mut backup_hints[backup_idx].parity_out, &entry);
                        }
                    }
                }
            }
        }

        if i % 10000 == 0 {
            progress_callback(i);
        }
    }
}
