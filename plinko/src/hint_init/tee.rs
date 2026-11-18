//! Constant-time forms of the two HintInit phases (`tee` feature, experimental).
//!
//! See `docs/constant_time_mode.md` for what this code hides and what it does
//! not hide.

use super::{
    fill_entry_words, for_each_block, hint_cutoffs_with, record_at, words_per_record, Accumulators,
    HintCutoffs, Layout, PROGRESS_STEP,
};
use crate::constant_time::{ct_eq_u64, ct_lt_u64, ct_select_usize};
use crate::hints::HintSet;
use crate::iprf::{IprfTee, IprfTeeTable};
use crate::subsets::{SubsetPrf, INVALID_CUTOFF};

/// Computes the same cutoffs as [`super::hint_cutoffs`] with a fixed amount
/// of work for each hint ([`SubsetPrf::cutoff_ct`]).
///
/// A worker writes only to its own part of the result, and the parts depend
/// only on the hint counts and on `threads`. So more workers add no timing
/// that depends on a secret value.
pub fn hint_cutoffs_ct(
    subset_prf: &SubsetPrf,
    num_blocks: usize,
    num_regular: usize,
    num_backup: usize,
    threads: usize,
) -> HintCutoffs {
    hint_cutoffs_with(
        SubsetPrf::cutoff_ct,
        subset_prf,
        num_blocks,
        num_regular,
        num_backup,
        threads,
    )
}

/// Computes the parities of `hints` from `database` with a fixed amount of
/// work for each entry.
///
/// The arguments and the result are those of [`super::fill_parities`], with
/// the constant-time iPRF of each block. The parities are equal to those of
/// [`super::fill_parities`] for every thread count.
///
/// The PMNS bin boundaries of each block are computed once with
/// `IprfTee::table_ct`, whose work depends only on the hint count and the
/// block size. The table is read at the offset of the entry, which is public.
/// Each entry runs `MAX_PREIMAGES` slots, and a padding record gets the same
/// work as a real record. Block membership is one AES call on the hint index
/// and one masked compare against the cutoff of the hint.
///
/// Not hidden: the cutoffs and the parity accumulators are read and written
/// at the hint index, so the memory access pattern depends on that index.
/// With more than one worker, two workers that update the same parity word
/// at the same time delay each other, and which words those are depends on
/// the hint indices of both workers.
///
/// # Panics
/// Panics if `hints` has no regular hint or no backup hint (index 0 of each
/// is the target of the dummy accesses), if `database` or `block_iprfs` does
/// not fit the geometry of `hints`, or if a PMNS ball count exceeds its
/// public bound (`IprfTeeTable::overflowed`). The hints would be wrong after
/// an overflow; the caller keeps its probability negligible with the hint
/// count per block.
pub fn fill_parities_ct(
    hints: &mut HintSet,
    database: &[u8],
    block_iprfs: &[IprfTee],
    subset_prf: &SubsetPrf,
    threads: usize,
    progress_callback: impl Fn(usize) + Sync,
) {
    assert!(
        hints.num_regular() >= 1 && hints.num_backup() >= 1,
        "CT path requires at least 1 regular and 1 backup hint for safe dummy indexing"
    );
    let layout: Layout = Layout::new(hints, database, block_iprfs.len());
    let accumulators: Accumulators = Accumulators::new(&layout);
    let regular_cutoffs: &[u32] = hints.regular_cutoffs();
    let backup_cutoffs: &[u32] = hints.backup_cutoffs();

    for_each_block(
        layout.num_blocks,
        threads,
        progress_callback,
        |block, report| {
            let table: IprfTeeTable<'_> = block_iprfs[block].table_ct();
            assert!(
                !table.overflowed(),
                "PMNS ball count of block {block} exceeds the constant-time bound; \
                 reduce total_hints or increase w"
            );
            let mut entry: Vec<u64> = vec![0u64; words_per_record(layout.record_size)];
            let mut unreported: usize = 0;

            for offset in 0..layout.block_size {
                let i: usize = block * layout.block_size + offset;
                // The number of real entries is public.
                if i < layout.num_entries {
                    fill_entry_words(record_at(database, layout.record_size, i), &mut entry);
                } else {
                    entry.fill(0);
                }

                let (indices, count) = table.inverse_ct(offset as u64);

                for (t, &idx) in indices.iter().enumerate() {
                    let in_range = ct_lt_u64(t as u64, count as u64);

                    let j = idx as usize;

                    let is_regular = ct_lt_u64(j as u64, layout.num_regular as u64);
                    let backup_idx = j.wrapping_sub(layout.num_regular);
                    let is_valid_backup = ct_lt_u64(backup_idx as u64, layout.num_backup as u64);
                    let is_backup = (1 - is_regular) & is_valid_backup;

                    let select_value = subset_prf.select(idx, block) as u64;

                    let regular_idx = ct_select_usize(is_regular, j, 0);
                    let regular_cutoff = regular_cutoffs[regular_idx] as u64;
                    let in_regular_subset = ct_lt_u64(select_value, regular_cutoff);

                    let backup_idx_clamped = ct_select_usize(is_valid_backup, backup_idx, 0);
                    let backup_cutoff = backup_cutoffs[backup_idx_clamped] as u64;
                    let backup_has_subset = 1 - ct_eq_u64(backup_cutoff, INVALID_CUTOFF as u64);
                    let in_backup_subset = ct_lt_u64(select_value, backup_cutoff);

                    let update_regular = in_range & is_regular & in_regular_subset;
                    let update_backup_in = in_range & is_backup & in_backup_subset;
                    let update_backup_out =
                        in_range & is_backup & backup_has_subset & (1 - in_backup_subset);

                    accumulators
                        .regular
                        .xor_masked(regular_idx, &entry, update_regular);
                    accumulators
                        .backup_in
                        .xor_masked(backup_idx_clamped, &entry, update_backup_in);
                    accumulators.backup_out.xor_masked(
                        backup_idx_clamped,
                        &entry,
                        update_backup_out,
                    );
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
