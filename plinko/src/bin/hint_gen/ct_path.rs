use plinko::constant_time::{ct_eq_u64, ct_lt_u64, ct_select_usize, ct_xor_masked};
use plinko::hints::HintSet;
use plinko::iprf::{IprfTee, IprfTeeTable};
use plinko::subsets::{SubsetPrf, INVALID_CUTOFF};

use crate::hint_gen::types::record_at;

/// Immutable inputs for constant-time hint processing.
pub struct CtProcessInput<'a> {
    pub db_bytes: &'a [u8],
    /// Bytes of one database record; equal to the record size of the hint set.
    pub record_size: usize,
    pub n_entries: usize,
    pub n_effective: usize,
    pub w: usize,
    pub num_regular: usize,
    pub num_backup: usize,
    pub block_iprfs_ct: &'a [IprfTee],
    pub subset_prf: &'a SubsetPrf,
}

/// Process database entries in constant-time for TEE execution.
///
/// The PMNS bin boundaries of each block are computed once with
/// `IprfTee::table_ct`, whose work depends only on the hint count and `w`.
/// The table is read at the offset of the entry, which is public.
///
/// Block membership is one AES call on the hint index and one masked compare
/// against the cutoff of the hint. The cutoffs and parities are read at the
/// hint index, so the memory access pattern still depends on that index.
///
/// # Panics
/// Panics if a PMNS ball count exceeds its public bound
/// (`IprfTeeTable::overflowed`). `validate_hint_params` keeps the probability
/// negligible. The hints would be wrong otherwise.
///
/// # Safety Requirements
/// - `num_backup` must be >= 1 (CT indexing clamps to index 0 for dummy access)
/// - `hints` must have at least 1 regular and 1 backup hint
///
/// `hints` holds the cutoffs; the function XORs the records into its parities.
/// The record size and the number of real entries are public.
pub fn process_entries_ct(
    input: CtProcessInput<'_>,
    hints: &mut HintSet,
    progress_callback: impl Fn(usize),
) {
    assert!(
        input.num_backup >= 1 && hints.num_regular() >= 1 && hints.num_backup() >= 1,
        "CT path requires at least 1 regular and 1 backup hint for safe dummy indexing"
    );
    assert_eq!(input.record_size, hints.record_size());
    debug_assert_eq!(input.num_regular, hints.num_regular());
    debug_assert_eq!(input.num_backup, hints.num_backup());
    debug_assert_eq!(input.n_effective, input.block_iprfs_ct.len() * input.w);
    let padding_record: Vec<u8> = vec![0u8; input.record_size];
    let mut table: Option<IprfTeeTable<'_>> = None;
    for i in 0..input.n_effective {
        let block = i / input.w;
        let offset = i % input.w;
        if offset == 0 {
            let block_table: IprfTeeTable<'_> = input.block_iprfs_ct[block].table_ct();
            assert!(
                !block_table.overflowed(),
                "PMNS ball count of block {block} exceeds the constant-time bound; \
                 reduce total_hints or increase w"
            );
            table = Some(block_table);
        }

        let entry: &[u8] = if i < input.n_entries {
            record_at(input.db_bytes, input.record_size, i)
        } else {
            &padding_record
        };

        let (indices, count) = table
            .as_ref()
            .expect("the table is built at offset 0 of each block")
            .inverse_ct(offset as u64);

        for (t, &idx) in indices.iter().enumerate() {
            let in_range = ct_lt_u64(t as u64, count as u64);

            let j = idx as usize;

            let is_regular = ct_lt_u64(j as u64, input.num_regular as u64);
            let backup_idx = j.wrapping_sub(input.num_regular);
            let is_valid_backup = ct_lt_u64(backup_idx as u64, input.num_backup as u64);
            let is_backup = (1 - is_regular) & is_valid_backup;

            let select_value = input.subset_prf.select(idx, block) as u64;

            let regular_idx = ct_select_usize(is_regular, j, 0);
            let regular_cutoff = hints.regular_cutoffs()[regular_idx] as u64;
            let in_regular_subset = ct_lt_u64(select_value, regular_cutoff);

            let backup_idx_clamped = ct_select_usize(is_valid_backup, backup_idx, 0);
            let backup_cutoff = hints.backup_cutoffs()[backup_idx_clamped] as u64;
            let backup_has_subset = 1 - ct_eq_u64(backup_cutoff, INVALID_CUTOFF as u64);
            let in_backup_subset = ct_lt_u64(select_value, backup_cutoff);

            let update_regular = in_range & is_regular & in_regular_subset;
            let update_backup_in = in_range & is_backup & in_backup_subset;
            let update_backup_out =
                in_range & is_backup & backup_has_subset & (1 - in_backup_subset);

            ct_xor_masked(hints.regular_parity_mut(regular_idx), entry, update_regular);
            ct_xor_masked(
                hints.backup_parity_in_mut(backup_idx_clamped),
                entry,
                update_backup_in,
            );
            ct_xor_masked(
                hints.backup_parity_out_mut(backup_idx_clamped),
                entry,
                update_backup_out,
            );
        }

        if i % 10000 == 0 {
            progress_callback(i);
        }
    }
}
