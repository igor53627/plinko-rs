use plinko::constant_time::{ct_eq_u64, ct_lt_u64, ct_select_usize, ct_xor_32_masked};
use plinko::iprf::{IprfTee, IprfTeeTable};
use plinko::subsets::{SubsetPrf, INVALID_CUTOFF};

use crate::hint_gen::types::{parity_word_at, BackupHint, RegularHint, PARITY_WORD_SIZE};

/// Immutable inputs for constant-time hint processing.
pub struct CtProcessInput<'a> {
    pub db_bytes: &'a [u8],
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
/// - `regular_hints` and `backup_hints` must have at least 1 element
pub fn process_entries_ct(
    input: CtProcessInput<'_>,
    regular_hints: &mut [RegularHint],
    backup_hints: &mut [BackupHint],
    progress_callback: impl Fn(usize),
) {
    assert!(
        input.num_backup >= 1 && !regular_hints.is_empty() && !backup_hints.is_empty(),
        "CT path requires at least 1 regular and 1 backup hint for safe dummy indexing"
    );
    debug_assert_eq!(input.n_effective, input.block_iprfs_ct.len() * input.w);
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

        let entry: [u8; PARITY_WORD_SIZE] = if i < input.n_entries {
            parity_word_at(input.db_bytes, i)
        } else {
            [0u8; PARITY_WORD_SIZE]
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
            let regular_cutoff = regular_hints[regular_idx].cutoff as u64;
            let in_regular_subset = ct_lt_u64(select_value, regular_cutoff);

            let backup_idx_clamped = ct_select_usize(is_valid_backup, backup_idx, 0);
            let backup_cutoff = backup_hints[backup_idx_clamped].cutoff as u64;
            let backup_has_subset = 1 - ct_eq_u64(backup_cutoff, INVALID_CUTOFF as u64);
            let in_backup_subset = ct_lt_u64(select_value, backup_cutoff);

            let update_regular = in_range & is_regular & in_regular_subset;
            let update_backup_in = in_range & is_backup & in_backup_subset;
            let update_backup_out =
                in_range & is_backup & backup_has_subset & (1 - in_backup_subset);

            ct_xor_32_masked(
                &mut regular_hints[regular_idx].parity,
                &entry,
                update_regular,
            );
            ct_xor_32_masked(
                &mut backup_hints[backup_idx_clamped].parity_in,
                &entry,
                update_backup_in,
            );
            ct_xor_32_masked(
                &mut backup_hints[backup_idx_clamped].parity_out,
                &entry,
                update_backup_out,
            );
        }

        if i % 10000 == 0 {
            progress_callback(i);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hint_gen::driver::{init_hints, HintParams};
    use crate::hint_gen::fast_path::{process_entries_fast, FastProcessInput};
    use crate::hint_gen::keys::derive_block_keys;
    use crate::hint_gen::types::DB_ENTRY_SIZE;
    use plinko::iprf::{Iprf, PrfKey128};

    struct Scenario {
        master_seed: [u8; 32],
        n_entries: usize,
        w: usize,
        c: usize,
        lambda: usize,
        /// Regular and backup hint that get `INVALID_CUTOFF`, to model a tie at the cutoff.
        invalid: Option<(usize, usize)>,
    }

    type Parities = (Vec<[u8; 32]>, Vec<[u8; 32]>, Vec<[u8; 32]>);

    impl Scenario {
        fn params(&self) -> HintParams {
            let num_regular: usize = self.lambda * self.w;
            HintParams {
                num_regular,
                num_backup: num_regular,
                total_hints: 2 * num_regular,
            }
        }

        /// A v3 database: the parity word of entry `i` is non-zero and distinct.
        fn db_bytes(&self) -> Vec<u8> {
            let mut db: Vec<u8> = vec![0u8; self.n_entries * DB_ENTRY_SIZE];
            for i in 0..self.n_entries {
                let record: &mut [u8] = &mut db[i * DB_ENTRY_SIZE..(i + 1) * DB_ENTRY_SIZE];
                record[0..8].copy_from_slice(&(i as u64 + 1).to_le_bytes());
                record[31] = 0x80 | (i as u8);
                record[32..40].fill(0xEE);
            }
            db
        }

        fn init(&self, subset_prf: &SubsetPrf) -> (Vec<RegularHint>, Vec<BackupHint>) {
            let (mut regular, mut backup) = init_hints(subset_prf, self.c, &self.params());
            if let Some((regular_idx, backup_idx)) = self.invalid {
                regular[regular_idx].cutoff = INVALID_CUTOFF;
                backup[backup_idx].cutoff = INVALID_CUTOFF;
            }
            (regular, backup)
        }

        fn parities_of(regular: &[RegularHint], backup: &[BackupHint]) -> Parities {
            (
                regular.iter().map(|h| h.parity).collect(),
                backup.iter().map(|h| h.parity_in).collect(),
                backup.iter().map(|h| h.parity_out).collect(),
            )
        }

        fn run_fast(&self) -> Parities {
            let params: HintParams = self.params();
            let db: Vec<u8> = self.db_bytes();
            let subset_prf: SubsetPrf = SubsetPrf::new(&self.master_seed);
            let (mut regular, mut backup) = self.init(&subset_prf);
            let keys: Vec<PrfKey128> = derive_block_keys(&self.master_seed, self.c);
            let block_iprfs: Vec<Iprf> = keys
                .iter()
                .map(|key| Iprf::new(*key, params.total_hints as u64, self.w as u64))
                .collect();
            process_entries_fast(
                FastProcessInput {
                    db_bytes: &db,
                    n_entries: self.n_entries,
                    n_effective: self.c * self.w,
                    w: self.w,
                    num_regular: params.num_regular,
                    num_backup: params.num_backup,
                    block_iprfs: &block_iprfs,
                    subset_prf: &subset_prf,
                },
                &mut regular,
                &mut backup,
                |_| {},
            );
            Self::parities_of(&regular, &backup)
        }

        fn run_ct(&self) -> Parities {
            let params: HintParams = self.params();
            let db: Vec<u8> = self.db_bytes();
            let subset_prf: SubsetPrf = SubsetPrf::new(&self.master_seed);
            let (mut regular, mut backup) = self.init(&subset_prf);
            let keys: Vec<PrfKey128> = derive_block_keys(&self.master_seed, self.c);
            let block_iprfs_ct: Vec<IprfTee> = keys
                .iter()
                .map(|key| IprfTee::new(*key, params.total_hints as u64, self.w as u64))
                .collect();
            process_entries_ct(
                CtProcessInput {
                    db_bytes: &db,
                    n_entries: self.n_entries,
                    n_effective: self.c * self.w,
                    w: self.w,
                    num_regular: params.num_regular,
                    num_backup: params.num_backup,
                    block_iprfs_ct: &block_iprfs_ct,
                    subset_prf: &subset_prf,
                },
                &mut regular,
                &mut backup,
                |_| {},
            );
            Self::parities_of(&regular, &backup)
        }

        /// Plinko Fig. 7, hint by hint: hint `j` takes the entry at offset
        /// `iF.F(k_alpha, j)` of each block `alpha`. Uses `forward` only, so it
        /// does not share the inverse or the streaming loop with the code under test.
        fn run_definition(&self) -> Parities {
            let params: HintParams = self.params();
            let db: Vec<u8> = self.db_bytes();
            let subset_prf: SubsetPrf = SubsetPrf::new(&self.master_seed);
            let keys: Vec<PrfKey128> = derive_block_keys(&self.master_seed, self.c);
            let block_iprfs: Vec<Iprf> = keys
                .iter()
                .map(|key| Iprf::new(*key, params.total_hints as u64, self.w as u64))
                .collect();

            let entry_at = |block: usize, hint_id: u64| -> [u8; 32] {
                let offset: usize = block_iprfs[block].forward(hint_id) as usize;
                let index: usize = block * self.w + offset;
                if index < self.n_entries {
                    parity_word_at(&db, index)
                } else {
                    [0u8; 32]
                }
            };
            let xor = |dst: &mut [u8; 32], src: [u8; 32]| {
                for (d, s) in dst.iter_mut().zip(src) {
                    *d ^= s;
                }
            };
            let subset_of = |hint_id: u64, size: usize| -> Vec<bool> {
                let mut values: Vec<u32> = Vec::new();
                subset_prf.select_values(hint_id, self.c, &mut values);
                let mut order: Vec<usize> = (0..self.c).collect();
                order.sort_by_key(|&block| values[block]);
                let mut member: Vec<bool> = vec![false; self.c];
                for &block in &order[..size] {
                    member[block] = true;
                }
                member
            };

            let mut regular: Vec<[u8; 32]> = vec![[0u8; 32]; params.num_regular];
            let mut backup_in: Vec<[u8; 32]> = vec![[0u8; 32]; params.num_backup];
            let mut backup_out: Vec<[u8; 32]> = vec![[0u8; 32]; params.num_backup];
            let invalid_ids: Vec<usize> = self
                .invalid
                .map(|(regular_idx, backup_idx)| vec![regular_idx, params.num_regular + backup_idx])
                .unwrap_or_default();
            for j in (0..params.total_hints).filter(|j| !invalid_ids.contains(j)) {
                let hint_id: u64 = j as u64;
                if j < params.num_regular {
                    let member: Vec<bool> = subset_of(hint_id, self.c / 2 + 1);
                    for block in (0..self.c).filter(|&block| member[block]) {
                        xor(&mut regular[j], entry_at(block, hint_id));
                    }
                } else {
                    let b: usize = j - params.num_regular;
                    let member: Vec<bool> = subset_of(hint_id, self.c / 2);
                    for (block, &is_member) in member.iter().enumerate() {
                        if is_member {
                            xor(&mut backup_in[b], entry_at(block, hint_id));
                        } else {
                            xor(&mut backup_out[b], entry_at(block, hint_id));
                        }
                    }
                }
            }
            (regular, backup_in, backup_out)
        }
    }

    fn assert_all_paths_agree(scenario: &Scenario) -> Parities {
        let expected: Parities = scenario.run_definition();
        let fast: Parities = scenario.run_fast();
        let ct: Parities = scenario.run_ct();

        assert!(
            expected.0.iter().filter(|p| **p != [0u8; 32]).count() > expected.0.len() / 2,
            "the scenario must give non-trivial regular parities"
        );
        assert!(
            expected.1 != expected.2,
            "in and out parities of backup hints must differ"
        );

        assert!(fast.0 == expected.0, "fast path: regular parities");
        assert!(fast.1 == expected.1, "fast path: backup in-parities");
        assert!(fast.2 == expected.2, "fast path: backup out-parities");
        assert!(ct.0 == expected.0, "constant-time path: regular parities");
        assert!(ct.1 == expected.1, "constant-time path: backup in-parities");
        assert!(
            ct.2 == expected.2,
            "constant-time path: backup out-parities"
        );
        expected
    }

    #[test]
    fn fast_and_ct_paths_match_the_hint_definition() {
        assert_all_paths_agree(&Scenario {
            master_seed: [42u8; 32],
            n_entries: 32,
            w: 8,
            c: 4,
            lambda: 2,
            invalid: None,
        });
    }

    #[test]
    fn invalid_hints_take_no_entry_in_both_paths() {
        let zero: [u8; 32] = [0u8; 32];
        let (regular_idx, backup_idx): (usize, usize) = (3, 5);
        let mut scenario = Scenario {
            master_seed: [42u8; 32],
            n_entries: 32,
            w: 8,
            c: 4,
            lambda: 2,
            invalid: None,
        };

        let valid: Parities = assert_all_paths_agree(&scenario);
        assert_ne!(valid.0[regular_idx], zero);
        assert_ne!(valid.1[backup_idx], zero);
        assert_ne!(valid.2[backup_idx], zero);

        scenario.invalid = Some((regular_idx, backup_idx));
        let invalid: Parities = assert_all_paths_agree(&scenario);
        assert_eq!(invalid.0[regular_idx], zero);
        assert_eq!(invalid.1[backup_idx], zero);
        assert_eq!(invalid.2[backup_idx], zero);
    }

    /// 1280 hints, with a padded last block and six blocks.
    #[test]
    fn fast_and_ct_paths_match_the_hint_definition_above_1024_hints() {
        assert_all_paths_agree(&Scenario {
            master_seed: [7u8; 32],
            n_entries: 90,
            w: 16,
            c: 6,
            lambda: 40,
            invalid: None,
        });
    }
}
