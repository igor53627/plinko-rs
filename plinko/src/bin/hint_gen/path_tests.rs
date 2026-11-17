//! Tests of the streaming paths against the hint definition of Plinko Fig. 7.
//!
//! The standard path is always checked. The constant-time path is checked
//! with the `tee` feature.

#[cfg(feature = "tee")]
use crate::hint_gen::ct_path::{process_entries_ct, CtProcessInput};
use crate::hint_gen::driver::{init_hints, HintParams};
use crate::hint_gen::fast_path::{process_entries_fast, FastProcessInput};
use crate::hint_gen::keys::derive_block_keys;
use crate::hint_gen::types::{parity_word_at, BackupHint, RegularHint, DB_ENTRY_SIZE};
#[cfg(feature = "tee")]
use plinko::iprf::IprfTee;
use plinko::iprf::{Iprf, PrfKey128};
use plinko::subsets::{SubsetPrf, INVALID_CUTOFF};

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

    fn run_fast(&self, threads: usize) -> Parities {
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
                threads,
            },
            &mut regular,
            &mut backup,
            |_| {},
        );
        Self::parities_of(&regular, &backup)
    }

    #[cfg(feature = "tee")]
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

fn assert_paths_match_the_definition(scenario: &Scenario) -> Parities {
    let expected: Parities = scenario.run_definition();
    let fast: Parities = scenario.run_fast(1);
    // Fewer threads than blocks, and more threads than blocks.
    for threads in [2, 3, 16] {
        assert!(
            scenario.run_fast(threads) == fast,
            "fast path with {threads} threads"
        );
    }

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
    #[cfg(feature = "tee")]
    {
        let ct: Parities = scenario.run_ct();
        assert!(ct.0 == expected.0, "constant-time path: regular parities");
        assert!(ct.1 == expected.1, "constant-time path: backup in-parities");
        assert!(
            ct.2 == expected.2,
            "constant-time path: backup out-parities"
        );
    }
    expected
}

#[test]
fn streaming_paths_match_the_hint_definition() {
    assert_paths_match_the_definition(&Scenario {
        master_seed: [42u8; 32],
        n_entries: 32,
        w: 8,
        c: 4,
        lambda: 2,
        invalid: None,
    });
}

#[test]
fn invalid_hints_take_no_entry() {
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

    let valid: Parities = assert_paths_match_the_definition(&scenario);
    assert_ne!(valid.0[regular_idx], zero);
    assert_ne!(valid.1[backup_idx], zero);
    assert_ne!(valid.2[backup_idx], zero);

    scenario.invalid = Some((regular_idx, backup_idx));
    let invalid: Parities = assert_paths_match_the_definition(&scenario);
    assert_eq!(invalid.0[regular_idx], zero);
    assert_eq!(invalid.1[backup_idx], zero);
    assert_eq!(invalid.2[backup_idx], zero);
}

/// 1280 hints, with a padded last block and six blocks.
#[test]
fn streaming_paths_match_the_hint_definition_above_1024_hints() {
    assert_paths_match_the_definition(&Scenario {
        master_seed: [7u8; 32],
        n_entries: 90,
        w: 16,
        c: 6,
        lambda: 40,
        invalid: None,
    });
}
