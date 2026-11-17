//! Tests of the streaming paths against the hint definition of Plinko Fig. 7.
//!
//! The standard path is always checked. The constant-time path is checked
//! with the `tee` feature.

#[cfg(feature = "tee")]
use crate::hint_gen::ct_path::{process_entries_ct, CtProcessInput};
use crate::hint_gen::driver::{hint_cutoffs, HintParams};
use crate::hint_gen::fast_path::{process_entries_fast, FastProcessInput};
use crate::hint_gen::keys::derive_block_keys;
use crate::hint_gen::types::record_at;
use plinko::hints::{seed_check, HintGeometry, HintSet};
#[cfg(feature = "tee")]
use plinko::iprf::IprfTee;
use plinko::iprf::{Iprf, PrfKey128};
use plinko::subsets::{SubsetPrf, INVALID_CUTOFF};

struct Scenario {
    master_seed: [u8; 32],
    /// Bytes of one database record.
    record_size: usize,
    n_entries: usize,
    w: usize,
    c: usize,
    lambda: usize,
    /// Regular and backup hint that get `INVALID_CUTOFF`, to model a tie at the cutoff.
    invalid: Option<(usize, usize)>,
}

/// Regular parities, backup in-parities, backup out-parities.
type Parities = (Vec<Vec<u8>>, Vec<Vec<u8>>, Vec<Vec<u8>>);

impl Scenario {
    fn params(&self) -> HintParams {
        let num_regular: usize = self.lambda * self.w;
        HintParams {
            num_regular,
            num_backup: num_regular,
            total_hints: 2 * num_regular,
        }
    }

    /// A database in which every byte of a record depends on the record
    /// index and on the byte position, and the first and the last byte of a
    /// record are non-zero.
    fn db_bytes(&self) -> Vec<u8> {
        let mut db: Vec<u8> = vec![0u8; self.n_entries * self.record_size];
        for (i, record) in db.chunks_exact_mut(self.record_size).enumerate() {
            for (k, byte) in record.iter_mut().enumerate() {
                *byte = (i as u8).wrapping_mul(37) ^ (k as u8).wrapping_mul(101) ^ 0x5A;
            }
            record[0] |= 0x01;
            record[self.record_size - 1] |= 0x80;
        }
        db
    }

    fn init(&self, subset_prf: &SubsetPrf) -> HintSet {
        let (mut regular, mut backup) = hint_cutoffs(subset_prf, self.c, &self.params());
        if let Some((regular_idx, backup_idx)) = self.invalid {
            regular[regular_idx] = INVALID_CUTOFF;
            backup[backup_idx] = INVALID_CUTOFF;
        }
        let geometry = HintGeometry {
            num_entries: self.n_entries as u64,
            record_size: self.record_size as u32,
            block_size: self.w as u64,
            num_blocks: self.c as u64,
        };
        HintSet::new(geometry, seed_check(&self.master_seed), regular, backup).expect("hint set")
    }

    fn parities_of(hints: &HintSet) -> Parities {
        (
            (0..hints.num_regular())
                .map(|j| hints.regular_parity(j).to_vec())
                .collect(),
            (0..hints.num_backup())
                .map(|b| hints.backup_parity_in(b).to_vec())
                .collect(),
            (0..hints.num_backup())
                .map(|b| hints.backup_parity_out(b).to_vec())
                .collect(),
        )
    }

    fn run_fast(&self, threads: usize) -> Parities {
        let params: HintParams = self.params();
        let db: Vec<u8> = self.db_bytes();
        let subset_prf: SubsetPrf = SubsetPrf::new(&self.master_seed);
        let mut hints: HintSet = self.init(&subset_prf);
        let keys: Vec<PrfKey128> = derive_block_keys(&self.master_seed, self.c);
        let block_iprfs: Vec<Iprf> = keys
            .iter()
            .map(|key| Iprf::new(*key, params.total_hints as u64, self.w as u64))
            .collect();
        process_entries_fast(
            FastProcessInput {
                db_bytes: &db,
                record_size: self.record_size,
                n_entries: self.n_entries,
                n_effective: self.c * self.w,
                w: self.w,
                num_regular: params.num_regular,
                num_backup: params.num_backup,
                block_iprfs: &block_iprfs,
                subset_prf: &subset_prf,
                threads,
            },
            &mut hints,
            |_| {},
        );
        Self::parities_of(&hints)
    }

    #[cfg(feature = "tee")]
    fn run_ct(&self) -> Parities {
        let params: HintParams = self.params();
        let db: Vec<u8> = self.db_bytes();
        let subset_prf: SubsetPrf = SubsetPrf::new(&self.master_seed);
        let mut hints: HintSet = self.init(&subset_prf);
        let keys: Vec<PrfKey128> = derive_block_keys(&self.master_seed, self.c);
        let block_iprfs_ct: Vec<IprfTee> = keys
            .iter()
            .map(|key| IprfTee::new(*key, params.total_hints as u64, self.w as u64))
            .collect();
        process_entries_ct(
            CtProcessInput {
                db_bytes: &db,
                record_size: self.record_size,
                n_entries: self.n_entries,
                n_effective: self.c * self.w,
                w: self.w,
                num_regular: params.num_regular,
                num_backup: params.num_backup,
                block_iprfs_ct: &block_iprfs_ct,
                subset_prf: &subset_prf,
            },
            &mut hints,
            |_| {},
        );
        Self::parities_of(&hints)
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

        let zero: Vec<u8> = vec![0u8; self.record_size];
        let entry_at = |block: usize, hint_id: u64| -> &[u8] {
            let offset: usize = block_iprfs[block].forward(hint_id) as usize;
            let index: usize = block * self.w + offset;
            if index < self.n_entries {
                record_at(&db, self.record_size, index)
            } else {
                &zero
            }
        };
        let xor = |dst: &mut Vec<u8>, src: &[u8]| {
            assert_eq!(dst.len(), src.len());
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

        let mut regular: Vec<Vec<u8>> = vec![zero.clone(); params.num_regular];
        let mut backup_in: Vec<Vec<u8>> = vec![zero.clone(); params.num_backup];
        let mut backup_out: Vec<Vec<u8>> = vec![zero.clone(); params.num_backup];
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
        expected
            .0
            .iter()
            .filter(|p| p.iter().any(|&b| b != 0))
            .count()
            > expected.0.len() / 2,
        "the scenario must give non-trivial regular parities"
    );
    // The last byte of a record reaches the parities (it is the TAG byte of a v3 record).
    assert!(
        expected.0.iter().any(|p| p[scenario.record_size - 1] != 0),
        "the last byte of the records must be in the parities"
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
        record_size: 40,
        n_entries: 32,
        w: 8,
        c: 4,
        lambda: 2,
        invalid: None,
    });
}

#[test]
fn invalid_hints_take_no_entry() {
    let zero: Vec<u8> = vec![0u8; 40];
    let (regular_idx, backup_idx): (usize, usize) = (3, 5);
    let mut scenario = Scenario {
        master_seed: [42u8; 32],
        record_size: 40,
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
        record_size: 40,
        n_entries: 90,
        w: 16,
        c: 6,
        lambda: 40,
        invalid: None,
    });
}

/// Record sizes below one word, with a partial last word, and above the
/// 32 bytes that hint file version 1 had. The last block is padded.
#[test]
fn streaming_paths_match_the_hint_definition_for_other_record_sizes() {
    for record_size in [1usize, 5, 8, 13, 32, 45, 256] {
        let parities: Parities = assert_paths_match_the_definition(&Scenario {
            master_seed: [0x31u8; 32],
            record_size,
            n_entries: 27,
            w: 8,
            c: 4,
            lambda: 4,
            invalid: None,
        });
        assert!(
            parities.0.iter().all(|parity| parity.len() == record_size),
            "record size {record_size}"
        );
    }
}
