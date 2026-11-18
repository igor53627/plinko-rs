//! Tests of HintInit against the hint definition of Plinko Fig. 7.
//!
//! The standard streaming phase is always checked. The constant-time form is
//! checked with the `tee` feature.

use super::*;
use crate::hints::seed_check;
#[cfg(feature = "tee")]
use crate::iprf::IprfTee;
use crate::iprf::PrfKey128;
use crate::protocol::derive_block_keys;

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
    /// Number of regular hints; the scenario has as many backup hints.
    fn num_regular(&self) -> usize {
        self.lambda * self.w
    }

    fn total_hints(&self) -> usize {
        2 * self.num_regular()
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
        let (mut regular, mut backup) = hint_cutoffs(
            subset_prf,
            self.c,
            self.num_regular(),
            self.num_regular(),
            1,
        );
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

    fn block_iprfs(&self) -> Vec<Iprf> {
        let keys: Vec<PrfKey128> = derive_block_keys(&self.master_seed, self.c);
        keys.iter()
            .map(|key| Iprf::new(*key, self.total_hints() as u64, self.w as u64))
            .collect()
    }

    fn run_fast(&self, threads: usize) -> Parities {
        let db: Vec<u8> = self.db_bytes();
        let subset_prf: SubsetPrf = SubsetPrf::new(&self.master_seed);
        let mut hints: HintSet = self.init(&subset_prf);
        let last_report = AtomicUsize::new(0);
        fill_parities(
            &mut hints,
            &db,
            &self.block_iprfs(),
            &subset_prf,
            threads,
            |done| {
                last_report.fetch_max(done, Ordering::Relaxed);
            },
        );
        assert_eq!(
            last_report.load(Ordering::Relaxed),
            self.c * self.w,
            "the progress reports must end at the padded entry count"
        );
        Self::parities_of(&hints)
    }

    #[cfg(feature = "tee")]
    fn run_ct(&self, threads: usize) -> Parities {
        let db: Vec<u8> = self.db_bytes();
        let subset_prf: SubsetPrf = SubsetPrf::new(&self.master_seed);
        let mut hints: HintSet = self.init(&subset_prf);
        let keys: Vec<PrfKey128> = derive_block_keys(&self.master_seed, self.c);
        let block_iprfs_ct: Vec<IprfTee> = keys
            .iter()
            .map(|key| IprfTee::new(*key, self.total_hints() as u64, self.w as u64))
            .collect();
        fill_parities_ct(
            &mut hints,
            &db,
            &block_iprfs_ct,
            &subset_prf,
            threads,
            |_| {},
        );
        Self::parities_of(&hints)
    }

    /// Plinko Fig. 7, hint by hint: hint `j` takes the entry at offset
    /// `iF.F(k_alpha, j)` of each block `alpha`. Uses `forward` only, so it
    /// does not share the inverse or the streaming loop with the code under test.
    fn run_definition(&self) -> Parities {
        let num_regular: usize = self.num_regular();
        let db: Vec<u8> = self.db_bytes();
        let subset_prf: SubsetPrf = SubsetPrf::new(&self.master_seed);
        let block_iprfs: Vec<Iprf> = self.block_iprfs();

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

        let mut regular: Vec<Vec<u8>> = vec![zero.clone(); num_regular];
        let mut backup_in: Vec<Vec<u8>> = vec![zero.clone(); num_regular];
        let mut backup_out: Vec<Vec<u8>> = vec![zero.clone(); num_regular];
        let invalid_ids: Vec<usize> = self
            .invalid
            .map(|(regular_idx, backup_idx)| vec![regular_idx, num_regular + backup_idx])
            .unwrap_or_default();
        for j in (0..self.total_hints()).filter(|j| !invalid_ids.contains(j)) {
            let hint_id: u64 = j as u64;
            if j < num_regular {
                let member: Vec<bool> = subset_of(hint_id, self.c / 2 + 1);
                for block in (0..self.c).filter(|&block| member[block]) {
                    xor(&mut regular[j], entry_at(block, hint_id));
                }
            } else {
                let b: usize = j - num_regular;
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

/// Fewer threads than blocks, and more threads than blocks.
const OTHER_THREAD_COUNTS: [usize; 3] = [2, 3, 16];

fn assert_paths_match_the_definition(scenario: &Scenario) -> Parities {
    let expected: Parities = scenario.run_definition();
    let fast: Parities = scenario.run_fast(1);
    for threads in OTHER_THREAD_COUNTS {
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
        let ct: Parities = scenario.run_ct(1);
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

/// Six blocks, the last one partly padding, a record with a partial last
/// word, and one invalid hint of each kind.
#[cfg(feature = "tee")]
#[test]
fn constant_time_path_gives_the_same_parities_for_every_thread_count() {
    let scenario = Scenario {
        master_seed: [0x63u8; 32],
        record_size: 13,
        n_entries: 43,
        w: 8,
        c: 6,
        lambda: 2,
        invalid: Some((1, 2)),
    };
    let expected: Parities = scenario.run_definition();
    assert!(scenario.run_ct(1) == expected, "1 thread");
    for threads in OTHER_THREAD_COUNTS {
        assert!(scenario.run_ct(threads) == expected, "{threads} threads");
    }
}

/// A cutoff must select exactly the subset size of its kind of hint, for the
/// hint ID of the hint: `j` for a regular hint and `num_regular + b` for a
/// backup hint. The part borders of the workers fall inside both lists.
#[test]
fn cutoffs_select_the_subset_size_for_every_thread_count() {
    let subset_prf: SubsetPrf = SubsetPrf::new(&[0x19u8; 32]);
    let (num_blocks, num_regular, num_backup): (usize, usize, usize) = (64, 7, 5);
    let selected = |hint_id: usize, cutoff: u32| -> usize {
        (0..num_blocks)
            .filter(|&block| subset_prf.contains(hint_id as u64, block, cutoff))
            .count()
    };

    let one_thread: HintCutoffs = hint_cutoffs(&subset_prf, num_blocks, num_regular, num_backup, 1);
    assert_eq!(
        (one_thread.0.len(), one_thread.1.len()),
        (num_regular, num_backup)
    );
    for (j, &cutoff) in one_thread.0.iter().enumerate() {
        assert_eq!(selected(j, cutoff), num_blocks / 2 + 1, "regular hint {j}");
    }
    for (b, &cutoff) in one_thread.1.iter().enumerate() {
        assert_eq!(
            selected(num_regular + b, cutoff),
            num_blocks / 2,
            "backup hint {b}"
        );
    }

    // 0 counts as 1; 16 is more than the number of hints.
    for threads in [0usize, 2, 3, 4, 16] {
        assert_eq!(
            hint_cutoffs(&subset_prf, num_blocks, num_regular, num_backup, threads),
            one_thread,
            "{threads} threads"
        );
    }
    assert_eq!(
        hint_cutoffs(&subset_prf, num_blocks, 0, 0, 4),
        (Vec::new(), Vec::new())
    );
}

#[test]
fn record_at_returns_the_whole_record() {
    let db: Vec<u8> = (0u8..21).collect();
    assert_eq!(record_at(&db, 7, 0), [0, 1, 2, 3, 4, 5, 6]);
    assert_eq!(record_at(&db, 7, 2), [14, 15, 16, 17, 18, 19, 20]);
    assert_eq!(record_at(&db, 21, 0), db.as_slice());
}

/// The constant-time cutoffs must equal the standard cutoffs.
#[cfg(feature = "tee")]
#[test]
fn constant_time_cutoffs_equal_the_standard_cutoffs() {
    let subset_prf: SubsetPrf = SubsetPrf::new(&[0x19u8; 32]);
    for (num_blocks, num_regular, num_backup) in [(64usize, 7usize, 5usize), (2, 3, 3), (10, 40, 1)]
    {
        let expected: HintCutoffs =
            hint_cutoffs(&subset_prf, num_blocks, num_regular, num_backup, 1);
        for threads in [0usize, 1, 3, 16] {
            assert_eq!(
                hint_cutoffs_ct(&subset_prf, num_blocks, num_regular, num_backup, threads),
                expected,
                "{num_blocks} blocks, {threads} threads"
            );
        }
    }
}

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

        #[cfg(feature = "tee")]
        {
            parities.xor_masked(2, &words, 0);
            assert_eq!(parity_of(2), zero, "record size {record_size}, mask 0");
            parities.xor_masked(2, &words, 1);
            assert_eq!(parity_of(2), entry, "record size {record_size}, mask 1");
        }
    }
}

#[test]
#[should_panic(expected = "the database must hold num_entries records")]
fn a_database_of_another_length_is_rejected() {
    let scenario = Scenario {
        master_seed: [42u8; 32],
        record_size: 40,
        n_entries: 32,
        w: 8,
        c: 4,
        lambda: 2,
        invalid: None,
    };
    let subset_prf: SubsetPrf = SubsetPrf::new(&scenario.master_seed);
    let mut hints: HintSet = scenario.init(&subset_prf);
    let db: Vec<u8> = scenario.db_bytes();
    fill_parities(
        &mut hints,
        &db[..db.len() - 1],
        &scenario.block_iprfs(),
        &subset_prf,
        1,
        |_| {},
    );
}
