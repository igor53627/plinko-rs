//! Integration tests for constant-time HintInit
//!
//! Verifies that the constant-time (TEE) path produces identical results
//! to the non-constant-time path, ensuring correctness while providing
//! timing side-channel protection.
//!
//! Needs the `tee` feature.

#![cfg(feature = "tee")]

use plinko::constant_time::{ct_eq_u64, ct_lt_u64, ct_select_usize, ct_xor_32_masked};
use plinko::iprf::{Iprf, IprfTee};
use plinko::protocol::derive_block_keys;
use plinko::subsets::{backup_subset_size, regular_subset_size, SubsetPrf, INVALID_CUTOFF};
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;

struct RegularHint {
    parity: [u8; 32],
}

struct BackupHint {
    parity_in: [u8; 32],
    parity_out: [u8; 32],
}

/// Cutoffs of the regular hints (IDs `0..num_regular`) and of the backup
/// hints (IDs `num_regular..num_regular + num_backup`).
fn hint_cutoffs(
    subset_prf: &SubsetPrf,
    c: usize,
    num_regular: usize,
    num_backup: usize,
) -> (Vec<u32>, Vec<u32>) {
    let mut scratch: Vec<u32> = Vec::new();
    let regular: Vec<u32> = (0..num_regular)
        .map(|j| subset_prf.cutoff(j as u64, c, regular_subset_size(c), &mut scratch))
        .collect();
    let backup: Vec<u32> = (0..num_backup)
        .map(|b| {
            subset_prf.cutoff(
                (num_regular + b) as u64,
                c,
                backup_subset_size(c),
                &mut scratch,
            )
        })
        .collect();
    (regular, backup)
}

fn xor_32(dst: &mut [u8; 32], src: &[u8; 32]) {
    for i in 0..32 {
        dst[i] ^= src[i];
    }
}

/// Run HintInit using the fast (non-CT) path
fn hintinit_fast(
    db: &[[u8; 32]],
    master_seed: &[u8; 32],
    lambda: usize,
    w: usize,
) -> (Vec<RegularHint>, Vec<BackupHint>) {
    let n = db.len();
    let c = n.div_ceil(w);
    let n_effective = c * w;

    let num_regular = lambda * w;
    let num_backup = lambda * w;
    let total_hints = num_regular + num_backup;

    let block_keys = derive_block_keys(master_seed, c);

    let mut regular_hints: Vec<RegularHint> = (0..num_regular)
        .map(|_| RegularHint { parity: [0u8; 32] })
        .collect();
    let mut backup_hints: Vec<BackupHint> = (0..num_backup)
        .map(|_| BackupHint {
            parity_in: [0u8; 32],
            parity_out: [0u8; 32],
        })
        .collect();

    let subset_prf: SubsetPrf = SubsetPrf::new(master_seed);
    let (regular_cutoffs, backup_cutoffs) = hint_cutoffs(&subset_prf, c, num_regular, num_backup);

    let block_iprfs: Vec<Iprf> = block_keys
        .iter()
        .map(|key| Iprf::new(*key, total_hints as u64, w as u64))
        .collect();

    let mut process_entry = |i: usize, entry: [u8; 32]| {
        let block = i / w;
        let offset = i % w;

        let hint_indices = block_iprfs[block].inverse(offset as u64);

        for hint_id in hint_indices {
            let j = hint_id as usize;
            if j < num_regular {
                if subset_prf.contains(hint_id, block, regular_cutoffs[j]) {
                    xor_32(&mut regular_hints[j].parity, &entry);
                }
            } else {
                let backup_idx = j - num_regular;
                if backup_idx < num_backup {
                    let cutoff: u32 = backup_cutoffs[backup_idx];
                    if cutoff == INVALID_CUTOFF {
                        continue;
                    }
                    if subset_prf.contains(hint_id, block, cutoff) {
                        xor_32(&mut backup_hints[backup_idx].parity_in, &entry);
                    } else {
                        xor_32(&mut backup_hints[backup_idx].parity_out, &entry);
                    }
                }
            }
        }
    };

    for (i, entry) in db.iter().copied().enumerate() {
        process_entry(i, entry);
    }
    for i in n..n_effective {
        process_entry(i, [0u8; 32]);
    }

    (regular_hints, backup_hints)
}

/// Run HintInit using the constant-time (TEE) path
fn hintinit_ct(
    db: &[[u8; 32]],
    master_seed: &[u8; 32],
    lambda: usize,
    w: usize,
) -> (Vec<RegularHint>, Vec<BackupHint>) {
    let n = db.len();
    let c = n.div_ceil(w);
    let n_effective = c * w;

    let num_regular = lambda * w;
    let num_backup = lambda * w;
    let total_hints = num_regular + num_backup;

    let block_keys = derive_block_keys(master_seed, c);

    let mut regular_hints: Vec<RegularHint> = (0..num_regular)
        .map(|_| RegularHint { parity: [0u8; 32] })
        .collect();
    let mut backup_hints: Vec<BackupHint> = (0..num_backup)
        .map(|_| BackupHint {
            parity_in: [0u8; 32],
            parity_out: [0u8; 32],
        })
        .collect();

    let subset_prf: SubsetPrf = SubsetPrf::new(master_seed);
    let (regular_cutoffs, backup_cutoffs) = hint_cutoffs(&subset_prf, c, num_regular, num_backup);

    let block_iprfs_ct: Vec<IprfTee> = block_keys
        .iter()
        .map(|key| IprfTee::new(*key, total_hints as u64, w as u64))
        .collect();

    let mut process_entry = |i: usize, entry: [u8; 32]| {
        let block = i / w;
        let offset = i % w;

        let (indices, count) = block_iprfs_ct[block].inverse_ct(offset as u64);

        for (t, &idx) in indices.iter().enumerate() {
            let in_range = ct_lt_u64(t as u64, count as u64);

            let j = idx as usize;

            let is_regular = ct_lt_u64(j as u64, num_regular as u64);
            let backup_idx = j.wrapping_sub(num_regular);
            let is_valid_backup = ct_lt_u64(backup_idx as u64, num_backup as u64);
            let is_backup = (1 - is_regular) & is_valid_backup;

            let select_value = subset_prf.select(idx, block) as u64;

            let regular_idx = ct_select_usize(is_regular, j, 0);
            let in_regular_subset = ct_lt_u64(select_value, regular_cutoffs[regular_idx] as u64);

            let backup_idx_clamped = ct_select_usize(is_valid_backup, backup_idx, 0);
            let backup_cutoff = backup_cutoffs[backup_idx_clamped] as u64;
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
    };

    for (i, entry) in db.iter().copied().enumerate() {
        process_entry(i, entry);
    }
    for i in n..n_effective {
        process_entry(i, [0u8; 32]);
    }

    (regular_hints, backup_hints)
}

#[test]
fn test_ct_hintinit_matches_fast_path_tiny() {
    let master_seed = [42u8; 32];
    let lambda = 2;
    let w = 4;
    let n = 16;

    let mut rng = ChaCha20Rng::from_seed([1u8; 32]);
    let db: Vec<[u8; 32]> = (0..n)
        .map(|_| {
            let mut entry = [0u8; 32];
            rng.fill_bytes(&mut entry);
            entry
        })
        .collect();

    let (regular_fast, backup_fast) = hintinit_fast(&db, &master_seed, lambda, w);
    let (regular_ct, backup_ct) = hintinit_ct(&db, &master_seed, lambda, w);

    assert_eq!(
        regular_fast.len(),
        regular_ct.len(),
        "Regular hint count mismatch"
    );
    assert_eq!(
        backup_fast.len(),
        backup_ct.len(),
        "Backup hint count mismatch"
    );

    for (i, (fast, ct)) in regular_fast.iter().zip(regular_ct.iter()).enumerate() {
        assert_eq!(fast.parity, ct.parity, "Regular hint {} parity mismatch", i);
    }

    for (i, (fast, ct)) in backup_fast.iter().zip(backup_ct.iter()).enumerate() {
        assert_eq!(
            fast.parity_in, ct.parity_in,
            "Backup hint {} parity_in mismatch",
            i
        );
        assert_eq!(
            fast.parity_out, ct.parity_out,
            "Backup hint {} parity_out mismatch",
            i
        );
    }
}

#[test]
fn test_ct_hintinit_matches_fast_path_small() {
    let master_seed = [0xABu8; 32];
    let lambda = 2;
    let w = 4;
    let n = 32;

    let mut rng = ChaCha20Rng::from_seed([2u8; 32]);
    let db: Vec<[u8; 32]> = (0..n)
        .map(|_| {
            let mut entry = [0u8; 32];
            rng.fill_bytes(&mut entry);
            entry
        })
        .collect();

    let (regular_fast, backup_fast) = hintinit_fast(&db, &master_seed, lambda, w);
    let (regular_ct, backup_ct) = hintinit_ct(&db, &master_seed, lambda, w);

    for (i, (fast, ct)) in regular_fast.iter().zip(regular_ct.iter()).enumerate() {
        assert_eq!(fast.parity, ct.parity, "Regular hint {} parity mismatch", i);
    }

    for (i, (fast, ct)) in backup_fast.iter().zip(backup_ct.iter()).enumerate() {
        assert_eq!(
            fast.parity_in, ct.parity_in,
            "Backup hint {} parity_in mismatch",
            i
        );
        assert_eq!(
            fast.parity_out, ct.parity_out,
            "Backup hint {} parity_out mismatch",
            i
        );
    }
}

#[test]
#[ignore] // Slow due to MAX_PREIMAGES=512 iterations per entry in CT path
fn test_ct_hintinit_matches_fast_path_medium() {
    let master_seed = [0xCDu8; 32];
    let lambda = 4;
    let w = 8;
    let n = 64;

    let mut rng = ChaCha20Rng::from_seed([3u8; 32]);
    let db: Vec<[u8; 32]> = (0..n)
        .map(|_| {
            let mut entry = [0u8; 32];
            rng.fill_bytes(&mut entry);
            entry
        })
        .collect();

    let (regular_fast, backup_fast) = hintinit_fast(&db, &master_seed, lambda, w);
    let (regular_ct, backup_ct) = hintinit_ct(&db, &master_seed, lambda, w);

    for (i, (fast, ct)) in regular_fast.iter().zip(regular_ct.iter()).enumerate() {
        assert_eq!(fast.parity, ct.parity, "Regular hint {} parity mismatch", i);
    }

    for (i, (fast, ct)) in backup_fast.iter().zip(backup_ct.iter()).enumerate() {
        assert_eq!(
            fast.parity_in, ct.parity_in,
            "Backup hint {} parity_in mismatch",
            i
        );
        assert_eq!(
            fast.parity_out, ct.parity_out,
            "Backup hint {} parity_out mismatch",
            i
        );
    }
}

#[test]
fn test_ct_hintinit_zero_db() {
    let master_seed = [0u8; 32];
    let lambda = 2;
    let w = 4;
    let n = 16;

    let db: Vec<[u8; 32]> = vec![[0u8; 32]; n];

    let (regular_fast, backup_fast) = hintinit_fast(&db, &master_seed, lambda, w);
    let (regular_ct, backup_ct) = hintinit_ct(&db, &master_seed, lambda, w);

    for (fast, ct) in regular_fast.iter().zip(regular_ct.iter()) {
        assert_eq!(fast.parity, ct.parity);
        assert_eq!(fast.parity, [0u8; 32], "Zero DB should yield zero parity");
    }

    for (fast, ct) in backup_fast.iter().zip(backup_ct.iter()) {
        assert_eq!(fast.parity_in, ct.parity_in);
        assert_eq!(fast.parity_out, ct.parity_out);
    }
}

#[test]
fn test_ct_hintinit_single_nonzero_entry() {
    // With this seed, regular hints 1, 3 and 7 contain entry 7. About one seed
    // in five gives no such hint, and then the non-zero check below cannot hold.
    let master_seed = [5u8; 32];
    let lambda = 2;
    let w = 4;
    let n = 16;

    let mut db: Vec<[u8; 32]> = vec![[0u8; 32]; n];
    db[7] = [0xFFu8; 32];

    let (regular_fast, backup_fast) = hintinit_fast(&db, &master_seed, lambda, w);
    let (regular_ct, backup_ct) = hintinit_ct(&db, &master_seed, lambda, w);

    for (i, (fast, ct)) in regular_fast.iter().zip(regular_ct.iter()).enumerate() {
        assert_eq!(fast.parity, ct.parity, "Regular hint {} parity mismatch", i);
    }

    for (i, (fast, ct)) in backup_fast.iter().zip(backup_ct.iter()).enumerate() {
        assert_eq!(
            fast.parity_in, ct.parity_in,
            "Backup hint {} parity_in mismatch",
            i
        );
        assert_eq!(
            fast.parity_out, ct.parity_out,
            "Backup hint {} parity_out mismatch",
            i
        );
    }

    let has_nonzero_regular = regular_ct.iter().any(|h| h.parity != [0u8; 32]);
    assert!(
        has_nonzero_regular,
        "Should have at least one non-zero regular parity"
    );
}

#[test]
fn test_iprf_tee_inverse_ct_coverage() {
    let key = [42u8; 16];
    let domain = 256u64;
    let range = 16u64;

    let iprf = Iprf::new(key, domain, range);
    let iprf_tee = IprfTee::new(key, domain, range);

    for offset in 0..range {
        let fast_result = iprf.inverse(offset);
        let (ct_array, ct_count) = iprf_tee.inverse_ct(offset);
        let ct_result: Vec<u64> = ct_array[..ct_count].to_vec();

        let mut fast_sorted = fast_result.clone();
        let mut ct_sorted = ct_result.clone();
        fast_sorted.sort();
        ct_sorted.sort();

        assert_eq!(
            fast_sorted, ct_sorted,
            "iPRF inverse mismatch for offset {}",
            offset
        );
    }
}

#[test]
fn test_ct_xor_masked_correctness() {
    let mut dst = [0xAAu8; 32];
    let src = [0x55u8; 32];

    let mut test_dst = dst;
    ct_xor_32_masked(&mut test_dst, &src, 0);
    assert_eq!(test_dst, dst, "mask=0 should not modify dst");

    ct_xor_32_masked(&mut dst, &src, 1);
    assert_eq!(dst, [0xFFu8; 32], "mask=1 should XOR");
}
