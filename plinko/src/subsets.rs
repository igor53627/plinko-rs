//! Block subsets of hints, derived from a PRF and one cutoff per hint.
//!
//! Plinko Fig. 7 gives each regular hint a uniform subset of `c/2 + 1` blocks
//! and each backup hint a uniform subset of `c/2` blocks. This module stores
//! such a subset in 4 bytes, with the representation of RMS24 (Ren, Mughees,
//! Sun, CCS 2024, section 3.2):
//!
//! - `v[j][k] = PRF("select", j, k)` is a 32-bit value for hint `j` and block `k`.
//! - The cutoff of hint `j` is the value that has exactly `subset_size` values
//!   of `v[j]` below it.
//! - Block `k` is in the subset of hint `j` when `v[j][k] < cutoff`.
//!
//! The rank order of independent PRF values is a uniform permutation, so the
//! subset is uniform among the subsets of that size.
//!
//! If two values are equal at the boundary, no cutoff selects exactly
//! `subset_size` blocks. The hint is then invalid and its cutoff is
//! [`INVALID_CUTOFF`], as in the RMS24 reference implementation. This occurs
//! with probability of about `c / 2^33` per hint.
//!
//! # PRF
//!
//! The key is `SHA-256(master_seed || "plinko_select")[0..16]`. One AES-128
//! block gives the values of four consecutive blocks:
//! `AES-128(key, hint_be64 || (block / 4)_be64)`, read as four big-endian
//! `u32` words. Block `k` uses word `k % 4`.
//!
//! The key is client-secret: the server must not learn which blocks a hint
//! contains.

use aes::cipher::{generic_array::GenericArray, BlockEncrypt, KeyInit};
use aes::Aes128;
use sha2::{Digest, Sha256};

/// Cutoff of a hint that has no valid subset. No block is below this cutoff.
pub const INVALID_CUTOFF: u32 = 0;

const SELECT_KEY_LABEL: &[u8] = b"plinko_select";
const VALUES_PER_AES_BLOCK: usize = 4;
const BATCH_AES_BLOCKS: usize = 8;

/// Number of blocks in the subset of a regular hint: `c/2 + 1`.
#[inline]
pub fn regular_subset_size(num_blocks: usize) -> usize {
    num_blocks / 2 + 1
}

/// Number of blocks in the subset of a backup hint: `c/2`.
#[inline]
pub fn backup_subset_size(num_blocks: usize) -> usize {
    num_blocks / 2
}

/// Finds the cutoff that has exactly `select_count` of `values` below it.
///
/// Returns [`INVALID_CUTOFF`] when no such cutoff exists, because the values
/// at the boundary are equal. The order of `values` is not preserved.
///
/// # Panics
///
/// Panics if `select_count` is 0 or larger than `values.len()`.
///
/// # Examples
///
/// ```
/// use plinko::subsets::{find_cutoff, INVALID_CUTOFF};
///
/// assert_eq!(find_cutoff(&mut [50, 10, 40, 20, 30], 3), 40);
/// assert_eq!(find_cutoff(&mut [10, 20, 20, 30], 2), INVALID_CUTOFF);
/// ```
pub fn find_cutoff(values: &mut [u32], select_count: usize) -> u32 {
    assert!(
        select_count >= 1 && select_count <= values.len(),
        "select_count must be in [1, {}], got {select_count}",
        values.len()
    );

    if select_count == values.len() {
        let max: u32 = *values.iter().max().expect("values is not empty");
        return max.checked_add(1).unwrap_or(INVALID_CUTOFF);
    }

    let (below, cutoff, _) = values.select_nth_unstable(select_count);
    let cutoff: u32 = *cutoff;
    let max_below: u32 = *below.iter().max().expect("select_count >= 1");
    if max_below == cutoff {
        INVALID_CUTOFF
    } else {
        cutoff
    }
}

/// Constant-time form of [`find_cutoff`]: the same result, and `values` is
/// not changed.
///
/// The cutoff is the largest `t` that has at most `select_count` values
/// below it. The function finds `t` bit by bit, from the most significant
/// bit, and counts the values below a candidate with
/// [`crate::constant_time::ct_lt_u32`]. It reads all values 33 times in
/// index order. Its instructions and memory accesses depend only on
/// `values.len()` and `select_count`, which are public.
///
/// # Panics
///
/// Panics if `select_count` is 0 or larger than `values.len()`.
///
/// # Examples
///
/// ```
/// use plinko::subsets::{find_cutoff_ct, INVALID_CUTOFF};
///
/// assert_eq!(find_cutoff_ct(&[50, 10, 40, 20, 30], 3), 40);
/// assert_eq!(find_cutoff_ct(&[10, 20, 20, 30], 2), INVALID_CUTOFF);
/// ```
#[cfg(feature = "tee")]
pub fn find_cutoff_ct(values: &[u32], select_count: usize) -> u32 {
    use crate::constant_time::ct_lt_u32;

    assert!(
        select_count >= 1 && select_count <= values.len(),
        "select_count must be in [1, {}], got {select_count}",
        values.len()
    );
    assert!(
        u32::try_from(values.len()).is_ok(),
        "at most u32::MAX values"
    );

    if select_count == values.len() {
        // The cutoff is the maximum plus 1; a maximum of u32::MAX wraps to
        // INVALID_CUTOFF, which is 0.
        let max: u32 = values.iter().fold(0u32, |max, &value| {
            let keep_max: u32 = ct_lt_u32(value, max).wrapping_neg();
            (max & keep_max) | (value & !keep_max)
        });
        return max.wrapping_add(1);
    }

    let count_below = |cutoff: u32| -> u32 {
        values
            .iter()
            .fold(0u32, |count, &value| count + ct_lt_u32(value, cutoff))
    };
    let select_count: u32 = select_count as u32;

    let mut cutoff: u32 = 0;
    for bit in (0..u32::BITS).rev() {
        let candidate: u32 = cutoff | (1u32 << bit);
        let too_many: u32 = ct_lt_u32(select_count, count_below(candidate));
        cutoff |= (1u32 << bit) & too_many.wrapping_sub(1);
    }

    // Equal values at the boundary: fewer than `select_count` values are below.
    let has_tie: u32 = ct_lt_u32(count_below(cutoff), select_count);
    cutoff & has_tie.wrapping_sub(1)
}

/// PRF that defines the block subset of each hint.
///
/// Hint IDs are the global hint indices: regular hints are `0..num_regular`
/// and backup hint `b` is `num_regular + b`.
///
/// # Examples
///
/// ```
/// use plinko::subsets::{regular_subset_size, SubsetPrf, INVALID_CUTOFF};
///
/// let prf: SubsetPrf = SubsetPrf::new(&[7u8; 32]);
/// let num_blocks: usize = 10;
/// let mut scratch: Vec<u32> = Vec::new();
/// let cutoff: u32 = prf.cutoff(3, num_blocks, regular_subset_size(num_blocks), &mut scratch);
/// assert_ne!(cutoff, INVALID_CUTOFF);
///
/// let selected: usize = (0..num_blocks).filter(|&k| prf.contains(3, k, cutoff)).count();
/// assert_eq!(selected, 6);
/// ```
#[derive(Clone)]
pub struct SubsetPrf {
    cipher: Aes128,
}

impl SubsetPrf {
    /// Derives the select key from the master seed.
    pub fn new(master_seed: &[u8; 32]) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(master_seed);
        hasher.update(SELECT_KEY_LABEL);
        let hash = hasher.finalize();
        Self {
            cipher: Aes128::new(GenericArray::from_slice(&hash[0..16])),
        }
    }

    /// Returns the select value of `block` for hint `hint_id`.
    ///
    /// The time does not depend on `hint_id`.
    #[inline]
    pub fn select(&self, hint_id: u64, block: usize) -> u32 {
        let mut aes_block = GenericArray::from(aes_input(hint_id, block / VALUES_PER_AES_BLOCK));
        self.cipher.encrypt_block(&mut aes_block);
        word_of(aes_block.as_slice(), block % VALUES_PER_AES_BLOCK)
    }

    /// Returns `true` if `block` is in the subset that `cutoff` defines for hint `hint_id`.
    ///
    /// Always `false` for [`INVALID_CUTOFF`].
    #[inline]
    pub fn contains(&self, hint_id: u64, block: usize, cutoff: u32) -> bool {
        self.select(hint_id, block) < cutoff
    }

    /// Writes the select values of blocks `0..num_blocks` for hint `hint_id` into `values`.
    pub fn select_values(&self, hint_id: u64, num_blocks: usize, values: &mut Vec<u32>) {
        values.clear();
        values.reserve(num_blocks);

        let num_aes_blocks: usize = num_blocks.div_ceil(VALUES_PER_AES_BLOCK);
        let mut first: usize = 0;
        while first < num_aes_blocks {
            let batch_len: usize = BATCH_AES_BLOCKS.min(num_aes_blocks - first);
            let mut batch = [GenericArray::from([0u8; 16]); BATCH_AES_BLOCKS];
            for (i, aes_block) in batch.iter_mut().enumerate().take(batch_len) {
                *aes_block = GenericArray::from(aes_input(hint_id, first + i));
            }
            self.cipher.encrypt_blocks(&mut batch[..batch_len]);

            for aes_block in &batch[..batch_len] {
                for word in 0..VALUES_PER_AES_BLOCK {
                    if values.len() < num_blocks {
                        values.push(word_of(aes_block.as_slice(), word));
                    }
                }
            }
            first += batch_len;
        }
    }

    /// Computes the cutoff of hint `hint_id` that selects `subset_size` of `num_blocks` blocks.
    ///
    /// `scratch` is a reusable buffer. Returns [`INVALID_CUTOFF`] for an invalid hint.
    pub fn cutoff(
        &self,
        hint_id: u64,
        num_blocks: usize,
        subset_size: usize,
        scratch: &mut Vec<u32>,
    ) -> u32 {
        self.select_values(hint_id, num_blocks, scratch);
        find_cutoff(scratch, subset_size)
    }

    /// Computes the same cutoff as [`SubsetPrf::cutoff`] in constant time.
    ///
    /// The select values are AES outputs of `hint_id`, and the cutoff comes
    /// from [`find_cutoff_ct`]. The work depends only on `num_blocks` and
    /// `subset_size`.
    #[cfg(feature = "tee")]
    pub fn cutoff_ct(
        &self,
        hint_id: u64,
        num_blocks: usize,
        subset_size: usize,
        scratch: &mut Vec<u32>,
    ) -> u32 {
        self.select_values(hint_id, num_blocks, scratch);
        find_cutoff_ct(scratch, subset_size)
    }
}

#[inline]
fn aes_input(hint_id: u64, group: usize) -> [u8; 16] {
    let mut input = [0u8; 16];
    input[0..8].copy_from_slice(&hint_id.to_be_bytes());
    input[8..16].copy_from_slice(&(group as u64).to_be_bytes());
    input
}

#[inline]
fn word_of(aes_block: &[u8], word: usize) -> u32 {
    let start: usize = word * 4;
    u32::from_be_bytes(
        aes_block[start..start + 4]
            .try_into()
            .expect("an AES block holds four u32 words"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn count_below(values: &[u32], cutoff: u32) -> usize {
        values.iter().filter(|&&v| v < cutoff).count()
    }

    #[test]
    fn find_cutoff_selects_exactly_the_requested_count() {
        let values: [u32; 5] = [50, 10, 40, 20, 30];
        for (select_count, expected) in [(1usize, 20u32), (2, 30), (3, 40), (4, 50), (5, 51)] {
            let cutoff: u32 = find_cutoff(&mut values.clone(), select_count);
            assert_eq!(cutoff, expected, "select_count={select_count}");
            assert_eq!(count_below(&values, cutoff), select_count);
        }
    }

    #[test]
    fn find_cutoff_rejects_a_tie_at_the_boundary_only() {
        assert_eq!(find_cutoff(&mut [10, 20, 20, 30], 2), INVALID_CUTOFF);
        assert_eq!(find_cutoff(&mut [20, 30, 20, 10], 2), INVALID_CUTOFF);

        // Ties that do not cross the boundary are valid.
        assert_eq!(find_cutoff(&mut [10, 10, 20, 30], 2), 20);
        assert_eq!(find_cutoff(&mut [10, 20, 30, 30], 2), 30);
        assert_eq!(find_cutoff(&mut [10, 20, 30, 30], 1), 20);
        assert_eq!(find_cutoff(&mut [0, 0, 5, 9], 2), 5);
    }

    #[test]
    fn find_cutoff_select_all_handles_the_maximum_value() {
        assert_eq!(find_cutoff(&mut [5, 7], 2), 8);
        assert_eq!(find_cutoff(&mut [u32::MAX, 1], 2), INVALID_CUTOFF);
    }

    #[test]
    fn select_matches_an_independent_aes_computation() {
        // Reference: openssl enc -aes-128-ecb -nopad with
        // key = SHA-256(seed || "plinko_select")[0..16], seed = 00 01 .. 1f,
        // plaintext = hint_be64 || group_be64 for hint 0x0102030405060708, group 2.
        let mut seed = [0u8; 32];
        for (i, byte) in seed.iter_mut().enumerate() {
            *byte = i as u8;
        }
        let prf: SubsetPrf = SubsetPrf::new(&seed);
        let hint_id: u64 = 0x0102_0304_0506_0708;
        let expected: [u32; 4] = [0x3250_1eae, 0x4102_e47b, 0x8668_169a, 0x00c4_c0cb];
        for (word, expected_value) in expected.iter().enumerate() {
            assert_eq!(
                prf.select(hint_id, 8 + word),
                *expected_value,
                "word={word}"
            );
        }
    }

    #[test]
    fn select_values_agrees_with_select_at_batch_boundaries() {
        let prf: SubsetPrf = SubsetPrf::new(&[3u8; 32]);
        let mut values: Vec<u32> = Vec::new();
        for num_blocks in [1usize, 2, 3, 4, 5, 31, 32, 33, 70] {
            prf.select_values(11, num_blocks, &mut values);
            assert_eq!(values.len(), num_blocks);
            for (block, value) in values.iter().enumerate() {
                assert_eq!(
                    *value,
                    prf.select(11, block),
                    "c={num_blocks} block={block}"
                );
            }
        }
    }

    #[test]
    fn subsets_have_the_exact_size_for_regular_and_backup_hints() {
        let prf: SubsetPrf = SubsetPrf::new(&[9u8; 32]);
        let mut scratch: Vec<u32> = Vec::new();
        for num_blocks in [2usize, 4, 6, 10, 64, 100, 1000] {
            for hint_id in 0..200u64 {
                for subset_size in [
                    regular_subset_size(num_blocks),
                    backup_subset_size(num_blocks),
                ] {
                    let cutoff: u32 = prf.cutoff(hint_id, num_blocks, subset_size, &mut scratch);
                    assert_ne!(cutoff, INVALID_CUTOFF, "c={num_blocks} hint={hint_id}");
                    let selected: usize = (0..num_blocks)
                        .filter(|&block| prf.contains(hint_id, block, cutoff))
                        .count();
                    assert_eq!(selected, subset_size, "c={num_blocks} hint={hint_id}");
                }
            }
        }
    }

    #[test]
    fn invalid_cutoff_selects_no_block() {
        let prf: SubsetPrf = SubsetPrf::new(&[1u8; 32]);
        assert!((0..256).all(|block| !prf.contains(5, block, INVALID_CUTOFF)));
    }

    #[test]
    fn each_block_is_selected_by_about_half_of_the_backup_hints() {
        let prf: SubsetPrf = SubsetPrf::new(&[0x5Au8; 32]);
        let num_blocks: usize = 16;
        let num_hints: u64 = 4000;
        let mut scratch: Vec<u32> = Vec::new();
        let mut selected_by: Vec<u32> = vec![0; num_blocks];
        let mut pair_together: u32 = 0;

        for hint_id in 0..num_hints {
            let cutoff: u32 = prf.cutoff(
                hint_id,
                num_blocks,
                backup_subset_size(num_blocks),
                &mut scratch,
            );
            for (block, count) in selected_by.iter_mut().enumerate() {
                *count += u32::from(prf.contains(hint_id, block, cutoff));
            }
            // Blocks 0 and 1 share an AES block; they must not move together.
            pair_together +=
                u32::from(prf.contains(hint_id, 0, cutoff) == prf.contains(hint_id, 1, cutoff));
        }

        // Mean 2000, standard deviation about 32.
        for (block, count) in selected_by.iter().enumerate() {
            assert!((1800..=2200).contains(count), "block={block} count={count}");
        }
        // P[same side] = 7/15 for a uniform half; mean 1867.
        assert!(
            (1650..=2080).contains(&pair_together),
            "pair_together={pair_together}"
        );
    }

    #[test]
    fn subsets_differ_between_hints_and_between_seeds() {
        let num_blocks: usize = 64;
        let subset_of = |prf: &SubsetPrf, hint_id: u64| -> Vec<usize> {
            let mut scratch: Vec<u32> = Vec::new();
            let cutoff: u32 = prf.cutoff(
                hint_id,
                num_blocks,
                backup_subset_size(num_blocks),
                &mut scratch,
            );
            (0..num_blocks)
                .filter(|&block| prf.contains(hint_id, block, cutoff))
                .collect()
        };

        let prf_a: SubsetPrf = SubsetPrf::new(&[1u8; 32]);
        let prf_b: SubsetPrf = SubsetPrf::new(&[2u8; 32]);
        assert_eq!(subset_of(&prf_a, 0), subset_of(&prf_a, 0));
        assert_ne!(subset_of(&prf_a, 0), subset_of(&prf_a, 1));
        assert_ne!(subset_of(&prf_a, 0), subset_of(&prf_b, 0));
    }

    /// Equal values at the boundary, values at the ends of the `u32` range,
    /// and each select count of each length.
    #[cfg(feature = "tee")]
    #[test]
    fn find_cutoff_ct_equals_find_cutoff() {
        assert_eq!(find_cutoff_ct(&[50, 10, 40, 20, 30], 3), 40);
        assert_eq!(find_cutoff_ct(&[10, 20, 20, 30], 2), INVALID_CUTOFF);
        assert_eq!(find_cutoff_ct(&[7], 1), 8);
        assert_eq!(find_cutoff_ct(&[3, u32::MAX], 2), INVALID_CUTOFF);
        assert_eq!(find_cutoff_ct(&[u32::MAX, 3, u32::MAX], 1), u32::MAX);
        assert_eq!(find_cutoff_ct(&[0, 0, 5], 1), INVALID_CUTOFF);

        let mut state: u64 = 0x1234_5678_9ABC_DEF1;
        let mut next = || -> u64 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let (mut valid, mut invalid): (usize, usize) = (0, 0);
        for case in 0..3000usize {
            let len: usize = 1 + case % 40;
            let values: Vec<u32> = (0..len)
                .map(|_| match case % 4 {
                    0 => (next() % 6) as u32,
                    1 => u32::MAX - (next() % 4) as u32,
                    2 => [0, 0x8000_0000, u32::MAX][(next() % 3) as usize],
                    _ => next() as u32,
                })
                .collect();
            for select_count in 1..=len {
                let expected: u32 = find_cutoff(&mut values.clone(), select_count);
                assert_eq!(
                    find_cutoff_ct(&values, select_count),
                    expected,
                    "{values:?}, select_count {select_count}"
                );
                if expected == INVALID_CUTOFF {
                    invalid += 1;
                } else {
                    valid += 1;
                    assert_eq!(count_below(&values, expected), select_count);
                }
            }
        }
        assert!(
            valid > 10_000 && invalid > 10_000,
            "{valid} valid, {invalid} invalid"
        );
    }

    #[cfg(feature = "tee")]
    #[test]
    fn cutoff_ct_equals_cutoff_for_prf_values() {
        let prf: SubsetPrf = SubsetPrf::new(&[0x77u8; 32]);
        let mut scratch: Vec<u32> = Vec::new();
        for (num_blocks, hints) in [(2usize, 40u64), (10, 40), (64, 40), (1000, 20), (27_988, 4)] {
            for subset_size in [
                regular_subset_size(num_blocks),
                backup_subset_size(num_blocks),
            ] {
                for hint_id in 0..hints {
                    let expected: u32 = prf.cutoff(hint_id, num_blocks, subset_size, &mut scratch);
                    assert_eq!(
                        prf.cutoff_ct(hint_id, num_blocks, subset_size, &mut scratch),
                        expected,
                        "num_blocks {num_blocks}, subset_size {subset_size}, hint {hint_id}"
                    );
                }
            }
        }
    }
}
