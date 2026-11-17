//! Exact derandomized Binomial(count, 1/2) sampling for PMNS.
//!
//! The PMNS tree (paper Fig. 4) splits the balls of a node between its two
//! children with `Binomial(count, p; F(k, node))`. When the number of bins is a
//! power of two, every split has `p = 1/2`, and the sample is the number of
//! set bits among `count` fair coins. This module draws the coins from an
//! AES-128 counter-mode keystream that is unique to the node.
//!
//! The sampler uses integer arithmetic only. It gives the same result on every
//! platform, and its output is exactly Binomial(count, 1/2) when the keystream
//! is uniform. This is the case that the proof of Theorem 4.7 covers.
//!
//! # Keystream format
//!
//! Block `i` of a node is `AES-128(key, low_be32 || high_be32 || i_be64)`, with
//! `i` starting at 0. A block is read as a big-endian 128-bit integer. Coins
//! are taken block by block, most significant bit first. The sample is the
//! number of set bits among the first `count` coins.
//!
//! The caller must use a key that is not used for another purpose, and must
//! not use one key for two trees with different shapes. `Iprf` derives the key
//! from the block key, the domain and the range.

use aes::cipher::{generic_array::GenericArray, BlockEncrypt};
use aes::Aes128;

#[cfg(feature = "tee")]
use crate::constant_time::{ct_eq_u64, ct_min_u64, ct_saturating_sub_u64};

/// Number of coins in one keystream block.
const BLOCK_BITS: u64 = 128;

/// Blocks encrypted per call, so that the AES rounds of different blocks overlap.
const BATCH_BLOCKS: usize = 8;

/// A PMNS tree node, identified by the inclusive range of bins that it covers.
///
/// # Examples
///
/// ```
/// use plinko::binomial::PmnsNode;
///
/// let root: PmnsNode = PmnsNode { low: 0, high: 65_535 };
/// assert_eq!(root.high - root.low + 1, 65_536);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PmnsNode {
    /// First bin of the node.
    pub low: u32,
    /// Last bin of the node.
    pub high: u32,
}

/// Returns keystream blocks `first_block..first_block + BATCH_BLOCKS` of `node`.
fn keystream_batch(cipher: &Aes128, node: PmnsNode, first_block: u64) -> [u128; BATCH_BLOCKS] {
    let mut blocks = [GenericArray::from([0u8; 16]); BATCH_BLOCKS];
    for (i, block) in blocks.iter_mut().enumerate() {
        block[0..4].copy_from_slice(&node.low.to_be_bytes());
        block[4..8].copy_from_slice(&node.high.to_be_bytes());
        block[8..16].copy_from_slice(&(first_block + i as u64).to_be_bytes());
    }
    cipher.encrypt_blocks(&mut blocks);

    let mut words: [u128; BATCH_BLOCKS] = [0u128; BATCH_BLOCKS];
    for (word, block) in words.iter_mut().zip(blocks.iter()) {
        *word = u128::from_be_bytes((*block).into());
    }
    words
}

/// Samples Binomial(count, 1/2) for `node`.
///
/// Returns the number of set bits among the first `count` keystream coins of
/// the node. The running time depends on `count`. The `tee` feature has
/// `coin_binomial_ct`, whose running time depends on a public bound only.
///
/// # Examples
///
/// ```
/// use aes::cipher::KeyInit;
/// use aes::Aes128;
/// use plinko::binomial::{coin_binomial, PmnsNode};
///
/// let cipher: Aes128 = Aes128::new(&[7u8; 16].into());
/// let node: PmnsNode = PmnsNode { low: 0, high: 15 };
///
/// let left: u64 = coin_binomial(&cipher, node, 1000);
/// assert!(left <= 1000);
/// assert_eq!(left, coin_binomial(&cipher, node, 1000));
/// assert_eq!(coin_binomial(&cipher, node, 0), 0);
/// ```
pub fn coin_binomial(cipher: &Aes128, node: PmnsNode, count: u64) -> u64 {
    let mut ones: u64 = 0;
    let mut remaining: u64 = count;
    let mut next_block: u64 = 0;

    while remaining > 0 {
        for word in keystream_batch(cipher, node, next_block) {
            let kept: u64 = remaining.min(BLOCK_BITS);
            if kept == 0 {
                break;
            }
            ones += u64::from((word >> (BLOCK_BITS - kept)).count_ones());
            remaining -= kept;
        }
        next_block += BATCH_BLOCKS as u64;
    }

    ones
}

/// Samples Binomial(count, 1/2) for `node` without a `count`-dependent running time.
///
/// Returns the same value as [`coin_binomial`]. The function always encrypts
/// the blocks that `max_count` coins need, and selects the coins of the first
/// `count` with masks. The work depends only on `max_count`, which must be a
/// public bound such as the iPRF domain.
///
/// `count` must not exceed `max_count`.
///
/// # Examples
///
/// ```
/// use aes::cipher::KeyInit;
/// use aes::Aes128;
/// use plinko::binomial::{coin_binomial, coin_binomial_ct, PmnsNode};
///
/// let cipher: Aes128 = Aes128::new(&[7u8; 16].into());
/// let node: PmnsNode = PmnsNode { low: 0, high: 15 };
///
/// assert_eq!(
///     coin_binomial_ct(&cipher, node, 300, 4096),
///     coin_binomial(&cipher, node, 300)
/// );
/// ```
#[cfg(feature = "tee")]
pub fn coin_binomial_ct(cipher: &Aes128, node: PmnsNode, count: u64, max_count: u64) -> u64 {
    debug_assert!(
        count <= max_count,
        "coin_binomial_ct: count {count} exceeds max_count {max_count}"
    );

    let total_blocks: u64 = max_count.div_ceil(BLOCK_BITS);
    let mut ones: u64 = 0;
    let mut next_block: u64 = 0;

    while next_block < total_blocks {
        let words: [u128; BATCH_BLOCKS] = keystream_batch(cipher, node, next_block);
        for (i, word) in words.iter().enumerate() {
            let consumed: u64 = (next_block + i as u64) * BLOCK_BITS;
            let kept: u64 = ct_min_u64(ct_saturating_sub_u64(count, consumed), BLOCK_BITS);
            let kept_high: u64 = ct_min_u64(kept, 64);
            let kept_low: u64 = kept - kept_high;

            let high_half: u64 = (word >> 64) as u64;
            let low_half: u64 = *word as u64;
            ones += u64::from((high_half & top_bits_mask(kept_high)).count_ones());
            ones += u64::from((low_half & top_bits_mask(kept_low)).count_ones());
        }
        next_block += BATCH_BLOCKS as u64;
    }

    ones
}

/// Mask with the `kept` most significant bits set, for `kept` in `0..=64`.
#[cfg(feature = "tee")]
pub(crate) fn top_bits_mask(kept: u64) -> u64 {
    let nonzero: u64 = 1 ^ ct_eq_u64(kept, 0);
    (u64::MAX << ((64 - kept) & 63)) & nonzero.wrapping_neg()
}

#[cfg(test)]
mod tests {
    use super::*;
    use aes::cipher::KeyInit;

    fn test_cipher() -> Aes128 {
        Aes128::new(&GenericArray::from([0x42u8; 16]))
    }

    /// The keystream format from the module documentation, one bit at a time.
    fn coin_binomial_bitwise(cipher: &Aes128, node: PmnsNode, count: u64) -> u64 {
        let mut ones: u64 = 0;
        for coin in 0..count {
            let mut block = GenericArray::from([0u8; 16]);
            block[0..4].copy_from_slice(&node.low.to_be_bytes());
            block[4..8].copy_from_slice(&node.high.to_be_bytes());
            block[8..16].copy_from_slice(&(coin / 128).to_be_bytes());
            cipher.encrypt_block(&mut block);

            let position: usize = (coin % 128) as usize;
            let byte: u8 = block[position / 8];
            ones += u64::from((byte >> (7 - position % 8)) & 1);
        }
        ones
    }

    /// Counts on both sides of the block, half-block and batch boundaries.
    const BOUNDARY_COUNTS: [u64; 16] = [
        0, 1, 2, 63, 64, 65, 127, 128, 129, 191, 192, 193, 1023, 1024, 1025, 3000,
    ];

    #[test]
    fn test_matches_bitwise_keystream_definition() {
        let cipher: Aes128 = test_cipher();
        let node: PmnsNode = PmnsNode { low: 3, high: 12 };

        for count in BOUNDARY_COUNTS {
            assert_eq!(
                coin_binomial(&cipher, node, count),
                coin_binomial_bitwise(&cipher, node, count),
                "count={count}"
            );
        }
    }

    #[test]
    #[cfg(feature = "tee")]
    fn test_ct_variant_matches_for_every_bound() {
        let cipher: Aes128 = test_cipher();
        let node: PmnsNode = PmnsNode { low: 0, high: 255 };

        for count in BOUNDARY_COUNTS {
            let expected: u64 = coin_binomial(&cipher, node, count);
            for max_count in [count, count + 1, count + 127, 4096, 65_536] {
                assert_eq!(
                    coin_binomial_ct(&cipher, node, count, max_count),
                    expected,
                    "count={count} max_count={max_count}"
                );
            }
        }
    }

    #[test]
    #[cfg(feature = "tee")]
    fn test_top_bits_mask_selects_leading_bits() {
        assert_eq!(top_bits_mask(0), 0);
        assert_eq!(top_bits_mask(1), 1 << 63);
        assert_eq!(top_bits_mask(64), u64::MAX);
        for kept in 0..=64u64 {
            let mask: u64 = top_bits_mask(kept);
            assert_eq!(u64::from(mask.count_ones()), kept);
            assert_eq!(u64::from(mask.leading_ones()), kept);
        }
    }

    /// Every (node, block index) pair must give a different AES input.
    #[test]
    fn test_keystream_blocks_differ_between_nodes_and_counters() {
        let cipher: Aes128 = test_cipher();
        let nodes: [PmnsNode; 4] = [
            PmnsNode { low: 0, high: 1 },
            PmnsNode { low: 1, high: 0 },
            PmnsNode { low: 0, high: 256 },
            PmnsNode { low: 1, high: 1 },
        ];

        let mut words: Vec<u128> = Vec::new();
        for node in nodes {
            words.extend(keystream_batch(&cipher, node, 0));
            words.extend(keystream_batch(&cipher, node, BATCH_BLOCKS as u64));
        }

        let total: usize = words.len();
        words.sort_unstable();
        words.dedup();
        assert_eq!(words.len(), total);
    }

    /// Binomial(n, 1/2) CDF from the PMF ratio recurrence, started at the mode.
    fn reference_half_binomial_cdf(n: u64) -> Vec<f64> {
        let size: usize = n as usize + 1;
        let mode: usize = size / 2;

        let mut weights: Vec<f64> = vec![0.0; size];
        weights[mode] = 1.0;
        for k in mode..n as usize {
            weights[k + 1] = weights[k] * ((n as usize - k) as f64 / (k + 1) as f64);
        }
        for k in (1..=mode).rev() {
            weights[k - 1] = weights[k] * (k as f64 / (n as usize - k + 1) as f64);
        }

        let total: f64 = weights.iter().sum();
        let mut cumulative: f64 = 0.0;
        weights
            .iter()
            .map(|weight| {
                cumulative += weight / total;
                cumulative
            })
            .collect()
    }

    /// Largest gap between the Binomial(count, 1/2) CDF and the empirical CDF
    /// of one sample from each of `samples` different nodes.
    fn empirical_cdf_gap(count: u64, samples: u32) -> f64 {
        let cipher: Aes128 = test_cipher();
        let reference: Vec<f64> = reference_half_binomial_cdf(count);

        let mut histogram: Vec<u32> = vec![0; count as usize + 1];
        for low in 0..samples {
            let node: PmnsNode = PmnsNode { low, high: low + 1 };
            histogram[coin_binomial(&cipher, node, count) as usize] += 1;
        }

        let mut seen: u32 = 0;
        let mut gap: f64 = 0.0;
        for (k, hits) in histogram.iter().enumerate() {
            seen += hits;
            gap = gap.max((f64::from(seen) / f64::from(samples) - reference[k]).abs());
        }
        gap
    }

    /// Kolmogorov-Smirnov check. With 4000 samples the statistic exceeds 0.04
    /// with probability below 1e-5 when the sampler is correct. A sampler with
    /// a mean that is wrong by 5 percent gives a gap above 0.8 at count 4000.
    #[test]
    fn test_samples_follow_binomial_half_distribution() {
        let samples: u32 = 4000;
        for count in [1u64, 2, 7, 100, 127, 128, 1000, 1025, 4000, 12_000] {
            let gap: f64 = empirical_cdf_gap(count, samples);
            assert!(gap < 0.04, "count={count}: CDF gap {gap}");
        }
    }
}
