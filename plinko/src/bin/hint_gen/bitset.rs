/// A packed bitset over block indices, designed for constant-time membership
/// queries that avoid data-dependent branching.
pub struct BlockBitset {
    /// Packed 64-bit words storing one bit per block index.
    bits: Vec<u64>,
    /// Total number of blocks this bitset covers.
    num_blocks: usize,
}

impl BlockBitset {
    /// Creates a new all-zeros bitset that can hold `num_blocks` bits.
    pub fn new(num_blocks: usize) -> Self {
        let num_words = num_blocks.div_ceil(64);
        Self {
            bits: vec![0u64; num_words],
            num_blocks,
        }
    }

    /// Builds a bitset with the given block indices set to 1.
    pub fn from_sorted_blocks(blocks: &[usize], num_blocks: usize) -> Self {
        let mut bitset = Self::new(num_blocks);
        for &block in blocks {
            if block < num_blocks {
                bitset.bits[block / 64] |= 1u64 << (block % 64);
            }
        }
        bitset
    }

    /// Returns true if `block` is in the set.
    #[inline]
    pub fn contains(&self, block: usize) -> bool {
        self.contains_ct(block) == 1
    }

    /// Returns 1 if `block` is in the set, 0 otherwise, in constant time.
    #[inline]
    pub fn contains_ct(&self, block: usize) -> u64 {
        if block >= self.num_blocks {
            return 0;
        }
        let word_idx = block / 64;
        let bit_idx = block % 64;
        (self.bits[word_idx] >> bit_idx) & 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::seq::index::sample;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    #[test]
    fn test_bitset_membership() {
        let blocks = vec![1, 3, 5, 7, 9];
        let bitset = BlockBitset::from_sorted_blocks(&blocks, 10);
        assert_eq!(bitset.contains_ct(0), 0);
        assert_eq!(bitset.contains_ct(1), 1);
        assert_eq!(bitset.contains_ct(2), 0);
        assert_eq!(bitset.contains_ct(3), 1);
        assert_eq!(bitset.contains_ct(5), 1);
        assert_eq!(bitset.contains_ct(9), 1);
        assert_eq!(bitset.contains_ct(10), 0);
    }

    #[test]
    fn test_bitset_exhaustive_membership() {
        let c = 128;
        let mut rng = ChaCha20Rng::from_seed([42u8; 32]);
        let subset_size = c / 2 + 1;
        let mut blocks = sample(&mut rng, c, subset_size).into_vec();
        blocks.sort_unstable();

        let bitset = BlockBitset::from_sorted_blocks(&blocks, c);

        for idx in 0..c {
            let vec_result = blocks.binary_search(&idx).is_ok();
            let bitset_result = bitset.contains(idx);
            assert_eq!(
                vec_result, bitset_result,
                "Mismatch at index {idx}: vec={vec_result}, bitset={bitset_result}"
            );
            assert_eq!(
                bitset.contains_ct(idx),
                if vec_result { 1 } else { 0 },
                "CT mismatch at index {idx}"
            );
        }
    }

    #[test]
    fn test_bitset_edge_cases() {
        // Empty subset
        let bitset = BlockBitset::from_sorted_blocks(&[], 64);
        for i in 0..64 {
            assert!(!bitset.contains(i), "Empty bitset should contain nothing");
        }

        // Full subset
        let all: Vec<usize> = (0..64).collect();
        let bitset = BlockBitset::from_sorted_blocks(&all, 64);
        for i in 0..64 {
            assert!(bitset.contains(i), "Full bitset should contain everything");
        }
        assert!(!bitset.contains(64), "Out of range should return false");

        // Single element
        let bitset = BlockBitset::from_sorted_blocks(&[0], 128);
        assert!(bitset.contains(0));
        for i in 1..128 {
            assert!(!bitset.contains(i), "Only index 0 should be set");
        }

        // Boundary indices (last bit in each word)
        let bitset = BlockBitset::from_sorted_blocks(&[63, 127], 128);
        assert!(bitset.contains(63));
        assert!(bitset.contains(127));
        assert!(!bitset.contains(0));
        assert!(!bitset.contains(64));

        // First bit in each word
        let bitset = BlockBitset::from_sorted_blocks(&[0, 64], 128);
        assert!(bitset.contains(0));
        assert!(bitset.contains(64));
        assert!(!bitset.contains(1));
        assert!(!bitset.contains(63));

        // Zero-size bitset
        let bitset = BlockBitset::from_sorted_blocks(&[], 0);
        assert!(!bitset.contains(0));
    }
}
