//! Constant-time variants of the PRP and the iPRF (experimental).
//!
//! Compiled only with the `tee` feature. This code is not part of the trust
//! model (`docs/trust_model.md`): the client runs HintInit, and no enclave
//! does. `docs/constant_time_mode.md` gives the properties and the limits.

use super::*;

/// Constant-time Sometimes-Recurse PRP for TEE execution
///
/// Full-domain security variant using constant-time operations to prevent
/// timing side-channels. Uses branchless logic throughout.
///
/// Round counts are computed per Morris-Rogaway Section 5 "Strategy 1" (equal error split)
/// with 128-bit security by default.
pub struct SwapOrNotSrTee {
    cipher: Aes128,
    domain: u64,
    max_levels: usize,
    security_bits: u32,
}

impl SwapOrNotSrTee {
    pub fn new(key: PrfKey128, domain: u64) -> Self {
        Self::with_security(key, domain, DEFAULT_SECURITY_BITS)
    }

    pub fn with_security(key: PrfKey128, domain: u64, security_bits: u32) -> Self {
        assert!(domain > 0, "SwapOrNotSrTee domain must be positive");
        let cipher = Aes128::new(&GenericArray::from(key));
        let max_levels = if domain <= 1 {
            0
        } else {
            (domain as f64).log2().ceil() as usize
        };

        Self {
            cipher,
            domain,
            max_levels,
            security_bits,
        }
    }

    fn derive_round_key(&self, level: usize, round: usize, n: u64) -> u64 {
        let mut input = [0u8; 16];
        let combined = ((level as u64) << 32) | (round as u64);
        input[0..8].copy_from_slice(&combined.to_be_bytes());
        input[8..16].copy_from_slice(&n.to_be_bytes());

        let mut block = GenericArray::from(input);
        self.cipher.encrypt_block(&mut block);

        u64::from_be_bytes(block[0..8].try_into().unwrap()) % n
    }

    fn prf_bit_ct(&self, level: usize, round: usize, canonical: u64) -> u64 {
        let mut input = [0u8; 16];
        let combined = ((level as u64) << 32) | (round as u64) | 0x80000000;
        input[0..8].copy_from_slice(&combined.to_be_bytes());
        input[8..16].copy_from_slice(&canonical.to_be_bytes());

        let mut block = GenericArray::from(input);
        self.cipher.encrypt_block(&mut block);

        (block[0] & 1) as u64
    }

    fn round_ct(&self, level: usize, round_num: usize, n: u64, x: u64) -> u64 {
        use crate::constant_time::{ct_lt_u64, ct_select_u64};

        let k_i = self.derive_round_key(level, round_num, n);
        let partner = (k_i + n - (x % n)) % n;

        let x_lt_partner = ct_lt_u64(x, partner);
        let canonical = ct_select_u64(x_lt_partner, partner, x);

        let swap = self.prf_bit_ct(level, round_num, canonical);
        ct_select_u64(swap, partner, x)
    }

    fn apply_rounds_forward(&self, level: usize, n: u64, x: u64) -> u64 {
        let t = sr_t_rounds_with_security(n, self.domain, self.security_bits);
        let mut val = x;
        for r in 0..t {
            val = self.round_ct(level, r, n, val);
        }
        val
    }

    fn apply_rounds_inverse(&self, level: usize, n: u64, y: u64) -> u64 {
        let t = sr_t_rounds_with_security(n, self.domain, self.security_bits);
        let mut val = y;
        for r in (0..t).rev() {
            val = self.round_ct(level, r, n, val);
        }
        val
    }

    pub fn forward(&self, x: u64) -> u64 {
        use crate::constant_time::{ct_ge_u64, ct_select_u64};

        debug_assert!(x < self.domain, "x must be < domain");
        let mut val = x % self.domain;
        let mut n = self.domain;
        let mut result = 0u64;
        let mut done = 0u64;

        for level in 0..self.max_levels {
            let should_skip = ct_ge_u64(1, n) | done;

            val = ct_select_u64(should_skip, val, self.apply_rounds_forward(level, n, val));

            let half = n / 2;
            let exited = ct_ge_u64(val, half);
            result = ct_select_u64(exited & (1 - done), val, result);
            done |= exited;

            n = ct_select_u64(should_skip, n, half);
        }

        ct_select_u64(done, result, val)
    }

    pub fn inverse(&self, y: u64) -> u64 {
        use crate::constant_time::{ct_eq_u64, ct_ge_u64, ct_gt_u64, ct_lt_u64, ct_select_u64};

        debug_assert!(y < self.domain, "y must be < domain");
        let y = y % self.domain;

        let mut sizes = [0u64; 64];
        let max_levels = self.max_levels.min(sizes.len());
        let mut n = self.domain;
        let mut stopped: u64 = 0;

        for size in sizes.iter_mut().take(max_levels) {
            let should_record = ct_gt_u64(n, 1) & ct_eq_u64(stopped, 0);
            *size = ct_select_u64(should_record, n, 0);

            let half = n / 2;
            let y_in_right = ct_ge_u64(y, half);
            let stopping_now = should_record & y_in_right;
            stopped = ct_select_u64(stopping_now, 1, stopped);

            n = ct_select_u64(should_record & ct_eq_u64(stopped, 0), half, n);
        }

        let mut val = y;
        for (level, &n_lvl) in sizes.iter().take(max_levels).enumerate().rev() {
            let skip = ct_lt_u64(n_lvl, 2);
            val = ct_select_u64(skip, val, self.apply_rounds_inverse(level, n_lvl, val));
        }
        val
    }

    /// Inverts every element of `vals` in place. Each result equals
    /// `inverse` of the element.
    ///
    /// Every element runs every round of every stage; a mask keeps the
    /// elements that a stage does not hold. The number of AES blocks and the
    /// memory accesses depend only on the domain and on `vals.len()`, not on
    /// the values. [`SwapOrNotSrTee::inverse`] runs only the stages that hold
    /// its input, so its time depends on the input.
    ///
    /// # Precondition
    /// Each element must be below the domain. This is not checked in release
    /// builds, to avoid a branch on a secret value.
    pub fn inverse_many_ct(&self, vals: &mut [u64]) {
        use crate::constant_time::{ct_lt_u64, ct_max_u64};

        debug_assert!(
            vals.iter().all(|&y| y < self.domain),
            "each value must be < domain"
        );
        let inputs: Vec<u64> = vals.to_vec();
        let mut active: Vec<u64> = vec![0u64; vals.len()];
        let mut partners: Vec<u64> = vec![0u64; vals.len()];
        let mut blocks: Vec<Block> = vec![Block::default(); vals.len()];

        let sizes: Vec<u64> = sr_level_sizes(self.domain, self.max_levels);
        for (level, &n) in sizes.iter().enumerate().rev() {
            for (mask, &input) in active.iter_mut().zip(&inputs) {
                *mask = ct_lt_u64(input, n).wrapping_neg();
            }
            let t: usize = sr_t_rounds_with_security(n, self.domain, self.security_bits);
            for round in (0..t).rev() {
                let k_i: u64 = self.derive_round_key(level, round, n);
                let prefix: [u8; 8] = sr_swap_prefix(level, round);

                for i in 0..vals.len() {
                    // A value outside the stage is replaced by 0 to keep x < n.
                    let x: u64 = vals[i] & active[i];
                    let wrap: u64 = n & ct_lt_u64(k_i, x).wrapping_neg();
                    let partner: u64 = k_i.wrapping_sub(x).wrapping_add(wrap);
                    partners[i] = partner;
                    blocks[i][0..8].copy_from_slice(&prefix);
                    blocks[i][8..16].copy_from_slice(&ct_max_u64(x, partner).to_be_bytes());
                }
                self.cipher.encrypt_blocks(&mut blocks);
                for i in 0..vals.len() {
                    let swap: u64 = u64::from(blocks[i][0] & 1).wrapping_neg() & active[i];
                    vals[i] ^= swap & (vals[i] ^ partners[i]);
                }
            }
        }
    }
}

/// Constant-time SwapOrNot PRP for TEE execution
///
/// Functionally equivalent to SwapOrNot but uses branchless operations
/// to prevent timing side-channels.
pub struct SwapOrNotTee {
    cipher: Aes128,
    domain: u64,
    num_rounds: usize,
}

impl SwapOrNotTee {
    pub fn new(key: PrfKey128, domain: u64) -> Self {
        assert!(domain > 0, "SwapOrNot domain must be positive");
        let cipher = Aes128::new(&GenericArray::from(key));
        let num_rounds = ((domain as f64).log2().ceil() as usize) * 6 + 6;
        Self {
            cipher,
            domain,
            num_rounds,
        }
    }

    fn derive_round_key(&self, round: usize) -> u64 {
        let mut input = [0u8; 16];
        input[0..8].copy_from_slice(&(round as u64).to_be_bytes());
        input[8..16].copy_from_slice(&self.domain.to_be_bytes());
        let mut block = GenericArray::from(input);
        self.cipher.encrypt_block(&mut block);
        u64::from_be_bytes(block[0..8].try_into().unwrap()) % self.domain
    }

    /// Returns 1 if should swap, 0 otherwise (constant-time)
    fn prf_bit_ct(&self, round: usize, canonical: u64) -> u64 {
        let mut input = [0u8; 16];
        input[0..8].copy_from_slice(&(round as u64 | 0x8000000000000000).to_be_bytes());
        input[8..16].copy_from_slice(&canonical.to_be_bytes());
        let mut block = GenericArray::from(input);
        self.cipher.encrypt_block(&mut block);
        (block[0] & 1) as u64
    }

    /// Constant-time round - no secret-dependent branches
    fn round_ct(&self, round_num: usize, x: u64) -> u64 {
        use crate::constant_time::{ct_lt_u64, ct_select_u64};

        let k_i = self.derive_round_key(round_num);
        let partner = (k_i + self.domain - (x % self.domain)) % self.domain;

        // Constant-time max: canonical = max(x, partner)
        let x_lt_partner = ct_lt_u64(x, partner);
        let canonical = ct_select_u64(x_lt_partner, partner, x);

        // Constant-time swap decision
        let swap = self.prf_bit_ct(round_num, canonical);
        ct_select_u64(swap, partner, x)
    }

    pub fn forward(&self, x: u64) -> u64 {
        let mut val = x;
        for round in 0..self.num_rounds {
            val = self.round_ct(round, val);
        }
        val
    }

    pub fn inverse(&self, y: u64) -> u64 {
        let mut val = y;
        for round in (0..self.num_rounds).rev() {
            val = self.round_ct(round, val);
        }
        val
    }
}

/// Maximum preimages for fixed-size array return
pub const MAX_PREIMAGES: usize = 512;

/// Constant-time iPRF for TEE execution
///
/// Functionally equivalent to Iprf but uses branchless operations and
/// fixed iteration counts to prevent timing side-channels.
/// Uses SwapOrNotSrTee for full-domain PRP security.
pub struct IprfTee {
    cipher: Aes128,
    prp: SwapOrNotSrTee,
    domain: u64,
    range: u64,
    tree_depth: usize,
}

impl IprfTee {
    /// Creates a new TEE-safe iPRF instance.
    ///
    /// # Panics
    /// Panics if `m` is not a power of two of at most 2^32.
    pub fn new(key: PrfKey128, n: u64, m: u64) -> Self {
        assert_pmns_range(m);

        let tree_depth = m.trailing_zeros() as usize;
        let cipher = pmns_cipher(&key, n, m);

        let mut prp_key = [0u8; 16];
        let mut hasher = Sha256::new();
        hasher.update(key);
        hasher.update(b"prp");
        let hash = hasher.finalize();
        prp_key.copy_from_slice(&hash[0..16]);

        let prp = SwapOrNotSrTee::new(prp_key, n);

        Self {
            cipher,
            prp,
            domain: n,
            range: m,
            tree_depth,
        }
    }

    /// Forward evaluation.
    ///
    /// # Security Note
    /// This method is NOT constant-time: the internal `trace_ball` loop branches
    /// on `ball_index < left_count`. For PIR query privacy, only `inverse_ct` needs
    /// to be constant-time (the server computes inverse, not forward).
    ///
    /// # Precondition
    /// `x` must be in range `[0, domain)`. This is a precondition, not runtime-checked
    /// in release builds, to avoid timing side-channels.
    pub fn forward(&self, x: u64) -> u64 {
        debug_assert!(x < self.domain, "IprfTee::forward: x must be < domain");
        let permuted = self.prp.forward(x);
        self.trace_ball(permuted, self.domain, self.range)
    }

    /// Constant-time inverse returning fixed-size array with validity mask.
    ///
    /// Each call samples every tree level again. To invert many outputs of
    /// one instance, use [`IprfTee::table_ct`].
    ///
    /// # Precondition
    /// `y` must be in range `[0, range)`. This is a precondition, not runtime-checked
    /// in release builds, to avoid timing side-channels.
    pub fn inverse_ct(&self, y: u64) -> ([u64; MAX_PREIMAGES], usize) {
        debug_assert!(y < self.range, "IprfTee::inverse_ct: y must be < range");

        let (ball_start, ball_count) = self.trace_ball_inverse_ct(y);
        self.preimages_ct(ball_start, ball_count)
    }

    /// Computes all bin boundaries of the PMNS tree once, with work that
    /// depends only on the domain and the range.
    ///
    /// The node of a subtree with `b` bins is sampled with the public bound
    /// `min(domain, MAX_PREIMAGES * b)` on its ball count. If a count exceeds
    /// its bound, or a bin holds more than [`MAX_PREIMAGES`] balls, the table
    /// reports it through [`IprfTeeTable::overflowed`] and its preimages are
    /// not valid. With at most `MAX_PREIMAGES / 2` expected balls per bin, the
    /// probability is below 2^-120 per node.
    pub fn table_ct(&self) -> IprfTeeTable<'_> {
        use crate::constant_time::{ct_lt_u64, ct_min_u64};

        let max_preimages: u64 = MAX_PREIMAGES as u64;
        let mut overflow: u64 = 0;

        let loads: Vec<u64> = pmns_level_counts(
            self.domain,
            self.range,
            self.tree_depth as u32,
            |node, count| {
                let bins: u64 = u64::from(node.high - node.low) + 1;
                let bound: u64 = max_preimages.saturating_mul(bins).min(self.domain);
                overflow |= ct_lt_u64(bound, count);
                crate::binomial::coin_binomial_ct(
                    &self.cipher,
                    node,
                    ct_min_u64(count, bound),
                    bound,
                )
            },
        );
        for &load in &loads {
            overflow |= ct_lt_u64(max_preimages, load);
        }

        IprfTeeTable {
            iprf: self,
            starts: bin_starts(&loads),
            overflow,
        }
    }

    /// Applies the inverse PRP to the balls `ball_start..ball_start + ball_count`
    /// with a fixed number of iterations.
    fn preimages_ct(&self, ball_start: u64, ball_count: u64) -> ([u64; MAX_PREIMAGES], usize) {
        use crate::constant_time::{ct_lt_u64, ct_select_u64};

        // Constant-time min(ball_count, MAX_PREIMAGES) to avoid data-dependent branch
        let max_preimages = MAX_PREIMAGES as u64;
        let exceeds_max = ct_lt_u64(max_preimages, ball_count);
        let count_u64 = ct_select_u64(exceeds_max, max_preimages, ball_count);
        let count = count_u64 as usize;
        let mut result = [0u64; MAX_PREIMAGES];
        let mut valid = [0u64; MAX_PREIMAGES];

        for (i, (slot, keep)) in result.iter_mut().zip(valid.iter_mut()).enumerate() {
            let i_u64 = i as u64;
            let in_range = ct_lt_u64(i_u64, count as u64);
            let z = ball_start + i_u64;
            // Only invert valid z values; use 0 for out-of-range
            let z_valid = ct_lt_u64(z, self.domain);
            *slot = ct_select_u64(z_valid, z, 0);
            *keep = in_range & z_valid;
        }
        self.prp.inverse_many_ct(&mut result);
        for (slot, &keep) in result.iter_mut().zip(valid.iter()) {
            *slot = ct_select_u64(keep, *slot, 0);
        }

        (result, count)
    }

    /// Constant-time trace_ball_inverse with fixed iteration count
    fn trace_ball_inverse_ct(&self, y: u64) -> (u64, u64) {
        use crate::constant_time::{ct_le_u64, ct_lt_u64, ct_select_u64};

        if self.range == 1 {
            return (0, self.domain);
        }

        let mut low = 0u64;
        let mut high = self.range - 1;
        let mut ball_count = self.domain;
        let mut ball_start = 0u64;

        for _level in 0..self.tree_depth {
            let should_continue = ct_lt_u64(low, high);

            let mid = (low + high) / 2;

            let left_count = crate::binomial::coin_binomial_ct(
                &self.cipher,
                pmns_node(low, high),
                ball_count,
                self.domain,
            );

            let go_left = ct_le_u64(y, mid);

            let new_low_left = low;
            let new_high_left = mid;
            let new_count_left = left_count;
            let new_start_left = ball_start;

            let new_low_right = mid + 1;
            let new_high_right = high;
            let new_count_right = ball_count.wrapping_sub(left_count);
            let new_start_right = ball_start + left_count;

            let new_low = ct_select_u64(go_left, new_low_left, new_low_right);
            let new_high = ct_select_u64(go_left, new_high_left, new_high_right);
            let new_count = ct_select_u64(go_left, new_count_left, new_count_right);
            let new_start = ct_select_u64(go_left, new_start_left, new_start_right);

            low = ct_select_u64(should_continue, new_low, low);
            high = ct_select_u64(should_continue, new_high, high);
            ball_count = ct_select_u64(should_continue, new_count, ball_count);
            ball_start = ct_select_u64(should_continue, new_start, ball_start);
        }

        (ball_start, ball_count)
    }

    fn trace_ball(&self, x_prime: u64, n: u64, m: u64) -> u64 {
        if m == 1 {
            return 0;
        }
        let mut low = 0u64;
        let mut high = m - 1;
        let mut ball_count = n;
        let mut ball_index = x_prime;

        while low < high {
            let mid = (low + high) / 2;
            let left_count =
                crate::binomial::coin_binomial(&self.cipher, pmns_node(low, high), ball_count);

            if ball_index < left_count {
                high = mid;
                ball_count = left_count;
            } else {
                low = mid + 1;
                ball_index -= left_count;
                ball_count -= left_count;
            }
        }
        low
    }

    /// Helper to get Vec<u64> like standard inverse (for testing)
    pub fn inverse(&self, y: u64) -> Vec<u64> {
        let (arr, count) = self.inverse_ct(y);
        arr[..count].to_vec()
    }
}

/// All bin boundaries of one [`IprfTee`], from [`IprfTee::table_ct`].
///
/// The table is derived from the key and is as secret as the key.
pub struct IprfTeeTable<'a> {
    iprf: &'a IprfTee,
    /// `starts[y]` is the first ball of bin `y`; `starts[range]` is the domain.
    starts: Vec<u64>,
    overflow: u64,
}

impl IprfTeeTable<'_> {
    /// Returns the same value as [`IprfTee::inverse_ct`] when the table did not overflow.
    ///
    /// `y` is read as a table index, so it must be public, as the offset of a
    /// database entry is.
    ///
    /// # Precondition
    /// `y` must be in range `[0, range)`.
    pub fn inverse_ct(&self, y: u64) -> ([u64; MAX_PREIMAGES], usize) {
        debug_assert!(
            y < self.iprf.range,
            "IprfTeeTable::inverse_ct: y must be < range"
        );
        let ball_start: u64 = self.starts[y as usize];
        let ball_count: u64 = self.starts[y as usize + 1].wrapping_sub(ball_start);
        self.iprf.preimages_ct(ball_start, ball_count)
    }

    /// Returns `true` if a ball count exceeded its public bound.
    ///
    /// The preimages of an overflowed table are not valid. The caller must
    /// stop; this reveals only that an event of negligible probability occurred.
    pub fn overflowed(&self) -> bool {
        self.overflow != 0
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::{boundary_ranges, pmns_test_key, table_ball_ranges};
    use super::*;

    #[test]
    fn test_swap_or_not_tee_matches_standard() {
        let key = [3u8; 16];
        let domain = 1000u64;
        let prp = SwapOrNot::new(key, domain);
        let prp_tee = SwapOrNotTee::new(key, domain);

        for x in 0..100 {
            assert_eq!(
                prp.forward(x),
                prp_tee.forward(x),
                "forward mismatch at x={}",
                x
            );
            assert_eq!(
                prp.inverse(x),
                prp_tee.inverse(x),
                "inverse mismatch at x={}",
                x
            );
        }
    }

    /// The fast and TEE PMNS trees must give the same ball range for every bin.
    ///
    /// The domain is not a multiple of the keystream block size, so node
    /// counts end inside a block.
    #[test]
    fn test_pmns_tee_matches_standard() {
        let domain: u64 = 100_003;
        let range: u64 = 64;

        for index in 0..2u8 {
            let key: PrfKey128 = pmns_test_key(index);
            let iprf: Iprf = Iprf::new(key, domain, range);
            let iprf_tee: IprfTee = IprfTee::new(key, domain, range);

            for y in 0..range {
                let balls: Vec<u64> = iprf.trace_ball_inverse(y, domain, range).collect();
                let (start, count) = iprf_tee.trace_ball_inverse_ct(y);
                assert_eq!(
                    (balls.first().copied(), balls.len() as u64),
                    ((count > 0).then_some(start), count),
                    "key {index}, bin {y}"
                );
            }
        }
    }

    /// The constant-time table must equal the per-call constant-time walk
    /// when the bin loads are within the bound.
    ///
    /// With 4096 balls and 64 bins, the nodes below depth 3 are sampled with a
    /// bound that is smaller than the domain.
    #[test]
    fn test_pmns_tee_table_matches_tree_walk() {
        let domain: u64 = 4_099;
        let range: u64 = 64;

        for index in 0..2u8 {
            let iprf_tee: IprfTee = IprfTee::new(pmns_test_key(index), domain, range);
            let table: IprfTeeTable<'_> = iprf_tee.table_ct();
            assert!(!table.overflowed(), "key {index}");

            let walked: Vec<(u64, u64)> = (0..range)
                .map(|y| iprf_tee.trace_ball_inverse_ct(y))
                .collect();
            assert_eq!(table_ball_ranges(&table.starts), walked, "key {index}");
        }
    }

    /// Loads above the bound must be reported, not returned as valid preimages.
    #[test]
    fn test_pmns_tee_table_reports_overflow() {
        // About 1563 balls per bin: above MAX_PREIMAGES.
        let dense: IprfTee = IprfTee::new(pmns_test_key(0), 100_003, 64);
        assert!(dense.table_ct().overflowed());

        // 1024 balls in 2 bins: the root bound equals the domain, so no node
        // overflows, but the larger bin holds more than MAX_PREIMAGES balls.
        let small: IprfTee = IprfTee::new(pmns_test_key(0), 1_024, 2);
        let table: IprfTeeTable<'_> = small.table_ct();
        let (_, first_load) = small.trace_ball_inverse_ct(0);
        assert_ne!(first_load, 512, "the test key must give unequal bins");
        assert_eq!(table_ball_ranges(&table.starts)[0], (0, first_load));
        assert!(table.overflowed());

        // 600 balls in 2 bins: both bins are within the bound.
        let sparse: IprfTee = IprfTee::new(pmns_test_key(0), 600, 2);
        let (_, first_load) = sparse.trace_ball_inverse_ct(0);
        assert!(first_load.max(600 - first_load) <= MAX_PREIMAGES as u64);
        assert!(!sparse.table_ct().overflowed());
    }

    /// `inverse_ct` of the table must equal `inverse_ct` of the instance, PRP included.
    #[test]
    fn test_iprf_tee_table_inverse_matches_inverse_ct() {
        let iprf_tee: IprfTee = IprfTee::new(pmns_test_key(6), 96, 4);
        let table: IprfTeeTable<'_> = iprf_tee.table_ct();
        assert!(!table.overflowed());

        let mut seen: usize = 0;
        for y in 0..4u64 {
            let (expected, expected_count) = iprf_tee.inverse_ct(y);
            let (got, got_count) = table.inverse_ct(y);
            assert_eq!(got_count, expected_count, "y={y}");
            assert_eq!(got[..got_count], expected[..expected_count], "y={y}");
            seen += got_count;
        }
        assert_eq!(seen, 96);
    }

    #[test]
    #[ignore = "slow with 128-bit security; run with --ignored"]
    fn test_iprf_tee_matches_standard() {
        let key = [4u8; 16];
        let domain = 500u64;
        let range = 64u64;
        let iprf = Iprf::new(key, domain, range);
        let iprf_tee = IprfTee::new(key, domain, range);

        for x in 0..100 {
            let y = iprf.forward(x);
            let y_tee = iprf_tee.forward(x);
            assert_eq!(y, y_tee, "forward mismatch at x={}", x);
        }

        for y in 0..range {
            let preimages = iprf.inverse(y);
            let preimages_tee = iprf_tee.inverse(y);
            assert_eq!(preimages, preimages_tee, "inverse mismatch at y={}", y);
        }
    }

    #[test]
    #[ignore = "slow with 128-bit security; run with --ignored"]
    fn test_swap_or_not_sr_tee_matches_standard() {
        let key = [7u8; 16];
        let domain = 64u64;
        let prp = SwapOrNotSr::new(key, domain);
        let prp_tee = SwapOrNotSrTee::new(key, domain);

        for x in 0..domain {
            assert_eq!(
                prp.forward(x),
                prp_tee.forward(x),
                "SR TEE forward mismatch at x={}",
                x
            );
        }

        for y in 0..domain {
            assert_eq!(
                prp.inverse(y),
                prp_tee.inverse(y),
                "SR TEE inverse mismatch at y={}",
                y
            );
        }
    }

    #[test]
    #[ignore = "slow with 128-bit security; run with --ignored"]
    fn test_swap_or_not_sr_tee_inverse_roundtrip() {
        let key = [8u8; 16];
        let domain = 100u64;
        let prp_tee = SwapOrNotSrTee::new(key, domain);

        for x in 0..domain {
            let y = prp_tee.forward(x);
            let x_recovered = prp_tee.inverse(y);
            assert_eq!(
                x, x_recovered,
                "SR TEE inverse roundtrip failed for x={}",
                x
            );
        }
    }

    /// `inverse_many_ct` must equal `inverse` for values in any order.
    #[test]
    fn test_sr_tee_inverse_many_ct_matches_inverse_on_small_domains() {
        for domain in 1..=33u64 {
            let key: PrfKey128 = [domain as u8; 16];
            let prp: SwapOrNotSr = SwapOrNotSr::with_security(key, domain, 16);
            let prp_tee: SwapOrNotSrTee = SwapOrNotSrTee::with_security(key, domain, 16);

            let mut vals: Vec<u64> = (0..domain).rev().chain(0..domain).collect();
            let expected: Vec<u64> = vals.iter().map(|&y| prp.inverse(y)).collect();
            prp_tee.inverse_many_ct(&mut vals);
            assert_eq!(vals, expected, "domain {domain}");
        }
    }

    /// Full round counts, domains above 65536, values of every stage depth in
    /// one call.
    #[test]
    fn test_sr_tee_inverse_many_ct_matches_inverse_at_stage_boundaries() {
        for domain in [100_003u64, 131_072] {
            let key: PrfKey128 = [0x5a; 16];
            let prp: SwapOrNotSr = SwapOrNotSr::new(key, domain);
            let prp_tee: SwapOrNotSrTee = SwapOrNotSrTee::new(key, domain);

            let mut vals: Vec<u64> = boundary_ranges(domain, 6).into_iter().flatten().collect();
            vals.reverse();
            vals.extend([0, 0, domain - 1, 1, domain / 3]);
            let expected: Vec<u64> = vals.iter().map(|&y| prp.inverse(y)).collect();
            let expected_tee: Vec<u64> = vals.iter().map(|&y| prp_tee.inverse(y)).collect();
            assert_eq!(expected_tee, expected, "domain {domain}");

            prp_tee.inverse_many_ct(&mut vals);
            assert_eq!(vals, expected, "domain {domain}");
        }
    }

    #[test]
    fn test_sr_tee_with_low_security() {
        let key = [10u8; 16];
        let domain = 32u64;

        let prp = SwapOrNotSr::with_security(key, domain, 32);
        let prp_tee = SwapOrNotSrTee::with_security(key, domain, 32);

        for x in 0..domain {
            assert_eq!(
                prp.forward(x),
                prp_tee.forward(x),
                "SR/TEE forward mismatch at x={}",
                x
            );
            let y = prp.forward(x);
            assert_eq!(
                prp.inverse(y),
                prp_tee.inverse(y),
                "SR/TEE inverse mismatch at y={}",
                y
            );
        }
    }

    /// The constant-time table inverse must return the single-value PRP
    /// inverse of each ball of the bin, in ball order, on a domain above 65536.
    #[test]
    fn test_iprf_tee_table_inverse_matches_single_value_prp() {
        let key: PrfKey128 = pmns_test_key(7);
        let domain: u64 = 100_003;
        let range: u64 = 512;

        let iprf: Iprf = Iprf::new(key, domain, range);
        let iprf_tee: IprfTee = IprfTee::new(key, domain, range);
        let tee_table: IprfTeeTable<'_> = iprf_tee.table_ct();
        assert!(!tee_table.overflowed());

        for y in [0u64, 255, 511] {
            let balls: Range<u64> = iprf.trace_ball_inverse(y, domain, range);
            let expected: Vec<u64> = balls.map(|z| iprf.prp.inverse(z)).collect();
            assert!(expected.len() > 100, "y={y}");

            let (got, count) = tee_table.inverse_ct(y);
            assert_eq!(got[..count], expected, "IprfTeeTable::inverse_ct, y={y}");
            assert!(got[count..].iter().all(|&x| x == 0), "y={y}");
        }
    }
}
