//! Invertible Pseudorandom Function (iPRF) for Plinko PIR
//!
//! Implementation based on the Plinko paper (2024-318):
//! - iPRF is built from Swap-or-Not PRP + PMNS (Pseudorandom Multinomial Sampler)
//! - Forward: iF.F(k, x) = S(k_pmns, P(k_prp, x))
//! - Inverse: iF.F^{-1}(k, y) = {P^{-1}(k_prp, z) : z ∈ S^{-1}(k_pmns, y)}
//!
//! The Swap-or-Not PRP is based on Morris-Rogaway (eprint 2013/560).

use aes::cipher::{generic_array::GenericArray, BlockEncrypt, KeyInit};
use aes::{Aes128, Block};
use sha2::{Digest, Sha256};
use std::ops::Range;
use std::sync::OnceLock;

use crate::binomial::PmnsNode;

pub type PrfKey128 = [u8; 16];

/// Swap-or-Not small-domain PRP (Morris-Rogaway 2013)
///
/// Inner shuffle used by SwapOrNotSr. Provides security for q < (1-epsilon)*N queries.
/// For full-domain security (all N queries), use SwapOrNotSr which wraps this with
/// the Sometimes-Recurse transformation.
pub struct SwapOrNot {
    cipher: Aes128,
    domain: u64,
    num_rounds: usize,
}

impl SwapOrNot {
    /// Creates a new SwapOrNot PRP instance for a finite domain.
    ///
    /// The `key` is a 16-byte AES-128 key used to initialize the internal cipher.
    /// `domain` is the size of the finite domain (must be greater than zero); the
    /// PRP operates over the integer set [0, domain).
    ///
    /// # Panics
    ///
    /// Panics if `domain` is zero.
    ///
    pub fn new(key: PrfKey128, domain: u64) -> Self {
        assert!(domain > 0, "SwapOrNot domain must be positive");
        let cipher = Aes128::new(&GenericArray::from(key));
        // ~6 * log2(N) rounds for full security
        let num_rounds = ((domain as f64).log2().ceil() as usize) * 6 + 6;

        Self {
            cipher,
            domain,
            num_rounds,
        }
    }

    /// Derives the round-specific key K_i from the PRP key and domain using AES.
    ///
    /// The function encrypts a 16-byte block formed from `round` (big-endian u64) and `domain`
    /// (big-endian u64), interprets the upper 64 bits of the ciphertext as a u64, and reduces
    /// it modulo `self.domain` to produce a value in the domain range.
    ///
    /// # Parameters
    ///
    /// - `round`: Round index (0-based) for which to derive the round key.
    ///
    /// # Returns
    ///
    /// `K_i` — a `u64` in the range `0..self.domain` representing the derived round key.
    fn derive_round_key(&self, round: usize) -> u64 {
        let mut input = [0u8; 16];
        input[0..8].copy_from_slice(&(round as u64).to_be_bytes());
        input[8..16].copy_from_slice(&self.domain.to_be_bytes());

        let mut block = GenericArray::from(input);
        self.cipher.encrypt_block(&mut block);

        u64::from_be_bytes(block[0..8].try_into().unwrap()) % self.domain
    }

    /// Derives the single-bit pseudorandom decision used to choose whether to swap in a Swap-or-Not round.
    fn prf_bit(&self, round: usize, canonical: u64) -> bool {
        let mut input = [0u8; 16];
        input[0..8].copy_from_slice(&(round as u64 | 0x8000000000000000).to_be_bytes());
        input[8..16].copy_from_slice(&canonical.to_be_bytes());

        let mut block = GenericArray::from(input);
        self.cipher.encrypt_block(&mut block);

        (block[0] & 1) == 1
    }

    /// Applies one round of the Swap-or-Not permutation to a value within the domain.
    ///
    /// The round computes a partner value K_i - x (mod N) using a round-specific key,
    /// chooses the canonical representative between the input and its partner, and
    /// uses a round-dependent PRF bit to decide whether to swap to the partner or
    /// keep the input. The operation is involutory: applying the same round again
    /// undoes the effect.
    fn round(&self, round_num: usize, x: u64) -> u64 {
        let k_i = self.derive_round_key(round_num);
        // Partner: K_i - X mod N
        let partner = (k_i + self.domain - (x % self.domain)) % self.domain;
        // Canonical representative: max(X, X')
        let canonical = x.max(partner);

        if self.prf_bit(round_num, canonical) {
            partner
        } else {
            x
        }
    }

    /// Forward PRP: encrypt by running rounds 0, 1, ..., r-1
    pub fn forward(&self, x: u64) -> u64 {
        let mut val = x;
        for round in 0..self.num_rounds {
            val = self.round(round, val);
        }
        val
    }

    /// Inverse PRP: decrypt by running rounds r-1, r-2, ..., 0
    pub fn inverse(&self, y: u64) -> u64 {
        let mut val = y;
        for round in (0..self.num_rounds).rev() {
            val = self.round(round, val);
        }
        val
    }
}

/// Default security parameter (bits) for SR PRP
pub const DEFAULT_SECURITY_BITS: u32 = 128;

/// Per-stage SN round count from Morris-Rogaway Eq. (2) with Strategy 1 (split error equally).
///
/// For SR shuffle with canonical split, we need each SN stage to achieve error ε/p
/// where p = number of nontrivial stages ≈ lg(N₀).
///
/// Per Eq. (2): t_N ≥ 7.23 lg N - 4.82 lg(ε/p)
///            = 7.23 lg N - 4.82 lg ε + 4.82 lg p
///            = 7.23 lg N + 4.82 λ + 4.82 lg p   (for ε = 2^(-λ))
///
/// Parameters:
/// - `n`: Current domain size at this SR level
/// - `num_stages`: Total number of nontrivial stages p = |G'(N₀)|
/// - `lambda`: Security parameter in bits (ε = 2^(-λ))
fn sr_t_rounds_with_params(n: u64, num_stages: usize, lambda: u32) -> usize {
    if n <= 2 {
        return if n == 2 { 1 } else { 0 };
    }

    let log_n = (n as f64).log2();
    let log_p = if num_stages > 0 {
        (num_stages as f64).log2()
    } else {
        0.0
    };

    let rounds = 7.23 * log_n + 4.82 * (lambda as f64) + 4.82 * log_p;
    rounds.ceil() as usize
}

/// Compute the number of nontrivial SR stages for domain N₀.
///
/// Nontrivial stages are those with N ≥ 3 (we exclude N=1 and N=2).
/// For canonical split p_N = floor(N/2), we have:
/// - G(N₀) = {N₀, floor(N₀/2), floor(N₀/4), ...}
/// - G'(N₀) = G(N₀) \ {1, 2}
fn sr_num_stages(n0: u64) -> usize {
    if n0 <= 2 {
        return 0;
    }
    let mut count = 0;
    let mut n = n0;
    while n >= 3 {
        count += 1;
        n /= 2;
    }
    count
}

/// Round count with explicit security parameter.
///
/// For λ-bit security (distinguishing advantage ε = 2^(-λ)):
/// - Total error budget ε is split equally among p = |G'(N₀)| stages
/// - Each stage gets ε/p error budget
/// - Per Eq. (2), t_N = ceil(7.23 lg N + 4.82 lg(p/ε))
///   = ceil(7.23 lg N + 4.82 lg p + 4.82 λ)
fn sr_t_rounds_with_security(n: u64, n0: u64, lambda: u32) -> usize {
    let num_stages = sr_num_stages(n0);
    sr_t_rounds_with_params(n, num_stages, lambda)
}

/// Stage sizes of the Sometimes-Recurse shuffle: `domain`, then its halves,
/// while the size is above 1. Stage `l` permutes the values below `sizes[l]`.
fn sr_level_sizes(domain: u64, max_levels: usize) -> Vec<u64> {
    let mut sizes: Vec<u64> = Vec::with_capacity(max_levels);
    let mut n: u64 = domain;
    while n > 1 && sizes.len() < max_levels {
        sizes.push(n);
        n /= 2;
    }
    sizes
}

/// AES input prefix of the swap decision of one round.
fn sr_swap_prefix(level: usize, round: usize) -> [u8; 8] {
    (((level as u64) << 32) | (round as u64) | 0x80000000).to_be_bytes()
}

/// Sometimes-Recurse PRP wrapper (Morris-Rogaway Fig. 1)
///
/// Provides full-domain security: secure even when adversary queries all N elements.
/// Uses SwapOrNot as inner shuffle with recursive halving.
///
/// Round counts are computed per Morris-Rogaway Section 5 "Strategy 1" (equal error split)
/// with 128-bit security by default.
pub struct SwapOrNotSr {
    cipher: Aes128,
    domain: u64,
    max_levels: usize,
    security_bits: u32,
}

impl SwapOrNotSr {
    pub fn new(key: PrfKey128, domain: u64) -> Self {
        Self::with_security(key, domain, DEFAULT_SECURITY_BITS)
    }

    pub fn with_security(key: PrfKey128, domain: u64, security_bits: u32) -> Self {
        assert!(domain > 0, "SwapOrNotSr domain must be positive");
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

    fn prf_bit(&self, level: usize, round: usize, canonical: u64, _n: u64) -> bool {
        let mut input = [0u8; 16];
        let combined = ((level as u64) << 32) | (round as u64) | 0x80000000;
        input[0..8].copy_from_slice(&combined.to_be_bytes());
        input[8..16].copy_from_slice(&canonical.to_be_bytes());

        let mut block = GenericArray::from(input);
        self.cipher.encrypt_block(&mut block);

        (block[0] & 1) == 1
    }

    fn round(&self, level: usize, round_num: usize, n: u64, x: u64) -> u64 {
        let k_i = self.derive_round_key(level, round_num, n);
        let partner = (k_i + n - (x % n)) % n;
        let canonical = x.max(partner);

        if self.prf_bit(level, round_num, canonical, n) {
            partner
        } else {
            x
        }
    }

    fn apply_rounds_forward(&self, level: usize, n: u64, x: u64) -> u64 {
        let t = sr_t_rounds_with_security(n, self.domain, self.security_bits);
        let mut val = x;
        for r in 0..t {
            val = self.round(level, r, n, val);
        }
        val
    }

    fn apply_rounds_inverse(&self, level: usize, n: u64, y: u64) -> u64 {
        let t = sr_t_rounds_with_security(n, self.domain, self.security_bits);
        let mut val = y;
        for r in (0..t).rev() {
            val = self.round(level, r, n, val);
        }
        val
    }

    pub fn forward(&self, x: u64) -> u64 {
        assert!(x < self.domain, "x must be < domain");
        let mut val = x;
        let mut n = self.domain;

        for level in 0..self.max_levels {
            if n <= 1 {
                break;
            }
            val = self.apply_rounds_forward(level, n, val);
            let half = n / 2;
            if val >= half {
                return val;
            }
            n = half;
        }
        val
    }

    pub fn inverse(&self, y: u64) -> u64 {
        assert!(y < self.domain, "y must be < domain");

        let mut sizes = Vec::with_capacity(self.max_levels);
        let mut n = self.domain;
        for _ in 0..self.max_levels {
            if n <= 1 {
                break;
            }
            sizes.push(n);
            let half = n / 2;
            if y >= half {
                break;
            }
            n = half;
        }

        let mut val = y;
        for (level, &n) in sizes.iter().enumerate().rev() {
            val = self.apply_rounds_inverse(level, n, val);
        }
        val
    }

    /// Inverts every value of `balls`. Element `i` of the result equals
    /// `inverse(balls.start + i)`.
    ///
    /// A value passes stage `l` if it is below the stage size, so the values
    /// of a range that pass a stage are a prefix of the range. All of them
    /// share the round key of a round, and their swap decisions are one batch
    /// of AES blocks. A single `inverse` call derives the round key again for
    /// each value and waits for each AES block before the next round.
    ///
    /// # Panics
    /// Panics if `balls.end` is above the domain.
    ///
    /// # Examples
    ///
    /// ```
    /// use plinko::iprf::SwapOrNotSr;
    ///
    /// let prp: SwapOrNotSr = SwapOrNotSr::with_security([3u8; 16], 1000, 16);
    /// let batch: Vec<u64> = prp.inverse_range(490..510);
    /// let single: Vec<u64> = (490..510).map(|y| prp.inverse(y)).collect();
    /// assert_eq!(batch, single);
    /// ```
    pub fn inverse_range(&self, balls: Range<u64>) -> Vec<u64> {
        assert!(balls.end <= self.domain, "balls must be < domain");
        let mut vals: Vec<u64> = balls.clone().collect();
        let mut partners: Vec<u64> = vec![0u64; vals.len()];
        let mut blocks: Vec<Block> = vec![Block::default(); vals.len()];

        let sizes: Vec<u64> = sr_level_sizes(self.domain, self.max_levels);
        for (level, &n) in sizes.iter().enumerate().rev() {
            let active: usize = n.saturating_sub(balls.start).min(vals.len() as u64) as usize;
            if active == 0 {
                continue;
            }
            let t: usize = sr_t_rounds_with_security(n, self.domain, self.security_bits);
            for round in (0..t).rev() {
                let k_i: u64 = self.derive_round_key(level, round, n);
                let prefix: [u8; 8] = sr_swap_prefix(level, round);

                // The swap bits are random, so both loops use masks, not branches.
                for i in 0..active {
                    // vals[i] < n and k_i < n, so this is (k_i - vals[i]) mod n.
                    let wrap: u64 = n & u64::from(k_i < vals[i]).wrapping_neg();
                    let partner: u64 = k_i.wrapping_sub(vals[i]).wrapping_add(wrap);
                    partners[i] = partner;
                    blocks[i][0..8].copy_from_slice(&prefix);
                    blocks[i][8..16].copy_from_slice(&vals[i].max(partner).to_be_bytes());
                }
                self.cipher.encrypt_blocks(&mut blocks[..active]);
                for i in 0..active {
                    let swap: u64 = u64::from(blocks[i][0] & 1).wrapping_neg();
                    vals[i] ^= swap & (vals[i] ^ partners[i]);
                }
            }
        }
        vals
    }
}

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

/// Invertible PRF built from Sometimes-Recurse PRP + PMNS
///
/// Uses SwapOrNotSr for full-domain PRP security (secure even when all N elements queried).
pub struct Iprf {
    cipher: Aes128,
    prp: SwapOrNotSr,
    domain: u64,
    range: u64,
    /// Number of top tree levels whose left counts are kept after the first use.
    cached_levels: u32,
    /// Left counts of the nodes in the first `cached_levels` levels, in
    /// breadth-first order. Derived from the key and as secret as the key.
    top_left_counts: OnceLock<Box<[u32]>>,
}

impl Iprf {
    /// Creates a new iPRF instance for input domain `n` and output range `m`.
    ///
    /// The constructor derives the PMNS keystream key (SHA-256(key || "pmns" || n || m)),
    /// computes the PMNS tree depth as log2(m), and derives a separate 128-bit key
    /// (SHA-256(key || "prp")) to initialize the internal Sometimes-Recurse PRP over the
    /// input domain `n`.
    ///
    /// # Panics
    /// Panics if `m` is not a power of two of at most 2^32. Every PMNS split then has
    /// probability 1/2, which the exact sampler in [`crate::binomial`] requires.
    pub fn new(key: PrfKey128, n: u64, m: u64) -> Self {
        Self::with_cached_levels(key, n, m, 0)
    }

    /// Creates an iPRF that keeps the left counts of the first `levels` PMNS
    /// tree levels in memory. All outputs equal those of [`Iprf::new`].
    ///
    /// The first `forward` or `inverse` call computes the cache; it costs
    /// `levels * n / 128` AES blocks. Later calls sample only the lower levels,
    /// whose nodes hold few balls. The cache takes `4 * (2^levels - 1)` bytes.
    /// It is derived from the key and must stay with the owner of the key.
    ///
    /// `levels` is limited to the tree depth. The cache stores counts as
    /// `u32`; for a domain above `u32::MAX` nothing is cached.
    ///
    /// # Panics
    /// Panics if `m` is not a power of two of at most 2^32.
    ///
    /// # Examples
    ///
    /// ```
    /// use plinko::iprf::Iprf;
    ///
    /// let plain: Iprf = Iprf::new([7u8; 16], 4096, 64);
    /// let cached: Iprf = Iprf::with_cached_levels([7u8; 16], 4096, 64, 4);
    /// assert_eq!(cached.forward(1234), plain.forward(1234));
    /// assert_eq!(cached.inverse(17), plain.inverse(17));
    /// ```
    pub fn with_cached_levels(key: PrfKey128, n: u64, m: u64, levels: u32) -> Self {
        assert_pmns_range(m);

        let tree_depth: u32 = m.trailing_zeros();
        let cached_levels: u32 = if n <= u64::from(u32::MAX) {
            levels.min(tree_depth)
        } else {
            0
        };
        let cipher = pmns_cipher(&key, n, m);

        // Derive a separate key for PRP from main key
        let mut prp_key = [0u8; 16];
        let mut hasher = Sha256::new();
        hasher.update(key);
        hasher.update(b"prp");
        let hash = hasher.finalize();
        prp_key.copy_from_slice(&hash[0..16]);

        let prp = SwapOrNotSr::new(prp_key, n);

        Self {
            cipher,
            prp,
            domain: n,
            range: m,
            cached_levels,
            top_left_counts: OnceLock::new(),
        }
    }

    /// Forward evaluation: P(x) then S(P(x))
    pub fn forward(&self, x: u64) -> u64 {
        if x >= self.domain {
            return 0;
        }
        // Apply PRP first, then PMNS
        let permuted = self.prp.forward(x);
        self.trace_ball(permuted, self.domain, self.range)
    }

    /// Compute all input values `x` in the domain that map to the given output `y`.
    ///
    /// If `y` is outside the iPRF's output range, an empty vector is returned. The returned
    /// vector contains every `x` in `[0, self.domain)` such that `self.forward(x) == y` (order not specified).
    ///
    /// Note: Multiple preimages per output are possible since this is an iPRF, not a bijection.
    pub fn inverse(&self, y: u64) -> Vec<u64> {
        if y >= self.range {
            return vec![];
        }
        // First find all PMNS preimages, then apply inverse PRP to each
        let balls: Range<u64> = self.trace_ball_inverse(y, self.domain, self.range);
        self.prp.inverse_range(balls)
    }

    /// Computes all bin boundaries of the PMNS tree once.
    ///
    /// Use the table to invert many outputs of this instance, as HintInit does
    /// for the offsets of one database block. A full tree costs about
    /// `log2(m) * n / 128` AES blocks and `8 * (m + 1)` bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// use plinko::iprf::{Iprf, IprfTable};
    ///
    /// let iprf: Iprf = Iprf::new([7u8; 16], 4096, 64);
    /// let table: IprfTable<'_> = iprf.table();
    /// for y in 0..64 {
    ///     assert_eq!(table.inverse(y), iprf.inverse(y));
    /// }
    /// ```
    pub fn table(&self) -> IprfTable<'_> {
        let loads: Vec<u64> = pmns_level_counts(
            self.domain,
            self.range,
            self.range.trailing_zeros(),
            |node, count| crate::binomial::coin_binomial(&self.cipher, node, count),
        );
        IprfTable {
            iprf: self,
            starts: bin_starts(&loads),
        }
    }

    /// Left counts of the cached top levels, computed on the first call.
    fn top_left_counts(&self) -> &[u32] {
        if self.cached_levels == 0 {
            return &[];
        }
        self.top_left_counts.get_or_init(|| {
            let mut left_counts: Vec<u32> = Vec::with_capacity((1usize << self.cached_levels) - 1);
            pmns_level_counts(
                self.domain,
                self.range,
                self.cached_levels,
                |node, count| {
                    let left: u64 = crate::binomial::coin_binomial(&self.cipher, node, count);
                    // The domain fits in u32 when cached_levels > 0.
                    left_counts.push(left as u32);
                    left
                },
            );
            left_counts.into_boxed_slice()
        })
    }

    /// Left count of the node at `depth` with breadth-first index `node_index`.
    #[inline]
    fn left_count(
        &self,
        top: &[u32],
        depth: u32,
        node_index: u64,
        node: PmnsNode,
        ball_count: u64,
    ) -> u64 {
        if depth < self.cached_levels {
            u64::from(top[node_index as usize])
        } else {
            crate::binomial::coin_binomial(&self.cipher, node, ball_count)
        }
    }

    /// Determines which PMNS bin a ball falls into for a given ball index.
    ///
    /// Given a total of `n` balls partitioned into `m` bins, performs the PMNS forward
    /// trace for ball with index `x_prime` (0-based) and returns the 0-based bin index
    /// that the ball is assigned to. If `m == 1`, returns `0`.
    fn trace_ball(&self, x_prime: u64, n: u64, m: u64) -> u64 {
        if m == 1 {
            return 0;
        }

        let top: &[u32] = self.top_left_counts();
        let mut low = 0u64;
        let mut high = m - 1;
        let mut ball_count = n;
        let mut ball_index = x_prime;
        let mut depth: u32 = 0;
        let mut node_index: u64 = 0;

        while low < high {
            let mid = (low + high) / 2;

            let left_count =
                self.left_count(top, depth, node_index, pmns_node(low, high), ball_count);

            if ball_index < left_count {
                high = mid;
                ball_count = left_count;
                node_index = 2 * node_index + 1;
            } else {
                low = mid + 1;
                ball_index -= left_count;
                ball_count -= left_count;
                node_index = 2 * node_index + 2;
            }
            depth += 1;
        }
        low
    }

    /// Returns the contiguous range of ball indices that fall into PMNS bin `y`.
    ///
    /// For `m == 1` this is all balls `[0..n)`. Otherwise the method walks the PMNS
    /// binary partitioning tree deterministically (using `coin_binomial`) to
    /// compute the starting index and count of balls assigned to bin `y`, and
    /// returns the range `[start, start + count)` as a `Vec<u64>`.
    ///
    /// # Parameters
    ///
    /// - `y`: target bin index in `[0, m)`.
    /// - `n`: total number of balls (domain size).
    /// - `m`: total number of bins (range size).
    ///
    /// # Returns
    ///
    /// The ball indices that map to bin `y` (the closed-open range
    /// `start..start+count`).
    fn trace_ball_inverse(&self, y: u64, n: u64, m: u64) -> Range<u64> {
        if m == 1 {
            return 0..n;
        }

        let top: &[u32] = self.top_left_counts();
        let mut low = 0u64;
        let mut high = m - 1;
        let mut ball_count = n;
        let mut ball_start = 0u64;
        let mut depth: u32 = 0;
        let mut node_index: u64 = 0;

        while low < high {
            let mid = (low + high) / 2;

            let left_count =
                self.left_count(top, depth, node_index, pmns_node(low, high), ball_count);

            if y <= mid {
                high = mid;
                ball_count = left_count;
                node_index = 2 * node_index + 1;
            } else {
                low = mid + 1;
                ball_start += left_count;
                ball_count -= left_count;
                node_index = 2 * node_index + 2;
            }
            depth += 1;
        }

        ball_start..ball_start + ball_count
    }
}

/// All bin boundaries of one [`Iprf`], from [`Iprf::table`].
///
/// The table is derived from the key and is as secret as the key.
pub struct IprfTable<'a> {
    iprf: &'a Iprf,
    /// `starts[y]` is the first ball of bin `y`; `starts[range]` is the domain.
    starts: Vec<u64>,
}

impl IprfTable<'_> {
    /// Returns the same preimages as [`Iprf::inverse`], in the same order.
    pub fn inverse(&self, y: u64) -> Vec<u64> {
        if y >= self.iprf.range {
            return vec![];
        }
        let balls: Range<u64> = self.starts[y as usize]..self.starts[y as usize + 1];
        self.iprf.prp.inverse_range(balls)
    }
}

/// Walks the first `levels` levels of the PMNS tree with `n` balls and `m`
/// bins in breadth-first order.
///
/// `split(node, count)` returns the number of balls that go to the left child.
/// Returns the ball counts of the nodes at depth `levels`, from left to right.
/// With `levels = log2(m)` these are the bin loads.
fn pmns_level_counts(
    n: u64,
    m: u64,
    levels: u32,
    mut split: impl FnMut(PmnsNode, u64) -> u64,
) -> Vec<u64> {
    debug_assert!(levels <= m.trailing_zeros());

    let mut counts: Vec<u64> = vec![n];
    for depth in 0..levels {
        let bins: u64 = m >> depth;
        let mut next: Vec<u64> = Vec::with_capacity(counts.len() * 2);
        for (i, &count) in counts.iter().enumerate() {
            let low: u64 = i as u64 * bins;
            let left: u64 = split(pmns_node(low, low + bins - 1), count);
            next.push(left);
            next.push(count - left);
        }
        counts = next;
    }
    counts
}

/// Prefix sums of the bin loads: entry `y` is the first ball of bin `y`, and
/// the last entry is the number of balls.
fn bin_starts(loads: &[u64]) -> Vec<u64> {
    let mut starts: Vec<u64> = Vec::with_capacity(loads.len() + 1);
    let mut next_start: u64 = 0;
    starts.push(next_start);
    for &load in loads {
        next_start += load;
        starts.push(next_start);
    }
    starts
}

/// Checks that a PMNS tree with `m` bins has only splits with probability 1/2
/// and that its bin indices fit the node encoding.
fn assert_pmns_range(m: u64) {
    assert!(
        m.is_power_of_two() && m <= 1 << 32,
        "iPRF range must be a power of two of at most 2^32, got {m}"
    );
}

/// Derives the PMNS keystream cipher for a tree with `n` balls and `m` bins.
///
/// The key depends on `n` and `m`, so two trees with different shapes never
/// share a keystream, even when they use the same iPRF key.
fn pmns_cipher(key: &PrfKey128, n: u64, m: u64) -> Aes128 {
    let mut hasher = Sha256::new();
    hasher.update(key);
    hasher.update(b"pmns");
    hasher.update(n.to_be_bytes());
    hasher.update(m.to_be_bytes());
    let hash = hasher.finalize();

    let mut pmns_key: PrfKey128 = [0u8; 16];
    pmns_key.copy_from_slice(&hash[0..16]);
    Aes128::new(&GenericArray::from(pmns_key))
}

/// Names the PMNS node that covers bins `low..=high`.
///
/// Both bounds are below the range, which [`assert_pmns_range`] limits to 2^32.
fn pmns_node(low: u64, high: u64) -> PmnsNode {
    debug_assert!(low <= high && high <= u64::from(u32::MAX));
    PmnsNode {
        low: low as u32,
        high: high as u32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_swap_or_not_inverse() {
        let key = [0u8; 16];
        let domain = 1000u64;
        let prp = SwapOrNot::new(key, domain);

        for x in 0..100 {
            let y = prp.forward(x);
            let x_recovered = prp.inverse(y);
            assert_eq!(x, x_recovered, "PRP inverse failed for x={}", x);
        }
    }

    #[test]
    fn test_swap_or_not_is_permutation() {
        let key = [1u8; 16];
        let domain = 100u64;
        let prp = SwapOrNot::new(key, domain);

        let mut outputs: Vec<u64> = (0..domain).map(|x| prp.forward(x)).collect();
        outputs.sort();
        outputs.dedup();
        assert_eq!(outputs.len(), domain as usize, "PRP is not a permutation");
    }

    #[test]
    fn test_iprf_inverse_contains_preimage() {
        let key = [2u8; 16];
        let domain = 1000u64;
        let range = 128u64;
        let iprf = Iprf::new(key, domain, range);

        for x in 0..50 {
            let y = iprf.forward(x);
            let preimages = iprf.inverse(y);
            assert!(
                preimages.contains(&x),
                "iPRF inverse for y={} does not contain original x={}",
                y,
                x
            );
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 64,
            .. ProptestConfig::default()
        })]

        #[test]
        fn swap_or_not_inverse_roundtrip(
            key in any::<[u8; 16]>(),
            domain in 1u64..10_000,
            x in any::<u64>(),
        ) {
            let x = x % domain;
            let prp = SwapOrNot::new(key, domain);

            let y = prp.forward(x);
            prop_assert!(y < domain);

            let x2 = prp.inverse(y);
            prop_assert_eq!(x, x2);
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 32,
            .. ProptestConfig::default()
        })]

        #[test]
        fn swap_or_not_is_permutation_for_small_domains(
            key in any::<[u8; 16]>(),
            domain in 1u64..65,
        ) {
            let prp = SwapOrNot::new(key, domain);

            let mut outputs: Vec<u64> = (0..domain).map(|x| prp.forward(x)).collect();
            outputs.sort_unstable();
            outputs.dedup();

            prop_assert_eq!(outputs.len() as u64, domain);
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 64,
            .. ProptestConfig::default()
        })]

        #[test]
        fn iprf_inverse_contains_preimage_and_is_consistent(
            key in any::<[u8; 16]>(),
            n in 1u64..1_000,
            m_log in 0u32..11,
            x_raw in any::<u64>(),
        ) {
            let m = 1u64 << m_log;
            let x = x_raw % n;

            let iprf = Iprf::new(key, n, m);

            let y = iprf.forward(x);
            prop_assert!(y < m);

            let preimages = iprf.inverse(y);
            prop_assert!(preimages.contains(&x));

            for &x2 in &preimages {
                prop_assert!(x2 < n);
                prop_assert_eq!(y, iprf.forward(x2));
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 32,
            .. ProptestConfig::default()
        })]

        #[test]
        fn iprf_inverse_partitions_domain(
            key in any::<[u8; 16]>(),
            n in 1u64..64,
            m_log in 0u32..8,
        ) {
            let m = 1u64 << m_log;

            let iprf = Iprf::new(key, n, m);
            let mut seen = vec![false; n as usize];

            for y in 0..m {
                for x in iprf.inverse(y) {
                    prop_assert!(x < n);
                    prop_assert!(
                        !seen[x as usize],
                        "element {} appears in multiple inverse bins", x
                    );
                    seen[x as usize] = true;
                }
            }

            prop_assert!(seen.iter().all(|&b| b));
        }
    }

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

    fn pmns_test_key(index: u8) -> PrfKey128 {
        let mut key: PrfKey128 = [0x5Au8; 16];
        key[0] = index;
        key
    }

    /// PMNS bin loads must look like 16384 balls thrown into 64 bins.
    ///
    /// Pearson's statistic has 63 degrees of freedom: mean 63, standard
    /// deviation about 11.2.
    #[test]
    fn test_pmns_bin_loads_are_multinomial() {
        let domain: u64 = 16_384;
        let range: u64 = 64;
        let expected_load: f64 = domain as f64 / range as f64;

        for index in 0..8u8 {
            let iprf: Iprf = Iprf::new(pmns_test_key(index), domain, range);
            let loads: Vec<u64> = (0..range)
                .map(|y| iprf.trace_ball_inverse(y, domain, range).count() as u64)
                .collect();

            assert_eq!(loads.iter().sum::<u64>(), domain, "key {index}");

            let chi_square: f64 = loads
                .iter()
                .map(|&load| (load as f64 - expected_load).powi(2) / expected_load)
                .sum();
            assert!(
                chi_square < 155.0,
                "key {index}: chi-square {chi_square} for loads {loads:?}"
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

    /// Ball range `(start, count)` of each bin, from the per-call tree walk.
    fn walked_ball_ranges(iprf: &Iprf, domain: u64, range: u64) -> Vec<(u64, u64)> {
        let mut ranges: Vec<(u64, u64)> = Vec::new();
        let mut next_start: u64 = 0;
        for y in 0..range {
            let balls: Vec<u64> = iprf.trace_ball_inverse(y, domain, range).collect();
            if let Some(&first) = balls.first() {
                assert_eq!(
                    first,
                    next_start,
                    "bin {y} does not start after bin {}",
                    y.max(1) - 1
                );
            }
            ranges.push((next_start, balls.len() as u64));
            next_start += balls.len() as u64;
        }
        assert_eq!(next_start, domain, "the bins must hold every ball");
        ranges
    }

    fn table_ball_ranges(starts: &[u64]) -> Vec<(u64, u64)> {
        starts
            .windows(2)
            .map(|pair| (pair[0], pair[1] - pair[0]))
            .collect()
    }

    /// The table of all bin boundaries must equal the per-call tree walk.
    ///
    /// The domains are not multiples of the keystream block size, and one
    /// range has a single bin.
    #[test]
    fn test_pmns_table_matches_tree_walk() {
        for (domain, range) in [(100_003u64, 64u64), (5_000, 1_024), (777, 2), (300, 1)] {
            for index in 0..2u8 {
                let iprf: Iprf = Iprf::new(pmns_test_key(index), domain, range);
                let walked: Vec<(u64, u64)> = walked_ball_ranges(&iprf, domain, range);
                assert_eq!(
                    table_ball_ranges(&iprf.table().starts),
                    walked,
                    "domain {domain}, range {range}, key {index}"
                );
            }
        }
    }

    /// Cached top levels must not change the tree, for every number of levels
    /// from none to more than the tree depth.
    #[test]
    fn test_pmns_cached_levels_match_uncached_tree() {
        let domain: u64 = 100_003;
        let range: u64 = 64;
        let key: PrfKey128 = pmns_test_key(3);

        let plain: Iprf = Iprf::new(key, domain, range);
        let walked: Vec<(u64, u64)> = walked_ball_ranges(&plain, domain, range);
        let ball_indices = || (0..domain).step_by(97).chain([domain - 1]);
        let bins: Vec<u64> = ball_indices()
            .map(|ball| plain.trace_ball(ball, domain, range))
            .collect();
        assert!(
            bins.iter().any(|&bin| bin != bins[0]),
            "balls must reach several bins"
        );

        for levels in [1u32, 2, 3, 5, 6, 9] {
            let cached: Iprf = Iprf::with_cached_levels(key, domain, range, levels);
            assert_eq!(cached.cached_levels, levels.min(6));
            assert_eq!(
                walked_ball_ranges(&cached, domain, range),
                walked,
                "levels {levels}"
            );
            let cached_bins: Vec<u64> = ball_indices()
                .map(|ball| cached.trace_ball(ball, domain, range))
                .collect();
            assert_eq!(cached_bins, bins, "levels {levels}");
            assert_eq!(
                cached.top_left_counts().len(),
                (1usize << levels.min(6)) - 1,
                "levels {levels}"
            );
        }
    }

    /// A domain above `u32::MAX` disables the cache instead of truncating counts.
    #[test]
    fn test_pmns_cache_is_off_for_a_domain_above_u32() {
        let iprf: Iprf = Iprf::with_cached_levels(pmns_test_key(0), 1 << 33, 4, 2);
        assert_eq!(iprf.cached_levels, 0);
        assert!(iprf.top_left_counts().is_empty());
    }

    /// `forward` and `inverse` with the cache and with the table must equal
    /// the plain instance, PRP included.
    #[test]
    fn test_iprf_table_and_cache_match_plain_iprf() {
        let key: PrfKey128 = pmns_test_key(5);
        let domain: u64 = 700;
        let range: u64 = 16;

        let plain: Iprf = Iprf::new(key, domain, range);
        let cached: Iprf = Iprf::with_cached_levels(key, domain, range, 3);
        let table: IprfTable<'_> = plain.table();

        let mut seen: u64 = 0;
        for y in 0..range {
            let expected: Vec<u64> = plain.inverse(y);
            assert_eq!(table.inverse(y), expected, "table, y={y}");
            assert_eq!(cached.inverse(y), expected, "cache, y={y}");
            seen += expected.len() as u64;
        }
        assert_eq!(seen, domain);
        assert!(table.inverse(range).is_empty());

        for x in (0..domain).step_by(7) {
            assert_eq!(cached.forward(x), plain.forward(x), "x={x}");
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

    /// Two trees with the same key and different shapes must not share coins.
    #[test]
    fn test_pmns_keystream_depends_on_tree_shape() {
        let key: PrfKey128 = pmns_test_key(0);
        let range: u64 = 64;
        let loads = |domain: u64| -> Vec<u64> {
            let iprf: Iprf = Iprf::new(key, domain, range);
            (0..range)
                .map(|y| iprf.trace_ball_inverse(y, domain, range).count() as u64)
                .collect()
        };

        let small: Vec<u64> = loads(16_384);
        let large: Vec<u64> = loads(16_385);
        let equal_bins: usize = small.iter().zip(&large).filter(|(a, b)| a == b).count();
        assert!(
            equal_bins < 16,
            "{equal_bins} of 64 bins have equal loads for domains 16384 and 16385"
        );
    }

    #[test]
    #[should_panic(expected = "iPRF range must be a power of two")]
    fn test_iprf_rejects_range_that_is_not_a_power_of_two() {
        let _ = Iprf::new([0u8; 16], 1000, 100);
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
    fn test_swap_or_not_sr_inverse() {
        let key = [5u8; 16];
        let domain = 100u64;
        let prp = SwapOrNotSr::new(key, domain);

        for x in 0..domain {
            let y = prp.forward(x);
            let x_recovered = prp.inverse(y);
            assert_eq!(x, x_recovered, "SR inverse failed for x={}", x);
        }
    }

    #[test]
    #[ignore = "slow with 128-bit security; run with --ignored"]
    fn test_swap_or_not_sr_is_permutation() {
        let key = [6u8; 16];
        let domain = 64u64;
        let prp = SwapOrNotSr::new(key, domain);

        let mut outputs: Vec<u64> = (0..domain).map(|x| prp.forward(x)).collect();
        outputs.sort();
        outputs.dedup();
        assert_eq!(
            outputs.len(),
            domain as usize,
            "SR PRP is not a permutation"
        );
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

    /// Ranges of at most `width` values around the stage boundaries of `domain`.
    fn boundary_ranges(domain: u64, width: u64) -> Vec<Range<u64>> {
        let mut ranges: Vec<Range<u64>> =
            vec![0..width.min(domain), domain.saturating_sub(width)..domain];
        let mut size: u64 = domain / 2;
        for _ in 0..3 {
            let low: u64 = size.saturating_sub(width / 2);
            ranges.push(low..(low + width).min(domain));
            size /= 2;
        }
        ranges
    }

    /// `inverse_range` must equal `inverse` for every range of a small domain,
    /// including the domains with one and two values.
    #[test]
    fn test_sr_inverse_range_matches_inverse_on_small_domains() {
        for domain in 1..=33u64 {
            let prp: SwapOrNotSr = SwapOrNotSr::with_security([domain as u8; 16], domain, 16);
            let single: Vec<u64> = (0..domain).map(|y| prp.inverse(y)).collect();
            for start in 0..=domain {
                for end in start..=domain {
                    assert_eq!(
                        prp.inverse_range(start..end),
                        single[start as usize..end as usize],
                        "domain {domain}, range {start}..{end}"
                    );
                }
            }
        }
    }

    /// Full round counts, domains above 65536: a range that crosses a stage
    /// boundary holds values with different stage counts.
    #[test]
    fn test_sr_inverse_range_matches_inverse_at_stage_boundaries() {
        for domain in [100_003u64, 131_072] {
            let prp: SwapOrNotSr = SwapOrNotSr::new([0x5a; 16], domain);
            for balls in boundary_ranges(domain, 40) {
                let single: Vec<u64> = balls.clone().map(|y| prp.inverse(y)).collect();
                assert_eq!(
                    prp.inverse_range(balls.clone()),
                    single,
                    "domain {domain}, range {balls:?}"
                );
                for (&x, y) in single.iter().zip(balls) {
                    assert_eq!(prp.forward(x), y, "domain {domain}");
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "balls must be < domain")]
    fn test_sr_inverse_range_rejects_a_ball_outside_the_domain() {
        let prp: SwapOrNotSr = SwapOrNotSr::with_security([1u8; 16], 100, 16);
        prp.inverse_range(90..101);
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

    /// The batched iPRF inverses must return the single-value PRP inverse of
    /// each ball of the bin, in ball order, on a domain above 65536.
    #[test]
    fn test_iprf_inverse_matches_single_value_prp() {
        let key: PrfKey128 = pmns_test_key(7);
        let domain: u64 = 100_003;
        let range: u64 = 512;

        let iprf: Iprf = Iprf::new(key, domain, range);
        let table: IprfTable<'_> = iprf.table();
        let iprf_tee: IprfTee = IprfTee::new(key, domain, range);
        let tee_table: IprfTeeTable<'_> = iprf_tee.table_ct();
        assert!(!tee_table.overflowed());

        for y in [0u64, 255, 511] {
            let balls: Range<u64> = iprf.trace_ball_inverse(y, domain, range);
            let expected: Vec<u64> = balls.map(|z| iprf.prp.inverse(z)).collect();
            assert!(expected.len() > 100, "y={y}");

            assert_eq!(iprf.inverse(y), expected, "Iprf::inverse, y={y}");
            assert_eq!(table.inverse(y), expected, "IprfTable::inverse, y={y}");
            let (got, count) = tee_table.inverse_ct(y);
            assert_eq!(got[..count], expected, "IprfTeeTable::inverse_ct, y={y}");
            assert!(got[count..].iter().all(|&x| x == 0), "y={y}");
            for &x in expected.iter().step_by(37) {
                assert_eq!(iprf.forward(x), y);
            }
        }
    }

    #[test]
    fn test_sr_num_stages() {
        assert_eq!(sr_num_stages(1), 0);
        assert_eq!(sr_num_stages(2), 0);
        assert_eq!(sr_num_stages(3), 1);
        assert_eq!(sr_num_stages(4), 1);
        assert_eq!(sr_num_stages(5), 1);
        assert_eq!(sr_num_stages(6), 2);
        assert_eq!(sr_num_stages(8), 2);
        assert_eq!(sr_num_stages(16), 3);
        assert_eq!(sr_num_stages(64), 5);
        assert_eq!(sr_num_stages(1000), 9);
        assert_eq!(sr_num_stages(1_000_000), 19);
    }

    #[test]
    fn test_sr_t_rounds_with_params_paper_example() {
        let n = 10u64.pow(16);
        let epsilon = 10.0_f64.powi(-10);
        let lambda = (-epsilon.log2()).ceil() as u32;

        let num_stages = sr_num_stages(n);
        let rounds_at_top = sr_t_rounds_with_params(n, num_stages, lambda);

        assert!(
            rounds_at_top >= 500,
            "Expected 500+ per-stage rounds for N=10^16, eps=10^-10 (got {})",
            rounds_at_top
        );
        assert!(
            rounds_at_top <= 700,
            "Per-stage rounds too high for N=10^16, eps=10^-10 (got {})",
            rounds_at_top
        );
    }

    #[test]
    fn test_sr_t_rounds_128bit_security() {
        let n0 = 1000u64;
        let lambda = 128u32;

        let rounds_at_n0 = sr_t_rounds_with_security(n0, n0, lambda);
        let rounds_at_half = sr_t_rounds_with_security(n0 / 2, n0, lambda);

        assert!(
            rounds_at_n0 > 600,
            "128-bit security needs significant rounds (got {} for N=1000)",
            rounds_at_n0
        );

        assert!(
            rounds_at_half < rounds_at_n0,
            "Smaller levels should have fewer rounds"
        );
    }

    #[test]
    fn test_sr_t_rounds_lower_security() {
        let n0 = 1000u64;
        let lambda_high = 128u32;
        let lambda_low = 64u32;

        let rounds_high = sr_t_rounds_with_security(n0, n0, lambda_high);
        let rounds_low = sr_t_rounds_with_security(n0, n0, lambda_low);

        assert!(
            rounds_high > rounds_low,
            "Higher security should need more rounds"
        );

        let expected_diff = 4.82 * ((lambda_high - lambda_low) as f64);
        let actual_diff = (rounds_high - rounds_low) as f64;
        assert!(
            (actual_diff - expected_diff).abs() < 2.0,
            "Round difference should be ~4.82 * delta_lambda"
        );
    }

    #[test]
    fn test_sr_with_custom_security() {
        let key = [9u8; 16];
        let domain = 64u64;

        let prp_32 = SwapOrNotSr::with_security(key, domain, 32);

        for x in 0..domain {
            let y = prp_32.forward(x);
            let x_recovered = prp_32.inverse(y);
            assert_eq!(x, x_recovered, "32-bit SR inverse failed at x={}", x);
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

    #[test]
    fn test_sr_is_permutation_low_security() {
        let key = [11u8; 16];
        let domain = 64u64;
        let prp = SwapOrNotSr::with_security(key, domain, 32);

        let mut outputs: Vec<u64> = (0..domain).map(|x| prp.forward(x)).collect();
        outputs.sort();
        outputs.dedup();
        assert_eq!(
            outputs.len(),
            domain as usize,
            "SR PRP is not a permutation"
        );
    }
}
