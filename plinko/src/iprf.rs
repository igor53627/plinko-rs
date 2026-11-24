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

#[cfg(feature = "tee")]
mod tee;
#[cfg(feature = "tee")]
pub use tee::{IprfTee, IprfTeeTable, SwapOrNotSrTee, SwapOrNotTee, MAX_PREIMAGES};

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
        let partner = sub_mod(k_i, x % self.domain, self.domain);
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
#[cfg(any(feature = "tee", test))]
fn sr_t_rounds_with_security(n: u64, n0: u64, lambda: u32) -> usize {
    let num_stages = sr_num_stages(n0);
    sr_t_rounds_with_params(n, num_stages, lambda)
}

/// Stage sizes of the Sometimes-Recurse shuffle: `domain`, then its halves,
/// while the size is above 1. Stage `l` permutes the values below `sizes[l]`.
#[cfg(feature = "tee")]
fn sr_level_sizes(domain: u64, max_levels: usize) -> Vec<u64> {
    let mut sizes: Vec<u64> = Vec::with_capacity(max_levels);
    let mut n: u64 = domain;
    while n > 1 && sizes.len() < max_levels {
        sizes.push(n);
        n /= 2;
    }
    sizes
}

/// Maps a 128-bit value to `0..n`: `floor(r * n / 2^128)`.
///
/// For a uniform `r`, each result has `floor(2^128 / n)` or `ceil(2^128 / n)`
/// preimages, so the statistical distance from uniform is at most `n / 2^128`.
/// The function has two widening multiplications and no division, so its time
/// does not depend on `r`.
fn reduce_below(r: u128, n: u64) -> u64 {
    let high: u128 = u128::from((r >> 64) as u64) * u128::from(n);
    let low: u128 = u128::from(r as u64) * u128::from(n);
    ((high + (low >> 64)) >> 64) as u64
}

/// Round key of round `round` of the Sometimes-Recurse stage `level`, which
/// has size `n`: a value in `0..n`.
///
/// The key is one AES block, `AES(level || round || n)`, reduced with
/// [`reduce_below`]. `level`, `round` and `n` are public.
///
/// The stage of size 2 has the fixed key 1. Its partner function is then
/// `x -> 1 - x`, and the swap bit of its one round alone decides the order of
/// the two values. With key 0 the round never swaps, so a random key gives a
/// swap with probability 1/4 and not 1/2.
#[cfg(any(feature = "tee", test))]
fn sr_round_key(cipher: &Aes128, level: usize, round: usize, n: u64) -> u64 {
    if n == 2 {
        return 1;
    }
    let mut input = [0u8; 16];
    let combined = ((level as u64) << 32) | (round as u64);
    input[0..8].copy_from_slice(&combined.to_be_bytes());
    input[8..16].copy_from_slice(&n.to_be_bytes());

    let mut block = GenericArray::from(input);
    cipher.encrypt_block(&mut block);

    reduce_below(u128::from_be_bytes(block.into()), n)
}

/// AES input prefix of the swap decision of one round.
fn sr_swap_prefix(level: usize, round: usize) -> [u8; 8] {
    (((level as u64) << 32) | (round as u64) | 0x80000000).to_be_bytes()
}

/// Number of round keys derived per `encrypt_blocks` call in [`SwapOrNotSr`].
const SR_KEY_BATCH: usize = 16;

/// Derives round keys for rounds `first_round..first_round + out.len()` of stage
/// `(level, n)` in one batched AES call.
///
/// Matches [`sr_round_key`] for every round. `out.len()` must not exceed
/// [`SR_KEY_BATCH`].
#[inline]
fn sr_round_keys_batch(cipher: &Aes128, level: usize, first_round: usize, n: u64, out: &mut [u64]) {
    debug_assert!(out.len() <= SR_KEY_BATCH);
    if n == 2 {
        out.fill(1);
        return;
    }
    let level_hi: u64 = (level as u64) << 32;
    let n_be: [u8; 8] = n.to_be_bytes();
    let mut blocks: [Block; SR_KEY_BATCH] = [Block::default(); SR_KEY_BATCH];
    for (i, block) in blocks.iter_mut().enumerate().take(out.len()) {
        let combined: u64 = level_hi | ((first_round + i) as u64);
        block[0..8].copy_from_slice(&combined.to_be_bytes());
        block[8..16].copy_from_slice(&n_be);
    }
    cipher.encrypt_blocks(&mut blocks[..out.len()]);
    for (key_slot, block) in out.iter_mut().zip(blocks.iter()) {
        *key_slot = reduce_below(u128::from_be_bytes((*block).into()), n);
    }
}

/// Applies one branchless Swap-or-Not round to `val` (with `val < n` and `k_i < n`).
#[inline]
fn sr_round_step(cipher: &Aes128, level: usize, round: usize, n: u64, k_i: u64, val: u64) -> u64 {
    // val < n and k_i < n, so this is (k_i - val) mod n without division or overflow.
    let wrap: u64 = n & u64::from(k_i < val).wrapping_neg();
    let partner: u64 = k_i.wrapping_sub(val).wrapping_add(wrap);
    let canonical: u64 = val.max(partner);

    let mut block: Block = Block::default();
    block[0..8].copy_from_slice(&sr_swap_prefix(level, round));
    block[8..16].copy_from_slice(&canonical.to_be_bytes());
    cipher.encrypt_block(&mut block);

    let swap: u64 = u64::from(block[0] & 1).wrapping_neg();
    val ^ (swap & (val ^ partner))
}

/// Runs `t` forward Swap-or-Not rounds of stage `(level, n)` for `L` independent
/// PRPs in lockstep so hardware AES pipelines overlap across lanes.
#[inline]
fn forward_stage_lanes<const L: usize>(
    prps: &[&SwapOrNotSr],
    ids: &[usize; L],
    level: usize,
    n: u64,
    t: usize,
    vals: &mut [u64],
) {
    let ciphers: [&Aes128; L] = std::array::from_fn(|l| &prps[ids[l]].cipher);
    let mut lane_vals: [u64; L] = std::array::from_fn(|l| vals[ids[l]]);
    let mut lane_keys: [[u64; SR_KEY_BATCH]; L] = [[0u64; SR_KEY_BATCH]; L];
    let mut partners: [u64; L] = [0u64; L];
    let mut swap_blocks: [Block; L] = [Block::default(); L];

    let mut start: usize = 0;
    while start < t {
        let batch_len: usize = (t - start).min(SR_KEY_BATCH);
        for l in 0..L {
            sr_round_keys_batch(ciphers[l], level, start, n, &mut lane_keys[l][..batch_len]);
        }
        for round in start..start + batch_len {
            let r_off: usize = round - start;
            let prefix: [u8; 8] = sr_swap_prefix(level, round);
            for l in 0..L {
                let val: u64 = lane_vals[l];
                let k_i: u64 = lane_keys[l][r_off];
                let wrap: u64 = n & u64::from(k_i < val).wrapping_neg();
                let partner: u64 = k_i.wrapping_sub(val).wrapping_add(wrap);
                partners[l] = partner;
                swap_blocks[l][0..8].copy_from_slice(&prefix);
                swap_blocks[l][8..16].copy_from_slice(&val.max(partner).to_be_bytes());
            }
            for l in 0..L {
                ciphers[l].encrypt_block(&mut swap_blocks[l]);
            }
            for l in 0..L {
                let swap: u64 = u64::from(swap_blocks[l][0] & 1).wrapping_neg();
                lane_vals[l] ^= swap & (lane_vals[l] ^ partners[l]);
            }
        }
        start += batch_len;
    }
    for l in 0..L {
        vals[ids[l]] = lane_vals[l];
    }
}

/// Sometimes-Recurse PRP wrapper (Morris-Rogaway Fig. 1)
///
/// Provides full-domain security: secure even when adversary queries all N elements.
/// Uses SwapOrNot as inner shuffle with recursive halving.
///
/// Round counts are computed per Morris-Rogaway Section 5 "Strategy 1" (equal error split)
/// with 128-bit security by default.
///
/// A round key is a 128-bit AES output reduced to the stage size without a division
/// (statistical distance from uniform at most `n / 2^128`), and the stage of size 2
/// has one round with the fixed key 1. See "Round Keys" in `docs/hint_generation.md`.
pub struct SwapOrNotSr {
    cipher: Aes128,
    domain: u64,
    max_levels: usize,
    num_stages: usize,
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
        let num_stages = sr_num_stages(domain);

        Self {
            cipher,
            domain,
            max_levels,
            num_stages,
            security_bits,
        }
    }

    fn apply_rounds_forward(&self, level: usize, n: u64, x: u64) -> u64 {
        let t = sr_t_rounds_with_params(n, self.num_stages, self.security_bits);
        let mut val = x;
        let mut keys: [u64; SR_KEY_BATCH] = [0u64; SR_KEY_BATCH];
        let mut start: usize = 0;
        while start < t {
            let len: usize = (t - start).min(SR_KEY_BATCH);
            sr_round_keys_batch(&self.cipher, level, start, n, &mut keys[..len]);
            for (i, &k_i) in keys[..len].iter().enumerate() {
                val = sr_round_step(&self.cipher, level, start + i, n, k_i, val);
            }
            start += len;
        }
        val
    }

    fn apply_rounds_inverse(&self, level: usize, n: u64, y: u64) -> u64 {
        let t = sr_t_rounds_with_params(n, self.num_stages, self.security_bits);
        let mut val = y;
        let mut keys: [u64; SR_KEY_BATCH] = [0u64; SR_KEY_BATCH];
        let mut end: usize = t;
        while end > 0 {
            let start: usize = end.saturating_sub(SR_KEY_BATCH);
            let len: usize = end - start;
            sr_round_keys_batch(&self.cipher, level, start, n, &mut keys[..len]);
            for i in (0..len).rev() {
                val = sr_round_step(&self.cipher, level, start + i, n, keys[i], val);
            }
            end = start;
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

    /// Evaluates [`SwapOrNotSr::forward`] at `x` for every PRP in `prps`.
    ///
    /// Element `i` of the result equals `prps[i].forward(x)`. All PRPs must
    /// share the same domain and security parameter.
    ///
    /// Interleaving independent PRPs in 8-way batches lets CPUs with pipelined
    /// AES units overlap their independent encryptions on a single core while
    /// producing outputs bit-identical to calling [`SwapOrNotSr::forward`] on
    /// each PRP.
    ///
    /// # Panics
    /// Panics if `x` is not below the domain or if any PRP in `prps` has a
    /// different domain or security parameter from `prps[0]`.
    ///
    /// # Examples
    ///
    /// ```
    /// use plinko::iprf::SwapOrNotSr;
    ///
    /// let prps: Vec<SwapOrNotSr> = (0u8..10)
    ///     .map(|k| SwapOrNotSr::with_security([k; 16], 1000, 16))
    ///     .collect();
    /// let refs: Vec<&SwapOrNotSr> = prps.iter().collect();
    /// let batch: Vec<u64> = SwapOrNotSr::forward_many(&refs, 123);
    /// let scalar: Vec<u64> = prps.iter().map(|p| p.forward(123)).collect();
    /// assert_eq!(batch, scalar);
    /// ```
    pub fn forward_many(prps: &[&Self], x: u64) -> Vec<u64> {
        const LANES: usize = 8;

        if prps.is_empty() {
            return Vec::new();
        }
        let domain: u64 = prps[0].domain;
        let max_levels: usize = prps[0].max_levels;
        let num_stages: usize = prps[0].num_stages;
        let security_bits: u32 = prps[0].security_bits;
        assert!(x < domain, "x must be < domain");
        for prp in &prps[1..] {
            assert_eq!(prp.domain, domain, "all PRPs must share the same domain");
            assert_eq!(
                prp.security_bits, security_bits,
                "all PRPs must share the same security_bits"
            );
        }

        let mut vals: Vec<u64> = vec![x; prps.len()];
        let mut active: Vec<usize> = (0..prps.len()).collect();
        let mut n: u64 = domain;

        for level in 0..max_levels {
            if n <= 1 || active.is_empty() {
                break;
            }
            let t: usize = sr_t_rounds_with_params(n, num_stages, security_bits);
            let mut chunks = active.chunks_exact(LANES);
            for chunk in chunks.by_ref() {
                let ids: [usize; LANES] = chunk.try_into().expect("chunks_exact(LANES)");
                forward_stage_lanes::<LANES>(prps, &ids, level, n, t, &mut vals);
            }
            let rem: &[usize] = chunks.remainder();
            if rem.len() >= 4 {
                let ids: [usize; 4] = rem[..4].try_into().expect("len >= 4");
                forward_stage_lanes::<4>(prps, &ids, level, n, t, &mut vals);
                for &idx in &rem[4..] {
                    vals[idx] = prps[idx].apply_rounds_forward(level, n, vals[idx]);
                }
            } else {
                for &idx in rem {
                    vals[idx] = prps[idx].apply_rounds_forward(level, n, vals[idx]);
                }
            }

            let half: u64 = n / 2;
            active.retain(|&idx| vals[idx] < half);
            n = half;
        }
        vals
    }

    pub fn inverse(&self, y: u64) -> u64 {
        assert!(y < self.domain, "y must be < domain");

        let mut sizes: [u64; 64] = [0u64; 64];
        let mut num_levels: usize = 0;
        let mut n = self.domain;
        for _ in 0..self.max_levels {
            if n <= 1 {
                break;
            }
            sizes[num_levels] = n;
            num_levels += 1;
            let half = n / 2;
            if y >= half {
                break;
            }
            n = half;
        }

        let mut val = y;
        for level in (0..num_levels).rev() {
            val = self.apply_rounds_inverse(level, sizes[level], val);
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
        if balls.is_empty() {
            return Vec::new();
        }
        let mut vals: Vec<u64> = balls.clone().collect();
        let mut partners: Vec<u64> = vec![0u64; vals.len()];
        let mut blocks: Vec<Block> = vec![Block::default(); vals.len()];
        let mut keys: [u64; SR_KEY_BATCH] = [0u64; SR_KEY_BATCH];

        let mut sizes: [u64; 64] = [0u64; 64];
        let mut num_levels: usize = 0;
        let mut size_n: u64 = self.domain;
        while size_n > 1 && num_levels < self.max_levels {
            sizes[num_levels] = size_n;
            num_levels += 1;
            size_n /= 2;
        }

        for level in (0..num_levels).rev() {
            let n: u64 = sizes[level];
            let active: usize = n.saturating_sub(balls.start).min(vals.len() as u64) as usize;
            if active == 0 {
                continue;
            }
            let t: usize = sr_t_rounds_with_params(n, self.num_stages, self.security_bits);
            let mut end: usize = t;
            while end > 0 {
                let start: usize = end.saturating_sub(SR_KEY_BATCH);
                let len: usize = end - start;
                sr_round_keys_batch(&self.cipher, level, start, n, &mut keys[..len]);
                for k_idx in (0..len).rev() {
                    let round: usize = start + k_idx;
                    let k_i: u64 = keys[k_idx];
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
                end = start;
            }
        }
        vals
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
    ///
    /// # Panics
    /// Panics if `x` is not in `[0, domain)`. The function has no value there,
    /// and bin 0 is a valid output.
    pub fn forward(&self, x: u64) -> u64 {
        assert!(
            x < self.domain,
            "Iprf::forward: x={x} is outside the domain [0, {})",
            self.domain
        );
        // Apply PRP first, then PMNS
        let permuted = self.prp.forward(x);
        self.trace_ball(permuted, self.domain, self.range)
    }

    /// Evaluates [`Iprf::forward`] at `x` for every iPRF in `iprfs`.
    ///
    /// Element `i` of the result equals `iprfs[i].forward(x)`. All iPRFs must
    /// have the same domain.
    ///
    /// # Panics
    /// Panics if `x` is not in `[0, domain)` or if the iPRFs do not share the
    /// same domain.
    ///
    /// # Examples
    ///
    /// ```
    /// use plinko::iprf::Iprf;
    ///
    /// let iprfs: Vec<Iprf> = (0u8..10)
    ///     .map(|k| Iprf::new([k; 16], 4096, 64))
    ///     .collect();
    /// let refs: Vec<&Iprf> = iprfs.iter().collect();
    /// let batch: Vec<u64> = Iprf::forward_many(&refs, 123);
    /// let scalar: Vec<u64> = iprfs.iter().map(|f| f.forward(123)).collect();
    /// assert_eq!(batch, scalar);
    /// ```
    pub fn forward_many(iprfs: &[&Self], x: u64) -> Vec<u64> {
        if iprfs.is_empty() {
            return Vec::new();
        }
        let domain: u64 = iprfs[0].domain;
        assert!(
            x < domain,
            "Iprf::forward_many: x={x} is outside the domain [0, {domain})"
        );
        let prps: Vec<&SwapOrNotSr> = iprfs.iter().map(|iprf| &iprf.prp).collect();
        let permuted: Vec<u64> = SwapOrNotSr::forward_many(&prps, x);
        iprfs
            .iter()
            .zip(permuted)
            .map(|(iprf, p)| iprf.trace_ball(p, iprf.domain, iprf.range))
            .collect()
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

/// Returns `(k - x) mod n` for `k < n` and `x < n`.
///
/// `k + n - x` overflows for `n` above 2^63; this form does not.
#[inline]
fn sub_mod(k: u64, x: u64, n: u64) -> u64 {
    let wrap: u64 = n & u64::from(k < x).wrapping_neg();
    k.wrapping_sub(x).wrapping_add(wrap)
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

    pub(super) fn pmns_test_key(index: u8) -> PrfKey128 {
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

    /// Ball range `(start, count)` of each bin, from the per-call tree walk.
    pub(super) fn walked_ball_ranges(iprf: &Iprf, domain: u64, range: u64) -> Vec<(u64, u64)> {
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

    pub(super) fn table_ball_ranges(starts: &[u64]) -> Vec<(u64, u64)> {
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
    #[should_panic(expected = "outside the domain")]
    fn test_iprf_forward_rejects_input_outside_the_domain() {
        let iprf = Iprf::new([7u8; 16], 100, 16);
        let _ = iprf.forward(100);
    }

    #[test]
    #[should_panic(expected = "iPRF range must be a power of two")]
    fn test_iprf_rejects_range_that_is_not_a_power_of_two() {
        let _ = Iprf::new([0u8; 16], 1000, 100);
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

    /// Ranges of at most `width` values around the stage boundaries of `domain`.
    pub(super) fn boundary_ranges(domain: u64, width: u64) -> Vec<Range<u64>> {
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

    /// A stage above 2^63: `k_i + n` does not fit in 64 bits there.
    #[test]
    fn test_prp_partner_does_not_overflow_on_the_largest_domain() {
        let prp: SwapOrNotSr = SwapOrNotSr::with_security([3u8; 16], u64::MAX, 16);
        for y in [0, 1, u64::MAX / 2, u64::MAX - 1] {
            let batch: u64 = prp.inverse_range(y..y + 1)[0];
            assert_eq!(prp.inverse(y), batch, "y={y}");
            assert_eq!(prp.forward(batch), y, "y={y}");
        }

        let one_level: SwapOrNot = SwapOrNot::new([3u8; 16], u64::MAX);
        for x in [0, 1, u64::MAX / 2, u64::MAX - 1] {
            let y: u64 = one_level.forward(x);
            assert!(y < u64::MAX, "x={x}");
            assert_eq!(one_level.inverse(y), x, "x={x}");
        }

        assert_eq!(sub_mod(0, u64::MAX - 1, u64::MAX), 1);
        assert_eq!(sub_mod(u64::MAX - 1, 0, u64::MAX), u64::MAX - 1);
        assert_eq!(sub_mod(5, 7, 10), 8);
    }

    #[test]
    #[should_panic(expected = "balls must be < domain")]
    fn test_sr_inverse_range_rejects_a_ball_outside_the_domain() {
        let prp: SwapOrNotSr = SwapOrNotSr::with_security([1u8; 16], 100, 16);
        prp.inverse_range(90..101);
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

        for y in [0u64, 255, 511] {
            let balls: Range<u64> = iprf.trace_ball_inverse(y, domain, range);
            let expected: Vec<u64> = balls.map(|z| iprf.prp.inverse(z)).collect();
            assert!(expected.len() > 100, "y={y}");

            assert_eq!(iprf.inverse(y), expected, "Iprf::inverse, y={y}");
            assert_eq!(table.inverse(y), expected, "IprfTable::inverse, y={y}");
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

    /// `reduce_below(r, n)` must be `floor(r * n / 2^128)`.
    #[test]
    fn test_reduce_below_is_the_scaled_floor() {
        // For a power of two the result is the top bits of `r`.
        let values: [u128; 5] = [
            0,
            1,
            u128::MAX,
            0x0123_4567_89AB_CDEF_FEDC_BA98_7654_3210,
            (1u128 << 127) + 5,
        ];
        for bits in [1u32, 7, 32, 63] {
            for r in values {
                assert_eq!(
                    u128::from(reduce_below(r, 1u64 << bits)),
                    r >> (128 - bits),
                    "bits {bits}, r {r:#x}"
                );
            }
        }

        // For r = q * 2^64 the result is floor(q * n / 2^64), one 128-bit product.
        for n in [1u64, 3, 10, 1000, 100_003, u64::MAX] {
            for q in [0u64, 1, 12_345, u64::MAX / 3, u64::MAX] {
                let expected: u64 = ((u128::from(q) * u128::from(n)) >> 64) as u64;
                assert_eq!(
                    reduce_below(u128::from(q) << 64, n),
                    expected,
                    "n {n}, q {q}"
                );
            }
            assert_eq!(reduce_below(0, n), 0);
            assert_eq!(reduce_below(u128::MAX, n), n - 1);
        }

        // The low half of `r` counts. For n = 3 the results change above
        // (2^128 - 1) / 3 and above two times that value.
        let third: u128 = u128::MAX / 3;
        assert_eq!(reduce_below(third, 3), 0);
        assert_eq!(reduce_below(third + 1, 3), 1);
        assert_eq!(reduce_below(2 * third, 3), 1);
        assert_eq!(reduce_below(2 * third + 1, 3), 2);
    }

    #[test]
    fn test_sr_round_key_is_below_the_stage_size() {
        let cipher = Aes128::new(&GenericArray::from([0x21u8; 16]));
        for n in [2u64, 3, 5, 1000, 65_536, 100_003, u64::MAX] {
            for round in 0..200 {
                let key: u64 = sr_round_key(&cipher, 3, round, n);
                assert!(key < n, "n {n}, round {round}");
                if n == 2 {
                    assert_eq!(key, 1, "the stage of size 2 has the fixed key 1");
                }
            }
        }

        // 600 keys of a stage of size 3: mean 200 for each value, sd 11.5.
        let mut counts: [u32; 3] = [0; 3];
        for round in 0..600 {
            counts[sr_round_key(&cipher, 1, round, 3) as usize] += 1;
        }
        assert!(
            counts.iter().all(|&count| (130..=270).contains(&count)),
            "{counts:?}"
        );
        // The key depends on the level and on the stage size.
        assert_ne!(
            (0..8)
                .map(|round| sr_round_key(&cipher, 1, round, 1000))
                .collect::<Vec<u64>>(),
            (0..8)
                .map(|round| sr_round_key(&cipher, 2, round, 1000))
                .collect::<Vec<u64>>()
        );
    }

    /// The stage of size 2 must swap its values with probability 1/2. With a
    /// random round key it was 1/4: 4000 keys give a mean of 2000 with a
    /// standard deviation of 32, where the old code gave about 1000.
    #[test]
    fn test_sr_stage_of_size_2_is_uniform() {
        let mut swaps: u32 = 0;
        for index in 0..4000u32 {
            let mut key: PrfKey128 = [0x6Bu8; 16];
            key[0..4].copy_from_slice(&index.to_be_bytes());
            let prp: SwapOrNotSr = SwapOrNotSr::new(key, 2);
            let image: u64 = prp.forward(0);
            assert_eq!(prp.forward(1), 1 - image);
            assert_eq!(prp.inverse(image), 0);
            swaps += image as u32;
        }
        assert!((1800..=2200).contains(&swaps), "swaps: {swaps}");
    }

    #[test]
    fn test_sr_round_keys_batch_matches_scalar_sr_round_key() {
        let cipher: Aes128 = Aes128::new(&GenericArray::from([0xA5u8; 16]));
        for n in [2u64, 3, 100, u64::MAX] {
            for level in [0usize, 1, 5] {
                for first_round in [0usize, 1, 15, 16, 17] {
                    for len in [1usize, 7, 15, SR_KEY_BATCH] {
                        let mut batch = vec![0u64; len];
                        sr_round_keys_batch(&cipher, level, first_round, n, &mut batch);
                        let scalar: Vec<u64> = (first_round..first_round + len)
                            .map(|r| sr_round_key(&cipher, level, r, n))
                            .collect();
                        assert_eq!(
                            batch, scalar,
                            "n={n} level={level} first_round={first_round} len={len}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_sr_forward_many_matches_scalar_across_domains_and_batch_sizes() {
        for domain in [1u64, 2, 3, 4, 5, 7, 16, 17, 100] {
            let max_count: usize = 19;
            let prps: Vec<SwapOrNotSr> = (0..max_count)
                .map(|i| {
                    let mut key: PrfKey128 = [0x3Cu8; 16];
                    key[0] = i as u8;
                    key[15] = (domain & 0xFF) as u8;
                    SwapOrNotSr::with_security(key, domain, 16)
                })
                .collect();

            let xs: Vec<u64> = vec![0, domain / 2, domain - 1];
            for count in [0usize, 1, 2, 3, 4, 5, 7, 8, 9, 12, 13, 16, 19] {
                let refs: Vec<&SwapOrNotSr> = prps[..count].iter().collect();
                for &x in &xs {
                    let batch: Vec<u64> = SwapOrNotSr::forward_many(&refs, x);
                    let scalar: Vec<u64> = refs.iter().map(|p| p.forward(x)).collect();
                    assert_eq!(batch, scalar, "domain={domain} count={count} x={x}");
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "all PRPs must share the same domain")]
    fn test_sr_forward_many_rejects_mismatched_domain() {
        let p0: SwapOrNotSr = SwapOrNotSr::with_security([1u8; 16], 10, 16);
        let p1: SwapOrNotSr = SwapOrNotSr::with_security([2u8; 16], 12, 16);
        let _ = SwapOrNotSr::forward_many(&[&p0, &p1], 3);
    }

    #[test]
    fn test_iprf_forward_many_matches_scalar() {
        let domain: u64 = 512;
        let range: u64 = 32;
        for cached_levels in [0u32, 3] {
            let iprfs: Vec<Iprf> = (0..13u8)
                .map(|k| Iprf::with_cached_levels([k; 16], domain, range, cached_levels))
                .collect();
            let refs: Vec<&Iprf> = iprfs.iter().collect();
            for x in [0u64, 1, 127, 256, 511] {
                let batch: Vec<u64> = Iprf::forward_many(&refs, x);
                let scalar: Vec<u64> = iprfs.iter().map(|f| f.forward(x)).collect();
                assert_eq!(batch, scalar, "cached_levels={cached_levels} x={x}");
            }
        }
    }
}
