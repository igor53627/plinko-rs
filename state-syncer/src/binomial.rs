//! True derandomized binomial sampling for PMNS.
//!
//! This module implements a proper Binomial(n, p) sampler using inverse-CDF
//! transform. When `prf_output` is uniform, the output is distributed according
//! to the binomial distribution, as required by the Plinko paper specification.
//!
//! Algorithm:
//! - Small count (<=1024): Exact inverse-CDF with PMF recurrence, O(n)
//! - Large count (>1024): Binary search over CDF using regularized incomplete beta, O(log n)
//!
//! For TEE environments, use `binomial_sample_tee` which is constant-time and
//! matches the simplified arithmetic formula in BinomialSpec.v exactly.

/// Threshold for switching between exact and approximate algorithms.
/// Below this, we use exact PMF summation. Above, we use binary search with beta function.
const EXACT_THRESHOLD: u64 = 1024;

/// Maximum value of u64 as f64 for normalization
const U64_MAX_F64: f64 = u64::MAX as f64;

/// True derandomized binomial sampler.
///
/// Given `count` trials with success probability `p = num/denom`, returns a value
/// distributed as Binomial(count, p) when `prf_output` is uniform over u64.
///
/// This implements the paper's specification:
/// "Binomial(n, p; r) will always return the same value, but over the random
/// choice of r with sufficient entropy, the output will be distributed according
/// to a binomial."
///
/// # Arguments
/// * `count` - Number of trials (n)
/// * `num` - Numerator of probability (must be <= denom)
/// * `denom` - Denominator of probability (must be > 0)
/// * `prf_output` - PRF output used as randomness source
///
/// # Returns
/// A value in [0, count] distributed as Binomial(count, num/denom)
pub fn binomial_sample(count: u64, num: u64, denom: u64, prf_output: u64) -> u64 {
    if denom == 0 {
        return 0;
    }
    if count == 0 {
        return 0;
    }
    if num == 0 {
        return 0;
    }
    if num >= denom {
        return count;
    }

    let mut p = num as f64 / denom as f64;
    let u = (prf_output as f64 + 0.5) / (U64_MAX_F64 + 1.0);

    // Use symmetry to keep p <= 0.5, avoiding underflow in q^n for exact branch
    // Binomial(n, p) = n - Binomial(n, 1-p)
    let use_complement = p > 0.5;
    if use_complement {
        p = 1.0 - p;
    }

    let k = if count <= EXACT_THRESHOLD {
        binomial_inverse_exact(count, p, u)
    } else {
        binomial_inverse_binary_search(count, p, u)
    };

    if use_complement {
        count - k
    } else {
        k
    }
}

/// Exact inverse-CDF for small count using PMF recurrence.
///
/// Complexity: O(count)
fn binomial_inverse_exact(n: u64, p: f64, u: f64) -> u64 {
    let q = 1.0 - p;

    let mut pmf = q.powi(n as i32);
    let mut cdf = pmf;

    if u <= cdf {
        return 0;
    }

    let p_over_q = p / q;

    for k in 1..=n {
        pmf *= ((n - k + 1) as f64 / k as f64) * p_over_q;
        cdf += pmf;
        if u <= cdf {
            return k;
        }
    }

    n
}

/// Inverse-CDF using binary search with regularized incomplete beta function.
///
/// Uses the identity: P(X <= k) = I_{1-p}(n-k, k+1)
/// where I_x(a,b) is the regularized incomplete beta function.
///
/// Complexity: O(log count)
fn binomial_inverse_binary_search(n: u64, p: f64, u: f64) -> u64 {
    let mut lo: u64 = 0;
    let mut hi: u64 = n;

    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let cdf = binomial_cdf(n, p, mid);

        if u <= cdf {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }

    lo
}

/// Compute binomial CDF: P(X <= k) for X ~ Binomial(n, p)
///
/// Uses the identity: P(X <= k) = I_{1-p}(n-k, k+1)
/// where I_x(a,b) is the regularized incomplete beta function.
fn binomial_cdf(n: u64, p: f64, k: u64) -> f64 {
    if k >= n {
        return 1.0;
    }

    let a = (n - k) as f64;
    let b = (k + 1) as f64;
    let x = 1.0 - p;

    puruspe::betai(a, b, x)
}

/// Maximum count for constant-time exact binomial sampling.
/// For counts above this, the O(n) iteration is too expensive.
/// In PMNS/Plinko contexts, `count` is always <= CT_BINOMIAL_MAX_COUNT,
/// which is chosen to exceed the maximum possible ball_count at any recursion
/// level (e.g., ~40k for Ethereum addresses at first recursion level).
pub const CT_BINOMIAL_MAX_COUNT: u64 = 65536;

/// TEE-safe constant-time exact binomial sampler.
///
/// Samples from Binomial(count, num/denom) distribution using inverse-CDF
/// with fixed iteration count to prevent timing side-channels.
///
/// Security properties:
/// - Always iterates exactly `CT_BINOMIAL_MAX_COUNT + 1` times (no early exit)
/// - Uses constant-time float comparison and selection
/// - No branches depend on `count` or `prf_output` (both are secret)
/// - Loop bounds depend only on public constants
///
/// Complexity: O(CT_BINOMIAL_MAX_COUNT) - fixed cost regardless of count
///
/// # TEE Usage
/// In TEE/constant-time contexts, callers MUST ensure `count <= CT_BINOMIAL_MAX_COUNT`.
/// This is a protocol-level invariant; exceeding it is a programmer error.
///
/// # Panics
/// Panics in debug mode if count > CT_BINOMIAL_MAX_COUNT.
#[inline]
pub fn binomial_sample_tee(count: u64, num: u64, denom: u64, prf_output: u64) -> u64 {
    use crate::constant_time::{ct_eq_u64, ct_select_u64};

    // Edge cases with public parameters only - these branches are on public values
    // (num, denom are fixed protocol parameters like 1/2)
    if denom == 0 {
        return 0;
    }
    if num == 0 {
        return 0;
    }
    if num >= denom {
        return count;
    }

    debug_assert!(
        count <= CT_BINOMIAL_MAX_COUNT,
        "binomial_sample_tee: count {} exceeds CT_BINOMIAL_MAX_COUNT {}",
        count,
        CT_BINOMIAL_MAX_COUNT
    );

    let mut p = num as f64 / denom as f64;
    let u = (prf_output as f64 + 0.5) / (U64_MAX_F64 + 1.0);

    // Use symmetry to keep p <= 0.5, avoiding underflow in q^n
    // This branch is on public parameter p
    let use_complement = p > 0.5;
    if use_complement {
        p = 1.0 - p;
    }

    // Handle count == 0 via CT masking instead of branching (count may be secret)
    let count_is_zero = ct_eq_u64(count, 0);
    let k = binomial_inverse_ct(count, p, u);

    // Apply complement if needed, but return 0 if count was 0
    let result = if use_complement { count - k } else { k };
    ct_select_u64(count_is_zero, 0, result)
}

/// Constant-time inverse-CDF binomial sampler with fixed iterations.
///
/// Always iterates exactly CT_BINOMIAL_MAX_COUNT+1 times regardless of n or
/// when the answer is found. Uses constant-time selection to track the result.
/// Loop bounds depend only on CT_BINOMIAL_MAX_COUNT (public constant), not on n.
///
/// Uses log-space computation starting from mode to avoid underflow for large n.
/// For n=40k and p=0.5, starting from k=0 would cause q^n to underflow to 0.
///
/// Strategy: Compute PMF at mode (stable), then expand outward using recurrence.
/// Track partial sums for CDF computation.
fn binomial_inverse_ct(n: u64, p: f64, u: f64) -> u64 {
    use crate::constant_time::{ct_f64_le, ct_f64_lt, ct_le_u64, ct_lt_u64, ct_select_f64, ct_select_u64};

    debug_assert!(
        n <= CT_BINOMIAL_MAX_COUNT,
        "binomial_inverse_ct: n={} exceeds CT_BINOMIAL_MAX_COUNT={}",
        n,
        CT_BINOMIAL_MAX_COUNT
    );

    let q = 1.0 - p;

    // Compute mode: floor((n+1)*p), clamped to [0, n]
    let mode = ((n as f64 + 1.0) * p).floor() as u64;
    let mode = mode.min(n);

    // Compute log(PMF) at mode using log-gamma approximation
    let log_pmf_mode = {
        let log_p = p.ln();
        let log_q = q.ln();
        let log_binom = lgamma_approx((n + 1) as f64)
            - lgamma_approx((mode + 1) as f64)
            - lgamma_approx((n - mode + 1) as f64);
        log_binom + (mode as f64) * log_p + ((n - mode) as f64) * log_q
    };
    let pmf_mode = log_pmf_mode.exp();

    let p_over_q = p / q;
    let q_over_p = q / p;

    // Strategy: Sweep left from mode to build "left" PMF values,
    // then sweep right from mode. Track cumulative sum.
    // 
    // We'll compute:
    // - cdf_below_mode = sum of PMF(0..mode-1)
    // - For k >= mode: CDF(k) = cdf_below_mode + sum of PMF(mode..k)
    // - For k < mode: we need to know partial sum up to k

    // Phase 1: Sweep left from mode-1 to 0, compute cdf_below_mode
    // Also compute "running sum from left" which gives us partial CDF at each k < mode
    // We store this in a lookup structure using the sweep
    
    // For left sweep: PMF(k) = PMF(k+1) * (k+1)/(n-k) * q/p
    let mut cdf_below_mode = 0.0f64;
    let mut pmf_left = pmf_mode;
    
    // We also need to find result if u falls in the left portion
    // For that we need CDF(k) = sum(PMF(0..k)) for k < mode
    // CDF(k) = cdf_below_mode - sum(PMF(k+1..mode-1))
    
    // During left sweep (from mode-1 down to 0), we compute:
    // running_sum_right[offset] = sum(PMF(mode-1-offset+1 .. mode-1)) = sum(PMF(mode-offset .. mode-1))
    // After sweep, cdf_below_mode = sum(PMF(0..mode-1))
    
    // running_sum from right (cumulative from mode-1 going left)
    let mut running_sum_right = 0.0f64;
    
    // We need to store partial sums to check later. Since we can't use arrays,
    // we'll do the check during this sweep itself.
    
    // During left sweep, at each step we have:
    // - pmf_left = PMF(mode-1-offset) after applying recurrence
    // - running_sum_right = sum(PMF(mode-1-offset+1 .. mode-1))
    // - CDF(mode-1-offset) = cdf_full - running_sum_right - pmf(mode) - pmf(mode+1) - ...
    // This is complex. Let's use a different strategy.
    
    // New strategy: Two-phase with result selection
    // Phase A: Compute all left PMFs and their partial sums (left-to-right)
    // Phase B: Compute all right PMFs, building full CDF, find result
    
    // For Phase A, we first need PMF(0), but that underflows.
    // So instead, we compute partial sums from mode going left.
    
    // Let's use "expanding search" from mode:
    // - Check if u is left of mode: u < CDF(mode-1)
    // - If yes, binary search or linear search left
    // - If no, linear search right
    
    // For CT, we can't branch on this. Instead:
    // Compute CDF(mode-1) = cdf_below_mode
    // Compute CDF(mode) = cdf_below_mode + pmf_mode
    // Compute CDF(mode+1), CDF(mode+2), etc.
    // Also compute CDF(mode-2), CDF(mode-3), etc. by subtracting
    
    // Build cdf_below_mode and pmf array for left portion
    // Actually, we need to iterate and track "cumulative from mode going left"
    
    // Let's define: left_cumsum[offset] = sum(PMF(mode-1), PMF(mode-2), ..., PMF(mode-1-offset+1))
    //             = sum(PMF(mode-offset .. mode-1))
    // Then CDF(mode-1-offset) = cdf_below_mode - left_cumsum[offset]
    // And cdf_below_mode = final left_cumsum after mode iterations
    
    // First pass: sweep left, compute cdf_below_mode
    for offset in 0..CT_BINOMIAL_MAX_COUNT {
        let valid = ct_lt_u64(offset, mode);
        let k = mode.saturating_sub(offset + 1);
        
        let k_plus_1 = k + 1;
        let n_minus_k = n.saturating_sub(k).max(1);
        let ratio = (k_plus_1 as f64 / n_minus_k as f64) * q_over_p;
        let pmf_k = pmf_left * ratio;
        pmf_left = ct_select_f64(valid, pmf_k, pmf_left);
        
        cdf_below_mode += ct_select_f64(valid, pmf_left, 0.0);
    }
    
    // Now cdf_below_mode = sum(PMF(0..mode-1))
    
    // Second pass: check left portion first (k < mode), from k=0 upward
    // We need CDF(k) = sum(PMF(0..k)) for k < mode
    // CDF(0) = PMF(0)
    // CDF(1) = PMF(0) + PMF(1)
    // etc.
    // We already computed these PMFs going from mode left, but in reverse order.
    // 
    // To compute CDF(k) for k < mode incrementally:
    // We need to iterate k = 0, 1, 2, ..., mode-1
    // But we computed PMF going mode-1, mode-2, ..., 0
    // 
    // Strategy: Use the PMF at 0 we computed (pmf_left after the sweep),
    // then iterate forward using the forward recurrence.
    
    let mut result = 0u64;
    let mut found = 0u64;
    let mut cdf = 0.0f64;
    
    // pmf_left now holds PMF(0) (after the left sweep ended at k=0)
    // Actually, the left sweep computed PMF going from mode toward 0,
    // so pmf_left is PMF(0) at the end (or pmf_mode if mode=0)
    let pmf_0 = pmf_left;
    let mut pmf = pmf_0;
    
    // Iterate k = 0, 1, ..., mode-1
    for k in 0..CT_BINOMIAL_MAX_COUNT {
        let valid = ct_lt_u64(k, mode);
        let in_support = ct_le_u64(k, n);
        let both_valid = valid & in_support;
        
        // Add PMF(k) to CDF
        let pmf_k = ct_select_f64(both_valid, pmf, 0.0);
        cdf += pmf_k;
        
        // Check if u <= cdf
        let cond = ct_f64_le(u, cdf) & both_valid;
        let is_first = (1 ^ found) & cond;
        result = ct_select_u64(is_first, k, result);
        found |= cond;
        
        // Compute PMF(k+1) = PMF(k) * (n-k)/(k+1) * p/q
        let k_plus_1 = k + 1;
        let n_minus_k = n.saturating_sub(k);
        let ratio = (n_minus_k as f64 / k_plus_1.max(1) as f64) * p_over_q;
        pmf *= ratio;
    }
    
    // Third pass: sweep right from mode, accumulate CDF, find result
    // CDF continues from cdf_below_mode
    cdf = cdf_below_mode;
    let mut pmf_right = pmf_mode;
    
    for offset in 0..=CT_BINOMIAL_MAX_COUNT {
        let k = mode.saturating_add(offset);
        let in_support = ct_le_u64(k, n);
        
        // Add current PMF to CDF
        let pmf_k = ct_select_f64(in_support, pmf_right, 0.0);
        cdf += pmf_k;
        
        // Check if u <= cdf (only if not already found in left portion)
        let cond = ct_f64_le(u, cdf) & in_support;
        let is_first = (1 ^ found) & cond;
        result = ct_select_u64(is_first, k, result);
        found |= cond;
        
        // Compute next PMF: PMF(k+1) = PMF(k) * (n-k)/(k+1) * p/q
        let k_plus_1 = k + 1;
        let n_minus_k = n.saturating_sub(k);
        let ratio = (n_minus_k as f64 / k_plus_1.max(1) as f64) * p_over_q;
        pmf_right *= ratio;
    }
    
    ct_select_u64(found, result, n)
}

/// Lanczos approximation for log-gamma function.
/// Accurate for x >= 0.5.
fn lgamma_approx(x: f64) -> f64 {
    use std::f64::consts::PI;

    if x < 0.5 {
        // Reflection formula: Gamma(x) * Gamma(1-x) = pi / sin(pi*x)
        // So: lgamma(x) = ln(pi) - ln(sin(pi*x)) - lgamma(1-x)
        PI.ln() - (PI * x).sin().abs().ln() - lgamma_approx(1.0 - x)
    } else {
        let g = 7.0;
        let coeffs = [
            0.99999999999980993,
            676.5203681218851,
            -1259.1392167224028,
            771.32342877765313,
            -176.61502916214059,
            12.507343278686905,
            -0.13857109526572012,
            9.9843695780195716e-6,
            1.5056327351493116e-7,
        ];

        let x = x - 1.0;
        let mut sum = coeffs[0];
        for (i, &c) in coeffs.iter().enumerate().skip(1) {
            sum += c / (x + i as f64);
        }

        let t = x + g + 0.5;
        0.5 * (2.0 * PI).ln() + (x + 0.5) * t.ln() - t + sum.ln()
    }
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edge_cases() {
        assert_eq!(binomial_sample(0, 1, 2, 12345), 0);
        assert_eq!(binomial_sample(10, 0, 2, 12345), 0);
        assert_eq!(binomial_sample(10, 2, 2, 12345), 10);
        assert_eq!(binomial_sample(10, 3, 2, 12345), 10);
        assert_eq!(binomial_sample(10, 1, 0, 12345), 0);
    }

    #[test]
    fn test_range_bounds() {
        for count in [1, 10, 100, 1000, 2000] {
            for prf in [0u64, u64::MAX / 2, u64::MAX] {
                let result = binomial_sample(count, 1, 2, prf);
                assert!(result <= count, "result {} > count {}", result, count);
            }
        }
    }

    #[test]
    fn test_deterministic() {
        let r1 = binomial_sample(100, 1, 2, 42);
        let r2 = binomial_sample(100, 1, 2, 42);
        assert_eq!(r1, r2);
    }

    #[test]
    fn test_distribution_mean() {
        let n = 100u64;
        let num = 1u64;
        let denom = 2u64;
        let expected_mean = n as f64 * (num as f64 / denom as f64);

        let samples: u64 = 10000;
        let mut sum: u64 = 0;
        for i in 0..samples {
            let prf = (i as u64).wrapping_mul(0x9E3779B97F4A7C15);
            sum += binomial_sample(n, num, denom, prf);
        }

        let actual_mean = sum as f64 / samples as f64;
        let tolerance = 2.0;
        assert!(
            (actual_mean - expected_mean).abs() < tolerance,
            "mean {} too far from expected {}",
            actual_mean,
            expected_mean
        );
    }

    #[test]
    fn test_full_support() {
        let n = 20u64;
        let mut seen = vec![false; (n + 1) as usize];

        for i in 0..100000u64 {
            let prf = i.wrapping_mul(0x9E3779B97F4A7C15);
            let k = binomial_sample(n, 1, 2, prf);
            seen[k as usize] = true;
        }

        let count_seen = seen.iter().filter(|&&x| x).count();
        assert!(
            count_seen >= 15,
            "should see most values in [0,20], only saw {}",
            count_seen
        );
    }

    #[test]
    fn test_exact_vs_binary_search_consistency() {
        let n = 1024u64;
        let p = 0.3;

        for i in 0..1000u64 {
            let prf = i.wrapping_mul(0x9E3779B97F4A7C15);
            let u = (prf as f64 + 0.5) / (U64_MAX_F64 + 1.0);

            let exact = binomial_inverse_exact(n, p, u);
            let binary = binomial_inverse_binary_search(n, p, u);

            assert!(
                exact == binary || exact.abs_diff(binary) <= 1,
                "mismatch at prf={}: exact={}, binary={}",
                prf,
                exact,
                binary
            );
        }
    }

    #[test]
    fn test_large_count() {
        let n = 1_000_000u64;
        let result = binomial_sample(n, 1, 2, 0x123456789ABCDEF0);
        assert!(result <= n, "result should be <= n");
    }

    #[test]
    fn test_high_probability() {
        let n = 1024u64;
        let mut sum: u64 = 0;
        let samples = 1000;

        for i in 0..samples {
            let prf = (i as u64).wrapping_mul(0x9E3779B97F4A7C15);
            let k = binomial_sample(n, 9, 10, prf);
            assert!(k <= n, "result {} > count {}", k, n);
            sum += k;
        }

        let actual_mean = sum as f64 / samples as f64;
        let expected_mean = n as f64 * 0.9;
        let tolerance = 20.0;
        assert!(
            (actual_mean - expected_mean).abs() < tolerance,
            "mean {} too far from expected {} for p=0.9",
            actual_mean,
            expected_mean
        );
    }

    #[test]
    fn test_symmetry_correctness() {
        let n = 100u64;
        let samples = 1000u64;
        let mut sum_low: u64 = 0;
        let mut sum_high: u64 = 0;

        for i in 0..samples {
            let prf = i.wrapping_mul(0x9E3779B97F4A7C15);

            let k_low = binomial_sample(n, 1, 4, prf);
            let k_high = binomial_sample(n, 3, 4, prf);

            assert!(k_low <= n);
            assert!(k_high <= n);

            sum_low += k_low;
            sum_high += k_high;
        }

        let mean_low = sum_low as f64 / samples as f64;
        let mean_high = sum_high as f64 / samples as f64;

        assert!(
            mean_high > mean_low,
            "mean for p=0.75 ({}) should exceed mean for p=0.25 ({})",
            mean_high,
            mean_low
        );
    }

    // ============================================================================
    // TEE Binomial Sampler Tests (binomial_sample_tee) - True Binomial CT Version
    // ============================================================================

    #[test]
    fn test_tee_matches_standard_binomial() {
        // Verify binomial_sample_tee produces same results as binomial_sample
        // (both use true binomial distribution, but TEE version is constant-time)
        for count in [1, 10, 50, 100, 500] {
            for (num, denom) in [(1, 2), (1, 4), (3, 4), (1, 10), (9, 10)] {
                for i in 0..50u64 {
                    let prf = i.wrapping_mul(0x9E3779B97F4A7C15);
                    let standard = binomial_sample(count, num, denom, prf);
                    let tee = binomial_sample_tee(count, num, denom, prf);
                    assert_eq!(
                        standard, tee,
                        "Mismatch: count={}, num={}, denom={}, prf={}: standard={}, tee={}",
                        count, num, denom, prf, standard, tee
                    );
                }
            }
        }
    }

    #[test]
    fn test_tee_edge_cases() {
        // denom = 0 always returns 0
        assert_eq!(binomial_sample_tee(100, 50, 0, 12345), 0);
        assert_eq!(binomial_sample_tee(0, 0, 0, 0), 0);

        // count = 0 always returns 0
        assert_eq!(binomial_sample_tee(0, 1, 2, 0), 0);
        assert_eq!(binomial_sample_tee(0, 1, 2, u64::MAX), 0);

        // num = 0 always returns 0
        assert_eq!(binomial_sample_tee(100, 0, 10, 5), 0);

        // num >= denom returns count
        assert_eq!(binomial_sample_tee(10, 20, 10, 0), 10);
        assert_eq!(binomial_sample_tee(10, 10, 10, 12345), 10);
    }

    #[test]
    fn test_tee_range_bounds() {
        // Result should always be in [0, count]
        for count in [1, 10, 100, 500, 1000] {
            for (num, denom) in [(1, 2), (1, 4), (3, 4), (1, 10)] {
                for i in 0..100u64 {
                    let prf = i.wrapping_mul(0x9E3779B97F4A7C15);
                    let result = binomial_sample_tee(count, num, denom, prf);
                    assert!(
                        result <= count,
                        "Range violation: count={}, num={}, denom={}, result={}",
                        count,
                        num,
                        denom,
                        result
                    );
                }
            }
        }
    }

    #[test]
    fn test_tee_deterministic() {
        let r1 = binomial_sample_tee(100, 1, 2, 42);
        let r2 = binomial_sample_tee(100, 1, 2, 42);
        assert_eq!(r1, r2);
    }

    #[test]
    fn test_tee_distribution_mean() {
        // The CT version should have the same distribution as true binomial
        let n = 100u64;
        let num = 1u64;
        let denom = 2u64;
        let expected_mean = n as f64 * (num as f64 / denom as f64);

        let samples: u64 = 10000;
        let mut sum: u64 = 0;
        for i in 0..samples {
            let prf = (i as u64).wrapping_mul(0x9E3779B97F4A7C15);
            sum += binomial_sample_tee(n, num, denom, prf);
        }

        let actual_mean = sum as f64 / samples as f64;
        let tolerance = 2.0;
        assert!(
            (actual_mean - expected_mean).abs() < tolerance,
            "TEE mean {} too far from expected {}",
            actual_mean,
            expected_mean
        );
    }

    #[test]
    fn test_tee_full_support() {
        // Should see values across the full support [0, n]
        let n = 20u64;
        let mut seen = vec![false; (n + 1) as usize];

        for i in 0..100000u64 {
            let prf = i.wrapping_mul(0x9E3779B97F4A7C15);
            let k = binomial_sample_tee(n, 1, 2, prf);
            seen[k as usize] = true;
        }

        let count_seen = seen.iter().filter(|&&x| x).count();
        assert!(
            count_seen >= 15,
            "TEE should see most values in [0,20], only saw {}",
            count_seen
        );
    }

    #[test]
    fn test_tee_symmetry() {
        // Binomial(n, p) should be symmetric with Binomial(n, 1-p)
        let n = 100u64;
        let samples = 1000u64;
        let mut sum_low: u64 = 0;
        let mut sum_high: u64 = 0;

        for i in 0..samples {
            let prf = i.wrapping_mul(0x9E3779B97F4A7C15);
            sum_low += binomial_sample_tee(n, 1, 4, prf);
            sum_high += binomial_sample_tee(n, 3, 4, prf);
        }

        let mean_low = sum_low as f64 / samples as f64;
        let mean_high = sum_high as f64 / samples as f64;

        assert!(
            mean_high > mean_low,
            "TEE mean for p=0.75 ({}) should exceed mean for p=0.25 ({})",
            mean_high,
            mean_low
        );
    }

    #[test]
    fn test_tee_at_threshold() {
        // Test at exactly the threshold (should work without fallback)
        let count = super::CT_BINOMIAL_MAX_COUNT;
        let result = binomial_sample_tee(count, 1, 2, 12345);
        assert!(result <= count);
    }

    #[test]
    fn test_tee_large_count_ethereum() {
        // Test with ~40k count (Ethereum address use case)
        // This verifies numerical stability - old implementation would underflow
        let n = 40000u64;
        let num = 1u64;
        let denom = 2u64;
        let expected_mean = n as f64 * (num as f64 / denom as f64);

        let samples = 100u64;
        let mut sum: u64 = 0;
        for i in 0..samples {
            let prf = (i as u64).wrapping_mul(0x9E3779B97F4A7C15);
            let result = binomial_sample_tee(n, num, denom, prf);
            assert!(result <= n, "result {} > n {}", result, n);
            sum += result;
        }

        let actual_mean = sum as f64 / samples as f64;
        // Allow larger tolerance for fewer samples
        let tolerance = 500.0;
        assert!(
            (actual_mean - expected_mean).abs() < tolerance,
            "TEE large-n mean {} too far from expected {} (tolerance {})",
            actual_mean,
            expected_mean,
            tolerance
        );
    }
}
