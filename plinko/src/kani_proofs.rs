//! Kani and proptest harnesses for iPRF security-critical functions.
//!
//! - Kani: the coin mask of the PMNS sampler (pure integer arithmetic).
//! - proptest: sampler, SwapOrNot and Iprf properties (AES makes Kani infeasible).
//!
//! Run with: `cargo kani --tests`

#[cfg(kani)]
mod kani_harnesses {
    /// Proof: the mask keeps exactly the `kept` most significant bits.
    #[kani::proof]
    fn proof_top_bits_mask_keeps_leading_bits() {
        let kept: u64 = kani::any();
        kani::assume(kept <= 64);

        let mask: u64 = crate::binomial::top_bits_mask(kept);
        kani::assert(
            u64::from(mask.leading_ones()) == kept,
            "mask must have kept leading ones",
        );
        kani::assert(
            u64::from(mask.count_ones()) == kept,
            "mask must have no other bits",
        );
    }
}

#[cfg(test)]
mod proptest_harnesses {
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn test_coin_binomial_bounded_and_ct_equal(
            key in proptest::array::uniform16(0u8..),
            low in 0u32..1_000,
            width in 0u32..1_000,
            count in 0u64..5_000,
            slack in 0u64..1_000,
        ) {
            use aes::cipher::KeyInit;
            use crate::binomial::{coin_binomial, coin_binomial_ct, PmnsNode};

            let cipher = aes::Aes128::new(&key.into());
            let node = PmnsNode { low, high: low + width };
            let result = coin_binomial(&cipher, node, count);
            prop_assert!(result <= count, "result {} > count {}", result, count);
            prop_assert_eq!(result, coin_binomial_ct(&cipher, node, count, count + slack));
        }

        #[test]
        fn test_swap_or_not_roundtrip(
            key in proptest::array::uniform16(0u8..),
            domain in 1u64..1000,
            x_ratio in 0.0f64..1.0,
        ) {
            use crate::iprf::SwapOrNot;
            let x = (x_ratio * (domain as f64)) as u64 % domain;
            let prp = SwapOrNot::new(key, domain);
            let y = prp.forward(x);
            let x_recovered = prp.inverse(y);
            prop_assert_eq!(x, x_recovered, "PRP roundtrip failed");
        }

        #[test]
        fn test_swap_or_not_output_in_domain(
            key in proptest::array::uniform16(0u8..),
            domain in 1u64..1000,
            x_ratio in 0.0f64..1.0,
        ) {
            use crate::iprf::SwapOrNot;
            let x = (x_ratio * (domain as f64)) as u64 % domain;
            let prp = SwapOrNot::new(key, domain);
            let y = prp.forward(x);
            prop_assert!(y < domain, "PRP output {} >= domain {}", y, domain);
        }

        #[test]
        fn test_iprf_forward_in_range(
            key in proptest::array::uniform16(0u8..),
            n in 10u64..500,
            m_log in 1u32..8,
            x_ratio in 0.0f64..1.0,
        ) {
            use crate::iprf::Iprf;
            let m = 1u64 << m_log;
            let x = (x_ratio * (n as f64)) as u64 % n;
            let iprf = Iprf::new(key, n, m);
            let y = iprf.forward(x);
            prop_assert!(y < m, "iPRF output {} >= range {}", y, m);
        }

        #[test]
        fn test_iprf_inverse_contains_original(
            key in proptest::array::uniform16(0u8..),
            n in 10u64..500,
            m_log in 1u32..8,
            x_ratio in 0.0f64..1.0,
        ) {
            use crate::iprf::Iprf;
            let m = 1u64 << m_log;
            let x = (x_ratio * (n as f64)) as u64 % n;
            let iprf = Iprf::new(key, n, m);
            let y = iprf.forward(x);
            let preimages = iprf.inverse(y);
            prop_assert!(
                preimages.contains(&x),
                "iPRF inverse({}) doesn't contain original x={}",
                y, x
            );
        }

        #[test]
        fn test_iprf_inverse_all_valid(
            key in proptest::array::uniform16(0u8..),
            n in 10u64..200,
            m_log in 1u32..7,
            y_ratio in 0.0f64..1.0,
        ) {
            use crate::iprf::Iprf;
            let m = 1u64 << m_log;
            let y = (y_ratio * (m as f64)) as u64 % m;
            let iprf = Iprf::new(key, n, m);
            let preimages = iprf.inverse(y);
            for x in preimages {
                prop_assert!(x < n, "iPRF inverse element {} >= domain {}", x, n);
            }
        }
    }
}
