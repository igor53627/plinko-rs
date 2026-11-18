//! Plinko PIR cryptographic primitives
//!
//! This crate provides core Plinko PIR primitives:
//! - `iprf`: Invertible PRF implementation (paper §4.2)
//! - `db`: Database loading and Plinko parameter derivation
//! - `binomial`: True derandomized binomial sampling for PMNS
//! - `subsets`: Block subsets of hints, from a PRF and one cutoff per hint
//! - `hints`: Hint set of a client and its on-disk format
//! - `delta`: Changes between two database versions and their on-disk format
//! - `hint_init`: HintInit (cutoffs and parities of a hint set) on worker threads
//!
//! Two opt-in features add experimental code that is not part of the trust
//! model (`docs/trust_model.md`):
//! - `tee`: `constant_time` and the constant-time PRP and iPRF variants
//! - `gpu`: `gpu`, a hint throughput benchmark (CUDA with the `cuda` feature)

pub mod binomial;
mod checked_io;
#[cfg(feature = "tee")]
pub mod constant_time;
pub mod db;
pub mod delta;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod hint_init;
pub mod hints;
pub mod iprf;
pub mod protocol;
pub mod schema40;
pub mod subsets;

#[cfg(any(kani, test))]
#[path = "kani_proofs.rs"]
mod kani_proofs;
