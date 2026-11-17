//! Plinko PIR cryptographic primitives
//!
//! This crate provides core Plinko PIR primitives:
//! - `iprf`: Invertible PRF implementation (paper §4.2)
//! - `db`: Database loading and Plinko parameter derivation
//! - `constant_time`: Data-oblivious operations for TEE execution
//! - `binomial`: True derandomized binomial sampling for PMNS
//! - `subsets`: Block subsets of hints, from a PRF and one cutoff per hint
//! - `hints`: Hint set of a client and its on-disk format

pub mod binomial;
pub mod constant_time;
pub mod db;
pub mod gpu;
pub mod hints;
pub mod iprf;
pub mod protocol;
pub mod schema40;
pub mod subsets;

#[cfg(any(kani, test))]
#[path = "kani_proofs.rs"]
mod kani_proofs;
