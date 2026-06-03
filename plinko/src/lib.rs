//! Plinko PIR cryptographic primitives
//!
//! This crate provides core Plinko PIR primitives:
//! - `iprf`: Invertible PRF implementation (paper §4.2)
//! - `db`: Database loading and Plinko parameter derivation
//! - `constant_time`: Data-oblivious operations for TEE execution
//! - `binomial`: True derandomized binomial sampling for PMNS

pub mod binomial;
pub mod constant_time;
pub mod db;
pub mod gpu;
pub mod iprf;
pub mod protocol;
pub mod schema40;

#[cfg(any(kani, test))]
#[path = "kani_proofs.rs"]
mod kani_proofs;
