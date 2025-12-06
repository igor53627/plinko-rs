//! State Syncer library for Plinko PIR
//!
//! This module exposes the iPRF implementation for testing and verification.

pub mod iprf;

#[cfg(any(kani, test))]
#[path = "kani_proofs.rs"]
mod kani_proofs;
