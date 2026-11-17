//! Plinko PIR hint generation coordinator.
//!
//! Orchestrates the offline phase of the Plinko PIR scheme: derives keys,
//! computes the cutoff of each block subset, and accumulates XOR parities over
//! the database to produce regular and backup hints that a client stores for
//! online queries.

#[cfg(feature = "tee")]
pub mod ct_path;
pub mod driver;
pub mod fast_path;
pub mod keys;
#[cfg(test)]
mod path_tests;
pub mod types;

#[cfg(feature = "tee")]
pub use driver::validate_hint_params;
pub use driver::{
    compute_geometry, hint_cutoffs, parse_or_generate_seed, print_results, validate_args,
    worker_threads, write_hint_file, HintParams,
};
pub use keys::derive_block_keys;
pub use types::Args;
