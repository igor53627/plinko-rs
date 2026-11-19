//! Plinko PIR hint generation coordinator.
//!
//! Command-line side of the offline phase of the Plinko PIR scheme: options,
//! database geometry, seed handling and the hint file. HintInit is
//! `plinko::hint_init`: the cutoff of each block subset and the XOR parities
//! over the database.

pub mod driver;
pub mod types;

#[cfg(feature = "tee")]
pub use driver::validate_hint_params;
pub use driver::{
    compute_geometry, parse_or_generate_seed, print_results, validate_args, worker_threads,
    write_hint_file, HintParams,
};
pub use types::Args;
