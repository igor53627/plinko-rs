//! Synthetic dataset generator for testing GPU hint generation.
//!
//! Generates a configurable-size database with random entries in the 48-byte schema,
//! suitable for benchmarking on Modal H100/H200 GPUs.
//!
//! Default: 0.1% of mainnet scale (~330K accounts, ~1.4M storage slots)

use clap::Parser;
use eyre::Result;
use indicatif::{ProgressBar, ProgressStyle};
use plinko::schema48::{AccountEntry48, CodeId, StorageEntry48, ENTRY_SIZE};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::{
    fs::File,
    io::{BufWriter, Write},
    path::PathBuf,
};

/// Mainnet scale (approximate as of 2024)
const MAINNET_ACCOUNTS: u64 = 330_000_000;
const MAINNET_STORAGE: u64 = 1_400_000_000;

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Generate synthetic PIR database for benchmarking"
)]
struct Args {
    /// Output directory for artifacts
    #[arg(long, default_value = "data/synthetic")]
    output_dir: PathBuf,

    /// Scale factor as percentage of mainnet (default: 0.1%)
    #[arg(long, default_value_t = 0.1)]
    scale_percent: f64,

    /// Number of accounts (overrides scale_percent)
    #[arg(long)]
    accounts: Option<u64>,

    /// Number of storage slots (overrides scale_percent)
    #[arg(long)]
    storage: Option<u64>,

    /// Random seed for reproducibility
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Number of unique bytecode hashes to simulate
    #[arg(long, default_value_t = 1000)]
    unique_bytecodes: u32,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Calculate counts based on scale or explicit values
    let num_accounts = args
        .accounts
        .unwrap_or_else(|| ((MAINNET_ACCOUNTS as f64) * (args.scale_percent / 100.0)) as u64);
    let num_storage = args
        .storage
        .unwrap_or_else(|| ((MAINNET_STORAGE as f64) * (args.scale_percent / 100.0)) as u64);
    let total_entries = num_accounts + num_storage;
    let total_bytes = total_entries * ENTRY_SIZE as u64;
    let total_mb = total_bytes as f64 / (1024.0 * 1024.0);

    println!("Synthetic Dataset Generator");
    println!("===========================");
    println!("Scale: {:.3}% of mainnet", args.scale_percent);
    println!("Accounts: {}", num_accounts);
    println!("Storage slots: {}", num_storage);
    println!("Total entries: {}", total_entries);
    println!("Total size: {:.2} MB ({} bytes)", total_mb, total_bytes);
    println!("Seed: {}", args.seed);
    println!();

    // Create output directory
    std::fs::create_dir_all(&args.output_dir)?;
    let db_path = args.output_dir.join("database.bin");
    println!("Writing to: {:?}", db_path);

    // Initialize RNG
    let mut rng = ChaCha8Rng::seed_from_u64(args.seed);

    // Create file and writer
    let file = File::create(&db_path)?;
    let mut writer = BufWriter::with_capacity(1024 * 1024, file); // 1MB buffer

    // Progress bar
    let pb = ProgressBar::new(total_entries);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
            )
            .unwrap()
            .progress_chars("#>-"),
    );

    // Generate accounts
    println!("Generating {} accounts...", num_accounts);
    for i in 0..num_accounts {
        // Random address
        let mut address = [0u8; 20];
        rng.fill(&mut address);

        // Random balance (varies from 0 to ~10^20 wei, fitting in 128 bits)
        let mut balance_256 = [0u8; 32];
        // Fill lower 16 bytes with random data
        rng.fill(&mut balance_256[..16]);

        // Random nonce (0 to 10000 for most accounts)
        let nonce: u64 = if rng.gen_bool(0.9) {
            rng.gen_range(0..10000)
        } else {
            rng.gen_range(10000..1_000_000)
        };

        // Code ID: ~90% EOA, ~10% contracts
        let code_id = if rng.gen_bool(0.1) {
            CodeId::new(rng.gen_range(1..=args.unique_bytecodes))
        } else {
            CodeId::new(0) // EOA
        };

        let entry = AccountEntry48::new(&balance_256, nonce, code_id, &address);
        writer.write_all(&entry.to_bytes())?;

        if i % 10000 == 0 {
            pb.set_position(i);
        }
    }

    // Generate storage slots
    println!("Generating {} storage slots...", num_storage);
    for i in 0..num_storage {
        // Random address (for TAG)
        let mut address = [0u8; 20];
        rng.fill(&mut address);

        // Random slot key
        let mut slot_key = [0u8; 32];
        rng.fill(&mut slot_key);

        // Random storage value
        let mut value = [0u8; 32];
        // Most storage values are small, some are large
        if rng.gen_bool(0.7) {
            // Small value (fits in 8 bytes)
            let small_val: u64 = rng.gen();
            value[..8].copy_from_slice(&small_val.to_le_bytes());
        } else {
            // Full 256-bit value
            rng.fill(&mut value);
        }

        let entry = StorageEntry48::new(&value, &address, &slot_key);
        writer.write_all(&entry.to_bytes())?;

        if i % 10000 == 0 {
            pb.set_position(num_accounts + i);
        }
    }

    pb.finish_with_message("Done!");
    writer.flush()?;

    // Write metadata
    let meta_path = args.output_dir.join("metadata.json");
    let json = format!(
        r#"{{
  "schema_version": 2,
  "entry_size_bytes": {},
  "synthetic": true,
  "seed": {},
  "scale_percent": {},
  "accounts": {},
  "storage_slots": {},
  "total_entries": {},
  "unique_bytecodes": {},
  "size_bytes": {}
}}"#,
        ENTRY_SIZE,
        args.seed,
        args.scale_percent,
        num_accounts,
        num_storage,
        total_entries,
        args.unique_bytecodes,
        total_bytes
    );
    std::fs::write(&meta_path, json)?;

    println!();
    println!("Generated:");
    println!("  Database: {:?} ({:.2} MB)", db_path, total_mb);
    println!("  Metadata: {:?}", meta_path);

    // Print Plinko params preview
    let target_chunk = ((4.0 * total_entries as f64).sqrt()) as u64;
    let mut chunk_size = 1u64;
    while chunk_size < target_chunk {
        chunk_size *= 2;
    }
    let set_size = total_entries.div_ceil(chunk_size);
    let set_size = set_size.div_ceil(4) * 4;

    println!();
    println!("Plinko parameters:");
    println!("  chunk_size (w): {}", chunk_size);
    println!("  set_size (c): {}", set_size);
    println!(
        "  capacity: {} (overhead: {:.2}%)",
        chunk_size * set_size,
        100.0 * (chunk_size * set_size - total_entries) as f64 / total_entries as f64
    );

    Ok(())
}
