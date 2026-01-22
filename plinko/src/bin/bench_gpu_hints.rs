//! GPU Hint Generation Benchmark
//!
//! Benchmarks hint generation on GPU (H100/H200) vs CPU.
//! Designed to run on Modal with synthetic datasets.
//!
//! Usage:
//!   cargo run -p plinko --bin bench_gpu_hints --features cuda -- --db data/synthetic/database.bin

use clap::Parser;
use eyre::Result;
use plinko::db::Database48;
use plinko::schema48::ENTRY_SIZE;
use std::path::PathBuf;
use std::time::Instant;

#[cfg(feature = "cuda")]
use plinko::gpu::{GpuHintGenerator, IprfBlockKey, PlinkoParams};

#[derive(Parser, Debug)]
#[command(author, version, about = "Benchmark GPU hint generation")]
struct Args {
    /// Path to database.bin (48-byte schema)
    #[arg(long)]
    db: PathBuf,

    /// Security parameter lambda (default: 128)
    #[arg(long, default_value_t = 128)]
    lambda: u32,

    /// Override chunk size (w). Default: derived from database size.
    /// Mainnet value: 131072 (2^17)
    #[arg(long)]
    chunk_size: Option<u64>,

    /// Number of warmup iterations
    #[arg(long, default_value_t = 3)]
    warmup: u32,

    /// Number of benchmark iterations
    #[arg(long, default_value_t = 10)]
    iterations: u32,

    /// Run CPU baseline for comparison
    #[arg(long, default_value_t = false)]
    cpu_baseline: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("GPU Hint Generation Benchmark");
    println!("==============================");
    println!();

    // Load database
    println!("Loading database: {:?}", args.db);
    let db = Database48::load(&args.db)?;
    let num_entries = db.num_entries;

    // Use override chunk_size if provided, otherwise use derived value
    let (chunk_size, set_size) = if let Some(w) = args.chunk_size {
        // Recalculate set_size for the overridden chunk_size
        let mut c = num_entries.div_ceil(w);
        c = c.div_ceil(4) * 4; // Round to multiple of 4
        println!("  Using override chunk_size (mainnet params)");
        (w, c)
    } else {
        (db.chunk_size, db.set_size)
    };

    println!("  Entries: {}", num_entries);
    println!("  Chunk size (w): {} {}", chunk_size, if args.chunk_size.is_some() { "(override)" } else { "" });
    println!("  Set size (c): {}", set_size);
    println!("  Entry size: {} bytes", ENTRY_SIZE);
    println!("  Database size: {:.2} MB", (num_entries * ENTRY_SIZE as u64) as f64 / 1e6);
    println!();

    // Calculate hint parameters
    let total_hints = 2 * args.lambda * chunk_size as u32;
    let blocks_per_regular = set_size / 2 + 1;
    let blocks_per_backup = set_size / 2;

    println!("Hint parameters:");
    println!("  Lambda: {}", args.lambda);
    println!("  Total hints: {}", total_hints);
    println!("  Regular hints: {} (blocks/hint: {})", args.lambda * chunk_size as u32, blocks_per_regular);
    println!("  Backup hints: {} (blocks/hint: {})", args.lambda * chunk_size as u32, blocks_per_backup);
    println!();

    // Create fake block keys (for benchmarking, actual keys don't matter)
    let block_keys: Vec<[u8; 16]> = (0..set_size)
        .map(|i| {
            let mut key = [0u8; 16];
            key[0..8].copy_from_slice(&i.to_le_bytes());
            key
        })
        .collect();

    // Create fake hint subsets (for benchmarking, use random pattern)
    let subset_bytes_per_hint = (set_size as usize + 7) / 8;
    let mut hint_subsets = vec![0u8; total_hints as usize * subset_bytes_per_hint];

    // Fill with pattern: each hint includes ~50% of blocks
    for hint_idx in 0..total_hints as usize {
        for block_idx in 0..set_size as usize {
            // Include block if (hint_idx + block_idx) is even
            if (hint_idx + block_idx) % 2 == 0 {
                let byte_idx = hint_idx * subset_bytes_per_hint + (block_idx / 8);
                let bit_mask = 1u8 << (block_idx % 8);
                hint_subsets[byte_idx] |= bit_mask;
            }
        }
    }

    // Get database bytes
    let entries = &db.mmap[..];

    // Store GPU mean time for speedup comparison
    #[allow(unused_mut)]
    let mut gpu_mean_time: Option<f64> = None;

    #[cfg(feature = "cuda")]
    {
        println!("Initializing GPU...");
        let gpu = GpuHintGenerator::new(0)?;
        println!("  Device: {}", gpu.device_name());
        println!();

        // Convert block keys to GPU format
        let gpu_block_keys: Vec<IprfBlockKey> = block_keys
            .iter()
            .map(|k| IprfBlockKey { key: *k })
            .collect();

        let params = PlinkoParams {
            num_entries,
            chunk_size,
            set_size,
            lambda: args.lambda,
            total_hints,
            blocks_per_hint: blocks_per_regular as u32,
            _pad: 0,
        };

        // Warmup
        println!("Warmup ({} iterations)...", args.warmup);
        for _ in 0..args.warmup {
            let _ = gpu.generate_hints(entries, &gpu_block_keys, &hint_subsets, params)?;
        }
        println!();

        // Benchmark
        println!("Benchmarking ({} iterations)...", args.iterations);
        let mut times = Vec::with_capacity(args.iterations as usize);

        for i in 0..args.iterations {
            let start = Instant::now();
            let hints = gpu.generate_hints(entries, &gpu_block_keys, &hint_subsets, params)?;
            let elapsed = start.elapsed();
            times.push(elapsed.as_secs_f64());

            println!(
                "  Iteration {}: {:.3} ms ({} hints)",
                i + 1,
                elapsed.as_secs_f64() * 1000.0,
                hints.len()
            );
        }

        // Statistics
        let mean = times.iter().sum::<f64>() / times.len() as f64;
        gpu_mean_time = Some(mean);
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = times[times.len() / 2];
        let min = times[0];
        let max = times[times.len() - 1];

        println!();
        println!("Results (GPU):");
        println!("  Mean:   {:.3} ms", mean * 1000.0);
        println!("  Median: {:.3} ms", median * 1000.0);
        println!("  Min:    {:.3} ms", min * 1000.0);
        println!("  Max:    {:.3} ms", max * 1000.0);
        println!();

        // Throughput
        let hints_per_sec = total_hints as f64 / mean;
        let entries_per_sec = (total_hints as f64 * blocks_per_regular as f64 * chunk_size as f64) / mean;
        println!("Throughput:");
        println!("  {:.2} hints/sec", hints_per_sec);
        println!("  {:.2} M entries processed/sec", entries_per_sec / 1e6);
    }

    #[cfg(not(feature = "cuda"))]
    {
        println!("CUDA feature not enabled. Build with --features cuda");
        println!();
    }

    // CPU baseline (optional)
    if args.cpu_baseline {
        println!();
        println!("Running CPU baseline...");

        let cpu = plinko::gpu::CpuHintGenerator::new();

        let start = Instant::now();
        let hints = cpu.generate_hints(
            entries,
            &block_keys,
            &hint_subsets,
            num_entries,
            chunk_size,
            set_size,
            total_hints,
        );
        let elapsed = start.elapsed();

        println!("CPU time: {:.3} ms ({} hints)", elapsed.as_secs_f64() * 1000.0, hints.len());

        if let Some(gpu_mean) = gpu_mean_time {
            let speedup = elapsed.as_secs_f64() / gpu_mean;
            println!("GPU speedup: {:.1}x", speedup);
        }
    }

    Ok(())
}
