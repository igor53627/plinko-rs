//! GPU Hint Generation Benchmark
//!
//! Benchmarks hint generation on GPU (H100/H200) vs CPU.
//! Designed to run on Modal with synthetic datasets.
//!
//! Usage:
//!   cargo run -p plinko --bin bench_gpu_hints --features cuda -- --db data/synthetic/database.bin

use clap::Parser;
use eyre::Result;
use plinko::db::{derive_plinko_params, Database40};
use plinko::schema40::ENTRY_SIZE;
use std::path::PathBuf;
use std::time::Instant;

#[cfg(feature = "cuda")]
use plinko::gpu::{GpuHintGenerator, IprfBlockKey, PlinkoParams};

#[derive(Parser, Debug)]
#[command(author, version, about = "Benchmark GPU hint generation")]
struct Args {
    /// Path to database.bin (40-byte schema v3). Required unless --synthetic-entries is used.
    #[arg(long, required_unless_present = "synthetic_entries")]
    db: Option<PathBuf>,

    /// Generate synthetic database in-memory with N entries (avoids disk I/O)
    #[arg(long)]
    synthetic_entries: Option<u64>,

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

    /// Maximum hints to generate (for memory-constrained benchmarks)
    /// If set, limits total_hints and extrapolates full time
    #[arg(long)]
    max_hints: Option<u32>,

    /// Override set size (c) to simulate larger database workload
    /// Mainnet value: 16404
    #[arg(long)]
    set_size: Option<u64>,

    /// Output file path to save generated hints (binary format, 32 bytes per hint)
    /// If specified, runs in production mode (1 iteration, saves output)
    #[arg(long)]
    output: Option<PathBuf>,

    /// Starting hint index (for distributed generation)
    #[arg(long, default_value_t = 0)]
    hint_start: u32,

    /// Number of hints to generate (for distributed generation)
    /// If not specified, generates all hints
    #[arg(long)]
    hint_count: Option<u32>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("GPU Hint Generation Benchmark");
    println!("==============================");
    println!();

    // Load database or generate synthetic
    let (num_entries, chunk_size, set_size, entries_vec, entries_slice) =
        if let Some(n) = args.synthetic_entries {
            println!("Mode: Synthetic In-Memory ({} entries)", n);

            let (w, c) = derive_plinko_params(n);

            let total_bytes = n as usize * ENTRY_SIZE;
            println!(
                "Allocating {:.2} GB for database...",
                total_bytes as f64 / 1e9
            );

            // Allocate and fill with pattern to avoid zero-page optimizations
            let mut v = vec![0xAAu8; total_bytes];
            // Ensure vector is actually substantiated in memory
            v[0] = 0x01;
            v[total_bytes - 1] = 0x02;

            (n, w, c, Some(v), None) // Return ownership of vec in Option to keep it alive
        } else {
            let path = args.db.as_ref().unwrap();
            println!("Loading database: {:?}", path);
            let db = Database40::load(path)?;
            (db.num_entries, db.chunk_size, db.set_size, None, Some(db)) // Keep db alive
        };

    // Use override chunk_size if provided, otherwise use derived value
    let (chunk_size, mut set_size) = if let Some(w) = args.chunk_size {
        // Recalculate set_size for the overridden chunk_size
        let mut c = num_entries.div_ceil(w);
        c = c.div_ceil(4) * 4; // Round to multiple of 4
        println!("  Using override chunk_size (mainnet params)");
        (w, c)
    } else {
        (chunk_size, set_size)
    };

    // Override set_size if provided (to simulate mainnet workload on smaller DB)
    if let Some(c) = args.set_size {
        println!("  Using override set_size (mainnet params): {}", c);
        set_size = c;
    }

    // Get reference to entries slice
    let entries = if let Some(ref v) = entries_vec {
        &v[..]
    } else {
        &entries_slice.as_ref().unwrap().mmap[..]
    };

    println!("  Entries: {}", num_entries);
    println!(
        "  Chunk size (w): {} {}",
        chunk_size,
        if args.chunk_size.is_some() {
            "(override)"
        } else {
            ""
        }
    );
    println!("  Set size (c): {}", set_size);
    println!("  Entry size: {} bytes", ENTRY_SIZE);
    println!(
        "  Database size: {:.2} MB",
        (num_entries * ENTRY_SIZE as u64) as f64 / 1e6
    );
    println!();

    // Calculate hint parameters
    let full_total_hints = 2 * args.lambda * chunk_size as u32;
    let blocks_per_regular = set_size / 2 + 1;
    let blocks_per_backup = set_size / 2;

    // Determine hint range for this worker
    let hint_start = args.hint_start;
    if hint_start > full_total_hints {
        return Err(eyre::eyre!(
            "hint_start ({}) exceeds full_total_hints ({})",
            hint_start,
            full_total_hints
        ));
    }

    let remaining_hints = full_total_hints - hint_start;
    let hint_count = args
        .hint_count
        .unwrap_or(remaining_hints)
        .min(remaining_hints);

    // Apply max_hints limit if specified (for benchmarking)
    #[allow(unused_variables)]
    let (total_hints, hints_scale_factor) = if let Some(max) = args.max_hints {
        if max == 0 {
            return Err(eyre::eyre!("max_hints must be > 0"));
        }
        if max < hint_count {
            println!(
                "  NOTE: Limiting hints to {} (of {}) due to memory constraints",
                max, hint_count
            );
            (max, hint_count as f64 / max as f64)
        } else {
            (hint_count, 1.0)
        }
    } else {
        (hint_count, 1.0)
    };

    // Production mode check
    let production_mode = args.output.is_some();
    if production_mode {
        println!(
            "  PRODUCTION MODE: Will save hints to {:?}",
            args.output.as_ref().unwrap()
        );
        println!(
            "  Hint range: {} to {} ({} hints)",
            hint_start,
            hint_start.saturating_add(total_hints),
            total_hints
        );
    }

    println!("Hint parameters:");
    println!("  Lambda: {}", args.lambda);
    println!("  Full total hints: {}", full_total_hints);
    println!("  Benchmark hints: {}", total_hints);
    println!(
        "  Regular hints: {} (blocks/hint: {})",
        args.lambda * chunk_size as u32,
        blocks_per_regular
    );
    println!(
        "  Backup hints: {} (blocks/hint: {})",
        args.lambda * chunk_size as u32,
        blocks_per_backup
    );
    println!();

    // Create fake block keys (for benchmarking, actual keys don't matter)
    // ChaCha uses 256-bit keys (8 × u32)
    let block_keys: Vec<[u32; 8]> = (0..set_size)
        .map(|i| {
            let mut key = [0u32; 8];
            key[0] = i as u32;
            key[1] = (i >> 32) as u32;
            key
        })
        .collect();

    // Create fake hint subsets (for benchmarking, use random pattern)
    // IMPORTANT: In production mode with distributed workers, we must ensure
    // hint_subsets corresponds to the correct range.
    // Ideally we would accept real subsets or generate them deterministically.
    // For now, we generate synthetic subsets for range 0..total_hints, which
    // is consistent with the GPU generating hints 0..total_hints locally.
    // The GPU kernel will apply hint_start offset for the iPRF value.
    let subset_bytes_per_hint = (set_size as usize).div_ceil(8);
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

    // Store GPU mean time for speedup comparison
    #[allow(unused_mut)]
    let mut gpu_mean_time: Option<f64> = None;

    #[cfg(feature = "cuda")]
    {
        use std::fs::File;
        use std::io::Write;

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
            hint_start_offset: hint_start, // Pass hint_start to kernel
        };

        if production_mode {
            // Production mode: single run, save output
            println!(
                "Generating {} hints (starting at {})...",
                total_hints, hint_start
            );
            println!("Calling GPU generate_hints...");
            let start = Instant::now();
            let hints = gpu.generate_hints(entries, &gpu_block_keys, &hint_subsets, params)?;
            let elapsed = start.elapsed();
            println!(
                "GPU generate_hints completed in {:.3}s",
                elapsed.as_secs_f64()
            );

            println!(
                "  Generation time: {:.3} s ({} hints)",
                elapsed.as_secs_f64(),
                hints.len()
            );
            println!(
                "  Throughput: {:.0} hints/sec",
                hints.len() as f64 / elapsed.as_secs_f64()
            );

            // Save to file
            let output_path = args.output.as_ref().unwrap();
            println!("Saving hints to {:?}...", output_path);
            let mut file = File::create(output_path)?;

            // Write hints as raw bytes (40 bytes per hint - strip 8 bytes of padding)
            for hint in &hints {
                file.write_all(&hint.parity[0..40])?;
            }
            file.flush()?;

            let file_size = hints.len() * 40;
            println!(
                "  Saved {} hints ({} bytes, {:.2} MB)",
                hints.len(),
                file_size,
                file_size as f64 / 1e6
            );
            println!();
            println!("OUTPUT_FILE={}", output_path.display());
            println!("HINTS_GENERATED={}", hints.len());
            println!("GENERATION_TIME_MS={:.0}", elapsed.as_secs_f64() * 1000.0);
        } else {
            // Benchmark mode: warmup + multiple iterations
            println!("Warmup ({} iterations)...", args.warmup);
            for i in 0..args.warmup {
                println!("Warmup iteration {}...", i + 1);
                let _ = gpu.generate_hints(entries, &gpu_block_keys, &hint_subsets, params)?;
            }
            println!();

            println!("Benchmarking ({} iterations)...", args.iterations);
            let mut times = Vec::with_capacity(args.iterations as usize);

            for i in 0..args.iterations {
                println!("Benchmark iteration {}...", i + 1);
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
            println!("Results (GPU - {} hints):", total_hints);
            println!("  Mean:   {:.3} ms", mean * 1000.0);
            println!("  Median: {:.3} ms", median * 1000.0);
            println!("  Min:    {:.3} ms", min * 1000.0);
            println!("  Max:    {:.3} ms", max * 1000.0);

            if hints_scale_factor > 1.0 {
                let extrapolated_mean = mean * hints_scale_factor;
                println!();
                println!("Extrapolated (full {} hints):", full_total_hints);
                println!("  Est. time: {:.3} sec", extrapolated_mean);
            }
            println!();

            // Throughput
            let hints_per_sec = total_hints as f64 / mean;
            let entries_per_sec =
                (total_hints as f64 * blocks_per_regular as f64 * chunk_size as f64) / mean;
            println!("Throughput:");
            println!("  {:.2} hints/sec", hints_per_sec);
            println!("  {:.2} M entries processed/sec", entries_per_sec / 1e6);
        }
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

        println!(
            "CPU time: {:.3} ms ({} hints)",
            elapsed.as_secs_f64() * 1000.0,
            hints.len()
        );

        if let Some(gpu_mean) = gpu_mean_time {
            let speedup = elapsed.as_secs_f64() / gpu_mean;
            println!("GPU speedup: {:.1}x", speedup);
        }
    }

    Ok(())
}
