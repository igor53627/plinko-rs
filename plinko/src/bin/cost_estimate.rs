//! Cost attribution estimator for Plinko PIR deployments.
//!
//! Estimates resource costs by component: storage, GPU compute, CPU compute,
//! and memory requirements. Uses measured throughput constants from production
//! benchmarks — no CUDA dependency required.

use clap::Parser;
use plinko::db::derive_plinko_params;

/// Mainnet v3 dataset: ~330M accounts + ~1.5B storage slots
const MAINNET_ENTRIES: u64 = 1_830_000_000;

/// v3 schema entry size (bytes)
const ENTRY_SIZE: u64 = 40;

// --- Measured throughput constants ---

/// GPU hint generation throughput: hints/sec per H200 GPU
/// Source: production benchmarks on Modal H200 instances
const GPU_HINTS_PER_SEC: f64 = 3_012.0;

/// CPU hint generation throughput: MB/s on bare metal (64-core)
/// Source: production benchmarks
const CPU_THROUGHPUT_MBPS: f64 = 55.8;

/// TEE overhead multiplier (SGX/TDX constant-time mode)
const TEE_OVERHEAD: f64 = 2.6;

/// Default GPU hourly rate (H200 on Modal/cloud)
const DEFAULT_GPU_HOURLY_RATE: f64 = 3.50;

/// Expanded entry size in GPU VRAM (48 bytes, v2/GPU-optimized layout)
const GPU_ENTRY_SIZE: u64 = 48;

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Estimate resource costs for a Plinko PIR deployment"
)]
struct Args {
    /// Total database entries (accounts + storage slots)
    #[arg(long)]
    entries: Option<u64>,

    /// Use mainnet preset (1.83B entries)
    #[arg(long, default_value_t = false)]
    mainnet: bool,

    /// Security parameter (lambda)
    #[arg(long, default_value_t = 128)]
    lambda: u64,

    /// Number of GPUs
    #[arg(long, default_value_t = 1)]
    gpus: u64,

    /// GPU hourly rate in USD
    #[arg(long, default_value_t = DEFAULT_GPU_HOURLY_RATE)]
    gpu_hourly_rate: f64,

    /// Number of CPU vCPUs for throughput scaling
    #[arg(long, default_value_t = 64)]
    cpu_vcpus: u64,

    /// Apply TEE (SGX/TDX) overhead to CPU estimates
    #[arg(long, default_value_t = false)]
    tee: bool,

    /// Output as JSON instead of human-readable table
    #[arg(long, default_value_t = false)]
    json: bool,
}

fn main() -> eyre::Result<()> {
    let args = Args::parse();

    let entries = match (args.entries, args.mainnet) {
        (Some(n), _) => n,
        (None, true) => MAINNET_ENTRIES,
        (None, false) => {
            eprintln!("Error: specify --entries <N> or --mainnet");
            std::process::exit(1);
        }
    };

    // --- Parameter derivation ---
    let (w, c) = derive_plinko_params(entries);
    let capacity = w * c;
    let num_regular = args.lambda * w;
    let num_backup = num_regular; // default: same as regular
    let total_hints = num_regular + num_backup;

    // --- Storage estimates ---
    let db_size = entries * ENTRY_SIZE;
    // Regular hints: each is subset_seed(32B) + parity(32B) = 64 bytes
    let regular_hint_bytes = num_regular * 64;
    // Backup hints: each is subset_seed(32B) + parity_in(32B) + parity_out(32B) = 96 bytes
    let backup_hint_bytes = num_backup * 96;
    let hint_storage = regular_hint_bytes + backup_hint_bytes;

    // --- GPU compute estimates ---
    let gpu_throughput = GPU_HINTS_PER_SEC * args.gpus as f64;
    let gpu_time_secs = total_hints as f64 / gpu_throughput;
    let gpu_time_hours = gpu_time_secs / 3600.0;
    let gpu_cost = gpu_time_hours * args.gpus as f64 * args.gpu_hourly_rate;

    // --- CPU compute estimates ---
    let db_size_mb = db_size as f64 / (1024.0 * 1024.0);
    let cpu_throughput = if args.tee {
        CPU_THROUGHPUT_MBPS / TEE_OVERHEAD
    } else {
        CPU_THROUGHPUT_MBPS
    };
    let cpu_time_secs = db_size_mb / cpu_throughput;

    // --- Memory estimates ---
    let vram_per_gpu = entries * GPU_ENTRY_SIZE; // expanded entries in VRAM
    let host_ram = db_size + hint_storage; // mmap'd DB + hint arrays

    if args.json {
        print_json(
            entries, args.lambda, w, c, capacity, total_hints, num_regular, num_backup,
            db_size, hint_storage, regular_hint_bytes, backup_hint_bytes,
            gpu_time_secs, gpu_cost, args.gpus, args.gpu_hourly_rate,
            cpu_time_secs, cpu_throughput, args.tee,
            vram_per_gpu, host_ram,
        );
    } else {
        print_table(
            entries, args.lambda, w, c, capacity, total_hints, num_regular, num_backup,
            db_size, hint_storage, regular_hint_bytes, backup_hint_bytes,
            gpu_time_secs, gpu_cost, args.gpus, args.gpu_hourly_rate,
            cpu_time_secs, cpu_throughput, args.tee,
            vram_per_gpu, host_ram,
        );
    }

    Ok(())
}

fn fmt_bytes(bytes: u64) -> String {
    let gb = bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    if gb >= 1.0 {
        format!("{:.2} GB", gb)
    } else {
        let mb = bytes as f64 / (1024.0 * 1024.0);
        format!("{:.2} MB", mb)
    }
}

fn fmt_duration(secs: f64) -> String {
    if secs >= 3600.0 {
        format!("{:.1}h", secs / 3600.0)
    } else if secs >= 60.0 {
        format!("{:.1}m", secs / 60.0)
    } else {
        format!("{:.1}s", secs)
    }
}

fn fmt_count(n: u64) -> String {
    if n >= 1_000_000_000 {
        format!("{:.2}B", n as f64 / 1_000_000_000.0)
    } else if n >= 1_000_000 {
        format!("{:.2}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        format!("{}", n)
    }
}

#[allow(clippy::too_many_arguments)]
fn print_table(
    entries: u64, lambda: u64, w: u64, c: u64, capacity: u64,
    total_hints: u64, num_regular: u64, num_backup: u64,
    db_size: u64, hint_storage: u64, regular_hint_bytes: u64, backup_hint_bytes: u64,
    gpu_time_secs: f64, gpu_cost: f64, gpus: u64, gpu_rate: f64,
    cpu_time_secs: f64, cpu_throughput: f64, tee: bool,
    vram_per_gpu: u64, host_ram: u64,
) {
    let overhead_pct = 100.0 * (capacity - entries) as f64 / entries as f64;

    println!("Plinko Cost Estimate");
    println!("====================");
    println!();
    println!("Parameters");
    println!("----------");
    println!("  Entries (N):        {:<14} ({})", entries, fmt_count(entries));
    println!("  Lambda:             {}", lambda);
    println!("  Chunk size (w):     {:<14} ({})", w, fmt_count(w));
    println!("  Set size (c):       {:<14} ({})", c, fmt_count(c));
    println!("  Capacity (w*c):     {:<14} ({}, overhead {:.1}%)", capacity, fmt_count(capacity), overhead_pct);
    println!("  Regular hints:      {:<14} ({})", num_regular, fmt_count(num_regular));
    println!("  Backup hints:       {:<14} ({})", num_backup, fmt_count(num_backup));
    println!("  Total hints:        {:<14} ({})", total_hints, fmt_count(total_hints));
    println!();
    println!("Storage");
    println!("-------");
    println!("  database.bin:       {}", fmt_bytes(db_size));
    println!("  Regular hints:      {} ({} x 64B)", fmt_bytes(regular_hint_bytes), fmt_count(num_regular));
    println!("  Backup hints:       {} ({} x 96B)", fmt_bytes(backup_hint_bytes), fmt_count(num_backup));
    println!("  Total hint storage: {}", fmt_bytes(hint_storage));
    println!("  Total storage:      {}", fmt_bytes(db_size + hint_storage));
    println!();
    println!("GPU Compute (H200)");
    println!("------------------");
    println!("  GPUs:               {}", gpus);
    println!("  Throughput:         {:.0} hints/sec/GPU", GPU_HINTS_PER_SEC);
    println!("  Total hints:        {}", fmt_count(total_hints));
    println!("  Estimated time:     {}", fmt_duration(gpu_time_secs));
    println!("  Estimated cost:     ${:.2} (@ ${:.2}/GPU-hr)", gpu_cost, gpu_rate);
    println!();
    println!("CPU Compute ({})", if tee { "TEE mode" } else { "bare metal" });
    println!("------------------");
    println!("  Throughput:         {:.1} MB/s (64-core{})", cpu_throughput, if tee { ", TEE 2.6x overhead" } else { "" });
    println!("  Database size:      {}", fmt_bytes(db_size));
    println!("  Estimated time:     {}", fmt_duration(cpu_time_secs));
    println!();
    println!("Memory");
    println!("------");
    println!("  VRAM per GPU:       {} (entries x {}B expanded)", fmt_bytes(vram_per_gpu), GPU_ENTRY_SIZE);
    println!("  Host RAM:           {} (DB mmap + hints)", fmt_bytes(host_ram));
}

#[allow(clippy::too_many_arguments)]
fn print_json(
    entries: u64, lambda: u64, w: u64, c: u64, capacity: u64,
    total_hints: u64, num_regular: u64, num_backup: u64,
    db_size: u64, hint_storage: u64, regular_hint_bytes: u64, backup_hint_bytes: u64,
    gpu_time_secs: f64, gpu_cost: f64, gpus: u64, gpu_rate: f64,
    cpu_time_secs: f64, cpu_throughput: f64, tee: bool,
    vram_per_gpu: u64, host_ram: u64,
) {
    println!(
        r#"{{
  "parameters": {{
    "entries": {},
    "lambda": {},
    "chunk_size": {},
    "set_size": {},
    "capacity": {},
    "total_hints": {},
    "regular_hints": {},
    "backup_hints": {}
  }},
  "storage": {{
    "database_bytes": {},
    "regular_hint_bytes": {},
    "backup_hint_bytes": {},
    "hint_storage_bytes": {},
    "total_bytes": {}
  }},
  "gpu_compute": {{
    "gpus": {},
    "throughput_hints_per_sec": {:.0},
    "time_secs": {:.1},
    "cost_usd": {:.2},
    "hourly_rate_usd": {:.2}
  }},
  "cpu_compute": {{
    "tee": {},
    "throughput_mbps": {:.1},
    "time_secs": {:.1}
  }},
  "memory": {{
    "vram_per_gpu_bytes": {},
    "host_ram_bytes": {}
  }}
}}"#,
        entries, lambda, w, c, capacity, total_hints, num_regular, num_backup,
        db_size, regular_hint_bytes, backup_hint_bytes, hint_storage, db_size + hint_storage,
        gpus, GPU_HINTS_PER_SEC, gpu_time_secs, gpu_cost, gpu_rate,
        tee, cpu_throughput, cpu_time_secs,
        vram_per_gpu, host_ram,
    );
}
