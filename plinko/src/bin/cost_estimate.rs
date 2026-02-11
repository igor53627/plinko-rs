//! Cost attribution estimator for Plinko PIR deployments.
//!
//! Estimates resource costs by component: storage, GPU compute, CPU compute,
//! and memory requirements. Uses measured throughput constants from production
//! benchmarks — no CUDA dependency required.

use clap::{value_parser, Parser};
use eyre::eyre;
use plinko::db::derive_plinko_params;
use serde::Serialize;

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
const BASELINE_CPU_VCPUS: u64 = 64;

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
    #[arg(long, value_parser = value_parser!(u64).range(1..))]
    entries: Option<u64>,

    /// Use mainnet preset (1.83B entries)
    #[arg(long, default_value_t = false)]
    mainnet: bool,

    /// Security parameter (lambda)
    #[arg(long, default_value_t = 128)]
    lambda: u64,

    /// Number of GPUs
    #[arg(long, default_value_t = 1, value_parser = value_parser!(u64).range(1..))]
    gpus: u64,

    /// GPU hourly rate in USD
    #[arg(long, default_value_t = DEFAULT_GPU_HOURLY_RATE)]
    gpu_hourly_rate: f64,

    /// Number of CPU vCPUs for throughput scaling
    #[arg(
        long,
        default_value_t = BASELINE_CPU_VCPUS,
        value_parser = value_parser!(u64).range(1..)
    )]
    cpu_vcpus: u64,

    /// Apply TEE (SGX/TDX) overhead to CPU estimates
    #[arg(long, default_value_t = false)]
    tee: bool,

    /// Output as JSON instead of human-readable table
    #[arg(long, default_value_t = false)]
    json: bool,
}

#[derive(Serialize)]
struct CostEstimate {
    parameters: Parameters,
    storage: Storage,
    gpu_compute: GpuCompute,
    cpu_compute: CpuCompute,
    memory: Memory,
}

#[derive(Serialize)]
struct Parameters {
    entries: u64,
    lambda: u64,
    chunk_size: u64,
    set_size: u64,
    capacity: u64,
    total_hints: u64,
    regular_hints: u64,
    backup_hints: u64,
}

#[derive(Serialize)]
struct Storage {
    database_bytes: u64,
    regular_hint_bytes: u64,
    backup_hint_bytes: u64,
    hint_storage_bytes: u64,
    total_bytes: u64,
}

#[derive(Serialize)]
struct GpuCompute {
    gpus: u64,
    throughput_hints_per_sec: f64,
    time_secs: f64,
    cost_usd: f64,
    hourly_rate_usd: f64,
}

#[derive(Serialize)]
struct CpuCompute {
    tee: bool,
    vcpus: u64,
    throughput_mbps: f64,
    time_secs: f64,
}

#[derive(Serialize)]
struct Memory {
    vram_per_gpu_bytes: u64,
    host_ram_bytes: u64,
}

fn main() -> eyre::Result<()> {
    let args = Args::parse();

    let entries = match (args.entries, args.mainnet) {
        (Some(n), _) => n,
        (None, true) => MAINNET_ENTRIES,
        (None, false) => return Err(eyre!("specify --entries <N> or --mainnet")),
    };

    // --- Parameter derivation ---
    let (w, c) = derive_plinko_params(entries);
    let capacity = checked_mul(w, c, "capacity")?;
    let num_regular = checked_mul(args.lambda, w, "num_regular")?;
    let num_backup = num_regular; // default: same as regular
    let total_hints = checked_add(num_regular, num_backup, "total_hints")?;

    if capacity < entries {
        return Err(eyre!(
            "derived capacity ({}) is smaller than entries ({})",
            capacity,
            entries
        ));
    }

    // --- Storage estimates ---
    let db_size = checked_mul(entries, ENTRY_SIZE, "database_bytes")?;
    // Regular hints: each is subset_seed(32B) + parity(32B) = 64 bytes
    let regular_hint_bytes = checked_mul(num_regular, 64, "regular_hint_bytes")?;
    // Backup hints: each is subset_seed(32B) + parity_in(32B) + parity_out(32B) = 96 bytes
    let backup_hint_bytes = checked_mul(num_backup, 96, "backup_hint_bytes")?;
    let hint_storage = checked_add(regular_hint_bytes, backup_hint_bytes, "hint_storage_bytes")?;
    let total_storage = checked_add(db_size, hint_storage, "total_storage_bytes")?;

    // --- GPU compute estimates ---
    let gpu_throughput = GPU_HINTS_PER_SEC * args.gpus as f64;
    let gpu_time_secs = total_hints as f64 / gpu_throughput;
    let gpu_time_hours = gpu_time_secs / 3600.0;
    let gpu_cost = gpu_time_hours * args.gpus as f64 * args.gpu_hourly_rate;

    // --- CPU compute estimates ---
    let db_size_mb = db_size as f64 / (1024.0 * 1024.0);
    let scaled_cpu_throughput =
        CPU_THROUGHPUT_MBPS * (args.cpu_vcpus as f64 / BASELINE_CPU_VCPUS as f64);
    let cpu_throughput = if args.tee {
        scaled_cpu_throughput / TEE_OVERHEAD
    } else {
        scaled_cpu_throughput
    };
    let cpu_time_secs = db_size_mb / cpu_throughput;

    ensure_finite(&[
        ("gpu_time_secs", gpu_time_secs),
        ("gpu_cost", gpu_cost),
        ("cpu_throughput", cpu_throughput),
        ("cpu_time_secs", cpu_time_secs),
    ])?;

    // --- Memory estimates ---
    let vram_per_gpu = checked_mul(entries, GPU_ENTRY_SIZE, "vram_per_gpu_bytes")?; // expanded entries in VRAM
    let host_ram = checked_add(db_size, hint_storage, "host_ram_bytes")?; // mmap'd DB + hint arrays

    let estimate = CostEstimate {
        parameters: Parameters {
            entries,
            lambda: args.lambda,
            chunk_size: w,
            set_size: c,
            capacity,
            total_hints,
            regular_hints: num_regular,
            backup_hints: num_backup,
        },
        storage: Storage {
            database_bytes: db_size,
            regular_hint_bytes,
            backup_hint_bytes,
            hint_storage_bytes: hint_storage,
            total_bytes: total_storage,
        },
        gpu_compute: GpuCompute {
            gpus: args.gpus,
            throughput_hints_per_sec: GPU_HINTS_PER_SEC,
            time_secs: gpu_time_secs,
            cost_usd: gpu_cost,
            hourly_rate_usd: args.gpu_hourly_rate,
        },
        cpu_compute: CpuCompute {
            tee: args.tee,
            vcpus: args.cpu_vcpus,
            throughput_mbps: cpu_throughput,
            time_secs: cpu_time_secs,
        },
        memory: Memory {
            vram_per_gpu_bytes: vram_per_gpu,
            host_ram_bytes: host_ram,
        },
    };

    if args.json {
        print_json(&estimate)?;
    } else {
        print_table(&estimate)?;
    }

    Ok(())
}

fn checked_mul(a: u64, b: u64, label: &str) -> eyre::Result<u64> {
    a.checked_mul(b)
        .ok_or_else(|| eyre!("overflow while computing {}", label))
}

fn checked_add(a: u64, b: u64, label: &str) -> eyre::Result<u64> {
    a.checked_add(b)
        .ok_or_else(|| eyre!("overflow while computing {}", label))
}

fn ensure_finite(values: &[(&str, f64)]) -> eyre::Result<()> {
    for (name, value) in values {
        if !value.is_finite() {
            return Err(eyre!(
                "computed non-finite value for {} ({}); reduce input magnitudes and retry",
                name,
                value
            ));
        }
    }
    Ok(())
}

fn compute_overhead_pct(entries: u64, capacity: u64) -> eyre::Result<f64> {
    if entries == 0 {
        return Err(eyre!("entries must be > 0"));
    }
    let overhead_entries = capacity.checked_sub(entries).ok_or_else(|| {
        eyre!(
            "capacity ({}) is smaller than entries ({})",
            capacity,
            entries
        )
    })?;
    Ok(100.0 * overhead_entries as f64 / entries as f64)
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

fn print_table(estimate: &CostEstimate) -> eyre::Result<()> {
    let overhead_pct =
        compute_overhead_pct(estimate.parameters.entries, estimate.parameters.capacity)?;

    println!("Plinko Cost Estimate");
    println!("====================");
    println!();
    println!("Parameters");
    println!("----------");
    println!(
        "  Entries (N):        {:<14} ({})",
        estimate.parameters.entries,
        fmt_count(estimate.parameters.entries)
    );
    println!("  Lambda:             {}", estimate.parameters.lambda);
    println!(
        "  Chunk size (w):     {:<14} ({})",
        estimate.parameters.chunk_size,
        fmt_count(estimate.parameters.chunk_size)
    );
    println!(
        "  Set size (c):       {:<14} ({})",
        estimate.parameters.set_size,
        fmt_count(estimate.parameters.set_size)
    );
    println!(
        "  Capacity (w*c):     {:<14} ({}, overhead {:.1}%)",
        estimate.parameters.capacity,
        fmt_count(estimate.parameters.capacity),
        overhead_pct
    );
    println!(
        "  Regular hints:      {:<14} ({})",
        estimate.parameters.regular_hints,
        fmt_count(estimate.parameters.regular_hints)
    );
    println!(
        "  Backup hints:       {:<14} ({})",
        estimate.parameters.backup_hints,
        fmt_count(estimate.parameters.backup_hints)
    );
    println!(
        "  Total hints:        {:<14} ({})",
        estimate.parameters.total_hints,
        fmt_count(estimate.parameters.total_hints)
    );
    println!();
    println!("Storage");
    println!("-------");
    println!(
        "  database.bin:       {}",
        fmt_bytes(estimate.storage.database_bytes)
    );
    println!(
        "  Regular hints:      {} ({} x 64B)",
        fmt_bytes(estimate.storage.regular_hint_bytes),
        fmt_count(estimate.parameters.regular_hints)
    );
    println!(
        "  Backup hints:       {} ({} x 96B)",
        fmt_bytes(estimate.storage.backup_hint_bytes),
        fmt_count(estimate.parameters.backup_hints)
    );
    println!(
        "  Total hint storage: {}",
        fmt_bytes(estimate.storage.hint_storage_bytes)
    );
    println!(
        "  Total storage:      {}",
        fmt_bytes(estimate.storage.total_bytes)
    );
    println!();
    println!("GPU Compute (H200)");
    println!("------------------");
    println!("  GPUs:               {}", estimate.gpu_compute.gpus);
    println!(
        "  Throughput:         {:.0} hints/sec/GPU",
        GPU_HINTS_PER_SEC
    );
    println!(
        "  Total hints:        {}",
        fmt_count(estimate.parameters.total_hints)
    );
    println!(
        "  Estimated time:     {}",
        fmt_duration(estimate.gpu_compute.time_secs)
    );
    println!(
        "  Estimated cost:     ${:.2} (@ ${:.2}/GPU-hr)",
        estimate.gpu_compute.cost_usd, estimate.gpu_compute.hourly_rate_usd
    );
    println!();
    println!(
        "CPU Compute ({})",
        if estimate.cpu_compute.tee {
            "TEE mode"
        } else {
            "bare metal"
        }
    );
    println!("------------------");
    println!(
        "  Throughput:         {:.1} MB/s ({}-vCPU{})",
        estimate.cpu_compute.throughput_mbps,
        estimate.cpu_compute.vcpus,
        if estimate.cpu_compute.tee {
            ", TEE 2.6x overhead"
        } else {
            ""
        }
    );
    println!(
        "  Database size:      {}",
        fmt_bytes(estimate.storage.database_bytes)
    );
    println!(
        "  Estimated time:     {}",
        fmt_duration(estimate.cpu_compute.time_secs)
    );
    println!();
    println!("Memory");
    println!("------");
    println!(
        "  VRAM per GPU:       {} (entries x {}B expanded)",
        fmt_bytes(estimate.memory.vram_per_gpu_bytes),
        GPU_ENTRY_SIZE
    );
    println!(
        "  Host RAM:           {} (DB mmap + hints)",
        fmt_bytes(estimate.memory.host_ram_bytes)
    );
    Ok(())
}

fn print_json(estimate: &CostEstimate) -> eyre::Result<()> {
    println!("{}", serde_json::to_string_pretty(estimate)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_mul_overflow_is_error() {
        assert!(checked_mul(u64::MAX, 2, "x").is_err());
    }

    #[test]
    fn checked_add_overflow_is_error() {
        assert!(checked_add(u64::MAX, 1, "x").is_err());
    }

    #[test]
    fn clap_rejects_zero_entries() {
        let args = Args::try_parse_from(["cost_estimate", "--entries", "0"]);
        assert!(args.is_err());
    }

    #[test]
    fn clap_rejects_zero_gpus() {
        let args = Args::try_parse_from(["cost_estimate", "--entries", "1", "--gpus", "0"]);
        assert!(args.is_err());
    }

    #[test]
    fn clap_rejects_zero_cpu_vcpus() {
        let args = Args::try_parse_from(["cost_estimate", "--entries", "1", "--cpu-vcpus", "0"]);
        assert!(args.is_err());
    }

    #[test]
    fn ensure_finite_rejects_infinity() {
        let result = ensure_finite(&[("x", f64::INFINITY)]);
        assert!(result.is_err());
    }

    #[test]
    fn compute_overhead_pct_rejects_invalid_capacity() {
        let result = compute_overhead_pct(10, 9);
        assert!(result.is_err());
    }
}
