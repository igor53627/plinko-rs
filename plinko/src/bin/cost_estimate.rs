//! Cost Attribution Estimator for Plinko PIR
//!
//! Estimates resource costs (storage, GPU compute, CPU compute, memory) for a
//! given database size and security configuration. Uses measured benchmark
//! constants from production runs — no CUDA dependency required.
//!
//! Usage:
//!   cargo run -p plinko --bin cost_estimate -- --mainnet
//!   cargo run -p plinko --bin cost_estimate -- --entries 100000000 --gpus 2

use clap::Parser;
use plinko::db::derive_plinko_params;

/// Measured GPU throughput: 3,012 hints/sec per H200 at production set_size (c=16404).
/// Source: docs/BENCHMARK_RESULTS.md — "Interleaved + High Occupancy" kernel.
const GPU_HINTS_PER_SEC_H200: f64 = 3_012.0;

/// Measured CPU throughput for hint generation: 55.8 MB/s on bare-metal 64-core.
/// This is the rate at which the CPU streams through the database file during
/// hint accumulation (XOR parity computation).
const CPU_THROUGHPUT_MBPS_BARE: f64 = 55.8;

/// TEE (Trusted Execution Environment) overhead multiplier.
/// CPU throughput under TEE is bare-metal throughput / TEE_OVERHEAD_FACTOR.
const TEE_OVERHEAD_FACTOR: f64 = 2.6;

/// V3 schema entry size (40 bytes per entry).
const ENTRY_SIZE_V3: u64 = 40;

/// GPU-expanded entry size (48 bytes, aligned for coalesced memory access).
const ENTRY_SIZE_GPU: u64 = 48;

/// Mainnet entry count (~1.83 billion entries, as extracted from Reth).
/// Source: docs/perf_kanban.md, scripts/modal_bench_mainnet.py
const MAINNET_ENTRIES: u64 = 1_830_000_000;

/// Default security parameter lambda.
const DEFAULT_LAMBDA: u64 = 128;

/// Modal H200 GPU hourly rate (USD).
/// Source: scripts/modal_run_bench.py gpu_prices dict.
const DEFAULT_GPU_HOURLY_RATE: f64 = 4.89;

/// Hint parity size in bytes (32 bytes = 256 bits).
/// Each regular hint stores a 32-byte parity. Each backup hint stores two
/// 32-byte parities (parity_in + parity_out) plus a 32-byte subset seed.
const HINT_PARITY_SIZE: u64 = 32;

/// Subset seed size stored per hint (32 bytes).
const HINT_SEED_SIZE: u64 = 32;

/// Account mapping entry size: 20B address + 4B index = 24 bytes.
const ACCOUNT_MAP_ENTRY_SIZE: u64 = 24;

/// Storage mapping entry size: 20B address + 32B slot key + 4B index = 56 bytes.
const STORAGE_MAP_ENTRY_SIZE: u64 = 56;

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Estimate Plinko PIR resource costs by component"
)]
struct Args {
    /// Number of database entries (mutually exclusive with --mainnet)
    #[arg(long, conflicts_with = "mainnet")]
    entries: Option<u64>,

    /// Use mainnet preset (~1.83B entries)
    #[arg(long)]
    mainnet: bool,

    /// Security parameter lambda
    #[arg(long, default_value_t = DEFAULT_LAMBDA)]
    lambda: u64,

    /// Number of GPUs for parallel hint generation
    #[arg(long, default_value_t = 1)]
    gpus: u64,

    /// GPU hourly rate in USD (default: Modal H200 at $4.89/hr)
    #[arg(long, default_value_t = DEFAULT_GPU_HOURLY_RATE)]
    gpu_hourly_rate: f64,

    /// Apply TEE overhead to CPU estimate
    #[arg(long)]
    tee: bool,

    /// Output as JSON instead of human-readable table
    #[arg(long)]
    json: bool,
}

fn format_bytes(bytes: u64) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = KB * 1024.0;
    const GB: f64 = MB * 1024.0;
    const TB: f64 = GB * 1024.0;
    let b = bytes as f64;
    if b >= TB {
        format!("{:.2} TB", b / TB)
    } else if b >= GB {
        format!("{:.2} GB", b / GB)
    } else if b >= MB {
        format!("{:.2} MB", b / MB)
    } else if b >= KB {
        format!("{:.2} KB", b / KB)
    } else {
        format!("{} B", bytes)
    }
}

fn format_duration(secs: f64) -> String {
    if secs < 60.0 {
        format!("{:.1}s", secs)
    } else if secs < 3600.0 {
        format!("{:.1} min", secs / 60.0)
    } else {
        let hours = secs / 3600.0;
        format!("{:.2} hr", hours)
    }
}

fn format_count(n: u64) -> String {
    if n >= 1_000_000_000 {
        format!("{:.3}B", n as f64 / 1e9)
    } else if n >= 1_000_000 {
        format!("{:.2}M", n as f64 / 1e6)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1e3)
    } else {
        format!("{}", n)
    }
}

fn main() {
    let args = Args::parse();

    let entries = if args.mainnet {
        MAINNET_ENTRIES
    } else if let Some(e) = args.entries {
        if e == 0 {
            eprintln!("Error: --entries must be > 0");
            std::process::exit(1);
        }
        e
    } else {
        eprintln!("Error: specify --entries <N> or --mainnet");
        std::process::exit(1);
    };

    // --- Parameter derivation (reuses crate logic) ---
    let (w, c) = derive_plinko_params(entries);
    let lambda = args.lambda;
    let num_regular = lambda * w;
    let num_backup = num_regular; // default: backup = regular
    let total_hints = num_regular + num_backup;
    let blocks_per_regular = c / 2 + 1;
    let blocks_per_backup = c / 2;

    // --- Storage ---
    let db_size = entries * ENTRY_SIZE_V3;

    // Regular hints: seed (32B) + parity (32B) = 64B each
    let regular_hint_storage = num_regular * (HINT_SEED_SIZE + HINT_PARITY_SIZE);
    // Backup hints: seed (32B) + parity_in (32B) + parity_out (32B) = 96B each
    let backup_hint_storage = num_backup * (HINT_SEED_SIZE + 2 * HINT_PARITY_SIZE);
    let total_hint_storage = regular_hint_storage + backup_hint_storage;

    // Mapping files (mainnet approximation: ~1.7% accounts, ~98.3% storage slots)
    // For non-mainnet, we estimate proportionally.
    let (num_accounts, num_storage_slots) = if args.mainnet {
        // Mainnet: ~30M accounts, ~1.8B storage slots
        (30_000_000u64, 1_800_000_000u64)
    } else {
        // Rough heuristic: ~1.7% accounts, rest storage
        let accts = (entries as f64 * 0.017).ceil() as u64;
        let slots = entries.saturating_sub(accts);
        (accts, slots)
    };
    let account_map_size = num_accounts * ACCOUNT_MAP_ENTRY_SIZE;
    let storage_map_size = num_storage_slots * STORAGE_MAP_ENTRY_SIZE;

    // --- GPU Compute ---
    let gpu_time_secs = total_hints as f64 / (GPU_HINTS_PER_SEC_H200 * args.gpus as f64);
    let gpu_hours = args.gpus as f64 * (gpu_time_secs / 3600.0);
    let gpu_cost = gpu_hours * args.gpu_hourly_rate;

    // --- CPU Compute ---
    let cpu_throughput = if args.tee {
        CPU_THROUGHPUT_MBPS_BARE / TEE_OVERHEAD_FACTOR
    } else {
        CPU_THROUGHPUT_MBPS_BARE
    };
    let db_size_mb = db_size as f64 / (1024.0 * 1024.0);
    let cpu_time_secs = db_size_mb / cpu_throughput;

    // --- Memory ---
    // GPU VRAM layout (from gpu.rs):
    //   1. Packed DB: num_entries * 48B (40B entries expanded to 48B aligned)
    //   2. Block keys: set_size * 32B (IprfBlockKey = 8 * u32)
    //   3. Hint subsets: total_hints * ceil(set_size / 8) bytes (bitset, 1 bit per block)
    //   4. Hint output: total_hints * 48B (HintOutput parity buffer)
    let vram_packed_db = entries * ENTRY_SIZE_GPU;
    let subset_bytes_per_hint = (c + 7) / 8;
    let vram_block_keys = c * 32;
    let vram_subsets = total_hints * subset_bytes_per_hint;
    let vram_output = total_hints * 48;
    let vram_total = vram_packed_db + vram_block_keys + vram_subsets + vram_output;

    // Host RAM: mmap'd database + hint arrays in memory
    let ram_db = db_size; // mmap'd, but counts toward resident memory
    let ram_regular = num_regular * (HINT_SEED_SIZE + HINT_PARITY_SIZE);
    let ram_backup = num_backup * (HINT_SEED_SIZE + 2 * HINT_PARITY_SIZE);
    let ram_total = ram_db + ram_regular + ram_backup;

    if args.json {
        print_json(
            entries, w, c, lambda, num_regular, num_backup, total_hints,
            blocks_per_regular, blocks_per_backup,
            db_size, regular_hint_storage, backup_hint_storage, total_hint_storage,
            account_map_size, storage_map_size,
            gpu_time_secs, gpu_hours, gpu_cost, &args,
            cpu_throughput, cpu_time_secs,
            vram_packed_db, vram_block_keys, vram_subsets, vram_output, vram_total,
            ram_db, ram_total,
        );
    } else {
        print_table(
            entries, w, c, lambda, num_regular, num_backup, total_hints,
            blocks_per_regular, blocks_per_backup,
            db_size, regular_hint_storage, backup_hint_storage, total_hint_storage,
            account_map_size, storage_map_size, num_accounts, num_storage_slots,
            gpu_time_secs, gpu_hours, gpu_cost, &args,
            cpu_throughput, cpu_time_secs,
            vram_packed_db, vram_block_keys, vram_subsets, vram_output, vram_total,
            ram_db, ram_total,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn print_table(
    entries: u64, w: u64, c: u64, lambda: u64,
    num_regular: u64, num_backup: u64, total_hints: u64,
    blocks_per_regular: u64, blocks_per_backup: u64,
    db_size: u64, regular_hint_storage: u64, backup_hint_storage: u64,
    total_hint_storage: u64,
    account_map_size: u64, storage_map_size: u64,
    num_accounts: u64, num_storage_slots: u64,
    gpu_time_secs: f64, gpu_hours: f64, gpu_cost: f64, args: &Args,
    cpu_throughput: f64, cpu_time_secs: f64,
    vram_packed_db: u64, vram_block_keys: u64, vram_subsets: u64,
    vram_output: u64, vram_total: u64,
    ram_db: u64, ram_total: u64,
) {
    println!("Plinko PIR Cost Estimate");
    println!("========================");
    println!();

    // Parameters
    println!("Parameters");
    println!("----------");
    println!("  Entries (N):           {:>14}  ({})", format_count(entries), entries);
    println!("  Chunk size (w):        {:>14}  (2^{})", format_count(w), (w as f64).log2().round() as u32);
    println!("  Set size (c):          {:>14}", format_count(c));
    println!("  Lambda:                {:>14}", lambda);
    println!("  Regular hints:         {:>14}  (lambda * w)", format_count(num_regular));
    println!("  Backup hints:          {:>14}  (lambda * w)", format_count(num_backup));
    println!("  Total hints:           {:>14}  (2 * lambda * w)", format_count(total_hints));
    println!("  Blocks/regular hint:   {:>14}  (c/2 + 1)", blocks_per_regular);
    println!("  Blocks/backup hint:    {:>14}  (c/2)", blocks_per_backup);
    println!();

    // Storage
    println!("Storage");
    println!("-------");
    println!("  database.bin:          {:>14}  ({} entries * {}B)", format_bytes(db_size), format_count(entries), ENTRY_SIZE_V3);
    println!("  account-mapping.bin:   {:>14}  ({} accounts * {}B)", format_bytes(account_map_size), format_count(num_accounts), ACCOUNT_MAP_ENTRY_SIZE);
    println!("  storage-mapping.bin:   {:>14}  ({} slots * {}B)", format_bytes(storage_map_size), format_count(num_storage_slots), STORAGE_MAP_ENTRY_SIZE);
    println!("  Regular hint storage:  {:>14}  ({} * {}B)", format_bytes(regular_hint_storage), format_count(num_regular), HINT_SEED_SIZE + HINT_PARITY_SIZE);
    println!("  Backup hint storage:   {:>14}  ({} * {}B)", format_bytes(backup_hint_storage), format_count(num_backup), HINT_SEED_SIZE + 2 * HINT_PARITY_SIZE);
    println!("  Total hint storage:    {:>14}", format_bytes(total_hint_storage));
    println!("  ─────────────────────────────────────");
    println!("  Total storage:         {:>14}", format_bytes(db_size + account_map_size + storage_map_size + total_hint_storage));
    println!();

    // GPU Compute
    println!("GPU Compute (H200)");
    println!("------------------");
    println!("  Throughput:            {:>14}  hints/sec/GPU", format_count(GPU_HINTS_PER_SEC_H200 as u64));
    println!("  GPUs:                  {:>14}", args.gpus);
    println!("  Wall-clock time:       {:>14}", format_duration(gpu_time_secs));
    println!("  GPU-hours:             {:>14.2} hr", gpu_hours);
    println!("  Rate:                  {:>14}", format!("${:.2}/GPU-hr", args.gpu_hourly_rate));
    println!("  Total GPU cost:        {:>14}", format!("${:.2}", gpu_cost));
    println!();

    // CPU Compute
    let mode = if args.tee { "TEE" } else { "bare metal" };
    println!("CPU Compute (64-core, {})", mode);
    println!("----------------------------");
    println!("  Throughput:            {:>14.1} MB/s", cpu_throughput);
    if args.tee {
        println!("  TEE overhead:          {:>14.1}x", TEE_OVERHEAD_FACTOR);
    }
    println!("  Database scan time:    {:>14}", format_duration(cpu_time_secs));
    println!();

    // Memory
    println!("Memory");
    println!("------");
    println!("  GPU VRAM (packed DB):  {:>14}  ({} * {}B)", format_bytes(vram_packed_db), format_count(entries), ENTRY_SIZE_GPU);
    println!("  GPU VRAM (block keys): {:>14}  ({} * 32B)", format_bytes(vram_block_keys), format_count(c));
    println!("  GPU VRAM (subsets):    {:>14}  (bitset, ceil(c/8) per hint)", format_bytes(vram_subsets));
    println!("  GPU VRAM (output):     {:>14}  ({} * 48B)", format_bytes(vram_output), format_count(total_hints));
    println!("  GPU VRAM total:        {:>14}", format_bytes(vram_total));
    println!("  Host RAM (DB mmap):    {:>14}", format_bytes(ram_db));
    println!("  Host RAM total:        {:>14}", format_bytes(ram_total));
}

#[allow(clippy::too_many_arguments)]
fn print_json(
    entries: u64, w: u64, c: u64, lambda: u64,
    num_regular: u64, num_backup: u64, total_hints: u64,
    blocks_per_regular: u64, blocks_per_backup: u64,
    db_size: u64, regular_hint_storage: u64, backup_hint_storage: u64,
    total_hint_storage: u64,
    account_map_size: u64, storage_map_size: u64,
    gpu_time_secs: f64, gpu_hours: f64, gpu_cost: f64, args: &Args,
    cpu_throughput: f64, cpu_time_secs: f64,
    vram_packed_db: u64, vram_block_keys: u64, vram_subsets: u64,
    vram_output: u64, vram_total: u64,
    ram_db: u64, ram_total: u64,
) {
    // Hand-rolled JSON to avoid adding serde dependency
    println!("{{");
    println!("  \"parameters\": {{");
    println!("    \"entries\": {},", entries);
    println!("    \"chunk_size_w\": {},", w);
    println!("    \"set_size_c\": {},", c);
    println!("    \"lambda\": {},", lambda);
    println!("    \"num_regular\": {},", num_regular);
    println!("    \"num_backup\": {},", num_backup);
    println!("    \"total_hints\": {},", total_hints);
    println!("    \"blocks_per_regular\": {},", blocks_per_regular);
    println!("    \"blocks_per_backup\": {}", blocks_per_backup);
    println!("  }},");
    println!("  \"storage_bytes\": {{");
    println!("    \"database_bin\": {},", db_size);
    println!("    \"account_mapping\": {},", account_map_size);
    println!("    \"storage_mapping\": {},", storage_map_size);
    println!("    \"regular_hints\": {},", regular_hint_storage);
    println!("    \"backup_hints\": {},", backup_hint_storage);
    println!("    \"total_hints\": {},", total_hint_storage);
    println!("    \"total\": {}", db_size + account_map_size + storage_map_size + total_hint_storage);
    println!("  }},");
    println!("  \"gpu_compute\": {{");
    println!("    \"throughput_hints_per_sec_per_gpu\": {},", GPU_HINTS_PER_SEC_H200 as u64);
    println!("    \"num_gpus\": {},", args.gpus);
    println!("    \"wall_clock_secs\": {:.1},", gpu_time_secs);
    println!("    \"gpu_hours\": {:.2},", gpu_hours);
    println!("    \"rate_per_gpu_hour_usd\": {:.2},", args.gpu_hourly_rate);
    println!("    \"total_cost_usd\": {:.2}", gpu_cost);
    println!("  }},");
    println!("  \"cpu_compute\": {{");
    println!("    \"throughput_mbps\": {:.1},", cpu_throughput);
    println!("    \"tee\": {},", args.tee);
    println!("    \"scan_time_secs\": {:.1}", cpu_time_secs);
    println!("  }},");
    println!("  \"memory_bytes\": {{");
    println!("    \"vram_packed_db\": {},", vram_packed_db);
    println!("    \"vram_block_keys\": {},", vram_block_keys);
    println!("    \"vram_subsets\": {},", vram_subsets);
    println!("    \"vram_output\": {},", vram_output);
    println!("    \"vram_total\": {},", vram_total);
    println!("    \"ram_db_mmap\": {},", ram_db);
    println!("    \"ram_total\": {}", ram_total);
    println!("  }}");
    println!("}}");
}
