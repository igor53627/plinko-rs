//! Benchmark and correctness harness for Multi-level PIR (`ePrint 2026/1273`, §4)
//! comparing Variant A (Hybrid: `Iprf` on early levels + `FlatAes` on upper levels),
//! Variant B (`FlatAes` on all levels), and baseline Fig. 7 Plinko.

use std::num::NonZeroUsize;
use std::time::{Duration, Instant};

use clap::{Parser, ValueEnum};
use plinko::hint_init::available_threads;
use plinko::multilevel::{
    BackendVariant, LevelBackend, LevelSpec, MultiLevelAnswer, MultiLevelClient, MultiLevelParams,
    MultiLevelServer, PendingMultiLevelQuery,
};
use plinko::protocol::{
    compute_hints_with_threads, Client as PlinkoClient, ProtocolParams, Server as PlinkoServer,
};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

/// Benchmark preset selecting the database scale.
#[derive(Debug, Clone, Copy, ValueEnum)]
enum Preset {
    /// 2^20 (~1M) records × 40 B (41.9 MB): full Fig. 7 Plinko vs 2-level A & B.
    Small,
    /// 2^24 (~16.7M) records × 40 B (671 MB): full 2-level and 6-level A & B.
    Medium,
    /// Ethereum mainnet v3 scale reference (1,834,095,877 × 40 B): real UpdateHint, Query,
    /// and per-block measured HintInit extrapolation for 2-level and 6-level A & B.
    Ethereum,
    /// Run all three presets.
    All,
}

#[derive(Debug, Parser)]
#[command(name = "bench_multilevel")]
struct Args {
    /// Scale preset to run.
    #[arg(long, value_enum, default_value_t = Preset::All)]
    preset: Preset,

    /// Worker thread count (defaults to available hardware threads; must be >= 1).
    #[arg(long)]
    threads: Option<NonZeroUsize>,

    /// Record size in bytes (must be >= 1).
    #[arg(long, default_value = "40")]
    record_size: NonZeroUsize,
}

fn make_synth_db(num_entries: usize, record_size: usize) -> Vec<u8> {
    // Fast deterministic pattern (avoids spending seconds in ChaCha20 for 671 MB).
    let mut db: Vec<u8> = vec![0u8; num_entries * record_size];
    for (i, chunk) in db.chunks_exact_mut(8).enumerate() {
        let v: u64 = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xA5A5_5A5A_1234_5678;
        chunk.copy_from_slice(&v.to_le_bytes());
    }
    db
}

fn fmt_bytes(bytes: usize) -> String {
    if bytes >= 1024 * 1024 * 1024 {
        format!("{:.2} GiB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    } else if bytes >= 1024 * 1024 {
        format!("{:.2} MiB", bytes as f64 / (1024.0 * 1024.0))
    } else if bytes >= 1024 {
        format!("{:.1} KiB", bytes as f64 / 1024.0)
    } else {
        format!("{bytes} B")
    }
}

fn fmt_dur(d: Duration) -> String {
    let us: f64 = d.as_secs_f64() * 1e6;
    if us >= 1e6 {
        format!("{:.2} s", d.as_secs_f64())
    } else if us >= 1e3 {
        format!("{:.2} ms", us / 1e3)
    } else {
        format!("{:.1} us", us)
    }
}

fn print_params_summary(label: &str, params: &MultiLevelParams) {
    println!(
        "\n--- {label} | N={} r={} B | levels={} | hints={} ({}) | p_fail={:.2e} ---",
        params.num_entries,
        params.record_size,
        params.levels.len(),
        params.total_hints_all_levels(),
        fmt_bytes(params.total_hint_bytes()),
        params.total_failure_probability(),
    );
    for (i, s) in params.levels.iter().enumerate() {
        println!(
            "  L{}: {:<7} w={:<6} c={:<9} M_reg={:<7} Q_bak={:<7} mem={:<10} upload={:<9} p_fail={:.3e}",
            i + 1,
            match s.backend {
                LevelBackend::Iprf => "Iprf",
                LevelBackend::FlatAes => "FlatAes",
            },
            s.block_size,
            s.num_blocks,
            s.num_primary,
            s.num_backup,
            fmt_bytes(s.hint_bytes(params.record_size)),
            fmt_bytes(s.query_upload_bytes()),
            s.failure_probability(),
        );
    }
}

fn run_plinko_baseline(db: &[u8], num_entries: usize, record_size: usize, threads: usize) {
    // Match ~2^-92 failure prob: lambda = 128 gives exp(-lambda/2) = exp(-64) ~= 1.6e-28.
    let w: usize = (num_entries as f64).sqrt().ceil() as usize;
    let w: usize = w.next_power_of_two();
    let lambda: usize = 128;
    let q_backup: usize = w; // fair budget comparison: Q = sqrt(N) backups
    let params: ProtocolParams =
        ProtocolParams::new(num_entries, record_size, Some(w), lambda, Some(q_backup)).unwrap();
    let seed: [u8; 32] = [0x11; 32];

    let hint_bytes: usize = 144
        + params.num_regular_hints * (4 + record_size)
        + params.num_backup_hints * (4 + 2 * record_size);

    println!(
        "\n--- Baseline Fig. 7 Plinko (lambda={lambda}, w={w}, Q={q_backup}) | N={num_entries} r={record_size} B | hints={} ({}) ---",
        params.num_total_hints,
        fmt_bytes(hint_bytes),
    );

    let t0: Instant = Instant::now();
    let hints = compute_hints_with_threads(&params, &seed, 0, db, threads).unwrap();
    let hint_init_wall: Duration = t0.elapsed();

    let query_seed: [u8; 32] = [0x44; 32];
    let mut client: PlinkoClient =
        PlinkoClient::from_hints(params.clone(), seed, query_seed, hints).unwrap();
    client.set_threads(threads);
    let mut server: PlinkoServer = PlinkoServer::new(params.clone(), 0, db).unwrap();
    let mut expected_db: Vec<u8> = db.to_vec();
    let mut rng: ChaCha20Rng = ChaCha20Rng::seed_from_u64(101);

    // Measure 20 queries + E2E verification.
    let q_count: usize = 20;
    let mut q_prep_total: Duration = Duration::ZERO;
    let mut q_srv_total: Duration = Duration::ZERO;
    for _ in 0..q_count {
        let idx: usize = rng.gen_range(0..num_entries);
        let tp: Instant = Instant::now();
        let prepared = client.prepare_query(idx).unwrap();
        q_prep_total += tp.elapsed();

        let ts: Instant = Instant::now();
        let ans = server.answer(&prepared.query).unwrap();
        q_srv_total += ts.elapsed();

        let got: Vec<u8> = client.reconstruct_and_replenish(&prepared, &ans).unwrap();
        assert_eq!(
            &got[..],
            &expected_db[idx * record_size..(idx + 1) * record_size]
        );
    }

    // Measure single UpdateHint (1 thread) and 1000-entry batch UpdateHint (1 thread & threads).
    let mut upd_1_total: Duration = Duration::ZERO;
    client.set_threads(1);
    for _ in 0..20 {
        let idx: usize = rng.gen_range(0..num_entries);
        let mut new_rec: Vec<u8> = vec![0u8; record_size];
        rng.fill(&mut new_rec[..]);
        let delta = server.apply_updates(&[(idx, new_rec.clone())]).unwrap();
        expected_db[idx * record_size..(idx + 1) * record_size].copy_from_slice(&new_rec);
        let tu: Instant = Instant::now();
        client.apply_delta(&delta).unwrap();
        upd_1_total += tu.elapsed();
    }

    let batch_size: usize = 1000;
    let updates_1: Vec<(usize, Vec<u8>)> = (0..batch_size)
        .map(|k| {
            let idx: usize = (k * (num_entries / batch_size)) % num_entries;
            let mut rec: Vec<u8> = vec![0u8; record_size];
            rng.fill(&mut rec[..]);
            (idx, rec)
        })
        .collect();
    let batch_delta_1 = server.apply_updates(&updates_1).unwrap();
    client.set_threads(1);
    let tb1: Instant = Instant::now();
    client.apply_delta(&batch_delta_1).unwrap();
    let batch_1t: Duration = tb1.elapsed();

    let updates_2: Vec<(usize, Vec<u8>)> = (0..batch_size)
        .map(|k| {
            let idx: usize = (k * (num_entries / batch_size) + 1) % num_entries;
            let mut rec: Vec<u8> = vec![0u8; record_size];
            rng.fill(&mut rec[..]);
            (idx, rec)
        })
        .collect();
    let batch_delta_2 = server.apply_updates(&updates_2).unwrap();
    client.set_threads(threads);
    let tbm: Instant = Instant::now();
    client.apply_delta(&batch_delta_2).unwrap();
    let batch_mt: Duration = tbm.elapsed();

    println!(
        "  HintInit ({threads}T): {} | Mem: {} | UpdateHint 1-rec (1T): {} | Batch 1000 (1T / {threads}T): {} / {} | Query prep ({threads}T): {} | Server: {}",
        fmt_dur(hint_init_wall),
        fmt_bytes(hint_bytes),
        fmt_dur(upd_1_total / 20),
        fmt_dur(batch_1t),
        fmt_dur(batch_mt),
        fmt_dur(q_prep_total / q_count as u32),
        fmt_dur(q_srv_total / q_count as u32),
    );
}

fn run_multilevel_full(
    label: &str,
    params: MultiLevelParams,
    db: &[u8],
    threads: usize,
    batch_size: usize,
) {
    print_params_summary(label, &params);
    let seed: [u8; 32] = [0x22; 32];
    let r: usize = params.record_size;
    let n: usize = params.num_entries;

    let t0: Instant = Instant::now();
    let mut client: MultiLevelClient =
        MultiLevelClient::hint_init(params.clone(), &seed, db, threads).unwrap();
    let hint_init_wall: Duration = t0.elapsed();

    let mut db_mut: Vec<u8> = db.to_vec();
    let mut rng: ChaCha20Rng = ChaCha20Rng::seed_from_u64(202);

    // Measure 30 queries with E2E verification + interleaved deltas.
    let q_count: usize = 30;
    let mut q_prep_total: Duration = Duration::ZERO;
    let mut q_srv_total: Duration = Duration::ZERO;
    let mut q_fin_total: Duration = Duration::ZERO;
    let mut level_hits: Vec<usize> = vec![0; params.levels.len()];

    for _ in 0..q_count {
        let idx: usize = rng.gen_range(0..n);
        let tp: Instant = Instant::now();
        let pending: PendingMultiLevelQuery = client.prepare_query(idx, &mut rng).unwrap();
        q_prep_total += tp.elapsed();
        level_hits[pending.level] += 1;

        let ts: Instant = Instant::now();
        let ans: MultiLevelAnswer =
            MultiLevelServer::answer(&db_mut, &params, &pending.query).unwrap();
        q_srv_total += ts.elapsed();

        let tf: Instant = Instant::now();
        let got: Vec<u8> = client.finish_query(pending, &ans, &mut rng).unwrap();
        q_fin_total += tf.elapsed();

        assert_eq!(&got[..], &db_mut[idx * r..(idx + 1) * r]);
    }

    // Measure single UpdateHint (1 thread).
    let mut upd_1_total: Duration = Duration::ZERO;
    for _ in 0..50 {
        let idx: usize = rng.gen_range(0..n);
        let mut delta: Vec<u8> = vec![0u8; r];
        rng.fill(&mut delta[..]);
        for (a, &b) in db_mut[idx * r..(idx + 1) * r].iter_mut().zip(&delta) {
            *a ^= b;
        }
        let tu: Instant = Instant::now();
        client.apply_delta(idx, &delta).unwrap();
        upd_1_total += tu.elapsed();
    }

    // Verify a post-delta query is still 100% exact.
    let check_idx: usize = rng.gen_range(0..n);
    let pending: PendingMultiLevelQuery = client.prepare_query(check_idx, &mut rng).unwrap();
    let ans: MultiLevelAnswer = MultiLevelServer::answer(&db_mut, &params, &pending.query).unwrap();
    let got: Vec<u8> = client.finish_query(pending, &ans, &mut rng).unwrap();
    assert_eq!(&got[..], &db_mut[check_idx * r..(check_idx + 1) * r]);

    // Measure batch UpdateHint (1T and multi-thread), verifying retrieval after each batch.
    let updates_1t: Vec<(usize, Vec<u8>)> = (0..batch_size)
        .map(|k| {
            let idx: usize = (k * (n / batch_size.max(1))) % n;
            let mut d: Vec<u8> = vec![0u8; r];
            rng.fill(&mut d[..]);
            (idx, d)
        })
        .collect();
    let tb1: Instant = Instant::now();
    client.apply_deltas_batch(&updates_1t, 1).unwrap();
    let batch_1t: Duration = tb1.elapsed();
    for (idx, d) in &updates_1t {
        for (a, &b) in db_mut[idx * r..(idx + 1) * r].iter_mut().zip(d) {
            *a ^= b;
        }
    }
    let verify_idx_1t: usize = updates_1t[updates_1t.len() / 2].0;
    let p_1t: PendingMultiLevelQuery = client.prepare_query(verify_idx_1t, &mut rng).unwrap();
    let a_1t: MultiLevelAnswer = MultiLevelServer::answer(&db_mut, &params, &p_1t.query).unwrap();
    let g_1t: Vec<u8> = client.finish_query(p_1t, &a_1t, &mut rng).unwrap();
    assert_eq!(
        &g_1t[..],
        &db_mut[verify_idx_1t * r..(verify_idx_1t + 1) * r]
    );

    let updates_mt: Vec<(usize, Vec<u8>)> = (0..batch_size)
        .map(|k| {
            let idx: usize = (k * (n / batch_size.max(1)) + 1) % n;
            let mut d: Vec<u8> = vec![0u8; r];
            rng.fill(&mut d[..]);
            (idx, d)
        })
        .collect();
    let tbm: Instant = Instant::now();
    client.apply_deltas_batch(&updates_mt, threads).unwrap();
    let batch_mt: Duration = tbm.elapsed();
    for (idx, d) in &updates_mt {
        for (a, &b) in db_mut[idx * r..(idx + 1) * r].iter_mut().zip(d) {
            *a ^= b;
        }
    }
    let verify_idx_mt: usize = updates_mt[updates_mt.len() / 2].0;
    let p_mt: PendingMultiLevelQuery = client.prepare_query(verify_idx_mt, &mut rng).unwrap();
    let a_mt: MultiLevelAnswer = MultiLevelServer::answer(&db_mut, &params, &p_mt.query).unwrap();
    let g_mt: Vec<u8> = client.finish_query(p_mt, &a_mt, &mut rng).unwrap();
    assert_eq!(
        &g_mt[..],
        &db_mut[verify_idx_mt * r..(verify_idx_mt + 1) * r]
    );

    println!(
        "  HintInit ({threads}T): {} | HintMem (est): {} | UpdateHint 1-rec (1T): {} | Batch {batch_size} (1T / {threads}T): {} / {}",
        fmt_dur(hint_init_wall),
        fmt_bytes(client.hint_memory_bytes()),
        fmt_dur(upd_1_total / 50),
        fmt_dur(batch_1t),
        fmt_dur(batch_mt),
    );
    println!(
        "  Query avg (1T prep / srv / finish): {} / {} / {} | level hits: {:?}",
        fmt_dur(q_prep_total / q_count as u32),
        fmt_dur(q_srv_total / q_count as u32),
        fmt_dur(q_fin_total / q_count as u32),
        level_hits,
    );
}

fn ethereum_6level_params(record_size: usize, variant: BackendVariant) -> MultiLevelParams {
    let n: usize = 1_834_095_877;
    let q_budget: usize = 65_536;
    // Exact 6-level schedule: w = 32768 -> 8192 -> 2048 -> 512 -> 128 -> 32
    let specs: [(usize, usize, usize); 6] = [
        (32_768, 65_536, 3 * q_budget),
        (8_192, 32_768, 65_536),
        (2_048, 16_384, 8_192),
        (512, 8_192, 512),
        (128, 4_096, 256),
        (32, 2_048, 128),
    ];
    let levels: Vec<LevelSpec> = specs
        .iter()
        .enumerate()
        .map(|(i, &(w, m, q))| {
            let backend: LevelBackend = match variant {
                BackendVariant::HybridA { iprf_levels } if i < iprf_levels => LevelBackend::Iprf,
                _ => LevelBackend::FlatAes,
            };
            LevelSpec {
                block_size: w,
                num_blocks: n.div_ceil(w),
                num_primary: m,
                num_backup: q,
                backend,
            }
        })
        .collect();
    MultiLevelParams::custom(n, record_size, levels).unwrap()
}

fn ethereum_2level_params(record_size: usize, variant: BackendVariant) -> MultiLevelParams {
    let n: usize = 1_834_095_877;
    let q_budget: usize = 65_536;
    MultiLevelParams::two_level(n, record_size, 64, q_budget, variant).unwrap()
}

fn run_ethereum_scale(label: &str, params: MultiLevelParams, threads: usize) {
    print_params_summary(label, &params);
    let seed: [u8; 32] = [0x33; 32];
    let r: usize = params.record_size;
    let n: usize = params.num_entries;

    let mut client: MultiLevelClient = MultiLevelClient::new_zeroed(params.clone(), &seed);
    let mut rng: ChaCha20Rng = ChaCha20Rng::seed_from_u64(303);

    // 1. Direct real measurement of UpdateHint (single record + 3000-record Ethereum block).
    let mut upd_1_total: Duration = Duration::ZERO;
    for _ in 0..50 {
        let idx: usize = rng.gen_range(0..n);
        let mut delta: Vec<u8> = vec![0u8; r];
        rng.fill(&mut delta[..]);
        let tu: Instant = Instant::now();
        client.apply_delta(idx, &delta).unwrap();
        upd_1_total += tu.elapsed();
    }

    let eth_block_updates: usize = 3_000;
    let updates: Vec<(usize, Vec<u8>)> = (0..eth_block_updates)
        .map(|_| {
            let idx: usize = rng.gen_range(0..n);
            let mut d: Vec<u8> = vec![0u8; r];
            rng.fill(&mut d[..]);
            (idx, d)
        })
        .collect();

    let tb1: Instant = Instant::now();
    client.apply_deltas_batch(&updates, 1).unwrap();
    let batch_1t: Duration = tb1.elapsed();

    let tbm: Instant = Instant::now();
    client.apply_deltas_batch(&updates, threads).unwrap();
    let batch_mt: Duration = tbm.elapsed();

    // 2. Direct real measurement of Query prepare_query (first matching level on 5 queries).
    let mut q_total: Duration = Duration::ZERO;
    let mut q_levels: Vec<usize> = Vec::new();
    let q_trials: usize = 5;
    for _ in 0..q_trials {
        let idx: usize = rng.gen_range(0..n);
        let tq: Instant = Instant::now();
        let pending: PendingMultiLevelQuery = client.prepare_query(idx, &mut rng).unwrap();
        q_total += tq.elapsed();
        q_levels.push(pending.level + 1);
    }

    // 3. Direct per-block HintInit rate measurement on each level -> exact core-hours extrapolation.
    let mut total_core_secs: f64 = 0.0;
    let mut total_wall_mt_secs: f64 = 0.0;
    let max_sample_bytes: usize = 160 * 1024 * 1024;
    println!("  Measured HintInit per-level (sample blocks -> extrapolated to full 1.834B):");
    for (lvl_idx, spec) in params.levels.iter().enumerate() {
        let bytes_per_block: usize = spec.block_size.saturating_mul(r).max(1);
        let budget_blocks: usize = (max_sample_bytes / bytes_per_block).max(1);
        let sample_blocks: usize = match spec.backend {
            LevelBackend::Iprf => (threads * 2).min(budget_blocks).min(spec.num_blocks).max(1),
            LevelBackend::FlatAes => budget_blocks.min(32_768).min(spec.num_blocks).max(1),
        };
        let eff_threads: usize = threads.min(sample_blocks).max(1);
        let sample_db: Vec<u8> = make_synth_db(sample_blocks * spec.block_size, r);

        // Measure 1-thread (true core-time) on a small subset when Iprf, or 1/4 sample when FlatAes.
        let sample_1t: usize = match spec.backend {
            LevelBackend::Iprf => 2.min(sample_blocks).max(1),
            LevelBackend::FlatAes => (sample_blocks / 4).max(1).min(sample_blocks),
        };
        let dt_1t: f64 =
            MultiLevelClient::bench_level_blocks(&params, &seed, lvl_idx, &sample_db, sample_1t, 1)
                .as_secs_f64();
        let level_core_secs: f64 = dt_1t * (spec.num_blocks as f64 / sample_1t as f64);

        let dt_mt: f64 = MultiLevelClient::bench_level_blocks(
            &params,
            &seed,
            lvl_idx,
            &sample_db,
            sample_blocks,
            eff_threads,
        )
        .as_secs_f64();
        let level_wall_secs: f64 = dt_mt * (spec.num_blocks as f64 / sample_blocks as f64);

        total_core_secs += level_core_secs;
        total_wall_mt_secs += level_wall_secs;
        println!(
            "    L{} ({:<7}): sample {sample_1t}T1/{sample_blocks}T{eff_threads} blks -> {:>7.2} core-h | wall ({eff_threads}T): {:>6.1} s",
            lvl_idx + 1,
            match spec.backend {
                LevelBackend::Iprf => "Iprf",
                LevelBackend::FlatAes => "FlatAes",
            },
            level_core_secs / 3600.0,
            level_wall_secs,
        );
    }

    println!(
        "  TOTAL Ethereum-scale (N=1.834B): HintMem (est)={} | HintInit={:.2} core-h (wall <= {threads}T: {:.1} s / {:.1} min)",
        fmt_bytes(client.hint_memory_bytes()),
        total_core_secs / 3600.0,
        total_wall_mt_secs,
        total_wall_mt_secs / 60.0,
    );
    println!(
        "  UpdateHint 1-rec (1T): {} | Eth block (3000 rec, 1T / {threads}T): {} / {} | Query prep avg (1T, levels {:?}): {}",
        fmt_dur(upd_1_total / 50),
        fmt_dur(batch_1t),
        fmt_dur(batch_mt),
        q_levels,
        fmt_dur(q_total / q_trials as u32),
    );
}

fn main() {
    let args: Args = Args::parse();
    let threads: usize = args
        .threads
        .map(NonZeroUsize::get)
        .unwrap_or_else(available_threads);
    let r: usize = args.record_size.get();

    println!("Multi-level PIR Prototype Benchmark (threads={threads}, record_size={r} B)");

    match args.preset {
        Preset::Small | Preset::All => {
            let n: usize = 1 << 20; // 1,048,576 records (41.9 MB)
            let q: usize = 1024;
            println!("\n============================================================");
            println!(
                "PRESET SMALL: N = 2^20 ({n}) records, DB = {}",
                fmt_bytes(n * r)
            );
            println!("============================================================");
            let db: Vec<u8> = make_synth_db(n, r);

            run_plinko_baseline(&db, n, r, threads);

            let p_2a: MultiLevelParams = MultiLevelParams::two_level(
                n,
                r,
                64,
                q,
                BackendVariant::HybridA { iprf_levels: 1 },
            )
            .unwrap();
            run_multilevel_full(
                "2-level Variant A (Iprf L1 + FlatAes L2)",
                p_2a,
                &db,
                threads,
                1000,
            );

            let p_2b: MultiLevelParams =
                MultiLevelParams::two_level(n, r, 64, q, BackendVariant::FlatB).unwrap();
            run_multilevel_full(
                "2-level Variant B (FlatAes L1-L2)",
                p_2b,
                &db,
                threads,
                1000,
            );
        }
        _ => {}
    }

    match args.preset {
        Preset::Medium | Preset::All => {
            let n: usize = 1 << 24; // 16,777,216 records (671 MB)
            let q: usize = 4096;
            println!("\n============================================================");
            println!(
                "PRESET MEDIUM: N = 2^24 ({n}) records, DB = {}",
                fmt_bytes(n * r)
            );
            println!("============================================================");
            let db: Vec<u8> = make_synth_db(n, r);

            let p_2a: MultiLevelParams = MultiLevelParams::two_level(
                n,
                r,
                64,
                q,
                BackendVariant::HybridA { iprf_levels: 1 },
            )
            .unwrap();
            run_multilevel_full(
                "2-level Variant A (Iprf L1 + FlatAes L2)",
                p_2a,
                &db,
                threads,
                1000,
            );

            let p_2b: MultiLevelParams =
                MultiLevelParams::two_level(n, r, 64, q, BackendVariant::FlatB).unwrap();
            run_multilevel_full(
                "2-level Variant B (FlatAes L1-L2)",
                p_2b,
                &db,
                threads,
                1000,
            );

            let p_6a: MultiLevelParams =
                MultiLevelParams::waterfall(n, r, 6, q, BackendVariant::HybridA { iprf_levels: 2 })
                    .unwrap();
            run_multilevel_full(
                "6-level Variant A (Iprf L1-L2 + FlatAes L3-L6)",
                p_6a,
                &db,
                threads,
                1000,
            );

            let p_6b: MultiLevelParams =
                MultiLevelParams::waterfall(n, r, 6, q, BackendVariant::FlatB).unwrap();
            run_multilevel_full(
                "6-level Variant B (FlatAes L1-L6)",
                p_6b,
                &db,
                threads,
                1000,
            );
        }
        _ => {}
    }

    match args.preset {
        Preset::Ethereum | Preset::All => {
            println!("\n============================================================");
            println!(
                "PRESET ETHEREUM SCALE REFERENCE: N = 1,834,095,877 records (73.4 GB), Q = 65,536"
            );
            println!("============================================================");

            run_ethereum_scale(
                "2-level Variant A (Iprf L1 + FlatAes L2)",
                ethereum_2level_params(r, BackendVariant::HybridA { iprf_levels: 1 }),
                threads,
            );
            run_ethereum_scale(
                "2-level Variant B (FlatAes L1-L2)",
                ethereum_2level_params(r, BackendVariant::FlatB),
                threads,
            );
            run_ethereum_scale(
                "6-level Variant A (Iprf L1-L2 + FlatAes L3-L6)",
                ethereum_6level_params(r, BackendVariant::HybridA { iprf_levels: 2 }),
                threads,
            );
            run_ethereum_scale(
                "6-level Variant B (FlatAes L1-L6)",
                ethereum_6level_params(r, BackendVariant::FlatB),
                threads,
            );
        }
        _ => {}
    }
}
