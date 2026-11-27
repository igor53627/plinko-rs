//! Benchmark for `plinko::keyword` (`KvScfTable`, `database.bin` builder, live
//! mutations, and 2-segment `plinko::multilevel` FlatB end-to-end lookup).

use clap::Parser;
use plinko::keyword::{
    BucketDelta, BuildStats, KeywordTableSpec, KvScfTable, DEFAULT_BUCKET_SIZE, DEFAULT_TAG_SIZE,
};
use plinko::multilevel::{BackendVariant, MultiLevelClient, MultiLevelParams, MultiLevelServer};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use std::path::PathBuf;
use std::time::{Duration, Instant};

#[derive(Parser, Debug)]
#[command(
    about = "Benchmark KvScfTable build, mutations, database.bin export, and 2-segment multilevel PIR"
)]
struct Args {
    /// Number of key-value pairs to build (default: 1,000,000).
    #[arg(long, default_value_t = 1_000_000)]
    num_keys: usize,

    /// Fixed key size in bytes (default: 20).
    #[arg(long, default_value_t = 20)]
    key_size: usize,

    /// Fixed value size in bytes (default: 32; 24 for compact EOA).
    #[arg(long, default_value_t = 32)]
    value_size: usize,

    /// Tag size in bytes (default: 8).
    #[arg(long, default_value_t = DEFAULT_TAG_SIZE)]
    tag_size: usize,

    /// Slots per bucket (default: 4).
    #[arg(long, default_value_t = DEFAULT_BUCKET_SIZE)]
    bucket_size: usize,

    /// Target build load factor (default: 0.90; use 0.85 for live insert headroom).
    #[arg(long, default_value_t = 0.90)]
    load: f64,

    /// Number of live updates, inserts, and deletes each to benchmark after build.
    #[arg(long, default_value_t = 1_000)]
    mutations: usize,

    /// Number of plaintext lookups to benchmark (half hits, half misses).
    #[arg(long, default_value_t = 100_000)]
    lookups: usize,

    /// Security parameter kappa for 2-segment multilevel PIR check.
    #[arg(long, default_value_t = 64)]
    kappa: usize,

    /// Query budget Q for 2-segment multilevel PIR check.
    #[arg(long, default_value_t = 1_024)]
    query_budget: usize,

    /// Worker threads for multilevel HintInit (default: available parallelism).
    #[arg(long)]
    threads: Option<usize>,

    /// Optional output path to benchmark writing `database.bin` to disk.
    #[arg(long)]
    out_db: Option<PathBuf>,

    /// Skip the 2-segment `plinko::multilevel` HintInit + online PIR check.
    #[arg(long, default_value_t = false)]
    skip_pir: bool,
}

fn fmt_bytes(bytes: usize) -> String {
    const KIB: f64 = 1024.0;
    const MIB: f64 = 1024.0 * 1024.0;
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    let b: f64 = bytes as f64;
    if b >= GIB {
        format!("{:.2} GiB", b / GIB)
    } else if b >= MIB {
        format!("{:.2} MiB", b / MIB)
    } else if b >= KIB {
        format!("{:.2} KiB", b / KIB)
    } else {
        format!("{bytes} B")
    }
}

fn synth_key(id: u64, key_size: usize, out: &mut [u8]) {
    let mut h = blake3::Hasher::new();
    h.update(b"bench-kw-key");
    h.update(&id.to_le_bytes());
    h.finalize_xof().fill(&mut out[..key_size]);
}

fn synth_val(id: u64, value_size: usize, out: &mut [u8]) {
    let mut h = blake3::Hasher::new();
    h.update(b"bench-kw-val");
    h.update(&id.to_le_bytes());
    h.finalize_xof().fill(&mut out[..value_size]);
}

fn apply_deltas_to_mirror(mirror: &mut [u8], spec: &KeywordTableSpec, deltas: &[BucketDelta]) {
    let rec: usize = spec.record_size();
    for d in deltas {
        let off: usize = d.global_index(spec) * rec;
        for i in 0..rec {
            mirror[off + i] ^= d.xor_delta[i];
        }
    }
}

fn main() {
    let args: Args = Args::parse();
    let threads: usize = args.threads.unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    });

    println!("=== plinko::keyword KV-SCF Benchmark ===");
    println!(
        "keys={}, key_size={} B, value_size={} B, tag_size={} B, b={}, target_load={:.2}",
        args.num_keys, args.key_size, args.value_size, args.tag_size, args.bucket_size, args.load
    );

    // Pre-generate deterministic synthetic keys and values in contiguous buffers.
    let t_gen: Instant = Instant::now();
    let mut keys_flat: Vec<u8> = vec![0u8; args.num_keys * args.key_size];
    let mut vals_flat: Vec<u8> = vec![0u8; args.num_keys * args.value_size];
    for i in 0..args.num_keys {
        synth_key(
            i as u64,
            args.key_size,
            &mut keys_flat[i * args.key_size..(i + 1) * args.key_size],
        );
        synth_val(
            i as u64,
            args.value_size,
            &mut vals_flat[i * args.value_size..(i + 1) * args.value_size],
        );
    }
    println!(
        "synthetic data ready in {:.2} ms",
        t_gen.elapsed().as_secs_f64() * 1e3
    );

    let spec: KeywordTableSpec = KeywordTableSpec::with_layout(
        args.num_keys,
        Some(args.key_size),
        args.value_size,
        args.tag_size,
        args.bucket_size,
        args.load,
        [0x42; 32],
    )
    .expect("valid KeywordTableSpec");

    println!(
        "geometry: s={} buckets/seg ({} total), record_size={} B, DB={} ({:.2e} false-match bound)",
        spec.buckets_per_segment(),
        spec.total_buckets(),
        spec.record_size(),
        fmt_bytes(spec.database_bytes()),
        spec.false_match_upper_bound()
    );

    let entries = (0..args.num_keys).map(|i| {
        (
            0x01u8,
            &keys_flat[i * args.key_size..(i + 1) * args.key_size],
            &vals_flat[i * args.value_size..(i + 1) * args.value_size],
        )
    });

    let t_build: Instant = Instant::now();
    let (mut table, stats): (KvScfTable, BuildStats) =
        KvScfTable::build(spec.clone(), entries).expect("build succeeded without stash");
    let build_elapsed: Duration = t_build.elapsed();
    let build_mkeys_s: f64 = args.num_keys as f64 / build_elapsed.as_secs_f64() / 1e6;

    println!(
        "build: {:.2} ms ({:.2} Mkeys/s), load={:.4}, total_kicks={}, max_kicks={}, tail(1%) mean={:.2} p99={} max={}",
        build_elapsed.as_secs_f64() * 1e3,
        build_mkeys_s,
        stats.load_factor,
        stats.total_kicks,
        stats.max_kicks_observed,
        stats.tail_mean_kicks,
        stats.tail_p99_kicks,
        stats.tail_max_kicks
    );

    // Optional disk export timing.
    if let Some(ref out_path) = args.out_db {
        let t_wr: Instant = Instant::now();
        table
            .write_database_file(out_path)
            .expect("write database.bin");
        let wr_s: f64 = t_wr.elapsed().as_secs_f64();
        let gb_s: f64 = spec.database_bytes() as f64 / wr_s / 1e9;
        println!(
            "write database.bin ({}): {:.2} ms ({:.2} GB/s)",
            out_path.display(),
            wr_s * 1e3,
            gb_s
        );
    }

    // Plaintext lookup benchmark (half hits, half misses; miss keys pre-generated
    // outside timer and probed under distinct kind 0x02 so even key_size=0 never collides).
    let hits: usize = (args.lookups / 2).min(args.num_keys);
    let misses: usize = args.lookups - hits;
    let mut miss_keys_flat: Vec<u8> = vec![0u8; misses * args.key_size];
    for i in 0..misses {
        synth_key(
            (args.num_keys + i + 1) as u64,
            args.key_size,
            &mut miss_keys_flat[i * args.key_size..(i + 1) * args.key_size],
        );
    }
    let t_lk: Instant = Instant::now();
    for i in 0..hits {
        let idx: usize = (i * 997) % args.num_keys;
        let k: &[u8] = &keys_flat[idx * args.key_size..(idx + 1) * args.key_size];
        let expected: &[u8] = &vals_flat[idx * args.value_size..(idx + 1) * args.value_size];
        let got: Option<&[u8]> = table.get(0x01, k).expect("lookup hit");
        assert_eq!(got, Some(expected));
    }
    let mut false_matches: usize = 0;
    for i in 0..misses {
        let mk: &[u8] = &miss_keys_flat[i * args.key_size..(i + 1) * args.key_size];
        if table.get(0x02, mk).expect("lookup miss").is_some() {
            false_matches += 1;
        }
    }
    let lk_elapsed: Duration = t_lk.elapsed();
    let lk_mops: f64 = (hits + misses) as f64 / lk_elapsed.as_secs_f64() / 1e6;
    println!(
        "plaintext lookup ({} hits + {} misses): {:.2} ms ({:.2} Mops/s), false_matches={}",
        hits,
        misses,
        lk_elapsed.as_secs_f64() * 1e3,
        lk_mops,
        false_matches
    );

    // Live mutations benchmark with exact XOR delta mirror check.
    let mut mirror: Vec<u8> = table.as_bytes().to_vec();
    let mut_count: usize = args.mutations.min(args.num_keys);
    let mut tmp_val: Vec<u8> = vec![0u8; args.value_size];
    let mut tmp_key: Vec<u8> = vec![0u8; args.key_size];

    // 1) Updates
    let t_upd: Instant = Instant::now();
    let mut upd_deltas: usize = 0;
    for i in 0..mut_count {
        let k: &[u8] = &keys_flat[i * args.key_size..(i + 1) * args.key_size];
        synth_val(
            (args.num_keys + 1_000_000 + i) as u64,
            args.value_size,
            &mut tmp_val,
        );
        let out = table.update(0x01, k, &tmp_val).expect("update existing");
        upd_deltas += out.deltas.len();
        apply_deltas_to_mirror(&mut mirror, &spec, &out.deltas);
    }
    let upd_ms: f64 = t_upd.elapsed().as_secs_f64() * 1e3;

    // 2) Deletes (cap starting offset at num_keys - mut_count so full delete range fits)
    let del_offset: usize = (args.num_keys - mut_count).min(mut_count);
    let t_del: Instant = Instant::now();
    let mut del_deltas: usize = 0;
    let mut deleted_count: usize = 0;
    for i in 0..mut_count {
        let idx: usize = del_offset + i;
        let k: &[u8] = &keys_flat[idx * args.key_size..(idx + 1) * args.key_size];
        let out = table.delete(0x01, k).expect("delete existing");
        del_deltas += out.deltas.len();
        apply_deltas_to_mirror(&mut mirror, &spec, &out.deltas);
        deleted_count += 1;
    }
    let del_ms: f64 = t_del.elapsed().as_secs_f64() * 1e3;

    // 3) Live inserts at final load
    let t_ins: Instant = Instant::now();
    let mut ins_deltas: usize = 0;
    let mut ins_kicks: usize = 0;
    let mut ins_max_deltas: usize = 0;
    let mut inserted_new: usize = 0;
    for i in 0..mut_count {
        synth_key(
            (args.num_keys + 2_000_000 + i) as u64,
            args.key_size,
            &mut tmp_key,
        );
        synth_val(
            (args.num_keys + 2_000_000 + i) as u64,
            args.value_size,
            &mut tmp_val,
        );
        let out = table.insert(0x01, &tmp_key, &tmp_val).expect("live insert");
        ins_deltas += out.deltas.len();
        ins_kicks += out.kicks;
        ins_max_deltas = ins_max_deltas.max(out.deltas.len());
        apply_deltas_to_mirror(&mut mirror, &spec, &out.deltas);
        inserted_new += 1;
    }
    let ins_ms: f64 = t_ins.elapsed().as_secs_f64() * 1e3;

    assert_eq!(
        mirror.as_slice(),
        table.as_bytes(),
        "XOR delta mirror diverged from KvScfTable"
    );
    println!(
        "mutations ({} each): update {:.2} ms ({:.2} deltas/op), delete {:.2} ms ({:.2} deltas/op), insert {:.2} ms ({:.2} deltas/op, max {}, mean {:.2} kicks)",
        mut_count,
        upd_ms,
        upd_deltas as f64 / mut_count.max(1) as f64,
        del_ms,
        del_deltas as f64 / deleted_count.max(1) as f64,
        ins_ms,
        ins_deltas as f64 / inserted_new.max(1) as f64,
        ins_max_deltas,
        ins_kicks as f64 / inserted_new.max(1) as f64
    );

    if args.skip_pir {
        return;
    }

    // 2-segment plinko::multilevel FlatB benchmark on the built 1M-key table.
    let s: usize = spec.buckets_per_segment();
    let rec: usize = spec.record_size();
    let ml_params = match MultiLevelParams::two_level(
        s,
        rec,
        args.kappa,
        args.query_budget,
        BackendVariant::FlatB,
    ) {
        Ok(p) => p,
        Err(err) => {
            println!("skipping 2-segment multilevel PIR check (params rejected: {err})");
            return;
        }
    };

    let t_hi: Instant = Instant::now();
    let mut c0 = MultiLevelClient::hint_init(
        ml_params.clone(),
        &[0x11; 32],
        table.segment_bytes(0).unwrap(),
        threads,
    )
    .expect("seg 0 HintInit");
    let mut c1 = MultiLevelClient::hint_init(
        ml_params.clone(),
        &[0x22; 32],
        table.segment_bytes(1).unwrap(),
        threads,
    )
    .expect("seg 1 HintInit");
    let hi_ms: f64 = t_hi.elapsed().as_secs_f64() * 1e3;

    let total_hints_bytes: usize = c0.hint_memory_bytes() + c1.hint_memory_bytes();
    let l0_upload: usize = 2 * ml_params.levels[0].query_upload_bytes();
    println!(
        "2-segment multilevel FlatB (kappa={}, Q={}, {}T): HintInit={:.2} ms, hints={} ({}/seg), L0 upload={} (2 segs)",
        args.kappa,
        args.query_budget,
        threads,
        hi_ms,
        fmt_bytes(total_hints_bytes),
        fmt_bytes(c0.hint_memory_bytes()),
        fmt_bytes(l0_upload),
    );

    // Reserve 1 query for the post-update verification so total queries stay within query_budget.
    let live_check: usize = 100.min(inserted_new);
    let post_check_queries: usize = usize::from(live_check > 0 && args.query_budget > 0);
    let pir_queries: usize = 100.min(args.query_budget.saturating_sub(post_check_queries));
    let mut rng = ChaCha20Rng::seed_from_u64(1337);
    let mut prep_total: Duration = Duration::ZERO;
    let mut ans_total: Duration = Duration::ZERO;
    let mut fin_total: Duration = Duration::ZERO;
    for i in 0..pir_queries {
        let (kind, key_slice, expected): (u8, Vec<u8>, Option<Vec<u8>>) = if i % 2 == 0
            && (inserted_new > 0 || args.num_keys > 0)
        {
            if inserted_new > 0 {
                let mut k: Vec<u8> = vec![0u8; args.key_size];
                let mut v: Vec<u8> = vec![0u8; args.value_size];
                let id: u64 = (args.num_keys + 2_000_000 + (i / 2) % inserted_new) as u64;
                synth_key(id, args.key_size, &mut k);
                synth_val(id, args.value_size, &mut v);
                (0x01, k, Some(v))
            } else {
                let idx: usize = (i / 2) % args.num_keys;
                let k: Vec<u8> = keys_flat[idx * args.key_size..(idx + 1) * args.key_size].to_vec();
                let v: Vec<u8> =
                    vals_flat[idx * args.value_size..(idx + 1) * args.value_size].to_vec();
                (0x01, k, Some(v))
            }
        } else {
            let mut k: Vec<u8> = vec![0u8; args.key_size];
            synth_key(
                (args.num_keys + 9_000_000 + i) as u64,
                args.key_size,
                &mut k,
            );
            (0x02, k, None)
        };

        let loc = spec.locate(kind, &key_slice).unwrap();
        let tp = Instant::now();
        let p0 = c0.prepare_query(loc.bucket_seg0, &mut rng).unwrap();
        let p1 = c1.prepare_query(loc.bucket_seg1, &mut rng).unwrap();
        prep_total += tp.elapsed();

        let ta = Instant::now();
        let a0 = MultiLevelServer::answer(
            table.segment_bytes(0).unwrap(),
            &ml_params,
            &p0.query,
            c0.epoch(),
        )
        .unwrap();
        let a1 = MultiLevelServer::answer(
            table.segment_bytes(1).unwrap(),
            &ml_params,
            &p1.query,
            c1.epoch(),
        )
        .unwrap();
        ans_total += ta.elapsed();

        let tf = Instant::now();
        let b0 = c0.finish_query(&p0, &a0, &mut rng).unwrap();
        let b1 = c1.finish_query(&p1, &a1, &mut rng).unwrap();
        let got = spec
            .scan_buckets(&b0, &b1, &loc.tag, None)
            .unwrap()
            .map(|v| v.to_vec());
        fin_total += tf.elapsed();

        assert_eq!(got, expected);
    }

    // Apply live updates through `apply_deltas_batch` and verify via PIR.
    let mut seg0_deltas: Vec<(usize, Vec<u8>)> = Vec::new();
    let mut seg1_deltas: Vec<(usize, Vec<u8>)> = Vec::new();
    for i in 0..live_check {
        synth_key(
            (args.num_keys + 2_000_000 + i) as u64,
            args.key_size,
            &mut tmp_key,
        );
        synth_val(
            (args.num_keys + 5_000_000 + i) as u64,
            args.value_size,
            &mut tmp_val,
        );
        let out = table.update(0x01, &tmp_key, &tmp_val).unwrap();
        for d in out.deltas {
            if d.segment == 0 {
                seg0_deltas.push((d.bucket_idx, d.xor_delta));
            } else {
                seg1_deltas.push((d.bucket_idx, d.xor_delta));
            }
        }
    }
    let t_apply = Instant::now();
    if !seg0_deltas.is_empty() {
        c0.apply_deltas_batch(&seg0_deltas, threads).unwrap();
    }
    if !seg1_deltas.is_empty() {
        c1.apply_deltas_batch(&seg1_deltas, threads).unwrap();
    }
    let apply_ms: f64 = t_apply.elapsed().as_secs_f64() * 1e3;

    // Verify first updated key via PIR after delta sync.
    if post_check_queries > 0 {
        synth_key(
            (args.num_keys + 2_000_000) as u64,
            args.key_size,
            &mut tmp_key,
        );
        synth_val(
            (args.num_keys + 5_000_000) as u64,
            args.value_size,
            &mut tmp_val,
        );
        let loc = spec.locate(0x01, &tmp_key).unwrap();
        let p0 = c0.prepare_query(loc.bucket_seg0, &mut rng).unwrap();
        let p1 = c1.prepare_query(loc.bucket_seg1, &mut rng).unwrap();
        let a0 = MultiLevelServer::answer(
            table.segment_bytes(0).unwrap(),
            &ml_params,
            &p0.query,
            c0.epoch(),
        )
        .unwrap();
        let a1 = MultiLevelServer::answer(
            table.segment_bytes(1).unwrap(),
            &ml_params,
            &p1.query,
            c1.epoch(),
        )
        .unwrap();
        let b0 = c0.finish_query(&p0, &a0, &mut rng).unwrap();
        let b1 = c1.finish_query(&p1, &a1, &mut rng).unwrap();
        let got = spec.scan_buckets(&b0, &b1, &loc.tag, None).unwrap();
        assert_eq!(got, Some(tmp_val.as_slice()));
    }

    let denom: f64 = pir_queries.max(1) as f64;
    println!(
        "2-segment PIR online ({} lookups): prep={:.1} us, server={:.1} us, finish={:.1} us | apply {} deltas={:.2} ms",
        pir_queries,
        prep_total.as_secs_f64() * 1e6 / denom,
        ans_total.as_secs_f64() * 1e6 / denom,
        fin_total.as_secs_f64() * 1e6 / denom,
        seg0_deltas.len() + seg1_deltas.len(),
        apply_ms,
    );
}
