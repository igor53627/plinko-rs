//! Standalone zero-import WebAssembly smoke test for `plinko::keyword` (2-segment `KV-SCF`)
//! covering `KvScfTable::build_with_seed_retry`, streaming 2-segment `MultiLevelHintBuilder`,
//! `KeywordClient<MultiLevelClient>` single/batch lookups, `CachePolicy::LocalHit`, live table
//! mutations + atomic `apply_epoch_update` (segment XOR deltas + runtime stash), `PLNKKEYW`
//! checkpoint serialization/corruption/budget/reload, `abort_lookup`, WAL `retire_hints`, and
//! `KeywordClient<protocol::Client>` round-trip with `PLNKKEYW` + `PLNKSTAT`.
//!
//! Build and run via `scripts/test_wasm_keyword.sh`.

use plinko::keyword::{
    BatchLookupPlan, BucketDelta, CachePolicy, KeyLocation, KeywordBatchAnswer, KeywordClient,
    KeywordClientError, KeywordEpochUpdate, KeywordLookupAnswer, KeywordLookupQuery,
    KeywordTableSpec, KvScfTable, LookupPlan, StashEntry, StashOp,
};
use plinko::multilevel::{
    BackendVariant, MultiLevelAnswer, MultiLevelClient, MultiLevelHintBuilder, MultiLevelParams,
    MultiLevelQuery, MultiLevelServer,
};
use plinko::protocol::{
    Client as ProtocolClient, ProtocolParams, Response as ProtocolResponse,
    Server as ProtocolServer,
};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

fn make_key(id: u64) -> [u8; 20] {
    let mut out = [0u8; 20];
    let b = id.to_le_bytes();
    out[0..8].copy_from_slice(&b);
    out[8..16].copy_from_slice(&id.wrapping_mul(0x9E37_79B9_7F4A_7C15).to_le_bytes());
    out[16..20].copy_from_slice(&(id as u32 ^ 0xA5A5_5A5A).to_le_bytes());
    out
}

fn make_val(id: u64, rev: u8) -> [u8; 24] {
    let mut out = [0u8; 24];
    let b = id.to_le_bytes();
    out[0..8].copy_from_slice(&b);
    out[8..16].copy_from_slice(&(!id).to_le_bytes());
    out[16..24].fill(rev);
    out
}

fn answer_single_ml(
    table: &KvScfTable,
    ml_params: &MultiLevelParams,
    kw: &KeywordClient<MultiLevelClient>,
    query: &KeywordLookupQuery<MultiLevelQuery>,
) -> Result<KeywordLookupAnswer<MultiLevelAnswer>, i32> {
    let seg0 = table.segment_bytes(0).ok_or(-20)?;
    let seg1 = table.segment_bytes(1).ok_or(-20)?;
    let a0 = MultiLevelServer::answer(
        seg0,
        ml_params,
        &query.segments[0],
        kw.backends()[0].epoch(),
    )
    .map_err(|_| -21)?;
    let a1 = MultiLevelServer::answer(
        seg1,
        ml_params,
        &query.segments[1],
        kw.backends()[1].epoch(),
    )
    .map_err(|_| -22)?;
    Ok(KeywordLookupAnswer {
        table_epoch: kw.spec().table_epoch(),
        state_epoch: kw.spec().state_epoch(),
        segments: [a0, a1],
    })
}

fn split_deltas(deltas: Vec<BucketDelta>) -> [Vec<(usize, Vec<u8>)>; 2] {
    let mut seg0: Vec<(usize, Vec<u8>)> = Vec::new();
    let mut seg1: Vec<(usize, Vec<u8>)> = Vec::new();
    for d in deltas {
        if d.segment == 0 {
            seg0.push((d.bucket_idx, d.xor_delta));
        } else {
            seg1.push((d.bucket_idx, d.xor_delta));
        }
    }
    [seg0, seg1]
}

fn run_multilevel_keyword_smoke() -> Result<(), i32> {
    let num_keys: usize = 8_192;
    let domain: u8 = 0x01;
    let keys: Vec<[u8; 20]> = (0..num_keys as u64).map(make_key).collect();
    let vals: Vec<[u8; 24]> = (0..num_keys as u64).map(|i| make_val(i, 1)).collect();
    let entries: Vec<(u8, &[u8], &[u8])> = (0..num_keys)
        .map(|i| (domain, keys[i].as_slice(), vals[i].as_slice()))
        .collect();

    let (mut table, _stats) =
        KvScfTable::build_with_seed_retry(&entries, Some(20), 24, 8, 4, 0.85, [0x42; 32], 4)
            .map_err(|_| -2)?;
    let spec: KeywordTableSpec = table.spec().clone();

    let ml_params: MultiLevelParams = MultiLevelParams::two_level(
        spec.buckets_per_segment(),
        spec.record_size(),
        16,
        64,
        BackendVariant::FlatB,
    )
    .map_err(|_| -3)?;
    let seg_seeds: [[u8; 32]; 2] = [[0x11; 32], [0x22; 32]];

    // 1. Streaming 2-segment HintInit with unaligned chunk size inside zero-import WASM.
    let mut clients: Vec<MultiLevelClient> = Vec::with_capacity(2);
    for (seg, seed) in seg_seeds.iter().enumerate() {
        let seg_bytes: &[u8] = table.segment_bytes(seg).ok_or(-4)?;
        let mut builder: MultiLevelHintBuilder =
            MultiLevelHintBuilder::with_chunk_bytes(ml_params.clone(), seed, 1, 8192)
                .map_err(|_| -4)?;
        for chunk in seg_bytes.chunks(3_001) {
            builder.push_bytes(chunk).map_err(|_| -4)?;
        }
        clients.push(builder.finish().map_err(|_| -4)?);
    }
    let [c0, c1]: [MultiLevelClient; 2] = clients.try_into().map_err(|_| -4)?;
    let mut kw: KeywordClient<MultiLevelClient> =
        KeywordClient::new(spec.clone(), [c0, c1], CachePolicy::LocalHit).map_err(|_| -5)?;
    let mut rng: ChaCha20Rng = ChaCha20Rng::seed_from_u64(0x2026_2082);

    // 2. Single hit, single miss, and cached repeat hit.
    let LookupPlan::Query { query, pending } = kw
        .prepare_lookup(domain, &keys[0], &mut rng)
        .map_err(|_| -6)?
    else {
        return Err(-6);
    };
    let ans = answer_single_ml(&table, &ml_params, &kw, &query)?;
    let got = kw.finish_lookup(&pending, &ans, &mut rng).map_err(|_| -7)?;
    if got.as_deref() != Some(vals[0].as_slice()) {
        return Err(-8);
    }

    let miss_key: [u8; 20] = make_key(999_999);
    let LookupPlan::Query {
        query: mq,
        pending: mp,
    } = kw
        .prepare_lookup(domain, &miss_key, &mut rng)
        .map_err(|_| -9)?
    else {
        return Err(-9);
    };
    let mans = answer_single_ml(&table, &ml_params, &kw, &mq)?;
    let mgot = kw.finish_lookup(&mp, &mans, &mut rng).map_err(|_| -10)?;
    if mgot.is_some() {
        return Err(-11);
    }

    // Repeat lookup of keys[0] must hit local bucket cache with zero network queries.
    match kw
        .prepare_lookup(domain, &keys[0], &mut rng)
        .map_err(|_| -12)?
    {
        LookupPlan::Local(Some(v)) if v == vals[0] => {}
        _ => return Err(-13),
    }

    // 3. 1-RTT batch lookup mixing cached hit, uncached hit, and miss.
    let miss_key2: [u8; 20] = make_key(888_888);
    let batch_req: [(u8, &[u8]); 3] = [
        (domain, keys[0].as_slice()),
        (domain, keys[7].as_slice()),
        (domain, miss_key2.as_slice()),
    ];
    let BatchLookupPlan::Query {
        query: bq,
        pending: bp,
    } = kw.prepare_batch(&batch_req, &mut rng).map_err(|_| -14)?
    else {
        return Err(-14);
    };
    let seg0_bytes: &[u8] = table.segment_bytes(0).ok_or(-15)?;
    let seg1_bytes: &[u8] = table.segment_bytes(1).ok_or(-15)?;
    let mut b_a0: Vec<MultiLevelAnswer> = Vec::with_capacity(bq.segments[0].len());
    for q in &bq.segments[0] {
        b_a0.push(
            MultiLevelServer::answer(seg0_bytes, &ml_params, q, kw.backends()[0].epoch())
                .map_err(|_| -15)?,
        );
    }
    let mut b_a1: Vec<MultiLevelAnswer> = Vec::with_capacity(bq.segments[1].len());
    for q in &bq.segments[1] {
        b_a1.push(
            MultiLevelServer::answer(seg1_bytes, &ml_params, q, kw.backends()[1].epoch())
                .map_err(|_| -15)?,
        );
    }
    let bans = KeywordBatchAnswer {
        table_epoch: kw.spec().table_epoch(),
        state_epoch: kw.spec().state_epoch(),
        segments: [b_a0, b_a1],
    };
    let bres = kw.finish_batch(&bp, &bans, &mut rng).map_err(|_| -16)?;
    if bres.len() != 3
        || bres[0].as_deref() != Some(vals[0].as_slice())
        || bres[1].as_deref() != Some(vals[7].as_slice())
        || bres[2].is_some()
    {
        return Err(-17);
    }

    // 4. Live table mutations (update + delete + insert) + runtime stash via apply_epoch_update.
    let updated_val0: [u8; 24] = make_val(0, 9);
    let inserted_key: [u8; 20] = make_key(100_000);
    let inserted_val: [u8; 24] = make_val(100_000, 3);
    let stash_key: [u8; 20] = make_key(200_000);
    let stash_val: [u8; 24] = make_val(200_000, 5);
    let stash_loc: KeyLocation = spec.locate(domain, &stash_key).map_err(|_| -18)?;

    let mut all_deltas: Vec<BucketDelta> = Vec::new();
    let out_upd = table
        .update(domain, &keys[0], &updated_val0)
        .map_err(|_| -18)?;
    all_deltas.extend(out_upd.deltas);
    let out_del = table.delete(domain, &keys[7]).map_err(|_| -18)?;
    all_deltas.extend(out_del.deltas);
    let out_ins = table
        .insert(domain, &inserted_key, &inserted_val)
        .map_err(|_| -18)?;
    all_deltas.extend(out_ins.deltas);

    let update = KeywordEpochUpdate {
        table_epoch: spec.table_epoch(),
        from_state_epoch: 0,
        to_state_epoch: 1,
        segment_deltas: split_deltas(all_deltas),
        stash_ops: vec![StashOp::Insert(StashEntry {
            bucket_seg0: stash_loc.bucket_seg0,
            tag: stash_loc.tag,
            value: stash_val.to_vec(),
        })],
    };
    kw.apply_epoch_update(&update, 1).map_err(|_| -19)?;

    // Verify updated key, deleted key, newly inserted key, and runtime-stash key via PIR.
    let checks: [(&[u8], Option<&[u8]>); 4] = [
        (&keys[0], Some(updated_val0.as_slice())),
        (&keys[7], None),
        (&inserted_key, Some(inserted_val.as_slice())),
        (&stash_key, Some(stash_val.as_slice())),
    ];
    for (k, expected) in checks {
        let got_val: Option<Vec<u8>> =
            match kw.prepare_lookup(domain, k, &mut rng).map_err(|_| -23)? {
                LookupPlan::Local(v) => v,
                LookupPlan::Query { query, pending } => {
                    let ans = answer_single_ml(&table, &ml_params, &kw, &query)?;
                    kw.finish_lookup(&pending, &ans, &mut rng)
                        .map_err(|_| -24)?
                }
            };
        if got_val.as_deref() != expected {
            return Err(-25);
        }
    }

    // 5. PLNKKEYW checkpoint serialization, corruption rejection, budget cap, and reload.
    let ckpt: Vec<u8> = kw.to_bytes().map_err(|_| -26)?;
    let mut bad_ckpt: Vec<u8> = ckpt.clone();
    bad_ckpt[ckpt.len() / 2] ^= 0x01;
    if !matches!(
        KeywordClient::<MultiLevelClient>::from_bytes(&bad_ckpt, &seg_seeds),
        Err(KeywordClientError::Checkpoint(_))
    ) {
        return Err(-27);
    }
    if KeywordClient::<MultiLevelClient>::read_checkpoint_with_budget(
        ckpt.as_slice(),
        &seg_seeds,
        64,
    )
    .is_ok()
    {
        return Err(-28);
    }

    // 6. Abort + WAL retire_hints on reloaded checkpoint.
    let LookupPlan::Query {
        query: _aq,
        mut pending,
    } = kw
        .prepare_lookup(domain, &keys[42], &mut rng)
        .map_err(|_| -29)?
    else {
        return Err(-29);
    };
    let [h0, h1]: [(usize, u32); 2] = pending.hint_ids();
    kw.abort_lookup(&mut pending).map_err(|_| -30)?;
    if kw.queries_in_flight() != 0 {
        return Err(-31);
    }

    let mut reloaded: KeywordClient<MultiLevelClient> =
        KeywordClient::<MultiLevelClient>::from_bytes(&ckpt, &seg_seeds).map_err(|_| -32)?;
    if reloaded.spec().state_epoch() != 1 || reloaded.stash_len() != 1 {
        return Err(-33);
    }
    reloaded.retire_hints(&[h0], &[h1]).map_err(|_| -34)?;

    let LookupPlan::Query {
        query: rq,
        pending: rp,
    } = reloaded
        .prepare_lookup(domain, &keys[100], &mut rng)
        .map_err(|_| -35)?
    else {
        return Err(-35);
    };
    let rans = answer_single_ml(&table, &ml_params, &reloaded, &rq)?;
    let rgot = reloaded
        .finish_lookup(&rp, &rans, &mut rng)
        .map_err(|_| -36)?;
    if rgot.as_deref() != Some(vals[100].as_slice()) {
        return Err(-37);
    }

    Ok(())
}

fn run_protocol_keyword_smoke() -> Result<(), i32> {
    let num_keys: usize = 16;
    let domain: u8 = 0x01;
    let keys: Vec<[u8; 20]> = (0..num_keys as u64).map(|i| make_key(i + 500)).collect();
    let vals: Vec<[u8; 24]> = (0..num_keys as u64).map(|i| make_val(i + 500, 7)).collect();
    let entries: Vec<(u8, &[u8], &[u8])> = (0..num_keys)
        .map(|i| (domain, keys[i].as_slice(), vals[i].as_slice()))
        .collect();

    let (table, _) =
        KvScfTable::build_with_seed_retry(&entries, Some(20), 24, 8, 2, 0.80, [0x77; 32], 4)
            .map_err(|_| -41)?;
    let spec: KeywordTableSpec = table.spec().clone();

    let params: ProtocolParams = ProtocolParams::new(
        spec.buckets_per_segment(),
        spec.record_size(),
        None,
        4,
        None,
    )
    .map_err(|_| -42)?;
    let seg_seeds: [[u8; 32]; 2] = [[0x33; 32], [0x44; 32]];
    let query_seeds: [[u8; 32]; 2] = [[0x55; 32], [0x66; 32]];

    let s0 = ProtocolServer::new(params.clone(), 0, table.segment_bytes(0).ok_or(-43)?)
        .map_err(|_| -43)?;
    let s1 = ProtocolServer::new(params.clone(), 0, table.segment_bytes(1).ok_or(-43)?)
        .map_err(|_| -43)?;
    let c0 = ProtocolClient::offline_init(
        params.clone(),
        seg_seeds[0],
        query_seeds[0],
        0,
        table.segment_bytes(0).ok_or(-44)?,
    )
    .map_err(|_| -44)?;
    let c1 = ProtocolClient::offline_init(
        params,
        seg_seeds[1],
        query_seeds[1],
        0,
        table.segment_bytes(1).ok_or(-44)?,
    )
    .map_err(|_| -44)?;

    let mut kw: KeywordClient<ProtocolClient> =
        KeywordClient::new(spec, [c0, c1], CachePolicy::ConstantTraffic).map_err(|_| -45)?;
    let mut rng: ChaCha20Rng = ChaCha20Rng::seed_from_u64(0x9090);

    let LookupPlan::Query { query, pending } = kw
        .prepare_lookup(domain, &keys[3], &mut rng)
        .map_err(|_| -46)?
    else {
        return Err(-46);
    };
    let a0: ProtocolResponse = s0.answer(&query.segments[0]).map_err(|_| -47)?;
    let a1: ProtocolResponse = s1.answer(&query.segments[1]).map_err(|_| -47)?;
    let ans = KeywordLookupAnswer {
        table_epoch: kw.spec().table_epoch(),
        state_epoch: kw.spec().state_epoch(),
        segments: [a0, a1],
    };
    let got = kw
        .finish_lookup(&pending, &ans, &mut rng)
        .map_err(|_| -48)?;
    if got.as_deref() != Some(vals[3].as_slice()) {
        return Err(-49);
    }

    // Round-trip PLNKKEYW + PLNKSTAT checkpoint with fresh query seeds and query again.
    let ckpt: Vec<u8> = kw.to_bytes().map_err(|_| -50)?;
    let fresh_qseeds: [[u8; 32]; 2] = [[0xA1; 32], [0xB2; 32]];
    let mut reloaded: KeywordClient<ProtocolClient> =
        KeywordClient::<ProtocolClient>::from_bytes(&ckpt, &seg_seeds, &fresh_qseeds)
            .map_err(|_| -51)?;

    let LookupPlan::Query {
        query: q2,
        pending: p2,
    } = reloaded
        .prepare_lookup(domain, &keys[5], &mut rng)
        .map_err(|_| -52)?
    else {
        return Err(-52);
    };
    let a20: ProtocolResponse = s0.answer(&q2.segments[0]).map_err(|_| -53)?;
    let a21: ProtocolResponse = s1.answer(&q2.segments[1]).map_err(|_| -53)?;
    let ans2 = KeywordLookupAnswer {
        table_epoch: reloaded.spec().table_epoch(),
        state_epoch: reloaded.spec().state_epoch(),
        segments: [a20, a21],
    };
    let got2 = reloaded
        .finish_lookup(&p2, &ans2, &mut rng)
        .map_err(|_| -54)?;
    if got2.as_deref() != Some(vals[5].as_slice()) {
        return Err(-55);
    }

    Ok(())
}

/// Runs the full `plinko::keyword` 2-segment `KV-SCF` smoke suite (both `MultiLevelClient`
/// and `protocol::Client` backends) inside zero-import WebAssembly.
///
/// Returns `0` on success, or a negative step code on failure.
#[no_mangle]
pub extern "C" fn run_keyword_kvscf_smoke() -> i32 {
    if let Err(code) = run_multilevel_keyword_smoke() {
        return code;
    }
    if let Err(code) = run_protocol_keyword_smoke() {
        return code;
    }
    0
}
