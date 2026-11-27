//! End-to-end tests for `plinko::keyword` through the `plinko_hints` binary and
//! `plinko::protocol::Client`.
//!
//! Covers both documented `database.bin` execution modes (`docs/keyword_lookup.md`):
//! 1. Single combined `database.bin` (`2 * s` buckets) -> `plinko_hints` -> one
//!    `protocol::Client` querying `a0` and `s + a1`.
//! 2. Disjoint 2-segment files (`segment_0.bin`, `segment_1.bin`) -> `plinko_hints`
//!    -> `KeywordClient<protocol::Client>` + `KeywordSession` with batch lookups,
//!    live mutations (`insert`/`update`/`delete`), checkpoints, and WAL `retire_hints`.

use std::fs::{self, File};
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use plinko::delta::{Delta, DeltaHeader};
use plinko::hints::HintSet;
use plinko::keyword::{
    CachePolicy, KeywordBatchAnswer, KeywordBatchQuery, KeywordClient, KeywordEpochUpdate,
    KeywordLookupAnswer, KeywordLookupQuery, KeywordSession, KeywordTableSpec, KeywordTransport,
    KvScfTable, MutationOutcome, NUM_SEGMENTS,
};
use plinko::protocol::{
    Client as ProtocolClient, ProtocolError, ProtocolParams, Query as ProtocolQuery,
    Response as ProtocolResponse, Server as ProtocolServer,
};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

fn work_dir(name: &str) -> PathBuf {
    let dir: PathBuf = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("keyword_e2e_{name}_{}", std::process::id()));
    fs::create_dir_all(&dir).expect("create work dir");
    dir
}

fn remove_work_dir(dir: &Path) {
    let _ = fs::remove_dir_all(dir);
}

fn run_plinko_hints(
    db_path: &Path,
    seed: &[u8; 32],
    record_size: usize,
    entries_per_block: usize,
    lambda: usize,
    hint_path: &Path,
    seed_path: &Path,
) -> HintSet {
    let seed_hex: String = seed.iter().map(|b| format!("{b:02x}")).collect();
    fs::write(seed_path, format!("{seed_hex}\n")).expect("write seed file");

    let output: Output = Command::new(env!("CARGO_BIN_EXE_plinko_hints"))
        .arg("--db-path")
        .arg(db_path)
        .arg("--record-size")
        .arg(record_size.to_string())
        .arg("--entries-per-block")
        .arg(entries_per_block.to_string())
        .arg("--lambda")
        .arg(lambda.to_string())
        .arg("--seed-file")
        .arg(seed_path)
        .arg("--threads")
        .arg("2")
        .arg("--output")
        .arg(hint_path)
        .output()
        .expect("run plinko_hints");

    assert!(
        output.status.success(),
        "plinko_hints failed: stdout={}, stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let file: File = File::open(hint_path).expect("open hint file");
    HintSet::read_from(BufReader::new(file)).expect("read HintSet")
}

fn synth_kv(id: u64, key_size: usize, value_size: usize) -> (Vec<u8>, Vec<u8>) {
    let mut key: Vec<u8> = vec![0u8; key_size];
    for (chunk_idx, chunk) in key.chunks_mut(8).enumerate() {
        let mut x: u64 = id.wrapping_add((chunk_idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        x ^= x >> 31;
        let bytes: [u8; 8] = x.to_le_bytes();
        chunk.copy_from_slice(&bytes[..chunk.len()]);
    }
    let val: Vec<u8> = (0..value_size)
        .map(|i| (((id.wrapping_mul(131) ^ (i as u64).wrapping_mul(29)) & 0xFF) as u8) | 0x01)
        .collect();
    (key, val)
}

#[test]
fn single_database_bin_through_plinko_hints_and_protocol_client() {
    let dir: PathBuf = work_dir("single_db");
    let key_size: usize = 16;
    let value_size: usize = 24;
    let num_keys: usize = 96;
    let spec: KeywordTableSpec =
        KeywordTableSpec::with_layout(num_keys, Some(key_size), value_size, 8, 4, 0.85, [0xA1; 32])
            .expect("spec");

    let entries: Vec<(Vec<u8>, Vec<u8>)> = (0..num_keys as u64)
        .map(|id| synth_kv(id, key_size, value_size))
        .collect();
    let (mut table, stats) = KvScfTable::build(
        spec.clone(),
        entries
            .iter()
            .map(|(k, v)| (0x01u8, k.as_slice(), v.as_slice())),
    )
    .expect("build table");
    assert_eq!(stats.num_keys, num_keys);

    let db_path: PathBuf = dir.join("database.bin");
    table
        .write_database_file(&db_path)
        .expect("write database.bin");

    let master_seed: [u8; 32] = [0x5Cu8; 32];
    let query_seed: [u8; 32] = [0xA7u8; 32];
    let hints: HintSet = run_plinko_hints(
        &db_path,
        &master_seed,
        spec.record_size(),
        4,
        16,
        &dir.join("hints.bin"),
        &dir.join("seed.hex"),
    );

    let params: ProtocolParams = ProtocolParams::for_hint_set(&hints).expect("params");
    assert_eq!(params.num_entries, spec.total_buckets());
    assert_eq!(params.record_size, spec.record_size());

    let mut server: ProtocolServer =
        ProtocolServer::new(params.clone(), 0, table.as_bytes()).expect("server");
    let mut client: ProtocolClient =
        ProtocolClient::from_hints(params, master_seed, query_seed, hints).expect("client");

    // 1) Look up present keys and absent keys through the single combined database.
    for (k, expected_v) in entries.iter().take(6) {
        let loc = spec.locate(0x01, k).expect("locate");
        let [idx0, idx1] = loc.global_indices(&spec);
        assert_ne!(idx0, idx1);

        let p0 = client.prepare_query(idx0).expect("prep0");
        let p1 = client.prepare_query(idx1).expect("prep1");
        let r0 = server.answer(&p0.query).expect("ans0");
        let r1 = server.answer(&p1.query).expect("ans1");
        let b0 = client.reconstruct_and_replenish(&p0, &r0).expect("recon0");
        let b1 = client.reconstruct_and_replenish(&p1, &r1).expect("recon1");
        let found = spec.scan_buckets(&b0, &b1, &loc.tag, None).expect("scan");
        assert_eq!(found, Some(expected_v.as_slice()));
    }

    for miss_id in 10_000..10_004u64 {
        let (mk, _) = synth_kv(miss_id, key_size, value_size);
        let loc = spec.locate(0x01, &mk).expect("locate miss");
        let [idx0, idx1] = loc.global_indices(&spec);
        let p0 = client.prepare_query(idx0).expect("prep0");
        let p1 = client.prepare_query(idx1).expect("prep1");
        let r0 = server.answer(&p0.query).expect("ans0");
        let r1 = server.answer(&p1.query).expect("ans1");
        let b0 = client.reconstruct_and_replenish(&p0, &r0).expect("recon0");
        let b1 = client.reconstruct_and_replenish(&p1, &r1).expect("recon1");
        let found = spec.scan_buckets(&b0, &b1, &loc.tag, None).expect("scan");
        assert_eq!(found, None);
    }

    // 2) Live update + delete + insert -> single-DB Delta -> verify via PIR.
    let updated_val: Vec<u8> = vec![0x77; value_size];
    let (new_k, new_v) = synth_kv(50_000, key_size, value_size);
    let out_upd = table
        .update(0x01, &entries[0].0, &updated_val)
        .expect("update");
    let out_del = table.delete(0x01, &entries[1].0).expect("delete");
    let out_ins = table.insert(0x01, &new_k, &new_v).expect("insert");

    let mut delta_changes: Vec<(u64, Vec<u8>)> = Vec::new();
    let mut touched_global: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
    for out in [&out_upd, &out_del, &out_ins] {
        for d in &out.deltas {
            let gidx: usize = d.global_index(&spec);
            touched_global.insert(gidx);
            delta_changes.push((gidx as u64, d.xor_delta.clone()));
        }
    }
    let server_updates: Vec<(usize, Vec<u8>)> = touched_global
        .iter()
        .map(|&gidx| {
            let seg: usize = gidx / spec.buckets_per_segment();
            let bidx: usize = gidx % spec.buckets_per_segment();
            (gidx, table.bucket_bytes(seg, bidx).unwrap().to_vec())
        })
        .collect();
    let server_delta: Delta = server
        .apply_updates(&server_updates)
        .expect("server update");
    let expected_delta: Delta = Delta::new(
        DeltaHeader {
            num_entries: spec.total_buckets() as u64,
            record_size: spec.record_size() as u32,
            from_version: 0,
            to_version: 1,
        },
        delta_changes,
    )
    .expect("build delta");
    assert_eq!(server_delta, expected_delta);
    client.apply_delta(&expected_delta).expect("client delta");

    // Verify updated key, deleted key, and newly inserted key via PIR.
    for (k, expected) in [
        (&entries[0].0, Some(updated_val.as_slice())),
        (&entries[1].0, None),
        (&new_k, Some(new_v.as_slice())),
    ] {
        let loc = spec.locate(0x01, k).expect("locate");
        let [idx0, idx1] = loc.global_indices(&spec);
        let p0 = client.prepare_query(idx0).expect("prep0");
        let p1 = client.prepare_query(idx1).expect("prep1");
        let r0 = server.answer(&p0.query).expect("ans0");
        let r1 = server.answer(&p1.query).expect("ans1");
        let b0 = client.reconstruct_and_replenish(&p0, &r0).expect("recon0");
        let b1 = client.reconstruct_and_replenish(&p1, &r1).expect("recon1");
        assert_eq!(
            spec.scan_buckets(&b0, &b1, &loc.tag, None).unwrap(),
            expected
        );
    }

    remove_work_dir(&dir);
}

struct TwoSegmentProtocolTransport {
    table: KvScfTable,
    servers: [ProtocolServer; NUM_SEGMENTS],
}

impl TwoSegmentProtocolTransport {
    fn publish_outcomes(&mut self, outcomes: &[MutationOutcome]) -> KeywordEpochUpdate {
        let from_epoch: u64 = self.table.spec().state_epoch();
        let to_epoch: u64 = self.table.advance_state_epoch().unwrap();
        let update: KeywordEpochUpdate = KeywordEpochUpdate::from_outcomes(
            self.table.spec().table_epoch(),
            from_epoch,
            to_epoch,
            outcomes,
        );
        for seg in 0..NUM_SEGMENTS {
            if !update.segment_deltas[seg].is_empty() {
                let server_updates: Vec<(usize, Vec<u8>)> = update.segment_deltas[seg]
                    .iter()
                    .map(|&(bidx, _)| (bidx, self.table.bucket_bytes(seg, bidx).unwrap().to_vec()))
                    .collect();
                self.servers[seg]
                    .apply_updates(&server_updates)
                    .expect("server segment update");
            }
        }
        update
    }
}

impl KeywordTransport<ProtocolQuery, ProtocolResponse> for TwoSegmentProtocolTransport {
    type Error = ProtocolError;

    fn send_lookup(
        &mut self,
        query: KeywordLookupQuery<ProtocolQuery>,
    ) -> Result<KeywordLookupAnswer<ProtocolResponse>, Self::Error> {
        let a0 = self.servers[0].answer(&query.segments[0])?;
        let a1 = self.servers[1].answer(&query.segments[1])?;
        Ok(KeywordLookupAnswer {
            table_epoch: self.table.spec().table_epoch(),
            state_epoch: self.table.spec().state_epoch(),
            segments: [a0, a1],
        })
    }

    fn send_batch(
        &mut self,
        query: KeywordBatchQuery<ProtocolQuery>,
    ) -> Result<KeywordBatchAnswer<ProtocolResponse>, Self::Error> {
        let a0: Vec<ProtocolResponse> = query.segments[0]
            .iter()
            .map(|q| self.servers[0].answer(q))
            .collect::<Result<Vec<_>, _>>()?;
        let a1: Vec<ProtocolResponse> = query.segments[1]
            .iter()
            .map(|q| self.servers[1].answer(q))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(KeywordBatchAnswer {
            table_epoch: self.table.spec().table_epoch(),
            state_epoch: self.table.spec().state_epoch(),
            segments: [a0, a1],
        })
    }
}

#[test]
fn two_segment_keyword_client_through_plinko_hints_and_session() {
    let dir: PathBuf = work_dir("two_segment");
    let value_size: usize = 20;
    let num_keys: usize = 100;
    let spec: KeywordTableSpec =
        KeywordTableSpec::for_capacity(num_keys, value_size, 0.80, [0xC3; 32]).expect("spec");
    let entries: Vec<(Vec<u8>, Vec<u8>)> = (0..num_keys as u64)
        .map(|id| {
            (
                format!("acct-{id:05}").into_bytes(),
                synth_kv(id, 4, value_size).1,
            )
        })
        .collect();
    let (table, _) = KvScfTable::build(
        spec.clone(),
        entries
            .iter()
            .map(|(k, v)| (0x01u8, k.as_slice(), v.as_slice())),
    )
    .expect("build");

    // Write disjoint segment files and generate hints via `plinko_hints` for each segment.
    let seg0_path: PathBuf = dir.join("segment_0.bin");
    let seg1_path: PathBuf = dir.join("segment_1.bin");
    table
        .write_segments(
            File::create(&seg0_path).expect("seg0"),
            File::create(&seg1_path).expect("seg1"),
        )
        .expect("write_segments");

    let seg_seeds: [[u8; 32]; 2] = [[0x19; 32], [0x29; 32]];
    let q_seeds: [[u8; 32]; 2] = [[0x39; 32], [0x49; 32]];
    let h0: HintSet = run_plinko_hints(
        &seg0_path,
        &seg_seeds[0],
        spec.record_size(),
        4,
        16,
        &dir.join("hints_0.bin"),
        &dir.join("seed_0.hex"),
    );
    let h1: HintSet = run_plinko_hints(
        &seg1_path,
        &seg_seeds[1],
        spec.record_size(),
        4,
        16,
        &dir.join("hints_1.bin"),
        &dir.join("seed_1.hex"),
    );

    let p0: ProtocolParams = ProtocolParams::for_hint_set(&h0).expect("p0");
    let p1: ProtocolParams = ProtocolParams::for_hint_set(&h1).expect("p1");
    let s0: ProtocolServer =
        ProtocolServer::new(p0.clone(), 0, table.segment_bytes(0).unwrap()).expect("s0");
    let s1: ProtocolServer =
        ProtocolServer::new(p1.clone(), 0, table.segment_bytes(1).unwrap()).expect("s1");
    let c0: ProtocolClient =
        ProtocolClient::from_hints(p0, seg_seeds[0], q_seeds[0], h0).expect("c0");
    let c1: ProtocolClient =
        ProtocolClient::from_hints(p1, seg_seeds[1], q_seeds[1], h1).expect("c1");

    let kw: KeywordClient<ProtocolClient> =
        KeywordClient::new(spec, [c0, c1], CachePolicy::ConstantTraffic).expect("kw");
    let transport = TwoSegmentProtocolTransport {
        table,
        servers: [s0, s1],
    };
    let mut session = KeywordSession::new(kw, transport, ChaCha20Rng::seed_from_u64(1234));

    // Single lookup + batch lookup with duplicates and misses.
    assert_eq!(
        session.get(0x01, &entries[0].0).expect("get0").as_deref(),
        Some(entries[0].1.as_slice())
    );
    assert_eq!(session.get(0x01, b"nonexistent").expect("miss"), None);

    let batch = session
        .get_batch(&[
            (0x01, entries[0].0.as_slice()),
            (0x01, entries[1].0.as_slice()),
            (0x01, entries[1].0.as_slice()),
            (0x01, b"missing-in-batch"),
        ])
        .expect("batch");
    assert_eq!(batch[0].as_deref(), Some(entries[0].1.as_slice()));
    assert_eq!(batch[1].as_deref(), Some(entries[1].1.as_slice()));
    assert_eq!(batch[2].as_deref(), Some(entries[1].1.as_slice()));
    assert_eq!(batch[3], None);

    // Live insert + update + delete via KeywordEpochUpdate.
    let new_v0: Vec<u8> = vec![0xBB; value_size];
    let new_key = b"acct-live-inserted";
    let new_val: Vec<u8> = vec![0xCC; value_size];
    let out_upd = session
        .transport_mut()
        .table
        .update(0x01, &entries[0].0, &new_v0)
        .expect("upd");
    let out_del = session
        .transport_mut()
        .table
        .delete(0x01, &entries[2].0)
        .expect("del");
    let out_ins = session
        .transport_mut()
        .table
        .insert(0x01, new_key, &new_val)
        .expect("ins");
    let epoch_upd = session
        .transport_mut()
        .publish_outcomes(&[out_upd, out_del, out_ins]);
    session
        .client_mut()
        .apply_epoch_update(&epoch_upd, 2)
        .expect("apply_epoch_update");

    assert_eq!(
        session
            .get(0x01, &entries[0].0)
            .expect("get0_new")
            .as_deref(),
        Some(new_v0.as_slice())
    );
    assert_eq!(session.get(0x01, &entries[2].0).expect("get2_del"), None);
    assert_eq!(
        session.get(0x01, new_key).expect("get_ins").as_deref(),
        Some(new_val.as_slice())
    );

    // Checkpoint roundtrip + post-checkpoint WAL logging + `retire_hints` recovery.
    let ckpt: Vec<u8> = session.client().to_bytes().expect("to_bytes");
    let (mut orig_client, mut transport, mut rng) = session.into_parts();

    // Perform post-checkpoint lookups on the original client while logging WAL hint IDs.
    let mut wal0: Vec<usize> = Vec::new();
    let mut wal1: Vec<usize> = Vec::new();
    for (key, _) in entries.iter().take(6).skip(3) {
        let plinko::keyword::LookupPlan::Query { query, pending } = orig_client
            .prepare_lookup(0x01, key, &mut rng)
            .expect("post-ckpt prepare")
        else {
            unreachable!()
        };
        let [h0, h1] = pending.hint_ids();
        wal0.push(h0);
        wal1.push(h1);
        let ans = transport.send_lookup(query).expect("post-ckpt answer");
        orig_client
            .finish_lookup(&pending, &ans, &mut rng)
            .expect("post-ckpt finish");
    }

    // Restore from pre-crash checkpoint, replay WAL via `retire_hints`, and confirm
    // subsequent queries never reuse any WAL-logged hint ID.
    let mut restored: KeywordClient<ProtocolClient> =
        KeywordClient::<ProtocolClient>::from_bytes(&ckpt, &seg_seeds, &[[0x59; 32], [0x69; 32]])
            .expect("from_bytes");
    restored
        .retire_hints(&wal0, &wal1)
        .expect("retire WAL hints");
    let wal0_set: std::collections::HashSet<usize> = wal0.iter().copied().collect();
    let wal1_set: std::collections::HashSet<usize> = wal1.iter().copied().collect();

    let plinko::keyword::LookupPlan::Query { query, pending } = restored
        .prepare_lookup(0x01, new_key, &mut rng)
        .expect("post-restore prepare")
    else {
        unreachable!()
    };
    let [rh0, rh1] = pending.hint_ids();
    assert!(!wal0_set.contains(&rh0), "reused retired seg0 hint {rh0}");
    assert!(!wal1_set.contains(&rh1), "reused retired seg1 hint {rh1}");
    let ans = transport.send_lookup(query).expect("post-restore answer");
    assert_eq!(
        restored
            .finish_lookup(&pending, &ans, &mut rng)
            .expect("post-restore finish")
            .as_deref(),
        Some(new_val.as_slice())
    );

    remove_work_dir(&dir);
}
