//! Standalone zero-import WebAssembly smoke test for 2-level Variant B (`FlatAes`)
//! covering `HintInit`, `HintSet` serialization/deserialization, queries, failure-private
//! refresh, and interleaved `apply_delta` updates.
//!
//! Build and run via `scripts/test_wasm_multilevel.sh`.

use plinko::multilevel::{
    BackendVariant, MultiLevelAnswer, MultiLevelClient, MultiLevelError, MultiLevelParams,
    MultiLevelServer, PendingMultiLevelQuery,
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

/// Runs a full end-to-end 2-level Variant B lifecycle inside WebAssembly.
///
/// Returns `0` on success, or a negative step code on failure.
#[no_mangle]
pub extern "C" fn run_multilevel_2b_smoke() -> i32 {
    let num_entries: usize = 65_536;
    let record_size: usize = 40;
    let seed: [u8; 32] = [0x5A; 32];

    let mut db: Vec<u8> = vec![0u8; num_entries * record_size];
    for (i, chunk) in db.chunks_exact_mut(8).enumerate() {
        let v: u64 = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xCAFE_BABE_1234_5678;
        chunk.copy_from_slice(&v.to_le_bytes());
    }

    let params: MultiLevelParams =
        match MultiLevelParams::two_level(num_entries, record_size, 32, 128, BackendVariant::FlatB)
        {
            Ok(p) => p,
            Err(_) => return -1,
        };

    let client: MultiLevelClient = match MultiLevelClient::hint_init(params.clone(), &seed, &db, 1)
    {
        Ok(c) => c,
        Err(_) => return -2,
    };

    // 1. Fresh HintSet serialization + corruption rejection + round-trip reload.
    let fresh_bytes: Vec<u8> = match client.to_bytes() {
        Ok(b) => b,
        Err(_) => return -3,
    };
    let mut corrupted: Vec<u8> = fresh_bytes.clone();
    corrupted[fresh_bytes.len() / 2] ^= 0x01;
    if MultiLevelClient::from_bytes(&corrupted, &seed).err()
        != Some(MultiLevelError::ChecksumMismatch)
    {
        return -4;
    }

    let mut restored: MultiLevelClient = match MultiLevelClient::from_bytes(&fresh_bytes, &seed) {
        Ok(c) => c,
        Err(_) => return -5,
    };

    // 2. Queries + interleaved deltas (stressing PromoteIn / PromoteOut overrides).
    let mut rng: ChaCha20Rng = ChaCha20Rng::seed_from_u64(0x2026_1273);
    let mut queried: Vec<usize> = Vec::with_capacity(20);
    for step in 0..20 {
        let idx: usize = if step % 3 == 2 && !queried.is_empty() {
            queried[rng.gen_range(0..queried.len())]
        } else {
            rng.gen_range(0..num_entries)
        };
        let pending: PendingMultiLevelQuery = match restored.prepare_query(idx, &mut rng) {
            Ok(p) => p,
            Err(_) => return -6,
        };
        let ans: MultiLevelAnswer =
            match MultiLevelServer::answer(&db, &params, &pending.query, restored.epoch()) {
                Ok(a) => a,
                Err(_) => return -17,
            };
        let got: Vec<u8> = match restored.finish_query(&pending, &ans, &mut rng) {
            Ok(v) => v,
            Err(_) => return -7,
        };
        if got != db[idx * record_size..(idx + 1) * record_size] {
            return -8;
        }
        queried.push(idx);

        let mut delta: Vec<u8> = vec![0u8; record_size];
        rng.fill(&mut delta[..]);
        for (a, &b) in db[idx * record_size..(idx + 1) * record_size]
            .iter_mut()
            .zip(&delta)
        {
            *a ^= b;
        }
        if restored.apply_delta(idx, &delta).is_err() {
            return -9;
        }
    }

    // 3. Mid-session HintSet serialization (with active overrides and consumed backups) + reload.
    let mid_bytes: Vec<u8> = match restored.to_bytes() {
        Ok(b) => b,
        Err(_) => return -10,
    };
    if mid_bytes.len() >= fresh_bytes.len() {
        return -11;
    }

    let mut reloaded: MultiLevelClient = match MultiLevelClient::from_bytes(&mid_bytes, &seed) {
        Ok(c) => c,
        Err(_) => return -12,
    };

    // 4. Continued queries + deltas after reload.
    for step in 0..10 {
        let idx: usize = if step % 2 == 0 {
            queried[rng.gen_range(0..queried.len())]
        } else {
            rng.gen_range(0..num_entries)
        };
        let pending: PendingMultiLevelQuery = match reloaded.prepare_query(idx, &mut rng) {
            Ok(p) => p,
            Err(_) => return -13,
        };
        let ans: MultiLevelAnswer =
            match MultiLevelServer::answer(&db, &params, &pending.query, reloaded.epoch()) {
                Ok(a) => a,
                Err(_) => return -18,
            };
        let got: Vec<u8> = match reloaded.finish_query(&pending, &ans, &mut rng) {
            Ok(v) => v,
            Err(_) => return -14,
        };
        if got != db[idx * record_size..(idx + 1) * record_size] {
            return -15;
        }

        let mut delta: Vec<u8> = vec![0u8; record_size];
        rng.fill(&mut delta[..]);
        for (a, &b) in db[idx * record_size..(idx + 1) * record_size]
            .iter_mut()
            .zip(&delta)
        {
            *a ^= b;
        }
        if reloaded.apply_delta(idx, &delta).is_err() {
            return -16;
        }
    }

    0
}
