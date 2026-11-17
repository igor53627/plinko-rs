use std::collections::HashSet;

use plinko::protocol::{Client, Entry, ProtocolParams, Server, ENTRY_SIZE};
use plinko::subsets::{backup_subset_size, SubsetPrf};

fn make_entry(tag: u64) -> Entry {
    let mut entry = [0u8; ENTRY_SIZE];
    entry[0..8].copy_from_slice(&tag.to_le_bytes());
    for (i, byte) in entry.iter_mut().enumerate().skip(8) {
        *byte = (tag as u8).wrapping_add((i as u8).wrapping_mul(17));
    }
    entry
}

fn make_db(n: usize) -> Vec<Entry> {
    (0..n).map(|i| make_entry(i as u64 + 11)).collect()
}

#[test]
fn protocol_query_answer_reconstruct_roundtrip() {
    let params = ProtocolParams::new(
        97,       // n
        Some(16), // w
        4,        // lambda
        None,     // num_backup_hints = lambda * w
    )
    .expect("params");

    let entries = make_db(params.num_entries);
    let master_seed = [0x11u8; 32];
    let query_seed = [0x22u8; 32];

    let server = Server::new(params.clone(), &entries).expect("server init");
    let mut client = Client::offline_init(params.clone(), master_seed, query_seed, &entries)
        .expect("offline init");

    let mut tested = 0usize;
    #[allow(clippy::needless_range_loop)]
    for index in 0..params.num_entries {
        if tested == 6 {
            break;
        }
        let Ok(prepared) = client.prepare_query(index) else {
            continue;
        };
        let response = server.answer(&prepared.query).expect("answer");
        let got = client
            .reconstruct_and_replenish(prepared, response)
            .expect("reconstruct");
        assert_eq!(got, entries[index], "roundtrip mismatch at index {index}");
        tested += 1;
    }
    assert_eq!(tested, 6, "expected at least 6 queryable indices");

    assert_eq!(client.num_cached_entries(), tested);
    assert_eq!(
        client.remaining_backup_hints(),
        params.num_backup_hints - tested
    );
}

#[test]
fn protocol_cache_update_and_requery() {
    let params = ProtocolParams::new(
        128,      // n
        Some(16), // w
        3,        // lambda
        None,
    )
    .expect("params");

    let mut entries = make_db(params.num_entries);
    let master_seed = [0x33u8; 32];
    let query_seed = [0x44u8; 32];

    let mut server = Server::new(params.clone(), &entries).expect("server init");
    let mut client =
        Client::offline_init(params, master_seed, query_seed, &entries).expect("offline init");

    let mut target = None;
    let mut prepared_1_opt = None;
    #[allow(clippy::needless_range_loop)]
    for idx in 0..entries.len() {
        if let Ok(prepared) = client.prepare_query(idx) {
            target = Some(idx);
            prepared_1_opt = Some(prepared);
            break;
        }
    }
    let target = target.expect("no queryable index found");
    let prepared_1 = prepared_1_opt.expect("prepared query missing");
    let response_1 = server.answer(&prepared_1.query).expect("answer #1");
    let got_1 = client
        .reconstruct_and_replenish(prepared_1, response_1)
        .expect("reconstruct #1");
    assert_eq!(got_1, entries[target]);
    assert_eq!(client.num_cached_entries(), 1);

    let updated_value = make_entry(0xDEAD_BEEFu64);
    let deltas = server
        .apply_updates(&[(target, updated_value)])
        .expect("server update");
    client.apply_updates(&deltas).expect("client update");
    entries[target] = updated_value;

    let backups_before = client.remaining_backup_hints();
    let prepared_2 = client.prepare_query(target).expect("prepare #2");
    let response_2 = server.answer(&prepared_2.query).expect("answer #2");
    let got_2 = client
        .reconstruct_and_replenish(prepared_2, response_2)
        .expect("reconstruct #2");

    assert_eq!(got_2, entries[target], "updated value mismatch");
    assert_eq!(client.remaining_backup_hints(), backups_before - 1);
    assert!(
        client.num_cached_entries() >= 2,
        "cached re-query should also cache a decoy entry"
    );
}

#[test]
fn test_protocol_e2e_after_compaction() {
    let params = ProtocolParams::new(
        128,      // n
        Some(16), // w
        4,        // lambda
        None,
    )
    .expect("params");

    let entries = make_db(params.num_entries);
    let master_seed = [0xAAu8; 32];
    let query_seed = [0xBBu8; 32];

    let server = Server::new(params.clone(), &entries).expect("server init");
    let mut client = Client::offline_init(params.clone(), master_seed, query_seed, &entries)
        .expect("offline init");

    let mut tested = 0usize;
    for (index, expected) in entries.iter().enumerate() {
        if tested == 8 {
            break;
        }
        let Ok(prepared) = client.prepare_query(index) else {
            continue;
        };
        let response = server.answer(&prepared.query).expect("answer");
        let got = client
            .reconstruct_and_replenish(prepared, response)
            .expect("reconstruct");
        assert_eq!(got, *expected, "roundtrip mismatch at index {index}");
        tested += 1;
    }
    assert!(
        tested >= 6,
        "expected at least 6 queryable indices, got {tested}"
    );

    assert_eq!(client.num_cached_entries(), tested);
    assert_eq!(
        client.remaining_backup_hints(),
        params.num_backup_hints - tested
    );
}

/// Exercises apply_updates through promoted (potentially complemented) slots.
///
/// Queries multiple indices to force backup promotions — statistically ~50%
/// will have `complemented = true` — then mutates entries and verifies that
/// re-querying after updates still returns correct values.
#[test]
fn test_update_through_complemented_promoted_slots() {
    let params = ProtocolParams::new(
        128,      // n
        Some(16), // w
        3,        // lambda
        None,
    )
    .expect("params");

    let mut entries = make_db(params.num_entries);
    // The scenario checks below hold for about one master seed in twelve;
    // this seed is one of them. The value checks do not depend on the seed.
    let master_seed = [0x2Au8; 32];
    let query_seed = [0xDDu8; 32];

    let mut server = Server::new(params.clone(), &entries).expect("server init");
    let mut client = Client::offline_init(params.clone(), master_seed, query_seed, &entries)
        .expect("offline init");

    // Phase 1: Query a few indices, consuming backups (each promotes one).
    // Backups are consumed in order 0, 1, 2, ... so we can verify which
    // promotions set complemented=true by checking whether the queried block
    // falls in the backup's original subset.
    let num_phase1 = 3;
    let mut phase1_count = 0;
    let mut queried_indices = Vec::new();
    #[allow(clippy::needless_range_loop)]
    for idx in 0..entries.len() {
        if phase1_count == num_phase1 {
            break;
        }
        let Ok(prepared) = client.prepare_query(idx) else {
            continue;
        };
        let response = server.answer(&prepared.query).expect("answer");
        let got = client
            .reconstruct_and_replenish(prepared, response)
            .expect("reconstruct");
        assert_eq!(got, entries[idx], "pre-update mismatch at index {idx}");
        queried_indices.push(idx);
        phase1_count += 1;
    }
    assert_eq!(phase1_count, num_phase1);
    let backups_after_phase1 = client.remaining_backup_hints();

    // Verify that at least one promoted slot actually had complemented=true.
    // Backups are promoted in order, so backup j was used for queried_indices[j].
    let subset_prf: SubsetPrf = SubsetPrf::new(&master_seed);
    let mut scratch: Vec<u32> = Vec::new();
    let mut complemented_promoted_slots = HashSet::new();
    for (j, &idx) in queried_indices.iter().enumerate() {
        let hint_id: u64 = (params.num_regular_hints + j) as u64;
        let cutoff: u32 = subset_prf.cutoff(
            hint_id,
            params.num_blocks,
            backup_subset_size(params.num_blocks),
            &mut scratch,
        );
        let queried_block = idx / params.block_size;
        if subset_prf.contains(hint_id, queried_block, cutoff) {
            complemented_promoted_slots.insert(params.num_regular_hints + j);
        }
    }
    assert!(
        !complemented_promoted_slots.is_empty(),
        "test seed did not produce any complemented promotions; \
         change master_seed or query more indices"
    );

    // Phase 2: Update every entry in the database.
    let mut server_updates = Vec::new();
    for (idx, entry) in entries.iter_mut().enumerate() {
        let new_val = make_entry(0xF000 + idx as u64);
        server_updates.push((idx, new_val));
        *entry = new_val;
    }
    let deltas = server
        .apply_updates(&server_updates)
        .expect("server update");
    client.apply_updates(&deltas).expect("client update");

    // Phase 3: Query new indices after the update. These queries use hints
    // (including promoted slots with complemented=true) whose parities were
    // maintained through apply_updates. Correct reconstruction validates that
    // the complement flag was handled properly during the update.
    let num_phase3 = 3;
    let mut phase3_count = 0;
    let mut consumed_complemented_promoted_slot = false;
    // Start from a higher index to avoid re-querying phase-1 cached entries
    // via the cache path (which would bypass the hint-based reconstruction).
    #[allow(clippy::needless_range_loop)]
    for idx in (entries.len() / 2)..entries.len() {
        if phase3_count == num_phase3 {
            break;
        }
        let Ok(prepared) = client.prepare_query(idx) else {
            continue;
        };
        consumed_complemented_promoted_slot |=
            complemented_promoted_slots.contains(&prepared.consumed_hint_index());
        let response = server.answer(&prepared.query).expect("answer post-update");
        let got = client
            .reconstruct_and_replenish(prepared, response)
            .expect("reconstruct post-update");
        assert_eq!(got, entries[idx], "post-update mismatch at index {idx}");
        phase3_count += 1;
    }
    assert!(
        phase3_count >= 2,
        "need at least 2 post-update queries, got {phase3_count}"
    );
    assert!(
        consumed_complemented_promoted_slot,
        "post-update queries did not consume a complemented promoted slot"
    );
    assert_eq!(
        client.remaining_backup_hints(),
        backups_after_phase1 - phase3_count
    );
}
