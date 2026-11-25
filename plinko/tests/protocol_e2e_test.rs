use std::collections::HashSet;

use plinko::delta::{Delta, DeltaHeader};
use plinko::hints::HintSet;
use plinko::protocol::{
    compute_hints, compute_hints_with_threads, derive_block_keys, Client, PreparedQuery,
    ProtocolError, ProtocolParams, Query, Response, Server,
};
use plinko::subsets::{backup_subset_size, SubsetPrf};

/// Record size of the test databases: not a multiple of 8, and above 32.
const RECORD_SIZE: usize = 45;

fn make_entry(tag: u64) -> Vec<u8> {
    let mut entry: Vec<u8> = vec![0u8; RECORD_SIZE];
    entry[0..8].copy_from_slice(&tag.to_le_bytes());
    for (i, byte) in entry.iter_mut().enumerate().skip(8) {
        *byte = (tag as u8).wrapping_add((i as u8).wrapping_mul(17));
    }
    entry[RECORD_SIZE - 1] |= 0x80;
    entry
}

/// A database of `n` records, one after another.
fn make_db(n: usize) -> Vec<u8> {
    (0..n).flat_map(|i| make_entry(i as u64 + 11)).collect()
}

fn record(db: &[u8], index: usize) -> &[u8] {
    &db[index * RECORD_SIZE..(index + 1) * RECORD_SIZE]
}

fn set_record(db: &mut [u8], index: usize, value: &[u8]) {
    db[index * RECORD_SIZE..(index + 1) * RECORD_SIZE].copy_from_slice(value);
}

/// Prepares a query for the first index of `indices` that has a hint.
fn prepare_first(
    client: &mut Client,
    indices: impl Iterator<Item = usize>,
) -> (usize, PreparedQuery) {
    for index in indices {
        if let Ok(prepared) = client.prepare_query(index) {
            return (index, prepared);
        }
    }
    panic!("no queryable index found");
}

#[test]
fn protocol_query_answer_reconstruct_roundtrip() {
    let params = ProtocolParams::new(
        97, // n
        RECORD_SIZE,
        Some(16), // w
        4,        // lambda
        None,     // num_backup_hints = lambda * w
    )
    .expect("params");

    let entries = make_db(params.num_entries);
    let master_seed = [0x11u8; 32];
    let query_seed = [0x22u8; 32];

    let server = Server::new(params.clone(), 0, &entries).expect("server init");
    let mut client = Client::offline_init(params.clone(), master_seed, query_seed, 0, &entries)
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
            .reconstruct_and_replenish(&prepared, &response)
            .expect("reconstruct");
        assert_eq!(
            got,
            record(&entries, index),
            "roundtrip mismatch at index {index}"
        );
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
        128, // n
        RECORD_SIZE,
        Some(16), // w
        3,        // lambda
        None,
    )
    .expect("params");

    let mut entries = make_db(params.num_entries);
    let master_seed = [0x33u8; 32];
    let query_seed = [0x44u8; 32];

    let mut server = Server::new(params.clone(), 0, &entries).expect("server init");
    let mut client = Client::offline_init(params.clone(), master_seed, query_seed, 0, &entries)
        .expect("offline init");

    let mut target = None;
    let mut prepared_1_opt = None;
    #[allow(clippy::needless_range_loop)]
    for idx in 0..params.num_entries {
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
        .reconstruct_and_replenish(&prepared_1, &response_1)
        .expect("reconstruct #1");
    assert_eq!(got_1, record(&entries, target));
    assert_eq!(client.num_cached_entries(), 1);

    let updated_value = make_entry(0xDEAD_BEEFu64);
    let delta: Delta = server
        .apply_updates(&[(target, updated_value.clone())])
        .expect("server update");
    client.apply_delta(&delta).expect("client update");
    set_record(&mut entries, target, &updated_value);

    let backups_before = client.remaining_backup_hints();
    let prepared_2 = client.prepare_query(target).expect("prepare #2");
    let response_2 = server.answer(&prepared_2.query).expect("answer #2");
    let got_2 = client
        .reconstruct_and_replenish(&prepared_2, &response_2)
        .expect("reconstruct #2");

    assert_eq!(got_2, record(&entries, target), "updated value mismatch");
    assert_eq!(client.remaining_backup_hints(), backups_before - 1);
    assert!(
        client.num_cached_entries() >= 2,
        "cached re-query should also cache a decoy entry"
    );
}

#[test]
fn test_protocol_e2e_after_compaction() {
    let params = ProtocolParams::new(
        128, // n
        RECORD_SIZE,
        Some(16), // w
        4,        // lambda
        None,
    )
    .expect("params");

    let entries = make_db(params.num_entries);
    let master_seed = [0xAAu8; 32];
    let query_seed = [0xBBu8; 32];

    let server = Server::new(params.clone(), 0, &entries).expect("server init");
    let mut client = Client::offline_init(params.clone(), master_seed, query_seed, 0, &entries)
        .expect("offline init");

    let mut tested = 0usize;
    for (index, expected) in entries.chunks_exact(RECORD_SIZE).enumerate() {
        if tested == 8 {
            break;
        }
        let Ok(prepared) = client.prepare_query(index) else {
            continue;
        };
        let response = server.answer(&prepared.query).expect("answer");
        let got = client
            .reconstruct_and_replenish(&prepared, &response)
            .expect("reconstruct");
        assert_eq!(got, expected, "roundtrip mismatch at index {index}");
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
        128, // n
        RECORD_SIZE,
        Some(16), // w
        3,        // lambda
        None,
    )
    .expect("params");

    let mut entries = make_db(params.num_entries);
    // The scenario checks below hold for a part of the seed pairs (the hint
    // of a query is a random candidate); this pair is one of them. The value
    // checks do not depend on the seeds.
    let master_seed = [0x2Au8; 32];
    let query_seed = [0xD0u8; 32];

    let mut server = Server::new(params.clone(), 0, &entries).expect("server init");
    let mut client = Client::offline_init(params.clone(), master_seed, query_seed, 0, &entries)
        .expect("offline init");

    // Phase 1: Query a few indices, consuming backups (each promotes one).
    // Backups are consumed in order 0, 1, 2, ... so we can verify which
    // promotions set complemented=true by checking whether the queried block
    // falls in the backup's original subset.
    let num_phase1 = 3;
    let mut phase1_count = 0;
    let mut queried_indices = Vec::new();
    #[allow(clippy::needless_range_loop)]
    for idx in 0..params.num_entries {
        if phase1_count == num_phase1 {
            break;
        }
        let Ok(prepared) = client.prepare_query(idx) else {
            continue;
        };
        let response = server.answer(&prepared.query).expect("answer");
        let got = client
            .reconstruct_and_replenish(&prepared, &response)
            .expect("reconstruct");
        assert_eq!(
            got,
            record(&entries, idx),
            "pre-update mismatch at index {idx}"
        );
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
    for (idx, entry) in entries.chunks_exact_mut(RECORD_SIZE).enumerate() {
        let new_val: Vec<u8> = make_entry(0xF000 + idx as u64);
        entry.copy_from_slice(&new_val);
        server_updates.push((idx, new_val));
    }
    let delta: Delta = server
        .apply_updates(&server_updates)
        .expect("server update");
    client.apply_delta(&delta).expect("client update");

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
    for idx in (params.num_entries / 2)..params.num_entries {
        if phase3_count == num_phase3 {
            break;
        }
        let Ok(prepared) = client.prepare_query(idx) else {
            continue;
        };
        consumed_complemented_promoted_slot |=
            complemented_promoted_slots.contains(&prepared.hint_index());
        let response = server.answer(&prepared.query).expect("answer post-update");
        let got = client
            .reconstruct_and_replenish(&prepared, &response)
            .expect("reconstruct post-update");
        assert_eq!(
            got,
            record(&entries, idx),
            "post-update mismatch at index {idx}"
        );
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

#[test]
fn a_lost_answer_keeps_the_hint_and_a_retry_sends_the_same_query() {
    let params: ProtocolParams =
        ProtocolParams::new(128, RECORD_SIZE, Some(16), 4, None).expect("params");
    let mut entries: Vec<u8> = make_db(params.num_entries);
    let mut server: Server = Server::new(params.clone(), 0, &entries).expect("server init");
    let mut client: Client =
        Client::offline_init(params.clone(), [0x51u8; 32], [0x52u8; 32], 0, &entries)
            .expect("offline init");

    let (index, prepared) = prepare_first(&mut client, 0..params.num_entries);
    let backups_before: usize = client.remaining_backup_hints();
    assert_eq!(client.queries_in_flight(), 1);
    assert_eq!(
        client.prepare_query(index).err(),
        Some(ProtocolError::QueryInFlight { index })
    );

    let late: Response = server
        .answer(&prepared.query)
        .expect("answer that arrives late");

    let server_updates: Vec<(usize, Vec<u8>)> = (0..params.num_entries)
        .map(|idx| (idx, make_entry(0xA000 + idx as u64)))
        .collect();
    let delta: Delta = server
        .apply_updates(&server_updates)
        .expect("server update");
    client.apply_delta(&delta).expect("client update");
    for (idx, value) in &server_updates {
        set_record(&mut entries, *idx, value);
    }

    // The first answer comes from the database before the update. With the
    // parity of the hint after the update it would give a wrong record.
    assert_eq!(
        client.reconstruct_and_replenish(&prepared, &late),
        Err(ProtocolError::DatabaseVersionMismatch {
            expected: 1,
            got: 0
        })
    );
    assert_eq!(client.queries_in_flight(), 1);
    assert_eq!(client.remaining_backup_hints(), backups_before);

    let response = server.answer(&prepared.query).expect("answer to the retry");
    assert_eq!(response.database_version, 1);
    let got: Vec<u8> = client
        .reconstruct_and_replenish(&prepared, &response)
        .expect("reconstruct");

    assert_eq!(got, record(&entries, index));
    assert_eq!(client.queries_in_flight(), 0);
    assert_eq!(client.num_cached_entries(), 1);
    assert_eq!(client.remaining_backup_hints(), backups_before - 1);
}

#[test]
fn a_query_in_flight_reserves_its_hint_and_a_backup_hint() {
    let params: ProtocolParams =
        ProtocolParams::new(128, RECORD_SIZE, Some(16), 4, Some(2)).expect("params");
    let entries: Vec<u8> = make_db(params.num_entries);
    let server: Server = Server::new(params.clone(), 0, &entries).expect("server init");
    let mut client: Client =
        Client::offline_init(params.clone(), [0x61u8; 32], [0x62u8; 32], 0, &entries)
            .expect("offline init");

    let (index_a, prepared_a) = prepare_first(&mut client, 0..params.num_entries);
    let (index_b, prepared_b) = prepare_first(&mut client, (index_a + 1)..params.num_entries);
    assert_ne!(prepared_a.hint_index(), prepared_b.hint_index());
    assert_eq!(client.queries_in_flight(), 2);
    assert_eq!(client.remaining_backup_hints(), 2);

    for index in 0..params.num_entries {
        assert_eq!(
            client.prepare_query(index).err(),
            Some(ProtocolError::NoBackupHintsLeft),
            "index {index}"
        );
    }
    assert_eq!(client.queries_in_flight(), 2);

    let response_b = server.answer(&prepared_b.query).expect("answer b");
    let response_a = server.answer(&prepared_a.query).expect("answer a");
    assert_eq!(
        client.reconstruct_and_replenish(&prepared_b, &response_b),
        Ok(record(&entries, index_b).to_vec())
    );
    assert_eq!(
        client.reconstruct_and_replenish(&prepared_a, &response_a),
        Ok(record(&entries, index_a).to_vec())
    );
    assert_eq!(client.remaining_backup_hints(), 0);
}

#[test]
fn the_client_stops_at_the_query_limit() {
    let query_limit: usize = 5;
    let params: ProtocolParams =
        ProtocolParams::new(128, RECORD_SIZE, Some(16), 4, Some(query_limit)).expect("params");
    let entries: Vec<u8> = make_db(params.num_entries);
    let server: Server = Server::new(params.clone(), 0, &entries).expect("server init");
    let mut client: Client =
        Client::offline_init(params.clone(), [0x71u8; 32], [0x72u8; 32], 0, &entries)
            .expect("offline init");

    let mut answered: Vec<usize> = Vec::new();
    for (index, expected) in entries.chunks_exact(RECORD_SIZE).enumerate() {
        let prepared: PreparedQuery = match client.prepare_query(index) {
            Ok(prepared) => prepared,
            Err(ProtocolError::NoHintForIndex { .. }) => continue,
            Err(ProtocolError::NoBackupHintsLeft) => break,
            Err(other) => panic!("unexpected error at index {index}: {other}"),
        };
        let response = server.answer(&prepared.query).expect("answer");
        assert_eq!(
            client.reconstruct_and_replenish(&prepared, &response),
            Ok(expected.to_vec())
        );
        answered.push(index);
    }

    assert_eq!(answered.len(), query_limit);
    assert_eq!(client.remaining_backup_hints(), 0);
    assert_eq!(client.num_cached_entries(), query_limit);

    // After the limit, a new index and a cached index (decoy path) both stop
    // before a query is built.
    for index in 0..params.num_entries {
        assert_eq!(
            client.prepare_query(index).err(),
            Some(ProtocolError::NoBackupHintsLeft),
            "index {index}"
        );
    }
    assert_eq!(client.queries_in_flight(), 0);
    assert_eq!(client.num_cached_entries(), query_limit);
}

#[test]
fn an_abandoned_query_removes_its_hint_and_uses_no_backup_hint() {
    let params: ProtocolParams =
        ProtocolParams::new(128, RECORD_SIZE, Some(16), 4, None).expect("params");
    let entries: Vec<u8> = make_db(params.num_entries);
    let server: Server = Server::new(params.clone(), 0, &entries).expect("server init");
    let mut client: Client =
        Client::offline_init(params.clone(), [0x81u8; 32], [0x82u8; 32], 0, &entries)
            .expect("offline init");

    let (index, prepared) = prepare_first(&mut client, 0..params.num_entries);
    let abandoned_hint: usize = prepared.hint_index();
    let backups_before: usize = client.remaining_backup_hints();

    assert_eq!(client.abandon_query(prepared), Ok(()));
    assert_eq!(client.queries_in_flight(), 0);
    assert_eq!(client.num_cached_entries(), 0);
    assert_eq!(client.remaining_backup_hints(), backups_before);

    // The index is not cached, so a new query goes to the same index. It must
    // not use the hint that the server saw.
    match client.prepare_query(index) {
        Ok(second) => {
            assert_ne!(second.hint_index(), abandoned_hint);
            let response = server.answer(&second.query).expect("answer");
            assert_eq!(
                client.reconstruct_and_replenish(&second, &response),
                Ok(record(&entries, index).to_vec())
            );
        }
        Err(error) => assert_eq!(error, ProtocolError::NoHintForIndex { index }),
    }
}

#[test]
fn a_prepared_query_of_another_client_is_rejected() {
    let params: ProtocolParams =
        ProtocolParams::new(128, RECORD_SIZE, Some(16), 4, None).expect("params");
    let entries: Vec<u8> = make_db(params.num_entries);
    let server: Server = Server::new(params.clone(), 0, &entries).expect("server init");
    let mut client: Client =
        Client::offline_init(params.clone(), [0x91u8; 32], [0x92u8; 32], 0, &entries)
            .expect("offline init");
    let mut other: Client =
        Client::offline_init(params.clone(), [0x91u8; 32], [0x93u8; 32], 0, &entries)
            .expect("offline init");

    let (_, foreign_a) = prepare_first(&mut other, 0..params.num_entries);
    let (_, foreign_b) = prepare_first(&mut other, (params.num_entries / 2)..params.num_entries);
    let response = server.answer(&foreign_a.query).expect("answer");
    let backups_before: usize = client.remaining_backup_hints();

    assert_eq!(
        client.reconstruct_and_replenish(&foreign_a, &response),
        Err(ProtocolError::UnknownPreparedQuery)
    );
    assert_eq!(
        client.abandon_query(foreign_b),
        Err(ProtocolError::UnknownPreparedQuery)
    );
    assert_eq!(client.remaining_backup_hints(), backups_before);
    assert_eq!(client.num_cached_entries(), 0);
}

#[test]
fn the_server_rejects_a_malformed_query() {
    let params: ProtocolParams =
        ProtocolParams::new(128, RECORD_SIZE, Some(16), 4, None).expect("params");
    let entries: Vec<u8> = make_db(params.num_entries);
    let server: Server = Server::new(params.clone(), 0, &entries).expect("server init");
    let mut client: Client =
        Client::offline_init(params.clone(), [0xA1u8; 32], [0xA2u8; 32], 0, &entries)
            .expect("offline init");

    let (_, prepared) = prepare_first(&mut client, 0..params.num_entries);
    let valid: Query = prepared.query.clone();
    let half: usize = params.num_blocks / 2;
    assert_eq!(valid.offsets.len(), half);
    assert!(server.answer(&valid).is_ok());

    let mut short_mask: Query = valid.clone();
    short_mask.mask.clear();
    assert_eq!(
        server.answer(&short_mask),
        Err(ProtocolError::MaskLengthMismatch {
            expected_bytes: params.num_blocks.div_ceil(8),
            got: 0,
        })
    );

    let mut one_sided: Query = valid.clone();
    one_sided.mask.fill(0xFF);
    assert_eq!(
        server.answer(&one_sided),
        Err(ProtocolError::QueryShapeMismatch {
            expected_offsets: half,
            got: half,
        })
    );

    let mut short_offsets: Query = valid.clone();
    short_offsets.offsets.pop();
    assert_eq!(
        server.answer(&short_offsets),
        Err(ProtocolError::QueryShapeMismatch {
            expected_offsets: half,
            got: half - 1,
        })
    );

    let mut large_offset: Query = valid;
    large_offset.offsets[0] = params.block_size as u32;
    assert!(matches!(
        server.answer(&large_offset),
        Err(ProtocolError::OffsetOutOfRange { offset, .. }) if offset == params.block_size
    ));
}

/// Every byte of a record comes back, for sizes below one word, at a word
/// boundary, with a partial last word, and above the 32 bytes of the old format.
#[test]
fn records_of_other_sizes_are_returned_whole() {
    for record_size in [1usize, 7, 8, 33, 40, 256] {
        let params: ProtocolParams =
            ProtocolParams::new(100, record_size, Some(16), 4, None).expect("params");
        let record_of = |index: usize, salt: u8| -> Vec<u8> {
            (0..record_size)
                .map(|k| (index as u8).wrapping_mul(29) ^ (k as u8).wrapping_mul(131) ^ salt)
                .collect()
        };
        let mut database: Vec<u8> = (0..params.num_entries)
            .flat_map(|index| record_of(index, 0x11))
            .collect();
        let mut server: Server = Server::new(params.clone(), 0, &database).expect("server init");
        let mut client: Client =
            Client::offline_init(params.clone(), [0xB1u8; 32], [0xB2u8; 32], 0, &database)
                .expect("offline init");

        let (first, prepared) = prepare_first(&mut client, 0..params.num_entries);
        let response: Response = server.answer(&prepared.query).expect("answer");
        assert_eq!(response.parity_0.len(), record_size);
        assert_eq!(response.parity_1.len(), record_size);
        assert_eq!(
            client.reconstruct_and_replenish(&prepared, &response),
            Ok(record_of(first, 0x11)),
            "record size {record_size}"
        );

        let updates: Vec<(usize, Vec<u8>)> = (0..params.num_entries)
            .filter(|index| index % 2 == 0 || *index == first)
            .map(|index| (index, record_of(index, 0xE7)))
            .collect();
        let delta: Delta = server.apply_updates(&updates).expect("server update");
        assert_eq!(delta.header().record_size as usize, record_size);
        assert_eq!(delta.len(), updates.len());
        assert!(delta
            .changes()
            .all(|(_, change)| change.len() == record_size));
        client.apply_delta(&delta).expect("client update");
        for (index, value) in &updates {
            database[index * record_size..(index + 1) * record_size].copy_from_slice(value);
            assert_eq!(server.record(*index), Some(value.as_slice()));
        }

        let mut checked: usize = 0;
        for index in (first..params.num_entries).chain(0..first) {
            if checked == 6 {
                break;
            }
            let Ok(prepared) = client.prepare_query(index) else {
                continue;
            };
            let response: Response = server.answer(&prepared.query).expect("answer");
            let got: Vec<u8> = client
                .reconstruct_and_replenish(&prepared, &response)
                .expect("reconstruct");
            assert_eq!(
                got,
                &database[index * record_size..(index + 1) * record_size],
                "record size {record_size}, index {index}"
            );
            checked += 1;
        }
        assert_eq!(checked, 6, "record size {record_size}");
    }
}

/// The padded database has `num_blocks * block_size * record_size` bytes.
#[test]
fn parameters_of_a_database_that_does_not_fit_in_memory_are_rejected() {
    // 2^32 on a 64-bit target: the largest block size.
    let block: usize = 1usize << (usize::BITS / 2);
    for (num_entries, record_size, block_size) in [
        // The records fit in `usize`; the padded blocks of 4-byte records
        // take `isize::MAX + 1` bytes.
        ((1usize << (usize::BITS - 3)) - 1, 4usize, block),
        // An odd block count at the limit of `usize`.
        (usize::MAX, 1, 1),
        // One byte above `isize::MAX` after the block count becomes even.
        (usize::MAX / 2, 1, 1),
    ] {
        assert_eq!(
            ProtocolParams::new(num_entries, record_size, Some(block_size), 1, Some(1)).err(),
            Some(ProtocolError::InvalidParams(
                "the padded database does not fit in memory"
            )),
            "num_entries {num_entries}, record_size {record_size}, block_size {block_size}"
        );
    }
    // The largest padded database: `isize::MAX - 1` bytes in an even block count.
    let params: ProtocolParams =
        ProtocolParams::new(usize::MAX / 2 - 1, 1, Some(1), 1, Some(1)).expect("params");
    assert_eq!(params.n_effective_entries(), usize::MAX / 2 - 1);
}

/// The block size is the range of an iPRF, which is at most 2^32.
#[test]
#[cfg(target_pointer_width = "64")]
fn a_block_size_above_the_iprf_range_is_rejected() {
    assert_eq!(
        ProtocolParams::new(1, 1, Some(1 << 33), 1, Some(1)).err(),
        Some(ProtocolError::InvalidParams(
            "block_size must be a power of two of at most 2^32"
        ))
    );
    let params: ProtocolParams =
        ProtocolParams::new(1, 1, Some(1 << 32), 1, Some(1)).expect("params");
    assert_eq!(params.num_blocks, 2);
}

/// The fields of the parameters are public. A changed block size or block
/// count that does not hold all records must give no server and no hints:
/// the last records would be in no block.
#[test]
fn parameters_that_new_does_not_derive_are_rejected() {
    let params: ProtocolParams =
        ProtocolParams::new(96, RECORD_SIZE, Some(16), 4, None).expect("params");
    assert_eq!(params.num_blocks, 6);
    let entries: Vec<u8> = make_db(params.num_entries);
    let changed: [ProtocolParams; 4] = [
        ProtocolParams {
            block_size: 8,
            ..params.clone()
        },
        ProtocolParams {
            num_blocks: 4,
            ..params.clone()
        },
        ProtocolParams {
            num_total_hints: params.num_regular_hints,
            ..params.clone()
        },
        ProtocolParams {
            num_regular_hints: params.num_regular_hints + 1,
            ..params.clone()
        },
    ];
    for (case, invalid) in changed.iter().enumerate() {
        assert!(
            matches!(
                Server::new(invalid.clone(), 0, &entries),
                Err(ProtocolError::InvalidParams(_))
            ),
            "case {case}: server"
        );
        assert!(
            matches!(
                compute_hints(invalid, &[0xC1u8; 32], 0, &entries),
                Err(ProtocolError::InvalidParams(_))
            ),
            "case {case}: hints"
        );
        assert!(
            matches!(
                Client::offline_init(invalid.clone(), [0xC1u8; 32], [0xC2u8; 32], 0, &entries),
                Err(ProtocolError::InvalidParams(_))
            ),
            "case {case}: client"
        );
    }

    let hints: HintSet = compute_hints(&params, &[0xC1u8; 32], 0, &entries).expect("hints");
    assert!(Client::from_hints(params, [0xC1u8; 32], [0xC2u8; 32], hints).is_ok());
}

#[test]
fn values_of_another_record_size_are_rejected() {
    let params: ProtocolParams =
        ProtocolParams::new(128, RECORD_SIZE, Some(16), 4, None).expect("params");
    let entries: Vec<u8> = make_db(params.num_entries);

    assert!(matches!(
        Server::new(params.clone(), 0, &entries[1..]),
        Err(ProtocolError::InvalidParams(_))
    ));
    assert!(matches!(
        Client::offline_init(params.clone(), [0xC1u8; 32], [0xC2u8; 32], 0, &entries[1..]),
        Err(ProtocolError::InvalidParams(_))
    ));
    for record_size in [0usize, (1 << 20) + 1] {
        assert!(matches!(
            ProtocolParams::new(128, record_size, Some(16), 4, None),
            Err(ProtocolError::InvalidParams(_))
        ));
    }

    // The fields of the parameters are public: a record size outside the
    // range must not give a server.
    for record_size in [0usize, (1 << 20) + 1] {
        let invalid = ProtocolParams {
            record_size,
            ..params.clone()
        };
        let database: Vec<u8> = vec![0u8; params.num_entries * record_size];
        assert!(matches!(
            Server::new(invalid, 0, &database),
            Err(ProtocolError::InvalidParams(_))
        ));
    }

    let mut server: Server = Server::new(params.clone(), 0, &entries).expect("server init");
    let mut client: Client =
        Client::offline_init(params.clone(), [0xC1u8; 32], [0xC2u8; 32], 0, &entries)
            .expect("offline init");
    let mismatch = ProtocolError::RecordSizeMismatch {
        expected: RECORD_SIZE,
        got: RECORD_SIZE - 1,
    };

    // The server changes no record when one value of a batch has a wrong size.
    let short: Vec<u8> = vec![0xFFu8; RECORD_SIZE - 1];
    assert_eq!(
        server.apply_updates(&[(3, make_entry(0x77)), (4, short.clone())]),
        Err(mismatch.clone())
    );
    assert_eq!(server.record(3), Some(record(&entries, 3)));
    assert_eq!(server.record(params.num_entries), None);

    assert_eq!(
        server.database_version(),
        0,
        "a failed update makes no version"
    );

    // The client changes no hint for a delta of another record size or of
    // another database size: a query after it still gives the record of the
    // database.
    let header = DeltaHeader {
        num_entries: params.num_entries as u64,
        record_size: RECORD_SIZE as u32,
        from_version: 0,
        to_version: 1,
    };
    let other_record_size: Delta = Delta::new(
        DeltaHeader {
            record_size: RECORD_SIZE as u32 - 1,
            ..header
        },
        vec![(0, short.clone())],
    )
    .expect("delta");
    assert_eq!(
        client.apply_delta(&other_record_size),
        Err(ProtocolError::DeltaMismatch("record size"))
    );
    let all_records: Vec<(u64, Vec<u8>)> = (0..params.num_entries as u64)
        .map(|index| (index, vec![0xFFu8; RECORD_SIZE]))
        .collect();
    let other_database_size: Delta = Delta::new(
        DeltaHeader {
            num_entries: params.num_entries as u64 + 1,
            ..header
        },
        all_records,
    )
    .expect("delta");
    assert_eq!(
        client.apply_delta(&other_database_size),
        Err(ProtocolError::DeltaMismatch("number of records"))
    );
    assert_eq!(client.database_version(), 0);
    let (index, prepared) = prepare_first(&mut client, 0..params.num_entries);
    let response: Response = server.answer(&prepared.query).expect("answer");
    assert_eq!(
        client.reconstruct_and_replenish(&prepared, &response),
        Ok(record(&entries, index).to_vec())
    );

    // A response of another size is not an answer. The query stays in
    // flight, and the client accepts the answer to a retry.
    let (index, prepared) = prepare_first(&mut client, (index + 1)..params.num_entries);
    let backups_before: usize = client.remaining_backup_hints();
    let cached_before: usize = client.num_cached_entries();
    let response: Response = server.answer(&prepared.query).expect("answer");
    let mut short_first: Response = response.clone();
    short_first.parity_0.pop();
    let mut short_second: Response = response.clone();
    short_second.parity_1.pop();
    for malformed in [short_first, short_second] {
        assert_eq!(
            client.reconstruct_and_replenish(&prepared, &malformed),
            Err(mismatch.clone())
        );
        assert_eq!(client.queries_in_flight(), 1);
        assert_eq!(client.remaining_backup_hints(), backups_before);
        assert_eq!(client.num_cached_entries(), cached_before);
    }
    assert_eq!(
        client.reconstruct_and_replenish(&prepared, &response),
        Ok(record(&entries, index).to_vec())
    );
    assert_eq!(client.queries_in_flight(), 0);
    // An accepted answer ends the query: the same value is not accepted again.
    assert_eq!(
        client.reconstruct_and_replenish(&prepared, &response),
        Err(ProtocolError::UnknownPreparedQuery)
    );
}

/// Regular parities, backup "in" parities and backup "out" parities.
fn parities_of(hints: &HintSet) -> Vec<Vec<u8>> {
    (0..hints.num_regular())
        .map(|j| hints.regular_parity(j).to_vec())
        .chain((0..hints.num_backup()).map(|b| hints.backup_parity_in(b).to_vec()))
        .chain((0..hints.num_backup()).map(|b| hints.backup_parity_out(b).to_vec()))
        .collect()
}

/// The client must accept an answer only from the database version of its
/// hints. The rejected answer here would give a wrong record: the record
/// changed between the two versions.
#[test]
fn an_answer_of_a_later_version_is_accepted_after_the_delta() {
    let params: ProtocolParams =
        ProtocolParams::new(128, RECORD_SIZE, Some(16), 4, None).expect("params");
    let mut entries: Vec<u8> = make_db(params.num_entries);
    let mut server: Server = Server::new(params.clone(), 41, &entries).expect("server init");
    let mut client: Client =
        Client::offline_init(params.clone(), [0xE1u8; 32], [0xE2u8; 32], 41, &entries)
            .expect("offline init");
    assert_eq!(
        (server.database_version(), client.database_version()),
        (41, 41)
    );

    let (index, prepared) = prepare_first(&mut client, 0..params.num_entries);
    let old_value: Vec<u8> = record(&entries, index).to_vec();
    let new_value: Vec<u8> = make_entry(0xC0FFEE);
    // Two writes of one record in one call give one change: first value
    // XOR last value.
    let delta: Delta = server
        .apply_updates(&[(index, make_entry(0xBAD)), (index, new_value.clone())])
        .expect("server update");
    set_record(&mut entries, index, &new_value);
    assert_eq!(server.database_version(), 42);
    assert_eq!(delta.len(), 1);
    assert_eq!(
        delta.header(),
        DeltaHeader {
            num_entries: 128,
            record_size: RECORD_SIZE as u32,
            from_version: 41,
            to_version: 42,
        }
    );

    let response: Response = server.answer(&prepared.query).expect("answer");
    assert_eq!(response.database_version, 42);
    let backups_before: usize = client.remaining_backup_hints();
    for _ in 0..2 {
        assert_eq!(
            client.reconstruct_and_replenish(&prepared, &response),
            Err(ProtocolError::DatabaseVersionMismatch {
                expected: 41,
                got: 42
            })
        );
    }
    assert_eq!(client.queries_in_flight(), 1);
    assert_eq!(client.remaining_backup_hints(), backups_before);
    assert_eq!(client.num_cached_entries(), 0);

    // The client catches up, and the answer that it has is then valid.
    client.apply_delta(&delta).expect("client update");
    assert_eq!(client.database_version(), 42);
    let got: Vec<u8> = client
        .reconstruct_and_replenish(&prepared, &response)
        .expect("reconstruct");
    assert_eq!(got, new_value);
    assert_ne!(got, old_value);
    assert_eq!(client.queries_in_flight(), 0);
    assert_eq!(client.remaining_backup_hints(), backups_before - 1);
}

/// A delta must apply only to the version that it starts from. A delta that
/// repeats, or one that comes before its predecessor, must change no hint.
#[test]
fn deltas_apply_in_version_order_and_one_time_only() {
    // 120 records in 8 blocks of 16: the last block is partly padding.
    let params: ProtocolParams =
        ProtocolParams::new(120, RECORD_SIZE, Some(16), 4, None).expect("params");
    let master_seed: [u8; 32] = [0xF1u8; 32];
    let mut entries: Vec<u8> = make_db(params.num_entries);
    let mut server: Server = Server::new(params.clone(), 5, &entries).expect("server init");
    let mut client: Client =
        Client::offline_init(params.clone(), master_seed, [0xF2u8; 32], 5, &entries)
            .expect("offline init");

    // Three deltas. Record 7 changes in each of them, and record 119 is the
    // last record of the database. The third delta has no change.
    let batches: [Vec<usize>; 3] = [
        vec![7, 0, 33, 119],
        (0..params.num_entries).filter(|i| i % 5 == 2).collect(),
        Vec::new(),
    ];
    let mut deltas: Vec<Delta> = Vec::new();
    for (round, batch) in batches.iter().enumerate() {
        let updates: Vec<(usize, Vec<u8>)> = batch
            .iter()
            .map(|&index| {
                (
                    index,
                    make_entry(0x5000 + 0x100 * round as u64 + index as u64),
                )
            })
            .collect();
        deltas.push(server.apply_updates(&updates).expect("server update"));
        for (index, value) in &updates {
            set_record(&mut entries, *index, value);
        }
    }
    assert_eq!(server.database_version(), 8);
    assert_eq!(deltas[0].len(), 4);
    assert!(deltas[2].is_empty());
    assert_eq!(
        deltas
            .iter()
            .map(|d| d.header().to_version)
            .collect::<Vec<u64>>(),
        [6, 7, 8]
    );

    let version_error =
        |expected: u64, got: u64| Err(ProtocolError::DatabaseVersionMismatch { expected, got });
    let hints_of = |client: &Client| client.unused_hints().expect("no query yet").clone();

    // Out of order: the second and the third delta before the first.
    let at_5: HintSet = hints_of(&client);
    assert_eq!(client.apply_delta(&deltas[1]), version_error(5, 6));
    assert_eq!(client.apply_delta(&deltas[2]), version_error(5, 7));
    assert!(
        hints_of(&client) == at_5,
        "a rejected delta changed the hints"
    );

    client.apply_delta(&deltas[0]).expect("first delta");
    assert_eq!(client.database_version(), 6);
    let at_6: HintSet = hints_of(&client);
    assert!(parities_of(&at_6) != parities_of(&at_5));
    assert_eq!(at_6.database_version(), 6);
    assert_eq!(at_6.init_database(), at_5.init_database());
    assert_eq!(at_6.init_database().version, 5);

    // A repeat, and a gap.
    assert_eq!(client.apply_delta(&deltas[0]), version_error(6, 5));
    assert_eq!(client.apply_delta(&deltas[2]), version_error(6, 7));
    assert!(
        hints_of(&client) == at_6,
        "a rejected delta changed the hints"
    );

    client.apply_delta(&deltas[1]).expect("second delta");
    let at_7: HintSet = hints_of(&client);
    client.apply_delta(&deltas[2]).expect("third delta");
    assert_eq!(client.database_version(), 8);
    assert!(
        parities_of(&hints_of(&client)) == parities_of(&at_7),
        "a delta with no change must change no parity"
    );
    assert_eq!(client.apply_delta(&deltas[2]), version_error(8, 7));

    // One delta from version 5 to version 7 with the changes of the first
    // two deltas. Record 7 changes in both.
    let combined: Delta = Delta::new(
        DeltaHeader {
            from_version: 5,
            to_version: 7,
            ..deltas[0].header()
        },
        deltas[..2]
            .iter()
            .flat_map(|delta| delta.changes())
            .map(|(index, change)| (index, change.to_vec()))
            .collect(),
    )
    .expect("combined delta");
    assert!(combined.len() < deltas[0].len() + deltas[1].len());
    let mut skipper: Client =
        Client::from_hints(params.clone(), master_seed, [0xF3u8; 32], at_5).expect("client");
    skipper
        .apply_delta(&combined)
        .expect("delta over two versions");
    assert_eq!(skipper.database_version(), 7);
    assert!(parities_of(&hints_of(&skipper)) == parities_of(&at_7));

    // The hints after the deltas are the hints of HintInit on the new database.
    let fresh: HintSet = compute_hints(&params, &master_seed, 8, &entries).expect("hints");
    assert!(
        parities_of(&hints_of(&client)) == parities_of(&fresh),
        "the parities after the deltas differ from HintInit on the new database"
    );

    // Changed and unchanged records, from the server at version 8.
    let mut checked: usize = 0;
    for index in [7usize, 119, 2, 33, 1, 118].into_iter().chain(40..60) {
        if checked == 8 {
            break;
        }
        let Ok(prepared) = client.prepare_query(index) else {
            continue;
        };
        assert!(client.unused_hints().is_none(), "the hint set is in use");
        let response: Response = server.answer(&prepared.query).expect("answer");
        assert_eq!(
            client.reconstruct_and_replenish(&prepared, &response),
            Ok(record(&entries, index).to_vec()),
            "index {index}"
        );
        checked += 1;
    }
    assert_eq!(checked, 8);
}

/// A record in the padding of the last block counts as a zero record. So a
/// database with fewer records and the same blocks has the same hints as the
/// database with zero records in their place (`docs/update_strategy.md`,
/// "A Database That Grows").
#[test]
fn padding_records_count_as_zero_records() {
    // 97 and 120 records both take 8 blocks of 16.
    let short_params: ProtocolParams =
        ProtocolParams::new(97, RECORD_SIZE, Some(16), 4, None).expect("params");
    let long_params: ProtocolParams =
        ProtocolParams::new(120, RECORD_SIZE, Some(16), 4, None).expect("params");
    assert_eq!(short_params.num_blocks, long_params.num_blocks);
    let master_seed: [u8; 32] = [0xD7u8; 32];

    let short_db: Vec<u8> = make_db(97);
    let mut long_db: Vec<u8> = short_db.clone();
    long_db.resize(120 * RECORD_SIZE, 0u8);

    let short_hints: HintSet =
        compute_hints(&short_params, &master_seed, 0, &short_db).expect("hints");
    let long_hints: HintSet =
        compute_hints(&long_params, &master_seed, 0, &long_db).expect("hints");
    assert_eq!(short_hints.regular_cutoffs(), long_hints.regular_cutoffs());
    assert_eq!(short_hints.backup_cutoffs(), long_hints.backup_cutoffs());
    assert!(parities_of(&short_hints) == parities_of(&long_hints));

    // With other bytes in records 97 to 119 the parities differ.
    let full_db: Vec<u8> = make_db(120);
    let full_hints: HintSet =
        compute_hints(&long_params, &master_seed, 0, &full_db).expect("hints");
    assert!(parities_of(&full_hints) != parities_of(&long_hints));
}

/// The hint set must not depend on the number of worker threads.
#[test]
fn hint_sets_are_equal_for_every_thread_count() {
    // 120 records in 8 blocks of 16: the last block is partly padding.
    let params: ProtocolParams =
        ProtocolParams::new(120, RECORD_SIZE, Some(16), 4, None).expect("params");
    assert_eq!(params.num_blocks, 8);
    let entries: Vec<u8> = make_db(params.num_entries);
    let master_seed: [u8; 32] = [0xD1u8; 32];

    let one_thread: HintSet =
        compute_hints_with_threads(&params, &master_seed, 0, &entries, 1).expect("1 thread");
    assert!(
        (0..one_thread.num_backup()).any(|b| one_thread
            .backup_parity_out(b)
            .iter()
            .any(|&byte| byte != 0)),
        "the scenario must give non-zero parities"
    );
    // 0 counts as 1; 3 is fewer than the blocks and 11 is more.
    for threads in [0usize, 3, 11] {
        let hints: HintSet =
            compute_hints_with_threads(&params, &master_seed, 0, &entries, threads).expect("hints");
        assert!(hints == one_thread, "{threads} threads");
    }
    let all_cores: HintSet = compute_hints(&params, &master_seed, 0, &entries).expect("all cores");
    assert!(all_cores == one_thread, "all available cores");
}

#[test]
fn block_keys_are_distinct_and_depend_on_the_seed() {
    let keys: HashSet<[u8; 16]> = derive_block_keys(&[1u8; 32], 100).into_iter().collect();
    assert_eq!(keys.len(), 100);
    assert_eq!(
        derive_block_keys(&[1u8; 32], 3),
        derive_block_keys(&[1u8; 32], 100)[..3]
    );
    assert_ne!(
        derive_block_keys(&[1u8; 32], 3),
        derive_block_keys(&[2u8; 32], 3)
    );
}

fn state_of(client: &Client) -> Vec<u8> {
    let mut bytes: Vec<u8> = Vec::new();
    client.write_state(&mut bytes).expect("write state");
    bytes
}

/// A hint that the server saw must not go out in a second query, also when
/// the client starts again. The test has three starts of one client: from
/// the newest state, from an old state with no log (the control: hints go
/// out a second time), and from an old state with the log of hints.
#[test]
fn a_restarted_client_does_not_send_a_hint_a_second_time() {
    let params: ProtocolParams =
        ProtocolParams::new(128, RECORD_SIZE, Some(16), 8, None).expect("params");
    let entries: Vec<u8> = make_db(params.num_entries);
    let server: Server = Server::new(params.clone(), 3, &entries).expect("server init");
    let master_seed: [u8; 32] = [0xF1u8; 32];
    let mut client: Client =
        Client::offline_init(params.clone(), master_seed, [0xF2u8; 32], 3, &entries)
            .expect("offline init");

    // Hints that the server saw.
    let mut sent: HashSet<usize> = HashSet::new();

    // First run: the client writes its state before each query leaves it.
    let mut state: Vec<u8> = Vec::new();
    let mut last: Option<(usize, Query)> = None;
    for index in 0..30 {
        let Ok(prepared) = client.prepare_query(index) else {
            continue;
        };
        state = state_of(&client);
        assert!(sent.insert(prepared.hint_index()), "index {index}");
        let response: Response = server.answer(&prepared.query).expect("answer");
        assert_eq!(
            client.reconstruct_and_replenish(&prepared, &response),
            Ok(record(&entries, index).to_vec())
        );
        last = Some((index, prepared.query.clone()));
    }
    let (last_index, last_query) = last.expect("a query");
    let answered_before: usize = client.num_cached_entries();
    assert!(answered_before >= 20, "{answered_before} answered queries");
    drop(client);

    // The newest state has the last query in flight: its answer came after
    // the write. The client gets the same message again, not a new hint.
    let (mut client, in_flight) =
        Client::read_state(master_seed, [0xF3u8; 32], state.as_slice()).expect("read state");
    assert_eq!(client.database_version(), 3);
    assert_eq!(client.num_cached_entries(), answered_before - 1);
    assert_eq!(in_flight.len(), 1);
    assert!(in_flight[0].query == last_query);
    assert!(sent.contains(&in_flight[0].hint_index()));
    let response: Response = server.answer(&in_flight[0].query).expect("answer");
    assert_eq!(
        client.reconstruct_and_replenish(&in_flight[0], &response),
        Ok(record(&entries, last_index).to_vec())
    );

    // Second run: one state, then a log of the hints of the later queries.
    let old_state: Vec<u8> = state_of(&client);
    let mut log: Vec<usize> = Vec::new();
    for index in 30..60 {
        let Ok(prepared) = client.prepare_query(index) else {
            continue;
        };
        log.push(prepared.hint_index());
        assert!(sent.insert(prepared.hint_index()), "index {index}");
        let response: Response = server.answer(&prepared.query).expect("answer");
        assert_eq!(
            client.reconstruct_and_replenish(&prepared, &response),
            Ok(record(&entries, index).to_vec())
        );
    }
    assert!(log.len() >= 20, "{} logged queries", log.len());
    drop(client);

    // Control: the old state with no log. The client does not know the later
    // queries, and it sends their hints again.
    let (mut rolled_back, _) =
        Client::read_state(master_seed, [0xF4u8; 32], old_state.as_slice()).expect("read state");
    let sent_again: usize = (30..60)
        .filter_map(|index| rolled_back.prepare_query(index).ok())
        .filter(|prepared| sent.contains(&prepared.hint_index()))
        .count();
    assert!(
        sent_again > 0,
        "the control found no hint that went out again"
    );

    // The old state with the log: no hint goes out a second time, until the
    // client has no backup hint left.
    let (mut client, in_flight) =
        Client::read_state(master_seed, [0xF5u8; 32], old_state.as_slice()).expect("read state");
    assert!(in_flight.is_empty());
    assert_eq!(
        client.retire_hints(&[log[0], params.num_total_hints]),
        Err(ProtocolError::InvalidParams(
            "a hint to retire is not a hint of this client"
        ))
    );
    assert_eq!(
        state_of(&client),
        old_state,
        "a rejected list changes nothing"
    );
    assert_eq!(client.retire_hints(&log), Ok(()));
    assert_eq!(client.retire_hints(&log), Ok(()));
    assert!(client.unused_hints().is_none());

    let mut queries: usize = 0;
    // The bound ends the loop if no index has a hint while backup hints remain.
    for index in (0..params.num_entries)
        .cycle()
        .take(20 * params.num_entries)
    {
        let prepared: PreparedQuery = match client.prepare_query(index) {
            Ok(prepared) => prepared,
            Err(ProtocolError::NoHintForIndex { .. }) => continue,
            Err(ProtocolError::NoBackupHintsLeft) => break,
            Err(other) => panic!("index {index}: {other}"),
        };
        assert!(
            sent.insert(prepared.hint_index()),
            "index {index}: hint {} went out a second time",
            prepared.hint_index()
        );
        let response: Response = server.answer(&prepared.query).expect("answer");
        assert_eq!(
            client.reconstruct_and_replenish(&prepared, &response),
            Ok(record(&entries, index).to_vec())
        );
        queries += 1;
    }
    assert!(queries >= 60, "{queries} queries after the restart");
    assert_eq!(client.remaining_backup_hints(), 0);
}

/// `retire_hints` can remove a backup hint that a query in flight counted
/// on. The query that then has no backup hint must stay in flight, with its
/// hint, and the caller can abandon it.
#[test]
fn a_query_with_no_backup_hint_after_retire_hints_stays_in_flight() {
    let params: ProtocolParams =
        ProtocolParams::new(128, RECORD_SIZE, Some(16), 4, Some(2)).expect("params");
    let entries: Vec<u8> = make_db(params.num_entries);
    let server: Server = Server::new(params.clone(), 0, &entries).expect("server init");
    let master_seed: [u8; 32] = [0xF6u8; 32];
    let mut client: Client =
        Client::offline_init(params.clone(), master_seed, [0xF7u8; 32], 0, &entries)
            .expect("offline init");

    let (index_a, prepared_a) = prepare_first(&mut client, 0..params.num_entries);
    let (_, prepared_b) = prepare_first(&mut client, (index_a + 1)..params.num_entries);
    assert_eq!(client.retire_hints(&[params.num_regular_hints + 1]), Ok(()));
    assert_eq!(client.remaining_backup_hints(), 1);
    assert_eq!(client.queries_in_flight(), 2);

    // A state with more queries in flight than backup hints is valid.
    let state: Vec<u8> = state_of(&client);
    let (restored, in_flight) =
        Client::read_state(master_seed, [0xF8u8; 32], state.as_slice()).expect("read state");
    assert_eq!(in_flight.len(), 2);
    assert_eq!(restored.remaining_backup_hints(), 1);
    assert_eq!(state_of(&restored), state);

    let response_a: Response = server.answer(&prepared_a.query).expect("answer a");
    let response_b: Response = server.answer(&prepared_b.query).expect("answer b");
    assert_eq!(
        client.reconstruct_and_replenish(&prepared_a, &response_a),
        Ok(record(&entries, index_a).to_vec())
    );
    let before: Vec<u8> = state_of(&client);
    assert_eq!(
        client.reconstruct_and_replenish(&prepared_b, &response_b),
        Err(ProtocolError::NoBackupHintsLeft)
    );
    assert_eq!(state_of(&client), before, "the error changes no state");
    assert_eq!(client.queries_in_flight(), 1);
    assert_eq!(client.abandon_query(prepared_b), Ok(()));
    assert_eq!(client.queries_in_flight(), 0);
    assert_eq!(client.num_cached_entries(), 1);
}

/// The log can hold the hint of a query that is in flight in the state. The
/// query must end, as with `abandon_query`, and the state must stay valid.
#[test]
fn retire_hints_ends_a_query_in_flight_that_uses_the_hint() {
    let params: ProtocolParams =
        ProtocolParams::new(128, RECORD_SIZE, Some(16), 4, None).expect("params");
    let entries: Vec<u8> = make_db(params.num_entries);
    let server: Server = Server::new(params.clone(), 0, &entries).expect("server init");
    let master_seed: [u8; 32] = [0xF9u8; 32];
    let mut client: Client =
        Client::offline_init(params.clone(), master_seed, [0xFAu8; 32], 0, &entries)
            .expect("offline init");

    let (index, retired) = prepare_first(&mut client, 0..params.num_entries);
    let (other_index, other) = prepare_first(&mut client, (index + 1)..params.num_entries);
    let backups_before: usize = client.remaining_backup_hints();
    assert_eq!(client.retire_hints(&[retired.hint_index()]), Ok(()));
    assert_eq!(client.queries_in_flight(), 1);
    assert_eq!(client.remaining_backup_hints(), backups_before);

    let response: Response = server.answer(&retired.query).expect("answer");
    assert_eq!(
        client.reconstruct_and_replenish(&retired, &response),
        Err(ProtocolError::UnknownPreparedQuery)
    );
    assert_eq!(client.num_cached_entries(), 0);

    let state: Vec<u8> = state_of(&client);
    let (_, in_flight) =
        Client::read_state(master_seed, [0xFBu8; 32], state.as_slice()).expect("read state");
    assert_eq!(in_flight.len(), 1);
    assert_eq!(in_flight[0].hint_index(), other.hint_index());
    assert!(in_flight[0].query == other.query);

    // The index of the retired query is free for a query with another hint.
    match client.prepare_query(index) {
        Ok(second) => assert_ne!(second.hint_index(), retired.hint_index()),
        Err(error) => assert_eq!(error, ProtocolError::NoHintForIndex { index }),
    }
    let response: Response = server.answer(&other.query).expect("answer");
    assert_eq!(
        client.reconstruct_and_replenish(&other, &response),
        Ok(record(&entries, other_index).to_vec())
    );
}

/// The library gives no integrity (`docs/integrity.md`): a changed parity
/// gives a wrong record with no error. Two properties must hold then. The
/// error stays in the hints that hold it, with no other damage. The messages
/// of the client do not depend on the bytes of an answer, so a server gets
/// no data from the messages after a wrong answer.
#[test]
fn a_wrong_answer_gives_a_wrong_record_and_the_same_later_messages() {
    let params: ProtocolParams =
        ProtocolParams::new(64, RECORD_SIZE, Some(8), 6, None).expect("params");
    let entries: Vec<u8> = make_db(params.num_entries);
    let server: Server = Server::new(params.clone(), 0, &entries).expect("server init");
    let new_client = || -> Client {
        Client::offline_init(params.clone(), [0x03u8; 32], [0x77u8; 32], 0, &entries)
            .expect("offline init")
    };
    let mut honest: Client = new_client();
    let mut cheated: Client = new_client();

    // Hints of the cheated client with a parity that holds the error.
    let mut wrong_hints: HashSet<usize> = HashSet::new();
    let mut wrong_records: usize = 0;
    let mut queries: usize = 0;
    for index in (0..params.num_entries).map(|k| (k * 37) % params.num_entries) {
        let (expected, prepared) = match (honest.prepare_query(index), cheated.prepare_query(index))
        {
            (Ok(expected), Ok(prepared)) => (expected, prepared),
            (Err(expected), Err(error)) => {
                assert_eq!(error, expected, "index {index}");
                continue;
            }
            _ => panic!("index {index}: one client has a query and the other has an error"),
        };
        assert!(
            prepared.query == expected.query,
            "message for index {index}"
        );
        assert_eq!(prepared.hint_index(), expected.hint_index());

        let good: Response = server.answer(&prepared.query).expect("answer");
        let mut bad: Response = good.clone();
        let first: bool = queries == 0;
        if first {
            bad.parity_0[3] ^= 0x10;
            bad.parity_1[3] ^= 0x10;
        }
        let correct: Vec<u8> = honest
            .reconstruct_and_replenish(&expected, &good)
            .expect("reconstruct");
        assert_eq!(correct, record(&entries, index));
        let got: Vec<u8> = cheated
            .reconstruct_and_replenish(&prepared, &bad)
            .expect("a wrong answer gives no error");
        queries += 1;

        let is_wrong: bool = first || wrong_hints.contains(&prepared.hint_index());
        let mut want: Vec<u8> = correct;
        if is_wrong {
            want[3] ^= 0x10;
            wrong_records += 1;
            // `Recon` promotes the backup hints in the order of their numbers.
            wrong_hints.insert(params.num_regular_hints + cheated.num_cached_entries() - 1);
        }
        assert_eq!(got, want, "index {index}, query {queries}");
    }

    assert!(queries >= 40, "{queries} queries");
    assert_eq!(wrong_records, 3, "the error spreads through promoted hints");
}

#[test]
fn parallel_query_and_delta_match_single_thread() {
    // 144 blocks -> 72 real blocks per query (>= PARALLEL_QUERY_MIN_BLOCKS = 64),
    // exercising the multi-threaded path of `prepare_query`.
    let num_entries: usize = 144 * 4;
    let params: ProtocolParams =
        ProtocolParams::new(num_entries, RECORD_SIZE, Some(4), 16, Some(32)).expect("params");
    assert_eq!(params.num_blocks, 144);

    let db: Vec<u8> = make_db(num_entries);
    let master_seed: [u8; 32] = [0x31u8; 32];
    let query_seed: [u8; 32] = [0x74u8; 32];
    let hints: HintSet = compute_hints(&params, &master_seed, 0, &db).expect("hints");

    let mut server: Server = Server::new(params.clone(), 0, &db).expect("server");
    let mut client_1t: Client =
        Client::from_hints(params.clone(), master_seed, query_seed, hints.clone()).expect("1t");
    let mut client_4t: Client =
        Client::from_hints(params.clone(), master_seed, query_seed, hints).expect("4t");
    client_1t.set_threads(1);
    client_4t.set_threads(4);
    assert_eq!(client_1t.threads(), 1);
    assert_eq!(client_4t.threads(), 4);

    // Run real queries and decoy queries (repeated indices) before the delta.
    for index in [0usize, 5, 17, 5, 100, 311, 17, 575] {
        let p1: PreparedQuery = client_1t.prepare_query(index).expect("prepare 1t");
        let p4: PreparedQuery = client_4t.prepare_query(index).expect("prepare 4t");
        assert_eq!(p1.query, p4.query, "query mismatch for index {index}");
        assert_eq!(p1.hint_index(), p4.hint_index());

        let resp: Response = server.answer(&p1.query).expect("answer");
        let v1: Vec<u8> = client_1t
            .reconstruct_and_replenish(&p1, &resp)
            .expect("recon 1t");
        let v4: Vec<u8> = client_4t
            .reconstruct_and_replenish(&p4, &resp)
            .expect("recon 4t");
        assert_eq!(v1, record(&db, index));
        assert_eq!(v1, v4);
    }

    // Apply a 270-entry delta (> PARALLEL_DELTA_BATCH_SIZE = 256) to exercise
    // multi-batch parallel `apply_delta`, including cached indices (5, 17).
    let updates: Vec<(usize, Vec<u8>)> = (0..270usize)
        .map(|i| {
            let idx: usize = match i {
                0 => 5,
                1 => 17,
                _ => (i * 2) % num_entries,
            };
            (idx, make_entry(10_000 + i as u64))
        })
        .collect();
    let delta: Delta = server.apply_updates(&updates).expect("apply updates");

    client_1t.apply_delta(&delta).expect("delta 1t");
    client_4t.apply_delta(&delta).expect("delta 4t");

    let mut state_1t: Vec<u8> = Vec::new();
    let mut state_4t: Vec<u8> = Vec::new();
    client_1t.write_state(&mut state_1t).expect("write 1t");
    client_4t.write_state(&mut state_4t).expect("write 4t");
    assert_eq!(
        state_1t, state_4t,
        "client states must be byte-identical after parallel queries and delta"
    );

    // Query updated records (both cached and uncached) after the delta.
    for &(idx, ref expected) in &updates[..6] {
        let p1: PreparedQuery = client_1t.prepare_query(idx).expect("post-delta 1t");
        let p4: PreparedQuery = client_4t.prepare_query(idx).expect("post-delta 4t");
        assert_eq!(p1.query, p4.query);
        let resp: Response = server.answer(&p1.query).expect("post-delta answer");
        let v1: Vec<u8> = client_1t
            .reconstruct_and_replenish(&p1, &resp)
            .expect("post-delta recon 1t");
        let v4: Vec<u8> = client_4t
            .reconstruct_and_replenish(&p4, &resp)
            .expect("post-delta recon 4t");
        assert_eq!(&v1, expected);
        assert_eq!(v1, v4);
    }
}

#[test]
fn backup_availability_tracks_promotions_and_retirements() {
    let num_entries: usize = 64;
    let params: ProtocolParams =
        ProtocolParams::new(num_entries, RECORD_SIZE, Some(8), 16, Some(4)).expect("params");
    let db: Vec<u8> = make_db(num_entries);
    let master_seed: [u8; 32] = [0x44u8; 32];
    let query_seed: [u8; 32] = [0x55u8; 32];
    let hints: HintSet = compute_hints(&params, &master_seed, 0, &db).expect("hints");

    let server: Server = Server::new(params.clone(), 0, &db).expect("server");
    let mut client: Client =
        Client::from_hints(params.clone(), master_seed, query_seed, hints).expect("client");

    let initial_backups: usize = client.remaining_backup_hints();
    assert_eq!(initial_backups, 4);

    let first_backup_id: usize = params.num_regular_hints;
    // Retire a middle backup (backup index 1) and the lowest backup (backup index 0).
    client
        .retire_hints(&[first_backup_id + 1])
        .expect("retire middle backup");
    assert_eq!(client.remaining_backup_hints(), 3);

    // Retiring the same backup a second time must be a no-op (no underflow).
    client
        .retire_hints(&[first_backup_id + 1])
        .expect("retire already-retired backup");
    assert_eq!(client.remaining_backup_hints(), 3);

    // First query promotes backup 0; next_backup_hint must then skip retired backup 1 to 2.
    let p0: PreparedQuery = client.prepare_query(0).expect("query 0");
    let r0: Response = server.answer(&p0.query).expect("answer 0");
    assert_eq!(
        client.reconstruct_and_replenish(&p0, &r0).expect("recon 0"),
        record(&db, 0)
    );
    assert_eq!(client.remaining_backup_hints(), 2);

    // Retiring an already-promoted backup must not decrement remaining_backup_hints.
    client
        .retire_hints(&[first_backup_id])
        .expect("retire promoted backup");
    assert_eq!(client.remaining_backup_hints(), 2);

    // Roundtrip through client state and exhaust the remaining 2 backups (indices 2 and 3).
    let mut state: Vec<u8> = Vec::new();
    client.write_state(&mut state).expect("write state");
    let (mut restored, pending) =
        Client::read_state(master_seed, [0x66u8; 32], state.as_slice()).expect("read state");
    assert!(pending.is_empty());
    assert_eq!(restored.remaining_backup_hints(), 2);

    for idx in [1usize, 2] {
        let p: PreparedQuery = restored.prepare_query(idx).expect("query");
        let r: Response = server.answer(&p.query).expect("answer");
        assert_eq!(
            restored.reconstruct_and_replenish(&p, &r).expect("recon"),
            record(&db, idx)
        );
    }
    assert_eq!(restored.remaining_backup_hints(), 0);
    assert_eq!(
        restored.prepare_query(3).err(),
        Some(ProtocolError::NoBackupHintsLeft)
    );
}
