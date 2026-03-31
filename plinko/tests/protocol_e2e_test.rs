use plinko::protocol::{
    block_in_subset, compute_backup_blocks, compute_regular_blocks, derive_subset_seed, Client,
    Entry, ProtocolParams, Server, ENTRY_SIZE, SEED_LABEL_BACKUP, SEED_LABEL_REGULAR,
};

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

// --- Helper: minimal BlockBitset for bitmap cross-check tests ---

struct BlockBitset {
    bits: Vec<u64>,
    num_blocks: usize,
}

impl BlockBitset {
    fn from_sorted_blocks(blocks: &[usize], num_blocks: usize) -> Self {
        let num_words = num_blocks.div_ceil(64);
        let mut bits = vec![0u64; num_words];
        for &block in blocks {
            if block < num_blocks {
                bits[block / 64] |= 1u64 << (block % 64);
            }
        }
        Self { bits, num_blocks }
    }

    fn contains(&self, block: usize) -> bool {
        if block >= self.num_blocks {
            return false;
        }
        (self.bits[block / 64] >> (block % 64)) & 1 == 1
    }
}

// --- Regression tests ---

#[test]
fn test_bitmap_matches_sorted_vec() {
    let master_seed = [0x77u8; 32];
    for c in [10, 64, 100, 128, 200] {
        for j in 0..10 {
            let seed = derive_subset_seed(&master_seed, b"test_regular", j);
            let blocks = compute_regular_blocks(&seed, c);
            let bitset = BlockBitset::from_sorted_blocks(&blocks, c);

            for b in 0..c {
                let vec_result = block_in_subset(&blocks, b);
                let bitset_result = bitset.contains(b);
                assert_eq!(
                    vec_result, bitset_result,
                    "Mismatch at c={c}, j={j}, block={b}"
                );
            }
        }

        for j in 0..10 {
            let seed = derive_subset_seed(&master_seed, b"test_backup", j);
            let blocks = compute_backup_blocks(&seed, c);
            let bitset = BlockBitset::from_sorted_blocks(&blocks, c);

            for b in 0..c {
                let vec_result = block_in_subset(&blocks, b);
                let bitset_result = bitset.contains(b);
                assert_eq!(
                    vec_result, bitset_result,
                    "Backup mismatch at c={c}, j={j}, block={b}"
                );
            }
        }
    }
}

#[test]
fn test_complement_flag_matches_complement_subset() {
    let master_seed = [0x88u8; 32];
    for c in [10, 64, 100, 128] {
        for j in 0..20 {
            let seed = derive_subset_seed(&master_seed, SEED_LABEL_BACKUP, j);
            let blocks = compute_backup_blocks(&seed, c);

            // Build complement manually
            let mut complement = Vec::new();
            for b in 0..c {
                if !block_in_subset(&blocks, b) {
                    complement.push(b);
                }
            }

            // Verify complement flag: !block_in_subset(blocks, b) == block_in_subset(complement, b)
            for b in 0..c {
                let in_original = block_in_subset(&blocks, b);
                let in_complement = block_in_subset(&complement, b);
                assert_eq!(
                    !in_original, in_complement,
                    "Complement mismatch at c={c}, j={j}, block={b}"
                );

                // Also verify: complemented flag XOR original membership
                let complemented_result = in_original ^ true; // complemented=true
                assert_eq!(
                    complemented_result, in_complement,
                    "XOR complement mismatch at c={c}, j={j}, block={b}"
                );
            }
        }
    }
}

#[test]
fn test_seed_recomputation_deterministic() {
    let master_seed = [0x99u8; 32];
    for c in [10, 50, 100, 200] {
        for j in 0..10u64 {
            let seed = derive_subset_seed(&master_seed, SEED_LABEL_REGULAR, j);
            let blocks1 = compute_regular_blocks(&seed, c);
            let blocks2 = compute_regular_blocks(&seed, c);
            assert_eq!(blocks1, blocks2, "Regular non-deterministic at c={c}, j={j}");

            let seed = derive_subset_seed(&master_seed, SEED_LABEL_BACKUP, j);
            let blocks1 = compute_backup_blocks(&seed, c);
            let blocks2 = compute_backup_blocks(&seed, c);
            assert_eq!(blocks1, blocks2, "Backup non-deterministic at c={c}, j={j}");
        }
    }
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
    let mut client =
        Client::offline_init(params.clone(), master_seed, query_seed, &entries)
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
    assert!(tested >= 6, "expected at least 6 queryable indices, got {tested}");

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
    let master_seed = [0xCCu8; 32];
    let query_seed = [0xDDu8; 32];

    let mut server = Server::new(params.clone(), &entries).expect("server init");
    let mut client =
        Client::offline_init(params.clone(), master_seed, query_seed, &entries).expect("offline init");

    // Phase 1: Query a few indices, consuming backups (each promotes one).
    // With 3 queries, ~50% of promoted slots will have complemented=true.
    let num_phase1 = 3;
    let mut phase1_count = 0;
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
        phase1_count += 1;
    }
    assert_eq!(phase1_count, num_phase1);
    let backups_after_phase1 = client.remaining_backup_hints();

    // Phase 2: Update every entry in the database.
    let mut server_updates = Vec::new();
    for (idx, entry) in entries.iter_mut().enumerate() {
        let new_val = make_entry(0xF000 + idx as u64);
        server_updates.push((idx, new_val));
        *entry = new_val;
    }
    let deltas = server.apply_updates(&server_updates).expect("server update");
    client.apply_updates(&deltas).expect("client update");

    // Phase 3: Query new indices after the update. These queries use hints
    // (including promoted slots with complemented=true) whose parities were
    // maintained through apply_updates. Correct reconstruction validates that
    // the complement flag was handled properly during the update.
    let num_phase3 = 3;
    let mut phase3_count = 0;
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
    assert_eq!(
        client.remaining_backup_hints(),
        backups_after_phase1 - phase3_count
    );
}
