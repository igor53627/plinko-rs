use plinko::protocol::{Client, Entry, ProtocolParams, Server, ENTRY_SIZE};

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
