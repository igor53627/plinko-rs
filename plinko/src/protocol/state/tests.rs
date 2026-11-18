use super::*;
use crate::protocol::{Query, Response, Server};

const NUM_ENTRIES: usize = 97;
const RECORD_SIZE: usize = 13;
const BLOCK_SIZE: usize = 16;
const LAMBDA: usize = 4;
const NUM_REGULAR: usize = LAMBDA * BLOCK_SIZE;
const NUM_BACKUP: usize = NUM_REGULAR;
const MASTER_SEED: [u8; 32] = [0xA1u8; 32];
const CHECKSUM_SIZE: usize = 32;
const HINT_HEADER_SIZE: usize = 112;
const ANSWERED_SIZE: usize = 17 + RECORD_SIZE;

fn make_database(num_entries: usize, record_size: usize) -> Vec<u8> {
    (0..num_entries * record_size)
        .map(|i| (i * 7 + i / record_size + 3) as u8)
        .collect()
}

fn record(database: &[u8], index: usize) -> &[u8] {
    &database[index * RECORD_SIZE..(index + 1) * RECORD_SIZE]
}

fn state_bytes(client: &Client) -> Vec<u8> {
    let mut bytes: Vec<u8> = Vec::new();
    client.write_state(&mut bytes).expect("write state");
    bytes
}

fn read(bytes: &[u8]) -> Result<(Client, Vec<PreparedQuery>), ClientStateError> {
    Client::read_state(MASTER_SEED, [0xC3u8; 32], bytes)
}

fn read_error(bytes: &[u8]) -> ClientStateError {
    match read(bytes) {
        Ok(_) => panic!("the state was accepted"),
        Err(error) => error,
    }
}

fn answer(server: &Server, client: &mut Client, prepared: &PreparedQuery) -> Vec<u8> {
    let response: Response = server.answer(&prepared.query).expect("answer");
    client
        .reconstruct_and_replenish(prepared, &response)
        .expect("reconstruct")
}

/// Prepares a query for the first index with a hint that has no answer and no query in flight.
fn prepare_new(client: &mut Client) -> PreparedQuery {
    for index in 0..NUM_ENTRIES {
        if client.cache.contains_key(&index) || client.in_flight.contains_key(&index) {
            continue;
        }
        if let Ok(prepared) = client.prepare_query(index) {
            return prepared;
        }
    }
    panic!("no index with a hint");
}

fn fields(prepared: &PreparedQuery) -> (Query, usize, usize, usize, bool) {
    (
        prepared.query.clone(),
        prepared.requested_index,
        prepared.queried_index,
        prepared.hint_idx,
        prepared.real_is_first,
    )
}

fn assert_same_state(got: &Client, want: &Client) {
    assert_eq!(format!("{:?}", got.params), format!("{:?}", want.params));
    assert!(got.hints == want.hints, "hint sets differ");
    assert_eq!(got.slots, want.slots);
    assert_eq!(got.backup_available, want.backup_available);
    assert_eq!(got.cache, want.cache);
    assert_eq!(got.in_flight, want.in_flight);
    assert_eq!(got.retired_backups, want.retired_backups);
    assert_eq!(got.queried, want.queried);
}

/// A client with each kind of state that the format stores.
struct Busy {
    server: Server,
    client: Client,
    database: Vec<u8>,
    /// Hints of the queries that have an answer or were abandoned, and the retired hints.
    sent_hints: Vec<usize>,
    /// The two queries in flight: one for a new index, one decoy.
    in_flight: Vec<PreparedQuery>,
    /// The two backup hints that `retire_hints` removed, in the order of their numbers.
    retired_backups: Vec<usize>,
}

fn busy_client() -> Busy {
    let params: ProtocolParams =
        ProtocolParams::new(NUM_ENTRIES, RECORD_SIZE, Some(BLOCK_SIZE), LAMBDA, None)
            .expect("params");
    assert_eq!(
        (params.num_regular_hints, params.num_backup_hints),
        (NUM_REGULAR, NUM_BACKUP)
    );
    let mut database: Vec<u8> = make_database(NUM_ENTRIES, RECORD_SIZE);
    let mut server: Server = Server::new(params.clone(), 5, &database).expect("server");
    let mut client: Client =
        Client::offline_init(params, MASTER_SEED, [0xB2u8; 32], 5, &database).expect("client");
    let mut sent_hints: Vec<usize> = Vec::new();

    // 37 and 97 have no common factor, so the indices go through all blocks.
    let mut answered: usize = 0;
    for index in (0..NUM_ENTRIES).map(|k| (k * 37) % NUM_ENTRIES) {
        if answered == 30 {
            break;
        }
        let Ok(prepared) = client.prepare_query(index) else {
            continue;
        };
        assert_eq!(
            answer(&server, &mut client, &prepared),
            record(&database, index)
        );
        sent_hints.push(prepared.hint_idx);
        answered += 1;
    }
    assert_eq!(answered, 30);

    let updates: Vec<(usize, Vec<u8>)> = (0..NUM_ENTRIES)
        .step_by(3)
        .map(|index| (index, vec![index as u8 ^ 0x5A; RECORD_SIZE]))
        .collect();
    let delta = server.apply_updates(&updates).expect("server update");
    client.apply_delta(&delta).expect("client update");
    for (index, value) in &updates {
        database[index * RECORD_SIZE..(index + 1) * RECORD_SIZE].copy_from_slice(value);
    }

    let abandoned: PreparedQuery = prepare_new(&mut client);
    sent_hints.push(abandoned.hint_idx);
    client.abandon_query(abandoned).expect("abandon");

    let new_query: PreparedQuery = prepare_new(&mut client);
    let mut cached: Vec<usize> = client.cache.keys().copied().collect();
    cached.sort_unstable();
    let decoy: PreparedQuery = cached
        .iter()
        .find_map(|&index| client.prepare_query(index).ok())
        .expect("decoy query");
    assert_ne!(decoy.requested_index, decoy.queried_index);

    let regular: usize = (0..NUM_REGULAR)
        .find(|&j| client.slots[j].is_some() && !client.hint_is_reserved(j))
        .expect("a regular hint that is present");
    let mut retired_backups: Vec<usize> = (0..NUM_BACKUP)
        .rev()
        .filter(|&b| client.backup_available[b])
        .take(2)
        .collect();
    retired_backups.sort_unstable();
    assert_eq!(retired_backups.len(), 2);
    let retired: Vec<usize> = vec![
        regular,
        NUM_REGULAR + retired_backups[1],
        NUM_REGULAR + retired_backups[0],
    ];
    client.retire_hints(&retired).expect("retire");
    sent_hints.extend(&retired);

    // The scenario must hold each state of a promoted hint.
    let promoted: Vec<Option<HintSlot>> = client
        .cache
        .values()
        .map(|entry| client.slots[entry.hint_idx])
        .collect();
    let sent_later: usize = promoted.iter().filter(|slot| slot.is_none()).count();
    let complemented: usize = promoted
        .iter()
        .filter(|slot| slot.is_some_and(|slot| slot.complemented))
        .count();
    let not_complemented: usize = promoted.len() - sent_later - complemented;
    assert!(
        sent_later > 0 && complemented > 0 && not_complemented > 0,
        "promoted hints: {sent_later} sent later, {complemented} complemented, \
         {not_complemented} not complemented"
    );

    Busy {
        server,
        client,
        database,
        sent_hints,
        in_flight: vec![new_query, decoy],
        retired_backups,
    }
}

/// Offsets of the sections of a state file, from the format table of the module.
struct Layout {
    sent: usize,
    answered: usize,
    in_flight: usize,
    retired: usize,
    checksum: usize,
}

impl Layout {
    fn of(client: &Client) -> Self {
        let size: usize = client.params.record_size;
        let hint_body: usize = HINT_HEADER_SIZE
            + client.params.num_regular_hints * (4 + size)
            + client.params.num_backup_hints * (4 + 2 * size);
        let sent: usize = 12 + hint_body;
        let answered: usize = sent + client.params.num_regular_hints.div_ceil(8);
        let in_flight: usize = answered + 8 + client.cache.len() * (17 + size);
        let retired: usize = in_flight + 8 + client.in_flight.len() * 25;
        let checksum: usize = retired + 8 + client.retired_backups.len() * 8;
        Self {
            sent,
            answered,
            in_flight,
            retired,
            checksum,
        }
    }

    fn answered_entry(&self, k: usize) -> usize {
        self.answered + 8 + k * ANSWERED_SIZE
    }

    fn in_flight_entry(&self, k: usize) -> usize {
        self.in_flight + 8 + k * 25
    }

    fn retired_entry(&self, k: usize) -> usize {
        self.retired + 8 + k * 8
    }
}

/// A change of the bytes of a state file.
type Patch<'a> = Box<dyn Fn(&mut [u8]) + 'a>;

fn put(bytes: &mut [u8], offset: usize, value: usize) {
    bytes[offset..offset + 8].copy_from_slice(&(value as u64).to_le_bytes());
}

fn swap_neighbors(bytes: &mut [u8], first: usize, size: usize) {
    let (head, tail) = bytes[first..first + 2 * size].split_at_mut(size);
    head.swap_with_slice(tail);
}

/// Writes the checksum of the content, so that only the content can be wrong.
fn sealed(mut bytes: Vec<u8>) -> Vec<u8> {
    let body: usize = bytes.len() - CHECKSUM_SIZE;
    let hash: blake3::Hash = blake3::hash(&bytes[..body]);
    bytes[body..].copy_from_slice(hash.as_bytes());
    bytes
}

fn error_after(bytes: &[u8], patch: impl FnOnce(&mut [u8])) -> ClientStateError {
    let mut patched: Vec<u8> = bytes.to_vec();
    patch(&mut patched);
    read_error(&sealed(patched))
}

fn invalid_after(bytes: &[u8], patch: impl FnOnce(&mut [u8])) -> &'static str {
    match error_after(bytes, patch) {
        ClientStateError::Invalid(what) => what,
        other => panic!("unexpected error: {other}"),
    }
}

/// The client from the state must continue as the client that wrote it: the
/// same queries in flight, and the same hints, cache and backup hints after
/// the same answers and the same delta.
#[test]
fn a_client_from_a_state_continues_as_the_client_that_wrote_it() {
    let mut busy: Busy = busy_client();
    let bytes: Vec<u8> = state_bytes(&busy.client);
    let (mut restored, prepared) = read(&bytes).expect("read state");

    assert_same_state(&restored, &busy.client);
    assert_eq!(state_bytes(&restored), bytes);
    assert!(restored.unused_hints().is_none());

    busy.in_flight
        .sort_unstable_by_key(|query| query.queried_index);
    assert_eq!(
        prepared.iter().map(fields).collect::<Vec<_>>(),
        busy.in_flight.iter().map(fields).collect::<Vec<_>>()
    );

    for query in &prepared {
        let response: Response = busy.server.answer(&query.query).expect("answer");
        let got: Vec<u8> = restored
            .reconstruct_and_replenish(query, &response)
            .expect("reconstruct after the restart");
        assert_eq!(got, record(&busy.database, query.requested_index));
        assert_eq!(
            busy.client.reconstruct_and_replenish(query, &response),
            Ok(got)
        );
    }
    assert_same_state(&restored, &busy.client);

    let updates: Vec<(usize, Vec<u8>)> = (0..NUM_ENTRIES)
        .map(|index| (index, vec![index as u8 ^ 0xC6; RECORD_SIZE]))
        .collect();
    let delta = busy.server.apply_updates(&updates).expect("server update");
    restored
        .apply_delta(&delta)
        .expect("delta after the restart");
    busy.client.apply_delta(&delta).expect("delta");
    assert_same_state(&restored, &busy.client);
    assert_eq!(
        state_bytes(&restored),
        state_bytes(&busy.client),
        "state after the answers and the delta"
    );
}

#[test]
fn the_state_file_has_the_documented_layout() {
    let busy: Busy = busy_client();
    let client: &Client = &busy.client;
    let bytes: Vec<u8> = state_bytes(client);

    let mut hint_file: Vec<u8> = Vec::new();
    client.hints.write_to(&mut hint_file).expect("hint file");
    let hint_body: usize = hint_file.len() - CHECKSUM_SIZE;

    assert_eq!(&bytes[0..8], b"PLNKSTAT");
    assert_eq!(u32_at(&bytes, 8), 1);
    assert!(bytes[12..12 + hint_body] == hint_file[..hint_body]);

    let mut at: usize = 12 + hint_body;
    for j in 0..NUM_REGULAR {
        let sent: bool = (bytes[at + j / 8] >> (j % 8)) & 1 == 1;
        assert_eq!(sent, busy.sent_hints.contains(&j), "regular hint {j}");
    }
    at += NUM_REGULAR / 8;

    let mut answered: Vec<usize> = client.cache.keys().copied().collect();
    answered.sort_unstable();
    assert_eq!(answered.len(), 30);
    assert_eq!(u64_at(&bytes, at), 30);
    at += 8;
    for index in answered {
        let entry: &CacheEntry = &client.cache[&index];
        let sent_later: bool = busy.sent_hints.contains(&entry.hint_idx);
        let flags: u8 = match client.slots[entry.hint_idx] {
            Some(slot) => 1 + 2 * u8::from(slot.complemented),
            None => 0,
        };
        assert_eq!(flags == 0, sent_later, "promoted hint of index {index}");
        assert_eq!(u64_at(&bytes, at), index as u64);
        assert_eq!(
            u64_at(&bytes, at + 8),
            (entry.hint_idx - NUM_REGULAR) as u64
        );
        assert_eq!(bytes[at + 16], flags);
        assert_eq!(
            &bytes[at + 17..at + ANSWERED_SIZE],
            record(&busy.database, index)
        );
        at += ANSWERED_SIZE;
    }

    let mut in_flight: Vec<&PreparedQuery> = busy.in_flight.iter().collect();
    in_flight.sort_unstable_by_key(|query| query.queried_index);
    assert_eq!(u64_at(&bytes, at), 2);
    at += 8;
    for query in in_flight {
        assert_eq!(u64_at(&bytes, at), query.queried_index as u64);
        assert_eq!(u64_at(&bytes, at + 8), query.requested_index as u64);
        assert_eq!(u64_at(&bytes, at + 16), query.hint_idx as u64);
        assert_eq!(bytes[at + 24], u8::from(query.real_is_first));
        at += 25;
    }

    assert_eq!(u64_at(&bytes, at), 2);
    at += 8;
    for &backup_idx in &busy.retired_backups {
        assert_eq!(u64_at(&bytes, at), backup_idx as u64);
        at += 8;
    }

    assert_eq!(bytes.len(), at + CHECKSUM_SIZE);
    assert_eq!(&bytes[at..], blake3::hash(&bytes[..at]).as_bytes());

    let layout: Layout = Layout::of(client);
    assert_eq!(layout.checksum, at);
}

#[test]
fn damage_to_a_state_file_is_detected() {
    let bytes: Vec<u8> = state_bytes(&busy_client().client);

    for position in 0..bytes.len() {
        for bit in 0..8 {
            let mut damaged: Vec<u8> = bytes.clone();
            damaged[position] ^= 1 << bit;
            assert!(
                read(&damaged).is_err(),
                "byte {position}, bit {bit}: the damage was not detected"
            );
        }
    }

    for len in 0..bytes.len() {
        match read_error(&bytes[..len]) {
            ClientStateError::Io(error) => {
                assert_eq!(error.kind(), io::ErrorKind::UnexpectedEof, "{len} bytes");
            }
            other => panic!("{len} bytes: unexpected error: {other}"),
        }
    }

    let mut longer: Vec<u8> = bytes.clone();
    longer.push(0);
    assert!(matches!(
        read_error(&longer),
        ClientStateError::TrailingData
    ));

    let last: usize = bytes.len() - 1;
    let mut wrong_hash: Vec<u8> = bytes;
    wrong_hash[last] ^= 0x80;
    assert!(matches!(
        read_error(&wrong_hash),
        ClientStateError::ChecksumMismatch
    ));
}

#[test]
fn a_state_of_another_master_seed_or_format_is_rejected() {
    let bytes: Vec<u8> = state_bytes(&busy_client().client);

    let mut other_seed: [u8; 32] = MASTER_SEED;
    other_seed[31] ^= 1;
    assert!(matches!(
        Client::read_state(other_seed, [0xC3u8; 32], bytes.as_slice()).err(),
        Some(ClientStateError::Mismatch(ProtocolError::HintSetMismatch(
            "master seed"
        )))
    ));

    assert!(matches!(
        error_after(&bytes, |bytes| bytes[7] = b'E'),
        ClientStateError::BadMagic
    ));
    assert!(matches!(
        error_after(&bytes, |bytes| bytes[8] = 2),
        ClientStateError::UnsupportedVersion(2)
    ));
    // The record size of the hint set: offset 12 of the hint file header.
    assert!(matches!(
        error_after(&bytes, |bytes| bytes[24..28].fill(0)),
        ClientStateError::Hints(HintFileError::InvalidRecordSize(0))
    ));

    let mut hint_file: Vec<u8> = Vec::new();
    busy_client()
        .client
        .hints
        .write_to(&mut hint_file)
        .expect("hint file");
    assert!(matches!(read_error(&hint_file), ClientStateError::BadMagic));
}

/// Each case has a correct checksum, so only the rule of the field rejects it.
#[test]
fn a_state_with_a_field_that_breaks_a_rule_is_rejected() {
    let busy: Busy = busy_client();
    let client: &Client = &busy.client;
    let bytes: Vec<u8> = state_bytes(client);
    let layout: Layout = Layout::of(client);
    assert!(read(&sealed(bytes.clone())).is_ok());

    let answered: [usize; 2] = [layout.answered_entry(0), layout.answered_entry(1)];
    let in_flight: [usize; 2] = [layout.in_flight_entry(0), layout.in_flight_entry(1)];
    let retired: [usize; 2] = [layout.retired_entry(0), layout.retired_entry(1)];
    let first_backup: usize = u64_at(&bytes, answered[0] + 8) as usize;
    let queried: usize = u64_at(&bytes, in_flight[0]) as usize;
    let first_hint: usize = u64_at(&bytes, in_flight[0] + 16) as usize;
    let cached: usize = *client.cache.keys().min().expect("cache");
    let not_cached: usize = (0..NUM_ENTRIES)
        .find(|index| !client.cache.contains_key(index) && *index != queried)
        .expect("an index with no answer");

    // Regular hints that are present and not reserved.
    let block: usize = client.params.block_of(queried);
    let offset: u64 = client.params.offset_in_block(queried) as u64;
    let free: Vec<usize> = (0..NUM_REGULAR)
        .filter(|&j| client.slots[j].is_some() && !client.hint_is_reserved(j))
        .collect();
    let has_block = |j: &&usize| -> bool {
        let slot: HintSlot = client.slots[**j].expect("present");
        client.slot_block_in_subset(**j, &slot, block)
    };
    let other_offset: usize = *free
        .iter()
        .filter(has_block)
        .find(|&&j| client.iprfs[block].forward(j as u64) != offset)
        .expect("a hint with the block at another offset");
    let other_blocks: usize = *free
        .iter()
        .find(|j| !has_block(j))
        .expect("a hint without the block");
    let sent_regular: usize = *busy
        .sent_hints
        .iter()
        .find(|&&j| j < NUM_REGULAR)
        .expect("a regular hint that was sent");

    let cases: Vec<(&str, Patch)> = vec![
        (
            "a regular hint with no valid cutoff is marked as sent",
            Box::new(|bytes| {
                let cutoff: usize = 12 + HINT_HEADER_SIZE + other_blocks * (4 + RECORD_SIZE);
                bytes[cutoff..cutoff + 4].copy_from_slice(&INVALID_CUTOFF.to_le_bytes());
                bytes[layout.sent + other_blocks / 8] |= 1 << (other_blocks % 8);
            }),
        ),
        (
            "more answered queries than backup hints",
            Box::new(|bytes| put(bytes, layout.answered, NUM_BACKUP + 1)),
        ),
        (
            "index of an answered query",
            Box::new(|bytes| put(bytes, answered[0], NUM_ENTRIES)),
        ),
        (
            "the answered queries are not in the order of their indices",
            Box::new(|bytes| swap_neighbors(bytes, answered[0], ANSWERED_SIZE)),
        ),
        (
            "the answered queries are not in the order of their indices",
            Box::new(|bytes| bytes.copy_within(answered[0]..answered[0] + 8, answered[1])),
        ),
        (
            "backup hint of an answered query",
            Box::new(|bytes| put(bytes, answered[0] + 8, NUM_BACKUP)),
        ),
        (
            "a backup hint is promoted two times, or it has no valid cutoff",
            Box::new(|bytes| put(bytes, answered[1] + 8, first_backup)),
        ),
        (
            "flags of an answered query",
            Box::new(|bytes| bytes[answered[0] + 16] = 4),
        ),
        (
            "flags of an answered query",
            Box::new(|bytes| bytes[answered[0] + 16] = 2),
        ),
        (
            "more queries in flight than free backup hints",
            Box::new(|bytes| put(bytes, layout.in_flight, NUM_BACKUP + 1)),
        ),
        (
            "queried index of a query in flight",
            Box::new(|bytes| put(bytes, in_flight[0], NUM_ENTRIES)),
        ),
        (
            "the queries in flight are not in the order of their queried indices",
            Box::new(|bytes| swap_neighbors(bytes, in_flight[0], 25)),
        ),
        (
            "a query in flight goes to an index that has an answer",
            Box::new(|bytes| put(bytes, in_flight[0], cached)),
        ),
        (
            "requested index of a query in flight",
            Box::new(|bytes| put(bytes, in_flight[0] + 8, not_cached)),
        ),
        (
            "requested index of a query in flight",
            Box::new(|bytes| put(bytes, in_flight[0] + 8, NUM_ENTRIES)),
        ),
        (
            "a query in flight has no hint, or two of them have the same hint",
            Box::new(|bytes| put(bytes, in_flight[0] + 16, NUM_REGULAR + NUM_BACKUP)),
        ),
        (
            "a query in flight has no hint, or two of them have the same hint",
            Box::new(|bytes| put(bytes, in_flight[0] + 16, sent_regular)),
        ),
        (
            "a query in flight has no hint, or two of them have the same hint",
            Box::new(|bytes| put(bytes, in_flight[1] + 16, first_hint)),
        ),
        (
            "flags of a query in flight",
            Box::new(|bytes| bytes[in_flight[0] + 24] = 2),
        ),
        (
            "the hint of a query in flight does not hold its index",
            Box::new(|bytes| put(bytes, in_flight[0] + 16, other_blocks)),
        ),
        (
            "the hint of a query in flight does not hold its index",
            Box::new(|bytes| put(bytes, in_flight[0] + 16, other_offset)),
        ),
        (
            "more retired backup hints than free backup hints",
            Box::new(|bytes| put(bytes, layout.retired, NUM_BACKUP + 1)),
        ),
        (
            "a retired backup hint is promoted, or it has no valid cutoff",
            Box::new(|bytes| put(bytes, retired[0], NUM_BACKUP)),
        ),
        (
            "a retired backup hint is promoted, or it has no valid cutoff",
            Box::new(|bytes| put(bytes, retired[0], first_backup)),
        ),
        (
            "the retired backup hints are not in the order of their numbers",
            Box::new(|bytes| swap_neighbors(bytes, retired[0], 8)),
        ),
    ];
    for (number, (expected, patch)) in cases.iter().enumerate() {
        assert_eq!(invalid_after(&bytes, patch), *expected, "case {number}");
    }
}

/// 12 regular hints: the last byte of the bitmap has 4 bits with no hint.
#[test]
fn a_state_before_the_first_query_holds_the_hint_set() {
    let params: ProtocolParams = ProtocolParams::new(30, 5, Some(4), 3, None).expect("params");
    assert_eq!(params.num_regular_hints, 12);
    let database: Vec<u8> = make_database(30, 5);
    let client: Client =
        Client::offline_init(params, MASTER_SEED, [0xB2u8; 32], 0, &database).expect("client");
    let bytes: Vec<u8> = state_bytes(&client);
    let layout: Layout = Layout::of(&client);
    assert_eq!(bytes.len(), layout.checksum + CHECKSUM_SIZE);
    assert!(bytes[layout.sent..layout.checksum].iter().all(|&b| b == 0));

    let (restored, prepared) = read(&bytes).expect("read state");
    assert!(prepared.is_empty());
    assert_same_state(&restored, &client);
    assert!(restored.unused_hints() == Some(&client.hints));

    for bit in 4..8 {
        assert_eq!(
            invalid_after(&bytes, |bytes| bytes[layout.sent + 1] |= 1 << bit),
            "a bit after the last regular hint is set",
            "bit {bit}"
        );
    }
}

/// After `retire_hints`, no query uses a hint of the list and no answer
/// promotes a backup hint of the list, also after a restart.
#[test]
fn retired_hints_are_not_used_and_not_promoted() {
    let busy: Busy = busy_client();
    let (mut client, prepared) = read(&state_bytes(&busy.client)).expect("read state");
    let backups_left: usize = client.remaining_backup_hints();
    assert_eq!(backups_left, NUM_BACKUP - 30 - 2);

    let mut later_hints: Vec<usize> = Vec::new();
    for query in &prepared {
        answer(&busy.server, &mut client, query);
        later_hints.push(query.hint_idx);
    }
    for index in (0..NUM_ENTRIES).cycle() {
        let query: PreparedQuery = match client.prepare_query(index) {
            Ok(query) => query,
            Err(ProtocolError::NoBackupHintsLeft) => break,
            Err(ProtocolError::NoHintForIndex { .. }) => continue,
            Err(other) => panic!("index {index}: {other}"),
        };
        assert_eq!(
            answer(&busy.server, &mut client, &query),
            record(&busy.database, index)
        );
        later_hints.push(query.hint_idx);
    }

    assert_eq!(later_hints.len(), backups_left);
    assert_eq!(client.remaining_backup_hints(), 0);
    for hint in &later_hints {
        assert!(
            !busy.sent_hints.contains(hint),
            "hint {hint} was sent before"
        );
    }
    for &backup_idx in &busy.retired_backups {
        assert!(client
            .cache
            .values()
            .all(|entry| entry.hint_idx != NUM_REGULAR + backup_idx));
        assert!(client.slots[NUM_REGULAR + backup_idx].is_none());
    }
}
