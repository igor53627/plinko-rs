//! End-to-end tests from the `plinko_hints` binary to the protocol client.
//!
//! The generator reads a v3 `database.bin` and writes a hint file. The client
//! loads the file and answers queries against a server that holds the same
//! database.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use plinko::hints::{HintFileError, HintSet};
use plinko::protocol::{
    compute_hints, Client, Entry, ProtocolError, ProtocolParams, Server, ENTRY_SIZE,
};
use plinko::schema40::ENTRY_SIZE as DB_ENTRY_SIZE;

const MASTER_SEED: [u8; 32] = [0x5Cu8; 32];
const QUERY_SEED: [u8; 32] = [0xA7u8; 32];

fn make_entry(tag: u64) -> Entry {
    let mut entry = [0u8; ENTRY_SIZE];
    entry[0..8].copy_from_slice(&tag.to_le_bytes());
    for (i, byte) in entry.iter_mut().enumerate().skip(8) {
        *byte = (tag as u8).wrapping_mul(31).wrapping_add(i as u8) | 0x01;
    }
    entry
}

fn make_db(num_entries: usize) -> Vec<Entry> {
    (0..num_entries).map(|i| make_entry(i as u64 + 1)).collect()
}

/// A v3 `database.bin`: the protocol entry is the first 32 bytes of a record.
/// Bytes 32..40 are outside the parity window and must not change a hint.
fn database_bytes(entries: &[Entry]) -> Vec<u8> {
    let mut bytes: Vec<u8> = Vec::with_capacity(entries.len() * DB_ENTRY_SIZE);
    for (i, entry) in entries.iter().enumerate() {
        bytes.extend_from_slice(entry);
        bytes.extend_from_slice(&[0xE0 | (i as u8 & 0x0F); DB_ENTRY_SIZE - ENTRY_SIZE]);
    }
    bytes
}

struct Scenario {
    name: &'static str,
    num_entries: usize,
    block_size: usize,
    lambda: usize,
    /// Value of `--threads`; `None` leaves the default of the binary.
    threads: Option<usize>,
}

struct Generated {
    entries: Vec<Entry>,
    hint_path: PathBuf,
}

fn work_dir(name: &str) -> PathBuf {
    let dir: PathBuf = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("hint_file_e2e_{name}"));
    fs::create_dir_all(&dir).expect("create work dir");
    dir
}

/// Writes the database and the seed file, then runs `plinko_hints --output`.
fn generate(scenario: &Scenario) -> Generated {
    let dir: PathBuf = work_dir(scenario.name);
    let db_path: PathBuf = dir.join("database.bin");
    let seed_path: PathBuf = dir.join("seed.hex");
    let hint_path: PathBuf = dir.join("hints.bin");

    let entries: Vec<Entry> = make_db(scenario.num_entries);
    fs::write(&db_path, database_bytes(&entries)).expect("write database");
    let seed_hex: String = MASTER_SEED.iter().map(|b| format!("{b:02x}")).collect();
    fs::write(&seed_path, format!("{seed_hex}\n")).expect("write seed file");
    // An earlier output with permissive mode: the run must replace it, not reuse it.
    fs::write(&hint_path, b"stale").expect("write stale hint file");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&hint_path, fs::Permissions::from_mode(0o644))
            .expect("set mode of stale hint file");
    }

    let output: Output = Command::new(env!("CARGO_BIN_EXE_plinko_hints"))
        .arg("--db-path")
        .arg(&db_path)
        .arg("--lambda")
        .arg(scenario.lambda.to_string())
        .arg("--entries-per-block")
        .arg(scenario.block_size.to_string())
        .arg("--seed-file")
        .arg(&seed_path)
        .arg("--output")
        .arg(&hint_path)
        .args(
            scenario
                .threads
                .map(|threads| format!("--threads={threads}")),
        )
        .output()
        .expect("run plinko_hints");
    assert!(
        output.status.success(),
        "plinko_hints failed: {}\n{}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );

    let leftovers: Vec<String> = fs::read_dir(&dir)
        .expect("list work dir")
        .map(|entry| {
            entry
                .expect("dir entry")
                .file_name()
                .to_string_lossy()
                .into_owned()
        })
        .filter(|name| name.ends_with(".tmp"))
        .collect();
    assert!(
        leftovers.is_empty(),
        "temporary files remain: {leftovers:?}"
    );

    Generated { entries, hint_path }
}

fn load(path: &Path) -> HintSet {
    let file = fs::File::open(path).expect("open hint file");
    HintSet::read_from(std::io::BufReader::new(file)).expect("read hint file")
}

/// Queries `count` indices that are spread over the database and checks each answer.
/// Returns the queried indices. An index with no hint is skipped.
fn query_and_check(
    client: &mut Client,
    server: &Server,
    entries: &[Entry],
    candidates: impl Iterator<Item = usize>,
    count: usize,
) -> Vec<usize> {
    let mut queried: Vec<usize> = Vec::new();
    for index in candidates {
        if queried.len() == count {
            break;
        }
        let prepared = match client.prepare_query(index) {
            Ok(prepared) => prepared,
            Err(ProtocolError::NoHintForIndex { .. }) => continue,
            Err(error) => panic!("prepare_query({index}) failed: {error}"),
        };
        let response = server.answer(&prepared.query).expect("answer");
        let got: Entry = client
            .reconstruct_and_replenish(prepared, response)
            .expect("reconstruct");
        assert_eq!(got, entries[index], "wrong answer for index {index}");
        queried.push(index);
    }
    assert_eq!(queried.len(), count, "too few indices have a hint");
    queried
}

/// Loads the generated file into a client and runs queries, an update, and more queries.
fn run_online_phase(scenario: &Scenario, generated: &Generated, hints: HintSet) {
    let params: ProtocolParams = ProtocolParams::for_hint_set(&hints).expect("params of the file");
    let expected: ProtocolParams = ProtocolParams::new(
        scenario.num_entries,
        Some(scenario.block_size),
        scenario.lambda,
        None,
    )
    .expect("expected params");
    assert_eq!(params.num_blocks, expected.num_blocks);
    assert_eq!(params.num_regular_hints, expected.num_regular_hints);
    assert_eq!(params.num_backup_hints, expected.num_backup_hints);

    let mut entries: Vec<Entry> = generated.entries.clone();
    let mut server: Server = Server::new(params.clone(), &entries).expect("server");
    let mut client: Client =
        Client::from_hints(params.clone(), MASTER_SEED, QUERY_SEED, hints).expect("client");

    let n: usize = entries.len();
    let spread = |start: usize| (0..n).map(move |k| (start + k * 37) % n);
    let first: Vec<usize> = query_and_check(&mut client, &server, &entries, spread(0), 5);

    // Change one index that was queried (it is in the client cache and in a
    // promoted hint) and many indices that were not.
    let mut updates: Vec<(usize, Entry)> = Vec::new();
    for index in (0..n).filter(|index| index % 3 != 1 || *index == first[0]) {
        let value: Entry = make_entry(0xBEEF_0000 + index as u64);
        entries[index] = value;
        updates.push((index, value));
    }
    let deltas = server.apply_updates(&updates).expect("server update");
    client.apply_updates(&deltas).expect("client update");

    // The first candidates are a changed entry and an unchanged entry that
    // were not queried before the update.
    let fresh_changed: usize = (0..n)
        .find(|index| index % 3 == 0 && !first.contains(index))
        .expect("a changed index that was not queried");
    let fresh_unchanged: usize = (0..n)
        .find(|index| index % 3 == 1 && !first.contains(index))
        .expect("an unchanged index that was not queried");
    let candidates = [fresh_changed, fresh_unchanged]
        .into_iter()
        .chain(spread(n / 2));
    let second: Vec<usize> = query_and_check(&mut client, &server, &entries, candidates, 6);
    assert!(
        second.iter().any(|index| index % 3 != 1),
        "no changed entry was queried after the update"
    );

    // A cached index is answered with the updated value.
    let prepared = client.prepare_query(first[0]).expect("re-query");
    let response = server.answer(&prepared.query).expect("answer");
    let got: Entry = client
        .reconstruct_and_replenish(prepared, response)
        .expect("reconstruct");
    assert_eq!(got, entries[first[0]], "stale value for a cached index");
}

#[test]
fn generator_file_equals_the_client_hint_set_and_answers_queries() {
    // 97 entries in 8 blocks of 16: the last block is partly padding, and the
    // eighth block is all padding.
    let scenario = Scenario {
        name: "small",
        num_entries: 97,
        block_size: 16,
        lambda: 8,
        threads: None,
    };
    let generated: Generated = generate(&scenario);
    let hints: HintSet = load(&generated.hint_path);

    let params: ProtocolParams = ProtocolParams::for_hint_set(&hints).expect("params");
    let in_memory: HintSet =
        compute_hints(&params, &MASTER_SEED, &generated.entries).expect("compute_hints");
    assert!(
        in_memory
            .regular
            .iter()
            .any(|hint| hint.parity != [0u8; 32]),
        "the scenario must give non-zero parities"
    );
    assert!(
        hints == in_memory,
        "generator file differs from compute_hints"
    );

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode: u32 = fs::metadata(&generated.hint_path)
            .expect("metadata")
            .permissions()
            .mode();
        assert_eq!(
            mode & 0o077,
            0,
            "the hint file must not be group or world readable"
        );
    }

    run_online_phase(&scenario, &generated, hints);
}

/// The hint file must not depend on the number of worker threads.
#[test]
fn generator_file_is_the_same_for_every_thread_count() {
    // 8 blocks: fewer threads than blocks, and more threads than blocks.
    let files: Vec<Vec<u8>> = [("threads_1", 1), ("threads_3", 3), ("threads_11", 11)]
        .into_iter()
        .map(|(name, threads)| {
            let generated: Generated = generate(&Scenario {
                name,
                num_entries: 120,
                block_size: 16,
                lambda: 8,
                threads: Some(threads),
            });
            fs::read(&generated.hint_path).expect("read hint file")
        })
        .collect();

    let one_thread: HintSet = HintSet::read_from(files[0].as_slice()).expect("hint file");
    assert!(
        one_thread
            .backup
            .iter()
            .any(|hint| hint.parity_out != [0u8; 32]),
        "the scenario must give non-zero parities"
    );
    assert!(files[1] == files[0], "3 threads differ from 1 thread");
    assert!(files[2] == files[0], "11 threads differ from 1 thread");
}

/// 66,560 hints: the iPRF domain is above 65,536.
///
/// Takes minutes without hardware AES. Run with:
/// `cargo test --release -p plinko --test hint_file_e2e_test -- --ignored`
#[test]
#[ignore]
fn generator_file_answers_queries_above_65536_hints() {
    let scenario = Scenario {
        name: "large",
        num_entries: 1000,
        block_size: 512,
        lambda: 65,
        threads: None,
    };
    let generated: Generated = generate(&scenario);
    let hints: HintSet = load(&generated.hint_path);
    assert_eq!(hints.regular.len() + hints.backup.len(), 66_560);
    run_online_phase(&scenario, &generated, hints);
}

#[test]
fn client_rejects_a_hint_set_that_does_not_belong_to_it() {
    let entries: Vec<Entry> = make_db(40);
    let params: ProtocolParams = ProtocolParams::new(40, Some(8), 2, None).expect("params");
    let hints: HintSet = compute_hints(&params, &MASTER_SEED, &entries).expect("hints");

    let mismatch = |result: Result<Client, ProtocolError>| -> &'static str {
        match result {
            Err(ProtocolError::HintSetMismatch(what)) => what,
            Err(other) => panic!("expected HintSetMismatch, got {other}"),
            Ok(_) => panic!("expected HintSetMismatch, got a client"),
        }
    };

    assert!(Client::from_hints(params.clone(), MASTER_SEED, QUERY_SEED, hints.clone()).is_ok());

    let other_seed: [u8; 32] = [0x5Du8; 32];
    assert_eq!(
        mismatch(Client::from_hints(
            params.clone(),
            other_seed,
            QUERY_SEED,
            hints.clone()
        )),
        "master seed"
    );

    // 41 entries need the same six blocks; only num_entries differs.
    let other_size: ProtocolParams = ProtocolParams::new(41, Some(8), 2, None).expect("params");
    assert_eq!(other_size.num_blocks, params.num_blocks);
    assert_eq!(
        mismatch(Client::from_hints(
            other_size,
            MASTER_SEED,
            QUERY_SEED,
            hints.clone()
        )),
        "database geometry"
    );

    let other_lambda: ProtocolParams = ProtocolParams::new(40, Some(8), 3, None).expect("params");
    assert_eq!(
        mismatch(Client::from_hints(
            other_lambda,
            MASTER_SEED,
            QUERY_SEED,
            hints.clone()
        )),
        "number of regular hints"
    );

    let other_backup: ProtocolParams =
        ProtocolParams::new(40, Some(8), 2, Some(15)).expect("params");
    assert_eq!(
        mismatch(Client::from_hints(
            other_backup,
            MASTER_SEED,
            QUERY_SEED,
            hints.clone()
        )),
        "number of backup hints"
    );

    // A header whose block count is not the block count of its database size.
    let mut wrong_blocks: HintSet = hints.clone();
    wrong_blocks.geometry.num_blocks += 2;
    assert!(matches!(
        ProtocolParams::for_hint_set(&wrong_blocks),
        Err(ProtocolError::HintSetMismatch(_))
    ));
}

#[test]
fn generator_refuses_output_without_a_seed() {
    let dir: PathBuf = work_dir("no_seed");
    let db_path: PathBuf = dir.join("database.bin");
    let hint_path: PathBuf = dir.join("hints.bin");
    fs::write(&db_path, database_bytes(&make_db(8))).expect("write database");
    if hint_path.exists() {
        fs::rename(&hint_path, dir.join("hints.bin.old")).expect("move old hint file");
    }

    let output: Output = Command::new(env!("CARGO_BIN_EXE_plinko_hints"))
        .arg("--db-path")
        .arg(&db_path)
        .arg("--entries-per-block")
        .arg("4")
        .arg("--output")
        .arg(&hint_path)
        .output()
        .expect("run plinko_hints");
    assert!(!output.status.success(), "a run without a seed must fail");
    assert!(!hint_path.exists(), "no hint file must be written");
}

#[test]
fn a_corrupted_generator_file_is_rejected_on_load() {
    let entries: Vec<Entry> = make_db(40);
    let params: ProtocolParams = ProtocolParams::new(40, Some(8), 2, None).expect("params");
    let hints: HintSet = compute_hints(&params, &MASTER_SEED, &entries).expect("hints");
    let mut bytes: Vec<u8> = Vec::new();
    hints.write_to(&mut bytes).expect("write");

    let middle: usize = bytes.len() / 2;
    bytes[middle] ^= 0x10;
    assert!(matches!(
        HintSet::read_from(bytes.as_slice()),
        Err(HintFileError::ChecksumMismatch)
    ));
}
