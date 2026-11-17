//! End-to-end tests from the `plinko_hints` binary to the protocol client.
//!
//! The generator reads a database file of fixed-size records and writes a
//! hint file. The client loads the file and answers queries against a server
//! that holds the same database.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use plinko::hints::{HintFileError, HintGeometry, HintSet};
use plinko::protocol::{compute_hints, Client, ProtocolError, ProtocolParams, Server};

/// Record size of a v3 `database.bin`, the default of `plinko_hints --record-size`.
const V3_RECORD_SIZE: usize = 40;

const MASTER_SEED: [u8; 32] = [0x5Cu8; 32];
const QUERY_SEED: [u8; 32] = [0xA7u8; 32];

/// A record in which every byte is non-zero and depends on the tag and on the
/// byte position.
fn make_record(tag: u64, record_size: usize) -> Vec<u8> {
    (0..record_size)
        .map(|k| {
            let mixed: u8 = (tag as u8).wrapping_mul(31) ^ ((tag >> 8) as u8);
            (mixed ^ (k as u8).wrapping_mul(97)) | 0x01
        })
        .collect()
}

/// A database: `num_entries` records one after another.
fn make_db(num_entries: usize, record_size: usize) -> Vec<u8> {
    (0..num_entries)
        .flat_map(|i| make_record(i as u64 + 1, record_size))
        .collect()
}

fn record(database: &[u8], record_size: usize, index: usize) -> &[u8] {
    &database[index * record_size..(index + 1) * record_size]
}

struct Scenario {
    name: &'static str,
    /// Value of `--record-size`; `None` leaves the default of the binary (40).
    record_size: Option<usize>,
    num_entries: usize,
    block_size: usize,
    lambda: usize,
    /// Value of `--threads`; `None` leaves the default of the binary.
    threads: Option<usize>,
    /// Pass `--constant-time` (needs the `tee` feature).
    constant_time: bool,
}

impl Scenario {
    fn record_size(&self) -> usize {
        self.record_size.unwrap_or(V3_RECORD_SIZE)
    }
}

struct Generated {
    database: Vec<u8>,
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

    let database: Vec<u8> = make_db(scenario.num_entries, scenario.record_size());
    fs::write(&db_path, &database).expect("write database");
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
                .record_size
                .map(|record_size| format!("--record-size={record_size}")),
        )
        .args(
            scenario
                .threads
                .map(|threads| format!("--threads={threads}")),
        )
        .args(scenario.constant_time.then_some("--constant-time"))
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

    Generated {
        database,
        hint_path,
    }
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
    database: &[u8],
    record_size: usize,
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
        let got: Vec<u8> = client
            .reconstruct_and_replenish(prepared, response)
            .expect("reconstruct");
        assert_eq!(
            got,
            record(database, record_size, index),
            "wrong answer for index {index}"
        );
        queried.push(index);
    }
    assert_eq!(queried.len(), count, "too few indices have a hint");
    queried
}

/// Loads the generated file into a client and runs queries, an update, and more queries.
fn run_online_phase(scenario: &Scenario, generated: &Generated, hints: HintSet) {
    let params: ProtocolParams = ProtocolParams::for_hint_set(&hints).expect("params of the file");
    let record_size: usize = scenario.record_size();
    let expected: ProtocolParams = ProtocolParams::new(
        scenario.num_entries,
        record_size,
        Some(scenario.block_size),
        scenario.lambda,
        None,
    )
    .expect("expected params");
    assert_eq!(params.record_size, record_size);
    assert_eq!(params.num_entries, scenario.num_entries);
    assert_eq!(params.num_blocks, expected.num_blocks);
    assert_eq!(params.num_regular_hints, expected.num_regular_hints);
    assert_eq!(params.num_backup_hints, expected.num_backup_hints);

    let mut database: Vec<u8> = generated.database.clone();
    let mut server: Server = Server::new(params.clone(), &database).expect("server");
    let mut client: Client =
        Client::from_hints(params.clone(), MASTER_SEED, QUERY_SEED, hints).expect("client");

    let n: usize = params.num_entries;
    let spread = |start: usize| (0..n).map(move |k| (start + k * 37) % n);
    let first: Vec<usize> =
        query_and_check(&mut client, &server, &database, record_size, spread(0), 5);

    // Change one index that was queried (it is in the client cache and in a
    // promoted hint) and many indices that were not.
    let mut updates: Vec<(usize, Vec<u8>)> = Vec::new();
    for index in (0..n).filter(|index| index % 3 != 1 || *index == first[0]) {
        let value: Vec<u8> = make_record(0xBEEF_0000 + index as u64, record_size);
        database[index * record_size..(index + 1) * record_size].copy_from_slice(&value);
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
    let second: Vec<usize> =
        query_and_check(&mut client, &server, &database, record_size, candidates, 6);
    assert!(
        second.iter().any(|index| index % 3 != 1),
        "no changed entry was queried after the update"
    );

    // A cached index is answered with the updated value.
    let prepared = client.prepare_query(first[0]).expect("re-query");
    let response = server.answer(&prepared.query).expect("answer");
    let got: Vec<u8> = client
        .reconstruct_and_replenish(prepared, response)
        .expect("reconstruct");
    assert_eq!(
        got,
        record(&database, record_size, first[0]),
        "stale value for a cached index"
    );
}

/// Runs the generator, compares its file with `compute_hints`, and runs the
/// online phase with the file.
fn check_generator_file(scenario: &Scenario) -> Generated {
    let generated: Generated = generate(scenario);
    let hints: HintSet = load(&generated.hint_path);
    let record_size: usize = scenario.record_size();
    assert_eq!(hints.record_size(), record_size);

    let params: ProtocolParams = ProtocolParams::for_hint_set(&hints).expect("params");
    let in_memory: HintSet =
        compute_hints(&params, &MASTER_SEED, &generated.database).expect("compute_hints");
    // Every byte position of a record reaches the parities, the last one included.
    for position in 0..record_size {
        assert!(
            (0..in_memory.num_regular()).any(|j| in_memory.regular_parity(j)[position] != 0),
            "byte {position} of the records is in no regular parity"
        );
    }
    assert!(
        hints == in_memory,
        "generator file differs from compute_hints"
    );

    run_online_phase(scenario, &generated, hints);
    generated
}

#[test]
fn generator_file_equals_the_client_hint_set_and_answers_queries() {
    // 97 entries in 8 blocks of 16: the last block is partly padding, and the
    // eighth block is all padding. No `--record-size`: 40-byte v3 records.
    let scenario = Scenario {
        name: "small",
        record_size: None,
        num_entries: 97,
        block_size: 16,
        lambda: 8,
        threads: None,
        constant_time: false,
    };
    let generated: Generated = check_generator_file(&scenario);
    assert_eq!(
        fs::metadata(&generated.hint_path).expect("metadata").len(),
        96 + 128 * 44 + 128 * 84,
        "hint file size for 40-byte records"
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
}

/// Record sizes that are not 40: one byte, a partial last word, a size that
/// is not a multiple of 8 above the old 32-byte window, and a large record.
#[test]
fn generator_file_answers_queries_for_other_record_sizes() {
    for (name, record_size) in [
        ("record_1", 1usize),
        ("record_13", 13),
        ("record_45", 45),
        ("record_256", 256),
    ] {
        let generated: Generated = check_generator_file(&Scenario {
            name,
            record_size: Some(record_size),
            num_entries: 97,
            block_size: 16,
            lambda: 8,
            threads: Some(3),
            constant_time: false,
        });
        assert_eq!(
            fs::metadata(&generated.hint_path).expect("metadata").len() as usize,
            96 + 128 * (4 + record_size) + 128 * (4 + 2 * record_size),
            "hint file size for record size {record_size}"
        );
    }
}

/// The constant-time path must write the same file as the standard path.
#[cfg(feature = "tee")]
#[test]
fn constant_time_path_writes_the_same_file_for_each_record_size() {
    for (name, record_size) in [
        ("ct_record_1", 1usize),
        ("ct_record_13", 13),
        ("ct_record_40", 40),
        ("ct_record_45", 45),
        ("ct_record_256", 256),
    ] {
        // 43 entries in 6 blocks of 8: the last block is partly padding.
        let scenario_for = |name: &'static str, constant_time: bool| Scenario {
            name,
            record_size: Some(record_size),
            num_entries: 43,
            block_size: 8,
            lambda: 8,
            threads: Some(1),
            constant_time,
        };
        let standard: Generated = generate(&scenario_for(name, false));
        let standard_bytes: Vec<u8> = fs::read(&standard.hint_path).expect("read hint file");
        let constant_time: Generated = check_generator_file(&scenario_for(name, true));
        let constant_time_bytes: Vec<u8> =
            fs::read(&constant_time.hint_path).expect("read hint file");
        assert!(
            constant_time_bytes == standard_bytes,
            "record size {record_size}: the constant-time file differs"
        );
    }
}

/// A file whose length is not a multiple of the record size has no meaning.
#[test]
fn generator_rejects_a_database_that_is_not_whole_records() {
    let dir: PathBuf = work_dir("partial_record");
    let db_path: PathBuf = dir.join("database.bin");
    let seed_path: PathBuf = dir.join("seed.hex");
    let hint_path: PathBuf = dir.join("hints.bin");
    // 8 records of 40 bytes: 320 bytes are not a multiple of 45.
    fs::write(&db_path, make_db(8, V3_RECORD_SIZE)).expect("write database");
    let seed_hex: String = MASTER_SEED.iter().map(|b| format!("{b:02x}")).collect();
    fs::write(&seed_path, seed_hex).expect("write seed file");
    if hint_path.exists() {
        fs::rename(&hint_path, dir.join("hints.bin.old")).expect("move old hint file");
    }

    let output: Output = Command::new(env!("CARGO_BIN_EXE_plinko_hints"))
        .arg("--db-path")
        .arg(&db_path)
        .arg("--record-size=45")
        .arg("--entries-per-block=4")
        .arg("--seed-file")
        .arg(&seed_path)
        .arg("--output")
        .arg(&hint_path)
        .output()
        .expect("run plinko_hints");
    assert!(!output.status.success(), "the run must fail");
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("multiple of the record size"),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(!hint_path.exists(), "no hint file must be written");
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
                record_size: None,
                num_entries: 120,
                block_size: 16,
                lambda: 8,
                threads: Some(threads),
                constant_time: false,
            });
            fs::read(&generated.hint_path).expect("read hint file")
        })
        .collect();

    let one_thread: HintSet = HintSet::read_from(files[0].as_slice()).expect("hint file");
    assert!(
        (0..one_thread.num_backup()).any(|b| one_thread
            .backup_parity_out(b)
            .iter()
            .any(|&byte| byte != 0)),
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
        record_size: None,
        num_entries: 1000,
        block_size: 512,
        lambda: 65,
        threads: None,
        constant_time: false,
    };
    let generated: Generated = generate(&scenario);
    let hints: HintSet = load(&generated.hint_path);
    assert_eq!(hints.num_regular() + hints.num_backup(), 66_560);
    run_online_phase(&scenario, &generated, hints);
}

#[test]
fn client_rejects_a_hint_set_that_does_not_belong_to_it() {
    let entries: Vec<u8> = make_db(40, V3_RECORD_SIZE);
    let params: ProtocolParams =
        ProtocolParams::new(40, V3_RECORD_SIZE, Some(8), 2, None).expect("params");
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
    let other_size: ProtocolParams =
        ProtocolParams::new(41, V3_RECORD_SIZE, Some(8), 2, None).expect("params");
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

    // The same geometry and hint counts; only the record size differs.
    let other_record_size: ProtocolParams =
        ProtocolParams::new(40, 32, Some(8), 2, None).expect("params");
    assert_eq!(
        mismatch(Client::from_hints(
            other_record_size,
            MASTER_SEED,
            QUERY_SEED,
            hints.clone()
        )),
        "record size"
    );

    let other_lambda: ProtocolParams =
        ProtocolParams::new(40, V3_RECORD_SIZE, Some(8), 3, None).expect("params");
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
        ProtocolParams::new(40, V3_RECORD_SIZE, Some(8), 2, Some(15)).expect("params");
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
    let wrong_blocks: HintSet = HintSet::new(
        HintGeometry {
            num_blocks: hints.geometry().num_blocks + 2,
            ..hints.geometry()
        },
        hints.seed_check(),
        hints.regular_cutoffs().to_vec(),
        hints.backup_cutoffs().to_vec(),
    )
    .expect("hint set");
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
    fs::write(&db_path, make_db(8, V3_RECORD_SIZE)).expect("write database");
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
    let entries: Vec<u8> = make_db(40, V3_RECORD_SIZE);
    let params: ProtocolParams =
        ProtocolParams::new(40, V3_RECORD_SIZE, Some(8), 2, None).expect("params");
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
