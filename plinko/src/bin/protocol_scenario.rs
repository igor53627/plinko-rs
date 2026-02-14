//! Deterministic protocol scenario for Rust-vs-Python transcript comparisons.

use clap::Parser;
use eyre::{bail, Result};
use plinko::protocol::{Client, Entry, ProtocolParams, Server, ENTRY_SIZE};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::fmt::Write as _;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(about = "Run deterministic in-memory Plinko protocol scenario")]
struct Args {
    /// Optional path to write JSON summary.
    #[arg(long)]
    output: Option<PathBuf>,

    /// Number of entries in the scenario DB.
    #[arg(long, default_value_t = 64)]
    num_entries: usize,

    /// Entries per block.
    #[arg(long, default_value_t = 8)]
    block_size: usize,

    /// Security parameter lambda (regular hints = lambda * block_size).
    #[arg(long, default_value_t = 40)]
    lambda: usize,

    /// Number of backup hints.
    #[arg(long, default_value_t = 32)]
    num_backup_hints: usize,
}

#[derive(Serialize)]
struct SummaryParams {
    num_entries: usize,
    entry_size: usize,
    block_size: usize,
    num_blocks: usize,
    num_reg_hints: usize,
    num_backup_hints: usize,
}

#[derive(Serialize)]
struct RoundSummary {
    indices: Vec<usize>,
    result_sha256: Vec<String>,
}

#[derive(Serialize)]
struct ScenarioSummary {
    params: SummaryParams,
    rounds: Vec<RoundSummary>,
    updates: Vec<usize>,
    remaining_queries: usize,
}

fn seed_from_label(label: &[u8]) -> [u8; 32] {
    let hash = Sha256::digest(label);
    let mut seed = [0u8; 32];
    seed.copy_from_slice(&hash);
    seed
}

fn deterministic_entry(index: usize, salt: &str) -> Entry {
    let mut hasher = Sha256::new();
    hasher.update(format!("{salt}:{index}").as_bytes());
    let hash = hasher.finalize();
    let mut entry = [0u8; ENTRY_SIZE];
    entry.copy_from_slice(&hash[..ENTRY_SIZE]);
    entry
}

fn sha256_hex(value: &Entry) -> String {
    let hash = Sha256::digest(value);
    let mut out = String::with_capacity(64);
    for byte in hash {
        let _ = write!(out, "{byte:02x}");
    }
    out
}

fn run_query_round(
    client: &mut Client,
    server: &Server,
    expected_db: &[Entry],
    indices: &[usize],
) -> Result<RoundSummary> {
    let mut result_sha256 = Vec::with_capacity(indices.len());

    for &index in indices {
        let prepared = client.prepare_query(index)?;
        let response = server.answer(&prepared.query)?;
        let got = client.reconstruct_and_replenish(prepared, response)?;
        let expected = expected_db[index];
        if got != expected {
            bail!("result mismatch at index={index}");
        }
        result_sha256.push(sha256_hex(&got));
    }

    Ok(RoundSummary {
        indices: indices.to_vec(),
        result_sha256,
    })
}

fn main() -> Result<()> {
    let args = Args::parse();

    if args.num_entries <= 63 {
        bail!("num_entries must be > 63 for the fixed scenario indices");
    }

    let params = ProtocolParams::new(
        args.num_entries,
        Some(args.block_size),
        args.lambda,
        Some(args.num_backup_hints),
    )?;

    let mut expected_db: Vec<Entry> = (0..params.num_entries)
        .map(|i| deterministic_entry(i, "db-v1"))
        .collect();

    let master_seed = seed_from_label(b"plinko_protocol_scenario_master_seed_v1");
    let query_seed = seed_from_label(b"plinko_protocol_scenario_query_seed_v1");

    let mut server = Server::new(params.clone(), &expected_db)?;
    let mut client = Client::offline_init(params.clone(), master_seed, query_seed, &expected_db)?;

    let round_one = run_query_round(&mut client, &server, &expected_db, &[0, 5, 17, 33, 63])?;

    let updates_input = vec![
        (5usize, deterministic_entry(1005, "update-v1")),
        (42usize, deterministic_entry(1042, "update-v1")),
    ];
    let updates = updates_input
        .iter()
        .map(|(idx, _)| *idx)
        .collect::<Vec<_>>();
    let deltas = server.apply_updates(&updates_input)?;
    client.apply_updates(&deltas)?;
    for (idx, value) in updates_input {
        expected_db[idx] = value;
    }

    let round_two = run_query_round(&mut client, &server, &expected_db, &[5, 42, 1, 33])?;
    let round_three = run_query_round(&mut client, &server, &expected_db, &[5, 17])?;

    let summary = ScenarioSummary {
        params: SummaryParams {
            num_entries: params.num_entries,
            entry_size: ENTRY_SIZE,
            block_size: params.block_size,
            num_blocks: params.num_blocks,
            num_reg_hints: params.num_regular_hints,
            num_backup_hints: params.num_backup_hints,
        },
        rounds: vec![round_one, round_two, round_three],
        updates,
        remaining_queries: client.remaining_backup_hints(),
    };

    let output = serde_json::to_string_pretty(&summary)?;
    println!("{output}");
    if let Some(path) = args.output {
        std::fs::write(path, format!("{output}\n"))?;
    }

    Ok(())
}
