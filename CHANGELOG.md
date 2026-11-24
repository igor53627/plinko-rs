# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added

- `plinko`: opt-in `cli` feature (enabled in `default` alongside `logging`) gates CLI binaries (`plinko_hints`, `cost_estimate`, `gen_synthetic`, `protocol_scenario`, and `bench_gpu_hints` with `gpu`), `plinko::db::Database40`, and CLI/OS dependencies (`clap`, `eyre`, `indicatif`, `memmap2`, `serde`, `serde_json`, `rand/std`, `rand/std_rng`). Building with `--no-default-features` targets `wasm32-unknown-unknown` cleanly with no host/OS imports, and `hint_init` clamps worker threads to 1 on `wasm32` without calling `std::thread::scope`. CI now runs `wasm32-unknown-unknown` checks for `--no-default-features` and `--no-default-features --features tee`.
- `plinko::iprf`: `SwapOrNotSr::forward_many` and `Iprf::forward_many` evaluate independent PRPs/iPRFs in 8-way interleaved lockstep so pipelined hardware AES units overlap across lanes while returning bit-identical outputs to scalar `forward`.
- `plinko::protocol::Client`: `Client::threads` and `Client::set_threads` configure worker threads for `prepare_query` and `apply_delta` (defaulting to `hint_init::available_threads`, always 1 on `wasm32`).
- `plinko::schema40::CodeStoreError`: replaces `eyre::Result` in `CodeStore::from_bytes` so `schema40` has no `eyre` dependency.
- `docs/protocol-visualization.html` has a Russian version in the same file. A switch `English / Русский` beside the theme switch changes the text of the article and the text that the figures make. The page takes the language from the address (`?lang=ru` or `?lang=en`), then from the saved choice (`localStorage` key `plinko-lang`), then from the language of the browser. Pseudocode (but not its comments), formulas and the names `iPRF`, `HintInit` and `lambda` stay in English. The English text and the figures do not change. The two switches are now in a row of their own below the introduction, because they do not fit beside the title.

### Changed

- `plinko`: removed unused dependencies `ctr`, `once_cell`, and build-dependency `cc` from `plinko/Cargo.toml`.
- `plinko::iprf` and `plinko::binomial`: `SwapOrNotSr` derives round keys in 16-block `encrypt_blocks` batches and uses branchless partner/swap arithmetic without modulo division; `coin_binomial` encrypts only the AES blocks needed for `remaining` coins.
- `plinko::protocol::Client`: `prepare_query` uses `Iprf::forward_many` and worker threads (for `>= 64` real blocks), `apply_delta` computes iPRF inverses on worker threads (for `>= 8` changed records), and backup hint availability (`remaining_backup_hints` and `promote_backup`) runs in O(1) amortized time instead of scanning `backup_available`. On 1,048,576 records (`w = 1024`, `c = 1024`, `lambda = 64`, Apple M3 Ultra, `--cfg aes_armv8`), warm `prepare_query` median dropped from 11.58 ms to 2.82 ms on 1 thread (4.1x) and 0.64 ms on 32 threads (18.1x); a 200-change `apply_delta` dropped from 67.15 ms to 7.45 ms on 32 threads (9.0x).
- `docs/protocol-visualization.html`: on a screen narrower than 40rem, the table of the Swap-or-Not rounds uses a smaller font and keeps each result on one line.
- `README.md` header. The title is `plinko-rs`, the name of the repository; before, it was `Plinko RS`. The `Ask DeepWiki` badge uses a static shields.io image, because `deepwiki.com/badge.svg` answers the image proxy of GitHub with HTTP 429 and the badge showed as a broken image. New badges: CI status, license and paper. The demo badge links to the root of the GitHub Pages site.
- The workflow `Deploy to GitHub Pages` (`.github/workflows/pages.yml`) publishes only the protocol demo, `docs/protocol-visualization.html`, as `index.html` and under its own name. Before, it published the whole `docs/` directory, with the PDF of the paper and the transcript of the talk, and it failed on each push because GitHub Pages was not enabled. It runs only when the demo or the workflow changes.
- `docs/protocol-visualization.html` is written again as an article with five figures, after the fixes below. All figures use one toy model: a keyed hash, a Swap-or-Not permutation, the PMNS tree, the iPRF, and the hints of a database of 24 records. A tree of the sampler replaces the physics board, so the page loads no external file and has no sound. The last figure is new and has two actions: a query with its answer and the promotion of a backup hint, and the update of a record. The page has a light theme and a dark theme.

### Fixed

- `docs/protocol-visualization.html` had a horizontal scroll of the page at widths from 834 px to 942 px and at 384 px and below. The notes beside the text now go below it on a screen narrower than 64rem (before: 52rem), where they are at most 40rem wide, because their tables do not fit in a narrower column. At 22rem and below, a result in the table of the rounds can wrap.
- `docs/protocol-visualization.html` (the protocol demo):
  - The HintInit demo sent each entry to every hint whose subset holds the block. It now uses a toy iPRF inverse, as the pseudocode beside it does, and shows the two parities of a backup hint.
  - The pseudocode blocks had no indentation in a browser. The HintInit pseudocode used two index ranges for the backup hints.
  - The page showed the wrong title of the paper, gave `O(1)` for the client query and for the hints of an entry, and named the Ethereum state as a use. The text now agrees with `docs/protocol_overview.md`, and each demo states its toy parameters.
  - The board drew one bin too many, and `Reset` did not stop `Drop 10` and `Drop 50`. `Compute Inverse` could show the result of another `y`.
  - The page loads matter.js with a subresource integrity hash, and the other demos run if that file does not load. Sound is off at the start. The page has no horizontal scroll on a narrow screen.

## [0.2.0] - 2026-11-19

The target of the repository is now a PIR library for a database of fixed-size records
(`docs/protocol_overview.md`). The public API of the `plinko` crate, the file formats and the
hints for a seed differ from those of 0.1.0.

Not in this release:

- Lookup by key: a query takes an index (TASK-22, proposal in `docs/keyword_lookup.md`).
- Integrity of an answer or of a delta: a decision, see `docs/integrity.md`.
- A network transport, a program that writes delta files, and a hint refresh before the query limit.
- Plinko hints from the GPU path: it is a throughput benchmark (TASK-21).

### Added

- **Check on a second platform** (2026-11-19, AMD EPYC 9375F, x86-64 Linux, rustc 1.91.1): fmt, clippy, the tests of the default build and of the features `tee,gpu`, the ignored test above 65536 hints, the e2e script, the differential script and the hook tests pass. The hint files of the standard path and of the constant-time path (24 blocks of w = 1024) and the hint file of a run with the mainnet hint count (24 blocks of w = 65536, 1,073,741,968 bytes) are byte-identical to the files that an Apple M3 Ultra (arm64 macOS) writes from the same source. `docs/hint_generation.md` has the timing: 1.04 ms of thread time for each record
- **CUDA GPU hint generation pipeline**: end-to-end GPU-accelerated hint generation on NVIDIA H200
  - ChaCha8 iPRF initially, upgraded to ChaCha12 for security margin
  - Warp-level parallelism across CUDA threads for throughput scaling
  - Parallel CPU baseline with full ChaCha8 iPRF for comparison benchmarks
  - Modal scripts for H200 production benchmarking (`modal_bench_mainnet.py`, `modal_bench_distributed.py`)
- **40-byte schema v3**: compact on-disk format saving 17% storage vs 48-byte v2
  - New `schema40` module with CUDA kernel support
  - Mainnet v3 dataset published to PIR bucket
- **VRAM compaction and expansion strategies**: 40B → 32B compaction in VRAM with 40B → 48B expansion for aligned GPU loads
  - Chunked compaction for correctness on large datasets
  - Expansion strategy documentation (`EXPANSION_40B_48B.md`, now in `docs/archived/optimization/`)
- **Crate rename**: `state-syncer` renamed to `plinko` (issue #27)
  - Workspace, Docker, documentation, and binary references updated
  - Extractor binary renamed to `plinko`
  - LICENSE and crates.io metadata added for publishing
- **Formal verification**: eliminated axioms across Rocq specs (#78)
  - Bijection properties fully proved in `SwapOrNotSrSpec.v`
  - `BinomialSpec.v` and `IprfSpec.v` now have zero axioms
  - `binomial_sample_range_aux` axiom eliminated
  - Formal directory canonicalized under `plinko/formal` (#52)
- **Documentation restructuring**
  - README split into subdocs (`protocol_overview.md`, `verification.md`, `update_strategy.md`)
  - Ops docs added (`ARCHITECTURE.md`, `DEPLOYMENT.md`, `FEATURE_FLAGS.md`, and `API_ENDPOINTS.md`, now in `docs/archived/`)
  - GPU benchmark results and optimization kanban tracking
  - Legacy docs archived, stale binaries removed

- **Hint file**: `plinko_hints --output <file>` writes all hints to one versioned file (`PLNKHINT`, version 1: geometry, hint counts, seed check, cutoffs and parities, BLAKE3 checksum; see `docs/data_format.md`). The new library module `plinko::hints` (`HintSet`, `RegularHint`, `BackupHint`) is shared by the generator and the client.
- **Client loads a hint file**: `Client::from_hints` builds a client from a `HintSet` and rejects another geometry, other hint counts, or another master seed (`ProtocolError::HintSetMismatch`). `ProtocolParams::for_hint_set` derives the parameters from the file. `protocol::compute_hints` runs HintInit in memory; `Client::offline_init` now calls it.
- **Keyword lookup proposal** (`docs/keyword_lookup.md`): `database.bin` as a cuckoo hash table, so that a client computes the positions of a key without the mapping files. The document compares four layouts with computed mainnet numbers and a build simulation, and recommends 2 hash functions with 4 slots per bucket. Not implemented.
- **Trust model** (`docs/trust_model.md`): the client runs HintInit and holds the master seed. The document states what the server learns, the cost for the client, and the points that the privacy claim does not cover. The constant-time (TEE) mode and the GPU path are marked as experimental and not part of the model in `README.md`, `docs/DEPLOYMENT.md` and `docs/constant_time_mode.md`.
- **TEE and GPU code are opt-in build features (breaking)**: a build of the `plinko` crate without features has no `constant_time` module, no `IprfTee`, `SwapOrNotSrTee`, `SwapOrNotTee` or `coin_binomial_ct`, no `plinko_hints --constant-time`, no `gpu` module and no `bench_gpu_hints` binary. The new features `tee` and `gpu` add them; `cuda` and `parallel` imply `gpu`, so the GPU build commands are unchanged. The constant-time variants moved from `iprf.rs` to `plinko/src/iprf/tee.rs`. CI runs the tests without features and with `tee,gpu`.
- `plinko_hints --threads <n>`: the standard path processes database blocks on worker threads (default: the number of available cores). The hint file is byte-identical for every thread count. Measured with w = 65536 and 16,777,216 hints: 24 blocks in 41.9 s on 24 threads and in 117.0 s on 8 threads. Extrapolated mainnet cost: about 300 core-hours (see `docs/hint_generation.md`).
- `plinko_hints --seed-file <file>` reads the master seed from a file, so that the seed is not in the process list. `--output` requires `--seed` or `--seed-file`.
- End-to-end test `plinko/tests/hint_file_e2e_test.rs`: runs the `plinko_hints` binary, compares its file with the in-memory hint set, and checks queries and an update; an ignored test does the same with 66,560 hints.
- `scripts/local_e2e_hint_pipeline.sh` compares the hint files of the fast and the constant-time run byte for byte.

- **PMNS tree reuse**: `Iprf::table()` and `IprfTee::table_ct()` compute all bin boundaries of one iPRF once; HintInit builds one table per database block in both paths and in `protocol::compute_hints`. `Iprf::with_cached_levels` keeps the top tree levels after the first call; the protocol client caches 8 levels per block (`CLIENT_CACHED_PMNS_LEVELS`). Outputs do not change. Measured at w = 65536 with hardware AES: `forward` 445 us to 28.5 us, `inverse` 7.56 ms to 7.02 ms (the inverse PRP dominates).
- The constant-time path stops with a panic when a PMNS bin holds more than `MAX_PREIMAGES` balls. It truncated the bin silently before and gave a wrong parity.

- **Database versions (breaking)**: the publisher of a database numbers each state. `Server::new`, `Client::offline_init`, `compute_hints` and `compute_hints_with_threads` take the version; `plinko_hints --database-version <n>` (default 0) sets it for a hint file. `Response::database_version` carries the version of an answer. `Client::reconstruct_and_replenish` accepts an answer only of the version of the hints (`ProtocolError::DatabaseVersionMismatch`); before, an answer from another database state gave a wrong record with no error. `docs/update_strategy.md`, `docs/online_protocol.md`.
- **Integrity statement** (`docs/integrity.md`): the library gives no integrity of an answer or of a delta; this is a decision. The document states what the client checks and what it must trust in HintInit, in an answer and in a delta, the effect of a wrong answer (the wrong record stays in the cache and spreads through the promoted backup hint; the messages of the client do not change), three mechanisms that were not selected with their cost, and what an application can do (compare the HintInit hash with a value from the publisher, put an authenticator in each record). New test `a_wrong_answer_gives_a_wrong_record_and_the_same_later_messages` (TASK-24)
- **Client state file**: `Client::write_state` writes the state of a client that sent queries (magic `PLNKSTAT`, format version 1: the hint set, the regular hints that were sent, the answered queries with their promoted backup hints and records, the queries in flight, the retired backup hints, BLAKE3 checksum; see `docs/data_format.md`). `Client::read_state` builds a client from it and returns the queries in flight with the same message bytes. Before, a client that sent a query could start again only from its hint file, and it then sent hints a second time. `Client::retire_hints` removes the hints of queries that a state does not hold, from a log of `PreparedQuery::hint_index` that the caller keeps. The format cannot detect a state that is older than the last query; `docs/online_protocol.md` (Client State And Restart) has the rules for the caller. `Client::reconstruct_and_replenish` returns `NoBackupHintsLeft` with no change of the client if `retire_hints` removed the last backup hint (TASK-32)
- **Delta format and `plinko::delta`**: `Delta` holds the records that differ between two database versions as `old XOR new`, with a writer and a reader (`PLNKDLTA`, format version 1: record size, number of records, `from_version`, `to_version`, changes in index order, BLAKE3 checksum; `docs/data_format.md`). `Delta::new` combines the changes of one record and removes a zero change. `Client::apply_delta` accepts only a delta that starts at the version of the hints, so a delta that repeats or comes too early changes no hint. A delta is not bound to a database: the caller must give a client only the deltas of the database of its hints. Tests: a sequence of deltas through files on a hint set from a file, then queries of changed and unchanged records; the parities after the deltas equal those of HintInit on the new database.
- `Client::database_version`, `Server::database_version`, and `Client::unused_hints`, which returns the hint set until the client builds its first query, so that a client can write its hints after it applied deltas.
- `docs/update_strategy.md` states the rule for a database that grows: a delta keeps the number of records, `w` and `c`; a publisher reserves zero records for later data; a change of `w`, `c` or the record size needs a new HintInit. Test `padding_records_count_as_zero_records`.

### Changed

- **Exact integer PMNS sampler (breaking)**: `binomial_sample` and `binomial_sample_tee` are replaced by `coin_binomial` and `coin_binomial_ct`. A split is the number of set bits among the first `count` bits of an AES-128 counter keystream of the tree node, which is exactly Binomial(count, 1/2) for a uniform keystream (computationally indistinguishable from it under the AES PRF assumption) and identical on every platform and in both modes. All iPRF outputs change.
- **iPRF range must be a power of two (breaking)**: `Iprf::new` and `IprfTee::new` panic otherwise, `ProtocolParams::new` returns an error, and `plinko_hints --entries-per-block` rejects other values. The default `w` of `plinko_hints` is now the smallest power of two >= sqrt(N) (mainnet v3: 65536 instead of 42826), the same as the protocol client.
- **PMNS key separation**: the keystream key is `SHA-256(key || "pmns" || n || m)[0..16]`; the node id is no longer a truncated hash.
- Removed the `puruspe` dependency.
- **Block subsets from a PRF cutoff (breaking)**: a hint stores its block subset as one 32-bit cutoff (RMS24 section 3.2, with the subset sizes of Plinko Fig. 7). Block `k` is in the subset of hint `j` when `PRF("select", j, k) < cutoff`. The new library module `plinko::subsets` (`SubsetPrf`, `find_cutoff`) is shared by `plinko_hints` and the protocol client. Subsets differ from the earlier `ChaCha20Rng` + `index::sample` derivation for the same master seed.
- A hint with a tie at the cutoff boundary is invalid (cutoff 0): its parities stay zero and the client does not use it. `plinko_hints` logs `invalid_regular` and `invalid_backup` in place of `zero_regular_seeds` and `zero_backup_seeds`.
- Removed `BlockBitset`, `compute_regular_blocks`, `compute_backup_blocks`, `block_in_subset`, `derive_subset_seed`, `SEED_LABEL_REGULAR`, `SEED_LABEL_BACKUP`, and the `memory_bench` benchmark. Subset state is 4 bytes per hint in the generator and in the client.
- **Commit hashes changed (2026-11-19)**: 48 commits of the history before this release had a private email address of the owner as author, and three commit messages had it in a `Co-Authored-By` line. The address is now `igor53627@users.noreply.github.com`. The trees, the author names, the dates and the committers are the same. The first 15 commits keep their hashes; each later commit has a new hash (the tip of the old `main` was `906dcbc`, it is `7c1598d`), and 57 commits lost their signature. A clone of the old history must be made again. The commit references in the documents and in `backlog/` use the new hashes
- **Agent rules**: `AGENTS.md` has a local-only phase section (no GitHub, `backlog/` tasks, `jj`, CodeRabbit CLI review) and no longer describes the Codex Cloud workflow, the remote Rocq checker, DeepWiki updates, or PR formatting. It states that the formal models are not checked against the Rust source. The module table in `docs/hint_generation.md` has no file-size column. Since the release the section is "Workflow": changes go to `main` through a pull request on GitHub (`igor53627/plinko-rs`), work is tracked in `backlog/` and not in GitHub issues. The rules for the host `aya` are less strict: builds, tests and benchmarks in `/root/plinko-work` need no permission
- **Batched inverse PRP**: `SwapOrNotSr::inverse_range` and `SwapOrNotSrTee::inverse_many_ct` invert all balls of one offset in one call. A round derives its round key once and computes the swap decisions as one batch of AES blocks. `Iprf::inverse`, `IprfTable::inverse`, `IprfTee::inverse_ct` and `IprfTeeTable::inverse_ct` use them; outputs are unchanged. Measured at w = 65536 and 16,777,216 hints, one thread, hardware AES: 7.04 ms to 0.52 ms per entry in the standard path, 68.3 ms to 16.7 ms in the constant-time path.
- The constant-time path runs every Sometimes-Recurse stage for every slot. Before, the time of the inverse PRP depended on the position of the balls of an entry (15 us to 213 us per value for a domain of 65536).
- **Online protocol (breaking)**: `Client::prepare_query` reserves the hint and one backup hint; the hint leaves the client in `reconstruct_and_replenish` (or `abandon_query`, new). A retry sends the same `Query` again. `prepare_query` returns `NoBackupHintsLeft` before it builds a message, and `QueryInFlight` for an index that has no answer yet. `PreparedQuery::consumed_hint_index` is now `hint_index`; `Client::queries_in_flight` is new. The rules and the open points are in `docs/online_protocol.md`.
- The client tries the hint candidates of an index in a random order (Fig. 7, `GetHint`).
- **`Iprf::forward` panics for an input outside the domain (breaking)**. It returned bin 0, which is a valid output.
- **Formal model of PMNS follows the coin sampler**: `BinomialSpec.v` defines the left count of a node as the number of set coins among the first `count` coins of a keystream, as `coin_binomial` in `plinko/src/binomial.rs` does. `IprfSpec.v` has one symbol `node_coin` in place of `prf_eval` and `encode_node`. The old file specified a formula that the Rust code never used.
- **Trust base is generated**: `plinko/formal/scripts/trust_base.py` writes `plinko/formal/docs/TRUST_BASE.md` from the `.v` files and `docs/trust_base_notes.toml` (class and meaning of each `Axiom`, `Parameter` and `Admitted`, the known differences between the models and the Rust code). `make verify-axioms` fails when an item has no note or the file is out of date; it printed a warning before. `proofs/Assumptions.v` prints the assumptions of the main theorems.
- **Documents state that the proofs are about models**: `docs/verification.md`, `plinko/formal/README.md`, `README.md` and the headers of the `sims/` and `linking/` files no longer describe the Rust code as verified. `docs/Plinko.v` is marked as a sketch.
- **CI**: `formal.yml` runs the trust base check and compiles `docs/Plinko.v` (`make sketch`). The Rocq-of-Rust job had `continue-on-error`; it is now the manual workflow `rocq-of-rust.yml` and fails when a step fails.
- **Scope**: the target is a PIR library for a database of fixed-size records. The full Ethereum state is not a target (`backlog/decisions/decision-1`); `README.md`, `docs/protocol_overview.md`, `docs/ARCHITECTURE.md`, `docs/trust_model.md` and `AGENTS.md` state this, with the cost for a client as the reason. The extractor and the mainnet numbers stay as test data and a scale reference.
- **Performance claims replaced by measurements**: `docs/protocol_overview.md` and `README.md` no longer say "O(1) hint search", "O(1) updates" or "Ethereum-scale practicality". They give the work of each step and the numbers of runs on 16,777,216 records (40-byte records: HintInit 0.57 ms CPU for each record, query 53 ms of client CPU and 8.7 KB, update 0.53 ms for each changed record). `docs/update_strategy.md` separates what is implemented (`apply_updates` in memory) from what is not (delta feed, inserts, database version).
- **Benchmark documents marked as historical**: `tee-test/SEV-SNP-BENCHMARK.md` (a generator of December 2025 with regular hints only and no iPRF), `docs/BENCHMARK_RESULTS.md` and `docs/gpu_benchmark_commands.md` (GPU throughput kernel, not Plinko hints; the reports say ChaCha8 where the kernel has 12 rounds) have a note at the top. `docs/benchmarks.md` lists the numbers of the current code first. `plinko/docs/delta-format.md` is marked as a proposal that is not implemented.
- `tee-test/run-benchmark.sh` describes the current generator in place of "per-hint BLAKE3" and takes `PLINKO_DIR`, `DB_PATH`, `HINTS_BIN`, `THREADS` and `LAMBDA` from the environment.
- `README.md` names the paper authors Hoover, Patel, Persiano and Yeo.
- `docs/data_format.md` has a section "Limits of the extractor" (no single snapshot, block number source, lossy records, unstable CodeIDs, partial files after a failed run). These limits are not planned to be removed; the extractor is a source of test data.
- The TAG comment in `plinko/src/schema40.rs` gives the collision probability over the mainnet record counts (8.7% for all records) in place of the probability for one pair.
- `docs/protocol_overview.md` lists the open work for a general library (lookup by key, database version and delta feed, check of an answer); `docs/trust_model.md` and `docs/keyword_lookup.md` follow the rewritten backlog tasks.
- **Record size is a parameter (breaking)**: `plinko_hints --record-size <bytes>` (1 to 1,048,576; default 40) and `ProtocolParams::new(num_entries, record_size, block_size, lambda, num_backup_hints)`. A hint parity is the XOR of whole records. Before, the hints covered the first 32 bytes of a 40-byte record, so a client could not read bytes 32 to 39 (the TAG of a storage record). A database file that is not a whole number of records is an error.
- **Hint file version 2 (breaking)**: the header has a `record_size` field, a regular hint has `4 + r` bytes and a backup hint has `4 + 2r` bytes (`docs/data_format.md`). No code reads version 1; run `plinko_hints` again. `Client::from_hints` rejects a hint set with another record size.
- **`plinko::hints` (breaking)**: `HintSet` has private fields and flat parity buffers. `HintSet::new` makes a set with zero parities from the geometry, the seed check and the cutoffs; `regular_parity`, `backup_parity_in`, `backup_parity_out` and their `_mut` forms give one parity as a slice. `RegularHint` and `BackupHint` are removed. `HintFileError::InvalidRecordSize` replaces `UnsupportedParitySize`. `MAX_RECORD_SIZE` is 1 MiB.
- **`plinko::protocol` (breaking)**: `Entry` and `ENTRY_SIZE` are removed. `Server::new`, `Client::offline_init` and `compute_hints` take the database as one byte slice of `num_entries * record_size` bytes. The parities of `Response`, `EntryUpdate::delta` and the result of `reconstruct_and_replenish` are `Vec<u8>`. New: `Server::record` and `ProtocolError::RecordSizeMismatch`. `Server::apply_updates` and `Client::apply_updates` check all items before the first change. A response with a parity of another size ends the query as `abandon_query` does. `ProtocolParams::new` rejects a database that does not fit in memory with its padding (`num_blocks * block_size * record_size` bytes); it could overflow before.
- `constant_time::ct_xor_masked(dst, src, mask)` works on slices of equal length and replaces `ct_xor_32_masked` (breaking, feature `tee`).
- The standard path of `plinko_hints` does no iPRF inverse for a padding record of the last block. The hint file does not change, because a padding record is zero.
- `cost_estimate` counts 44 bytes per regular hint and 84 bytes per backup hint (cutoff + parities of 40 bytes). The labels of its text output said 36 and 68 bytes next to totals for 44 and 84; they now come from the same constants.
- **Measured for record sizes of 32, 40 and 256 bytes** (16,777,216 records, `docs/protocol_overview.md`): the CPU time of HintInit, of a query and of an update is within 1% to 2% for the three sizes, because the iPRF calls take nearly all of it. The hint file is 54.5 MB, 67.1 MB and 406.8 MB; the answer of the server takes 0.11 ms, 0.13 ms and 0.38 ms. At the mainnet hint count HintInit takes 2.1 GB of memory with 40-byte records (`docs/trust_model.md` said 1.8 GB, measured with 32-byte parities).
- Tests for record sizes that are not 32 bytes: `protocol_e2e_test.rs` (1, 7, 8, 33, 40, 45, 256 and wrong sizes), `hint_file_e2e_test.rs` from the `plinko_hints` binary (1, 13, 40, 45, 256; the constant-time file is equal for 1, 13, 40, 45 and 256), `hint_gen/path_tests.rs` (1, 5, 8, 13, 32, 45, 256).
- **HintInit is the library module `plinko::hint_init` (breaking)**: `hint_cutoffs`, `fill_parities` and, with the `tee` feature, `fill_parities_ct`. `plinko_hints` and `protocol::compute_hints` call the same code; before, the streaming loop and the block key derivation existed in the binary and in `plinko/src/protocol.rs`. Removed from the binary: `hint_gen/fast_path.rs`, `hint_gen/ct_path.rs`, `hint_gen/keys.rs`, `driver::hint_cutoffs` and `types::record_at`. The hints for a seed do not change: the hint file of a 24-block run (w = 65536) is byte-identical to the file of the code before the change.
- **Worker threads in all HintInit steps**: the cutoffs are computed on the worker threads; `protocol::compute_hints` and `Client::offline_init` use all available cores, and the new `protocol::compute_hints_with_threads` takes the thread count. Measured (Apple M3 Ultra, hardware AES): cutoffs of 262,144 hints at c = 27,988 in 10.58 s on 1 thread and 0.51 s on 24 threads; `compute_hints` on 24 blocks of w = 4096 in 47.7 s on 1 thread and 2.36 s on 24 threads. For `plinko_hints` at the mainnet geometry the gain is the cutoff phase only: about 11 minutes of about 13 hours (extrapolated).
- **`plinko_hints --constant-time --threads <n>`** is accepted (`tee` feature). The default of the constant-time path stays 1 thread: workers that update the same parity word delay each other, and which words those are depends on the hint indices (`docs/constant_time_mode.md`). Measured on 24 blocks of w = 1024: 294.6 s on 1 thread, 14.3 s on 24 threads, the same hint file as the standard path. The constant-time path now holds the parity accumulators in addition to the hint set, as the standard path does.
- `docs/constant_time_mode.md` states that the cutoffs are not computed in constant time (`select_nth_unstable` over the select values of a hint).

- **PRP round keys without a division, hint file version 3 (breaking)**: the round key of a Sometimes-Recurse stage of size `n` is a 128-bit AES output `r` mapped to `floor(r * n / 2^128)` (`reduce_below`, `sr_round_key` in `plinko/src/iprf.rs`). It was a 64-bit AES output modulo `n`: a division with a key-derived dividend in both paths. The hints of a seed change, so `HINT_FILE_VERSION` is 3 and the reader rejects version 2; run `plinko_hints` again. For a hint count that is not a power of two, the bias of a round key falls from `n / 2^64` to `n / 2^128` (`docs/hint_generation.md`, "Round Keys"). The 24-block run at w = 65536 on 24 threads takes 40.5 s (41.4 s before; inside the variation between runs).
- **Constant-time path (feature `tee`)**: `SwapOrNotSrTee::inverse` and `SwapOrNotSrTee::forward` run every round of every stage under a mask, so their time does not depend on the input (`test_sr_tee_time_does_not_depend_on_the_input`, release build only; the test fails with the old function). `plinko_hints --constant-time` computes the cutoffs with the new `hint_init::hint_cutoffs_ct` (`subsets::find_cutoff_ct`, `SubsetPrf::cutoff_ct`, `constant_time::ct_lt_u32`): a bit-by-bit search with a fixed amount of work for each hint, 102 us against 39 us for each hint at c = 27,988. The cutoffs and the hint file are equal to those of the standard path.
- `SwapOrNotTee` (feature `tee`) has no division on its input and panics for an input that is not below the domain (breaking). Its round key keeps the reduction modulo the domain of `SwapOrNot`.
- `docs/constant_time_mode.md` states one claim (on one thread, the sequence of operations depends only on public values), the attacker that it covers in part (one that measures time: the number and the kind of the operations are fixed, the cache-dependent time of the accesses at the hint index is not) and the attacker that it does not cover (one that sees memory accesses, such as the host of a confidential VM). It states what the accesses at the hint index reveal (the iPRF mapping, and from it the queried index up to `c/2` candidates) and lists what remains. The help text of `--constant-time` makes the same claim.

- **Hint file version 4 (breaking)**: the header has 112 bytes and holds the database version of the parities, and the version and the BLAKE3 hash of the database bytes that HintInit read (`HintSet::database_version`, `HintSet::init_database`, `DatabaseSnapshot`). `HintSet::new` takes a `DatabaseSnapshot`. `plinko_hints` hashes the database in one pass before the parity pass (1.1 s for 2 GiB in the page cache) and logs the hash. The hint records are byte-identical to those of version 3 (checked on the 24-block run at w = 65536); the reader rejects version 3, because it has no database version.
- **Updates go through deltas (breaking)**: `Server::apply_updates` moves the database to the next version and returns a `Delta`; `Client::apply_delta` replaces `Client::apply_updates`, and `protocol::EntryUpdate` is removed.
- **`Client::reconstruct_and_replenish` takes references and leaves the client unchanged on an error (breaking)**: after `DatabaseVersionMismatch` or `RecordSizeMismatch` the query stays in flight, and the caller can apply a delta or send the query again. Before, a malformed answer ended the query as `abandon_query` does.
- `plinko/docs/delta-format.md` (a proposal of 2025 that differed from the code) is removed; the delta format is in `docs/data_format.md`.
- `docs/update_strategy.md`, `docs/online_protocol.md` and `docs/trust_model.md` state that the hint file holds no state of the online phase: a client that loads its hint file again after queries uses hints a second time. A client that sent queries starts again from a state file (`Client::write_state`, `Client::read_state`); the format cannot detect a state that is older than the last query.
- **iPRF cipher**: upgraded from ChaCha8 to ChaCha12 for GPU hint generation (security margin)
- **Schema default**: v3 (40-byte) schema used by default in Modal benchmark scripts

### Fixed

- **Findings of the review of the 0.2.0 pull request** (TASK-33):
  - `plinko_hints` refuses an `--output` that is the database file or the seed file. The hint file takes the place of its path with a rename, so the run replaced the input and ended with exit code 0. It also refuses an `--output` that is a directory before HintInit, not after it.
  - `ProtocolParams::new` and `plinko_hints --entries-per-block` reject a block size above 2^32 (`protocol::MAX_BLOCK_SIZE`). Such a value passed validation and the iPRF constructor then stopped with a panic.
  - `plinko_hints` rejects a `--lambda` or `--backup-hints` whose hint count does not fit in 64 bits. In a release build the product wrapped, and the run wrote hints for a small count.
  - `Server::new`, `compute_hints`, `Client::offline_init` and `Client::from_hints` reject `ProtocolParams` with a changed public field (`InvalidParams`): a block size or a block count that does not hold `num_entries` records gave a server and hints that ignored the last records. `hint_init::fill_parities` and `fill_parities_ct` stop with a panic for such a geometry.
  - The parity accessors of `HintSet` check the hint index before they compute the byte offset. In a release build an index such as 2^63 with a record of 2 bytes wrapped to the parity of hint 0.
  - `Client::retire_hints` keeps `Client::unused_hints` when the list removes no hint (a hint with no valid cutoff). The client refused to export an unchanged hint set, and a state round trip gave another value.
  - The scalar `SwapOrNotSr` and `SwapOrNot` compute the partner of a round without `k_i + n`, which overflowed for a domain above 2^63 (a panic in a debug build, a wrong value in a release build). No hint set has such a domain; the values for all other domains do not change.
  - `scripts/local_e2e_hint_pipeline.sh` writes the seed to a new file, so that a seed file from an earlier run in a caller-supplied `WORK_DIR` does not keep a permissive mode.
  - `cost_estimate` derives `w`, `c` and the hint counts with `ProtocolParams::new`, the rule of `plinko_hints` and of the client. It used the geometry of the GPU benchmark (`db::derive_plinko_params`): for mainnet it printed w = 131072, 16,777,216 regular hints and 2.00 GB of hints, where a `plinko_hints` run gives w = 65536, 8,388,608 regular hints and 1.00 GB. The GPU section and the VRAM lines keep the geometry of the benchmark and are labelled as a throughput benchmark that does not compute Plinko hints; the JSON object `gpu_compute` has the new fields `chunk_size`, `set_size` and `total_hints`.
  - The CPU section of `cost_estimate` is the HintInit cost of the current `plinko_hints`: 0.59 ms of CPU time for each record in the standard path and 16.7 ms in the constant-time path (`--tee`), from the measurements in `docs/hint_generation.md` and `docs/constant_time_mode.md`. For mainnet it prints 301 core-hours (8,509 with `--tee`). It printed 20.9 minutes on 64 vCPUs (22 core-hours) from the throughput of the generator of December 2025 (55.8 MB/s) and a TEE factor of 2.6, and neither number describes this code. The standard number is scaled with lambda; only lambda = 128 was measured. `--tee` adds no overhead of a TEE, and it refuses a lambda above 128, as `plinko_hints --constant-time` does. The JSON object `cpu_compute` has the new fields `ms_per_entry` and `core_secs`; `throughput_mbps` is now derived from them.
  - `scripts/local_e2e_hint_pipeline.sh`, `differential_e2e_reference.sh` and `differential_protocol_transcript.sh` remove their work directory on exit only if they made it. A caller-supplied `WORK_DIR` was removed with all its files unless `KEEP_ARTIFACTS=1`.
  - `rocq-of-rust.yml` builds the translator after each checkout (the cache key does not hold the revision of the fork) and no longer asks for `pull-requests: write`.
  - The tests of `hint_file_e2e_test.rs` use a directory with the process number and remove it when they pass: two test runs with one target directory failed on the temporary files of each other. Two test loops over `prepare_query` have a bound.
  - Documents and doc comments: `docs/integrity.md` and `docs/hint_generation.md` state that the database file must not change during a `plinko_hints` run; `docs/online_protocol.md` and `plinko/src/protocol/state.rs` have the rule to keep the deltas that a state needs and to drop a returned query whose hint is in the log; `ProtocolError::NoBackupHintsLeft` separates reserved backup hints from an empty pool; `HintSet::new` states that it does not detect an allocation that the system refuses; `IprfTee` no longer claims that the full iPRF is constant-time, and `IprfTee::inverse_ct` states that it cuts a bin above `MAX_PREIMAGES` with no sign; `tee-test/SEV-SNP-BENCHMARK.md` no longer attributes its overhead number to AES-NI (the PRF mode of those runs is not recorded).
  - `plinko/formal/sims/SwapOrNotSim.v` and `SimTypes.v`: two comments said that the theorems lift properties to the Rust implementation and that the models are translated Rust. They now say that the theorems are statements about hand-written Rocq models, as the status text of each file does. No theorem and no proof changes.
- **Inconsistent axiom in the formal model**: `sr_base_case_no_recurse` (`SwapOrNotSrSpec.v`) contradicted the proven bijection of Sometimes-Recurse, so `False` was derivable. The axiom is removed; `sr_base_case_equiv` now has the condition as a hypothesis.
- **Decoy index was sequential**: a query for a cached index went to the smallest index that was not cached. It now goes to a uniform index among those that are not cached and not in flight (Fig. 7).
- **A lost answer lost the hint, and an update during a query gave a wrong answer**: `prepare_query` took the hint out of the client and kept a copy of its parity.
- **Two queries for one index before the first answer** promoted two backup hints with the same extra index; later updates of that entry missed one of them.
- **`Server::answer` panicked on a malformed query** (short mask, or a mask with more than `c/2` blocks in one subset). It returns `MaskLengthMismatch` or `QueryShapeMismatch`.
- **PMNS binomial sampler was not binomial for counts 1025 to about 12000**: `binomial_sample` delegated to `puruspe::betai`, which selects the wrong continued fraction in the lower tail. PMNS bin loads were not multinomial, and `Iprf` disagreed with `IprfTee` above 1024 hints. The floating-point sampler is removed (see Changed).
- **Constant-time mode above 65536 total hints**: `IprfTee::new` no longer panics; the new sampler has no domain limit.
- **Databases smaller than two blocks**: `plinko_hints` pads them to two blocks, as the protocol client does, instead of returning an error.
- **Schema alignment**: corrected `ENTRY_SIZE` and alignment for 40-byte schema in CUDA kernel
- **H200 misalignment crash**: reverted `ulong2` optimization that caused XID 13 crashes on H200 due to unaligned access
- **48B expansion correctness**: ensured algorithmic correctness with chunked compaction for large datasets
- The Sometimes-Recurse stage of size 2 swapped its two values with probability 1/4, not 1/2. It has the fixed round key 1 now (`test_sr_stage_of_size_2_is_uniform`). Only a PRP with a domain of exactly 2 was biased in a measurable way.

### Removed

- `plinko/formal/specs/TrueBinomialSpec.v` and its axiom `binomial_theorem_Z`: a model of the inverse-CDF sampler that the Rust code no longer has; no proof used it.
- `.beads` directory (unused tracking artifact)
- `analyze_storage_values.rs` (unused)
- Legacy binaries (`bench_hints_linux_amd64`, `state-syncer-linux-amd64`)

## [0.1.0] - 2025-12-18

### Added

- **Constant-time HintInit for TEE execution** (`--constant-time` flag): Issue #62
  - Eliminates timing side-channels that could leak iPRF mappings during hint generation
  - Uses `IprfTee::inverse_ct` with fixed MAX_PREIMAGES=512 iterations (no early exit)
  - `BlockBitset` struct for O(1) constant-time block membership testing
  - Branchless index clamping via `ct_select_usize` to prevent secret-dependent branches
  - `ct_xor_32_masked` for conditional parity XOR without control flow
  - Parameter guards: validates MAX_PREIMAGES sufficiency, num_regular/num_backup > 0
  - Security model: protects against timing side-channels; cache side-channels out of scope
    (would require ORAM for full memory-access obliviousness)
  - Verified against Plinko.v Coq spec and paper Fig. 7 HintInit algorithm
- **TEE constant-time exact binomial sampler** (`binomial_sample_tee`): Issue #41 fix
  - True Binomial(n,p) distribution using inverse-CDF with fixed iterations
  - O(CT_BINOMIAL_MAX_COUNT) complexity with constant-time float comparison and selection
  - No branches depend on `count` or `prf_output` (both treated as secret)
  - Uses log-gamma mode-based computation for numerical stability (supports n up to ~40k)
  - Uses `ct_f64_le`, `ct_select_f64` for constant-time IEEE 754 float operations
  - Threshold: `count <= CT_BINOMIAL_MAX_COUNT` (65536); no fallback - approximation removed entirely
  - `IprfTee` now uses `binomial_sample_tee` for TEE-safe execution
  - `IprfTee::new` asserts `n <= CT_BINOMIAL_MAX_COUNT` to enforce protocol invariant
  - `IprfTee::inverse_ct` uses constant-time min for MAX_PREIMAGES clamping
  - Non-TEE `Iprf` continues to use standard binomial sampler
- **Sometimes-Recurse PRP** (`SwapOrNotSr`, `SwapOrNotSrTee`): Morris-Rogaway Fig. 1 wrapper for full-domain security
  - Level-aware key derivation with proper domain separation
  - Paper-faithful round counts: `t_N = ceil(7.23 lg N + 4.82 lambda + 4.82 lg p)` per Morris-Rogaway Eq. (2)
  - Configurable security parameter (default 128-bit)
  - `with_security(key, domain, security_bits)` constructor for explicit security level
- **Constant-time utilities**: Added `ct_ge_u64` for branchless greater-or-equal comparison
- **Formal verification**: `SwapOrNotSrSpec.v` with Rocq proofs for SR round involution, range preservation, and bijection properties
- **Formal verification**: Completed proofs in `SwapOrNotSrSpec.v` - all lemmas now proved (no Admitted)
  - `sr_forward_inverse_id`, `sr_inverse_forward_id`: SR is a bijection (converted from axioms)
  - `sr_apply_rounds_inv_in_range`: inverse rounds preserve range
- **Formal verification**: Completed proofs in `TrueBinomialSpec.v`
  - `binom_quantile_le_n`: quantile bounded by n (converted from axiom, with proper preconditions)
  - `binom_cdf_as_Zsum`: CDF equals sum of PMF terms (fully proven)
  - Added helper lemmas: `Qle_bool_iff`, `binom_cdf_Q_eq`, `u64_to_Q_le_1`
- **Formal verification**: Linked PRP to SwapOrNot SR in `IprfSpec.v`
  - `prp_forward`/`prp_inverse` now defined as `sr_forward`/`sr_inverse`
  - 4 PRP axioms converted to proven lemmas
- **Formal verification**: Eliminated duplicate axioms
  - Removed 4 duplicate PRP axioms from `IprfProofs.v`
  - Removed duplicate `round_key_in_range` from `SwapOrNotProofs.v`
  - `sr_apply_rounds_inv_inverse`: inverse then forward is identity
  - `sr_forward_aux_in_range`, `sr_forward_range`: forward preserves range
  - `sr_inverse_aux_in_range`, `sr_inverse_range`: inverse preserves range
- **Formal verification**: Eliminated `binomial_sample_range_aux` axiom in `IprfSpec.v`
  - Added `binomial_sample_range_full` lemma to `BinomialSpec.v` (handles count=0 and count>0)
  - Fixed `binomial_sample_spec` to return 0 when count=0 (cleaner semantics)
  - `BinomialSpec.v` and `IprfSpec.v` now have zero axioms
- **Formal verification**: `TrueBinomialSpec.v` inverse-CDF binomial sampler specification
  - `binom_quantile_aux_bound`: quantile function is bounded
  - CDF monotonicity and completeness lemmas
  - `binom_quantile_le_n`: proven with proper preconditions (0 <= num < denom, u <= 1)
- **Documentation**: Alex Hoover talk transcript explaining full-domain PRP requirement for Plinko hint reuse security
- Seed-based subset derivation functions (`derive_subset_seed`, `compute_regular_blocks`, `compute_backup_blocks`)
- New unit tests for subset seed derivation and block computation
- CI improvements:
  - `OPAMJOBS=2` for parallel Rocq compilation
  - Better cache keys with Makefile hash and versioning (v2)
  - Conditional opam setup (skip if cached)
  - `ci-quick` / `ci-full` targets for PR vs main builds
  - Audit artifacts upload
  - GitHub step summary with verification table
  - `admit-count` and `verify-axioms` Makefile targets
  - Configurable `COQC`/`COQDEP` (avoids symlink brittleness)

### Changed

- **iPRF cipher**: Upgraded from AES-128 to ChaCha12 for GPU-native ARX operations; applies to both `GpuHintGenerator` and `CpuHintGenerator`
- **VRAM layout**: On-the-fly compaction strips Tag/Padding during upload, then chunked 48B expansion ensures Tag is included in parity computation
- **Modal scripts**: Updated for v3 schema with robust builder pattern and multi-GPU sharding support
- **iPRF**: `Iprf` and `IprfTee` now use `SwapOrNotSr`/`SwapOrNotSrTee` instead of plain `SwapOrNot` for full-domain security
- **Hint storage**: Hints now store a 32-byte seed instead of explicit block lists, significantly reducing memory footprint
- **Block count (c)**: Auto-bumped to even if odd (required for security proof); hard assertion enforced in production mode
- **Padding**: Replace `find_nearest_divisor` with padding - DB is padded with dummy zero entries instead of truncating tail entries
- **CLI**: `--allow-truncation` is now hidden (debug-only flag that violates security assumptions)
- **CLI docs**: Fixed description for `--entries-per-block` to reflect padding behavior

### Fixed

- **Schema v3 alignment**: Corrected ENTRY_SIZE and alignment for 40-byte schema in CUDA kernel
- **VRAM correctness**: Switched from 32B compaction to 48B expansion to ensure Tag is included in hint parity (algorithmic correctness fix)
- **H200 stability**: Reverted ulong2 vectorized loads that caused XID 13 misalignment crashes on H200
- **SR round counts**: Implemented paper-faithful per-stage epsilon-budget schedule (Morris-Rogaway Section 5, Strategy 1) for provable 128-bit security; previous heuristic had no proven bound
- Block count evenness now properly enforced for Plinko security proof compliance

### Removed

- `.beads` directory (unused build artifact)
- `analyze_storage_values.rs` (unused analysis tool)
- Legacy documentation binaries; archived into `docs/archived/`
- SonarQube scanner work artifacts (added to `.gitignore`)
