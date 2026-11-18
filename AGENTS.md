# Plinko RS - Agent Rules

## Local-Only Phase (No GitHub)

The repository is not on GitHub at this time; `origin` is a local Git server. Until the owner
publishes it again, this section replaces the GitHub rules of the global agent instructions
(issue tracking, PR workflow, branch cleanup) and the CI references below.

- Do not use `gh`. Do not create issues, pull requests, or review comments.
- Track work in `backlog/` with the `backlog` CLI. A task is necessary for work that spans more
  than one commit and for each deferred item (follow-up, revert, TODO). A small fix needs no task.
- Version control is `jj` in a colocated repository, without feature branches. `jj` does not run
  Git hooks, so check each commit message with `.githooks/commit-msg <file>` (see Commit Messages).
- Review local changes before a commit: `coderabbit review --uncommitted --agent`.
- Push only when the owner asks.

Remove this section when the repository is on GitHub again.

## Scope

The target is a PIR library for a database of fixed-size records. The full Ethereum state is not
a target (owner decision of 2026-11-17, `backlog/decisions/decision-1`; costs in
`docs/protocol_overview.md`). The Reth extractor, the v3 record format and the mainnet numbers
stay as test data and as a scale reference. Do not describe the repository as a solution for
Ethereum-scale state, and do not propose HintInit by a server, a cloud or a TEE.

## Canonical Protocol Reference

Prefer these for implementation and verification (see also `README.md`):

1. **`docs/hint_generation.md`** — HintInit (Fig. 7), crate layout
2. **`docs/data_format.md`** — v3 `database.bin` / mappings, mainnet reference counts
3. **`docs/Plinko.v`** and **`plinko/formal/`** — formal specs and proofs
4. **`docs/plinko_paper_part6_algorithms.json`** — parsed pseudocode when cross-checking the paper

**`docs/plinko_paper_index.json`** is an agent catalog of paper parts 1–6. Dataset layout: **`docs/data_format.md`** (v3). Use local extract `--limit`, synthetic data, or in-repo tests.

If the parsed JSON and the paper PDF (`docs/2024-318.pdf`) disagree, the PDF is correct.

## Host: root@aya

- **Reth DB location**: `/mnt/mainnet/data`
- **Working directory**: `/mnt/mainnet/plinko`
- **DO NOT** create artifacts outside `/mnt/mainnet/plinko` without explicit user permission

## Build Commands

```bash
# Build the extractor (workspace root package)
cargo build --release

# Build the hints generator
cargo build --release -p plinko --bin plinko_hints
```

## Run Commands

```bash
# Extract from Reth DB (on aya)
./target/release/plinko \
  --db-path /mnt/mainnet/data \
  --output-dir /mnt/mainnet/plinko

# Generate hints (standard mode)
./target/release/plinko_hints \
  --db-path /mnt/mainnet/plinko/database.bin \
  --lambda 128

# Generate hints (constant-time mode; experimental, build with --features tee)
./target/release/plinko_hints \
  --db-path /mnt/mainnet/plinko/database.bin \
  --lambda 128 --constant-time
```

`--entries-per-block` (w) must be a power of two. The default is the smallest power of two that
is at least sqrt(N).

### Run With journald Logging (aya)

Use `systemd-run` so stdout/stderr land in the system journal with a named unit:

```bash
systemd-run --unit plinko_hints_0p01_prodW --working-directory=/mnt/mainnet/plinko \
  bash -lc 'RUST_BACKTRACE=1 ./target/release/plinko_hints \
    --db-path /mnt/mainnet/plinko/tmp/sample_0p01pct.db \
    --entries-per-block 65536 \
    --lambda 127 \
    --seed 000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f'

# View logs
journalctl -u plinko_hints_0p01_prodW -n 200
```

## Data Format

- A database is an array of fixed-size records. The record size is a parameter:
  `plinko_hints --record-size` (default 40), `ProtocolParams::record_size`, and the
  `record_size` field of the hint file (version 3, `docs/data_format.md`).
- A hint parity is the XOR of whole records. No code may read a part of a record into a parity.
- v3 `database.bin` (extractor, `gen_synthetic`): 40-byte records, layout in `docs/data_format.md`.

## Style

- Do NOT use emojis in code, comments, or documentation

## Commit Messages

This repo enforces [Conventional Commits](https://www.conventionalcommits.org/).

- **Format**: `<type>[(<scope>)][!]: <description>` (max 72 chars)
- **Types**: `feat`, `fix`, `perf`, `docs`, `chore`, `ci`, `test`, `refactor`, `style`, `build`, `revert`
- **Normalizer**: `.githooks/prepare-commit-msg` auto-fixes safe issues before validation (lowercase type, expand aliases like `feature`->`feat`, insert missing space after colon)
- **Validator**: `.githooks/commit-msg` validates locally; run `scripts/setup-hooks.sh` to enable
- **CI**: `.github/workflows/commit-lint.yml` normalizes then validates all PR commits (same 3 transforms as local hook, so contributors without local hooks still pass)
- **Tests**: `scripts/test_commit_hooks.sh` table-driven tests for normalization, validation, and exemptions
- **Exemptions**: merge commits, `fixup!`/`squash!` prefixes, `bd sync` messages (all three enforcement points use identical `case` syntax)

## Protocol Compliance

The paper is the specification. When modifying Rust code in `plinko/`, make sure that the change
agrees with Fig. 4 (PMNS) and Fig. 7 (HintInit, Query, Answer, Recon, UpdateHint, GetHint).

### Verification Checklist

Before committing Rust changes to hint generation or iPRF:

- [ ] iPRF instantiation: `Iprf::new(key, domain, range)` where domain=total_hints, range=w
- [ ] w is a power of two, and the generator and the client derive the same (w, c)
- [ ] Per-block keys: c keys derived from master seed
- [ ] Regular hints: subset size c/2+1, single parity XOR'd when block in P
- [ ] Backup hints: subset size c/2, parity_in when block in P, parity_out otherwise
- [ ] Streaming: (block, offset) = (i/w, i mod w), use block's key for iPRF.inverse(offset)
- [ ] `Iprf` and `IprfTee` give the same output for the same key (`cargo test --release -p plinko --features tee,gpu`)
- [ ] The default build has no TEE or GPU code: both are opt-in features outside the trust model (`docs/trust_model.md`, `docs/FEATURE_FLAGS.md`)

### Formal Models

`docs/Plinko.v` and `plinko/formal/` are hand-written models of the algorithms. No tool checks
them against the Rust source, and they do not model the AES keystream of the PMNS sampler, key
derivation, or 40-byte records. A Rust test is the evidence for Rust behaviour; a proof is not.

- Build: `cd plinko/formal && make` (Rocq is installed locally).
- Do not add `Axiom` or `Admitted`. Do not describe Rust code as formally verified.
- `plinko/formal/docs/TRUST_BASE.md` is generated. After a change of an `Axiom`, `Parameter` or
  `Admitted`, edit `plinko/formal/docs/trust_base_notes.toml` and run `make trust-base`.
- When a Rust change makes a statement in `docs/verification.md` or
  `plinko/formal/docs/TRUST_BASE.md` false, correct the statement in the same change.
- The Codex Cloud and remote checker workflow is in `docs/codex-cloud-verification.md`. It is
  not necessary for local work.

### Reference Files

| Spec | Location |
|------|----------|
| Paper (PDF) | `docs/2024-318.pdf` |
| Paper HintInit | `docs/plinko_paper_part6_algorithms.json` (algorithm_id: plinko_hintinit) |
| Coq HintInit | `docs/Plinko.v` (hint_init, process_db_entry) |
| iPRF Spec | `plinko/formal/specs/IprfSpec.v` |
| PRP Spec | `plinko/formal/specs/SwapOrNotSrSpec.v` |

## Documentation Sync Rules

When running GPU benchmarks:
- **ALWAYS** check `docs/gpu_benchmark_commands.md` for the latest command templates and data locations.
- **UPDATE** `docs/gpu_benchmark_commands.md` with new results after significant benchmark runs.

When modifying `plinko/src/hint_init.rs`, `plinko/src/hint_init/` or `plinko/src/bin/hint_gen/`:
- Update `docs/hint_generation.md` if module structure or API changes
- Update `docs/constant_time_mode.md` if CT security model changes
