# Plinko RS - Agent Rules

## Code Environment

- **Codex Code Env ID**: `694e8588110c8191945f9b9dfbf0b7d1`

## Codex Cloud Tasks

For complex or long-running tasks (formal proofs, large refactors), use Codex Cloud:

```bash
# Submit a task to Codex Cloud (runs remotely with gpt-5.2-codex + xhigh reasoning)
codex cloud exec --env 694e8588110c8191945f9b9dfbf0b7d1 --attempts 3 "Your task description"

# Check task status
codex cloud status <task_id>

# View diff from completed task
codex cloud diff <task_id>

# Apply changes locally
codex cloud apply <task_id>
```

Options:
- `--attempts N`: Run N parallel attempts (best-of-N), use 3 for complex proofs
- `--branch <branch>`: Target a specific git branch

## GitHub CLI hygiene

- When using `gh` to create or edit issues/PRs, ensure newlines are real (not literal `\n`); prefer `--body-file` or a here-doc.
- If a posted issue/PR body contains literal `\n`, fix it immediately with `gh issue edit`/`gh pr edit`.

## Codex Cloud + Remote Rocq Verification

For Coq/Rocq proof synthesis with verification, use the remote build server.
See [docs/codex-cloud-verification.md](docs/codex-cloud-verification.md) for full details.

### Quick Start

```bash
# Submit proof task with verification
codex cloud exec --env 694e8588110c8191945f9b9dfbf0b7d1 --attempts 3 "
Prove \`lemma_name\` in plinko/formal/specs/SomeSpec.v

Verify using:
curl -s -X POST http://108.61.166.134/verify-project \\
  -H 'Content-Type: application/json' \\
  -d '{\"files\": {...}, \"main\": \"Plinko/Specs/SomeSpec.v\", \"timeout\": 120}'

Read .v files from plinko/formal/specs/, key as Plinko/Specs/<name>.v.
"
```

### Verification Server

- **Host**: `108.61.166.134:80`
- **Endpoints**: `GET /health`, `POST /verify`, `POST /verify-project`
- **Service**: `rocq-verify.service` (Python API + Rocq 9.1.0)

### Check Server Status

```bash
ssh root@108.61.166.134 "systemctl status rocq-verify nginx"
ssh root@108.61.166.134 "tail -f /var/log/nginx/access.log"
```

### Evaluating Multiple Attempts

When using `--attempts N`, always check ALL attempts before applying:

```bash
# View each attempt's diff
for i in 1 2 3; do echo "=== Attempt $i ==="; codex cloud diff <task_id> --attempt $i | head -20; done

# Apply specific attempt
codex cloud apply <task_id> --attempt <N>
```

**Evaluation criteria (in priority order):**

1. **Compiles** - Must verify with remote Rocq server (mandatory, disqualifies if fails)
2. **No new axioms/Admitted** - Solution shouldn't introduce new proof debt
3. **Minimal changes** - Fewer lines = less risk of breaking other proofs
4. **Semantic correctness** - Doesn't change function definitions unless necessary
5. **Proof quality**:
   - Direct proofs > convoluted ones
   - Reuses existing lemmas > duplicates logic
   - Clear structure (induction base/step separated)
6. **Helper lemmas** - General/reusable > one-off hacks

**Quick comparison:**
```bash
# Line count per attempt
for i in 1 2 3; do echo "Attempt $i: $(codex cloud diff <task_id> --attempt $i 2>&1 | grep -c '^[+-]') lines"; done

# Check for new Admitted/Axiom
for i in 1 2 3; do echo "Attempt $i:"; codex cloud diff <task_id> --attempt $i 2>&1 | grep -E "^\+.*Admitted|^\+.*Axiom"; done
```

## Canonical Protocol Reference

**`docs/plinko_paper_index.json`** is the canonical source of truth for Plinko protocol implementation details. It indexes:
- Parsed paper content (`plinko_paper_part*.json`)
- Regression test data location (Cloudflare R2)
- Video resources
- Agent usage notes for different tasks

Always consult this file first when implementing or verifying protocol logic.

## Host: root@aya

- **Reth DB location**: `/mnt/mainnet/data`
- **Working directory**: `/mnt/mainnet/plinko`
- **DO NOT** create artifacts outside `/mnt/mainnet/plinko` without explicit user permission

## Build Commands

```bash
# Build plinko binary
cargo build --release

# Build hints generator
cd plinko && cargo build --release --bin plinko_hints
```

## Run Commands

```bash
# Extract from Reth DB (on aya)
./target/release/plinko \
  --db-path /mnt/mainnet/data \
  --output-dir /mnt/mainnet/plinko

# Generate hints (standard mode)
./plinko/target/release/plinko_hints \
  --db-path /mnt/mainnet/plinko/database.bin \
  --lambda 128

# Generate hints (constant-time mode for TEE)
./plinko/target/release/plinko_hints \
  --db-path /mnt/mainnet/plinko/database.bin \
  --lambda 128 --constant-time
```

### Run With journald Logging (aya)

Use `systemd-run` so stdout/stderr land in the system journal with a named unit:

```bash
systemd-run --unit plinko_hints_0p01_prodW --working-directory=/mnt/mainnet/plinko \
  bash -lc 'RUST_BACKTRACE=1 ./target/release/plinko_hints \
    --db-path /mnt/mainnet/plinko/tmp/sample_0p01pct.db \
    --entries-per-block 49177 \
    --lambda 127 \
    --seed 000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f'

# View logs
journalctl -u plinko_hints_0p01_prodW -n 200
```

## Data Format

- Accounts: 3 words (96 bytes) - nonce, balance, bytecode_hash
- Storage: 1 word (32 bytes) - value

## Style

- Do NOT use emojis in code, comments, or documentation

## Commit Messages

This repo enforces [Conventional Commits](https://www.conventionalcommits.org/).

- **Format**: `<type>[(<scope>)][!]: <description>` (max 72 chars)
- **Types**: `feat`, `fix`, `perf`, `docs`, `chore`, `ci`, `test`, `refactor`, `style`, `build`, `revert`
- **Normalizer**: `.githooks/prepare-commit-msg` auto-fixes safe issues before validation (lowercase type, expand aliases like `feature`->`feat`, insert missing space after colon)
- **Validator**: `.githooks/commit-msg` validates locally; run `scripts/setup-hooks.sh` to enable
- **CI**: `.github/workflows/commit-lint.yml` validates all PR commits
- **Exemptions**: merge commits, `fixup!`/`squash!` prefixes, `bd sync` messages

## PR Formatting

- Use real newlines in PR descriptions/comments; never include literal "\n"
- Use Markdown headings ("## Summary", "## Testing") with dash bullet lists

## Formal Verification Compliance

When modifying Rust code in `plinko/`, ALWAYS verify changes don't break alignment with:

1. **Paper specification** (`docs/plinko_paper_*.json`)
   - HintInit must match Fig. 7 pseudocode
   - iPRF domain/range must be correct (domain=num_hints, range=w)
   - Key generation: one iPRF key per block

2. **Coq specification** (`docs/Plinko.v`)
   - `RegularHint`: block subset of c/2+1, single parity
   - `BackupHint`: block subset of c/2, dual parities (in/out)
   - `hint_init`, `process_db_entry` logic must match

3. **Rocq proofs** (`plinko/formal/`)
   - `specs/*.v`: iPRF, SwapOrNot, binomial specs
   - `proofs/*.v`: Verified properties
   - Do NOT change Rust semantics that would invalidate proofs

### Verification Checklist

Before committing Rust changes to hint generation or iPRF:

- [ ] iPRF instantiation: `Iprf::new(key, domain, range)` where domain=total_hints, range=w
- [ ] Per-block keys: c keys derived from master seed
- [ ] Regular hints: subset size c/2+1, single parity XOR'd when block in P
- [ ] Backup hints: subset size c/2, parity_in when block in P, parity_out otherwise
- [ ] Streaming: (block, offset) = (i/w, i mod w), use block's key for iPRF.inverse(offset)

### Reference Files

| Spec | Location |
|------|----------|
| Paper HintInit | `docs/plinko_paper_part6_algorithms.json` (algorithm_id: plinko_hintinit) |
| Coq HintInit | `docs/Plinko.v` (hint_init, process_db_entry) |
| iPRF Spec | `plinko/formal/specs/IprfSpec.v` |
| PRP Spec | `plinko/formal/specs/SwapOrNotSpec.v` |

## Documentation Sync Rules

When running GPU benchmarks:
- **ALWAYS** check `docs/gpu_benchmark_commands.md` for the latest command templates and data locations.
- **UPDATE** `docs/gpu_benchmark_commands.md` with new results after significant benchmark runs.

When modifying `plinko/src/bin/hint_gen/`:
- Update `docs/hint_generation.md` if module structure or API changes
- Update `docs/constant_time_mode.md` if CT security model changes
- Keep the module table in `docs/hint_generation.md` current with file sizes

## Devin DeepWiki

Update `.devin/wiki.json` when making changes worth documenting:
- Add repo_notes for new features, data locations, or architectural changes
- Update pages array if new components need dedicated documentation
- DeepWiki regenerates on next Devin indexing
