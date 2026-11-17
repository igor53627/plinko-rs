# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added

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
  - Expansion strategy documentation (`EXPANSION_40B_48B.md`)
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
  - Ops docs added (`ARCHITECTURE.md`, `DEPLOYMENT.md`, `API_ENDPOINTS.md`, `FEATURE_FLAGS.md`)
  - GPU benchmark results and optimization kanban tracking
  - Legacy docs archived, stale binaries removed

- **Hint file**: `plinko_hints --output <file>` writes all hints to one versioned file (`PLNKHINT`, version 1: geometry, hint counts, seed check, cutoffs and parities, BLAKE3 checksum; see `docs/data_format.md`). The new library module `plinko::hints` (`HintSet`, `RegularHint`, `BackupHint`) is shared by the generator and the client.
- **Client loads a hint file**: `Client::from_hints` builds a client from a `HintSet` and rejects another geometry, other hint counts, or another master seed (`ProtocolError::HintSetMismatch`). `ProtocolParams::for_hint_set` derives the parameters from the file. `protocol::compute_hints` runs HintInit in memory; `Client::offline_init` now calls it.
- `plinko_hints --seed-file <file>` reads the master seed from a file, so that the seed is not in the process list. `--output` requires `--seed` or `--seed-file`.
- End-to-end test `plinko/tests/hint_file_e2e_test.rs`: runs the `plinko_hints` binary, compares its file with the in-memory hint set, and checks queries and an update; an ignored test does the same with 66,560 hints.
- `scripts/local_e2e_hint_pipeline.sh` compares the hint files of the fast and the constant-time run byte for byte.

- **PMNS tree reuse**: `Iprf::table()` and `IprfTee::table_ct()` compute all bin boundaries of one iPRF once; HintInit builds one table per database block in both paths and in `protocol::compute_hints`. `Iprf::with_cached_levels` keeps the top tree levels after the first call; the protocol client caches 8 levels per block (`CLIENT_CACHED_PMNS_LEVELS`). Outputs do not change. Measured at w = 65536 with hardware AES: `forward` 445 us to 28.5 us, `inverse` 7.56 ms to 7.02 ms (the inverse PRP dominates).
- The constant-time path stops with a panic when a PMNS bin holds more than `MAX_PREIMAGES` balls. It truncated the bin silently before and gave a wrong parity.

### Changed

- **Exact integer PMNS sampler (breaking)**: `binomial_sample` and `binomial_sample_tee` are replaced by `coin_binomial` and `coin_binomial_ct`. A split is the number of set bits among the first `count` bits of an AES-128 counter keystream of the tree node, which is exactly Binomial(count, 1/2) for a uniform keystream (computationally indistinguishable from it under the AES PRF assumption) and identical on every platform and in both modes. All iPRF outputs change.
- **iPRF range must be a power of two (breaking)**: `Iprf::new` and `IprfTee::new` panic otherwise, `ProtocolParams::new` returns an error, and `plinko_hints --entries-per-block` rejects other values. The default `w` of `plinko_hints` is now the smallest power of two >= sqrt(N) (mainnet v3: 65536 instead of 42826), the same as the protocol client.
- **PMNS key separation**: the keystream key is `SHA-256(key || "pmns" || n || m)[0..16]`; the node id is no longer a truncated hash.
- Removed the `puruspe` dependency.
- **Block subsets from a PRF cutoff (breaking)**: a hint stores its block subset as one 32-bit cutoff (RMS24 section 3.2, with the subset sizes of Plinko Fig. 7). Block `k` is in the subset of hint `j` when `PRF("select", j, k) < cutoff`. The new library module `plinko::subsets` (`SubsetPrf`, `find_cutoff`) is shared by `plinko_hints` and the protocol client. Subsets differ from the earlier `ChaCha20Rng` + `index::sample` derivation for the same master seed.
- A hint with a tie at the cutoff boundary is invalid (cutoff 0): its parities stay zero and the client does not use it. `plinko_hints` logs `invalid_regular` and `invalid_backup` in place of `zero_regular_seeds` and `zero_backup_seeds`.
- Removed `BlockBitset`, `compute_regular_blocks`, `compute_backup_blocks`, `block_in_subset`, `derive_subset_seed`, `SEED_LABEL_REGULAR`, `SEED_LABEL_BACKUP`, and the `memory_bench` benchmark. Subset state is 4 bytes per hint in the generator and in the client.
- `cost_estimate` counts 36 bytes per regular hint and 68 bytes per backup hint (cutoff + parities).
- **Agent rules**: `AGENTS.md` has a local-only phase section (no GitHub, `backlog/` tasks, `jj`, CodeRabbit CLI review) and no longer describes the Codex Cloud workflow, the remote Rocq checker, DeepWiki updates, or PR formatting. It states that the formal models are not checked against the Rust source. The module table in `docs/hint_generation.md` has no file-size column.

### Fixed

- **PMNS binomial sampler was not binomial for counts 1025 to about 12000**: `binomial_sample` delegated to `puruspe::betai`, which selects the wrong continued fraction in the lower tail. PMNS bin loads were not multinomial, and `Iprf` disagreed with `IprfTee` above 1024 hints. The floating-point sampler is removed (see Changed).
- **Constant-time mode above 65536 total hints**: `IprfTee::new` no longer panics; the new sampler has no domain limit.
- **Databases smaller than two blocks**: `plinko_hints` pads them to two blocks, as the protocol client does, instead of returning an error.
- **Schema alignment**: corrected `ENTRY_SIZE` and alignment for 40-byte schema in CUDA kernel
- **H200 misalignment crash**: reverted `ulong2` optimization that caused XID 13 crashes on H200 due to unaligned access
- **48B expansion correctness**: ensured algorithmic correctness with chunked compaction for large datasets

### Changed

- **iPRF cipher**: upgraded from ChaCha8 to ChaCha12 for GPU hint generation (security margin)
- **Schema default**: v3 (40-byte) schema used by default in Modal benchmark scripts

### Removed

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
- Legacy documentation binaries; archived into `docs/archive/`
- SonarQube scanner work artifacts (added to `.gitignore`)
