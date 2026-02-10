# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added

- **GPU-accelerated hint generation**: CUDA kernel for massively parallel hint computation
  - CUDA kernel (`hint_kernel.cu`) with ChaCha12 block cipher and SwapOrNot iPRF
  - `GpuHintGenerator` using cudarc for GPU execution
  - Warp-level parallelism: warp shuffle broadcasts block keys (32x fewer memory reads), ballot sync skips unneeded blocks
  - Parallel CPU baseline (`CpuHintGenerator`) with full ChaCha8 iPRF and rayon parallelism for accurate GPU speedup comparison
  - Evolved cipher from AES-128 (software S-box bottleneck) to ChaCha8 (33x speedup) to ChaCha12 (enhanced security)
  - `gen_synthetic` binary for generating test datasets
  - `bench_gpu_hints` binary for GPU vs CPU benchmarking
  - Modal scripts for H100/H200 distributed benchmarking with multi-GPU sharding
  - Production benchmarks: ~133-175k hints/sec on 50x H200 GPUs
- **Schema v3 (40-byte entries)**: 17% storage reduction over 48-byte schema
  - `AccountEntry40`: Balance(16) + Nonce(4) + CodeID(4) + TAG(8) + Pad(8)
  - `StorageEntry40`: Value(32) + TAG(8) - no padding needed
  - Reduced nonce from 8B (u64) to 4B (u32) - sufficient for 4B transactions
  - Mainnet footprint: 103 GB -> 86 GB; 12% faster GPU hint generation
  - Chunked VRAM upload: 48B expansion during upload to include Tag in parity, avoids VRAM overflow
  - Reverted ulong2 vectorized loads due to XID 13 misalignment crashes on H200
- **Documentation restructuring**
  - Split README into focused subdocs (ops, benchmarks, architecture)
  - Added ops docs and PIR dataset links for mainnet v3 data
  - GPU benchmark notes and expansion strategy documentation
- **Repository rename and crate move**: consolidated `state-syncer` into the `plinko` crate
  - Workspace now uses `plinko` for hint generation and iPRF/CT primitives
  - Docker artifacts updated to build/run `plinko` instead of `state-syncer`
  - Documentation references updated to the new crate paths

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

- **iPRF cipher**: Upgraded from AES-128 to ChaCha12 for GPU-native ARX operations and improved security margin
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
