# Multi-level PIR prototype (`plinko::multilevel`)

Experimental implementation of multi-level client-preprocessing single-server PIR based on [ePrint 2026/1273](https://eprint.iacr.org/2026/1273) (Section 4), extended with dynamic `UpdateHint` delta updates, versioned checkpoint serialization, and zero-import WebAssembly support.

## Scope and status

- `plinko::multilevel` lives alongside the canonical 1-level Fig. 7 implementation in `plinko::protocol` and does not modify `plinko::protocol`.
- Interactive walkthrough (English and Russian): [igor53627.github.io/plinko-rs](https://igor53627.github.io/plinko-rs/) (`docs/protocol-visualization.html`, Part 6).
- **Database download is unchanged:** as in 1-level Plinko, the client runs `HintInit` locally over the full database (`docs/trust_model.md`). The full Ethereum state is not a target (`docs/protocol_overview.md#scope`): at Ethereum v3 scale (1,834,095,877 x 40-byte records) the client still downloads 73.4 GB, even though 2-level Flat B reduces extrapolated `HintInit` CPU from ~300 core-hours to ~0.64 core-hours.

## How it works

1-level Plinko uses `w ~ sqrt(N)` and `2 * lambda * w` hints so that a single query misses with probability at most `exp(-lambda / 2)`. Multi-level PIR trades a larger fallback query message (paid rarely) for drastically fewer hints and faster `HintInit`:

```text
Level 0 (fast path)   : block_size w_0 ~ sqrt(N)/2, M_0 = ceil(2 * w_0 * ln(kappa))
                        -> misses with probability ~1 / kappa
                           |
                           v (only on Level 0 miss)
Level 1 (fallback)    : ratio = next_power_of_two(kappa), block_size w_1 = w_0 / ratio (>= 2),
                        M_1 = 2 * w_1 * kappa
                        -> combined miss probability ~exp(-kappa)
```

- **Parameter builders** (`plinko/src/multilevel.rs`):
  - `MultiLevelParams::two_level(num_entries, record_size, kappa, query_budget, variant)` — 2-level Table 2 geometry; rounds `kappa` up to `ratio = next_power_of_two(kappa)`, sets `w_1 = w_0 / ratio` (rejecting inputs where `ratio > w_0 / 2`, i.e. `w_1 < 2`), and sets `M_1 = 2 * w_1 * kappa` (for example, `N = 65,536`, `kappa = 33` yields `w_0 = 128`, `ratio = 64`, `w_1 = 2`, `M_1 = 132`).
  - `MultiLevelParams::waterfall(num_entries, record_size, num_levels, query_budget, variant)` — `d`-level geometric waterfall (`w_i` dividing by four across levels, `w_i = w_0 / 4^i`).
  - `MultiLevelParams::custom(num_entries, record_size, levels)` — validates level geometry and checks primary/backup/accumulator storage sizes fit within `isize::MAX` before allocation.
- **Per-level backends (`BackendVariant`):**
  - `HybridA { iprf_levels }`: uses `Iprf` on the first `iprf_levels` levels and batched `FlatAes` on the remaining levels.
  - `FlatB` (Variant B): uses 8-way batched AES-128 (`FlatAes`) on all levels. Because total hints drop by ~11x-30x, scanning the hint table per block with hardware AES is substantially faster than inverting Swap-or-Not/PMNS iPRFs during `HintInit`, `prepare_query`, and `apply_deltas_batch`.
- **Failure-private `Refresh` (Lemma 9):**
  - When a query hits level `l`, all skipped levels `0..l` suffered a miss. To keep their hint distributions independent of past queries, `finish_query` collects the `A` active, non-reserved primary slots on each failed level `i < l` (excluding reserved and retired slots), samples `k ~ Binomial(A, 1 / (2 * w_i))` over those `A` slots so each available slot retains exact probability `1 / (2 * w_i)`, and replaces the `k` chosen slots with backups via `PromoteIn(x, value)` (replacing 0 slots when `A = 0`).
  - On the winning level `l`, the spent slot is always replaced via `PromoteIn(x, value)`. Then, with probability `(1 - 1 / (2 * w_l))^A_l` where `A_l` counts active slots on level `l` (non-retired slots not reserved by other in-flight queries, including the spent slot), `finish_query` sweeps the matching active, non-reserved primary slots returned by `find_matching_slots` plus the just-replaced slot with `PromoteOut(x)`. All backup budgets across failed levels and the winning level are verified before any refresh replacement is committed; if a backup budget check fails (`BackupExhausted`), `finish_query` removes the pending query, marks the winning slot `retired`, and returns the recovered record without applying refresh replacements.
  - Tail probabilities below `2^-53` use 32-bit chunked Bernoulli sampling (`sample_bernoulli_neg_exp`).
- **Dynamic `UpdateHint` (`apply_delta`, `apply_deltas_batch`):**
  - Each record change `(index, delta)` XORs `delta` into matching primary slots (respecting `override_entry` on promoted slots via an `override_index` map) and unconsumed backups on every level.
  - `apply_deltas_batch` evaluates per-block inverses across worker threads using fallible `std::thread::Builder::spawn_scoped` with inline fallback.
- **Single-pass streaming `HintInit` (`MultiLevelHintBuilder`, `hint_init_from_reader`, `hint_init_from_file`):**
  - Because every level's `block_size` is a power of two, their least common multiple equals `max(block_size)`. Streaming `HintInit` reads the database once in bounded chunks (default `DEFAULT_STREAM_CHUNK_BYTES`: 64 MiB native / 4 MiB `wasm32`, aligned to a multiple of `max(block_size) * record_size` and capped to the database size) and updates all levels simultaneously into persistent per-level accumulators (with total multi-worker local buffer memory bounded across all levels), producing bit-identical hints to in-memory `MultiLevelClient::hint_init`.
  - `MultiLevelHintBuilder` accepts arbitrary unaligned `push_bytes` slices (works on `wasm32-unknown-unknown` with zero host imports).
  - `hint_init_from_reader[_with_chunk_bytes]` streams from any EOF-terminated `std::io::Read` (handling short reads / `EINTR` and probing one byte past the expected length to reject trailing data; because that probe waits for EOF, use a bounded reader such as `reader.take(expected)` -- which cannot detect trailing bytes outside its bound -- or `MultiLevelHintBuilder::push_bytes` on a persistent connection).
  - `hint_init_from_file[_with_chunk_bytes]` (native) pre-checks regular file size before allocating accumulators; the file must remain unchanged for the entire call and match the snapshot the server answers against.

## Safety invariants and `PLNKMLH1` checkpoint format

- **In-flight slot reservation and recovery:** `prepare_query` reserves the selected primary slot so concurrent calls never emit two queries from the same hint. `finish_query(&pending, &answer, &mut rng)` and `abort_query(&pending)` borrow `PendingMultiLevelQuery` so callers retain the handle to retry or abort after `AnswerLengthMismatch` or server `SnapshotMismatch`. Calling `abort_query` releases the in-flight reservation while permanently marking the slot `retired` (persisted in slot flags bit 3) so an exposed hint is never reused.
- **Session and snapshot-epoch binding:** each `MultiLevelClient` mints a unique `session_id` (renewed on reload) so pending handles cannot be replayed across client instances. `MultiLevelClient::epoch()` increments on each successful `apply_delta` and each non-empty successful `apply_deltas_batch` (empty batches do not advance the epoch) and is stored in the `PLNKMLH1` header; `MultiLevelServer::answer(&db, &params, &query, snapshot_epoch)` stamps the server's epoch onto `MultiLevelAnswer`, and `finish_query` rejects both client-side and server-side epoch mismatches before decoding or refreshing.
- **Serialization (`write_hints`, `read_hints`, `read_hints_with_limit`, `to_bytes`, `from_bytes`, `from_bytes_with_limit`):**
  - Binary format `PLNKMLH1` (version 1): 36-byte header (magic, version, `num_entries`, `record_size`, `num_levels`, `epoch`), per-level metadata, slot flags (`selector`, `has_override`, `override_color`, `retired`), active overrides, primary parities, remaining unconsumed backup parities only (`next_backup..num_backup`), and a trailing 32-byte BLAKE3 checksum.
  - `write_hints` refuses to checkpoint while any query is in flight (`QueryInFlight`).
  - `read_hints` enforces a default 512 MiB decoded memory budget (`MAX_CHECKPOINT_BYTES`, customizable via `read_hints_with_limit` / `from_bytes_with_limit`) with fallible allocations and compact remaining-backup storage (`backup_offset`).

## Measured benchmarks

Measured on Apple M3 Ultra (32 cores, `RUSTFLAGS="--cfg aes_armv8"`, record size `r = 40 B`) via `bench_multilevel`:

### Small and medium scales (full end-to-end runs)

| Scale | Scheme | Hint memory | HintInit (32T) | UpdateHint (1 rec, 1T) |
|-------|--------|-------------|----------------|------------------------|
| `2^20` (1,048,576 recs) | 1-level Fig. 7 Plinko | 5.58 MiB | 10.86 s | 6.9 us |
| `2^20` (1,048,576 recs) | 2-level Flat B (`kappa=32`) | 490.8 KiB (11.6x smaller) | 74.1 ms (146x faster) | 324.4 us |
| `2^24` (16,777,216 recs) | 2-level Flat B (`kappa=64`) | 1.86 MiB | 1.09 s | -- |
| `2^24` (16,777,216 recs) | 6-level Flat B | 1.81 MiB | 3.44 s | -- |

### Ethereum mainnet v3 scale reference (`1,834,095,877` x 40 B, `Q = 65,536`)

Extrapolated from real per-block `bench_level_blocks` timings (separating one-time accumulator setup from per-block scan) plus directly measured `apply_deltas_batch` (3,000-record block) and `prepare_query`:

| Scheme | Hint memory | HintInit CPU (extrapolated) | HintInit wall (32T) | 3,000-rec UpdateHint (32T) | Query prep (1T) |
|--------|-------------|-----------------------------|---------------------|----------------------------|-----------------|
| 1-level Fig. 7 Plinko | ~1.0 GiB+ | ~300 core-h | -- | ~1.6 s (1T equiv.) | ~53 ms (`2^24`) |
| 2-level Flat B (`kappa=64`) | 29.44 MiB | 0.64 core-h | ~1.9 min | 41.4 ms | 3.37 ms |

Remember: this scale reference measures compute and hint memory only; a client still downloads the full 73.4 GB database.

## WebAssembly (`wasm32-unknown-unknown`)

`plinko::multilevel` compiles cleanly with `--no-default-features --target wasm32-unknown-unknown` (clamping worker counts to 1 without spawning threads and importing zero host functions).

The standalone smoke crate [`scripts/wasm_multilevel_smoke`](../scripts/wasm_multilevel_smoke) compiles a ~178 KiB release `.wasm` module with **0 host imports** and runs full 2-level Variant B `HintInit`, `PLNKMLH1` round-trip serialization + tamper rejection, queries, refresh, mid-session reload, and `apply_delta` inside Node.js (~175 ms). It runs in the `WebAssembly` job of GitHub Actions CI.

## Commands

```bash
# Run multilevel unit tests (release + hardware AES on Apple Silicon)
RUSTFLAGS="--cfg aes_armv8" cargo test --release -p plinko multilevel

# Run multilevel tests without CLI/default features
cargo test --manifest-path plinko/Cargo.toml --no-default-features --lib multilevel

# Run the multi-level benchmark (2^20, 2^24, and Ethereum scale reference)
RUSTFLAGS="--cfg aes_armv8" cargo run --release -p plinko --bin bench_multilevel

# Run the zero-import WASM E2E smoke test in Node.js
./scripts/test_wasm_multilevel.sh
```
