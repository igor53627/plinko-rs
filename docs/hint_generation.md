# Plinko PIR Hint Generator

Implements Plinko's HintInit (Fig. 7 of the paper).

Reads a database file that is an array of fixed-size records. `--record-size` gives the
record size in bytes (1 to 1,048,576; default 40, the v3 `database.bin` record). Geometry
uses the record count (`file_size / record_size`); a file that is not a whole number of
records is an error. A hint parity is the XOR of whole records, so it has the record size.
The GPU benchmark (`plinko/src/gpu.rs`, `gpu` feature) works with 40-byte records only.

## Key Design (per paper)

- Generates c iPRF keys (one per block), not one global key
- Regular hints: block subset of size c/2+1, single parity
- Backup hints: block subset of size c/2, dual parities (in/out)
- A block subset is one 32-bit cutoff per hint (see [Block Subsets](#block-subsets))
- iPRF domain = total hints (λw + q), range = w (block size)
- `w` must be a power of two. The default is the smallest power of two >= sqrt(N), the same rule as the protocol client (`plinko::protocol::default_block_size`). Mainnet v3: w = 65536, c = 27988

## Module Structure

HintInit is the library module `plinko::hint_init`. The binary `plinko_hints`
and `plinko::protocol::compute_hints` (which `Client::offline_init` uses) call
the same functions, so there is one implementation of the cutoffs and of the
streaming loop.

| File                                | Purpose                                                                               |
|-------------------------------------|---------------------------------------------------------------------------------------|
| `plinko/src/hint_init.rs`           | `hint_cutoffs` and `fill_parities` (standard streaming phase), both on worker threads |
| `plinko/src/hint_init/tee.rs`       | `hint_cutoffs_ct` and `fill_parities_ct`, the constant-time phases; `tee` feature     |
| `plinko/src/hint_init/tests.rs`     | Tests of both streaming phases against the hint definition of Fig. 7                  |
| `plinko/src/bin/plinko_hints.rs`    | The binary: reads the options, maps the database, calls `plinko::hint_init`           |
| `plinko/src/bin/hint_gen/types.rs`  | `Args` (with `--record-size` and `--threads`)                                         |
| `plinko/src/bin/hint_gen/driver.rs` | Geometry, validation, seed handling, thread count, result log, hint file output       |

## API Notes

- `hint_init::hint_cutoffs(subset_prf, num_blocks, num_regular, num_backup, threads)`
  computes the cutoffs.
- `hint_init::fill_parities(hints, database, block_iprfs, subset_prf, threads, progress)`
  writes the parities into a `plinko::hints::HintSet`, which holds the geometry and the
  cutoffs and has the record size of the database. `hint_init::fill_parities_ct` is the
  same with `IprfTee`, and `hint_init::hint_cutoffs_ct` gives the same cutoffs with a
  fixed amount of work for each hint (`tee` feature).
- `protocol::compute_hints` uses all available cores;
  `protocol::compute_hints_with_threads` takes the thread count.
- The block keys come from `protocol::derive_block_keys` in the binary and in the library.
- `driver::print_results(duration, file_len, hints, params)` logs the counts.
- Related CPU benchmark/generator path in `plinko/src/gpu.rs` (`gpu` feature) now uses `CpuHintInput` for `generate_hints`.
- The constant-time variants of the PRP and the iPRF are in `plinko/src/iprf/tee.rs` (`tee` feature). See [FEATURE_FLAGS.md](FEATURE_FLAGS.md).

## Usage

```bash
# Build (from repo root)
cargo build --release -p plinko --bin plinko_hints

# Make a master seed (client-secret)
(umask 077 && openssl rand -hex 32 > ./data/master_seed.hex)

# Generate hints (standard mode; 40-byte records by default)
./target/release/plinko_hints \
  --db-path ./data/database.bin \
  --lambda 128 \
  --seed-file ./data/master_seed.hex \
  --output ./data/hints.bin

# A database of 256-byte records
./target/release/plinko_hints \
  --db-path ./data/records.bin \
  --record-size 256 \
  --lambda 128 \
  --seed-file ./data/master_seed.hex \
  --output ./data/hints.bin

# Generate hints (constant-time mode; experimental, build with --features tee)
./target/release/plinko_hints \
  --db-path ./data/database.bin \
  --lambda 128 --constant-time \
  --seed-file ./data/master_seed.hex \
  --output ./data/hints.bin
```

Without `--output` the run only logs counts. `--output` requires `--seed` or
`--seed-file`, because a client cannot use the hints without the master seed.
`--seed` puts the seed in the process list; use `--seed-file`.

See [constant_time_mode.md](constant_time_mode.md) for TEE security details.

## PMNS Tree Reuse

The iPRF of a block is a PRP followed by a PMNS tree with `w` bins. The tree
is the same for all offsets of the block, so the code computes it once:

- `Iprf::table()` and `IprfTee::table_ct()` compute all bin boundaries of one
  block. HintInit (both paths, and `protocol::compute_hints`) builds one table
  per database block and reads it for each offset.
- `Iprf::with_cached_levels(key, n, m, levels)` keeps the left counts of the
  top tree levels after the first call. The protocol client uses 8 levels
  (`CLIENT_CACHED_PMNS_LEVELS`): 1020 bytes per block, about 29 MB for the
  27,988 blocks of mainnet (computed). The cache is derived from the block key
  and is client-secret.

All outputs are equal to those of the plain `Iprf`
(`test_pmns_table_matches_tree_walk`, `test_pmns_cached_levels_match_uncached_tree`,
`test_pmns_tee_table_matches_tree_walk`).

Measured at w = 65536 and 16,777,216 hints on one thread (Apple M3 Ultra,
hardware AES through `--cfg aes_armv8`):

| Operation                               | Before  | After                          |
|-----------------------------------------|---------|--------------------------------|
| `forward` (client, 8 cached levels)     | 445 us  | 28.5 us                        |
| `inverse` (client, 8 cached levels)     | 7.56 ms | 7.02 ms                        |
| `inverse` per entry in HintInit (table) | 7.56 ms | 7.03 ms, plus 4.2 ms per block |
| `inverse_ct` per entry in HintInit      | 72.5 ms | 68.0 ms, plus 9.2 ms per block |
| Cache build, first call per block       | -       | 1.7 ms                         |

The tree is no longer a relevant cost. What remained in `inverse` was the inverse
PRP of about 256 balls per offset (512 fixed iterations in the constant-time
path); see the next section.

## Batched Inverse PRP

The PRP is a Sometimes-Recurse shuffle. Stage `l` permutes the values below
`domain / 2^l` with about 640 to 810 Swap-or-Not rounds (128-bit security,
16,777,216 hints). A round costs one AES block for the round key and one AES
block for the swap decision of a value.

The balls of one offset are a contiguous range. The code inverts the range in
one call:

- `SwapOrNotSr::inverse_range(balls)` (standard path and client). The values
  that a stage holds are a prefix of the range. A round derives its round key
  once for the range, and the swap decisions of the range are one batch of AES
  blocks (`encrypt_blocks`), which the CPU computes in parallel. The loops use
  masks in place of branches, because the swap bits are random.
- `SwapOrNotSrTee::inverse_many_ct(vals)` (constant-time path). All 512 slots
  run every round of every stage; a mask keeps a value that a stage does not
  hold. See [constant_time_mode.md](constant_time_mode.md#fixed-prp-work).

`Iprf::inverse`, `IprfTable::inverse`, `IprfTee::inverse_ct` and
`IprfTeeTable::inverse_ct` use these functions. The single-value functions
`SwapOrNotSr::inverse` and `SwapOrNotSrTee::inverse` are unchanged and are the
reference of the tests. All outputs are equal to those of the single-value
functions (`test_sr_inverse_range_matches_inverse_at_stage_boundaries`,
`test_sr_tee_inverse_many_ct_matches_inverse_at_stage_boundaries`,
`test_iprf_inverse_matches_single_value_prp`; domains 100,003 and 131,072).

Measured at w = 65536 and 16,777,216 hints on one thread (Apple M3 Ultra,
hardware AES through `--cfg aes_armv8`), per offset:

| Operation                           | Before  | After   |
|-------------------------------------|---------|---------|
| `inverse` in HintInit (table)       | 7.04 ms | 0.52 ms |
| `inverse` (client, 8 cached levels) | 7.06 ms | 0.52 ms |
| `inverse` (no table, no cache)      | 7.53 ms | 1.08 ms |
| `inverse_ct` in HintInit (table)    | 68.3 ms | 16.7 ms |
| `inverse_ct` (no table)             | 72.5 ms | 21.1 ms |

`forward` is a single-value function and is not changed.

## Worker Threads

Both phases of HintInit run on `--threads` worker threads (`plinko::hint_init`).
The default is the number of available cores; with `--constant-time` it is 1
(see [constant_time_mode.md](constant_time_mode.md#worker-threads)).

- **Cutoffs.** Each worker computes the cutoffs of one contiguous part of the
  hints.
- **Streaming.** A worker takes the next block from a shared counter, builds
  the PMNS table of the block, and XORs the entries of the block into shared
  parity accumulators. The accumulators are atomic 64-bit words that the
  workers update with XOR.

XOR is commutative, so the hint file does not depend on the thread count or on
the schedule (`generator_file_is_the_same_for_every_thread_count`,
`hint_sets_are_equal_for_every_thread_count` for `protocol::compute_hints`, and
the unit tests in `plinko/src/hint_init/tests.rs` with 1, 2, 3 and 16 threads
for both streaming phases and with 0 to 16 threads for the cutoffs).

An accumulator holds a record as 64-bit words in little-endian order; a record
size that is not a multiple of 8 leaves the last word partly unused. The
accumulators take the record size, rounded up to 8 bytes, per regular hint and
two times that per backup hint in addition to the hints, until the streaming
phase ends. The constant-time path uses the same accumulators.

The measurements below were made with 32-byte parities, before the record size
was a parameter. Runs with records of 32, 40 and 256 bytes gave CPU times within 1% of each other
([protocol_overview.md](protocol_overview.md#measured-16777216-records)); the
peak memory grows with the record size.

Measured with w = 65536 and 16,777,216 hints (Apple M3 Ultra, 24 performance
cores, hardware AES), full `plinko_hints` runs with `--output`:

| Database            | Threads | Wall time | CPU time | CPU per entry | Peak memory |
|---------------------|---------|-----------|----------|---------------|-------------|
| 2 blocks (5.2 MB)   | 1       | 70.6 s    | 69.6 s   | 0.53 ms       | 1.69 GB     |
| 2 blocks (5.2 MB)   | 2       | 37.1 s    | 71.9 s   | 0.55 ms       | 1.69 GB     |
| 24 blocks (62.9 MB) | 8       | 117.0 s   | 904.5 s  | 0.58 ms       | 1.76 GB     |
| 24 blocks (62.9 MB) | 24      | 41.9 s    | 924.1 s  | 0.59 ms       | 1.78 GB     |

The hint files of the two 2-block runs are byte-identical, and so are those of
the 24-block runs.

The 24-block run on 24 threads with 40-byte parities (2026-11-17, the record
size as a parameter): 42.5 s of wall time, 925.3 s of CPU time and a peak memory
footprint of 2.11 GB (2.18 GB of resident memory with the mapped database
pages). The hint file has 1,073,741,920 bytes. The memory is the hint set and
the accumulators of the worker threads: `2 * lambda * w * 3r` bytes, which is
2.01 GB for `r = 40`, plus the cutoffs.

Cutoffs on worker threads (2026-11-18). The same 24-block run on 24 threads,
before and after the change:

| Step                 | Cutoffs on one thread | Cutoffs on 24 threads |
|----------------------|-----------------------|-----------------------|
| Cutoff phase         | 2.03 s                | 0.20 s                |
| Whole run, wall time | 42.5 s                | 41.4 s                |
| Whole run, CPU time  | 925.3 s               | 931.6 s               |

The hint file of the run after the change is byte-identical to the file of the
run before it. This database has 24 blocks, so a cutoff is cheap here; the cost
of a cutoff grows with `c`. `hint_cutoffs` alone at c = 27,988 for 262,144
hints: 10.58 s on 1 thread (40 us per hint), 1.48 s on 8 threads and 0.51 s on
24 threads, with equal cutoffs. `hint_cutoffs_ct` (`tee` feature) takes 26.7 s
on 1 thread (102 us per hint) and 1.26 s on 24 threads for the same input
([constant_time_mode.md](constant_time_mode.md#fixed-cutoff-work)).

Round keys without a division (2026-11-18, hint file version 3; see
[Round Keys](#round-keys)). The 24-block run on 24 threads: 40.5 s of wall
time and 915.8 s of CPU time, against 41.4 s and 931.6 s before. The
difference is inside the variation between runs.

`protocol::compute_hints` used one thread before; it now calls the same code.
24 blocks of w = 4096 (98,304 records of 40 bytes, 1,048,576 hints): 47.7 s on
1 thread and 2.36 s on 24 threads, with equal hint sets. The timings of
`hint_cutoffs` and of `compute_hints` come from scratch programs that are not
in the repository.

### Mainnet Cost

Mainnet v3 has 27,988 blocks of 65,536 entries (1,834,221,568 entries with
padding). No mainnet run was done; these numbers are extrapolations from the
24-block runs above:

- About 300 core-hours (0.59 ms per entry), where the code before the batched
  PRP needed about 3,600 core-hours (7.04 ms per entry).
- About 13 hours of wall time on 24 cores of this machine.
- The cutoffs of the hints are computed before the streaming phase: about 11
  minutes of CPU time at c = 27,988 (40 us per hint), which is about 32 s on 24
  threads (extrapolated from 262,144 hints). They ran on one thread before.
  The wall time of the whole HintInit stays at about 13 hours on 24 cores.

A build without hardware AES is about 20 times slower. On arm64 the `aes` crate
uses hardware AES only with `RUSTFLAGS="--cfg aes_armv8"`.

## Round Keys

The round key of a round in a stage of size `n` is one AES block,
`AES(level || round || n)`, mapped to `0..n` as `floor(r * n / 2^128)` with the
128-bit block value `r` (`reduce_below` in `plinko/src/iprf.rs`).
`SwapOrNotSr` and `SwapOrNotSrTee` share this function (`sr_round_key`).

- **No division.** The mapping is two widening multiplications. Before hint
  file version 3 the key was a 64-bit AES output modulo `n`: a division whose
  dividend depends on the block key, in the standard path and in the
  constant-time path.
- **Bias.** For a power of two `n` the result is the top bits of `r` and is
  exactly uniform. The default hint count (`2 * lambda * w` with lambda = 128)
  is a power of two, so all stage sizes are powers of two. For other hint
  counts the statistical distance of a round key from uniform is at most
  `n / 2^128`; the sum over all round keys of one block key is about
  `1600 * total_hints / 2^128` (2^-93 for 16.8 million hints; computed). With
  the old reduction the sum was about `1600 * total_hints / 2^64` (2^-29).
- **Stage of size 2.** This stage has one round and the fixed round key 1. The
  partner of `x` is then `1 - x`, and the swap bit alone decides the order of
  the two values: a swap with probability 1/2
  (`test_sr_stage_of_size_2_is_uniform`). With a random key the round never
  swaps for key 0, which gave a swap with probability 1/4 (measured before the
  change over 48,000 keys on a domain of 2: 0.2495). As the last stage of a
  larger domain the old behavior had no measurable effect; a PRP with a domain
  of exactly 2 was biased.

The change gives other hints for the same seed. A hint file of version 2 is
not valid for this code, and the reader rejects it
([`data_format.md`](data_format.md#hint-file-hintsbin)).

The one-level `SwapOrNot` is not changed: its round key is an AES output
modulo the domain. HintInit and the protocol do not use it.

## Hint File

`--output` writes one file with all regular and backup hints. The layout is in
[`data_format.md`](data_format.md#hint-file-hintsbin) and the code is in the
library module `plinko::hints`.

- The generator writes to `<output>.<pid>.tmp`, which it creates exclusively,
  and then renames it. On Unix the file has mode 0600.
- The file holds no seed. It holds an 8-byte check value of the seed, so that a
  client detects a wrong seed.
- The file and the seed are client-secret. Whoever holds both can link each
  query of the client to its hint.
- The fast path and the constant-time path give byte-identical files for the
  same seed and database (`scripts/local_e2e_hint_pipeline.sh` checks this for
  40-byte records, and `constant_time_path_writes_the_same_file_for_each_record_size`
  in `hint_file_e2e_test.rs` for 1, 13, 40, 45 and 256 bytes; both need the `tee` feature).

The protocol client loads the file:

```rust
use plinko::hints::HintSet;
use plinko::protocol::{Client, ProtocolParams};

let file = std::fs::File::open("hints.bin")?;
let hints = HintSet::read_from(std::io::BufReader::new(file))?;
let params = ProtocolParams::for_hint_set(&hints)?;
let client = Client::from_hints(params, master_seed, query_rng_seed, hints)?;
```

`Client::from_hints` rejects a hint set with another record size, another
geometry, other hint counts, or another master seed. `plinko/tests/hint_file_e2e_test.rs` runs the
binary, loads the file, and checks queries and an update.

## Block Subsets

Each hint stores its block subset as one 32-bit cutoff. The scheme is the
median-cutoff representation of RMS24 (Ren, Mughees, Sun, CCS 2024, section 3.2),
with the subset sizes of Plinko Fig. 7. The code is in the library module
`plinko::subsets`; `plinko_hints` and the protocol client both use it.

- `v[j][k] = PRF("select", j, k)` is a 32-bit value for hint `j` and block `k`.
- The cutoff of hint `j` has exactly `c/2 + 1` values of `v[j]` below it for a
  regular hint, and exactly `c/2` for a backup hint.
- Block `k` is in the subset of hint `j` when `v[j][k] < cutoff`.
- Hint IDs are global: regular hints are `0..num_regular`, and backup hint `b`
  is `num_regular + b`. The iPRF uses the same IDs.

The rank order of independent PRF values is a uniform permutation. The subset
is thus uniform among the subsets of that size, as Fig. 7 requires.

PRF: the key is `SHA-256(master_seed || "plinko_select")[0..16]`. One AES-128
block `AES(key, hint_be64 || (block / 4)_be64)` gives four big-endian `u32`
values; block `k` uses word `k % 4`. The key is client-secret.

### Invalid hints

If the two values at the subset boundary are equal, no cutoff selects the exact
number of blocks. The cutoff is then `0` (`INVALID_CUTOFF`), and the hint is
invalid: its parities stay zero, and the client does not use it for a query, a
promotion, or an update. S3PIR (the RMS24 reference code) uses the same rule.
The probability is about `c / 2^33` per hint. For mainnet (c = 27988,
16,777,216 hints) that is about 55 invalid hints (computed, not measured).
`plinko_hints` logs the counts as `invalid_regular` and `invalid_backup`.

### Memory

| Representation                      | Per hint         | Mainnet, 16,777,216 hints |
|-------------------------------------|------------------|---------------------------|
| Sorted block list (client, removed) | about `4c` bytes | about 1.9 TB              |
| Packed bitmap (generator, removed)  | `ceil(c/64) * 8` | about 59 GB               |
| Cutoff                              | 4 bytes          | 67 MB                     |

The values are computed from c = 27988, not measured.

### Cost

A membership test is one AES-128 block. A cutoff needs `c/4` AES blocks and a
selection (`select_nth_unstable`) over `c` values. The compute bottleneck
remains the Swap-Or-Not rounds in the iPRF.

Measured at c = 27988 on one thread (Apple M3 Ultra, hardware AES through
`--cfg aes_armv8`):

| Operation                       | Time                                   |
|---------------------------------|----------------------------------------|
| Membership test (`contains`)    | 1.4 ns                                 |
| Cutoff of one hint              | 44 us                                  |
| Cutoffs of all 16,777,216 hints | about 740 s (extrapolated from 20,000) |
