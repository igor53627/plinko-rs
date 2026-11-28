# Keyword Lookup: Design Specification

Status: implemented in `plinko::keyword` (`plinko/src/keyword.rs`,
`plinko/src/keyword/client.rs`), benchmarked by `bench_keyword`, and covered by
`plinko/tests/keyword_e2e_test.rs` (TASK-22).

This document specifies private key-value lookup over `plinko-rs` for a general
fixed-size key-value database, parameterized by key length, value size, tag
size, bucket size, and load factor. The full Ethereum state is not a target
([protocol_overview.md](protocol_overview.md#scope)); large-scale counts below
are scale references only.

## Problem

Index PIR (`plinko::multilevel` and `plinko::protocol`) retrieves a record at
an integer index `0..N`. A client that knows a key (such as an identifier,
address, or composite key) cannot ask the server for its index without
revealing the key, and storing an explicit key-to-index map on the client costs
`O(N)` space.

We reduce keyword PIR to index PIR using a **2-segment Key-Value Segmented
Cuckoo Filter (`KV-SCF`)**, adapted from RisePIR (`ePrint 2026/2082`, CANS
2026) with modular partial-key cuckoo hashing. It replaces the earlier
unsegmented Cuckoo proposal (`d=2, b=4`) and works over both
`plinko::multilevel` (recommended) and 1-level `plinko::protocol`.

## Parameters

A table is defined by its public `KeywordTableSpec`:

| Parameter | Symbol | Default | Constraints / Notes |
|---|---|---|---|
| Segments | `d` | `2` | Fixed to 2 in v1; each lookup issues `d = 2` index PIR queries |
| Slots per bucket | `b` | `4` | `>= 1`; `b = 4` balances load (0.85-0.90) and bucket size (`b = 2` lowers backup hint size) |
| Tag size (bytes) | `tag_size` | `8` | `8` (64-bit, default) or `16` (128-bit, for adversarial key spaces) |
| Value size (bytes) | `value_size` | caller | Fixed per table (`>= 1`); e.g. 16 B, 24 B, 32 B |
| Key size (bytes) | `key_size` | arbitrary | Fixed or variable-length byte slices; length-prefixed in hashing |
| Target build load | `load` | `0.85`-`0.90` | `0 < load < 0.95`; `0.90` for mostly static tables, `0.80`-`0.85` for live inserts |
| Buckets per segment | `s` | derived | `max(1, ceil(ceil(num_keys / (b * load)) / 2))` (`s >= 1`, even for `num_keys = 0`; no power-of-two rounding required) |
| Table seed | `table_seed` | random 32 B | Public; regenerated on build failure or major epoch rotation |
| Table epoch (major) | `table_epoch` | `0` | Increments only on seed or geometry change (requires fresh `HintInit`) |
| State epoch (minor) | `state_epoch` | `0` | Increments on every delta batch (`UpdateHint` / `apply_deltas_batch`) |

Derived sizes:

- `slot_size = tag_size + value_size`
- `record_size = b * slot_size` (one PIR entry = one bucket)
- `total_buckets = 2 * s`
- `database_bytes = 2 * s * record_size`
- **PIR queries per lookup:** **`2`** (one query in segment 0, one query in
  segment 1).

## Layout

```text
database.bin = segment_0 | segment_1                  (2 * s * record_size bytes)
segment_j    = bucket[0] | ... | bucket[s - 1]        (s buckets, j in {0, 1})
bucket       = slot[0] | ... | slot[b - 1]            (one PIR entry, record_size bytes)
slot         = tag (tag_size bytes) | value (value_size bytes)
```

Storing `segment_0` followed contiguously by `segment_1` serves both execution
modes from one file:

1. **Two independent segment PIR instances (recommended for
   `plinko::multilevel`):** segment 0 is slice `[0 .. s * record_size]` and
   segment 1 is `[s * record_size .. 2 * s * record_size]`. Each segment has
   its own hint set, so a lookup sends both queries in **1 RTT** with zero hint
   contention and no `pos_0 == pos_1` edge case.
2. **Single-database PIR instance (compatible with `plinko_hints` and 1-level
   `plinko::protocol`):** treat `database.bin` as one table of `2 * s` entries
   and query indices `a0` and `s + a1`. Because `a0 < s <= s + a1`, the two
   indices are still guaranteed distinct.

An **empty slot** is all bytes zero (`tag == [0; tag_size]` and
`value == [0; value_size]`).

## Hashing And Partial-Key Eviction

Hashing uses keyed **BLAKE3** (deterministic across native and `wasm32`,
already a dependency of `plinko`) with the 32-byte public `table_seed`. A
1-byte `kind` namespace separates distinct key domains stored in one table
(use `0x00` for single-domain tables).

For input `(kind, key)`:

1. Compute 32 bytes via BLAKE3 keyed XOF with domain separation and explicit
   key length (`key.len() as u32` little-endian):
   `digest = blake3_keyed(table_seed, "plinko-kvscf-v1" || kind || u32_le(key.len()) || key)`
2. Primary bucket offset in segment 0:
   `a0 = u64_le(digest[0..8]) mod s`
3. Tag:
   `tag = digest[8 .. 8 + tag_size]`.
   If all bytes of `tag` are zero, set `tag[0] = 1` so no valid key maps to
   the empty-slot sentinel (costs 1 of `2^t` tag values and avoids build
   restarts on zero tags).
4. Alternate bucket offset in segment 1 via a second keyed BLAKE3 hash of
   `tag` alone:
   `delta = u64_le(blake3_keyed(table_seed, "plinko-kvscf-alt-v1" || tag)[0..8]) mod s`
   `a1 = (a0 + delta) mod s`

**Partial-key eviction:** during build and live inserts, the server always
knows which segment (`0` or `1`) an evicted slot sits in, so it recovers the
alternate bucket strictly from `(segment, bucket_idx, tag)` without keeping
original keys or a 4-byte `other_bucket` side array:

- Evicting from segment 0 at `a0`: target in segment 1 is `(a0 + delta) mod s`.
- Evicting from segment 1 at `a1`: target in segment 0 is
  `(a1 + s - delta) mod s`.

Using modular addition/subtraction instead of XOR works for any `s >= 1`,
avoiding up to 2x capacity padding from rounding `s` to a power of two.

## Lookup Protocol

1. The client computes `(a0, a1, tag)` locally from `KeywordTableSpec` and
   `(kind, key)`. The key never leaves the client.
2. The client prepares 2 index PIR queries: `a0` against segment 0 and `a1`
   against segment 1 (sent together in 1 RTT).
3. Upon receiving both buckets, the client verifies `table_epoch`, global
   `state_epoch`, and each segment's own backend epoch (see
   [Client Interface](#client-interface-plinkokeyword)), then collects all
   matches across both sources:
   - Scan all `2 * b` slots of the two buckets for `slot.tag == tag`.
   - If a bounded public runtime-stash is present in epoch metadata, also
     check the stash for `(a0, tag)` locally (while still issuing the 2 PIR
     queries unless `CachePolicy::LocalHit` is explicitly enabled).
4. Combine the bucket matches and runtime-stash matches before choosing the
   outcome:
   - Zero total matches: return `None` (not found).
   - One total match: return `Some(value)`.
   - More than one total match (whether within the buckets or across a bucket
     and the stash): return `Err(DuplicateTagMatch)`.

## Tag Size And False-Match Probability

Let `t = 8 * tag_size` be the tag width in bits (`2^t - 1` non-zero tags).

1. **Non-member false match (key not in table):**
   Model a fresh non-member query's keyed BLAKE3 digest as producing
   `(a0, raw_tag)` uniform over `[0, s) x {0, 1}^t` relative to the stored
   table identities (this bound does not apply to keys adversarially chosen
   against a known `table_seed` and snapshot). By the modular relation
   `a1 = (a0 + h1(tag)) mod s`, a query `(a0, tag)` matches a stored table
   entry `(a0', tag')` in either segment if and only if `a0 == a0'` and
   `tag == tag'`.
   - For a fixed primary bucket `a0`, let `C(a0)` be the number of distinct
     stored tags in the table whose primary bucket is `a0` (with no assumption
     that stored tags are independent). Because the zero-tag remap
     (`raw_tag = [0; tag_size] -> [1, 0, ...]`) gives at most one stored tag
     per `a0` mass `2 / 2^t` while every other tag has mass `1 / 2^t`, the
     conditional false-match probability given `a0` is at most
     `(C(a0) + 1) / 2^t`.
   - Averaging over uniform `a0 in [0, s)` and using
     `sum_{a0} C(a0) = N <= d * b * s` yields the conservative table-only
     upper bound reported by `KeywordTableSpec::false_match_upper_bound()`
     (when the runtime-stash is empty or disabled):
     `(N / s + 1) / 2^t <= (d * b + 1) / 2^t`:
     - `tag_size = 8` (`t = 64`, `d = 2, b = 4`): `<= 9 / 2^64 ~= 4.9e-19`
       (`~2^-61` per lookup).
     - `tag_size = 16` (`t = 128`, `d = 2, b = 4`): `<= 9 / 2^128 ~= 2.6e-38`
       (`~2^-125` per lookup).
   - When the optional runtime-stash is enabled with at most `S_stash`
     distinct entries matching `a0` (worst-case `S_stash` equals the full
     stash capacity, for example 256), counting those additional distinct tags
     adds `S_stash / 2^t`, giving `(d * b + S_stash + 1) / 2^t`
     (`<= 265 / 2^64 ~= 1.44e-17` for `tag_size = 8, S_stash = 256`).

2. **All-or-nothing collision for stored keys (RisePIR Lemma 1):**
   Because `delta = h1(tag)` depends only on `tag`, two distinct keys `k != k'`
   with `tag(k) == tag(k')` satisfy `a1(k) == a1(k')` if and only if
   `a0(k) == a0(k')`. Thus two keys with the same tag either share both
   candidate buckets or share neither.
   - For a table of `N` honest keys, a given key collides with another stored
     key with probability `<= N / (s * (2^t - 1)) = (d * b * load) / (2^t - 1)`
     (`~3.9e-19` at `t = 64, load = 0.90`).
   - Across the entire table (birthday bound), the probability that any pair
     collides on `(a0, tag)` is `~= (d * b * load * N) / (2 * 2^t)`:
     `~2^-38` at `N = 2^24` and `~2e-11` at `N = 10^8` for `t = 64`. The
     builder rejects duplicate `(a0, tag)` pairs during build.
   - **Adversarial keys (public `table_seed`):** grinding a collision against
     one specific victim key requires `~s * 2^t` hashes (`~2^88` for
     `s = 1.5e7, t = 64`); grinding a collision against any of `N` stored keys
     requires `~2^t / (d * b * load)` hashes (`~2^61` for `t = 64`, `~2^125`
     for `t = 128`). Use `tag_size = 16` when untrusted parties can freely
     grind keys against a known `table_seed`.

## Mutations After Build (Inserts, Updates, Deletes)

Every mutation on the table produces a small batch of `(segment, bucket_idx,
xor_delta)` updates that feed directly into `MultiLevelClient::apply_deltas_batch`
or `protocol::Client::apply_delta`:

- **Value update:** locate the slot matching `tag` in `(seg 0, a0)` or
  `(seg 1, a1)` (or, when the optional runtime-stash is enabled, the stash
  entry matching `(a0, tag)`). Overwriting a bucket slot produces **1 bucket
  delta** and does not change load; overwriting a stash entry emits
  `StashOp::Update(a0, tag, new_value)` and no bucket delta.
- **Delete:** locate the matching slot in `(seg 0, a0)` or `(seg 1, a1)` and
  zero it out (**1 bucket delta**), or remove the matching `(a0, tag)` entry
  from the runtime-stash (`StashOp::Delete(a0, tag)`). Deletion is weak in the
  preprocessing PIR model: future lookups return `None`, while clients that
  logged earlier deltas retain old values.
- **Insert:**
  1. Reject if a slot with the same `tag` already exists in `(seg 0, a0)` or
     `(seg 1, a1)` -- or, when the runtime-stash is enabled, if `(a0, tag)` is
     already present in the stash (duplicate key or `(a0, tag)` collision).
  2. If either candidate bucket has a free slot, write `tag || value` there
     (**1 bucket delta**).
  3. Otherwise run random-walk partial-key eviction for up to `max_kicks`
     steps (default `500`), alternating between segment 0 and segment 1 and
     computing each displaced slot's alternate bucket from its stored `tag`.
     At `d = 2, b = 4`, an insert touches ~2-3 buckets on average at
     `load = 0.80-0.85` and ~5.4 buckets on average at `load = 0.90`.
  4. If `max_kicks` is exhausted, roll back the tentative walk and report
     `InsertError::TableFull` (no partial state left behind).

**Handling saturation without immediate rebuild:**

- Choose `load = 0.80-0.85` at build time when live inserts are expected
  (leaves 15-25% free slots and keeps eviction walks short).
- Optional **bounded public runtime-stash** (for example up to 256 entries,
  ~12 KB in epoch metadata): when a live insert hits `TableFull`, the server
  can append `(a0, tag, value)` to the public stash (`StashOp::Insert`) until a
  scheduled major epoch rotation (`table_epoch += 1` with larger `s` and fresh
  `table_seed`, which triggers a new `HintInit` on clients). Keying the stash
  by `(a0, tag)` matches the table's exact key identity (preventing a stashed
  key from matching a different or absent key that shares only `tag` with a
  different `a0`). Because the stash is public, each stashed entry publicly
  exposes `(a0, tag, value)` -- the same bucket offset, tag, and value that an
  on-table slot already exposes publicly in `database.bin` and delta streams.
- Build-time construction never uses a stash: if a key fails to place during
  build, the builder reports the unplaceable key (or retries with a new
  `table_seed` when retry is requested).

## Comparison With Other Keyword PIR Constructions

```text
+------------------------------+---------+---------+------------+-------------+----------------------------+
| Construction                 | Queries | Record  | Extra      | Dynamic     | Fit for plinko::multilevel |
|                              | /lookup | size    | client RAM | inserts     |                            |
+------------------------------+---------+---------+------------+-------------+----------------------------+
| Unsegmented Cuckoo C4        | 2       | 4*(t+v) | 32 B seed  | Yes (O(1))  | Shared hint pool; needs    |
| (prior docs/keyword_lookup)  |         |         |            |             | other_bucket + Rewind/Skip |
+------------------------------+---------+---------+------------+-------------+----------------------------+
| 2-segment KV-SCF (selected)  | 2       | 4*(t+v) | 32 B seed  | Yes (O(1))  | 2 disjoint PIR instances,  |
| (RisePIR, ePrint 2026/2082)  | (1/seg) |         |            |             | 1 RTT, partial-key evict   |
+------------------------------+---------+---------+------------+-------------+----------------------------+
| MPHF-KVS (future static opt) | 1       | 1*(t+v) | 1.5-2.7    | No (value   | 1 query, load 1.0, b=1;    |
| (ePrint 2025/2252)           |         | (b=1)   | bits/key   | update only)| rebuild on any key insert  |
+------------------------------+---------+---------+------------+-------------+----------------------------+
| BFF-KVS (ChalametPIR /       | 3       | 1*(t+v) | 32 B seed  | No (rebuild | 3x queries; ChalametPIR's  |
| KPIR_kvs / 2025/2252)        |         | (~1.13x)|            | on update)  | 1-query trick needs LWE    |
+------------------------------+---------+---------+------------+-------------+----------------------------+
| KPIR_index / KPIR_hash       | 1       | wide    | 0.5-8      | No / update | Built for Row-KOPIR; wide  |
| (USENIX Security 2025)       |         | window  | bits/key   | only        | records bloat XOR hints    |
+------------------------------+---------+---------+------------+-------------+----------------------------+
```

Note on sublinear PIR scaling: on linear-server PIR (SimplePIR/FrodoPIR),
splitting into `d = 2` segments halves server work. On sublinear `sqrt(N)`
PIR (`S * T = Omega(M)`), splitting preserves the `2x` space-time product, but
with `plinko::multilevel` (where hints are already 15-35x smaller than 1-level
Plinko) it delivers 1-RTT parallel lookup without hint contention, removes the
`other_bucket` side array, eliminates `pos_0 == pos_1` decoy logic, and lets
callers tune `b = 4` vs `b = 2` cleanly.

## Concrete Costs With `plinko::multilevel` (`FlatB`, `kappa = 100`)

All rows use exact `MultiLevelParams::two_level(..., kappa = 100, FlatB)`
geometry (`> 137` bits of session security across `2 * Q` segment queries).
CPU timings are calibrated extrapolations from Apple M3 Ultra (`--cfg
aes_armv8`, `docs/multilevel_pir.md`).

```text
+-----------------------------------+-----------+-----------+------------+------------+------------+-----------+-----------+
| Workload (d=2 KV-SCF, kappa=100)  | DB size   | Hints     | Primary /  | Upload L0  | Download / | HintInit  | 1k deltas |
|                                   | (stream)  | (2 segs)  | Backup     | (expected) | Server L0  | (8T / 32T)| (1 core)  |
+-----------------------------------+-----------+-----------+------------+------------+------------+-----------+-----------+
| 2^24 index-only baseline (40 B)   | 640.0 MiB |  1.90 MiB | 0.90/1.00  | 12.0 KiB   | 80 B /     | ~4.4 s /  | ~0.6 ms   |
| (no keyword lookup, Q=4096)       |           |           | MiB        | (18.4 KiB) | 0.31 MiB   | ~1.1 s    |           |
+-----------------------------------+-----------+-----------+------------+------------+------------+-----------+-----------+
| 2^24 keys, 40 B slot, b=4         | 711.1 MiB | 11.27 MiB | 3.45/7.82  |  6.1 KiB   | 640 B /    | ~2.4 s /  | ~0.6 ms   |
| load=0.90, Q=4096 (142.6 b/sess)  | (0.75 GB) |           | MiB        | (9.0 KiB)  | 0.69 MiB   | ~0.6 s    |           |
+-----------------------------------+-----------+-----------+------------+------------+------------+-----------+-----------+
| 2^24 keys, 40 B slot, b=2         | 800.0 MiB |  7.44 MiB | 3.53/3.91  |  7.5 KiB   | 320 B /    | ~3.2 s /  | ~0.9 ms   |
| load=0.80, Q=4096 (141.6 b/sess)  | (0.84 GB) |           | MiB        | (11.5 KiB) | 0.39 MiB   | ~0.8 s    |           |
+-----------------------------------+-----------+-----------+------------+------------+------------+-----------+-----------+
| 100M EOA, 24 B slot, b=4          |  2.82 GB  |  8.90 MiB | 4.21/4.69  | 21.0 KiB   | 384 B /    | ~10.3 s / | ~0.89 ms  |
| load=0.85, Q=4096 (140.2 b/sess)  | (2.63 GiB)|           | MiB        | (32.3 KiB) | 1.31 MiB   | ~2.6 s    |           |
+-----------------------------------+-----------+-----------+------------+------------+------------+-----------+-----------+
| 100M EOA, 32 B slot, b=4          |  3.76 GB  | 11.81 MiB | 5.55/6.26  | 21.0 KiB   | 512 B /    | ~11.8 s / | ~0.89 ms  |
| load=0.85, Q=4096 (140.2 b/sess)  | (3.51 GiB)|           | MiB        | (32.3 KiB) | 1.75 MiB   | ~2.9 s    |           |
+-----------------------------------+-----------+-----------+------------+------------+------------+-----------+-----------+
| 300M scale ref, 32 B slot, b=4    | 11.29 GB  | 17.36 MiB | 11.11/6.26 | 34.2 KiB   | 512 B /    | ~32.5 s / | ~1.5 ms   |
| load=0.85, Q=4096 (139.1 b/sess)  | (10.52GiB)|           | MiB        | (54.4 KiB) | 2.63 MiB   | ~8.1 s    |           |
+-----------------------------------+-----------+-----------+------------+------------+------------+-----------+-----------+
| 300M scale ref, 40 B slot, b=4    | 14.12 GB  | 21.62 MiB | 13.80/7.82 | 34.2 KiB   | 640 B /    | ~36.6 s / | ~1.5 ms   |
| load=0.85, Q=4096 (139.1 b/sess)  | (13.15GiB)|           | MiB        | (54.4 KiB) | 3.29 MiB   | ~9.1 s    |           |
+-----------------------------------+-----------+-----------+------------+------------+------------+-----------+-----------+
```

Notice that packing `b = 4` slots per bucket shrinks the number of PIR entries
by `3.6x`, which roughly **halves online upload and `HintInit` CPU** relative
to the 40-byte index-only baseline, while backup hints scale linearly with
`Q * record_size` (using `b = 2` or a smaller session budget `Q` halves backup
memory when RAM is tighter than upload). For wall-clock measurements of the
1M, 10M, and 100M EOA rows (including `b = 4` vs `b = 2` on 100M keys), see
[Measured Benchmarks](#measured-benchmarks-bench_keyword-apple-m3-ultra) below.

### Measured Benchmarks (`bench_keyword`, Apple M3 Ultra)

Measured on Apple M3 Ultra (32 cores, `RUSTFLAGS="--cfg aes_armv8"`, release
build, `v0.4.0`) with `key_size = 20 B` and `tag_size = 8 B` (`false_match`
bound `4.88e-19` for `b = 4`, `2.71e-19` for `b = 2`). All runs built with
**zero stash** (`0` unplaceable keys) and verified every PIR lookup and
post-delta lookup against plaintext `KvScfTable::get`.

Reproduce:

```bash
RUSTFLAGS="--cfg aes_armv8" cargo build --release -p plinko --bin bench_keyword
./target/release/bench_keyword --num-keys 100000000 --key-size 20 --value-size 24 \
  --tag-size 8 --bucket-size 4 --load 0.85 --kappa 100 --query-budget 4096
```

#### Scaling 1M to 10M keys (`b = 4`, `Q = 1024`, 32T)

```text
+-----------------------------------+-------------------+-------------------+-------------------+-------------------+
| Metric (key=20B, tag=8B, Q=1024)  | 1M default        | 1M compact EOA    | 10M default       | 10M high-sec      |
|                                   | val=32B, l=0.90   | val=24B, l=0.85   | val=32B, l=0.90   | val=32B, l=0.90   |
|                                   | kappa=64          | kappa=64          | kappa=64          | kappa=100         |
+-----------------------------------+-------------------+-------------------+-------------------+-------------------+
| Buckets / segment (record_size)   | 138,889 (160 B)   | 147,059 (128 B)   | 1,388,889 (160 B) | 1,388,889 (160 B) |
| DB size (database.bin)            | 42.39 MiB         | 35.90 MiB         | 423.86 MiB        | 423.86 MiB        |
| Build (1T, 0 stash)               | 318.9 ms (3.14M/s)| 279.5 ms (3.58M/s)| 4.09 s (2.44M/s)  | 4.09 s (2.45M/s)  |
| Evictions (max / tail p99)        | 43 / 19           | 20 / 9            | 52 / 20           | 52 / 20           |
| Plaintext lookup (100k, 50/50)    | 35.2 ms (2.84M/s) | 32.5 ms (3.08M/s) | 41.4 ms (2.42M/s) | 41.5 ms (2.41M/s) |
| Live update / delete (1k ops)     | 0.48 / 0.39 ms    | 0.45 / 0.35 ms    | 0.63 / 0.53 ms    | 0.63 / 0.53 ms    |
| Live insert (1k ops, deltas/op)   | 1.50 ms (3.79)    | 0.83 ms (1.90)    | 2.02 ms (3.85)    | 2.04 ms (3.85)    |
| 2-seg FlatB HintInit (32T)        | 53.38 ms          | 47.90 ms          | 313.24 ms         | 539.32 ms         |
| Hints (2 segments)                | 2.89 MiB          | 2.31 MiB          | 5.37 MiB          | 5.59 MiB          |
| Upload L0 (2 segments)            | 1.19 KiB          | 1.26 KiB          | 3.64 KiB          | 3.64 KiB          |
| Online PIR (prep / server / fin)  | 107 / 52 / 3.2 us | 92 / 30 / 2.9 us  | 212 / 176 / 8.8us | 363 / 287 / 9.4us |
| Apply 100 deltas (2 segs, 32T)    | 1.49 ms           | 1.36 ms           | 1.48 ms           | 1.44 ms           |
+-----------------------------------+-------------------+-------------------+-------------------+-------------------+
```

Additional 1M / 10M points:
- 1M high-sec (`val = 32 B`, `load = 0.90`, `kappa = 100`, `Q = 1024`):
  `HintInit = 107.84 ms`, `hints = 3.00 MiB`, `L0 upload = 1.19 KiB`,
  `prep = 107.1 us`, `server = 39.8 us`, `finish = 3.3 us`,
  `apply 100 deltas = 1.57 ms`.
- 10M compact EOA (`val = 24 B`, `load = 0.85`, `kappa = 64`, `Q = 1024`):
  `DB = 359.03 MiB`, `build = 3.69 s` (`2.71 Mkeys/s`), `HintInit = 249.33 ms`,
  `hints = 4.31 MiB`, `L0 upload = 3.86 KiB`, `apply 100 deltas = 1.41 ms`.
- Writing `database.bin` to local NVMe measured `4.62 ms` (`9.62 GB/s`) at 1M
  and `47.81 ms` (`9.30 GB/s`) at 10M.

#### 100M EOA profiles: `b = 4` (`load = 0.85`) vs `b = 2` (`load = 0.80`)

All 100M runs use `key_size = 20 B`, `tag_size = 8 B`, `kappa = 100`
(`140.2` bits of session security for `b = 4`), `Q = 4096`, and 32 threads:

```text
+-----------------------------------+-------------------+-------------------+-------------------+-------------------+
| Metric (100M EOA, kappa=100,      | 24 B slot (v=16B) | 24 B slot (v=16B) | 32 B slot (v=24B) | 32 B slot (v=24B) |
| Q=4096, 32T)                      | b=4, load=0.85    | b=2, load=0.80    | b=4, load=0.85    | b=2, load=0.80    |
+-----------------------------------+-------------------+-------------------+-------------------+-------------------+
| Bucket size (record_size)         | 96 B              | 48 B (0.50x)      | 128 B             | 64 B (0.50x)      |
| Buckets / segment                 | 14,705,883        | 31,250,000 (2.13x)| 14,705,883        | 31,250,000 (2.13x)|
| DB size (stream)                  | 2.63 GiB          | 2.79 GiB (+6.2%)  | 3.51 GiB          | 3.73 GiB (+6.2%)  |
| Build 100M keys (1T, 0 stash)     | 38.89 s (2.57M/s) | 49.60 s (2.02M/s) | 39.42 s (2.54M/s) | 46.00 s (2.17M/s) |
| Evictions (max / tail p99)        | 35 / 9            | 98 / 32           | 35 / 9            | 98 / 32           |
| Plaintext lookup (100k, 50/50)    | 41.99 ms (2.38M/s)| 42.57 ms (2.35M/s)| 42.93 ms (2.33M/s)| 48.42 ms (2.07M/s)|
| Live update / delete (1k ops)     | 0.82 / 0.71 ms    | 0.79 / 0.59 ms    | 0.86 / 0.73 ms    | 0.90 / 0.75 ms    |
| Live insert (1k ops, deltas/op)   | 1.37 ms (1.95)    | 2.59 ms (4.96)    | 1.42 ms (1.95)    | 3.04 ms (4.96)    |
| 2-seg FlatB HintInit (32T)        | 3.87 s            | 6.44 s (+66.5%)   | 3.71 s            | 5.81 s (+56.8%)   |
| Hints (2 segments)                | 8.90 MiB          | 6.72 MiB (-24.5%) | 11.81 MiB         | 8.85 MiB (-25.1%) |
|   per segment                     | 4.45 MiB          | 3.36 MiB          | 5.91 MiB          | 4.43 MiB          |
| Upload L0 (2 segments)            | 21.04 KiB         | 24.22 KiB (+15.1%)| 21.04 KiB         | 24.22 KiB (+15.1%)|
| Download L0 (2 segments)          | 384 B             | 192 B (-50.0%)    | 512 B             | 256 B (-50.0%)    |
| Online PIR (prep / server / fin)  | 1.36 / 0.81 / 17u | 1.63 / 0.81 / 36u | 1.36 / 0.91 / 17u | 1.63 / 0.54 / 36u |
|   total CPU per private lookup    | 2.19 ms           | 2.47 ms           | 2.29 ms           | 2.20 ms           |
| Apply 100 deltas (2 segs, 32T)    | 1.61 ms           | 1.65 ms           | 1.51 ms           | 1.99 ms           |
+-----------------------------------+-------------------+-------------------+-------------------+-------------------+
```

Key observations from the measured runs:
- **Exact geometry match:** measured `DB`, `hints`, and `L0 upload` on both
  100M `b = 4` profiles match the analytical table (`2.63 / 3.51 GiB`,
  `8.90 / 11.81 MiB`, `21.04 KiB`).
- **`HintInit` at 100M is AES/PRG-bound per bucket:** `b = 4` takes `3.71-3.87 s`
  on 32 cores (versus the `~2.6-2.9 s` bandwidth-weighted extrapolation), and
  `b = 2` takes `5.81-6.44 s` (`+57%` to `+67%`) because the bucket count grows
  `2.13x`.
- **`b = 4` vs `b = 2` tradeoff:** `b = 2` (`load = 0.80`) shrinks client hints
  by **~25%** (`6.72 MiB` / `8.85 MiB`) and halves reply download (`192-256 B`)
  at nearly identical total online CPU (`2.20-2.47 ms`), while `b = 4`
  (`load = 0.85`) cuts live-insert delta amplification by **2.5x** (`1.95` vs
  `4.96` deltas/insert), keeps `L0 upload` 15% smaller, and runs `HintInit`
  ~1.6x faster.
- **Flat measured delta latency across 1M-100M:** `bench_keyword` calls
  `apply_deltas_batch` across both segments (`~50` deltas per segment on 32
  threads); although FlatB delta work scales with hint count (`O(sqrt(N))`),
  thread-scope overhead dominates at this batch size, keeping measured wall
  time flat at `1.36-1.99 ms` per 100 deltas from 1M through 100M keys.

### Simulation Reference For `(d, b, load)`

Random-walk cuckoo simulation (67,108,864 slots, `max_kicks = 500`, evictions
measured on the last 1% of inserted keys):

| `d` | `b` | Load | Keys not placed | Evictions: mean | p99 | max |
|---|---|---|---|---|---|---|
| 2 | 1 | 0.45 | 470 | 2.8 | 59 | 500 |
| 3 | 1 | 0.85 | 0 | 15.5 | 120 | 420 |
| 2 | 2 | 0.80 | 0 | 7.5 | 74 | 268 |
| 2 | 2 | 0.85 | 50 | 26.8 | 222 | 500 |
| 2 | 4 | 0.90 | 0 | 5.4 | 41 | 177 |
| 2 | 4 | 0.95 | 3 | 25.9 | 159 | 500 |
| 2 | 8 | 0.95 | 0 | 5.3 | 35 | 102 |

RisePIR (`ePrint 2026/2082`, Table 2) further confirms that 2-segment `KV-SCF`
with `b = 4` achieves up to **97.4%** maximum load before insertion fails,
leaving comfortable margin at `load = 0.85-0.90`.

## Client Interface (`plinko::keyword`)

The library exposes a **sans-I/O state machine** matching `MultiLevelClient`
and `protocol::Client` (no async runtime or networking in `plinko`, zero host
imports on `wasm32-unknown-unknown`):

1. **`IndexPirClient` trait:** adapter covering segment geometry, query
   preparation/abort/cancel, WAL hint retirement, and staged atomic commits
   (`plan_finish_batch` / `commit_finish_batch` and
   `prepare_index_deltas` / `commit_index_deltas`), implemented out of the box
   for both `MultiLevelClient` and `protocol::Client`.
2. **`KeywordClient<B: IndexPirClient>`:** owns `[B; 2]` (segment 0 and
   segment 1), `KeywordTableSpec`, optional runtime-stash, and optional
   XOR-updated bucket cache:
   - `prepare_lookup(kind, key, rng) -> Result<LookupPlan<Q, P>>`
   - `finish_lookup(pending, answer, rng) -> Result<Option<Vec<u8>>>`
   - `abort_lookup(pending) -> Result<()>` (atomically rolls back both
     segments on transport failure or epoch mismatch)
   - `prepare_batch` / `finish_batch` / `abort_batch` (leverages
     `MultiLevelClient`'s existing `slot.reserved` tracking for 1-RTT multi-key
     batches). Because `protocol::Client::prepare_query` rejects a second
     concurrent query for the same bucket index with `QueryInFlight` (whether
     from a duplicate key or from two distinct keys sharing a bucket `a0` or
     `a1`), `prepare_batch` tracks already-scheduled bucket indices per segment
     within the batch:
     - Under `CachePolicy::LocalHit`, `prepare_batch` deduplicates repeated
       keys and skips cached/stashed hits locally; if two distinct uncached
       keys share a segment bucket `a_j`, `a_j` is queried once and its
       recovered bucket is shared across both entries in `finish_batch` (a
       decoy is substituted only if needed to keep wire pairs balanced, or the
       segment query is omitted when the transport supports unbalanced per-segment
       lists). This reveals the deduplicated non-cached batch count to the
       server.
     - Under `CachePolicy::Off` and `CachePolicy::ConstantTraffic`,
       `prepare_batch` emits one query pair per input key whenever each segment
       has enough distinct indices, using two-pass scheduling per segment so
       internal backend decoys cannot collide with real buckets in the same
       batch:
       1. Pass 1 schedules all distinct real bucket indices required by the
          batch that are not already cached inside the segment backend (so
          `protocol::Client` never performs internal decoy substitution during
          Pass 1).
       2. Pass 2 schedules all remaining slots (backend-cached buckets,
          duplicate keys, shared buckets, and keyword-cached/stashed hits).
          `IndexPirClient` exposes the reserved wire index
          (`PreparedQuery::queried_index()` on `protocol::Client`) so
          `prepare_batch` tracks every backend-reserved bucket
          and always picks a decoy index not currently in flight and not cached
          in segment `j`, sharing the single recovered bucket across all
          matching batch entries in `finish_batch`.
       If segment `j` has no available non-in-flight decoy index (for example
       `batch_len > s` on `protocol::Client`), `prepare_batch` rolls back its
       tentative reservations and returns an error instead of breaking the
       one-pair-per-key guarantee. Within-batch key equality and bucket
       collisions therefore stay hidden from the server (the caller's nominal
       batch size remains visible on the wire).
   - `apply_epoch_update(&KeywordEpochUpdate)`: validates `table_epoch` and
     `from_state_epoch == self.state_epoch`, applies each non-empty
     `segment_deltas[j]` to `backend[j]`, XORs deltas into cached buckets in
     place, applies `stash_ops` to the runtime-stash, and advances global
     `state_epoch` to `to_state_epoch`.
     **Per-segment epoch mapping:** `MultiLevelClient::epoch()` does not
     advance on an empty delta batch (`docs/multilevel_pir.md`), so after an
     update that touches only segment 0, `backend[0].epoch()` advances while
     `backend[1].epoch()` stays unchanged. Each segment answer `segments[j]`
     therefore carries its own segment backend epoch `seg_epoch_j` (not the
     global `state_epoch`), and `finish_lookup` checks both
     `answer.state_epoch == self.state_epoch` (global keyword state, covering
     the runtime-stash as well) and `seg_epoch_j == backend[j].epoch()` per
     segment independently.
   - `write_checkpoint` / `read_checkpoint` and `to_bytes` / `from_bytes`:
     persists `KeywordTableSpec`, `(table_epoch, state_epoch)`, runtime-stash,
     and both segment client states when no queries are in flight (matching
     `MultiLevelClient::write_hints`, which returns `QueryInFlight` while any
     query is reserved). **Restart and hint-reuse rule:** a checkpoint must
     never cause a previously sent hint to be sent a second time. Following
     the write-ahead log mechanism in
     [docs/online_protocol.md, "Client State And Restart"](online_protocol.md#client-state-and-restart),
     callers persist checkpoints at idle boundaries, durably append each sent
     segment hint identifier to a write-ahead log before transmitting the
     query (`PreparedQuery::hint_index` for `protocol::Client`, and the
     immutable `(level, hint_id)` for `MultiLevelClient`, which uniquely
     identifies both initial primary hints `0..num_primary` and promoted
     backup hints `num_primary..total_hints`), and on restore retire every
     logged hint before issuing any new query (`Client::retire_hints` for
     `protocol::Client`, and `MultiLevelClient::retire_hints`, which retires
     any active primary slot holding `hint_id` and permanently marks/advances
     past any logged backup `hint_id >= num_primary` so neither original
     primary hints nor promoted backups sent after the checkpoint can ever be
     reused). (Standalone
     `protocol::Client::write_state` also supports persisting in-flight
     queries, but the generic `KeywordClient` checkpoint contract relies on
     the idle-checkpoint + WAL-retirement path so it works identically across
     both backends.)
3. **Explicit `CachePolicy`:**
   - `Off` (default): always issues 2 PIR queries per input key (no local
     cache bypass, no batch deduplication).
   - `LocalHit`: returns cached/stash hits locally and deduplicates repeated
     keys in batches without spending backup hints `Q` (useful for local RPC
     proxies or repeated reads in one call; reveals hit/dedup counts on the
     wire).
   - `ConstantTraffic`: returns cached/stash/duplicate hits locally while still
     issuing 2 decoy queries per such input key to keep wire traffic uniform.
4. **Thin transport wrapper (`KeywordTransport` + `KeywordSession`):** optional
   sync convenience layer providing `get(kind, key)` and `get_batch(&keys)`
   with automatic `abort_lookup` / `abort_batch` on transport or completion
   error. Because `KeywordSession::get` and `get_batch` do not expose
   `pending.hint_ids()` before sending, `KeywordSession` does **not** support
   the WAL-before-send step required for checkpoint-and-WAL crash recovery;
   callers needing WAL recovery must drive `client_mut()` with
   `KeywordClient::prepare_lookup` (or `prepare_batch`), durably log
   `pending.hint_ids()` before sending, and complete via `finish_lookup` /
   `abort_lookup` (or `finish_batch` / `abort_batch`).
5. **OHTTP compatibility:** `KeywordLookupQuery` carries only
   `(table_epoch, state_epoch, segments: [Q; 2])` and omits local `session_id`
   and `query_id` fields, so callers can route online lookups through an OHTTP
   relay (RFC 9458) without forwarding those identifiers; any cross-request
   unlinkability guarantees depend on the transport and deployment controls.

## Limits

- **Server integrity:** wrong answers from a malicious server are not detected
  by PIR ([integrity.md](integrity.md)). Clients should verify a committed
  snapshot digest during `HintInit`.
- **Public seed:** an adversary who can freely insert keys into a table with a
  known `table_seed` can craft > `2 * b` keys sharing `(a0, a1)` to force
  eviction failure. Mitigations are build-time seed rotation, bounded
  runtime-stash, and 16-byte tags.
- **Client-side `HintInit` download:** as in index PIR, every new client
  streams the full `database.bin` once locally ([trust_model.md](trust_model.md)).
  `HintInit` must never be delegated to the server, a cloud, or a TEE.

## Implementation Status (TASK-22)

1. Library table core (`plinko/src/keyword.rs`): `KeywordTableSpec`, keyed
   BLAKE3 hashing and modular partial-key eviction, `KvScfTable` (build with
   unplaceable key reporting and optional seed retry, live
   `insert`/`update`/`upsert`/`delete` returning segment deltas and stash ops,
   `database.bin` and per-segment writers), plus `bench_keyword` CLI benchmark.
2. Client and session layer (`plinko/src/keyword/client.rs`): `IndexPirClient`
   implementations for `MultiLevelClient` and `protocol::Client`,
   `KeywordClient<B>` (`prepare_lookup`/`finish_lookup`/`abort_lookup`,
   two-pass `prepare_batch`/`finish_batch`/`abort_batch`, `apply_epoch_update`,
   `CachePolicy`, `retire_hints`, `PLNKKEYW` checkpointing), and
   `KeywordSession` / `KeywordTransport`.
3. End-to-end tests (`plinko/tests/keyword_e2e_test.rs` and unit tests in
   `plinko/src/keyword.rs` + `plinko/src/keyword/client.rs`): single-file
   `database.bin` and 2-segment flows through `plinko_hints` +
   `protocol::Client`, `MultiLevelClient` policies/updates/stash/checkpoints/WAL
   recovery, and `wasm32-unknown-unknown --no-default-features` compilation.
4. Documentation sync: `docs/data_format.md`, `docs/trust_model.md`,
   `docs/README.md`, `README.md`, and `CHANGELOG.md`.
