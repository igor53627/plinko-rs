# Keyword Lookup: Design Specification

Status: generalized design specification for TASK-22 (no code implements this
document yet).

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
   A lookup inspects at most `d * b` occupied slots (8 slots for `d = 2, b = 4`),
   each carrying an independent uniform non-zero tag. By union bound:
   `P_false_match <= (d * b) / (2^t - 1)`
   - `tag_size = 8` (`t = 64`, `d = 2, b = 4`): `<= 8 / (2^64 - 1) ~= 4.3e-19`
     (`~2^-61` per lookup).
   - `tag_size = 16` (`t = 128`, `d = 2, b = 4`): `<= 8 / (2^128 - 1) ~= 2.4e-38`
     (`~2^-125` per lookup).

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
memory when RAM is tighter than upload).

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

1. **`IndexPirClient` trait:** minimal adapter (~8 methods: `num_entries`,
   `record_size`, `epoch`, `remaining_queries`, `prepare_index_query`,
   `finish_index_query`, `abort_index_query`, `apply_index_deltas`),
   implemented out of the box for both `MultiLevelClient` and
   `protocol::Client`.
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
          (`PreparedQuery::queried_index()` on `protocol::Client`, added in
          `TASK-22`) so `prepare_batch` tracks every backend-reserved bucket
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
     `protocol::Client`, and a matching `MultiLevelClient::retire_hints` to be
     added in `TASK-22` that retires any active primary slot holding `hint_id`
     and permanently marks/advances past any logged backup
     `hint_id >= num_primary` so neither original primary hints nor promoted
     backups sent after the checkpoint can ever be reused). (Standalone
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
   with automatic `abort_lookup` on transport error.
5. **OHTTP compatibility:** `KeywordLookupQuery` carries only
   `(table_epoch, state_epoch, segments: [Q; 2])` and never puts `session_id`
   or `query_id` on the wire, so online lookups can be routed through an OHTTP
   relay (RFC 9458) to hide client IP while PIR protects the queried key even
   if relay and server collude.

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

## Implementation Steps (TASK-22)

1. Library module `plinko::keyword`: `KeywordTableSpec`, BLAKE3 keyed hashing
   and modular partial-key eviction, `KvScfBuilder` / `KvScfTable` (build,
   live `insert`/`update`/`delete` returning segment deltas, `database.bin`
   writer, unplaceable key reporting).
2. `IndexPirClient` implementations for `MultiLevelClient` and
   `protocol::Client`, plus `KeywordClient<B>` (`prepare_lookup`,
   `finish_lookup`, `abort_lookup`, `prepare_batch`/`finish_batch`,
   `apply_epoch_update`, `CachePolicy`, checkpointing) and `KeywordSession`.
3. End-to-end tests: build synthetic key-value tables at arbitrary
   `(key_size, value_size, load)` -> verify through `plinko_hints` +
   `protocol::Client` AND through `MultiLevelClient` (including `insert`,
   `update`, `delete`, not-found lookups, `abort_lookup`, and `wasm32`
   compilation).
4. Update `docs/data_format.md` and `docs/trust_model.md` ("Not Covered" list)
   once `plinko::keyword` lands.
