# Keyword Lookup: Design Proposal

Status: proposal for TASK-22. The layout decision is open; no code implements
this document yet.

The document was written for the Ethereum v3 data: its keys are addresses and
(address, slot) pairs, and its numbers are for the mainnet snapshot. The full
Ethereum state is no longer a target
([protocol_overview.md](protocol_overview.md#scope)). TASK-22 now asks for the
same lookup for a general key-value database, with the key size, the value
size and the load as parameters; this document is the start point and is not
general yet.

## Problem

Plinko retrieves the record at an integer index. A client has a key: an address,
or an (address, slot) pair. Today the step from the key to the index is in
`account-mapping.bin` and `storage-mapping.bin` (91.46 GB at mainnet, more than
`database.bin`), and no code reads them. A client that asks a server for the
index reveals its key.

The paper names the solution (Section 2, "Additions and Deletions"): "A standard
cuckoo-hashing technique (see [ZPZS24, ALP+ 21]) can reduce keyword-PIR to
standard PIR. In this case, add/delete/update translates to updating a couple
array entries."

Sources that were read for this proposal: the Plinko paper,
`keewoolee/rms24-plinko-spec` (`pir/keyword_pir`, 2 hash functions, 1 slot per
bucket, 3 buckets per item, a stash), and `~/pse/rms24-rs`
(`src/keyword_pir/mod.rs`, a bucketed table with a 72-byte collision file).

## Design

`database.bin` becomes a cuckoo hash table. The client computes the candidate
positions of a key from the key and a public seed. There are no mapping files.

```text
database.bin = bucket[0] | bucket[1] | ... | bucket[num_buckets - 1]
bucket       = slot[0] | ... | slot[b - 1]        (one PIR entry)
slot         = tag (8 bytes) | payload (32 bytes)
```

- **PIR entry = bucket.** One PIR query returns all `b` slots of one bucket.
- **Positions.** A key has `d` candidate buckets:
  `h = keccak256(table_seed || kind || key)`,
  `position_i = u64_le(h[8i .. 8i + 8]) mod num_buckets`.
- **Tag.** `tag = h[24 .. 32]`. The client finds its record by the tag.
- **Kind.** One byte that separates key spaces: `0x01` account (key = 20-byte
  address), `0x02` storage (key = address || slot key, 52 bytes). Other data
  sets use other kind values; the table code does not interpret the key.
- **Payload.** Account: balance (16) | nonce (4) | CodeID (4) | zero (8).
  Storage: value (32). These are the fields of the v3 record without its TAG.
- **Seed.** `table_seed` (32 bytes) and `num_buckets`, `b`, `d` are in
  `metadata.json`. They are public.
- **Empty slot.** All bytes zero. The builder selects another seed if a key has
  the tag 0.
- **No stash at build time.** If a key cannot be placed, the builder selects
  another seed and starts again.

### Lookup

1. The client computes the `d` positions and the tag of the key.
2. The client sends `d` PIR queries, one per position. It always sends `d`
   queries, also when the first answer holds the record, and also when two
   positions are equal or an index is in its cache (the existing client sends
   a query for a decoy index in that case).
3. The client compares the tag with the tag of each slot of the candidate
   buckets. If two positions are equal, it reads that bucket once. One match
   gives the payload. No match means that the key is not in the
   table. More than one match is an error.

The server sees `d` PIR queries per lookup. By the privacy of the index PIR the
queries do not reveal the positions. The number of lookups stays visible.

### Why The Slot Layout Changes

The v3 record has its TAG at bytes 24..32 for an account and at bytes 32..40 for
a storage slot. In a table that mixes both kinds this is not safe: bytes 24..32
of a storage record are value bytes, which the owner of a contract selects. A
contract can thus write a value that looks like the tag of a victim account
into a slot of a candidate bucket of that account. The proposal puts the tag at
one position for all kinds and binds the kind into the tag.

## Layout Options

Mainnet v3 has 1,834,095,877 keys. All numbers in this table are computed from
the formulas of the code and from per-operation measurements (0.59 ms CPU per
PIR entry in HintInit and 28.5 us per `forward` call, both measured at
w = 65536 with hardware AES). No table was built at mainnet scale.

| Option                   | Today (index) | B: d=3, b=1 | C2: d=2, b=2 | C4: d=2, b=4 | C8: d=2, b=8 |
|--------------------------|---------------|-------------|--------------|--------------|--------------|
| Load at build            | 1.00          | 0.85        | 0.80         | 0.90         | 0.95         |
| PIR entries (buckets)    | 1.834e9       | 2.158e9     | 1.146e9      | 0.509e9      | 0.241e9      |
| PIR entry size           | 40 B          | 40 B        | 80 B         | 160 B        | 320 B        |
| `database.bin`           | 73.4 GB       | 86.3 GB     | 91.7 GB      | 81.5 GB      | 77.2 GB      |
| w                        | 65536         | 65536       | 65536        | 32768        | 16384        |
| c                        | 27988         | 32926       | 17492        | 15548        | 14730        |
| Hints (lambda = 128)     | 16,777,216    | 16,777,216  | 16,777,216   | 8,388,608    | 4,194,304    |
| Hint file                | 1.07 GB       | 1.07 GB     | 2.08 GB      | 2.05 GB      | 2.03 GB      |
| HintInit                 | 301 core-h    | 354 core-h  | 188 core-h   | 83 core-h    | 40 core-h    |
| PIR queries per lookup   | 1             | 3           | 2            | 2            | 2            |
| Lookups per hint set     | 8,388,608     | 2,796,202   | 4,194,304    | 2,097,152    | 1,048,576    |
| Upload per lookup        | 59 KB         | 210 KB      | 74 KB        | 66 KB        | 63 KB        |
| Download per lookup      | 80 B          | 240 B       | 320 B        | 640 B        | 1280 B       |
| Client CPU per lookup    | 0.40 s        | 1.41 s      | 0.50 s       | 0.44 s       | 0.42 s       |

Notes:

- "Today" has no keyword lookup; it is the base for the comparison. Its hint
  file is given for a parity of 40 bytes, which is the size since TASK-31.
- The hint file grows with the PIR entry size, because a parity has the size
  of one PIR entry.
- HintInit cost follows the number of PIR entries, because the iPRF inverse is
  the cost and it is paid once per PIR entry.
- "Lookups per hint set" is the number of backup hints divided by `d`.
- The option of the reference spec (d = 2, b = 1) is not in the table. In the
  simulation below it does not build without a stash at a load of 0.45, and it
  needs more than 160 GB.

### Simulation Of The Table Build

Measured with a scratch program (67,108,864 slots, random-walk insertion, at
most 500 evictions per key). "Evictions" are counted for the last 1 percent of
the inserted keys, where the table is at its final load. The table gives the
run with the first seed. In the run with a second seed the mean values differ
by less than 2 percent, and the keys not placed are 487, 0, 24,905, 0, 67, 0,
1 and 0.

| d | b | Load | Keys not placed | Evictions: mean | p99 | max |
|---|---|------|-----------------|-----------------|-----|-----|
| 2 | 1 | 0.45 | 470             | 2.8             | 59  | 500 |
| 3 | 1 | 0.85 | 0               | 15.5            | 120 | 420 |
| 3 | 1 | 0.90 | 24,966          | 104             | 500 | 500 |
| 2 | 2 | 0.80 | 0               | 7.5             | 74  | 268 |
| 2 | 2 | 0.85 | 50              | 26.8            | 222 | 500 |
| 2 | 4 | 0.90 | 0               | 5.4             | 41  | 177 |
| 2 | 4 | 0.95 | 3               | 25.9            | 159 | 500 |
| 2 | 8 | 0.95 | 0               | 5.3             | 35  | 102 |

The mainnet table has about 30 times more slots; that size was not simulated.

## Recommendation

**C4: 2 hash functions, 4 slots per bucket, load 0.90 at build.**

- 2 PIR queries per lookup, where option B needs 3.
- HintInit drops from about 300 to about 83 core-hours, because there are 3.6
  times fewer PIR entries.
- An insert at the final load moves 5.4 records on average, where option B
  moves 15.5. Each moved record is one changed PIR entry for all clients.
- At a load of 0.95 the simulation does not place 3 keys (1 key with the
  second seed), so a load of 0.90 leaves room for growth before a rebuild or
  a stash is necessary.
- Cost: the hint file is 2.05 GB, where it is 1.07 GB with 40-byte entries.

C8 halves HintInit again at the same hint file size, but it also halves the
lookups per hint set, and a change of one record changes a 320-byte entry.

## Consequences For The Code

1. **The PIR entry size is a parameter** (done in TASK-31): `plinko::hints`,
   `plinko::protocol` and `plinko_hints --record-size` take the record size of
   the database, and the parity covers the full record. A bucket of this
   proposal is one record.
2. **A table builder.** The cuckoo build needs random access to the table.
   With d = 2 the builder keeps, for each slot, the index of the other
   candidate bucket of its key (4 bytes), so that it can move a record without
   its key: 8.2 GB in addition to the 81.5 GB table at mainnet (computed).
3. **The extractor** writes (kind, key, payload) records for the builder, not
   `database.bin` and the mapping files. `gen_synthetic` does the same, so
   that tests have keys.
4. **A keyword client** on top of `protocol::Client`: `lookup(kind, key)`
   sends `d` queries and checks the tags.
5. **`docs/data_format.md`** describes the new layout as v4. The mapping files
   and the v3 record order are removed.

## Limits

- **Tag collisions.** A tag has 64 bits. A false match needs another key with
  the same tag in a candidate bucket of the key. For keys that nobody selects
  this is negligible. An attacker who selects keys, for example slot keys of
  its own contract, needs about 2^91 hash evaluations for one given victim key
  and about 2^60 for some key of the table (estimate: 2^-64 for the tag, about
  2^-27 for a shared bucket, 2^31 keys). The result is a lookup error for that
  key, or a wrong value for a key that is not in the table. A 16-byte tag
  removes this at a cost of 8 bytes per slot; the proposal does not select it.
- **Wrong answers of the server** are not covered (TASK-24).
- **Inserts after the build** move records, and a key that cannot be placed
  needs a stash or a rebuild with a new seed. A new seed moves all records, so
  all clients must run HintInit again. An insert that moves records in a table
  of fixed capacity is a delta
  ([update_strategy.md](update_strategy.md#a-database-that-grows)).
- **The seed is public.** An attacker can search for keys that share both
  buckets. More than 8 such keys cannot be placed with b = 4. At build time the
  builder answers with a new seed; for inserts there is no answer yet.

## Implementation Steps

1. Library module `plinko::keyword`: hashing, slot and bucket codec, in-memory
   cuckoo builder, tag match. Unit tests with a build at a load of 0.90.
2. Entry size as a parameter in `hints`, `protocol` and `plinko_hints` (done
   in TASK-31; the end-to-end tests run with 1, 13, 40, 45 and 256 bytes).
3. Builder binary and the changes of `gen_synthetic` and the extractor.
4. Keyword client and an end-to-end test: build a table from synthetic
   accounts and slots, run `plinko_hints`, look up an address, an
   (address, slot) pair and a key that is not in the table.
5. Documents: `data_format.md` v4, `trust_model.md` ("Not Covered" list).
