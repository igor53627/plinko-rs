# Plinko Protocol Overview

Plinko ([ePrint 2024/318](https://eprint.iacr.org/2024/318); Hoover, Patel,
Persiano, Yeo) is a single-server PIR scheme with client preprocessing. The
client reads the full database once (HintInit) and keeps hints. After that a
query does not reveal the index to the server, and a change of a database
record costs the client a hint update in place of a new HintInit.

## Scope

The repository targets a PIR library for a database that is an array of
fixed-size records: HintInit by the client (`plinko_hints`), the online
protocol (`plinko::protocol`) and hint updates.

**The full Ethereum state is not a target.** This is a decision of the project
owner of 2026-11-17 (`backlog/decisions/decision-1`). The reason is the cost
that each client pays. For the mainnet v3 snapshot (1,834,095,877 records):

| Item                      | Mainnet v3                                                     | Source                                                              |
|---------------------------|----------------------------------------------------------------|---------------------------------------------------------------------|
| Download for HintInit     | 73.4 GB                                                        | [data_format.md](data_format.md)                                    |
| HintInit                  | About 300 core-hours                                           | Extrapolated, [hint_generation.md](hint_generation.md#mainnet-cost) |
| Query, client CPU         | About 0.4 s                                                    | Computed from the measured cost per iPRF call                       |
| HintInit again            | When an insert moves the record indices, and after `Q` queries | [online_protocol.md](online_protocol.md)                            |
| Lookup by address or slot | Mapping files of 91 GB, or a message that reveals the key      | [keyword_lookup.md](keyword_lookup.md)                              |
| Check of an answer        | None                                                           | [trust_model.md](trust_model.md#not-covered)                        |

The Reth extractor (`src/main.rs`), the v3 record format and the mainnet
numbers stay in the repository as a source of large test data and as a scale
reference.

The record size is a parameter: 1 to 1,048,576 bytes
(`plinko_hints --record-size`, `ProtocolParams::record_size`). A hint parity
is the XOR of whole records, so a query returns the whole record. The default
is 40 bytes, the record of the v3 `database.bin`.

## Cost

The costs below replace the "O(1)" statements of earlier versions of this
document. One iPRF call runs a keyed permutation with about 800 Swap-or-Not
rounds for each stage at lambda = 128 (`sr_t_rounds_with_params` in
`plinko/src/iprf.rs`), so "constant" was correct as an order and not useful as
a number.

| Step                     | Work                                                                               |
|--------------------------|------------------------------------------------------------------------------------|
| HintInit (client)        | One iPRF inverse for each database record: `2 * lambda` candidate hints on average |
| Find a hint for an index | One iPRF inverse                                                                   |
| Query (client)           | `c/2` iPRF forward calls, one for each block of the hint                           |
| Query message            | `c/8` bytes of block mask and `c/2` offsets of 4 bytes                             |
| Answer (server)          | XOR of `c` records into two parities                                               |
| Answer message           | Two parities: `2r` bytes                                                           |
| Update (client)          | One iPRF inverse for each changed record                                           |
| Query limit              | `Q` queries, where `Q` is the number of backup hints; then a new HintInit          |

`w` is the number of records in a block, `c` is the number of blocks and `r`
is the record size in bytes. The
default `w` is the smallest power of two that is at least `sqrt(n)`
(`--entries-per-block` sets another value). The paper gives a trade-off
between client storage and query time; the code has no other tuning for it.

### Measured, 16,777,216 Records

Three full runs from `plinko_hints` to queries and updates, 2026-11-17, one
for each record size: Apple M3 Ultra, hardware AES
(`RUSTFLAGS="--cfg aes_armv8"`), `w = 4096`, `c = 4096`, `lambda = 128`,
524,288 regular hints and 524,288 backup hints. The database of each run is a
file of random bytes. The client and the server ran in one process with the
database in memory; there is no network in these numbers.

| Step                                        | 32-byte records | 40-byte records | 256-byte records |
|---------------------------------------------|-----------------|-----------------|------------------|
| Database file, bytes                        | 536,870,912     | 671,088,640     | 4,294,967,296    |
| HintInit, wall time on 24 threads           | 410 s           | 408 s           | 412 s            |
| HintInit, CPU time                          | 9,664 s         | 9,596 s         | 9,687 s          |
| HintInit, peak memory without the database  | 115 MB          | 141 MB          | 824 MB           |
| Hint file, bytes                            | 54,526,048      | 67,108,960      | 406,847,584      |
| Query, client CPU (`prepare_query`), median | 53.8 ms         | 53.2 ms         | 53.1 ms          |
| Query, client CPU, 99th percentile          | 59.9 ms         | 58.8 ms         | 59.3 ms          |
| First query of a client                     | 281 ms          | 281 ms          | 281 ms           |
| Query message, bytes                        | 8,704           | 8,704           | 8,704            |
| Answer message, bytes                       | 64              | 80              | 512              |
| Answer, server CPU                          | 0.11 ms         | 0.13 ms         | 0.38 ms          |
| Recon, client CPU                           | 0.001 ms        | 0.001 ms        | 0.001 ms         |
| Update, client CPU for each changed record  | 0.54 ms         | 0.53 ms         | 0.52 ms          |

In each run all 1,000 queries and 200 queries after 2,000 updates returned
the record of the database. The first query is slower because the PMNS level
cache of the client is empty at that time. The peak memory is the "peak
memory footprint" of `/usr/bin/time -l`; the database is a mapped file, and
its pages are in addition (814 MB of resident memory in the 40-byte run).

The query and update timings come from a scratch program that calls `Client`
and `Server` of `plinko::protocol`; it is not in the repository. The server
time is for a database in memory, where a read of a record is fast. A build
without hardware AES is about 20 times slower.

The run with 32-byte records gives the numbers of the code before the record
size was a parameter (412 s and 9,682 s CPU for HintInit, measured then with
a 40-byte database and parities over 32 bytes).

### Scaling

- HintInit CPU is linear in the number of records: 0.57 to 0.58 ms for each
  record here, 0.59 ms at the mainnet hint count.
- The record size does not change the CPU time of HintInit, of a query or of
  an update in the measured range: the iPRF calls take nearly all of it. It
  changes the download for HintInit, the hint file, the memory of HintInit,
  the answer message and the XOR work of the server.
- The client CPU of a query grows with `c`: 53.2 ms for 2,048 forward calls is
  about 26 us for each of the `c/2` blocks of the hint. This number includes
  the one inverse of the query.
- The query message has `c/8 + 2c` bytes: 59,475 bytes at `c = 27,988`.
- The hint file has `144 + lambda * w * (8 + 3r)` bytes with the default
  number of backup hints (`lambda * w`): a regular hint takes `4 + r` bytes
  and a backup hint takes `4 + 2r` bytes.
- HintInit holds the hints two times until the streaming phase ends (the
  accumulators of the worker threads and the hint set): about
  `2 * lambda * w * 3r` bytes.

## Phases

```text
database.bin (array of fixed-size records; the server publishes it)
   -> HintInit by the client (plinko_hints --seed-file --output) -> hint file
   -> Query / Answer / Recon (plinko::protocol)
   -> UpdateHint (Client::apply_delta, one delta for each new database version)
```

The online steps are a library model in one process. The repository has no
network transport. Open work for a general library, each with a backlog task:

- Lookup by key (TASK-22, proposal in [keyword_lookup.md](keyword_lookup.md)).
- A producer and a transport for deltas
  ([update_strategy.md](update_strategy.md#not-implemented)). The database
  version, the delta format and the client state file for a restart exist.

The library gives no integrity of an answer or of a delta. This is a decision,
not open work ([integrity.md](integrity.md)).

## References

- Paper: `docs/2024-318.pdf`
- Trust model: [trust_model.md](trust_model.md)
- Online protocol and its deviations from the paper: [online_protocol.md](online_protocol.md)
- Formal models (not a proof about the Rust code): `plinko/formal/`; `docs/Plinko.v` is a sketch
