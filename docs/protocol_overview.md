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

| Item                           | Mainnet v3                                                    | Source                                        |
|--------------------------------|---------------------------------------------------------------|-----------------------------------------------|
| Download for HintInit          | 73.4 GB                                                       | [data_format.md](data_format.md)              |
| HintInit                       | About 300 core-hours                                          | Extrapolated, [hint_generation.md](hint_generation.md#mainnet-cost) |
| Query, client CPU              | About 0.4 s                                                   | Computed from the measured cost per iPRF call |
| HintInit again                 | When an insert moves the record indices, and after `Q` queries | TASK-23, [online_protocol.md](online_protocol.md) |
| Lookup by address or slot      | Mapping files of 91 GB, or a message that reveals the key     | TASK-22, [keyword_lookup.md](keyword_lookup.md) |
| Check of an answer             | None                                                          | TASK-24                                       |

The Reth extractor (`src/main.rs`), the v3 record format and the mainnet
numbers stay in the repository as a source of large test data and as a scale
reference.

The record size is not a parameter today: `database.bin` has 40-byte records
and the hints cover the first 32 bytes of each record.

## Cost

The costs below replace the "O(1)" statements of earlier versions of this
document. One iPRF call runs a keyed permutation with about 800 Swap-or-Not
rounds for each stage at lambda = 128 (`sr_t_rounds_with_params` in
`plinko/src/iprf.rs`), so "constant" was correct as an order and not useful as
a number.

| Step                      | Work                                                                                 |
|---------------------------|--------------------------------------------------------------------------------------|
| HintInit (client)         | One iPRF inverse for each database record: `2 * lambda` candidate hints on average    |
| Find a hint for an index  | One iPRF inverse                                                                      |
| Query (client)            | `c/2` iPRF forward calls, one for each block of the hint                              |
| Query message             | `c/8` bytes of block mask and `c/2` offsets of 4 bytes                                |
| Answer (server)           | XOR of `c` records into two parities                                                  |
| Update (client)           | One iPRF inverse for each changed record                                              |
| Query limit               | `Q` queries, where `Q` is the number of backup hints; then a new HintInit             |

`w` is the number of records in a block and `c` is the number of blocks. The
default `w` is the smallest power of two that is at least `sqrt(n)`
(`--entries-per-block` sets another value). The paper gives a trade-off
between client storage and query time; the code has no other tuning for it.

### Measured, 16,777,216 Records

One full run from `gen_synthetic` to queries and updates, 2026-11-17: Apple M3
Ultra, hardware AES (`RUSTFLAGS="--cfg aes_armv8"`), `w = 4096`, `c = 4096`,
`lambda = 128`, 524,288 regular hints and 524,288 backup hints. The client and
the server ran in one process with the database in memory; there is no network
in these numbers.

| Step                                   | Measured                                             |
|----------------------------------------|------------------------------------------------------|
| `database.bin`                         | 671,088,640 bytes                                    |
| HintInit, wall time on 24 threads      | 412 s                                                |
| HintInit, CPU time                     | 9,682 s (0.58 ms for each record)                    |
| HintInit, peak resident memory         | 790 MB, with the mapped database pages               |
| Hint file                              | 54,526,048 bytes                                     |
| Query, client CPU (`prepare_query`)    | 56.0 ms median, 63.3 ms at the 99th percentile       |
| First query of a client                | 289 ms; the PMNS level cache is empty at that time   |
| Query message                          | 8,704 bytes                                          |
| Answer message                         | 64 bytes                                             |
| Answer, server CPU                     | 0.09 ms                                              |
| Recon, client CPU                      | 0.002 ms                                             |
| Update, client CPU                     | 0.51 to 0.53 ms for each changed record (two runs)   |

All 1,000 queries and 200 queries after 2,000 updates returned the record of
the database. The query and update timings come from a scratch program that
calls `Client` and `Server` of `plinko::protocol`; it is not in the
repository. The server time is for a database in memory, where a read of a
record is fast. A build without hardware AES is about 20 times slower.

### Scaling

- HintInit CPU is linear in the number of records: 0.58 ms for each record
  here, 0.59 ms at the mainnet hint count.
- The client CPU of a query grows with `c`: 56.0 ms for 2,048 forward calls is
  about 27 us for each of the `c/2` blocks of the hint. This number includes
  the one inverse of the query.
- The query message has `c/8 + 2c` bytes: 59,475 bytes at `c = 27,988`.
- The hint file has `lambda * w * 104` bytes plus 96 bytes, with the default
  number of backup hints (`lambda * w`).

## Phases

```text
database.bin (array of fixed-size records; the server publishes it)
   -> HintInit by the client (plinko_hints --seed-file --output) -> hint file
   -> Query / Answer / Recon (plinko::protocol)
   -> UpdateHint (Client::apply_updates)
```

The online steps are a library model in one process. The repository has no
network transport, no delta feed for updates ([update_strategy.md](update_strategy.md)),
no lookup by key and no check of an answer. See
[trust_model.md](trust_model.md#not-covered).

## References

- Paper: `docs/2024-318.pdf`
- Trust model: [trust_model.md](trust_model.md)
- Online protocol and its deviations from the paper: [online_protocol.md](online_protocol.md)
- Formal models (not a proof about the Rust code): `plinko/formal/`; `docs/Plinko.v` is a sketch
