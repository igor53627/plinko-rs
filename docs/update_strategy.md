# Update Strategy

Plinko lets a client keep its hints current when database records change,
without a new HintInit. This document states what the repository implements
and what it does not.

## Database Versions

The publisher of a database gives each state of the database a version number
(`u64`). The library does not assign the first number; the caller gives it
(`plinko_hints --database-version`, `Server::new`, `compute_hints`). A later
state has a higher number.

| Item      | Carries                                                   | Check by the client                              |
|-----------|-----------------------------------------------------------|--------------------------------------------------|
| Hint set  | `database_version`: the version of its parities           | None; it is the state of the client              |
| Hint file | The same, and the version and the hash that HintInit read | The file checksum                                |
| Answer    | `Response::database_version`                              | Equal to the version of the hints                |
| Delta     | `from_version` and `to_version`                           | `from_version` equal to the version of the hints |

The version is a label that the server and the publisher state. The client
cannot check that the content belongs to the label (see
[Not Implemented](#not-implemented)).

## Delta

A delta (`plinko::delta::Delta`) holds the records that differ between two
versions, each as `old value XOR new value`. The file format is in
[data_format.md](data_format.md#delta-file).

- `Server::apply_updates` writes new values, moves the database to the next
  version (`database_version() + 1`), and returns the delta. A call with no
  update makes a new version with an empty delta.
- `Delta::new` takes the changes in any order. It combines the changes of one
  record and removes a change of zero bytes. `Delta::write_to` and
  `Delta::read_from` are the writer and the reader.
- `Client::apply_delta` applies a delta and moves the client to `to_version`.
  For a changed record it computes the iPRF inverse with the key of the block,
  which gives the candidate hints (`2 * lambda` on average). For each
  candidate:
  - A regular hint, or a backup hint that is promoted: XOR the change into the
    parity if the block subset of the hint contains the block.
  - A backup hint that is not used yet: XOR the change into `parity_in` if its
    block subset contains the block, and into `parity_out` if not.

  It also XORs the change into the cached answer of the record and into the
  parity of the hint that holds the record as its extra index.

### Order

`Client::apply_delta` accepts a delta only if its `from_version` is the
version of the hints. In each other case it returns `DatabaseVersionMismatch`
and changes no hint:

- A delta that the client applied before (it starts below the version of the
  client).
- A delta that comes before its predecessor (it starts above the version of
  the client).

A delta can span more than one version (`to_version` above `from_version + 1`).
The client then needs no delta in between. `Server::apply_updates` makes
deltas of one version only.

A delta for another number of records or another record size gives
`DeltaMismatch`.

**A delta is not bound to a database.** A version number identifies a state of
one database. `Client::apply_delta` accepts a delta of another database that
has the same record size, number of records and version, and the hints are
then wrong with no error. The caller must give a client only the deltas of the
database of its hints. A binding (for example a hash of the database before
and after the delta) is a part of TASK-24.

### Answers

`Client::reconstruct_and_replenish` accepts an answer only if its version is
the version of the hints. If not, it returns `DatabaseVersionMismatch`, the
client does not change, and the query stays in flight
([online_protocol.md](online_protocol.md#database-version-of-an-answer)).
Before this check, an answer from another database state gave a wrong record
with no error.

### Restart

The hint file stores the version of its parities
([data_format.md](data_format.md#hint-file-hintsbin)). A client that applied
deltas and sent no query can write its hints (`Client::unused_hints`,
`HintSet::write_to`). After a restart it loads the file, reads
`HintSet::database_version`, and applies the deltas that start there.

**This works only before the first query.** The hint file has no field for the
hints that queries used, for the promoted backup hints, or for the cache.
`Client::unused_hints` returns `None` after the first query. A client that
loads its hint file again after queries uses hints a second time, and the
server then gets the queried indices
([online_protocol.md](online_protocol.md#query-in-flight-lost-answer-retry)).
The code cannot detect this. A stored client state for the online phase is not
implemented.

Tests: `deltas_apply_in_version_order_and_one_time_only` and
`an_answer_of_a_later_version_is_accepted_after_the_delta` in
`protocol_e2e_test.rs`; `a_restarted_client_applies_the_deltas_after_its_hint_file`
in `hint_file_e2e_test.rs`, which also sends each delta through a file.

## Cost

Measured: 0.52 to 0.54 ms of client CPU for each changed record (one thread,
Apple M3 Ultra, hardware AES; 16,777,216 records, `w = 4096`; record sizes of
32, 40 and 256 bytes). The cost is one
iPRF inverse; it is 0.52 ms at the mainnet hint count
([trust_model.md](trust_model.md#cost-for-the-client)). A build without
hardware AES is about 20 times slower.

The cost does not depend on the database size beyond the iPRF domain. It is
not small: 10,000 changed records cost about 5 CPU-seconds for each client.

A delta file has `80 + count * (8 + r)` bytes for `count` changed records of
`r` bytes: 480,080 bytes for 10,000 records of 40 bytes (computed).

## A Database That Grows

A delta changes records at fixed indices. It keeps the number of records, the
record size, `w` and `c`.

The hints depend on the geometry (`w`, `c`) and on the record bytes. A record
in the padding of the last block counts as a zero record: a database with
fewer records and the same blocks has the same hints as the database with zero
records in their place (test `padding_records_count_as_zero_records`).

| Change of the database                                             | New HintInit                                                                                    |
|--------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|
| New value of a record                                              | No: a delta                                                                                     |
| A new record at an index below `c * w`, with `w` and `c` unchanged | No, in principle: it is a change of a zero record. The library does not implement it, see below |
| A new record at an index of `c * w` or above                       | Yes: `c` changes                                                                                |
| Another `w`                                                        | Yes: the block and the offset of each record change                                             |
| Another record size                                                | Yes                                                                                             |
| An insert that moves later records, or a delete that does so       | Yes, or one delta change for each moved record                                                  |

`w` and `c` follow from the number of records unless the caller sets `w`: the
default `w` is the smallest power of two that is at least `sqrt(n)`, so it
changes when `n` passes a power of 4, and `c` is `ceil(n / w)` made even.

**What the library implements:** a delta has the `num_entries` of the client,
and an index of `num_entries` or above is not valid. So the number of records
is fixed for a hint set. A publisher that wants to add records without a new
HintInit of the clients reserves them: it publishes the database with zero
records up to the capacity that it selects, and a new record is then a change
of a reserved record. The reserved records are part of the download and of
the HintInit work. A delta that raises `num_entries` inside the padding is not
implemented.

With the keyword table of [keyword_lookup.md](keyword_lookup.md) (not
implemented) an insert is a delta: the table has a fixed capacity, and an
insert writes one slot and moves some records to other slots. A rebuild of the
table with a new seed moves all records and needs a new HintInit.

## Not Implemented

- **Producer and transport.** The library has the delta format, a writer and a
  reader. No program writes delta files for a database on disk, and no service
  publishes them.
- **Client state after a query.** See [Restart](#restart).
- **Appends inside the padding.** See [A Database That Grows](#a-database-that-grows).
- **Authentication.** The checksum of a delta file detects damage only. A
  client cannot check that a delta holds the changes of the publisher, that a
  version number belongs to a content, or that an answer is correct (TASK-24).
  The hint file stores the BLAKE3 hash of the database bytes that HintInit
  read; a caller can compare it with a value from the publisher. The hash is
  not updated by a delta.
- **Amortized cost.** The client pays one iPRF inverse for each changed record
  of each delta, also for a record that changes again in the next delta.
