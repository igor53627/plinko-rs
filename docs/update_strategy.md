# Update Strategy

Plinko lets a client keep its hints current when database records change,
without a new HintInit. This document states what the repository implements
and what it does not.

## Implemented

- `Server::apply_updates` (`plinko/src/protocol.rs`) writes new values and
  returns one `EntryUpdate { index, delta }` for each changed record, where
  `delta = old value XOR new value`.
- `Client::apply_updates` applies the deltas to the hints. For a changed
  record it computes the iPRF inverse with the key of the block, which gives
  the candidate hints (`2 * lambda` on average). For each candidate:
  - A regular hint, or a backup hint that is promoted: XOR the delta into the
    parity if the block subset of the hint contains the block.
  - A backup hint that is not used yet: XOR the delta into `parity_in` if its
    block subset contains the block, and into `parity_out` if not.

  It also XORs the delta into the cached answer of the record and into the
  parity of the hint that holds the record as its extra index.

Both are in-memory functions. Tests: `protocol_e2e_test.rs` and
`hint_file_e2e_test.rs`.

## Cost

Measured: 0.51 to 0.53 ms of client CPU for each changed record (one thread,
Apple M3 Ultra, hardware AES; 16,777,216 records, `w = 4096`). The cost is one
iPRF inverse; it is 0.52 ms at the mainnet hint count
([trust_model.md](trust_model.md#cost-for-the-client)). A build without
hardware AES is about 20 times slower.

The cost does not depend on the database size beyond the iPRF domain. It is
not small: 10,000 changed records cost about 5 CPU-seconds for each client.

## Not Implemented

- **Delta feed.** No code writes or reads a delta file, and no service
  publishes changes. `plinko/docs/delta-format.md` is an old proposal that
  differs from the code.
- **Inserts and deletes.** An update changes a record at a fixed index. In the
  v3 layout (accounts first, then storage, in key order) a new record moves
  all later records, and a larger `w` changes the block of each record. Both
  need a new HintInit. The rule for a database that grows is a part of
  TASK-23.
- **Database version.** An answer and a delta carry no version, so a client
  cannot tell which database state they belong to
  ([online_protocol.md](online_protocol.md); TASK-23).
- **Authentication.** A client cannot check a delta, as it cannot check an
  answer (TASK-24).
