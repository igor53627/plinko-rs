# Online Protocol

This document describes the online phase of the client and server model in
`plinko/src/protocol.rs`. The model follows Fig. 7 of the Plinko paper (`Query`,
`Answer`, `Recon`, `GetHint`, `UpdateHint`). Each section states where the model
differs from the paper and why.

The parties and the secrets are in [trust_model.md](trust_model.md). The hint
set is in [hint_generation.md](hint_generation.md).

## Query

`Client::prepare_query(i)` does these steps.

1. **Index.** If `i` is not in the cache, the query goes to `i`. If `i` is in
   the cache, the query goes to a decoy: a uniform index among the indices that
   are not in the cache and not in flight. Fig. 7 states this as
   `While Q[i] != ⊥: i <-$ [n]`. The decoy query uses a hint and a backup hint,
   and its answer goes to the cache, the same as a real query.
2. **Hint.** The client computes the candidates `iF.F^-1(K[alpha], beta)`,
   puts them in a random order (Fig. 7, `GetHint`: "in random order"), and
   takes the first hint that is present, is not reserved by a query in flight,
   and contains the index.
3. **Message.** The client removes the block of the index from the subset of
   the hint, flips one coin to select which of the two subsets is the real one,
   and builds the message (see [Query Format](#query-format)).

Errors:

- `NoHintForIndex`: no hint contains the index. If the iPRF is a random
  function, the probability is about `e^(-lambda/2)` for each index (`lambda * w`
  hints, each contains the index with probability about `1/(2w)`). For a decoy,
  the caller can call `prepare_query` again; the client then draws a new decoy.
- `NoDecoyIndexAvailable`: every index is in the cache or in flight.
- `QueryInFlight`, `NoBackupHintsLeft`: see the sections below.

On an error, the client reserves no hint.

## Query Format

The paper and the model use different encodings.

| Source               | Message                                                       |
|----------------------|---------------------------------------------------------------|
| Plinko Fig. 7        | subset `P'` and one offset for each of the `c` blocks         |
| RMS24, Section 3.2   | bit vector `b` (`c` bits) and offset vector `r` (`c` offsets) |
| This model (`Query`) | `mask` (`c` bits) and `c/2` offsets                           |

In the model, bit `k` of `mask` puts block `k` in subset 0 or subset 1. The
`i`-th block of each subset reads `offsets[i]`. The offsets are those of the
real subset in block order; the dummy subset has no offsets of its own.

**This form is not in the RMS24 text and not in the Plinko paper.** RMS24
sends one offset for each block; its "half" refers to the step from two index
lists to `b` and `r`. The form with `c/2` offsets comes from the Python
reference `keewoolee/rms24-plinko-spec`.

Sizes at mainnet (`c = 27,988`, offsets as `u32`), computed:

| Form                 | Mask        | Offsets       | Total         |
|----------------------|-------------|---------------|---------------|
| One offset per block | 3,499 bytes | 111,952 bytes | 115,451 bytes |
| `c/2` shared offsets | 3,499 bytes | 55,976 bytes  | 59,475 bytes  |

### Argument For The Shared Offsets

This argument is ours. It is not a proof from a paper and it is not machine
checked.

The privacy proof of RMS24 (Section 3.5) has two parts:

1. The offset vector `r` consists of pseudorandom values that are independent
   of the query index.
2. The distribution of the bit vector `b` does not depend on the query index.

The shared form keeps both parts:

- `mask` is `b` with no change, so part 2 applies as written.
- `offsets` holds the offsets of the real subset. The block of the query index
  is not in the real subset, so its offset is not in the message. The other
  offsets of the hint are iPRF outputs for other blocks (or the offset of the
  extra index); part 1 of RMS24 treats the same values as independent of the
  query index.
- The coin is uniform and independent. The two subsets have the same size and
  read the same offset list, so the message does not identify the real subset.

Thus the message has the same distribution for each query index, on the same
assumptions as RMS24. The difference from RMS24 is that the dummy offsets are a
copy of the real offsets and not new random values. The argument above does not
use new random values for the dummy subset.

The conservative alternative is one offset per block, as in the two papers. It
makes the upload two times larger (see the table).

## Query In Flight, Lost Answer, Retry

A query is in flight from `prepare_query` until `reconstruct_and_replenish` or
`abandon_query`. During this time:

- The hint stays in the client and is reserved. No other query uses it.
- One backup hint is reserved for the promotion.
- `prepare_query` for the same index returns `QueryInFlight`. Two answers for
  one index would promote two backup hints with the same extra index, and the
  cache holds one promoted hint for each index.
- A decoy is not an index in flight.
- `apply_delta` changes the parity of a reserved hint the same as that of
  each other hint.

**Rule for a retry: send the same `Query` bytes again.** The caller keeps the
`PreparedQuery` and sends `prepared.query` again.

Why this is private: the server has the message already, and a copy has no new
data about the index. The server learns that a retry occurred. That depends on
the network, not on the index.

**Rule for a hint: a hint that was sent is not used for a second index.** The
subset of a hint for an index in block `alpha` contains the other blocks of the
hint with their offsets. If the same hint served a second index in block
`gamma`, the second message would contain block `alpha` with the offset of the
first index. The server would get the first index. The reservation and the
removal of the hint at `Recon` enforce this rule.

`abandon_query` stops a query with no answer. The client removes the hint and
does not promote a backup hint. **This is a deviation.** The proof of RMS24 for
later queries needs a new hint that contains the same index in place of the
used one ("the replenished hint must follow the same distribution as the one
just consumed"). After `abandon_query` the hint table has one hint less that
contains the index, with no replacement. This document does not analyze the
size of that bias. Use a retry; use `abandon_query` only when the server does
not answer at all.

## Client State And Restart

The rule for a hint must also hold when the client process stops and starts
again. A hint file cannot give this: it holds the hints of a client that has
sent no query. A client that loads its hint file again after queries sends
hints a second time.

`Client::write_state` writes the state of a client that sent queries, and
`Client::read_state` builds a client from it
([data_format.md](data_format.md#client-state-file)). The state holds the
hints with their database version, the regular hints that were sent, the
promoted backup hints, the cache and the queries in flight.

`read_state` returns the queries in flight. Each one has the same `Query`
bytes as before the restart: a message is a function of its hint and of the
stored order of the two subsets. The caller sends it again (the rule for a
retry) or calls `abandon_query`.

**Rules for the caller.** The code cannot enforce them.

1. **Record a query before it leaves the client**, and make the record
   durable first. A query that the server saw and the stored state does not
   hold has the same effect as a hint file that is loaded again. There are
   two ways:
   - Write the state after `prepare_query`. The state has the size of the
     hint file (1.07 GB with the mainnet parameters, computed), so this is a
     large write for each query.
   - Write the state at times. Between two writes, append
     `PreparedQuery::hint_index` of each query to a log. After a restart,
     load the state and pass the log to `Client::retire_hints`. The log of a
     state holds each query that the client prepared after the `write_state`
     call of that state, also a query that left the client before the state
     was durable. Keep the old state and its log until the new state and its
     log are durable.
2. **Without the log, load only the newest state.** The file has no field
   that shows its age, and the client has no second source to compare it
   with. A copy from a backup is an old state. `read_state` cannot detect it.
3. **Replace a state only with a complete one**, for example with a write to
   a new file and a rename.
4. **Use new random bytes as the query seed of each start.** The state does
   not hold the query randomness. With an old seed the client repeats random
   choices of queries that the server saw.
5. **Keep the file secret.** It holds the hints, the indices of the answered
   queries with their records, and the queried and the requested index of
   each query in flight. It does not hold the master seed.

`retire_hints` removes each hint of the log with no promotion, as
`abandon_query` does, so the deviation of the section above applies to each
of them. The
answers of those queries are not in the state: a later query for one of those
indices is a query for that index with another hint, not a decoy query. A
backup hint of the log is not promoted later. If it was the last backup hint
that a query in flight counted on, `reconstruct_and_replenish` returns
`NoBackupHintsLeft` for that query and changes nothing; call `abandon_query`.

Tests: `a_restarted_client_does_not_send_a_hint_a_second_time` in
`protocol_e2e_test.rs` starts one client from the newest state, from an old
state with no log (the control: hints go out a second time) and from an old
state with the log. `a_query_with_no_backup_hint_after_retire_hints_stays_in_flight`
is in the same file. The unit tests of `plinko/src/protocol/state.rs` compare
a client from a state with the client that wrote it, check the layout, and
flip each bit of a state file.

## Database Version Of An Answer

The parity of a hint belongs to one database version
([update_strategy.md](update_strategy.md#database-versions)). An answer
carries the version of the database that it comes from
(`Response::database_version`). `reconstruct_and_replenish` compares the two
before it changes the client:

- **Equal:** the client computes the record.
- **Not equal:** `DatabaseVersionMismatch`. The client does not change, and
  the query stays in flight. If the answer is newer than the hints, the caller
  applies the deltas up to that version and calls the function again with the
  same answer; no new message goes to the server. If the answer is older, the
  caller sends the same `Query` bytes again.

`reconstruct_and_replenish` takes the prepared query and the answer by
reference, so both stay with the caller after an error.

The version is what the server states. A server that states a wrong version
gives the client a wrong record, as a server that sends a wrong parity does
([integrity.md](integrity.md)). A version mismatch reveals to the server at most that the client
sends a query again; the message is the same.

## Query Limit

`q` is the number of backup hints (`num_backup_hints`, default `lambda * w`).
Each answered query promotes one backup hint, and each query in flight reserves
one.

When no backup hint is free, `prepare_query` returns `NoBackupHintsLeft` before
it builds a message. This also applies to an index in the cache, because the
decoy query needs a hint and a backup hint. No hint is used and nothing is sent.

After the limit, **the client must run HintInit again with a new master seed.**
The hints are a function of the master seed. The same seed gives the same
subsets and offsets, and the server saw those of each used hint. A second use
breaks the rule for a hint above. The code cannot check that a seed is new.

The amortized refresh of the paper (Section 5.2: the client streams `n/q`
entries after each query and builds the next hints in the cache) is out of
scope. At mainnet the default limit is `128 * 65,536 = 8,388,608` queries.

## iPRF Domain

`Iprf::forward(x)` panics if `x` is not in `[0, domain)`. Before, it returned
bin 0, which is a valid output. The client calls `forward` only with a hint
index below `num_total_hints`, which is the domain.

`IprfTee::forward` (feature `tee`, outside the trust model) has the same
precondition, with a check in debug builds only.

## Malformed Answer

`reconstruct_and_replenish` returns `RecordSizeMismatch` for a parity that
does not have the record size. The client does not change and the query stays
in flight, so the caller can send the query again. Before the database
version check, this error ended the query as `abandon_query` does.

## Server Checks

`Server::answer` returns an error, and does not panic, for a mask of the wrong
length, a mask that does not put `c/2` blocks in each subset, a wrong number of
offsets, and an offset outside a block.

## Not Covered

- **Wrong answers.** The client cannot check an answer, and it cannot check
  the database version that an answer states. The library gives no
  integrity ([integrity.md](integrity.md)).
- **An old client state.** A client that loads a state file that is older
  than its last query, with no log of the later hints, sends hints a second
  time, which breaks the rule for a hint above. The same applies to a hint
  file that is loaded again after queries. The code cannot detect this; see
  [Client State And Restart](#client-state-and-restart).
- **Bias after `abandon_query` and `retire_hints`.** See above.
- **Formal model.** No machine-checked statement covers the online phase or
  the shared offsets. The Rocq models have no Query, Answer and Recon
  ([verification.md](verification.md)).
- **Reference check.** `scripts/differential_protocol_transcript.sh` compares
  the answers and the number of queries left with the Python reference. It does
  not compare the messages. The reference selects the smallest free index as
  the decoy; this model selects a random one.
