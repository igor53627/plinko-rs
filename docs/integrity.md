# Integrity

This document states what a client can check and what it must trust: in
HintInit, in an answer and in a delta.

## Decision

**The library gives no integrity.** The protocol hides the queried index from
the server. It does not show the client that a record is correct. The client
must trust the server for each answer, and the server and the publisher for
each delta. An application that needs integrity puts an authenticator in each
record ([What An Application Can Do](#what-an-application-can-do)).

Reasons:

- The paper gives none. Its games compute the answer with the honest `Answer`
  algorithm. A check inside the protocol is a new construction, and the
  repository has no analysis for one.
- A record is opaque bytes for the library, and a parity covers all bytes of
  a record. An authenticator inside the record thus passes through HintInit,
  Query, Recon and UpdateHint with no change of the library.
- Each mechanism has a cost that depends on the application
  ([Mechanisms That Were Not Selected](#mechanisms-that-were-not-selected)).

## What The Client Checks

| Step     | The client checks                                                                                                                                                                     | The client must trust                                                                                                                            |
|----------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------|
| HintInit | It reads all records. `plinko_hints` stores the BLAKE3 hash of the database file in the hint file (`HintSet::init_database`); a caller can compare it with a value from the publisher | That the file is the database that the server answers from. With no value from the publisher, the hash only names the bytes that the client read |
| Answer   | The version of the answer is the version of the hints; each parity has the record size                                                                                                | The two parities, and the version that the answer states                                                                                         |
| Delta    | The format rules, the record size, the number of records, and that the delta starts from the version of the hints                                                                     | The changes, that the server applied the same changes, and that the delta belongs to the database of the hints                                   |
| Files    | The BLAKE3 checksum of a hint file, a state file and a delta file detects damage                                                                                                      | A checksum is not a signature: whoever changes a file can write a new checksum                                                                   |

The client is the only party that can read its hint file and its state file
([trust_model.md](trust_model.md#secrets)), so their checksums are sufficient
against damage. A delta comes from another party; its checksum does not show
who wrote it.

## Effect Of A Wrong Answer

The client computes `record = server parity XOR hint parity`. A server that
XORs a value into its parity gives the client the record XOR that value, with
no error.

- **The wrong record stays.** It goes into the cache, so a later query for the
  same index returns it again.
- **The error spreads.** `Recon` XORs the record into the parity of the backup
  hint that it promotes. A later query that uses this promoted hint gives a
  record with the same error, and that record goes into the next promoted
  hint.
- **The client cannot repair its hints.** It does not know which parity is
  wrong. After a wrong record is found by other means, run HintInit again
  with a new master seed
  ([online_protocol.md](online_protocol.md#query-limit) gives the reason for
  the new seed).
- **The messages of the client do not change.** A query is a function of the
  index, the hints that are present, the indices in the cache and in flight,
  and the query randomness. It does not depend on the bytes of an answer.

Test: `a_wrong_answer_gives_a_wrong_record_and_the_same_later_messages` in
`protocol_e2e_test.rs` changes one bit in the parities of one answer. It
compares each later message with that of a client that gets correct answers,
and it checks that a record is wrong exactly when its query uses a hint that
holds the error.

**The reaction of the application can leak.** If the server can see that the
application found a wrong record (the application stops, or it sends other
queries), the server gets data about the query. Example: the server changes
only one of the two parities. The record is wrong only if that parity is the
one of the real subset. The queried block is not in the real subset, so the
server then knows that the queried block is one of the `c/2` blocks of the
other subset. This is a conclusion from the message format
([online_protocol.md](online_protocol.md#query-format)); the repository has no
analysis of a server that repeats this. An application that checks records
must not show the result of the check to the server.

## Deltas

**The client must trust the deltas.**

- A delta holds `old value XOR new value` of each changed record. A wrong
  change gives a wrong parity in each hint that holds the record, and wrong
  records later, with no error.
- The hash in the hint file is the hash of the database that HintInit read. A
  delta does not change it. The client does not hold the records, so it
  cannot compute the hash of the next version from its hash and a delta.
  After the first delta the client has no hash of the database of its hints.
- A delta is bound to a version number, not to a database
  ([update_strategy.md](update_strategy.md#order)). The caller must give a
  client only the deltas of the database of its hints.
- The version of an answer and of a delta is a label. A server that states a
  wrong version gives the client a wrong record.

Not implemented: a hash of the database before and after the changes in the
header of a delta, with the current hash in the hint file. The client could
then detect a delta of another database, and a caller could compare the hash
of each version with a value from the publisher. The client could not check
the hash after the changes, so a wrong delta with correct hashes would pass.
The publisher would compute a hash of the full database for each version.

## Mechanisms That Were Not Selected

The numbers are computed from the formats. No code exists for these
mechanisms.

| Mechanism                                                                                                  | Added bytes                                                                                                  | Added queries                                | Limit                                                                                       |
|------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------|----------------------------------------------|---------------------------------------------------------------------------------------------|
| Authenticator of the publisher in each record (example: an Ed25519 signature over the index and the value) | 64 for each record in the database, in each parity of a hint and of an answer, and in each change of a delta | 0                                            | Needs a key that the server does not hold. It does not show that a record is the newest one |
| Merkle proof in each record                                                                                | `32 * ceil(log2 n)` for each record                                                                          | 0                                            | One changed record changes the proof of each record, so a delta holds all records           |
| Merkle path through PIR                                                                                    | A second database of about `2n` nodes of 32 bytes, with its own hints                                        | `ceil(log2 n)` for each lookup (binary tree) | Each added query uses a backup hint. The client needs the new root after each delta         |

With the mainnet parameters of
[data_format.md](data_format.md#hint-file-hintsbin) (lambda = 128, w = 65536),
64 bytes more in each record add 1,610,612,736 bytes to the hint file
(`lambda * w * 3 * 64`).

## What An Application Can Do

- **Compare the hash of HintInit.** Get the BLAKE3 hash of the database file
  from the publisher through a channel that the server does not control, and
  compare it with `HintSet::init_database`. This shows that HintInit read the
  database of the publisher. It shows nothing about later answers.
- **Put an authenticator in each record**, if the publisher is not the
  server. The publisher signs the index and the value of a record and stores
  the signature in the record. The client checks the signature after `Recon`.
  A server that withholds deltas from a client can still answer from an old
  version, with records that the publisher signed before.
- **Do not show the result of a check to the server**; see above.
- **After a wrong record, run HintInit again** with a new master seed.
