---
id: TASK-24
title: Let the client check an answer against a commitment of the database
status: Done
assignee: []
created_date: '2026-11-17T11:36:00Z'
updated_date: '2026-11-18T19:23:00Z'
labels:
  - security
  - design
dependencies:
  - TASK-23
references:
  - plinko/src/protocol.rs
  - plinko/src/hints.rs
  - docs/trust_model.md
  - docs/online_protocol.md
  - >-
    backlog/decisions/decision-1 -
    Target-a-PIR-library-for-fixed-size-records-not-the-full-Ethereum-state.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The client computes answer = server parity XOR hint parity. A server can XOR any value into its parity, and the client gets a wrong record with no error. The privacy game of the paper computes the answer with the honest Answer algorithm, so the paper gives no integrity. The client reads the full database once in HintInit, so it can compute a commitment of the database at that time; today plinko_hints computes none, and the hint file has no field for one. Nothing tells the client that the database that it used for HintInit is the database that the server published to other clients, and deltas (TASK-23) are not authenticated.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A document states what the client can check and what it must trust the server for: in HintInit, in an answer and in a delta
- [x] #2 The document selects a mechanism against a wrong answer (for example a per-record tag under a client key that the parity covers, or a Merkle path that the client fetches by PIR), with its cost in bytes and in queries, or it states that integrity is out of scope and gives the reason
- [x] #3 plinko_hints computes a hash of the database in its streaming pass and stores it in the hint file; a caller can compare it with a value that the publisher of the database gives
- [ ] #4 If a mechanism is selected: a test shows that the client rejects an answer in which the server changed one bit of a parity
- [x] #5 The document states how a delta changes the commitment, or that the client must trust the deltas
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Rewritten on 2026-11-17 after the scope decision (backlog/decisions/decision-1). Removed: the state root, the block hash and the trie leaf of Ethereum (metadata.json of the extractor). Moved to TASK-31: the parity covers the whole record (before: the TAG of storage records at bytes 32..40 is outside the parity). Done outside this task: the TAG collision comment in plinko/src/schema40.rs gives the probability for the mainnet record counts. The mechanisms in criterion 2 are examples, not a selection; no analysis of them was done.

From TASK-23 (2026-11-18): (1) Criterion 3 is done: plinko_hints hashes the database (BLAKE3) and the hint file (version 4) stores the hash with the version that HintInit read (HintSet::init_database). A delta does not update this hash. (2) A delta is bound to a version number only, not to a database: Client::apply_delta accepts a delta of another database with the same record size, number of records and version. Decide here on a binding, for example from_hash and to_hash of the database in the delta header, with the client keeping the current hash. (3) Response::database_version is a label that the server states; the client cannot check it.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Decision: the library gives no integrity of an answer or of a delta. docs/integrity.md states what the client checks and what it must trust in HintInit, in an answer and in a delta (criterion 1), gives the reasons for the decision and three mechanisms that were not selected with their cost in bytes and in queries (criterion 2), and states that the client must trust the deltas: a delta does not update the hash of the hint file and is bound to a version number only (criterion 5). Criterion 3 was done in TASK-23 (BLAKE3 hash of the database in the hint file). Criterion 4 does not apply: no mechanism is selected. New test a_wrong_answer_gives_a_wrong_record_and_the_same_later_messages: a changed parity gives a wrong record with no error, the error spreads only through the promoted backup hint, and the later messages of the client are those of a client with correct answers. Not implemented, for a decision of the owner: a hash of the database before and after the changes in the header of a delta (a format change; the publisher would hash the full database for each version; it detects a delta of another database, not a wrong delta). The decision has no record in backlog/decisions; the owner has not confirmed it.
<!-- SECTION:FINAL_SUMMARY:END -->
