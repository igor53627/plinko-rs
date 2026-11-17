---
id: TASK-24
title: Let the client verify an answer
status: To Do
assignee: []
created_date: '2026-11-17T11:36:00Z'
updated_date: '2026-11-17T17:33:00Z'
labels:
  - data-layer
  - security
dependencies: []
references:
  - plinko/src/schema40.rs
  - plinko/src/bin/hint_gen/types.rs
  - plinko/src/protocol.rs
  - src/main.rs
  - docs/data_format.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The parity covers the first 32 bytes of a record. A storage record has its TAG at bytes 32..40, so the client never gets the storage TAG through PIR. The client computes answer = server parity XOR hint parity, so a server can XOR a value into the balance and leave the TAG intact. metadata.json has no state root, block hash or hash of database.bin. The account record has no storage root, a 128-bit balance and a local CodeID, so a trie leaf cannot be rebuilt. The comment in schema40.rs gives 5e-20 as the TAG collision probability; that is for one pair, and at mainnet scale the chance of a collision is 8.7 percent over all records.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The parity window covers the TAG of storage records in the generator, the hint file and the client
- [ ] #2 metadata.json holds the block number, the block hash, the state root and a hash of database.bin
- [ ] #3 A document states what the client can verify and what it must trust the server for
- [ ] #4 The document selects a mechanism against a malleable answer, or states that integrity is out of scope
- [ ] #5 The TAG collision comment in schema40.rs gives the mainnet-scale probability
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Scope decision of 2026-11-17 (backlog/decisions/decision-1): the full Ethereum state is not a target. The link to a state root is Ethereum-specific. A check of an answer against a commitment of the database is a general need; the owner decides whether to change the task.
<!-- SECTION:NOTES:END -->
