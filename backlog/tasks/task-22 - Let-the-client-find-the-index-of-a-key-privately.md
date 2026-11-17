---
id: TASK-22
title: Let the client look up a record by key
status: To Do
assignee: []
created_date: '2026-11-17T11:36:00Z'
updated_date: '2026-11-17T17:43:00Z'
labels:
  - lookup
  - design
dependencies:
  - TASK-31
references:
  - docs/keyword_lookup.md
  - plinko/src/protocol.rs
  - https://github.com/keewoolee/rms24-plinko-spec
  - >-
    backlog/decisions/decision-1 -
    Target-a-PIR-library-for-fixed-size-records-not-the-full-Ethereum-state.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PIR takes an integer index. A client that knows a key and not the index of its record needs a private step from the key to the index. The library has none: a caller must hold a full map from keys to indices, or send the key to a party that holds the map, which defeats PIR. docs/keyword_lookup.md proposes a cuckoo table (2 hash functions, 4 slots per bucket, slot = tag and payload, 2 PIR queries per lookup); its numbers are for the Ethereum v3 snapshot, and no code implements it. keewoolee/rms24-plinko-spec (pir/keyword_pir) and ~/pse/rms24-rs (src/keyword_pir) have a cuckoo keyword PIR. After the scope decision (decision-1) the task is about a database of key and value records of any kind, not about Ethereum addresses and slots.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 docs/keyword_lookup.md describes the layout for a general key-value database, with the key size, the value size and the load as parameters, and gives the number of PIR queries per lookup
- [ ] #2 A table builder in the library writes database.bin in that layout from a list of key and value records, and reports a key that it cannot place
- [ ] #3 The client computes the candidate indices of a key from the public table parameters, without a key map and without a message that contains the key
- [ ] #4 A lookup returns the value of a key that is in the table. For a key that is not in the table it returns a not-found result, except with the false-match probability that the document states (criterion 6); the client checks the tag of each answer against its key
- [ ] #5 An end-to-end test on a synthetic key-value database goes from keys to the correct values through plinko_hints and the protocol client
- [ ] #6 The document states the tag size, the probability of a false match, and what an insert after the build needs
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Rewritten on 2026-11-17 after the scope decision (backlog/decisions/decision-1); before, the task was about Ethereum addresses and (address, slot) pairs and the mapping files of 91 GB. The earlier criterion 1 (a design document selects the layout) was checked for the Ethereum-sized proposal in docs/keyword_lookup.md; the owner did not confirm that layout, and the document is not general yet, so the new criterion 1 is open. Depends on TASK-31: a slot is a tag and a payload, so the parity must cover the whole record. Simulation crate of the proposal: /tmp/plinko-review-main/cuckoo_sim (scratch, not in the repository).
<!-- SECTION:NOTES:END -->
