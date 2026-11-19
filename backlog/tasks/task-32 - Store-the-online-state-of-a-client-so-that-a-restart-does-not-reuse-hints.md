---
id: TASK-32
title: Store the online state of a client so that a restart does not reuse hints
status: Done
assignee: []
created_date: '2026-11-18T16:31:00Z'
updated_date: '2026-11-18T19:01:00Z'
labels:
  - protocol
  - privacy
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The hint file holds the hints of a client that has sent no query. The used hints, the promoted backup hints (with their extra index and the complement flag), the cache and the queries in flight are in memory only. A client that loads its hint file again after queries uses hints a second time; the rule of docs/online_protocol.md (a hint that was sent is not used for a second index) is then broken, and the server gets queried indices. Client::from_hints cannot detect this. Client::unused_hints returns None after the first query, so there is nothing to write. Found in TASK-23.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A client can write its full state after queries (used hint slots, promoted backup hints, cache, database version) and a new process can load it
- [x] #2 A test queries, writes the state, loads it in a new client, and checks that no hint that was sent is used for another index
- [x] #3 A state that is older than the last query of the client is detected or the document states that the caller must prevent it (rollback of the state file)
- [x] #4 docs/online_protocol.md and docs/update_strategy.md describe the state file and its secrecy (it holds queried indices and records)
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Client::write_state and Client::read_state store and load the state of a client that sent queries (file PLNKSTAT, format version 1, in plinko/src/protocol/state.rs; layout in docs/data_format.md). read_state returns the queries in flight with the same message bytes. Client::retire_hints removes the hints of queries that a state does not hold, from a log of PreparedQuery::hint_index that the caller keeps. Criterion 3 is met by documentation: the format cannot detect a state that is older than the last query; docs/online_protocol.md (Client State And Restart) states the rules for the caller (record each query before it leaves the client, load only the newest state or pass the log, new query seed for each start, keep the file secret). Limits: a state has the size of the hint file, so a write for each query is not practical at mainnet parameters (the log is the alternative); retire_hints removes hints with no promotion, the same deviation as abandon_query, and the answers of those queries are lost; reconstruct_and_replenish returns NoBackupHintsLeft with no change if retire_hints removed the last backup hint. Tests: 7 unit tests in plinko/src/protocol/state/tests.rs (same state as the client that wrote it, layout, each flipped bit and each truncation, 25 invalid fields with a correct checksum) and 3 tests in protocol_e2e_test.rs (restart from the newest state, from an old state with no log as the control, from an old state with the log). 8 mutations of the code were each caught by a test.
<!-- SECTION:FINAL_SUMMARY:END -->
