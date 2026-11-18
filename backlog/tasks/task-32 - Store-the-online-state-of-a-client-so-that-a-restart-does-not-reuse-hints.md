---
id: TASK-32
title: Store the online state of a client so that a restart does not reuse hints
status: To Do
assignee: []
created_date: '2026-11-18T16:31:00Z'
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
- [ ] #1 A client can write its full state after queries (used hint slots, promoted backup hints, cache, database version) and a new process can load it
- [ ] #2 A test queries, writes the state, loads it in a new client, and checks that no hint that was sent is used for another index
- [ ] #3 A state that is older than the last query of the client is detected or the document states that the caller must prevent it (rollback of the state file)
- [ ] #4 docs/online_protocol.md and docs/update_strategy.md describe the state file and its secrecy (it holds queried indices and records)
<!-- AC:END -->
