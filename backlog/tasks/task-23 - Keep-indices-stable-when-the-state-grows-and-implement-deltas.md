---
id: TASK-23
title: Version the database and implement a delta feed for hint updates
status: Done
assignee: []
created_date: '2026-11-17T11:36:00Z'
updated_date: '2026-11-18T16:32:00Z'
labels:
  - updates
  - protocol
dependencies: []
references:
  - docs/update_strategy.md
  - plinko/docs/delta-format.md
  - plinko/src/protocol.rs
  - docs/online_protocol.md
  - >-
    backlog/decisions/decision-1 -
    Target-a-PIR-library-for-fixed-size-records-not-the-full-Ethereum-state.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Server::apply_updates and Client::apply_updates of plinko::protocol work in memory in one process. There is no format for a set of changes, no producer and no reader. An answer and a delta carry no database version, so a client cannot tell which database state an answer belongs to, and it cannot detect a delta that is missing or out of order. plinko/docs/delta-format.md is a proposal of 2025 that differs from the code (one forward call in place of the iPRF inverse, account records only). An update changes the value of a record at a fixed index; the library has no rule for a database that gets more records. The client pays one iPRF inverse for each changed record (0.51 to 0.53 ms, measured).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The database has a version number; an answer and a delta carry it, and the client rejects an answer or a delta of another version than it expects
- [x] #2 A delta format is specified in docs/data_format.md with the record size as a field, and plinko/docs/delta-format.md is removed or replaced by it
- [x] #3 A delta writer and a delta reader exist in the library, and a test applies a sequence of deltas to a hint set that was loaded from a file, then queries changed and unchanged records
- [x] #4 The client refuses a delta that is out of order or that repeats, and its hints do not change in that case
- [x] #5 docs/update_strategy.md states the rule for a database that grows: which appends need no new HintInit (records in the padding of the last block), and when a new HintInit is necessary (a change of w or c)
- [x] #6 The hint file stores the database version that the hints belong to, so that a client that restarts knows which deltas to apply
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Rewritten on 2026-11-17 after the scope decision (backlog/decisions/decision-1). Removed: stable indices for inserts in the Ethereum v3 layout (accounts first, then storage, in key order), and the dependency on the lookup task. The measured update cost is in docs/update_strategy.md and docs/protocol_overview.md since TASK-27. If TASK-22 is done, inserts into the cuckoo table move records; the rule for that belongs to this task (see Limits in docs/keyword_lookup.md).

TASK-23 result (2026-11-18): (1) Database version: Server::new, Client::offline_init, compute_hints and plinko_hints --database-version take it; Response::database_version; Client::reconstruct_and_replenish returns DatabaseVersionMismatch and leaves the query in flight. (2) plinko::delta::Delta with a file format (PLNKDLTA, format version 1), writer and reader; Server::apply_updates returns a Delta of one version; Client::apply_delta replaces Client::apply_updates. (3) Hint file version 4: database_version, init_version, init_hash (BLAKE3 of the database bytes; this is criterion 3 of TASK-24). (4) docs/update_strategy.md has the rule for a database that grows; plinko/docs/delta-format.md is removed. Decisions: the number of records is fixed for a hint set, and a publisher reserves zero records (a delta that raises num_entries inside the padding is not implemented); the library does not assign the first version number. Limits: (a) a delta is not bound to a database, only to a version number (CodeRabbit finding; a binding by hash belongs to TASK-24); (b) the version of an answer is a label of the server, not checked; (c) the hint file holds no state of the online phase, so a client that loads its file again after queries reuses hints, and a stored client state is not implemented; (d) no program writes delta files for a database on disk, and there is no transport.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Commit f1fc0ef2. The database has a version; an answer and a delta carry it, and the client accepts only the version of its hints (a mismatch leaves the query in flight and the hints unchanged). plinko::delta has the delta format, a writer and a reader; Client::apply_delta rejects a delta that repeats or is out of order. Hint file version 4 stores the version of its parities and the version and BLAKE3 hash of the database that HintInit read. docs/update_strategy.md has the rule for a database that grows. Tests: deltas through files on a hint set from the plinko_hints binary, restart from a written hint file, parities after the deltas equal to HintInit on the new database. Not solved: a delta and the version of an answer are not authenticated and a delta is not bound to a database (TASK-24); the hint file has no state of the online phase (TASK-32); no delta producer for files and no transport; appends inside the padding are not implemented. Not verified: CI, a mainnet-size run, a second CodeRabbit pass after the fixes of the first.
<!-- SECTION:FINAL_SUMMARY:END -->
