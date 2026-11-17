---
id: TASK-23
title: Version the database and implement a delta feed for hint updates
status: To Do
assignee: []
created_date: '2026-11-17T11:36:00Z'
updated_date: '2026-11-17T17:43:00Z'
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
- [ ] #1 The database has a version number; an answer and a delta carry it, and the client rejects an answer or a delta of another version than it expects
- [ ] #2 A delta format is specified in docs/data_format.md with the record size as a field, and plinko/docs/delta-format.md is removed or replaced by it
- [ ] #3 A delta writer and a delta reader exist in the library, and a test applies a sequence of deltas to a hint set that was loaded from a file, then queries changed and unchanged records
- [ ] #4 The client refuses a delta that is out of order or that repeats, and its hints do not change in that case
- [ ] #5 docs/update_strategy.md states the rule for a database that grows: which appends need no new HintInit (records in the padding of the last block), and when a new HintInit is necessary (a change of w or c)
- [ ] #6 The hint file stores the database version that the hints belong to, so that a client that restarts knows which deltas to apply
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Rewritten on 2026-11-17 after the scope decision (backlog/decisions/decision-1). Removed: stable indices for inserts in the Ethereum v3 layout (accounts first, then storage, in key order), and the dependency on the lookup task. The measured update cost is in docs/update_strategy.md and docs/protocol_overview.md since TASK-27. If TASK-22 is done, inserts into the cuckoo table move records; the rule for that belongs to this task (see Limits in docs/keyword_lookup.md).
<!-- SECTION:NOTES:END -->
