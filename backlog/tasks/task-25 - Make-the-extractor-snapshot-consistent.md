---
id: TASK-25
title: Make the extractor snapshot consistent
status: To Do
assignee: []
created_date: '2026-11-17T11:36:00Z'
updated_date: '2026-11-17T17:33:00Z'
labels:
  - extractor
dependencies: []
references:
  - src/main.rs
  - docs/data_format.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The extractor opens a new read transaction for each 100,000 items and reads the block number once in its own transaction. On a live node the output mixes many blocks. The block number comes from the CanonicalHeaders table in MDBX; Reth v1.8.1 keeps headers in static files, and an empty table gives block 0 with no error. CodeIDs follow first appearance, so they are not stable between two extractions. On a nonce error the extractor leaves partial files.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The extractor reads accounts and storage from one consistent state, or it refuses to run on a database that changes during the run
- [ ] #2 The block number comes from the source that Reth uses for the best block, and a missing value is an error
- [ ] #3 metadata.json states the block that the data belongs to, and a check against the state root exists or the document states that there is none
- [ ] #4 CodeIDs are stable between extractions, or the document states that they are not
- [ ] #5 A failed run leaves no partial output files
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Scope decision of 2026-11-17 (backlog/decisions/decision-1): the full Ethereum state is not a target. The extractor is now a source of test data only, so a consistent snapshot has low value. Candidate to close; waits for the owner.
<!-- SECTION:NOTES:END -->
