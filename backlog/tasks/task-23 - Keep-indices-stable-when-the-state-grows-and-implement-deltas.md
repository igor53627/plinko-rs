---
id: TASK-23
title: Keep indices stable when the state grows and implement deltas
status: To Do
assignee: []
created_date: '2026-11-17T11:36:00Z'
updated_date: '2026-11-17T17:33:00Z'
labels:
  - data-layer
  - updates
dependencies:
  - TASK-22
references:
  - docs/update_strategy.md
  - plinko/docs/delta-format.md
  - plinko/src/protocol.rs
  - src/main.rs
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The index is a running counter in key order: accounts first, then storage slots. One new account shifts all later accounts and all slots. docs/update_strategy.md describes only XOR at a fixed index, and apply_updates rejects an index at or above num_entries. plinko/docs/delta-format.md describes a service and a file format that no code implements; it has 4 words per entry, accounts only, and one forward call where the code uses the iPRF inverse. The client pays one iPRF inverse per changed entry.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 An insert or a delete of a key does not change the index of another entry between rebuilds, or the document gives the rebuild interval and its cost
- [ ] #2 The block size w does not change inside the growth window that the document states
- [ ] #3 A delta producer and a delta reader exist and a test applies a multi-block delta to client hints, or plinko/docs/delta-format.md is removed
- [ ] #4 The client update cost per changed entry is measured and docs/update_strategy.md and docs/protocol_overview.md state it
- [ ] #5 This layout and the layout of the index lookup task are one layout
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Scope decision of 2026-11-17 (backlog/decisions/decision-1): the full Ethereum state is not a target. The index shift on inserts is a property of the Ethereum v3 layout. A delta feed and a database version are still general needs; the owner decides whether to keep them as a smaller task.
<!-- SECTION:NOTES:END -->
