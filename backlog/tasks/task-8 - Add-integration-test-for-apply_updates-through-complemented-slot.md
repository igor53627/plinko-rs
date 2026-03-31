---
id: TASK-8
title: Add integration test for apply_updates through complemented slot
status: Open
assignee: []
created_date: '2026-03-31 08:00'
labels:
  - testing
  - protocol
dependencies: []
references:
  - 'https://github.com/igor53627/plinko-rs/issues/116'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The `apply_updates` path in `protocol.rs` calls `slot_block_in_subset()` on promoted slots which may have `complemented = true`. There is no test that promotes a backup where `queried_in_subset = true` (resulting in `complemented = true` on the promoted slot) and then applies an update that hits the complemented block membership logic.

This is the most important correctness path for the complement flag and should have a dedicated integration test.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Test promotes a backup where queried_block is in the original subset (complemented = true)
- [ ] #2 Test applies updates that include blocks both inside and outside the complemented subset
- [ ] #3 Test verifies query correctness after updates through the complemented slot
<!-- AC:END -->
