---
id: TASK-4
title: Add regression tests for block set representation changes
status: Done
assignee: []
created_date: '2026-03-31 05:12'
labels:
  - testing
dependencies:
  - TASK-1
  - TASK-2
references:
  - 'https://github.com/igor53627/plinko-rs/issues/116'
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add regression tests to ensure correctness is preserved when switching from Vec<usize> to BlockBitset (TASK-1) and from materialized subset_blocks to seed-based recomputation (TASK-2).

Tests should cover:
- BlockBitset membership agrees with sorted Vec<usize> binary search for all block indices
- Hint parity computation produces identical results with bitmap vs sorted array
- promote_backup with complement flag produces same subset as complement_subset()
- try_build_query produces identical queries with seed recomputation vs stored blocks
- End-to-end PIR correctness: full offline_init → query → answer → reconstruct cycle matches current behavior
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Test: BlockBitset membership matches binary search on sorted Vec<usize> for random and edge-case inputs
- [x] #2 Test: Hint parity output is identical under both block storage representations
- [x] #3 Test: promote_backup with complement flag matches complement_subset() output
- [x] #4 Test: try_build_query produces identical queries with seed recomputation vs stored Vec<usize>
- [x] #5 Test: End-to-end PIR correctness (offline_init → query → answer → reconstruct) unchanged
<!-- AC:END -->
