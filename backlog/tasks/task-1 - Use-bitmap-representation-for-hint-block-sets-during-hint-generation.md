---
id: TASK-1
title: Use bitmap representation for hint block sets during hint generation
status: Done
assignee: []
created_date: '2026-03-31 05:10'
labels:
  - optimization
  - memory
dependencies: []
references:
  - 'https://github.com/igor53627/plinko-rs/issues/116'
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replace Vec<Vec<usize>> storage of regular_hint_blocks and backup_hint_blocks with BlockBitset (bitmap) during hint generation.

Currently each hint stores c/2 block indices as usize (8 bytes each), totaling c/2 * 8 bytes per hint. A bitmap representation uses c/8 bytes per hint — a 32x reduction.

BlockBitset already exists in src/bin/hint_gen/bitset.rs and is used for the CT path. This task extends its use to the fast path and as the primary storage format in init_hints().
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 init_hints() returns BlockBitset instead of Vec<usize> for block sets
- [x] #2 fast_path.rs uses BlockBitset for membership checks instead of binary search
- [x] #3 Memory usage during hint generation reduced by ~32x for block storage
- [x] #4 All existing tests pass
<!-- AC:END -->
