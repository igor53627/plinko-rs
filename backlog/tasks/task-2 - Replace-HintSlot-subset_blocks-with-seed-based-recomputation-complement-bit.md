---
id: TASK-2
title: Replace HintSlot subset_blocks with seed-based recomputation + complement bit
status: To Do
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
In protocol.rs, HintSlot currently stores subset_blocks: Vec<usize> — a full materialized copy of the block set.

Since block sets are deterministically derivable from the subset_seed (already stored in RegularHint/BackupHint), the client can recompute blocks on-demand instead of persisting them.

For promoted backups, store a single bool indicating whether the subset was complemented (from promote_backup). This reduces persistent client storage from O(lambda * c) to O(lambda).

Key changes:
- HintSlot stores seed + complemented flag instead of Vec<usize>
- block_in_subset() recomputes from seed when needed
- promote_backup() sets the complement flag instead of calling complement_subset()
- try_build_query() recomputes blocks on-the-fly
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 HintSlot no longer stores Vec<usize> subset_blocks
- [ ] #2 Block membership checks recompute from seed on-demand
- [ ] #3 promote_backup stores a complement flag instead of materializing the complement set
- [ ] #4 Client persistent storage reduced from O(lambda*c) to O(lambda)
- [ ] #5 All existing tests pass
<!-- AC:END -->
