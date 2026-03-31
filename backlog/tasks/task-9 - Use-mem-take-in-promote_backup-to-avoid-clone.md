---
id: TASK-9
title: "Use mem::take in promote_backup to avoid clone"
status: Done
assignee: []
created_date: '2026-03-31 08:00'
labels:
  - performance
  - protocol
dependencies: []
references:
  - 'https://github.com/igor53627/plinko-rs/issues/116'
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`promote_backup` at line 645 does `backup.cached_blocks.clone()` to move the block list into the promoted `HintSlot`. Since the backup is consumed (`available = false`), its `cached_blocks` becomes dead. Using `std::mem::take` instead of `.clone()` avoids the allocation entirely.

```rust
// Current:
let cached_blocks = backup.cached_blocks.clone();

// Proposed:
let cached_blocks = std::mem::take(&mut backup.cached_blocks);
```
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 promote_backup uses mem::take instead of clone for cached_blocks
- [x] #2 All existing tests pass
<!-- AC:END -->
