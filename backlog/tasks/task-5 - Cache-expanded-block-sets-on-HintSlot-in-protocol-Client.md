---
id: TASK-5
title: Cache expanded block sets on HintSlot in protocol Client
status: Done
assignee: []
created_date: '2026-03-31 06:00'
labels:
  - performance
  - protocol
dependencies:
  - TASK-2
references:
  - 'https://github.com/igor53627/plinko-rs/issues/116'
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
After TASK-2, `slot_block_in_subset()` recomputes blocks from seed (ChaCha20 + sampling + Vec allocation) on every call. This is called in tight loops:

- `prepare_query()`: per-candidate in the IPRF inverse loop
- `try_build_query()`: per-block (iterates all `num_blocks`)
- `apply_updates()`: per-update per-hint-index

This is a significant performance regression vs the original pre-computed `Vec<usize>`.

The seed-only form is correct for serialized/on-disk storage, but the in-memory `Client` should lazily cache the expanded `Vec<usize>` (or `BlockBitset`) on each `HintSlot` and `BackupHint`.

Same issue applies to `apply_updates` for available backup hints, where `compute_backup_blocks` is called per-update.

**Fix:** Add a cached field (e.g. `cached_blocks: Option<Vec<usize>>`) to `HintSlot` and compute it once on first access, or eagerly populate it in `offline_init` and `promote_backup`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 slot_block_in_subset() does not allocate on every call
- [ ] #2 HintSlot caches the expanded block set (computed once from seed)
- [ ] #3 BackupHint in apply_updates does not recompute blocks per update
- [ ] #4 All existing tests pass
- [ ] #5 No measurable regression in protocol e2e test runtime
<!-- AC:END -->
