---
id: TASK-10
title: Add dedicated non-CT contains() to BlockBitset
status: Open
assignee: []
created_date: '2026-03-31 08:00'
labels:
  - performance
dependencies: []
references:
  - 'https://github.com/igor53627/plinko-rs/issues/116'
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`BlockBitset::contains()` currently delegates to `contains_ct()`, which means the non-CT fast path pays constant-time overhead (no early exit on out-of-bounds, no branch on the bit value).

A dedicated implementation would be:
```rust
pub fn contains(&self, block: usize) -> bool {
    if block >= self.num_blocks {
        return false;
    }
    (self.bits[block / 64] >> (block % 64)) & 1 == 1
}
```

This avoids entangling the CT and non-CT code paths.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 contains() uses a direct implementation instead of delegating to contains_ct()
- [ ] #2 All existing tests pass
- [ ] #3 Benchmark shows no regression (or improvement)
<!-- AC:END -->
