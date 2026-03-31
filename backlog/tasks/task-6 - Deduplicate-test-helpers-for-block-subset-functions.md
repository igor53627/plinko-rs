---
id: TASK-6
title: Deduplicate test helpers for block subset functions
status: Done
assignee: []
created_date: '2026-03-31 06:00'
labels:
  - tech-debt
  - testing
dependencies:
  - TASK-4
references:
  - 'https://github.com/igor53627/plinko-rs/issues/116'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`protocol_e2e_test.rs` and `ct_hintinit_test.rs` both reimplement private functions from `protocol.rs` and `bitset.rs`:

- `BlockBitset` (struct + `from_sorted_blocks` + `contains`/`contains_ct`)
- `compute_regular_blocks` / `compute_backup_blocks`
- `derive_subset_seed`
- `block_in_subset`

These local copies can silently diverge from the canonical implementations if the production code changes (e.g. subset size formula, RNG algorithm).

Options:
1. Make `compute_regular_blocks`, `compute_backup_blocks`, `derive_subset_seed` `pub(crate)` in `protocol.rs` and import them in tests
2. Move shared helpers to a `protocol::internals` or `protocol::test_utils` module
3. Re-export from a test utility crate

Additionally, `test_complement_flag_matches_complement_subset` tests local copies of the math, not the actual `slot_block_in_subset` production code path. An integration test through a promoted `Client` would provide stronger coverage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Test files import canonical implementations instead of reimplementing them
- [ ] #2 No local copies of compute_regular_blocks/compute_backup_blocks in test files
- [ ] #3 Complement flag test validates the production code path (not a local reimplementation)
<!-- AC:END -->
