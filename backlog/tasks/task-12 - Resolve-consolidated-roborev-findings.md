---
id: TASK-12
title: Resolve consolidated roborev findings
status: Done
assignee: []
created_date: '2026-06-07 21:18'
labels:
  - roborev
  - maintenance
dependencies: []
references:
  - roborev job 7594
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve the consolidated roborev findings from job 7594 after compacting open reviews across all branches.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Docker runtime image no longer copies a missing binary.
- [x] #2 Binomial large-parameter oracle, convergence tracking, and mainnet monotonicity tests are updated.
- [x] #3 Stale documentation findings are corrected.
- [x] #4 Targeted tests and roborev verification pass.
<!-- AC:END -->

## Verification

- `cargo build -p plinko --bins`
- `cargo test -p plinko --lib`
- `git diff --check`
- `roborev review --dirty --wait --agent codex --reasoning thorough` (job 7595, no issues)
