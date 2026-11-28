---
id: TASK-3
title: Add memory benchmarks for hint generation and client storage
status: Done
assignee: []
created_date: '2026-03-31 05:11'
labels:
  - testing
  - benchmarks
dependencies:
  - TASK-1
  - TASK-2
references:
  - 'https://github.com/igor53627/plinko-rs/issues/116'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add benchmarks to measure and validate memory improvements from issue #116 optimizations.

Include:
- Peak memory measurement during init_hints() for varying c values
- Client HintSlot storage measurement before/after seed-based compaction
- Comparison report showing memory reduction ratios
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Benchmark measures peak memory during hint generation
- [x] #2 Benchmark compares Vec<usize> vs BlockBitset storage
- [x] #3 Results confirm expected ~32x reduction for hint gen block storage
<!-- AC:END -->
