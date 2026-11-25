---
id: TASK-38
title: >-
  Add multi-level PIR prototype, HintSet serialization, benchmark, and WASM
  smoke test
status: Done
assignee: []
created_date: '2026-11-25T11:44:59Z'
updated_date: '2026-11-25T11:45:07Z'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement ePrint 2026/1273 §4 multi-level PIR prototype (2-level and d-level waterfall) with dynamic delta updates, Hybrid A (Iprf+FlatAes) vs Flat B (FlatAes) backends, BLAKE3-checksummed HintSet serialization, CLI benchmark, and zero-import wasm32 E2E smoke test.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 plinko::multilevel implements 2-level and waterfall HintInit, Query, Answer, failure-private Refresh, and apply_delta
- [x] #2 HintSet binary serialization (PLNKMLH1) passes fresh and mid-session round-trip and corruption tests
- [x] #3 bench_multilevel compares Hybrid A vs Flat B across 2^20, 2^24, and Ethereum v3 reference scales
- [x] #4 scripts/test_wasm_multilevel.sh executes 2-level Variant B in Node.js with zero WASM host imports
- [x] #5 CodeRabbit findings resolved and fmt, clippy, tests, doc, and wasm32 checks pass
<!-- AC:END -->
