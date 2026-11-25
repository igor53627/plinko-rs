---
id: TASK-37
title: >-
  Separate CLI dependencies, support wasm32, and optimize iPRF and Client
  query/update
status: Done
assignee: []
created_date: '2026-11-24T20:11:00Z'
updated_date: '2026-11-24T20:29:41Z'
labels:
  - performance
  - wasm
  - packaging
dependencies: []
references:
  - plinko/Cargo.toml
  - plinko/src/iprf.rs
  - plinko/src/protocol.rs
  - plinko/src/hint_init.rs
  - docs/FEATURE_FLAGS.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Separate CLI and OS-specific dependencies in plinko behind a cli feature so the library compiles and runs on wasm32-unknown-unknown without default features; optimize SwapOrNotSr forward/inverse and parallelize Client::prepare_query and Client::apply_delta while keeping outputs bit-identical.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 plinko builds and passes runtime checks on wasm32-unknown-unknown with --no-default-features and no CLI or getrandom dependencies
- [x] #2 Default build, tee, and gpu features keep all existing CLI binaries and pass cargo fmt, clippy, test, and doc
- [x] #3 SwapOrNotSr forward/inverse and Iprf operations produce bit-identical outputs with measurable speedup
- [x] #4 Client::prepare_query and Client::apply_delta support configurable worker threads, produce bit-identical outputs for any thread count, and track available backup hints in O(1)
- [x] #5 CI checks wasm32 compilation and docs/CHANGELOG reflect the feature flags and measured speedups
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-11-24: Separated CLI and OS-specific deps under the default cli feature, removed unused ctr/once_cell/cc, replaced eyre in CodeStore::from_bytes with CodeStoreError, made hint_init inline on single-worker and wasm32 targets, and added a wasm32-unknown-unknown CI job. Verified with a zero-import wasm32-unknown-unknown release cdylib running HintInit, HintSet I/O, Query/Answer/Recon, Delta I/O, and Client state save/restore in Node.js (254 KB, 283.8 ms). Optimized SwapOrNotSr with 16-block batched round keys, branchless partner/swap arithmetic, and 8-way interleaved forward_many; sliced coin_binomial to only encrypt needed blocks; added Client::threads/set_threads, parallel/batched prepare_query, parallel apply_delta, and O(1) amortized backup availability tracking. Benchmarks on Apple M3 Ultra (--cfg aes_armv8, N=1,048,576, w=1024, c=1024, lambda=64): warm prepare_query median 11.58 ms -> 2.82 ms (1t, 4.1x) / 0.64 ms (32t, 18.1x); cold first query 19.66 ms -> 11.16 ms (1t) / 1.79 ms (32t, 11.0x); 200-change apply_delta 67.15 ms -> 7.45 ms (32t, 9.0x); Iprf::forward_many 5.45 us/block vs 25.42 us/block baseline (4.66x single-core).
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
CLI dependencies gated behind the cli feature (default on), library builds and runs end-to-end on wasm32-unknown-unknown with --no-default-features, SwapOrNotSr/Iprf forward_many and Client query/delta optimizations achieve 4.1x single-core and 18.1x multi-core query speedup with bit-identical outputs.
<!-- SECTION:FINAL_SUMMARY:END -->
