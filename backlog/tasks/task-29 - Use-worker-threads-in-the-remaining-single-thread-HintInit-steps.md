---
id: TASK-29
title: Use worker threads in the remaining single-thread HintInit steps
status: To Do
assignee: []
created_date: '2026-11-17T12:40:00Z'
labels:
  - hint-gen
  - performance
dependencies: []
references:
  - plinko/src/bin/hint_gen/driver.rs
  - plinko/src/bin/hint_gen/fast_path.rs
  - plinko/src/protocol.rs
  - plinko/src/bin/hint_gen/ct_path.rs
  - docs/hint_generation.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-18 added worker threads only to the streaming phase of the standard path of plinko_hints (hint_gen/fast_path.rs). Three steps still use one thread. (1) driver::init_hints computes the cutoff of each hint: 44 us per hint at c = 27988, about 12 minutes for 16,777,216 hints (computed from the cutoff benchmark). (2) protocol::compute_hints, which Client::offline_init uses, has its own one-thread streaming loop; a client that runs HintInit through the library gets no gain from TASK-18 threads. (3) The constant-time path rejects --threads above 1. The streaming loop exists twice: in fast_path.rs and in protocol.rs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 init_hints computes the cutoffs on the worker threads, and the cutoffs are equal to the one-thread result
- [ ] #2 protocol::compute_hints uses worker threads, and its HintSet is equal to the one-thread result
- [ ] #3 plinko_hints and protocol::compute_hints share one streaming implementation, or the document states why they do not
- [ ] #4 The constant-time path uses worker threads, or docs/constant_time_mode.md states why it must not (shared accumulators are read and written at the secret hint index)
- [ ] #5 Wall time of a 24-block run at w = 65536 and 16,777,216 hints is measured before and after, cutoff phase included
- [ ] #6 docs/hint_generation.md states the new mainnet estimate
<!-- AC:END -->
