---
id: TASK-14
title: Reuse PMNS tree counts across offsets
status: Done
assignee: []
created_date: '2026-11-17T08:27:00Z'
updated_date: '2026-11-17T11:00:00Z'
labels:
  - iprf
  - performance
dependencies: []
references:
  - plinko/src/binomial.rs
  - plinko/src/iprf.rs
  - plinko/src/bin/hint_gen/fast_path.rs
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Each Iprf::inverse and Iprf::forward call generates the keystream of every node on its path again. At w = 65536 and 16.8M hints the root node alone is 131072 AES blocks. HintInit repeats this for each of the w offsets of a database block, and IprfTee does domain-sized sampling at every tree level. Compute the node counts once and reuse them: one full tree per database block in HintInit, and cached upper levels in the client. Found by a CodeRabbit CLI review on 2026-11-17.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 HintInit computes the PMNS tree of a database block once, not once per offset
- [x] #2 The client can cache upper tree levels; the cache stays on the client and is derived from the block key
- [x] #3 The constant-time path keeps work that does not depend on secret counts
- [x] #4 A test shows that outputs are identical to the uncached Iprf for the same key
- [x] #5 forward and inverse timings at w = 65536 are measured before and after
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Iprf::table and IprfTee::table_ct walk the tree breadth-first once (pmns_level_counts) and keep the prefix sums of the bin loads. HintInit builds one table per database block in fast_path, ct_path and protocol::compute_hints. Iprf::with_cached_levels keeps the left counts of the top levels in a OnceLock (u32 counts; off for a domain above u32::MAX); the client uses 8 levels, 1020 bytes per block.

table_ct samples a node of b bins with the public bound min(domain, MAX_PREIMAGES * b), clamps the count with ct_min_u64 and sets an overflow flag; ct_path panics on overflow. Before, a bin above MAX_PREIMAGES was truncated silently. AC 3 is checked by code reading and by equality tests, not by a timing measurement.

Measured at w = 65536, 16,777,216 hints, hardware AES, one thread: forward 445 us -> 28.5 us; inverse 7.56 ms -> 7.02 ms; inverse_ct 72.5 ms -> 68.0 ms; table build 4.2 ms, table_ct build 9.2 ms, cache build 1.7 ms. The inverse PRP of about 256 balls per offset dominates, so HintInit is still about 7 ms per entry.

Mutation checks: a wrong child index in trace_ball and a disabled leaf overflow check each made a new test fail.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
HintInit computes the PMNS tree once per block and the client caches 8 top levels. Outputs are unchanged. forward is 16 times faster; inverse changes by 7 percent because the inverse PRP dominates. Full tests, ignored tests, clippy, fmt, the local e2e script, and a CodeRabbit review (0 findings) pass.
<!-- SECTION:FINAL_SUMMARY:END -->
