---
id: TASK-18
title: Cut the inverse PRP cost of HintInit
status: Done
assignee: []
created_date: '2026-11-17T11:36:00Z'
updated_date: '2026-11-17T12:17:00Z'
labels:
  - iprf
  - performance
dependencies: []
references:
  - plinko/src/iprf.rs
  - plinko/src/bin/hint_gen/fast_path.rs
  - docs/hint_generation.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
After TASK-14 an iPRF inverse at w = 65536 and 16,777,216 hints still takes about 7 ms with hardware AES. The time is the inverse Sometimes-Recurse PRP of about 256 balls per offset: about 810 rounds on the top level and 2 AES calls per round, and the round key is derived again for each element in each round. HintInit over mainnet (1.83e9 entries) is about 7 ms per entry on one thread, and plinko_hints has one thread.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Round keys are derived once per (level, round), not once per element
- [x] #2 The AES calls of the balls of one offset are batched, or a measurement shows that this gives no gain
- [x] #3 forward and inverse outputs are identical to the current code for the same key (test on a domain above 65536)
- [x] #4 inverse and inverse_ct at w = 65536 are measured before and after with hardware AES
- [x] #5 plinko_hints uses more than one thread for the streaming phase, and the hint file is byte-identical to the one-thread file
- [x] #6 The PRP bias on domain 2 and the mod-n bias of the 64-bit round keys are fixed or documented
- [x] #7 docs/hint_generation.md states the measured core-hours for a mainnet HintInit
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
The inverse PRP of the balls of one offset is one batched call (SwapOrNotSr::inverse_range, SwapOrNotSrTee::inverse_many_ct): one round key per round, swap decisions as one AES batch. Outputs equal the single-value functions (tests on domains 100,003 and 131,072). Measured at w = 65536, 16,777,216 hints, one thread, hardware AES: inverse 7.04 ms -> 0.52 ms, inverse_ct 68.3 ms -> 16.7 ms per entry. The constant-time variant runs every stage for every slot. plinko_hints --threads runs the standard path on worker threads with atomic XOR accumulators; hint files are byte-identical for 1/2 threads (2 blocks) and 8/24/32 threads (24 blocks). Extrapolated mainnet HintInit: about 300 core-hours, about 13 h on 24 cores (no mainnet run). The round key mod-n bias and the one-round stage of size 2 are documented in docs/hint_generation.md, not fixed; the replacement is a note in TASK-20. Not done: threads in the constant-time path and in protocol::compute_hints; cutoff init is single-threaded.
<!-- SECTION:FINAL_SUMMARY:END -->
