---
id: TASK-18
title: Cut the inverse PRP cost of HintInit
status: To Do
assignee: []
created_date: '2026-11-17T11:36:00Z'
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
- [ ] #1 Round keys are derived once per (level, round), not once per element
- [ ] #2 The AES calls of the balls of one offset are batched, or a measurement shows that this gives no gain
- [ ] #3 forward and inverse outputs are identical to the current code for the same key (test on a domain above 65536)
- [ ] #4 inverse and inverse_ct at w = 65536 are measured before and after with hardware AES
- [ ] #5 plinko_hints uses more than one thread for the streaming phase, and the hint file is byte-identical to the one-thread file
- [ ] #6 The PRP bias on domain 2 and the mod-n bias of the 64-bit round keys are fixed or documented
- [ ] #7 docs/hint_generation.md states the measured core-hours for a mainnet HintInit
<!-- AC:END -->
