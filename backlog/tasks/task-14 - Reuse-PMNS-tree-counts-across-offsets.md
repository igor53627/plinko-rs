---
id: TASK-14
title: Reuse PMNS tree counts across offsets
status: To Do
assignee: []
created_date: '2026-11-17T08:27:00Z'
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
- [ ] #1 HintInit computes the PMNS tree of a database block once, not once per offset
- [ ] #2 The client can cache upper tree levels; the cache stays on the client and is derived from the block key
- [ ] #3 The constant-time path keeps work that does not depend on secret counts
- [ ] #4 A test shows that outputs are identical to the uncached Iprf for the same key
- [ ] #5 forward and inverse timings at w = 65536 are measured before and after
<!-- AC:END -->
