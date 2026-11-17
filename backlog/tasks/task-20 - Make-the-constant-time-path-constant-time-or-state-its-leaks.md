---
id: TASK-20
title: Make the constant-time path constant-time or state its leaks
status: To Do
assignee: []
created_date: '2026-11-17T11:36:00Z'
updated_date: '2026-11-17T12:12:00Z'
labels:
  - constant-time
  - hint-gen
dependencies:
  - TASK-19
references:
  - plinko/src/iprf.rs
  - plinko/src/bin/hint_gen/ct_path.rs
  - docs/constant_time_mode.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
SwapOrNotSrTee::inverse runs zero rounds on levels that the input does not reach: 15 us at z = 65535 and 213 us at z = 0 for domain 65536, and z depends on the key. ct_path reads cutoffs and parities at the secret hint index. The CLI help promises data-independent memory access, and docs/constant_time_mode.md says the opposite. The compiled code divides by secret values (x % n).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The time of SwapOrNotSrTee::inverse does not depend on its input (statistical timing test), or the document states the leak
- [ ] #2 The CLI help of --constant-time and docs/constant_time_mode.md make the same claim on memory access
- [ ] #3 The document states the attacker (for example an SEV-SNP host that sees page accesses) and what the hint index access reveals
- [ ] #4 Secret-dependent divisions in the constant-time code are removed or listed
- [ ] #5 The mode is kept only if the trust model task needs it
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
From TASK-18: the round key of a stage is a 64-bit AES output mod n (SwapOrNotSr and SwapOrNotSrTee derive_round_key). Replace it here with a reduction that has no division and a bias below n / 2^128 (for example a 128-bit multiply-shift), and give the stage of size 2 the fixed round key 1. This changes the hints for a seed. Analysis: docs/hint_generation.md, PRP Deviations From An Ideal Permutation.
<!-- SECTION:NOTES:END -->
