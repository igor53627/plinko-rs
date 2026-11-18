---
id: TASK-20
title: Make the constant-time path constant-time or state its leaks
status: Done
assignee: []
created_date: '2026-11-17T11:36:00Z'
updated_date: '2026-11-18T05:38:00Z'
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
- [x] #1 The time of SwapOrNotSrTee::inverse does not depend on its input (statistical timing test), or the document states the leak
- [x] #2 The CLI help of --constant-time and docs/constant_time_mode.md make the same claim on memory access
- [x] #3 The document states the attacker (for example an SEV-SNP host that sees page accesses) and what the hint index access reveals
- [x] #4 Secret-dependent divisions in the constant-time code are removed or listed
- [x] #5 The mode is kept only if the trust model task needs it
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
From TASK-18: the round key of a stage is a 64-bit AES output mod n (SwapOrNotSr and SwapOrNotSrTee derive_round_key). Replace it here with a reduction that has no division and a bias below n / 2^128 (for example a 128-bit multiply-shift), and give the stage of size 2 the fixed round key 1. This changes the hints for a seed. Analysis: docs/hint_generation.md, PRP Deviations From An Ideal Permutation.

From TASK-29 (2026-11-18): (1) The constant-time path computes the cutoffs with hint_init::hint_cutoffs, the same function as the standard path. It is not constant-time: select_nth_unstable over the secret select values of a hint. docs/constant_time_mode.md states this. (2) plinko_hints --constant-time accepts --threads; the default is 1 thread. With more workers, two workers that update the same parity word delay each other, and which words those are depends on the hint indices; the size of the effect is not measured. (3) The constant-time streaming loop is plinko/src/hint_init/tee.rs (fill_parities_ct), not plinko/src/bin/hint_gen/ct_path.rs. (4) plinko/tests/ct_hintinit_test.rs tests copies of both loops, not the library functions; plinko/src/hint_init/tests.rs tests the functions against the hint definition.

TASK-20 result (2026-11-18): (1) Round keys: sr_round_key with reduce_below, floor(r * n / 2^128) for a 128-bit AES output; the stage of size 2 has the fixed key 1. The hints of a seed change; hint file version 3. (2) SwapOrNotSrTee::inverse and ::forward run every stage under a mask; timing test test_sr_tee_time_does_not_depend_on_the_input (release only). (3) Cutoffs of the constant-time path: hint_cutoffs_ct, find_cutoff_ct (bit-by-bit search). (4) SwapOrNotTee: no division on the input; its round key keeps AES mod domain (listed; HintInit does not use it). (5) Criterion 5: the owner decided on 2026-11-17 to keep the TEE code outside the trust model and not to delete it (TASK-19, TASK-30); the mode stays an opt-in experimental feature. Limits: the memory accesses at the secret hint index stay and reveal the iPRF mapping to an observer of pages or cache lines; their cache-dependent time is visible to a timing attacker too (found by the CodeRabbit review; not measured); thread contention is not measured; the machine code was not checked with a tool; one machine (Apple M3 Ultra); no SEV-SNP run.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Commit e03d8205. The number and the kind of the operations of the constant-time path are fixed on one thread: PRP round keys without a division (reduce_below; hint file version 3, breaking), SwapOrNotSrTee::inverse and ::forward over every stage (timing test, release only), constant-time cutoffs (hint_cutoffs_ct). docs/constant_time_mode.md states the claim, the attackers and a list of what remains; the CLI help makes the same claim. Not solved: the memory accesses at the secret hint index reveal the iPRF mapping to an observer of pages or cache lines (analysis: the queried index is then one of c/2 candidates), and their cache-dependent time is visible to a timing attacker (not measured). Not verified: machine code, thread contention, a second machine, SEV-SNP. Measured (M3 Ultra, hardware AES): CT run on 24 blocks of w = 1024 291.3 s on 1 thread, file byte-identical to the standard path; cutoffs 102 us against 39 us for each hint at c = 27,988.
<!-- SECTION:FINAL_SUMMARY:END -->
