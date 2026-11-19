---
id: TASK-13
title: Replace f64 PMNS sampler with exact integer sampler
status: Done
assignee: []
created_date: '2026-11-17T08:27:00Z'
updated_date: '2026-11-17T08:27:00Z'
labels:
  - iprf
  - correctness
dependencies: []
references:
  - plinko/src/binomial.rs
  - plinko/src/iprf.rs
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
binomial_sample delegated to puruspe::betai for counts 1025 to about 12000 and returned values that were not binomial. PMNS bin loads were not multinomial, and Iprf disagreed with IprfTee above 1024 hints. Replace both f64 samplers with one exact integer sampler: the bit count of an AES-128 counter keystream per PMNS tree node. The iPRF range w must be a power of two.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Sampler output follows Binomial(count, 1/2) at ten selected counts from 1 to 12000 (Kolmogorov-Smirnov test)
- [x] #2 PMNS bin loads pass a chi-square test on domain 16384, range 64
- [x] #3 Iprf and IprfTee give the same PMNS tree; IprfTee has no 65536-hint limit
- [x] #4 plinko_hints and the protocol client derive the same default (w, c)
- [x] #5 puruspe dependency removed
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Commits: fix(iprf)!: replace f64 PMNS sampler with exact integer sampler; fix(hint_gen): pad small databases to two blocks like the client. Measured at w = 65536, 16.8M hints, hardware AES, one thread: 209 to 294 preimages per offset (median 256), forward 432 us, inverse 7.63 ms.
<!-- SECTION:FINAL_SUMMARY:END -->
