---
id: TASK-30
title: Move the TEE mode and the GPU path behind opt-in build features
status: Done
assignee: []
created_date: '2026-11-17T13:05:00Z'
updated_date: '2026-11-17T13:16:00Z'
labels:
  - hint-gen
  - tee
  - gpu
dependencies: []
references:
  - plinko/Cargo.toml
  - plinko/src/iprf.rs
  - plinko/src/lib.rs
  - plinko/src/bin/plinko_hints.rs
  - docs/trust_model.md
  - docs/FEATURE_FLAGS.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-19 decided that the client runs HintInit and that the TEE mode and the GPU path are outside the trust model. The code of both is still part of every build of the plinko crate: IprfTee, SwapOrNotSrTee and SwapOrNotTee are in iprf.rs between the model code, plinko_hints always has --constant-time, and the gpu module and bench_gpu_hints are always compiled. The default build must contain the model only.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A build of the plinko crate without features has no constant-time code path: no IprfTee, no SwapOrNotSrTee, no constant_time module, no --constant-time option
- [x] #2 A build without features has no gpu module and no bench_gpu_hints binary
- [x] #3 The features tee and gpu compile that code again; cuda and parallel imply gpu, so existing GPU build commands still work
- [x] #4 The TEE variants of the PRP and the iPRF are in their own file, not between the model code in iprf.rs
- [x] #5 The check of the standard path against the hint definition runs without features; the check of the constant-time path against it runs with the tee feature
- [x] #6 All tests pass without features and with --features tee,gpu; the e2e script still compares the two hint files
- [x] #7 docs/FEATURE_FLAGS.md, README.md, docs/DEPLOYMENT.md, docs/constant_time_mode.md, docs/hint_generation.md and AGENTS.md give the build commands with the features
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
The default build of the plinko crate contains only the trust-model code. Feature tee: constant_time, IprfTee, SwapOrNotSrTee, SwapOrNotTee (moved to plinko/src/iprf/tee.rs), coin_binomial_ct, plinko_hints --constant-time. Feature gpu: the gpu module and bench_gpu_hints (required-features); cuda and parallel imply gpu. Tests of the streaming paths against the hint definition are in hint_gen/path_tests.rs. Verified: tests pass without features (67 lib tests) and with tee,gpu (103 lib tests); the e2e script (built with tee) gives byte-identical fast and CT files; the default binary rejects --constant-time; --features parallel builds. Not verified: a build with --features cuda (no CUDA on this machine) and the Kani workflow change (cargo-kani is not installed). The gpu.rs doctest still fails under --ignored (TASK-21).
<!-- SECTION:FINAL_SUMMARY:END -->
