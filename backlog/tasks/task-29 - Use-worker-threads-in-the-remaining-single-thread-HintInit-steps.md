---
id: TASK-29
title: Use worker threads in the remaining single-thread HintInit steps
status: Done
assignee: []
created_date: '2026-11-17T12:40:00Z'
updated_date: '2026-11-18T04:15:00Z'
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
- [x] #1 init_hints computes the cutoffs on the worker threads, and the cutoffs are equal to the one-thread result
- [x] #2 protocol::compute_hints uses worker threads, and its HintSet is equal to the one-thread result
- [x] #3 plinko_hints and protocol::compute_hints share one streaming implementation, or the document states why they do not
- [x] #4 The constant-time path uses worker threads, or docs/constant_time_mode.md states why it must not (shared accumulators are read and written at the secret hint index)
- [x] #5 Wall time of a 24-block run at w = 65536 and 16,777,216 hints is measured before and after, cutoff phase included
- [x] #6 docs/hint_generation.md states the new mainnet estimate
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Commit bbafc210 (local jj, not pushed).

Done:
- HintInit is the library module plinko::hint_init (hint_cutoffs, fill_parities, and fill_parities_ct with the tee feature). plinko_hints and protocol::compute_hints call the same code; the second streaming loop (protocol.rs) and the second derive_block_keys (hint_gen/keys.rs) are removed. Note: driver::init_hints of the task text was named hint_cutoffs since TASK-31.
- Cutoffs on worker threads. At c = 27,988 for 262,144 hints: 10.58 s on 1 thread, 1.48 s on 8, 0.51 s on 24, equal cutoffs.
- protocol::compute_hints and Client::offline_init use all available cores; compute_hints_with_threads takes the thread count. 24 blocks of w = 4096: 47.7 s on 1 thread, 2.36 s on 24, equal hint sets.
- The constant-time path accepts --threads; its default stays 1 thread. 24 blocks of w = 1024: 294.6 s on 1 thread, 14.3 s on 24, the same file as the standard path.
- 24-block run at w = 65536 and 16,777,216 hints, 24 threads, before and after: 42.5 s and 41.4 s of wall time; cutoff phase 2.03 s and 0.20 s. The hint file is byte-identical to the file of the code before the change.
- Mainnet estimate in docs/hint_generation.md: the cutoff phase goes from about 11 minutes to about 32 s on 24 threads (extrapolated); the whole HintInit stays at about 13 hours on 24 cores.

Limits:
- The gain for plinko_hints at the mainnet geometry is the cutoff phase only.
- The timing effect of worker threads in the constant-time path (contention on a parity word depends on the hint indices) is not measured; it is in the notes of TASK-20.
- The constant-time path now holds the parity accumulators in addition to the hint set.
- The timings of hint_cutoffs and compute_hints come from scratch programs outside the repository. One machine (M3 Ultra, hardware AES), one run each.
- Not run: CI, a cuda build, SEV-SNP, a mainnet-size run.
<!-- SECTION:FINAL_SUMMARY:END -->
