---
id: TASK-16
title: Derive block subsets from a PRF median cutoff
status: Done
assignee: []
created_date: '2026-11-17T08:49:00Z'
updated_date: '2026-11-17T09:25:00Z'
labels:
  - protocol
  - hint-gen
  - memory
dependencies: []
references:
  - plinko/src/protocol.rs
  - plinko/src/bin/hint_gen/subsets.rs
  - plinko/src/bin/hint_gen/bitset.rs
  - https://dl.acm.org/doi/pdf/10.1145/3658644.3690266
  - https://github.com/keewoolee/rms24-plinko-spec
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Hints store block subsets explicitly: 59 GB of bitsets in the generator and about 1.9 TB of block lists in the client at mainnet (computed). Replace them with the RMS24 representation (CCS 2024, section 3.2), adapted to Plinko as in keewoolee/rms24-plinko-spec: v[j,k] = PRF(select, j, k), a 32-bit cutoff per hint, block k is in the subset when v[j,k] < cutoff. A regular hint cutoff selects c/2+1 blocks and a backup hint cutoff selects c/2 blocks (Plinko Fig. 7). A cutoff of 0 marks a hint with a tie at the boundary as invalid, as in S3PIR and rms24-rs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 One shared library module derives select values and cutoffs; plinko_hints and protocol.rs both use it
- [x] #2 Regular subsets have exactly c/2+1 blocks and backup subsets exactly c/2 blocks for every valid hint
- [x] #3 Generator and client keep 4 bytes of subset state per hint; explicit bitsets and block lists are removed
- [x] #4 A hint with a tie at the cutoff boundary is marked invalid and is never used for a query or an update
- [x] #5 Fast path and constant-time path give identical parities on a domain above 1024 hints
- [x] #6 Client query, backup promotion and update tests pass with the new subsets
- [x] #7 docs/hint_generation.md, docs/constant_time_mode.md and CHANGELOG.md describe the new derivation
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Module plinko::subsets: v[j][k] is word k%4 of AES-128(key, hint_be64 || (k/4)_be64), key = SHA-256(master_seed || "plinko_select")[0..16]. Hint IDs are global (backup b = num_regular + b). Values are 32 bits; a tie at the boundary gives cutoff 0 (invalid hint), probability about c/2^33 per hint (checked by simulation at 12 bits), about 55 hints at mainnet.

Tests: find_cutoff boundaries and ties; an AES test vector from openssl; exact subset sizes for c in 2..1000; fast and constant-time paths call the real process_entries_* functions and are compared with a per-hint definition that uses forward only, on 32 and on 1280 hints; invalid hints keep zero parities in both paths. The client has no test with a real tie; an invalid hint is an empty slot or an unavailable backup by construction.

Measured at c = 27988, hardware AES, one thread: contains 1.4 ns, cutoff 44 us per hint, about 740 s for 16,777,216 hints (extrapolated).

Two existing tests depended on their seed and got new seeds: test_ct_hintinit_single_nonzero_entry and test_update_through_complemented_promoted_slots. A scratch run of the second scenario passed its value checks for all 256 one-byte master seeds.

Removed: BlockBitset, the memory_bench benchmark, compute_*_blocks, block_in_subset, derive_subset_seed. The criterion dev-dependency is now unused and was left in place.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Block subsets are one 32-bit cutoff per hint (plinko::subsets), shared by plinko_hints and the protocol client. Full test suite, clippy, fmt, the local e2e script, and a CodeRabbit review (0 findings) pass.
<!-- SECTION:FINAL_SUMMARY:END -->
