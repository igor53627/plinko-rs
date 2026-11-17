---
id: TASK-17
title: Write hints to disk and load them in the client
status: Done
assignee: []
created_date: '2026-11-17T09:49:00Z'
updated_date: '2026-11-17T10:11:00Z'
labels:
  - protocol
  - hint-gen
  - format
dependencies: []
references:
  - plinko/src/bin/plinko_hints.rs
  - plinko/src/protocol.rs
  - docs/data_format.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
plinko_hints computes the hints and drops them, and the protocol client can only build hints from an in-memory database. Close the pipeline: a versioned hint file with the geometry in its header, a writer in plinko_hints, a loader in the client, and an end-to-end test from the generator binary to client queries.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A library module defines the hint set and its on-disk format (magic, version, geometry, hint counts, seed check, checksum); generator and client share it
- [x] #2 plinko_hints --output writes the hint file; the master seed comes from --seed or --seed-file and is never stored in the hint file
- [x] #3 The client loads a hint file and rejects a wrong seed, a geometry mismatch, an unknown version, truncation, and corruption
- [x] #4 The hint file of the generator equals the hint set that the client computes in memory for the same seed and database
- [x] #5 An end-to-end test runs the plinko_hints binary, loads the file, and gets correct answers for queries and for a query after an update
- [x] #6 The same end-to-end scenario passes with more than 65536 hints
- [x] #7 Fast and constant-time runs with the same seed give byte-identical hint files in scripts/local_e2e_hint_pipeline.sh
- [x] #8 docs/hint_generation.md, docs/data_format.md, docs/ARCHITECTURE.md and CHANGELOG.md describe the file and the commands
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Format PLNKHINT version 1 (plinko::hints): 64-byte header (magic, version, parity size, num_entries, block_size, num_blocks, num_regular, num_backup, seed check), 36-byte regular records, 68-byte backup records, BLAKE3 of all earlier bytes. The seed check is SHA-256(master_seed || "plinko_seed_check")[0..8]. Mainnet size: 872,415,328 bytes (computed).

Generator: --output writes through <output>.tmp and rename, mode 0600 on Unix; --seed-file reads 64 hex chars; --output requires --seed or --seed-file and is refused with --allow-truncation.

Client: compute_hints (in-memory HintInit) + Client::from_hints; offline_init is their composition, so all earlier protocol tests go through from_hints. from_hints does not recompute cutoffs or parities.

Tests: hints.rs unit tests (documented offsets, roundtrip, each flipped bit, each truncation, trailing data, header fields); hint_file_e2e_test (binary -> file == compute_hints, queries, update, re-query; mismatch rejections; no output without a seed). Mutation check: swapped backup parities in fast_path made the e2e test fail. The ignored test with 66,560 hints (w = 512, lambda = 65, 1000 entries) passed in 122 s in release with software AES. The fast/CT byte comparison in the script uses 4 hints; ct_path unit tests compare parities on 1280 hints.

Not done: the client model keeps 32-byte entries and has no loader for database.bin; a mainnet-scale run was not made.

After the CodeRabbit review: the temporary file is <output>.<pid>.tmp and is created with create_new, so two runs cannot share it and an existing file cannot keep a permissive mode.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
plinko_hints --output writes a versioned hint file (plinko::hints), and Client::from_hints loads it. The end-to-end test from the binary to client queries passes, also with 66,560 hints. Full test suite, clippy, fmt, the local e2e script, and a CodeRabbit review (2 findings fixed, then 0) pass.
<!-- SECTION:FINAL_SUMMARY:END -->
