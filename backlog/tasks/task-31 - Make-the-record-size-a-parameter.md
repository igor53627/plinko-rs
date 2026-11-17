---
id: TASK-31
title: Make the record size a parameter
status: Done
assignee: []
created_date: '2026-11-17T17:41:00Z'
updated_date: '2026-11-17T18:58:00Z'
labels:
  - core
  - protocol
dependencies: []
references:
  - plinko/src/bin/hint_gen/types.rs
  - plinko/src/protocol.rs
  - plinko/src/hints.rs
  - docs/data_format.md
  - >-
    backlog/decisions/decision-1 -
    Target-a-PIR-library-for-fixed-size-records-not-the-full-Ethereum-state.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
database.bin has 40-byte records (plinko/src/schema40.rs). The hints XOR the first 32 bytes of each record: PARITY_WORD_SIZE = 32 in plinko/src/bin/hint_gen/types.rs, ENTRY_SIZE = 32 in plinko/src/protocol.rs, and 32-byte parities in the hint file of version 1. Bytes 32..40 of a record are outside the parity, so a client cannot read them through PIR. After the scope decision (backlog/decisions/decision-1) the target is a PIR library for a database of fixed-size records; such a library needs the record size as a parameter, and the parity must cover the whole record.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 plinko_hints takes the record size from an option or from the database metadata, and the parity covers the whole record
- [x] #2 The hint file stores the record size in a new format version; the client rejects a file with another record size; docs/data_format.md describes the format
- [x] #3 Client, Server, Query and Response of plinko::protocol work with the record size of the database; no 32-byte constant is left in the protocol
- [x] #4 End-to-end tests from the plinko_hints binary to queries and an update pass for 40-byte records (all bytes of a v3 record) and for one size that is not a multiple of 8
- [x] #5 The standard path and the constant-time path (feature tee) write byte-identical hint files for each tested size
- [x] #6 HintInit, query and update costs are measured for 32-byte and for 256-byte records, and docs/protocol_overview.md states them
- [x] #7 docs/hint_generation.md, docs/data_format.md and the Data Format section of AGENTS.md describe the parameter
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Commit f3360714 (local jj, not pushed).

Done:
- plinko_hints --record-size (1 to 1,048,576 bytes, default 40). The record size comes from the option, not from metadata.json. A hint parity is the XOR of whole records.
- Hint file version 2 with a record_size field; version 1 is not read. Client::from_hints rejects another record size.
- plinko::hints: HintSet with private fields, flat parity buffers and accessors. plinko::protocol: no Entry and no ENTRY_SIZE; the database is a byte slice; records, deltas and parities are Vec<u8>.
- ProtocolParams::new rejects a padded database that does not fit in memory (CodeRabbit finding).
- Tests from the plinko_hints binary: 1, 13, 40, 45 and 256 bytes; the constant-time file is byte-identical for the same five sizes.
- Measured on 16,777,216 records (M3 Ultra, hardware AES, w = 4096, 24 threads), record sizes 32 / 40 / 256 bytes: HintInit 9,664 / 9,596 / 9,687 s CPU; query 53.8 / 53.2 / 53.1 ms; update 0.54 / 0.53 / 0.52 ms; hint file 54.5 / 67.1 / 406.8 MB; answer of the server 0.11 / 0.13 / 0.38 ms. HintInit memory at the mainnet hint count with 40-byte records: 2.11 GB.

Limits:
- The GPU benchmark (feature gpu) and gen_synthetic work with 40-byte records only.
- The formal model has a 32-byte zero entry (noted in TRUST_BASE.md).
- Not run: CI, a cuda build, SEV-SNP, a mainnet-size run. Timings are from one machine, one run for each size.
- The query and update timings come from a scratch program outside the repository.
<!-- SECTION:FINAL_SUMMARY:END -->
