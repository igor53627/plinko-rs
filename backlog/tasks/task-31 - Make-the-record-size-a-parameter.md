---
id: TASK-31
title: Make the record size a parameter
status: To Do
assignee: []
created_date: '2026-11-17T17:41:00Z'
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
- [ ] #1 plinko_hints takes the record size from an option or from the database metadata, and the parity covers the whole record
- [ ] #2 The hint file stores the record size in a new format version; the client rejects a file with another record size; docs/data_format.md describes the format
- [ ] #3 Client, Server, Query and Response of plinko::protocol work with the record size of the database; no 32-byte constant is left in the protocol
- [ ] #4 End-to-end tests from the plinko_hints binary to queries and an update pass for 40-byte records (all bytes of a v3 record) and for one size that is not a multiple of 8
- [ ] #5 The standard path and the constant-time path (feature tee) write byte-identical hint files for each tested size
- [ ] #6 HintInit, query and update costs are measured for 32-byte and for 256-byte records, and docs/protocol_overview.md states them
- [ ] #7 docs/hint_generation.md, docs/data_format.md and the Data Format section of AGENTS.md describe the parameter
<!-- AC:END -->
