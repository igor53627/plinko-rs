---
id: TASK-22
title: Let the client find the index of a key privately
status: To Do
assignee: []
created_date: '2026-11-17T11:36:00Z'
labels:
  - data-layer
  - design
dependencies: []
references:
  - src/main.rs
  - plinko/src/schema40.rs
  - docs/data_format.md
  - https://github.com/keewoolee/rms24-plinko-spec
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PIR works on an integer index. The step from an address or an (address, slot) pair to an index is in account-mapping.bin (8.44 GB) and storage-mapping.bin (83.02 GB), 91.46 GB together and more than database.bin. No code reads these files. The client must download them or tell a server its key, which defeats PIR. The TAG field is named Cuckoo TAG, but there is no cuckoo table. keewoolee/rms24-plinko-spec (pir/keyword_pir) and ~/pse/rms24-rs (src/keyword_pir) have a cuckoo keyword PIR.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A design document selects the layout (for example cuckoo hashing by key hash) and gives the number of PIR queries per lookup
- [ ] #2 The client computes the candidate indices of a key without the mapping files and without a message that contains the key
- [ ] #3 The extractor or a build tool writes database.bin in that layout
- [ ] #4 A test on a synthetic database goes from an address and from an (address, slot) pair to the correct value
- [ ] #5 The client checks the TAG of the answer against its key
<!-- AC:END -->
