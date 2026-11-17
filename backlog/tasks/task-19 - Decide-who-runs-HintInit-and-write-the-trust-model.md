---
id: TASK-19
title: Decide who runs HintInit and write the trust model
status: To Do
assignee: []
created_date: '2026-11-17T11:36:00Z'
labels:
  - design
  - docs
dependencies: []
references:
  - docs/constant_time_mode.md
  - docs/DEPLOYMENT.md
  - README.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The paper lets the client run HintInit, and the keys stay with the client. The repository has a GPU path and a TEE mode that move HintInit to another party, but no document says who holds the master seed. A party that holds the seed and the hints can link each query to its hint, so the client has no privacy against that party. The answer decides if the TEE mode and the GPU path are needed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A document states who runs HintInit, who holds the master seed, and what each party learns
- [ ] #2 The document gives the HintInit cost for that party from measured numbers on the current code
- [ ] #3 If a TEE runs HintInit: attestation, the seed channel into the TEE, and the hint export are specified
- [ ] #4 The TEE mode and the GPU path are each marked as part of the design or as not part of it
- [ ] #5 README.md, docs/DEPLOYMENT.md and docs/constant_time_mode.md agree with the document
<!-- AC:END -->
