---
id: TASK-19
title: Decide who runs HintInit and write the trust model
status: In Progress
assignee: []
created_date: '2026-11-17T11:36:00Z'
updated_date: '2026-11-17T12:51:00Z'
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
- [x] #1 A document states who runs HintInit, who holds the master seed, and what each party learns
- [x] #2 The document gives the HintInit cost for that party from measured numbers on the current code
- [x] #3 If a TEE runs HintInit: attestation, the seed channel into the TEE, and the hint export are specified
- [x] #4 The TEE mode and the GPU path are each marked as part of the design or as not part of it
- [x] #5 README.md, docs/DEPLOYMENT.md and docs/constant_time_mode.md agree with the document
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
docs/trust_model.md is written with the paper model as the decision: the client runs HintInit and holds the master seed; the TEE mode and the GPU path are experimental and not part of the model. AC 3 is met as a list of what a TEE design needs and does not have. The owner has not confirmed the decision yet; the task stays In Progress until then.
<!-- SECTION:NOTES:END -->
