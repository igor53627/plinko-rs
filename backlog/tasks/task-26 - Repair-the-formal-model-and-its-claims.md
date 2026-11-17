---
id: TASK-26
title: Repair the formal model and its claims
status: To Do
assignee: []
created_date: '2026-11-17T11:36:00Z'
labels:
  - formal
  - docs
dependencies: []
references:
  - plinko/formal/specs/SwapOrNotSrSpec.v
  - plinko/formal/linking/IprfLink.v
  - plinko/formal/specs/BinomialSpec.v
  - plinko/formal/sims/HintInitSim.v
  - docs/Plinko.v
  - .github/workflows/formal.yml
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The axiom sr_base_case_no_recurse in SwapOrNotSrSpec.v is inconsistent with a proved bijection; False can be derived. linking/IprfLink.v declares the Rust functions as Parameters with axioms that say Rust equals the spec, so nothing links the proofs to the code. BinomialSpec.v specifies a formula that the Rust code never used, and the code now uses the coin sampler. HintInitSim.v uses one iPRF without a key for all blocks, and its backup theorem is Admitted. docs/Plinko.v models 3 of 6 algorithms and is not built in CI. The Kani workflow calls harnesses that do not exist. TRUST_BASE.md is stale.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 No axiom set in plinko/formal derives False; sr_base_case_no_recurse is removed or corrected
- [ ] #2 TRUST_BASE.md lists each Axiom, Parameter and Admitted with its meaning, from a script that CI runs
- [ ] #3 The PMNS model matches the coin sampler (bit count of a keystream), or the documents state that the model differs from the code
- [ ] #4 Rust comments and documents do not claim that the code is verified; they state that the proofs are about a model
- [ ] #5 CI fails when a formal target fails, and the Kani workflow names only harnesses that exist
- [ ] #6 docs/Plinko.v is built in CI or marked as a sketch
<!-- AC:END -->
