---
id: TASK-26
title: Repair the formal model and its claims
status: Done
assignee: []
created_date: '2026-11-17T11:36:00Z'
updated_date: '2026-11-17T16:28:00Z'
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
- [x] #1 No axiom set in plinko/formal derives False; sr_base_case_no_recurse is removed or corrected
- [x] #2 TRUST_BASE.md lists each Axiom, Parameter and Admitted with its meaning, from a script that CI runs
- [x] #3 The PMNS model matches the coin sampler (bit count of a keystream), or the documents state that the model differs from the code
- [x] #4 Rust comments and documents do not claim that the code is verified; they state that the proofs are about a model
- [x] #5 CI fails when a formal target fails, and the Kani workflow names only harnesses that exist
- [x] #6 docs/Plinko.v is built in CI or marked as a sketch
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Removed the axiom sr_base_case_no_recurse (it contradicted the proven Sometimes-Recurse bijection); sr_base_case_equiv takes the condition as a hypothesis. BinomialSpec.v now models the coin sampler of binomial.rs (set coins among the first count coins of an abstract keystream); IprfSpec.v uses the symbol node_coin in place of prf_eval and encode_node; the proofs build. Removed TrueBinomialSpec.v and its axiom binomial_theorem_Z (model of a sampler that the code no longer has). scripts/trust_base.py writes docs/TRUST_BASE.md from the .v files and docs/trust_base_notes.toml (46 items, each with class and meaning, plus the known differences between the models and the code); make verify-axioms fails on an item with no note, a note with no item, or a stale file. proofs/Assumptions.v prints the assumptions of the main theorems: the SR bijection, the iPRF theorems and the regular-hint theorem use symbols only. Documents, file headers of sims/ and linking/, and the db.rs comment state that the proofs are about models. docs/Plinko.v is marked as a sketch and make sketch compiles it. CI: formal.yml runs the trust base check and the sketch; the Rocq-of-Rust job (continue-on-error) is now the manual workflow rocq-of-rust.yml. The Kani workflow already named only the existing harness. Not done: the consistency of the remaining axioms is an argument by hand (a model per axiom in the notes), not a checked proof; HintInitSim.v still has one iPRF with no key and an Admitted backup theorem; SwapOrNotSim.v and linking/ are unchanged apart from the status headers. Checks: clean make all, verify-axioms, sketch on Rocq 9.3.0 (CI uses 9.0.0, not run); six negative checks of the script; cargo test; CodeRabbit (4 minor: 3 fixed, 1 about the cache of the moved CI job not changed).
<!-- SECTION:FINAL_SUMMARY:END -->
