(** Assumptions.v - Prints the assumptions of the main theorems

    [make all] compiles this file, and the build log then shows, for each
    theorem below, every [Axiom] and [Parameter] that its proof uses.
    docs/TRUST_BASE.md gives the class and the meaning of each of them.

    This file proves nothing.
*)

Require Import Plinko.Specs.SwapOrNotSrSpec.
Require Import Plinko.Specs.IprfSpec.
Require Import Plinko.Proofs.SwapOrNotProofs.
Require Import Plinko.Proofs.IprfProofs.
Require Import Plinko.Sims.HintInitSim.
Require Import Plinko.Sims.SwapOrNotSim.

(** Sometimes-Recurse is a bijection. *)
Print Assumptions sr_forward_inverse_id.
Print Assumptions sr_inverse_forward_id.

(** Swap-or-Not on the domain 2^64 is a bijection. *)
Print Assumptions forward_is_bijection.

(** iPRF: the inverse of a bin is the set of its preimages; the sets partition the domain. *)
Print Assumptions iprf_inverse_contains_preimage_proof.
Print Assumptions iprf_inverse_consistent_proof.
Print Assumptions iprf_inverse_partitions_proof.
Print Assumptions iprf_inverse_disjoint_proof.

(** HintInit model: streaming parity equals batch parity (regular, then backup). *)
Print Assumptions hint_init_streaming_eq_batch.
Print Assumptions hint_init_backup_streaming_eq_batch.

(** One-level Swap-or-Not with the abstract AES. *)
Print Assumptions rust_forward_is_bijection.
