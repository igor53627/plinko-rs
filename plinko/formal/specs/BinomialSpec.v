(** * BinomialSpec: the coin sampler of the PMNS tree

    Model of [coin_binomial] in plinko/src/binomial.rs. A PMNS node splits its
    balls between its two children with Binomial(count, 1/2). The sample is the
    number of set coins among the first [count] coins of the keystream of the
    node.

    The keystream is abstract here: a function from the coin index to a bit.
    The Rust code takes the coins from an AES-128 counter-mode keystream. No
    statement in this file is about AES or about the distribution of the
    sample.
*)

From Stdlib Require Import ZArith.
From Stdlib Require Import Lia.

Open Scope Z_scope.

(** Number of set coins among coins [0 .. count-1]. *)
Fixpoint coin_count (coins : nat -> bool) (count : nat) : Z :=
  match count with
  | O => 0
  | S count' => coin_count coins count' + (if coins count' then 1 else 0)
  end.

(** The sample for [count] balls. *)
Definition coin_sample_spec (coins : nat -> bool) (count : Z) : Z :=
  coin_count coins (Z.to_nat count).

Lemma coin_count_range : forall coins count,
  0 <= coin_count coins count <= Z.of_nat count.
Proof.
  intros coins count.
  induction count as [|count' IH].
  - simpl. lia.
  - cbn [coin_count]. rewrite Nat2Z.inj_succ.
    destruct (coins count'); lia.
Qed.

(** No balls: the sample is 0. *)
Lemma coin_sample_zero : forall coins,
  coin_sample_spec coins 0 = 0.
Proof.
  intros coins. unfold coin_sample_spec. reflexivity.
Qed.

(** The sample is between 0 and the number of balls. *)
Lemma coin_sample_range : forall coins count,
  0 <= count ->
  0 <= coin_sample_spec coins count <= count.
Proof.
  intros coins count Hcount.
  unfold coin_sample_spec.
  pose proof (coin_count_range coins (Z.to_nat count)) as Hrange.
  rewrite Z2Nat.id in Hrange by lia.
  exact Hrange.
Qed.

Lemma coin_count_monotone : forall coins count1 count2,
  (count1 <= count2)%nat ->
  coin_count coins count1 <= coin_count coins count2.
Proof.
  intros coins count1 count2 Hle.
  induction Hle as [|count2' Hle IH].
  - lia.
  - cbn [coin_count]. destruct (coins count2'); lia.
Qed.

(** More balls: the sample does not decrease. *)
Lemma coin_sample_monotone : forall coins count1 count2,
  0 <= count1 ->
  count1 <= count2 ->
  coin_sample_spec coins count1 <= coin_sample_spec coins count2.
Proof.
  intros coins count1 count2 Hcount1 Hle.
  unfold coin_sample_spec.
  apply coin_count_monotone.
  apply Z2Nat.inj_le; lia.
Qed.

Close Scope Z_scope.
