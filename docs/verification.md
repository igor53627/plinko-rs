# Verification

Doc map: [`README.md`](README.md).

**The Rust code is not formally verified.** The Rocq files are hand-written
models of the algorithms. No tool translates the Rust source into them, and no
proof contains a term of the Rust code. The evidence for the behaviour of the
Rust code is the Rust tests.

## Rocq Models

Directory: `plinko/formal/`. Overview: [`../plinko/formal/README.md`](../plinko/formal/README.md).

- `SwapOrNotSpec.v`, `SwapOrNotSrSpec.v`: Swap-or-Not PRP and the Sometimes-Recurse wrapper
- `BinomialSpec.v`: the coin sampler of the PMNS tree (set bits among the first `count` coins of a keystream; the keystream is abstract)
- `IprfSpec.v`: PMNS tree walk and iPRF (PRP, then PMNS)
- `sims/HintInitSim.v`: HintInit, streaming parity against batch parity

Proven for the models: the PRPs are bijections; the iPRF inverse of a bin is
the set of all inputs that map to the bin; the inverse sets are disjoint and
cover the domain; for a regular hint the streaming parity equals the batch
parity. These statements hold for all coins and all round functions, so they
say nothing about a distribution.

Not stated in any file: correctness of the protocol, privacy of a query,
pseudorandomness, constant-time behaviour.

## Trust Base

[`../plinko/formal/docs/TRUST_BASE.md`](../plinko/formal/docs/TRUST_BASE.md)
lists each `Axiom`, `Parameter` and `Admitted` with its class and meaning, and
the known differences between the models and the Rust code. A script writes the
file; `make verify-axioms` fails when the file is out of date.

`docs/Plinko.v` is an older sketch (three of the six algorithms of Fig. 7, four
lemmas with no proof). `make sketch` compiles it.

## Rocq

```bash
cd plinko/formal
make all            # build the models and the proofs
make verify-axioms  # check docs/TRUST_BASE.md
make sketch         # compile docs/Plinko.v
```

## Kani

One harness, for the coin mask of the constant-time sampler (feature `tee`):

```bash
cd plinko && cargo kani --features tee --harness proof_top_bits_mask_keeps_leading_bits
```

## Proptest

```bash
cd plinko && cargo test --lib --features tee kani_proofs
```
