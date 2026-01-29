# Verification

The `plinko/formal/` directory contains Rocq (Coq) specifications and proofs aligned with the Plinko paper.

## Rocq Specs

- `SwapOrNotSpec.v`, `SwapOrNotSrSpec.v`: Swap-or-Not PRP and Sometimes-Recurse wrapper
- `IprfSpec.v`: Invertible PRF combining PRP + PMNS
- `BinomialSpec.v`, `TrueBinomialSpec.v`: Binomial sampling specifications

## Rocq Proofs

- `SwapOrNotProofs.v`: Round involution, forward/inverse identity, bijection
- `IprfProofs.v`: iPRF partition property, inverse consistency

## Trust Base (intentional axioms)

- Crypto: AES-128 encryption, key derivation properties
- Math: `binomial_theorem_Z`
- FFI: Rust-to-Rocq refinement axioms (verified via proptest)

## Kani

```bash
cd plinko && cargo kani --tests
```

## Proptest

```bash
cd plinko && cargo test --lib kani_proofs
```

## Rocq

```bash
cd plinko/formal && make
```
