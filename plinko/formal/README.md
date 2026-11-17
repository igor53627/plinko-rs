# Plinko Formal Models

Rocq models of the Plinko algorithms, with proofs of structural properties.

**The proofs are about hand-written models, not about the Rust code.** No tool
translates the Rust source into these models. The evidence for the behaviour of
the Rust code is the Rust tests. [`docs/TRUST_BASE.md`](docs/TRUST_BASE.md)
lists what is proven, each assumption, and the known differences between the
models and the code.

## Structure

```
formal/
  specs/                - Models
    CommonTypes.v       - Shared integer helpers
    DbSpec.v            - Model of derive_plinko_params (db.rs)
    BinomialSpec.v      - Coin sampler: set bits among the first `count` coins
    SwapOrNotSpec.v     - Swap-or-Not PRP on the domain 2^64
    SwapOrNotSrSpec.v   - Sometimes-Recurse PRP on a domain of any size
    IprfSpec.v          - PMNS tree walk and iPRF (PRP, then PMNS)

  proofs/               - Properties of the models
    DbProofs.v          - Invariants of the parameter function
    SwapOrNotProofs.v   - Bijection of Swap-or-Not
    PrpBridge.v         - Swap-or-Not as the PRP on the domain 2^64
    IprfProofs.v        - Inverse and partition properties of the iPRF
    IprfConcrete.v      - The same properties on the domain 2^64
    Assumptions.v       - Prints the assumptions of the main theorems

  sims/                 - Second-level models with Rust names (model only)
    SimTypes.v          - State records
    SwapOrNotSim.v      - One-level Swap-or-Not with an abstract AES
    IprfSim.v           - iPRF state and its relation to IprfSpec.v
    HintInitSim.v       - HintInit: streaming parity against batch parity

  linking/              - Sketch, not built by `make all`
    types.v             - Type links for RocqOfRust
    IprfLink.v          - Placeholders for Rust functions

  scripts/trust_base.py - Writes and checks docs/TRUST_BASE.md
  docs/TRUST_BASE.md    - Generated list of assumptions
  docs/trust_base_notes.toml - Class and meaning of each assumption
  Makefile
```

`../../docs/Plinko.v` is an older sketch. `make sketch` compiles it; nothing
here uses it.

## Building

Prerequisite: Rocq 9.x (`rocq` in the path). Python 3.11 or later for the
trust base script.

```bash
make all            # specs, sims, proofs
make verify-axioms  # fail if docs/TRUST_BASE.md is out of date or an item has no note
make trust-base     # write docs/TRUST_BASE.md
make sketch         # compile ../../docs/Plinko.v in artifacts/sketch
make audit          # print axioms, parameters and admits
```

`make linking` and `make translate` need RocqOfRust and are not part of
`make all`. The linking files contain placeholders and no translated Rust code.

## What Is Proven

For the models, with the assumptions of `docs/TRUST_BASE.md`:

- **Swap-or-Not** (`SwapOrNotProofs.v`): each round is its own inverse; forward
  and inverse are mutual inverses; forward is a bijection on `[0, 2^64)`.
- **Sometimes-Recurse** (`SwapOrNotSrSpec.v`): forward and inverse are mutual
  inverses on `[0, n)` for each `n > 0`.
- **PMNS and iPRF** (`IprfSpec.v`, `IprfProofs.v`): `forward x` is a bin below
  the range; `inverse y` is the set of all `x` with `forward x = y`; the inverse
  sets of different bins are disjoint and cover the domain. This holds for all
  coins and all round functions.
- **HintInit model** (`HintInitSim.v`): for a regular hint, the streaming
  parity equals the batch parity.
- **Parameter function** (`DbProofs.v`): the block size is a power of two and
  the capacity is not less than the number of entries.

## What Is Not Proven

- Any property of the Rust code.
- Correctness of the protocol (`Recon` returns `D[i]`) and privacy of a query.
- Any statement about a distribution: pseudorandomness of the PRP, or the
  binomial distribution of the bin loads.
- The HintInit theorem for backup hints (`Admitted`).

## Rules For A Change

- Do not add an `Axiom` or an `Admitted`.
- After a change to a `Parameter`, or to an existing `Axiom` or `Admitted`,
  update its note in `docs/trust_base_notes.toml` and run `make trust-base`.
  CI runs `make verify-axioms`.
- Build with `make all` before a commit.
