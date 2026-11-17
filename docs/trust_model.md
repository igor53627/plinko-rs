# Trust Model

This document states who runs each step of the protocol, who holds each secret,
and what each party learns. It applies to any database that is an array of
fixed-size records; nothing in it is specific to Ethereum state. The full
Ethereum state is not a target of the repository
([protocol_overview.md](protocol_overview.md#scope)); the mainnet numbers below
are a scale reference.

## Decision

**The client runs HintInit and holds the master seed. No other party gets the
seed or the hints.**

This is the model of the paper. Definition 5.1 writes `HintInit^D` for "an
algorithm that is run by the client that requires a streaming pass over the
database D", and the privacy game (Fig. 5) gives the adversary the queries and
nothing of the client state. The paper names one alternative, HintInit by the
server under fully-homomorphic encryption (Section 5.1); this repository does
not implement it.

## Parties

| Party  | Runs                                                      | Holds                                               |
|--------|-----------------------------------------------------------|-----------------------------------------------------|
| Client | HintInit (`plinko_hints`), Query, Recon, UpdateHint       | Master seed, hint file, cache of answers            |
| Server | Publishes `database.bin` and updates, Answer              | `database.bin`                                      |

There is no third party. A machine that the client rents to run HintInit holds
the master seed and is thus part of the client (see
[Alternatives](#alternatives)).

## Secrets

| Secret            | Where it is                                 | What a holder can do                                                          |
|-------------------|---------------------------------------------|-------------------------------------------------------------------------------|
| Master seed       | `--seed-file` of `plinko_hints`, the client | Compute all block keys and all block subsets, and so the set of each hint     |
| Hint file         | `--output` of `plinko_hints`, the client    | Client-secret; the parities and cutoffs belong to one client                  |
| PMNS level cache  | Client memory (29 MB at mainnet, computed)  | Derived from the block keys; as secret as the master seed                     |

A party that holds the master seed and sees the queries can find the queried
index of each query. For one offset of the query it computes the iPRF inverse
of the block, which gives about 256 candidate hints; the other offsets select
one hint; the block of that hint that is not in the query is the queried block.
This is a conclusion from the hint structure; there is no attack code in the
repository.

The hint file does not contain the master seed. It has a seed check value
(`plinko::hints::seed_check`), so that the client rejects a file of another
seed.

## What Each Party Learns

**The client** downloads the full database for HintInit, so it can read every
record at that time. After HintInit it needs only the hint file and the seed.

**The server** learns:

- That a client downloaded `database.bin`. The file is the same for all clients.
- The time and the number of the queries of a client.
- For each query: one bit mask over the blocks and `c/2` offsets
  (`plinko::protocol::Query`). It returns two parities.

By the privacy theorem of the paper the queries do not reveal the queried
index to the server, on these conditions:

- The master seed and the hints stay with the client.
- The client makes at most `Q` queries with one hint set. `Q` is the number of
  backup hints (8,388,608 at mainnet with lambda = 128). After that the client
  gets `NoBackupHintsLeft` and must run HintInit again.
- The primitives are pseudorandom. See "PRP Deviations From An Ideal
  Permutation" in [hint_generation.md](hint_generation.md).

**An observer of the network** sees the same as the server if the transport is
not encrypted. The repository has no transport.

## Not Covered

These points are outside the privacy claim today.

- **Key to index.** PIR takes an index. The step from a key to an index needs
  a full map from keys to indices (91 GB of mapping files for the mainnet test
  data) or a message that reveals the key (TASK-22).
- **Wrong answers.** The games of the paper compute the answer with the honest
  `Answer` algorithm. The client cannot check an answer, and it has no
  commitment of the database to compare with (TASK-24).
- **Database version.** An answer and an update carry no version of the
  database (TASK-23).
- **Test data from the extractor.** The extractor does not read one consistent
  state; see [data_format.md](data_format.md#limits-of-the-extractor). This is
  not planned.
- **Online protocol.** [online_protocol.md](online_protocol.md) lists the open
  points: the query format with shared offsets has an argument and no proof, an
  answer has no database version, `abandon_query` removes a hint with no
  replacement, and there is no hint refresh before `Q` queries.
- **Traffic analysis.** The time and the number of queries are visible.

## Cost For The Client

Mainnet v3: 1,834,095,877 records, w = 65536, c = 27988, lambda = 128. Details
and measurement conditions: [hint_generation.md](hint_generation.md#mainnet-cost).

| Item                               | Cost                                         | Source                                   |
|------------------------------------|----------------------------------------------|------------------------------------------|
| Download of `database.bin`         | 73.4 GB (73,363,835,080 bytes)               | `data_format.md`                         |
| HintInit, CPU                      | About 300 core-hours                         | Extrapolated from 24-block runs          |
| HintInit, wall time on 24 cores    | About 13 hours                               | Extrapolated from the 24-thread run      |
| HintInit, wall time on 8 cores     | About 37 hours                               | Extrapolated from the 8-thread run       |
| HintInit, memory                   | 1.8 GB plus the mapped database pages        | Measured, 24 blocks, mainnet hint count  |
| Hint file                          | 872 MB (872,415,328 bytes)                   | Computed                                 |
| Update of the hints                | 0.52 ms CPU per changed record               | Measured (`inverse`, one thread)         |
| Query, client side                 | About 0.4 s CPU                              | Computed: 13,994 `forward` calls, 28.5 us each |

The measurements are from an Apple M3 Ultra with hardware AES. No mainnet run
was done. A build without hardware AES is about 20 times slower.

`plinko_hints` maps `database.bin` from disk. It does not read the database
from a network stream, so the client needs the disk space for the file until
HintInit ends.

## Alternatives

| HintInit is run by                    | Seed is held by   | Privacy against the server                                   | Status                         |
|---------------------------------------|-------------------|--------------------------------------------------------------|--------------------------------|
| Client                                | Client            | Yes (paper model)                                            | This design; implemented       |
| Server, in the clear                  | Server            | None                                                         | Rejected                       |
| Rented machine or GPU cloud           | Its operator      | Only if the operator does not give the seed to the server    | Not part of the design         |
| TEE at the server or at a cloud       | The enclave       | Only if the enclave and its side channels hold               | Not part of the design         |
| Server under FHE (paper, Section 5.1) | Client            | Yes                                                          | Not implemented                |

The rented machine and the TEE add a trust assumption that single-server PIR
is meant to remove.

### TEE Mode

`plinko_hints --constant-time` is experimental and is not part of this trust
model. The default build does not contain it; it needs the `tee` build feature
([FEATURE_FLAGS.md](FEATURE_FLAGS.md)). A design in which a TEE runs HintInit needs all of these, and the
repository has none of them:

1. Remote attestation of the binary that runs in the enclave.
2. A channel that brings the master seed of the client into the enclave and is
   bound to the attestation.
3. An export of the hint file that only the client can read.
4. Code without secret-dependent time and memory access. The current path
   reads the hints at the secret hint index and divides by key-derived values
   (TASK-20).
5. A cost that is practical. The path takes 16.7 ms per record on one thread
   and has no worker threads: about 8,500 core-hours, or 355 days, for mainnet
   (computed from the per-record measurement).

A client can use the mode on its own machine. It gives the same hint file as
the standard path.

### GPU Path

`bench_gpu_hints` and `plinko/cuda/` are a throughput benchmark and are not
part of this trust model. The default build does not contain them; they need
the `gpu` or the `cuda` build feature. The output is not a Plinko hint set: the block keys
are public, there are two fixed block subsets, and there are no backup hints
(TASK-21). A GPU path that writes the same hint file as `plinko_hints` fits
this model only on hardware that the client controls.
