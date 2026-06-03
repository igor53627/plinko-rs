# README Split Design (2026-01-29)

## Goal
Shrink `README.md` to a concise entry point while preserving key information via subdocs. Keep an overview of the protocol and a short artifact summary in README, and move deep dives (verification, benchmarks, update strategy, schema details) into focused documents.

## Approach
- Keep README for quickstart + high-level context.
- Move long-form sections into subdocs with clear links.
- Add a protocol overview in README with top-5 features.

## Proposed Doc Layout

1) `docs/protocol_overview.md`
- Short protocol explanation.
- Top-5 features (A/B/C/D/G).
- Phase diagram: Extract → HintGen → Query/Update.

2) `docs/data_format.md`
- v3 40-byte schema layout.
- Mapping formats, code_store, metadata.
- Mainnet dataset sizes + R2 links.

3) `docs/verification.md`
- Formal verification overview.
- Rocq/Kani/proptest commands.

4) `docs/benchmarks.md`
- CPU/TEE benchmark table + links.
- GPU benchmark references (existing docs).

5) `docs/update_strategy.md`
- Incremental update algorithm.

Existing docs:
- `docs/hint_generation.md` stays as-is.

## README Outline
- Intro / Why
- Usage
- Output artifacts (short summary + link to `docs/data_format.md`)
- Protocol overview (top-5 features)
- Docs section with links to new subdocs + existing ones
- References

## Notes
- Avoid emojis.
- Keep README under ~150 lines.
