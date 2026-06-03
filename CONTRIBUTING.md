# Contributing

## Workflow

Do not push feature fixes directly to `main`. Use a short-lived branch and open a pull request.

```bash
git checkout main && git pull
git checkout -b <type>/<short-description>   # e.g. fix/binomial-cdf, docs/readme

# ... edit ...
cargo fmt -p plinko
cargo test --manifest-path plinko/Cargo.toml

git push -u origin HEAD
gh pr create --base main
```

The PR template (`.github/pull_request_template.md`) asks for **Summary** and **Testing** sections with real newlines, not literal `\n`.

## Commit messages

[Conventional Commits](https://www.conventionalcommits.org/) are enforced on PRs (see `.github/workflows/commit-lint.yml`).

Examples: `fix(plinko): ...`, `test(binomial): ...`, `docs: ...`, `style(plinko): ...`

Enable local hooks (optional but recommended):

```bash
./scripts/setup-hooks.sh
```

## CI checks (pull requests)

| Check | Workflow |
|-------|----------|
| Format | `ci.yml` — `cargo fmt --check` |
| Test | `ci.yml` |
| Clippy | `ci.yml` |
| Conventional Commits | `commit-lint.yml` |
| Local E2E Smoke | `local-e2e-smoke.yml` (PRs) |

Formal verification (`formal.yml`) runs on PRs that touch `plinko/formal/` or related paths.

## Branch protection

`main` requires pull requests and passing checks: **Format**, **Test**, **Clippy**, **Conventional Commits**.

Config reference: [`.github/branch-protection-main.json`](.github/branch-protection-main.json) (for admins re-applying). If the REST API rejects `required_approving_review_count: 0`, use branch settings in the GitHub UI (current `main` uses PRs + checks, zero required approvals).

## Rust / Plinko specifics

- Implementation specs: `docs/hint_generation.md`, `docs/data_format.md`, `docs/Plinko.v` (paper JSON index is auxiliary)
- Hint generation and TEE: `docs/hint_generation.md`, `docs/constant_time_mode.md`
- Changing hint/iPRF/binomial semantics: align with paper + `plinko/formal/` specs (see `AGENTS.md`)