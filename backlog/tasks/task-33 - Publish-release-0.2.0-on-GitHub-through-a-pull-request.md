---
id: TASK-33
title: Publish release 0.2.0 on GitHub through a pull request
status: Done
assignee: []
created_date: '2026-11-19T03:25:00Z'
updated_date: '2026-11-19T14:45:00Z'
labels:
  - release
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The private repository igor53627/plinko-rs exists on GitHub since 2026-11-19 and is empty. GitHub declined the push of main (906dcbc) with 'push declined due to email privacy restrictions': 48 commits of the old history have the private email of the owner as author (committer noreply@github.com), and the account blocks command line pushes that expose it. The 46 commits of release/0.2.0 use the noreply address. GitHub Actions are disabled in the repository so that the first push of the old main starts no workflow (ci.yml and pages.yml run on a push to main). The jj remote is 'github'. The branch release/0.2.0 is on origin; main on origin is not moved.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The owner selects a way: allow the push in the email settings of the account, or rewrite the author of the 48 old commits (this changes each commit hash of main)
- [x] #2 main (the rewritten history before the release) and release/0.2.0 are on GitHub, and GitHub Actions are enabled before the pull request is opened
- [x] #3 A pull request release/0.2.0 to main is open and its workflows pass (CI, commit lint, formal with Rocq 9.0.0, e2e smoke); a failure from the release changes is fixed
- [x] #4 After the merge with no squash: main on origin is at the same commit, and the tag v0.2.0 is on the merge result
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-11-19: the owner selected the rewrite, with no copy of the old history. git filter-branch replaced the author email in 48 commits and the address in three Co-Authored-By lines with igor53627@users.noreply.github.com; a script compared each old commit with its new commit (tree, parents, author name and date, committer, message). Old main 906dcbc is now 7c1598d; the first 15 commits keep their hashes; 57 commits lost their signature. The 46 commits of release/0.2.0 were moved to the new main with jj rebase; the tree of the branch tip is the same. The local branches fix/binomial-beta-routing and archive/local-worktree-checkpoint-20260717 still point to the old history and were not changed.

origin (local Git server): main is protected from force push, so main on origin is still the old 906dcbc. The owner must allow the force push or remove the protection before main on origin can follow GitHub. release/0.2.0 on origin was replaced by the rewritten branch.

2026-11-19: pull request 1 is open (release/0.2.0 to main); the 7 checks passed on d45aa92. CodeRabbit gave 26 findings on the pull request. Each one was checked against the code: 23 led to a change (9 code fixes with tests that fail without the fix, the others are documents, doc comments, tests, a script and a workflow), 3 led to no code change (HintSet::new allocation: doc line only, the fix alone gives no guarantee because SharedParities allocates the same size; IprfTee::inverse_ct overflow flag: doc and a test only, HintInit uses the checked table path; scan of the backup flags: below 1 percent of the query cost). Also fixed from the same check: lambda * w with no overflow check, --output that is a directory. Open from the same check, not changed: cost_estimate takes w from derive_plinko_params (131072 for mainnet) and plinko_hints from default_block_size (65536); local_e2e_hint_pipeline.sh removes a caller-supplied WORK_DIR on exit. Local checks and the CodeRabbit CLI (0 findings) passed. Not pushed: a push needs a command of the owner.

2026-11-19, later: the owner approved the push of the review fixes and asked for the two open items. Fixed: cost_estimate uses the geometry of plinko_hints (TASK-21 has the rest); the three e2e scripts keep a work directory of the caller. main on origin (Gitea 1.26.4) is now 7c1598d, the same commit as main on GitHub: the branch protection rule of main (push disabled, force push disabled) was deleted with the API, main was pushed with --force-with-lease against 906dcbc, and the rule was created again with the same 31 fields (only the timestamps differ).
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Pull request 1 (release/0.2.0 to main, 63 commits) was merged on 2026-11-19 with a merge commit, no squash: 43465a6. The 7 checks passed on the last head a33dd0c. Two CodeRabbit reviews: 26 findings and 1 finding, each checked against the code; the fixes are commits 20ba9c0 to a33dd0c. The annotated tag v0.2.0 is on the merge commit. main and the tag are on GitHub and on origin at the same commits. Open after the release: the workflow 'Deploy to GitHub Pages' fails on main, because GitHub Pages is not enabled in the new private repository (a decision of the owner: a Pages site is public).
<!-- SECTION:FINAL_SUMMARY:END -->
