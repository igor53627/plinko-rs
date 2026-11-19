---
id: TASK-33
title: Publish release 0.2.0 on GitHub through a pull request
status: To Do
assignee: []
created_date: '2026-11-19T03:25:00Z'
updated_date: '2026-11-19T06:30:00Z'
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
- [ ] #2 main (the rewritten history before the release) and release/0.2.0 are on GitHub, and GitHub Actions are enabled before the pull request is opened
- [ ] #3 A pull request release/0.2.0 to main is open and its workflows pass (CI, commit lint, formal with Rocq 9.0.0, e2e smoke); a failure from the release changes is fixed
- [ ] #4 After the merge with no squash: main on origin is at the same commit, and the tag v0.2.0 is on the merge result
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-11-19: the owner selected the rewrite, with no copy of the old history. git filter-branch replaced the author email in 48 commits and the address in three Co-Authored-By lines with igor53627@users.noreply.github.com; a script compared each old commit with its new commit (tree, parents, author name and date, committer, message). Old main 906dcbc is now 7c1598d; the first 15 commits keep their hashes; 57 commits lost their signature. The 46 commits of release/0.2.0 were moved to the new main with jj rebase; the tree of the branch tip is the same. The local branches fix/binomial-beta-routing and archive/local-worktree-checkpoint-20260717 still point to the old history and were not changed.

origin (local Git server): main is protected from force push, so main on origin is still the old 906dcbc. The owner must allow the force push or remove the protection before main on origin can follow GitHub. release/0.2.0 on origin was replaced by the rewritten branch.
<!-- SECTION:NOTES:END -->
