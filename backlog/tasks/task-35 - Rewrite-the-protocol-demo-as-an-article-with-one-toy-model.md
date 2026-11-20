---
id: TASK-35
title: Rewrite the protocol demo as an article with one toy model
status: Done
assignee: []
created_date: '2026-11-20T12:10:00Z'
updated_date: '2026-11-20T12:17:00Z'
labels:
  - docs
dependencies: []
references:
  - docs/protocol-visualization.html
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
After TASK-34 the demo page still had a physics board that did not run the sampler, a toy code of its own in each demo, and no figure for a query or for an update. The owner approved a new style on 2026-11-20: an article on graph paper with figures that run the algorithms.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 All figures use one toy model, and a check on many keys compares it with the definitions
- [x] #2 The page has figures for a query and for an update of a record
- [x] #3 The page loads no external file
- [x] #4 The page has no horizontal scroll at 1280 px and at 390 px, and no console error
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-11-20: commit on the branch docs/tutorial-rewrite, delivered through a pull request. Checks in Chromium on 300 keys: the permutation is a permutation and its inverse returns the input; the iPRF inverse equals a search of all inputs; the streaming HintInit gives the parities of the definition; 2400 queries return the record of the database; after each of 4022 updates all parities of the client agree with the database. The CodeRabbit CLI gave 11 minor findings, all valid, all fixed: figure count in CHANGELOG.md, a new HintInit keeps the database of the server, contrast of a spent hint, the cached value as the result of a repeated read, keyboard focus after a render (4 findings), the value in the accessible name of a record, a decoy that has a hint, no iPRF call for the block of the index. Not checked: Safari, a phone.

2026-11-20, review of pull request 2 by CodeRabbit: 1 minor finding, valid, fixed: the note of the selected hint in the HintInit figure has aria-live.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
docs/protocol-visualization.html is an article with five figures: the sampler tree, the Swap-or-Not permutation, the iPRF, HintInit, and the online figure with the actions query and update. The physics board, matter.js and the sound are removed. The site shows the new page after the pull request is merged, because the Pages workflow runs on a push to main.
<!-- SECTION:FINAL_SUMMARY:END -->
