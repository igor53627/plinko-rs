---
id: TASK-34
title: Publish the protocol demo on GitHub Pages
status: Done
assignee: []
created_date: '2026-11-20T10:11:00Z'
updated_date: '2026-11-20T10:11:00Z'
labels:
  - docs
  - ci
dependencies: []
references:
  - docs/protocol-visualization.html
  - .github/workflows/pages.yml
  - README.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
docs/protocol-visualization.html is an interactive page about the protocol, written in December 2025. The Pages workflow was removed on 2026-11-19 because GitHub Pages was not enabled. The owner decided on 2026-11-20 to review the page, to publish it on GitHub Pages and to make the repository public. The README badge of DeepWiki showed as a broken image.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The demo page agrees with docs/protocol_overview.md, and the HintInit demo computes the parities that the pseudocode beside it gives
- [x] #2 The Pages workflow publishes the demo page only; no other file of docs/ is on the site
- [x] #3 The repository is public, and https://igor53627.github.io/plinko-rs/ gives the demo page
- [x] #4 Each badge of README.md loads through the image proxy of GitHub
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-11-20: decisions of the owner: enable GitHub Pages, make the repository public, publish the demo page only, rename the README title to plinko-rs. Commits e10b6cb (page), 5860c77 (README), 24b744a (workflow). Checks: the page in Chromium at 1280 px and 390 px with no console error; a check in the page that the 8 regular parities and the 4 backup parity pairs equal the values from the definition; the live site gives the same bytes as docs/protocol-visualization.html, and 404 for README.md, the PDF of the paper and the transcript; the 5 badges of the rendered README give HTTP 200. Not checked: Safari, a phone, a 120 Hz display. Open, not planned: the bins of the board are not exactly below the gaps of the pins; with matter.js 0.19.0 the board runs at two times the speed on a 120 Hz display (from the code, not measured).
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
The demo page is corrected and is on https://igor53627.github.io/plinko-rs/ (also under /protocol-visualization.html, the link of the 0.2.0 README). The repository is public since 2026-11-20. The Pages workflow publishes the demo page only and runs when the page or the workflow changes. The HintInit demo uses a toy iPRF inverse and shows which candidate hints take an entry. README.md has the title plinko-rs and five badges; the DeepWiki badge uses a static image because deepwiki.com/badge.svg gives HTTP 429 to the image proxy of GitHub.
<!-- SECTION:FINAL_SUMMARY:END -->
