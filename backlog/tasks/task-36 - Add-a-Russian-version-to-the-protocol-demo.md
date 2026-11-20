---
id: TASK-36
title: Add a Russian version to the protocol demo
status: Done
assignee: []
created_date: '2026-11-20T13:51:00Z'
updated_date: '2026-11-20T13:56:00Z'
labels: []
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The owner asked for a Russian version of docs/protocol-visualization.html, delivered as a PR. The page is English only. A Chinese version was made and then dropped by the owner; its commit is in the local bookmark docs/tutorial-zh.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A switch on the page changes all text of the article and of the figures between English and Russian
- [x] #2 The English text and the figure output are the same as before the change
- [x] #3 The address ?lang=ru opens the Russian version; a click on the switch saves the choice
- [x] #4 The Russian page has no horizontal scroll at 1280 px and at 390 px, and no console error
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
One file with a switch; the English text stays in the HTML, the Russian text is in the tables TRANSLATIONS (71 keys) and MESSAGES (36 entries per language) of the script. Checks with Playwright: the figures of the English page give the same HTML as before for 150 keys (12,750 snapshots); 240 keys with 45 actions each give no exception and no 'undefined' in either language; the Russian page has no horizontal scroll at 1280 px and 390 px. The first Russian text made the table of the rounds wider than a 390 px screen; the column heads are shorter now and the table uses a smaller font on a narrow screen (both languages). A language from the address is not saved: only a click on the switch saves the choice. CodeRabbit CLI: no finding.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
docs/protocol-visualization.html has a Russian version behind a switch English / Русский, with ?lang=ru and a saved choice.
<!-- SECTION:FINAL_SUMMARY:END -->
