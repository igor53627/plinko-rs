---
id: TASK-15
title: Archive closed backlog tasks
status: Done
assignee: []
created_date: '2026-11-17T08:42:00Z'
updated_date: '2026-11-28T13:31:32Z'
labels:
  - maintenance
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
backlog/tasks holds 13 closed tasks (TASK-1 to TASK-13) next to the open ones. Archive the closed tasks with backlog cleanup so that the task list shows current work. Do this after the block subset representation is reworked: TASK-1, TASK-2 and TASK-5 describe the current bitset and cached-vector design and are the reference for that rework.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The block subset rework is complete, or the owner confirms that TASK-1, TASK-2 and TASK-5 are not necessary as a reference
- [x] #2 Closed tasks are archived with backlog cleanup; no task file is deleted by hand
- [x] #3 backlog task list shows only open tasks and recently closed tasks
- [x] #4 The empty backlog/.locks directory is not committed
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-11-28: Archived 33 closed tasks older than 1 week via backlog cleanup into backlog/archive/tasks/. Added backlog/.locks/ to .gitignore. backlog task list now shows 0 open tasks and 5 recently closed tasks (TASK-15, TASK-21, TASK-22, TASK-37, TASK-38).
<!-- SECTION:NOTES:END -->
