---
id: TASK-15
title: Archive closed backlog tasks
status: To Do
assignee: []
created_date: '2026-11-17T08:42:00Z'
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
- [ ] #1 The block subset rework is complete, or the owner confirms that TASK-1, TASK-2 and TASK-5 are not necessary as a reference
- [ ] #2 Closed tasks are archived with backlog cleanup; no task file is deleted by hand
- [ ] #3 backlog task list shows only open tasks and recently closed tasks
- [ ] #4 The empty backlog/.locks directory is not committed
<!-- AC:END -->
