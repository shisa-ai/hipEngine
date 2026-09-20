---
name: cleanup-fix
description: Apply fixes from the hipEngine cleanup audit queue (audit/audit.py queue) and close the rows out. Use when clearing doc-path-drift, axis-branch, unguarded-hip-test, stub, marker or ungoverned-flag-branch findings, when a triage decision says promote/remove/document and someone should do it, or when asked to lower the audit budget by actually fixing things.
tools: Bash, Read, Grep, Glob, Edit, Write
---

Read `audit/agents/fix.md` and follow it. That file is the assignment and is the
one to edit; this stub only registers the agent with Claude Code.
