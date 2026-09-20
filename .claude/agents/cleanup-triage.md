---
name: cleanup-triage
description: Decide what hipEngine cleanup-audit rows are, without changing code. Use when working through DEAD-FLAG, LOST-OPT, EXACTNESS-REJECT, ORPHAN-KERNEL, STALE-LEDGER or GATE-CATCH22 debt, when reviewing stale or expired decisions, or when `audit.py check` fails because untriaged rows grew. For applying fixes, use cleanup-fix instead.
tools: Bash, Read, Grep, Glob
---

Read `audit/agents/triage.md` and follow it. That file is the assignment and is
the one to edit; this stub only registers the agent with Claude Code.
