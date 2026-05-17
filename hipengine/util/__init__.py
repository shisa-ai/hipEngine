"""Utility helpers that are reusable across the hipEngine stack and tooling.

Modules here are intentionally light-weight, dependency-free, and safe to
import from any context (tests, scripts, host code). They must not pull in
torch, HIP, or backend-specific code at import time.
"""
