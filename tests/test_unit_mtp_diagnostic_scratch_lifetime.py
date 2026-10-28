"""Diagnostic verifier calls keep scratch live without an environment reader."""

import ast
from pathlib import Path

import pytest


_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("script", [
    "mtp_chain_e2e_smoke.py", "mtp_layer_drift_audit.py", "mtp_state_drift_audit.py",
])
def test_mtp_diagnostics_keep_verifier_scratch_without_env_switch(script):
    source = (_ROOT / "scripts" / script).read_text()
    assert "HIPENGINE_MTP_SKIP_CANONICALIZE_AFTER_VERIFY" not in source
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute)
             and node.func.attr in {"verify_chain_bulk_and_commit", "verify_tree_bulk_and_commit"}]
    assert calls
    values = [kw.value for node in calls for kw in node.keywords if kw.arg == "canonicalize_after"]
    assert len(values) == len(calls)
    for value in values:
        if script == "mtp_chain_e2e_smoke.py":
            assert isinstance(value, ast.Name) and value.id == "canonicalize_after_verify"
        else:
            assert isinstance(value, ast.Constant) and value.value is False
    if script == "mtp_chain_e2e_smoke.py":
        assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name) and target.id == "canonicalize_after_verify"
                               for target in node.targets)]
        assert len(assignments) == 2
        assert all(isinstance(node.value, ast.Constant) and node.value.value is False for node in assignments)
