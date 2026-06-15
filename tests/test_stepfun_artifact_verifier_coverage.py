from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_SCRIPT_RE = re.compile(r"--verify-[a-z-]+")


def _stepfun_artifact_scripts_with_verifiers() -> dict[str, list[str]]:
    scripts: dict[str, list[str]] = {}
    for path in sorted((REPO_ROOT / "scripts").glob("stepfun_*.py")):
        text = path.read_text()
        if "DEFAULT_OUTPUT" not in text and "--default-output" not in text:
            continue
        scripts[path.name] = sorted(set(ARTIFACT_SCRIPT_RE.findall(text)))
    return scripts


def test_stepfun_default_output_artifact_scripts_have_drift_verifiers() -> None:
    scripts = _stepfun_artifact_scripts_with_verifiers()

    assert scripts, "expected at least one StepFun default-output artifact script"
    assert [name for name, verifiers in scripts.items() if not verifiers] == []
    assert "--verify-manifest" in scripts["stepfun_final_blocker_manifest.py"]
    assert "--verify-rollup" in scripts["stepfun_remaining_blockers_rollup.py"]
    assert "--verify-blocker-status" in scripts["stepfun_kv_blocker_status.py"]
    assert "--verify-token-mismatch" in scripts["stepfun_oracle_token_mismatch.py"]
    assert "--verify-probe" in scripts["stepfun_llamacpp_logits_probe.py"]
    assert "--verify-patch-dry-run" in scripts[
        "stepfun_llamacpp_logits_helper_patch_dry_run.py"
    ]
