"""Public import surface of the ``hipengine`` root package.

The root package is the startup-isolation boundary: a bare ``import hipengine``
must stay CPU-safe -- no ``hipengine.llm`` (which transitively loads the
speculative package and the GPU kernel backends), no backend package, no torch
-- while the documented exports ``LLM``, ``SamplingParams``, and
``ExecutionProfile`` must keep working exactly as before through lazy PEP 562
attribute resolution. Every test here runs a fresh interpreter; in-process
checks cannot see startup imports that an earlier test already triggered.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Blocking hipengine.llm outright would also block the intentional lazy path,
# so the bare-import probe tracks loaded modules instead of rejecting them.
_BARE_IMPORT_PROBE = """\
import json, sys
import hipengine
loaded = sorted(
    name
    for name in sys.modules
    if name.startswith(("hipengine.llm", "hipengine.speculative", "hipengine.kernels.hip_", "hipengine.kernels.cuda_", "torch"))
)
print(json.dumps({
    "loaded": loaded,
    "execution_profile": repr(hipengine.ExecutionProfile),
    "in_all": [name for name in ("ExecutionProfile", "LLM", "SamplingParams") if name in hipengine.__all__],
    "dir_names": [name for name in ("ExecutionProfile", "LLM", "SamplingParams") if name in dir(hipengine)],
}))
"""

_LAZY_EXPORT_PROBE = """\
import json, sys
from hipengine import ExecutionProfile, LLM, SamplingParams
import hipengine
from hipengine.execution_profiles import ExecutionProfile as EagerExecutionProfile
checks = {
    "llm_identity": LLM is hipengine.LLM is sys.modules["hipengine.llm"].LLM,
    "sampling_identity": SamplingParams is hipengine.SamplingParams is sys.modules["hipengine.llm"].SamplingParams,
    "profile_identity": ExecutionProfile is EagerExecutionProfile,
    "repeat_access_cached": hipengine.LLM is LLM and hipengine.SamplingParams is SamplingParams,
    "llm_imported_after_access": "hipengine.llm" in sys.modules,
    "execution_profile_was_eager": "hipengine.execution_profiles" in sys.modules,
}
print(json.dumps(checks))
"""

_DIRECT_LLM_MODULE_PROBE = """\
import json, sys
import hipengine.llm
import hipengine
checks = {
    "package_attr_matches_module": hipengine.LLM is hipengine.llm.LLM
    and hipengine.SamplingParams is hipengine.llm.SamplingParams,
}
print(json.dumps(checks))
"""

_MISSING_ATTRIBUTE_PROBE = """\
import json, sys
import hipengine
try:
    hipengine.DefinitelyNotAnExport
except AttributeError as error:
    message = str(error)
else:
    message = "no AttributeError raised"
print(json.dumps({"message": message, "hasattr": hasattr(hipengine, "DefinitelyNotAnExport")}))
"""


def _run_probe(source: str) -> dict:
    result = subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=ROOT,
    )
    assert result.returncode == 0, f"probe failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_bare_import_loads_no_llm_no_speculative_no_backend_no_torch():
    """``import hipengine`` alone stays CPU-safe: only pure metadata loads."""

    summary = _run_probe(_BARE_IMPORT_PROBE)
    assert summary["loaded"] == []
    assert summary["execution_profile"].startswith("<enum 'ExecutionProfile'>")
    assert summary["in_all"] == ["ExecutionProfile", "LLM", "SamplingParams"]
    assert summary["dir_names"] == ["ExecutionProfile", "LLM", "SamplingParams"]


def test_lazy_exports_resolve_to_the_real_llm_classes_and_cache():
    """``from hipengine import LLM, SamplingParams`` binds the real classes."""

    summary = _run_probe(_LAZY_EXPORT_PROBE)
    assert summary == {
        "llm_identity": True,
        "sampling_identity": True,
        "profile_identity": True,
        "repeat_access_cached": True,
        "llm_imported_after_access": True,
        "execution_profile_was_eager": True,
    }


def test_direct_llm_module_import_keeps_package_attributes_in_sync():
    """Importing ``hipengine.llm`` directly still exposes it on the package."""

    summary = _run_probe(_DIRECT_LLM_MODULE_PROBE)
    assert summary == {"package_attr_matches_module": True}


def test_unknown_attribute_raises_attribute_error():
    """Lazy resolution must not turn typos into imports or other errors."""

    summary = _run_probe(_MISSING_ATTRIBUTE_PROBE)
    assert summary["hasattr"] is False
    assert "DefinitelyNotAnExport" in summary["message"]
    assert "has no attribute" in summary["message"]
