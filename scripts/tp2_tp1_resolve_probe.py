"""Dump the TP1 resident-control route's resolved GGUF linear variants.

The TP1 reference capture records no resolve log, so the TP2 bulk route's
dispatch decisions could not be diffed against it. This builds the same resident
control session the reference used, installs the diagnostic's resolve logger,
runs one teacher-forced prefill, and prints the leaves - the missing half.
"""
import sys, json
sys.path.insert(0, '.')
from scripts.tp2_bulk_prefill_diagnostic import _install_resolve_logger, _summarize_resolve_log
import hipengine.runtime.gguf_linear as gguf_linear

_install_resolve_logger(gguf_linear)
import scripts.tp2_bulk_prefill_diagnostic as diag
diag._RESOLVE_LOG = []

MODEL = "/models/gguf/Qwen3.8-27B-Q4_K_M.gguf"
from scripts.tp2_resident_control import create_resident_control
session = create_resident_control(MODEL, capacity=200, row_hook=lambda r: None)
try:
    out = session.teacher_forced_logits(list(range(64)))
    print("prefill logits", getattr(out, "shape", None), flush=True)
    session.generate(list(range(64)), max_new_tokens=8)
    print("decode exercised", flush=True)
finally:
    session.close()
print(json.dumps(_summarize_resolve_log(diag._RESOLVE_LOG), indent=1))
