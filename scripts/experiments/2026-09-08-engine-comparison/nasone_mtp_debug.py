import runpy
import sys

sys.path.insert(0, "/home/lhl/hipEngine-main")
from hipengine.runtime import qwen35_gguf_mtp as m

original = m._effective_target_verify_mode


def observed(requested, **kwargs):
    result = original(requested, **kwargs)
    print("VERIFY_MODE", requested, kwargs, result, flush=True)
    return result


m._effective_target_verify_mode = observed
original_prepare = m.Qwen35GGUFTransactionalVerifier.prepare


def prepare(self, batch, **kwargs):
    print("PREPARE", self.target.position, batch.tokens, batch.positions,
          kwargs["remaining_decode"], flush=True)
    result = original_prepare(self, batch, **kwargs)
    print("PREPARED", self.target.position, result.summary,
          result.native_graph_fallback_reason, flush=True)
    return result


m.Qwen35GGUFTransactionalVerifier.prepare = prepare
sys.argv[0] = "/home/lhl/hipEngine-main/scripts/qwen36_dense_gguf_suite.py"
runpy.run_path(sys.argv[0], run_name="__main__")
