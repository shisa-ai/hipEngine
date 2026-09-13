import runpy
import sys
import numpy as np

sys.path.insert(0, "/home/lhl/hipEngine-main")
from hipengine.runtime.qwen35_gguf_mtp import Qwen35GGUFTransactionalVerifier
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

original_prepare = Qwen35GGUFTransactionalVerifier.prepare
original_verify = Qwen35GGUFResidentSession.verify_target_block


def prepare(self, *args, **kwargs):
    kwargs["allow_graph"] = False
    return original_prepare(self, *args, **kwargs)


def verify(self, *args, **kwargs):
    kwargs["capture_layer_output_hidden"] = tuple(range(64))
    kwargs["capture_layer_boundary_hidden"] = tuple(range(64))
    kwargs["capture_lm_head_logits"] = True
    result = original_verify(self, *args, **kwargs)
    print("VERIFY_IDS", result.start_position, result.input_token_ids, result.token_ids, flush=True)
    for layer, arrays in result.layer_boundary_hidden.items():
        print("BOUNDARY", layer, {k: (int(np.isnan(v).sum()), float(np.nanmax(np.abs(v))))
                                  for k, v in arrays.items()}, flush=True)
    for layer, values in result.layer_output_hidden.items():
        print("LAYER", layer, int(np.isnan(values).sum()), float(np.nanmax(np.abs(values))), flush=True)
    raise RuntimeError("stop after first target layer capture")


Qwen35GGUFTransactionalVerifier.prepare = prepare
Qwen35GGUFResidentSession.verify_target_block = verify
sys.argv[0] = "/home/lhl/hipEngine-main/scripts/qwen36_dense_gguf_suite.py"
runpy.run_path(sys.argv[0], run_name="__main__")
