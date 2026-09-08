import runpy
import sys

sys.path.insert(0, "/home/lhl/hipEngine-main")
from hipengine.runtime.qwen35_gguf_mtp import Qwen35GGUFTransactionalVerifier

original = Qwen35GGUFTransactionalVerifier.prepare


def prepare(self, *args, **kwargs):
    kwargs["allow_graph"] = False
    return original(self, *args, **kwargs)


Qwen35GGUFTransactionalVerifier.prepare = prepare
sys.argv[0] = "/home/lhl/hipEngine-main/scripts/qwen36_dense_gguf_suite.py"
runpy.run_path(sys.argv[0], run_name="__main__")
