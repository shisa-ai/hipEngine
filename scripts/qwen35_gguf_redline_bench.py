import os
import runpy

from hipengine.core.hip import configure_default_graph_adapter
from hipengine.core.redline_graph import RedlineHipGraphAdapter

library = os.environ["HIPENGINE_REDLINE_HIPGRAPH_LIBRARY"]
module = os.environ["HIPENGINE_REDLINE_HIPGRAPH_MODULE"]
digest = os.environ["HIPENGINE_REDLINE_HIPGRAPH_SHA256"]
adapter = RedlineHipGraphAdapter.load(
    library_path=library,
    module_path=module,
    expected_sha256=digest,
    require_pm4=True,
)
configure_default_graph_adapter(adapter)
# Keep the already mapped interposer in this Python process without injecting
# the Python-enabled DSO into git/hipcc/other non-Python child processes.
os.environ.pop("LD_PRELOAD", None)
runpy.run_path("scripts/qwen35_gguf_bench.py", run_name="__main__")
