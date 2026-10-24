"""Reproduce the rejected graph-auto candidate without changing the eager default.

The original driver explicitly selects eager for its baseline arm. This wrapper
restores the measured candidate's automatic graph selection for its graph arm.
All model, workload, timing and validation logic stays in the frozen driver.
"""
from pathlib import Path
import runpy

from hipengine.generation.qwen35_paro_mtp2 import Qwen35ParoMTP2Adapter

original = Qwen35ParoMTP2Adapter.__dict__["_target_graph_mode"]
try:
    Qwen35ParoMTP2Adapter._target_graph_mode = staticmethod(lambda plan: "auto")
    runpy.run_path(str(Path(__file__).with_name("measure_graph.py")), run_name="__main__")
finally:
    Qwen35ParoMTP2Adapter._target_graph_mode = original
