"""Measure the row-capable vs initial-state-only verify journal footprint."""
import sys
sys.path.insert(0, "/home/lhl/hipEngine-ud")
from pathlib import Path
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

model = Path("/models/gguf/Qwen3.8-27B-Q4_K_M.gguf")
with Qwen35GGUFResidentSession(model, max_sequence_length=107) as target:
    scratch = target.scratch
    conv = [s for s in scratch.layer_conv_states if s is not None]
    rec = [s for s in scratch.layer_recurrent_states if s is not None]
    per_row = sum(int(s.nbytes) for s in (*conv, *rec))
    print("backend:", target.backend)
    print("scratch.max_positions:", scratch.max_positions)
    print("conv layers:", len(conv), "recurrent layers:", len(rec))
    print("state bytes per row: %.2f MiB" % (per_row / 2**20))
    for max_rows in (4,):
        row_capable = (max_rows + 1) * per_row
        initial_only = 1 * per_row
        print(f"max_rows={max_rows}: initial_state_only={initial_only/2**20:.2f} MiB "
              f"row_capable={row_capable/2**20:.2f} MiB delta={(row_capable-initial_only)/2**20:.2f} MiB")
