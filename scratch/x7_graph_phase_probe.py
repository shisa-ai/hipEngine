import time, numpy as np, hipengine
from pathlib import Path
from collections import defaultdict
MODEL = Path("/mnt/nvme1/models/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf")
llm = hipengine.LLM(model=str(MODEL))
runner = llm._get_text_generator()._ensure_runner()
vocab = int(runner.weights.config.vocab_size or 0)
rng = np.random.default_rng(7)
prompt = [int(t) for t in rng.integers(0, vocab, 1024)]
N = 128
pc = time.perf_counter
ph = defaultdict(list); on = {"v": True}
def wrap(obj, name, label):
    real = getattr(obj, name)
    def fn(*a, **k):
        if on["v"]:
            t0 = pc(); r = real(*a, **k); ph[label].append((pc() - t0) * 1e3); return r
        return real(*a, **k)
    setattr(obj, name, fn)
wrap(runner, "_stage_block_content", "stage")
wrap(runner, "_launch_block", "launch")
wrap(runner, "_collect_block", "collect")
rt = hipengine.core.hip.get_hip_runtime()

runner.reset()
logits = runner.forward(prompt)
tok = runner.next_token(logits)

# --- launched ---
ts = []
for _ in range(N):
    t0 = pc(); logits = runner.forward([tok], return_hidden=True)
    ts.append((pc() - t0) * 1e3); tok = runner.next_token(logits)
base = np.median(ts[5:])
print(f"launched: total {base:.3f} | stage {np.mean(ph['stage'][5:]):.3f} "
      f"enqueue {np.mean(ph['launch'][5:]):.3f} collect {np.mean(ph['collect'][5:]):.3f}")

# --- graph ---
runner.reset(); runner.forward(prompt)
logits = runner.forward([tok])   # warm + arms session
sess = runner._decode_graph
wrap(sess, "_capture_key", "key"); wrap(sess, "_retarget_appends", "retarget")
wrap(rt, "graph_launch", "glaunch")
tok = runner.next_token(logits)
ph.clear()
ts = []
for _ in range(N):
    t0 = pc(); logits = runner.forward([tok])
    ts.append((pc() - t0) * 1e3); tok = runner.next_token(logits)
g = np.median(ts[5:])
print(f"graph  : total {g:.3f} | stage {np.mean(ph['stage'][5:]):.3f} key {np.mean(ph['key'][5:]):.3f} "
      f"retarget {np.mean(ph['retarget'][5:]):.3f} glaunch {np.mean(ph['glaunch'][5:]):.3f} "
      f"collect {np.mean(ph['collect'][5:]):.3f} captures={sess.captures}")
