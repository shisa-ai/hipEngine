import json
from pathlib import Path
import subprocess

from engine_compare_runner import RAW, ROOT, run_case

MODEL = "/models/gguf/Qwen3.8-27B-Q4_K_M.gguf"
NASONE = Path("/tmp/llama-rdna3-nasone32")
STRIX = Path("/tmp/strix-llama-head")


def resume_name(name):
    attempt = 0
    while True:
        candidate = name if attempt == 0 else f"{name}-retry{attempt}"
        monitor = RAW / f"{candidate}-monitor.json"
        if not monitor.exists():
            return candidate, False
        value = json.loads(monitor.read_text())
        if value["returncode"] == 0 and not value["foreign_gpu1_owners"]:
            print("REUSE", candidate, flush=True)
            return candidate, True
        attempt += 1


def hip_suite(name, outputs, runs):
    name, complete = resume_name(name)
    if complete:
        return
    command = [
        "python3", "scripts/qwen36_dense_gguf_suite.py", "--model", MODEL,
        "--max-new-tokens", str(outputs), "--candidate-budgets", "3",
        "--target-verify-mode", "native", "--runs", str(runs), "--limit", "10",
        "--warmup", "--max-sequence-length", "1024",
        "--compiler-version-file", "/tmp/hipengine-hipcc-version.txt",
        "--require-cached-build", "--output", str(RAW / f"{name}.json"),
    ]
    assert run_case(name, command) == 0
    payload = json.loads((RAW / f"{name}.json").read_text())
    assert payload["correctness"]["all_exact_greedy"]
    assert payload["correctness"]["all_gpu_accept_match_cpu"]
    assert payload["memory_after_close"]["active_allocations"] == 0


def hip_shape(name, prompt, outputs=128, capacity=False):
    name, complete = resume_name(name)
    if complete:
        return
    command = [
        "python3", "scripts/qwen35_gguf_bench.py", "--model", MODEL,
        "--prompt-length", str(prompt), "--decode-tokens", str(outputs),
        "--persistent-session", "--force-bulk-prefill", "--public-ar-profile",
        "--compiler-version-file", "/tmp/hipengine-hipcc-version.txt",
        "--require-cached-build", "--json", str(RAW / f"{name}.json"),
    ]
    env = None
    if capacity:
        command += ["--kv-storage", "int8_per_token_head", "--kv-scale-dtype", "fp32",
                    "--warmup-runs", "0", "--measured-runs", "1", "--warmup-decode-tokens", "0"]
        env = {"HIPENGINE_GGUF_INT8_KV_ALLOW_UNVERIFIED_LONG": "1",
               "HIPENGINE_GGUF_INT8_KV_BF16_FULL_LAYERS": "none"}
    else:
        command += ["--graph-replay-decode"]
    assert run_case(name, command, env) == 0
    assert json.loads((RAW / f"{name}.json").read_text())["summary"]["finite_final_logits_all"]


def llama(name, source, mode="ar", outputs=25, repetitions=3, lengths="", context=1024, kv="bf16", env=None):
    name, complete = resume_name(name)
    if complete:
        return 0
    command = [
        "python3", "scripts/llamacpp_raw_suite_bench.py", "--server", str(source / "build/bin/llama-server"),
        "--source", str(source), "--model", MODEL, "--mode", mode,
        "--outputs", str(outputs), "--repetitions", str(repetitions),
        "--context", str(context), "--kv", kv, "--output", str(RAW / f"{name}.json"),
    ]
    if lengths:
        command += ["--repeat-lengths", lengths]
    return run_case(name, command, env)


hip_suite("hipengine-c1-short", 25, 3)
hip_suite("hipengine-c1-long", 129, 1)
hip_shape("hipengine-p512", 512)
hip_shape("hipengine-p8192", 8192)
assert llama("nasone32-shapes", NASONE, outputs=129, lengths="512,8192", context=8704) == 0
assert llama("nasone32-ar", NASONE) == 0
assert llama("nasone32-mtp", NASONE, mode="mtp") == 0
assert llama("nasone32-ar-sequential", NASONE, env={"GGML_CUDA_GDN_CHUNKED": "0"}) == 0
assert llama("nasone32-mtp-sequential", NASONE, mode="mtp", env={"GGML_CUDA_GDN_CHUNKED": "0"}) == 0
assert llama("nasone32-ar-long", NASONE, outputs=129, repetitions=1) == 0
assert llama("nasone32-mtp-long", NASONE, mode="mtp", outputs=129, repetitions=1) == 0
hip_shape("hipengine-pure-int8-128k", 131072, outputs=8, capacity=True)

# Pin HEAD when its lane starts; a resumed packet keeps the same revision.
if not (RAW / "strix-source.json").exists():
    subprocess.run(["git", "-C", str(STRIX), "fetch", "--depth", "100", "origin"], check=True)
    head = subprocess.check_output(
        ["git", "-C", str(STRIX), "ls-remote", "origin", "HEAD"], text=True,
    ).split()[0]
    subprocess.run(["git", "-C", str(STRIX), "checkout", "--detach", head], check=True)
    with (RAW / "strix-build.log").open("w") as log:
        subprocess.run(["cmake", "--build", str(STRIX / "build"), "--target", "llama-bench",
                        "llama-server", "-j", "16"], check=True, stdout=log, stderr=subprocess.STDOUT)
    assert not subprocess.check_output(["git", "-C", str(STRIX), "status", "--porcelain"])
    (RAW / "strix-source.json").write_text(json.dumps(dict(commit=head, path=str(STRIX)), indent=2) + "\n")

assert llama("strix-llama-shapes", STRIX, outputs=129, lengths="512,8192", context=8704) == 0
assert llama("strix-llama-ar", STRIX) == 0
assert llama("strix-llama-mtp", STRIX, mode="mtp") == 0
assert llama("strix-llama-ar-long", STRIX, outputs=129, repetitions=1) == 0
assert llama("strix-llama-mtp-long", STRIX, mode="mtp", outputs=129, repetitions=1) == 0
if llama("strix-llama-bf16-128k", STRIX, outputs=9, repetitions=1,
         lengths="131072", context=131328) != 0:
    assert llama("strix-llama-bf16-112k", STRIX, outputs=9, repetitions=1,
                 lengths="114688", context=114944) == 0
if llama("strix-llama-q8-224k", STRIX, outputs=9, repetitions=1,
         lengths="229376", context=229632, kv="q8_0") != 0:
    assert llama("strix-llama-q8-192k", STRIX, outputs=9, repetitions=1,
                 lengths="196608", context=196864, kv="q8_0") == 0
print("ALL COMPARISON CASES COMPLETE", flush=True)
