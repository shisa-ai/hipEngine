"""Exercise the registered Qwen4Exp server at admission-resolved native context."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-root",type=Path,required=True)
    p.add_argument("--compiler-version-file",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--target-tokens",type=int,default=8192)
    p.add_argument("--chat-only",action="store_true")
    a = p.parse_args()
    os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(a.compiler_version_file)
    os.environ["HIPENGINE_REQUIRE_CACHED_BUILD"] = "1"
    from fastapi.testclient import TestClient
    from hipengine import LLM
    from hipengine.core.memory import memory_stats
    from hipengine.server import ServerConfig,create_app
    from scripts.qwen4exp_canonical_ar_bench import _git_metadata,_host_metadata
    from scripts.qwen4exp_framework_family_refresh import check_host
    from scripts.qwen4_exp_qsa_retrieval_gate import build_retrieval_prompt
    check_host()
    llm = LLM(str(a.model_root),backend="hip_gfx1151",quant="gguf_ud_q4_k_xl",
              execution_profile="production",max_active_requests=2)
    config = ServerConfig(model=str(a.model_root),backend="hip_gfx1151",
        quant="gguf_ud_q4_k_xl",execution_profile="production",kv_storage="bf16",
        max_active_requests=2,speculative_mtp_serving="off")
    report = dict(schema=1,status="running",performance_claim=False,source=_git_metadata(ROOT),
                  host=_host_metadata(),command=sys.argv,configured_context=None)
    app = create_app(config,llm=llm)
    try:
        with TestClient(app) as client:
            ready = client.get("/ready")
            report["ready"] = ready.json()
            if ready.status_code != 200:
                raise AssertionError("native-context startup failed")
            generator = llm._text_generator
            capacity = int(generator.runner.max_sequence_length)
            report["resolved_context"] = capacity
            report["admission"] = generator.context_admission
            report["resident_runners"] = len(generator._resident_model_runner._all_runners)
            assert capacity == generator.model_plugin.native_context_length
            assert generator.runner.config.qsa_dense_equivalent_max_tokens == 2051
            prompt,needle = build_retrieval_prompt(generator.tokenizer,target_tokens=a.target_tokens)
            tokens = llm.tokenize(prompt)
            assert 2051 < len(tokens) < capacity
            report["retrieval_prompt_tokens"] = len(tokens)
            report["needle_position"] = needle
            if a.chat_only:
                _,separator,tail = prompt.partition("<|im_start|>user\n")
                archive,end,_ = tail.partition("<|im_end|>")
                assert separator and end
                endpoint = "/v1/chat/completions"
                payload = dict(model=config.model_id,messages=[dict(role="user",content=archive)],
                               max_tokens=128,temperature=0.,top_p=1.)
            else:
                endpoint = "/v1/completions"
                payload = dict(model=config.model_id,prompt=prompt,max_tokens=128,temperature=0.,top_p=1.)
            response = client.post(endpoint,json=payload)
            report["retrieval_endpoint"] = endpoint
            report["retrieval_status"] = response.status_code
            report["retrieval_response"] = response.json()
            if response.status_code != 200:
                raise AssertionError("served sparse-context request failed")
            choice = response.json()["choices"][0]
            text = choice["message"]["content"] if a.chat_only else choice["text"]
            if "VIOLET-7391" not in text:
                raise AssertionError("served sparse-context retrieval failed")
            excessive = (dict(messages=[dict(role="user",content=" a"*(capacity+1))])
                         if a.chat_only else dict(prompt=[int(tokens[0])]*(capacity+1)))
            rejected = client.post(endpoint,json=dict(
                model=config.model_id,**excessive,max_tokens=1,temperature=0.))
            report["over_limit_status"] = rejected.status_code
            report["over_limit_response"] = rejected.json()
            if rejected.status_code != 400 or rejected.json().get("error",{}).get("code") != "context_length_exceeded":
                raise AssertionError("over-limit request did not fail with context_length_exceeded")
            report["status"] = "passed"
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        raise
    finally:
        llm.close()
        report["memory_after_close"] = memory_stats()
        a.output.write_text(json.dumps(report,indent=2,default=str)+"\n")
        if report["memory_after_close"]["active_allocations"] != 0:
            raise RuntimeError("served context gate leaked tracked ownership")


if __name__=="__main__":
    main()
