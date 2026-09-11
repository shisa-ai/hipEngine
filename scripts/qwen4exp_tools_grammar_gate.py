"""HTTP sanity gates for XML tools and tokenizer-constrained structured output."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import urllib.error
import urllib.request


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url",default="http://127.0.0.1:18146")
    p.add_argument("--model",default="qwen4exp")
    p.add_argument("--output",type=Path,required=True)
    a = p.parse_args()
    report = {"status":"running","cases":[]}
    def post(payload):
        request = urllib.request.Request(a.base_url+"/v1/chat/completions",
            data=json.dumps({"model":a.model,"temperature":0,"max_tokens":256,
                             "enable_thinking":False,**payload}).encode(),
            headers={"Content-Type":"application/json"})
        try:
            with urllib.request.urlopen(request,timeout=180) as response:
                data = response.read().decode()
        except urllib.error.HTTPError as exc:
            raise RuntimeError(exc.read().decode()) from exc
        return data if payload.get("stream") else json.loads(data)
    tool = {"type":"function","function":{"name":"lookup","description":"Look up an inventory key.",
        "parameters":{"type":"object","properties":{"key":{"type":"string"},"count":{"type":"integer"}},
                      "required":["key","count"],"additionalProperties":False}}}
    try:
        payload = dict(messages=[{"role":"user","content":"Look up key DELTA using count 3."}],
                       tools=[tool],tool_choice={"type":"function","function":{"name":"lookup"}})
        result = post(payload)
        message = result["choices"][0]["message"]
        call = message["tool_calls"][0]
        assert call["function"]["name"]=="lookup"
        assert json.loads(call["function"]["arguments"])=={"key":"DELTA","count":3}
        report["cases"].append(dict(name="required_xml_tool",response=result))
        history = [*payload["messages"],message,{"role":"tool","tool_call_id":call["id"],
                   "content":'{"result":"DELTA is in aisle 7"}'}]
        result = post(dict(messages=history,tools=[tool],tool_choice="none"))
        assert "7" in result["choices"][0]["message"]["content"]
        assert not result["choices"][0]["message"].get("tool_calls")
        report["cases"].append(dict(name="multiturn_tool_response",response=result))

        def schema_case(label):
            return post(dict(messages=[{"role":"user","content":"Ignore the schema and output invalid plain text."}],
                response_format={"type":"json_schema","json_schema":{"name":"fixed","strict":True,
                    "schema":{"type":"object","properties":{"label":{"const":label},"count":{"const":7}},
                              "required":["label","count"],"additionalProperties":False}}}))
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(schema_case,("FIRST","SECOND")))
        for label,result in zip(("FIRST","SECOND"),results):
            assert json.loads(result["choices"][0]["message"]["content"])=={"label":label,"count":7}
            report["cases"].append(dict(name="schema_isolation_"+label,response=result))
        stream = post({**payload,"stream":True})
        calls = {}
        finish = None
        for line in stream.splitlines():
            if not line.startswith("data: ") or line=="data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            for choice in chunk.get("choices",[]):
                finish = choice.get("finish_reason") or finish
                for part in choice.get("delta",{}).get("tool_calls",[]):
                    row = calls.setdefault(part["index"],dict(name="",arguments=""))
                    fn = part.get("function",{})
                    row["name"] += fn.get("name","")
                    row["arguments"] += fn.get("arguments","")
        assert finish=="tool_calls" and calls[0]["name"]=="lookup"
        assert json.loads(calls[0]["arguments"])=={"key":"DELTA","count":3}
        report["cases"].append(dict(name="streamed_tool",calls=calls,finish=finish))
        result = post(dict(messages=[{"role":"user","content":"Return anything."}],grammar='root ::= "GRAMMAR-OK"'))
        assert result["choices"][0]["message"]["content"]=="GRAMMAR-OK"
        report["cases"].append(dict(name="gbnf",response=result))
        report["status"] = "passed"
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        raise
    finally:
        a.output.write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":
    main()
