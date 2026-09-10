"""Torch-free, request-owned grammar masks applied before greedy selection."""
import json
import re
from copy import deepcopy

import numpy as np
from llguidance import LLMatcher, LLTokenizer, StructTag, grammar_from


def _literal(value):
    return json.dumps(value,ensure_ascii=False)


def _string_rule(schema):
    supported = {"type","title","description","default","examples","deprecated","readOnly",
                 "writeOnly","$comment","enum","const","minLength","maxLength","pattern"}
    if set(schema)-supported:
        raise ValueError("unsupported XML string constraints: "+", ".join(sorted(set(schema)-supported)))
    values = schema.get("enum", [schema["const"]] if "const" in schema else None)
    if values is not None:
        if not all(isinstance(v,str) for v in values):
            raise ValueError("XML string enum must contain strings")
        values = [v for v in values if
                  ("const" not in schema or v==schema["const"]) and
                  len(v)>=schema.get("minLength",0) and
                  ("maxLength" not in schema or len(v)<=schema["maxLength"]) and
                  ("pattern" not in schema or re.search(schema["pattern"],v))]
        if not values:
            raise ValueError("XML string enum has no valid values")
        return "("+" | ".join(_literal(v) for v in values)+")"
    if "pattern" in schema:
        if "minLength" in schema or "maxLength" in schema:
            raise ValueError("XML string pattern combined with length bounds is not supported")
        pattern = schema["pattern"]
        prefix = "" if pattern.startswith("^") else "(.|\\n)*"
        end_anchor = pattern.endswith("$") and not pattern.endswith("\\$")
        suffix = "" if end_anchor else "(.|\\n)*"
        pattern = pattern.removeprefix("^")
        if end_anchor:
            pattern = pattern[:-1]
        return "/"+prefix+"(?:"+pattern.replace("/","\\/")+")"+suffix+"/"
    minimum = int(schema.get("minLength",0))
    maximum = schema.get("maxLength")
    extent = f"{{{minimum},{'' if maximum is None else int(maximum)}}}"
    return "/(.|\\n)"+extent+"/"


def xml_tool_grammar(tools, *, mode="auto", selected_name=None, parallel=True):
    tags,required_rules = [],[]
    for tool in tools:
        function = tool.get("function",tool)
        name = function["name"]
        if not isinstance(name,str) or not re.fullmatch(r"[^<>\s]+",name):
            raise ValueError("tool name cannot be represented by the XML protocol")
        if selected_name is not None and name != selected_name:
            continue
        schema = deepcopy(function.get("parameters",{"type":"object","properties":{}}))
        properties = schema.get("properties",{})
        if schema.get("type","object") != "object" or any(
            key in schema for key in ("anyOf","oneOf","allOf","$ref","patternProperties","dependentRequired")):
            raise ValueError("XML tools require an object schema with explicit properties")
        required = set(schema.get("required",()))
        if not required <= set(properties):
            raise ValueError("XML tool requires an undeclared parameter")
        rules,sequence = [],[]
        for number,(key,value_schema) in enumerate(properties.items()):
            if not isinstance(key,str) or not re.fullmatch(r"[^<>\s]+",key):
                raise ValueError("parameter name cannot be represented by the XML protocol")
            if not isinstance(value_schema,dict):
                raise ValueError("XML parameter schema must be an object")
            rule = f"parameter_{number}"
            if value_schema.get("type")=="string":
                value_name = f"value_{number}"
                rules.append(f'{value_name}[suffix="\\n</parameter>\\n"]: '+_string_rule(value_schema))
                rules.append(f'{rule}: {_literal("<parameter="+key+">"+chr(10))} {value_name}')
            else:
                definition = dict(value_schema)
                if "$defs" in schema:
                    definition["$defs"] = schema["$defs"]
                value = "%json "+json.dumps(definition)
                rules.append(f'{rule}: {_literal("<parameter="+key+">"+chr(10))} {value} '
                             f'{_literal(chr(10)+"</parameter>"+chr(10))}')
            sequence.append(rule if key in required else f"{rule}?")
        body = ('start: arguments "</function>\\n" </tool_call>\narguments: '+
                " ".join(sequence)+"\n"+"\n".join(rules))
        begin = "<tool_call>\n<function="+name+">\n"
        tags.append(StructTag(trigger="<tool_call>",begin=begin,grammar=body,end=""))
        index = len(required_rules)
        required_rules.append(dict(name=f"call_{index}",lark_grammar=
            'start: <tool_call> '+_literal(begin[len("<tool_call>"):])+
            ' arguments "</function>\\n" </tool_call>\narguments: '+" ".join(sequence)+"\n"+"\n".join(rules)))
    if not tags:
        raise ValueError("no available tool for requested tool choice")
    if mode=="auto" and parallel:
        return StructTag.to_grammar(tags)
    alternatives = " | ".join("@"+g["name"] for g in required_rules)
    if mode=="auto":
        root = dict(name="root",lark_grammar=f"start: text (({alternatives}) text)?\ntext: /(.|\\n)*/")
    else:
        suffix = f' ("\\n" ({alternatives}))*' if parallel else ""
        root = dict(name="root",lark_grammar=f"start: ({alternatives})"+suffix)
    return json.dumps(dict(grammars=[root,*required_rules]))


def _compile_body(spec):
    if spec["format"] == "qwen4exp_tools":
        if spec.get("answer") is not None and spec["mode"]=="auto":
            calls = json.loads(xml_tool_grammar(spec["tools"],mode="required",selected_name=spec.get("name"),
                                               parallel=spec.get("parallel",True)))["grammars"]
            calls[0]["name"] = "tool_root"
            answer = _compile_body(spec["answer"])
            answers = json.loads(answer)["grammars"] if answer.lstrip().startswith("{") else [
                {"lark_grammar":answer}]
            answers[0]["name"] = "answer_root"
            return json.dumps({"grammars":[
                {"name":"combined_root","lark_grammar":"start: @tool_root | @answer_root"},
                *calls,*answers]})
        return xml_tool_grammar(spec["tools"],mode=spec["mode"],selected_name=spec.get("name"),
                                parallel=spec.get("parallel",True))
    value = spec["value"]
    return grammar_from(spec["format"], value if isinstance(value,str) else json.dumps(value))


def compile_spec(spec):
    body = _compile_body(spec)
    if not spec.get("reasoning_open"):
        return body
    grammars = json.loads(body)["grammars"] if body.lstrip().startswith("{") else [
        {"name":"answer","lark_grammar":body}]
    first_name = grammars[0].get("name") or "answer"
    grammars[0]["name"] = first_name
    root = {"name":"reasoning_root","lark_grammar":
            'start: /(.|\\n)*/ </think> @'+first_name}
    return json.dumps({"grammars":[root,*grammars]})


class GrammarSession:
    """One mutable matcher per request; a tokenizer may be shared read-only."""
    def __init__(self, tokenizer, spec):
        self.matcher = LLMatcher(tokenizer,compile_spec(spec))
        if self.matcher.is_error():
            raise ValueError("invalid output grammar: "+self.matcher.get_error())
        self.eos = set(tokenizer.eos_tokens)
        self.done = False

    def select(self, logits, *, suppress=(), logit_bias=()):
        scores = np.asarray(logits,dtype=np.float32).copy()
        allowed = np.frombuffer(self.matcher.compute_logit_bias(),dtype=np.uint8)
        if self.matcher.is_error() or len(allowed)!=len(scores):
            raise ValueError("grammar mask failed: "+self.matcher.get_error())
        scores[allowed==0] = -np.inf
        for token,bias in logit_bias:
            scores[int(token)] += float(bias)
        for token in suppress:
            scores[int(token)] = -np.inf
        if not np.isfinite(scores).any():
            raise ValueError("grammar and token controls allow no finite token")
        token = int(np.argmax(scores))
        if not self.matcher.consume_token(token):
            raise ValueError("grammar rejected selected token: "+self.matcher.get_error())
        self.done = token in self.eos or self.matcher.is_stopped()
        return token


def make_tokenizer(tokenizer):
    return LLTokenizer(tokenizer.encoder.to_str(),n_vocab=len(tokenizer.tokens),
                       eos_token=tokenizer.eos_token_id)
