"""Qwen4Exp's embedded-template renderer and XML-parameter tool protocol."""
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
import json
import re

from jinja2.sandbox import ImmutableSandboxedEnvironment


def _raise_template_error(message):
    raise ValueError(str(message))


@lru_cache(maxsize=8)
def _compile_template(source):
    env = ImmutableSandboxedEnvironment(trim_blocks=True,lstrip_blocks=True)
    env.globals["raise_exception"] = _raise_template_error
    env.filters["tojson"] = lambda value,**kwargs: json.dumps(value,ensure_ascii=False,**kwargs)
    return env.from_string(source)


def render_qwen4_exp_chat(template, messages, *, tools=None, enable_thinking=False,
                         add_generation_prompt=True, reasoning_effort=None):
    if not template:
        raise ValueError("Qwen4Exp requires its embedded GGUF chat template")
    normalized = []
    for message in messages:
        row = deepcopy(dict(message) if isinstance(message,Mapping) else message.model_dump())
        for call in row.get("tool_calls") or ():
            function = call.get("function",call)
            arguments = function.get("arguments")
            if isinstance(arguments,str):
                function["arguments"] = json.loads(arguments) if arguments.strip() else {}
            if function.get("arguments") is not None and not isinstance(function["arguments"],dict):
                raise ValueError("tool arguments must be an object")
        normalized.append(row)
    kwargs = dict(messages=normalized,tools=tools,enable_thinking=enable_thinking,
                  add_generation_prompt=add_generation_prompt)
    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort
    return _compile_template(template).render(**kwargs)


@dataclass(frozen=True)
class ParsedToolCall:
    name: str
    arguments: str
    raw_text: str


@dataclass(frozen=True)
class ParsedToolOutput:
    content: str
    tool_calls: tuple[ParsedToolCall,...]
    invalid_blocks: tuple[str,...] = ()


_BLOCK = re.compile(r"<tool_call>(.*?)</tool_call>",re.DOTALL)
_FUNCTION = re.compile(r"\s*<function=([^<>\s]+)>(.*?)</function>\s*",re.DOTALL)
_PARAMETER = re.compile(r"<parameter=([^<>\s]+)>(.*?)</parameter>",re.DOTALL)


def _parameter_value(text,schema):
    # The trained format wraps each parameter in one newline, not JSON quoting.
    if text.startswith("\n"):
        text = text[1:]
    if text.endswith("\n"):
        text = text[:-1]
    kind = schema.get("type")
    if kind=="string":
        return text
    if kind in ("integer","number","boolean","object","array","null"):
        value = json.loads(text)
        valid = {
            "integer":type(value) is int,
            "number":type(value) in (int,float),
            "boolean":type(value) is bool,
            "object":isinstance(value,dict),
            "array":isinstance(value,list),
            "null":value is None,
        }
        if not valid[kind]:
            raise ValueError("parameter type mismatch")
        return value
    try:
        return json.loads(text)
    except ValueError:
        return text


class Qwen4ExpToolParser:
    name = "qwen4exp_xml_parameters"
    requires_declared_tools = True
    capabilities = {
        "parser":name,"format":"qwen_xml_parameters","strict_malformed_blocks_rejected":True,
        "json_compatibility":True,"streaming_validation":"buffer_then_emit_json_argument_fragments",
        "strict_decoding_scope":"xml_function_parameters_with_declared_schemas",
        "required_tool_start_forcing_scope":"grammar_mask",
        "specific_tool_name_prefix_forcing_scope":"grammar_mask",
        "strict_tool_schema_prefix_anchor":False,
        "tool_call_close_repair":False,
    }

    def parse_with_initial_reasoning(self,text,*,tools=None,initially_open=False):
        result = self.parse("<think>"+text if initially_open else text,tools=tools)
        if initially_open:
            return ParsedToolOutput(result.content.removeprefix("<think>"),result.tool_calls,result.invalid_blocks)
        return result

    def parse(self,model_output,*,tools=None):
        text = str(model_output)
        thinking_spans = [(m.start(),m.end()) for m in
                          re.finditer(r"<think>.*?(?:</think>|$)",text,re.DOTALL)]
        masked = text
        for start,end in thinking_spans:
            masked = masked[:start]+" "*(end-start)+masked[end:]
        schemas = {}
        for tool in tools or ():
            function = tool.get("function",tool)
            schemas[function.get("name")] = function.get("parameters",{}).get("properties",{})
        calls,invalid,parts = [],[],[]
        cursor = 0
        for block in _BLOCK.finditer(masked):
            gap = text[cursor:block.start()]
            parts.append(gap)
            masked_gap = masked[cursor:block.start()]
            if "<tool_call>" in masked_gap or "</tool_call>" in masked_gap:
                invalid.append(gap)
            raw,body = block.group(0),block.group(1)
            try:
                match = _FUNCTION.fullmatch(body)
                if match:
                    name,args_text = match.groups()
                    arguments = {}
                    end = 0
                    for parameter in _PARAMETER.finditer(args_text):
                        key,value = parameter.groups()
                        if args_text[end:parameter.start()].strip() or key in arguments:
                            raise ValueError("invalid or duplicate parameter")
                        arguments[key] = _parameter_value(value,schemas.get(name,{}).get(key,{}))
                        end = parameter.end()
                    if args_text[end:].strip():
                        raise ValueError("unparsed function content")
                else:
                    obj = json.loads(body)
                    obj = obj.get("function",obj)
                    name,arguments = obj["name"],obj.get("arguments",{})
                    if isinstance(arguments,str):
                        arguments = json.loads(arguments)
                if not isinstance(name,str) or not name or not isinstance(arguments,dict):
                    raise ValueError("invalid tool call")
                calls.append(ParsedToolCall(name,json.dumps(arguments,ensure_ascii=False,allow_nan=False),raw))
            except (ValueError,TypeError,KeyError,AttributeError):
                invalid.append(raw)
                parts.append(raw)
            cursor = block.end()
        tail = text[cursor:]
        if "<tool_call>" in masked[cursor:] or "</tool_call>" in masked[cursor:]:
            invalid.append(tail)
        parts.append(tail)
        return ParsedToolOutput("".join(parts).strip(),tuple(calls),tuple(invalid))


class Qwen4ExpReasoningParser:
    def __init__(self,tokenizer):
        self.tokenizer = tokenizer

    def initially_open(self,prompt):
        assistant = str(prompt).rsplit("<|im_start|>assistant",1)[-1]
        return assistant.rfind("<think>") > assistant.rfind("</think>")

    def initially_open_ids(self,ids):
        return self.initially_open(self.tokenizer.decode(ids,skip_special=False))
