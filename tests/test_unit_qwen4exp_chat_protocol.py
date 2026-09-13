import json
from pathlib import Path
from types import SimpleNamespace
import pytest

TEMPLATE = (Path(__file__).parent/"fixtures/qwen4exp_chat_template.jinja").read_text()
TOOLS = [{"type":"function","function":{"name":"lookup","parameters":{
    "type":"object","properties":{"key":{"type":"string"},"count":{"type":"integer"},
    "enabled":{"type":"boolean"},"options":{"type":"object"}},"required":["key"]}}}]


def test_embedded_template_tools_and_multiturn():
    from hipengine.chat.qwen4_exp import render_qwen4_exp_chat
    messages = [{"role":"system","content":"Be precise."},{"role":"user","content":"Find A"},
        {"role":"assistant","content":None,"tool_calls":[{"id":"one","type":"function",
         "function":{"name":"lookup","arguments":'{"key":"A","count":2}'}}]},
        {"role":"tool","tool_call_id":"one","content":"Found A"}]
    text = render_qwen4_exp_chat(TEMPLATE,messages,tools=TOOLS,enable_thinking=False)
    assert "# Tools\n" in text and "<IMPORTANT>" in text
    assert "<function=lookup>\n<parameter=key>\nA\n</parameter>" in text
    assert "<parameter=count>\n2\n</parameter>" in text
    assert "<|im_start|>user\n<tool_response>\nFound A\n</tool_response><|im_end|>" in text
    assert text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert messages[2]["tool_calls"][0]["function"]["arguments"] == '{"key":"A","count":2}'


def test_xml_typed_values_multiple_calls_and_visible_suffix():
    from hipengine.chat.qwen4_exp import Qwen4ExpToolParser
    text = ('Before\n<tool_call>\n<function=lookup>\n<parameter=key>\ntrue\n</parameter>\n'
            '<parameter=count>2</parameter><parameter=enabled>false</parameter>'
            '<parameter=options>{"items":[1,"二"]}</parameter></function>\n</tool_call>'
            '\n<tool_call><function=lookup><parameter=key>B</parameter></function></tool_call>\nAfter')
    result = Qwen4ExpToolParser().parse(text,tools=TOOLS)
    assert not result.invalid_blocks and len(result.tool_calls)==2
    assert json.loads(result.tool_calls[0].arguments)=={
        "key":"true","count":2,"enabled":False,"options":{"items":[1,"二"]}}
    assert "Before" in result.content and "After" in result.content


@pytest.mark.parametrize("body",[
    "<function=lookup><parameter=key>A</parameter><parameter=key>B</parameter></function>",
    "<function=lookup><parameter=count>no</parameter></function>",
    "<function=lookup><parameter=key>A</function>",
])
def test_malformed_xml_fails_closed(body):
    from hipengine.chat.qwen4_exp import Qwen4ExpToolParser
    result = Qwen4ExpToolParser().parse("<tool_call>"+body+"</tool_call>",tools=TOOLS)
    assert result.invalid_blocks and not result.tool_calls


def test_model_hook_and_generic_json_remain_supported():
    from hipengine.chat.qwen4_exp import Qwen4ExpToolParser
    from hipengine.server.api import _parse_chat_tool_calls_for_engine
    engine = SimpleNamespace(chat_tool_parser=Qwen4ExpToolParser())
    xml = "<tool_call><function=lookup><parameter=key>A</parameter></function></tool_call>"
    assert _parse_chat_tool_calls_for_engine(engine,xml,tools=TOOLS).tool_calls[0].name=="lookup"
    for target in (None,engine):
        result = _parse_chat_tool_calls_for_engine(target,
            '<tool_call>{"name":"lookup","arguments":{"key":"A"}}</tool_call>',tools=TOOLS)
        assert result.tool_calls[0].name=="lookup"
        assert json.loads(result.tool_calls[0].arguments)=={"key":"A"}


def test_structured_json_is_not_mistaken_for_bare_tool():
    from hipengine.chat.qwen4_exp import Qwen4ExpToolParser
    from hipengine.server.api import _parse_chat_tool_calls_for_engine
    engine = SimpleNamespace(chat_tool_parser=Qwen4ExpToolParser())
    text = '{"name":"lookup","arguments":{"key":"A"}}'
    result = _parse_chat_tool_calls_for_engine(engine,text,tools=None)
    assert not result.tool_calls and result.text==text
    result = _parse_chat_tool_calls_for_engine(engine,text,tools=TOOLS,allow_bare_json_tools=False)
    assert not result.tool_calls and result.text==text
    assert _parse_chat_tool_calls_for_engine(engine,text,tools=TOOLS).tool_calls
    literal = json.dumps({"example":"<tool_call><function=lookup></function></tool_call>"})
    result = _parse_chat_tool_calls_for_engine(engine,literal,tools=None)
    assert not result.tool_calls and result.text==literal


def test_reasoning_effort_reaches_embedded_template():
    from hipengine.chat.qwen4_exp import render_qwen4_exp_chat
    from hipengine.server.api import (
        ChatCompletionRequest,_thinking_control_from_request,_render_chat_prompt_with_model_protocol)
    request=ChatCompletionRequest(model="test",messages=[{"role":"user","content":"Hi"}],reasoning_effort="low")
    engine=SimpleNamespace(chat_template_reasoning_effort=True,
        render_chat_prompt=lambda messages,**kwargs: render_qwen4_exp_chat(TEMPLATE,messages,**kwargs))
    text=_render_chat_prompt_with_model_protocol(
        request,thinking=_thinking_control_from_request(request,chat_default_max_tokens=None),
        engine=engine,validate_tool_transcript=True)
    assert "Reasoning effort is set to low" in text
    assert text.endswith("<think>\n")


def test_lazy_engine_loads_model_protocol_before_first_render():
    from hipengine.chat.qwen4_exp import render_qwen4_exp_chat
    from hipengine.server.api import (
        ChatCompletionRequest,_thinking_control_from_request,_render_chat_prompt_with_model_protocol)
    request=ChatCompletionRequest(model="test",messages=[{"role":"user","content":"Hi"}],tools=TOOLS)
    owner=SimpleNamespace(render_chat_prompt=lambda messages,**kwargs:
                          render_qwen4_exp_chat(TEMPLATE,messages,**kwargs))
    calls=[]
    engine=SimpleNamespace(_text_generator=None,_get_text_generator=lambda: calls.append(True) or owner)
    text=_render_chat_prompt_with_model_protocol(
        request,thinking=_thinking_control_from_request(request,chat_default_max_tokens=None),
        engine=engine,validate_tool_transcript=True)
    assert calls==[True]
    assert "<function=example_function_name>" in text
