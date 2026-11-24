"""Streaming and final XML parsing must agree before tools may execute."""
import json

import pytest

from hipengine.server.chat_output import _parse_chat_tool_calls
from hipengine.server.tool_stream import XMLToolStream


TOOLS = [{'type': 'function', 'function': {'name': 'bash', 'parameters': {
    'type': 'object', 'properties': {'command': {'type': 'string'}},
    'required': ['command'],
}}}]
CALL = '<tool_call>\n<function=bash>\n<parameter=command>\npwd\n</parameter>\n</function>\n</tool_call>'


def _stream(text, width):
    stream = XMLToolStream(names=['bash'], string_typed=lambda name, key: True)
    events = []
    for offset in range(0, len(text), width):
        events.extend(stream.feed(text[offset:offset + width]))
    return stream, events


@pytest.mark.parametrize('width', [1, 2, 7, 10000])
@pytest.mark.parametrize('variant', ['parameter_space', 'function_space', 'compact', 'outer_tag_case'])
def test_xml_stream_and_final_parser_agree_on_name_spacing_and_compact_calls(width, variant):
    call = {
        'parameter_space': CALL.replace('parameter=command>', 'parameter= command \t>'),
        'function_space': CALL.replace('function=bash>', 'function= bash \t>'),
        'compact': CALL.replace('>\n', '>').replace('\n<', '<'),
        'outer_tag_case': CALL.replace('<tool_call>', '<ToOl_CaLl>').replace('</tool_call>', '</TOOL_CALL>'),
    }[variant]
    parsed = _parse_chat_tool_calls(call, tools=TOOLS)
    assert len(parsed.tool_calls) == 1
    stream, events = _stream(call, width)
    events.extend(stream.finish())
    assert [event[2] for event in events if event[2]] == ['bash']
    assert json.loads(''.join(event[3] for event in events if event[0] == 'tool')) == json.loads(parsed.tool_calls[0].arguments) == {'command': 'pwd'}


def test_fourteen_call_signature_is_reproduced_by_padded_final_parameter_name():
    # Synthetic reproduction of the recorded counters, not recovered model XML.
    text = '\n'.join([CALL] * 13 + [CALL.replace('parameter=command>', 'parameter=command >')])
    parsed = _parse_chat_tool_calls(text, tools=TOOLS)
    assert len(parsed.tool_calls) == 14
    stream, events = _stream(text, 1)
    events.extend(stream.finish())
    assert stream.state == 'text'
    for index in range(14):
        arguments = ''.join(event[3] for event in events if event[0] == 'tool' and event[1] == index)
        assert json.loads(arguments) == json.loads(parsed.tool_calls[index].arguments)


@pytest.mark.parametrize('key', ['command', ' command '])
def test_final_parser_rejects_duplicate_normalized_xml_parameters(key):
    text = CALL.replace('</function>', f'<parameter={key}>\necho overwritten\n</parameter>\n</function>')
    assert not _parse_chat_tool_calls(text, tools=TOOLS).tool_calls


@pytest.mark.parametrize('suffix', ['', '<parameter=command>', '<parameter=command >\nunfinished'])
def test_unfinished_xml_is_never_completed_by_stream_or_final_parser(suffix):
    text = '<tool_call>\n<function=bash>\n' + suffix
    stream, _events = _stream(text, 1)
    with pytest.raises(ValueError, match='incomplete'):
        list(stream.finish())
    assert not _parse_chat_tool_calls(text, tools=TOOLS).tool_calls
