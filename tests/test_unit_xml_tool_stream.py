import json

import pytest

from hipengine.server.tool_stream import XMLToolStream
from test_unit_xml_tool_constraint import CALL


@pytest.mark.parametrize('width', [1, 2, 7, 10000])
@pytest.mark.parametrize('compact', [False, True])
def test_xml_stream_preserves_arguments_across_boundaries(width, compact):
    value = 'echo "quoted"\ncat file.py\n'
    envelope = CALL.replace('>\n', '>').replace('\n<', '<') if compact else CALL
    text = 'Inspecting.\n' + envelope.replace('pwd', value if not compact else value.rstrip('\n'))
    stream = XMLToolStream(names=['bash'], string_typed=lambda name, key: True)
    events = []
    for offset in range(0, len(text), width):
        events.extend(stream.feed(text[offset:offset + width]))
    events.extend(stream.finish())
    assert ''.join(e[3] for e in events if e[0] == 'content') == 'Inspecting.\n'
    assert [e[2] for e in events if e[2]] == ['bash']
    assert json.loads(''.join(e[3] for e in events if e[0] == 'tool')) == {'command': value if not compact else value.rstrip('\n')}


def test_xml_stream_emits_arguments_before_parameter_closes():
    stream = XMLToolStream(names=['bash'], string_typed=lambda name, key: True)
    events = list(stream.feed(CALL.split('pwd')[0] + 'echo live'))
    assert ''.join(e[3] for e in events) == '{"command":"echo live'
    with pytest.raises(ValueError, match='incomplete'):
        list(stream.finish())


def test_xml_stream_nonstring_waits_and_parallel_calls_keep_indexes():
    stream = XMLToolStream(names=['bash'], string_typed=lambda name, key: False)
    events = list(stream.feed(CALL.replace('pwd', '[1,2]') + '\n' + CALL.replace('pwd', 'true')))
    events.extend(stream.finish())
    for index, expected in [(0, [1, 2]), (1, True)]:
        assert json.loads(''.join(e[3] for e in events if e[0] == 'tool' and e[1] == index)) == {'command': expected}
