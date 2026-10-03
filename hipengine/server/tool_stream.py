"""Incremental Qwen XML to OpenAI argument deltas.

Only string-typed values stream before their parameter closes. Other values wait
for JSON decoding. The caller must validate the final call before a success
finish event; clients must not execute unfinished calls.
"""
from __future__ import annotations

import json
import re


class XMLToolStream:
    def __init__(self, *, names, string_typed):
        self.names = set(names)
        self.string_typed = string_typed
        self.pending = ''
        self.state = 'text'
        self.index = -1
        self.name = ''
        self.key = ''
        self.keys = set()
        self.is_string = False
        self.buffered = False
        self.content = ''

    def feed(self, text):
        """Yield (field, tool_index, name, text) as source bytes arrive."""
        self.pending += text
        if self.buffered:
            return
        while self.pending:
            if self.state == 'text':
                markers = ('<tool_call>', '<|im_end|>', '<|endoftext|>')
                positions = [(self.pending.find(m), m) for m in markers if m in self.pending]
                if positions:
                    pos, marker = min(positions)
                    if pos:
                        self.content += self.pending[:pos]
                        yield 'content', -1, '', self.pending[:pos]
                    self.pending = self.pending[pos + len(marker):]
                    if marker == '<tool_call>':
                        self.state = 'function'
                    continue
                hold = max((n for m in markers for n in range(1, min(len(m), len(self.pending)) + 1)
                            if m.startswith(self.pending[-n:])), default=0)
                emit = len(self.pending) - hold
                if emit:
                    self.content += self.pending[:emit]
                    yield 'content', -1, '', self.pending[:emit]
                    self.pending = self.pending[emit:]
                return
            if self.state == 'function':
                self.pending = self.pending.lstrip()
                if not ('<function='.startswith(self.pending) or self.pending.startswith('<function=')):
                    # Legacy JSON and parser-repair envelopes keep the existing
                    # buffered compatibility path, without fabricated deltas.
                    self.buffered = True
                    return
                match = re.match(r'<function=([^<>\s]+)>', self.pending)
                if match is None:
                    return
                self.name = match[1]
                if self.name not in self.names:
                    raise ValueError('unknown streamed tool name')
                self.index += 1
                self.keys = set()
                self.pending = self.pending[match.end():]
                self.state = 'parameter'
                yield 'tool', self.index, self.name, '{'
                continue
            if self.state == 'parameter':
                self.pending = self.pending.lstrip()
                if self.pending.startswith('</function>'):
                    self.pending = self.pending[len('</function>'):]
                    self.state = 'end'
                    yield 'tool', self.index, '', '}'
                    continue
                match = re.match(r'<parameter=([^<>\s]+)>', self.pending)
                if match is None:
                    return
                self.key = match[1]
                if self.key in self.keys:
                    raise ValueError('duplicate streamed tool parameter')
                prefix = ',' if self.keys else ''
                self.keys.add(self.key)
                self.is_string = self.string_typed(self.name, self.key)
                self.pending = self.pending[match.end():]
                self.state = 'value_start'
                yield 'tool', self.index, '', prefix + json.dumps(self.key, ensure_ascii=False) + ':' + ('"' if self.is_string else '')
                continue
            if self.state == 'value_start':
                if self.pending.startswith('\n'):
                    self.pending = self.pending[1:]
                self.state = 'value'
                continue
            if self.state == 'value':
                closing = '</parameter>'
                pos = self.pending.find(closing)
                if pos >= 0:
                    value = self.pending[:pos].removesuffix('\n')
                    if self.is_string:
                        fragment = json.dumps(value, ensure_ascii=False)[1:-1] + '"'
                    else:
                        try:
                            value = json.loads(value)
                        except ValueError:
                            pass
                        fragment = json.dumps(value, ensure_ascii=False, separators=(',', ':'))
                    yield 'tool', self.index, '', fragment
                    self.pending = self.pending[pos + len(closing):]
                    self.state = 'parameter'
                    continue
                if self.is_string:
                    hold = max((n for n in range(1, min(len(closing), len(self.pending)) + 1)
                                if closing.startswith(self.pending[-n:])), default=0)
                    # Hold the optional newline before the closing marker too.
                    if self.pending[:len(self.pending) - hold].endswith('\n'):
                        hold += 1
                    emit = len(self.pending) - hold
                    if emit:
                        yield 'tool', self.index, '', json.dumps(self.pending[:emit], ensure_ascii=False)[1:-1]
                        self.pending = self.pending[emit:]
                return
            if self.state == 'end':
                self.pending = self.pending.lstrip()
                if not self.pending.startswith('</tool_call>'):
                    return
                self.pending = self.pending[len('</tool_call>'):]
                self.state = 'text'

    def finish(self):
        if self.state != 'text':
            raise ValueError('incomplete streamed XML tool call')
        text, self.pending = self.pending, ''
        if text:
            yield 'content', -1, '', text
