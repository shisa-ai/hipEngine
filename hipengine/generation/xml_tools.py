"""Prefix recognition for Qwen's XML function envelope (not general XML).

Parameter values are opaque text; schema validation belongs to the API parser.
Only structural suffixes at parameter boundaries are safe to force.
"""
from __future__ import annotations

import re


def xml_tool_prefix(text, *, names, mode, start, end, parallel, forbidden=()):
    """Return (valid, complete, branch, safe_close_suffix) for decoded text."""
    body = text.lstrip()
    seen = 0
    while True:
        if seen and body and not parallel:
            return False, False, 'invalid', ''
        marker = body.find(start)
        if marker < 0:
            if not body:
                return True, bool(seen), 'tool' if seen else 'undecided', ''
            partial = next((body[-n:] for n in range(min(len(body), len(start)), 0, -1)
                            if start.startswith(body[-n:])), '')
            if seen or mode == 'required':
                valid = start.startswith(body)
                return valid, False, 'tool', ''
            if any(value in body or value.startswith(body) for value in forbidden):
                return False, False, 'invalid', ''
            return True, False, 'tool' if partial else 'text', ''
        if marker and (seen or mode == 'required') and body[:marker].strip():
            return False, False, 'invalid', ''
        if any(value in body[:marker] for value in forbidden):
            return False, False, 'invalid', ''
        if seen and not parallel:
            return False, False, 'invalid', ''
        body = body[marker + len(start):].lstrip()
        headers = tuple('<function=' + name + '>' for name in names)
        if any(header.startswith(body) for header in headers):
            return True, False, 'tool', ''
        header = next((header for header in headers if body.startswith(header)), None)
        if header is None:
            return False, False, 'invalid', ''
        body = body[len(header):].lstrip()
        keys = set()
        while True:
            close = '</function>'
            if close.startswith(body):
                return True, False, 'tool', close[len(body):] + '\n' + end
            if body.startswith(close):
                body = body[len(close):].lstrip()
                if end.startswith(body):
                    if body != end:
                        return True, False, 'tool', end[len(body):]
                elif not body.startswith(end):
                    return False, False, 'invalid', ''
                body = body[len(end):].lstrip()
                seen += 1
                if not body:
                    return True, True, 'tool', ''
                break
            opening = '<parameter='
            if opening.startswith(body):
                return True, False, 'tool', ''
            if not body.startswith(opening):
                return False, False, 'invalid', ''
            rest = body[len(opening):]
            if '>' not in rest:
                return not bool(re.search(r'[<>\s]', rest)), False, 'tool', ''
            key, value = rest.split('>', 1)
            if not key or re.search(r'[<>\s]', key) or key in keys:
                return False, False, 'invalid', ''
            keys.add(key)
            closing = '</parameter>'
            where = value.find(closing)
            if where < 0:
                return True, False, 'tool', ''
            body = value[where + len(closing):].lstrip()
