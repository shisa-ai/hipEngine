"""Boundary contract for the pure chat-output parser owner.

The reasoning splitter, tool-output parsers, marker handling and the parser
dataclasses/constants were moved out of ``hipengine/server/api.py`` unchanged
into ``hipengine/server/chat_output.py``. These checks pin the properties the
move depends on:

* the owner imports only the standard library and neither the HTTP application
  nor any engine, model, runtime, dispatch or loading module;
* every moved symbol keeps the exact pre-move AST (the hashes below were taken
  from ``hipengine/server/api.py`` at the move base), and the parser names
  referenced by the application and the result-policy owner resolve to the
  identical objects;
* the parsers keep the byte-level results captured from the pre-move
  implementation: reasoning splits and source spans, chunk-split reasoning and
  terminal markers, prompt-open reasoning, terminal-marker stripping, JSON and
  Qwen XML tool envelopes (malformed, incomplete, duplicate and bare), string
  versus JSON argument typing, and tool-block source offsets.

The expected values are an oracle for behavior, not a restatement of the
module.
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from hipengine.server import chat_output

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "hipengine" / "server" / "chat_output.py"
API_PATH = REPO_ROOT / "hipengine" / "server" / "api.py"

CALL_ID = re.compile(r"call_[0-9a-f]{24}")

BLOCKED_IMPORT_PREFIXES = (
    "fastapi",
    "starlette",
    "torch",
    "hipengine.server.api",
    "hipengine.generation",
    "hipengine.runtime",
    "hipengine.models",
    "hipengine.kvcache",
    "hipengine.speculative",
    "hipengine.llm",
    "hipengine.kernels",
    "hipengine.dispatch",
    "hipengine.loading",
    "hipengine.quant",
)

# Every parser symbol the move put under the owner's ownership.
MOVED_NAMES = (
    "_CHAT_TEMPLATE_RESIDUE_MARKERS",
    "_CHAT_TEMPLATE_RESIDUE_ROLES",
    "_CHAT_TEMPLATE_TERMINAL_MARKERS",
    "_ParsedChatOutput",
    "_ParsedToolCall",
    "_REASONING_CLOSE_TAGS",
    "_REASONING_OPEN_TAGS",
    "_ReasoningPart",
    "_ReasoningSplit",
    "_ReasoningSplitter",
    "_TOOL_CALL_END_MARKER",
    "_TOOL_CALL_START_MARKER",
    "_ToolCallBlock",
    "_classify_chat_length_phase",
    "_find_marker",
    "_find_valid_tool_call_block",
    "_has_unclosed_tool_call",
    "_is_chat_template_control_residue",
    "_is_incomplete_duplicate_tool_prefix_control_residue",
    "_json_schema_resolves_to_string",
    "_looks_like_partial_json",
    "_parse_chat_tool_calls",
    "_parsed_tool_call_from_block_body",
    "_parsed_tool_call_from_function_xml",
    "_parsed_tool_call_from_json",
    "_parsed_tool_call_from_mapping",
    "_split_reasoning",
    "_strip_chat_template_terminal_edges",
    "_strip_chat_terminal_markers",
    "_strip_duplicate_tool_call_start",
    "_tag_suffix_len",
    "_tool_arguments_json",
    "_tool_call_name_prefix_text",
    "_tool_parameter_property_schema",
    "_valid_tool_call_blocks",
    "_xml_parameter_is_string_typed",
    "_xml_parameter_value",
)

# Owner-internal implementation detail; never imported from ``api``.
OWNER_ONLY_NAMES = ("_XML_FUNCTION_ENVELOPE_RE", "_XML_PARAMETER_RE")

# The subset ``hipengine.server.api`` imports for its own request handling.
# Result-policy consumers import the remaining parser helpers from this module
# directly.
API_APPLICATION_NAMES = (
    "_ParsedChatOutput",
    "_ParsedToolCall",
    "_ReasoningSplitter",
    "_TOOL_CALL_END_MARKER",
    "_TOOL_CALL_START_MARKER",
    "_classify_chat_length_phase",
    "_parse_chat_tool_calls",
    "_split_reasoning",
    "_strip_chat_terminal_markers",
    "_tool_arguments_json",
    "_xml_parameter_is_string_typed",
)

# The parser helpers the extracted result-policy owner imports directly.
RESULT_POLICY_NAMES = ("_find_marker", "_valid_tool_call_blocks")

# sha256(ast.dump(node, include_attributes=False))[:16] captured from
# ``hipengine/server/api.py`` before the move.
AST_HASHES = {
    "_CHAT_TEMPLATE_RESIDUE_MARKERS": "cdb982e8bdfac41b",
    "_CHAT_TEMPLATE_RESIDUE_ROLES": "c95b7d583eed8a80",
    "_CHAT_TEMPLATE_TERMINAL_MARKERS": "d5fee77afed4804b",
    "_ParsedChatOutput": "3e012f00baac7862",
    "_ParsedToolCall": "6f16e01fb474bb14",
    "_REASONING_CLOSE_TAGS": "a5edfa461ff2219b",
    "_REASONING_OPEN_TAGS": "78dc10b918ff4208",
    "_ReasoningPart": "14f0b6fbf165bd98",
    "_ReasoningSplit": "6755e243003296ca",
    "_ReasoningSplitter": "a67a573ee3e0ce62",
    "_TOOL_CALL_END_MARKER": "07e6d0f329e35148",
    "_TOOL_CALL_START_MARKER": "11794d34ec537426",
    "_ToolCallBlock": "d3ed65b566004f7a",
    "_XML_FUNCTION_ENVELOPE_RE": "c6149bb4e255d887",
    "_XML_PARAMETER_RE": "36740f1c76dce624",
    "_classify_chat_length_phase": "d4e13ecead37c5b8",
    "_find_marker": "b24d9ef0237aa5c9",
    "_find_valid_tool_call_block": "c5a29b242695e7c6",
    "_has_unclosed_tool_call": "2e74295acfddf749",
    "_is_chat_template_control_residue": "fcdbbd9652a594f3",
    "_is_incomplete_duplicate_tool_prefix_control_residue": "cdc960280aaaba5d",
    "_json_schema_resolves_to_string": "28d4aee2fa0d2f5d",
    "_looks_like_partial_json": "db29cbfd7f4b898b",
    "_parse_chat_tool_calls": "4499bf74b1707bd3",
    "_parsed_tool_call_from_block_body": "cb2955b4f42969a5",
    "_parsed_tool_call_from_function_xml": "2f4a6244f11ab789",
    "_parsed_tool_call_from_json": "7decb5f64b554148",
    "_parsed_tool_call_from_mapping": "fc670b2c2811fbda",
    "_split_reasoning": "89ffe8c56075c1d4",
    "_strip_chat_template_terminal_edges": "30d2a7429ecfd13f",
    "_strip_chat_terminal_markers": "8b1ee2193c58448a",
    "_strip_duplicate_tool_call_start": "ec340d354ea58f90",
    "_tag_suffix_len": "aac9e1dd0c5a1c0c",
    "_tool_arguments_json": "0068d0ce2604bb2e",
    "_tool_call_name_prefix_text": "ff64384e4fadbf82",
    "_tool_parameter_property_schema": "60a049c459f2f216",
    "_valid_tool_call_blocks": "b3baf570e61c6abd",
    "_xml_parameter_is_string_typed": "a567ddddab3d218d",
    "_xml_parameter_value": "6cfbce12f4d8bf02",
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "count": {"type": "integer"},
                    "flag": {"type": "boolean"},
                    "payload": {"type": "object"},
                },
            },
        },
    },
]

XML_MULTILINE = (
    "<tool_call>\n<function=lookup>\n<parameter=query>\nline one\nline two\n</parameter>\n"
    "<parameter=count>\n7\n</parameter>\n</function>\n</tool_call>"
)


THINK_OPEN = '<think>'
THINK_CLOSE = '</think>'


def _top_level_nodes(path: Path) -> dict[str, ast.AST]:
    nodes: dict[str, ast.AST] = {}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        name = getattr(node, "name", None)
        if name is None and isinstance(node, ast.Assign):
            targets = [target.id for target in node.targets if isinstance(target, ast.Name)]
            name = targets[0] if len(targets) == 1 else None
        if name is not None:
            nodes[name] = node
    return nodes


def _ast_hash(node: ast.AST) -> str:
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()[:16]


def _payload(parsed) -> dict:
    return {
        "text": parsed.text,
        "reasoning_initially_open": parsed.reasoning_initially_open,
        "tool_parser_name": parsed.tool_parser_name,
        "invalid_tool_call_blocks": list(parsed.invalid_tool_call_blocks),
        "tool_calls": [
            {
                "id": CALL_ID.sub("call_ID", call.id),
                "name": call.name,
                "arguments": call.arguments,
                "raw_text": call.raw_text,
            }
            for call in parsed.tool_calls
        ],
    }


def _call_payload(call) -> dict | None:
    if call is None:
        return None
    return {
        "id": CALL_ID.sub("call_ID", call.id),
        "name": call.name,
        "arguments": call.arguments,
        "raw_text": call.raw_text,
    }


def _parts(parts) -> list[list]:
    return [[part.field, part.text, part.source_start, part.source_end] for part in parts]


# --- module boundary -------------------------------------------------------


def test_owner_module_imports_only_the_standard_library() -> None:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "owner must not use relative imports"
            imported.append(node.module or "")
    for module in imported:
        assert not module.startswith(BLOCKED_IMPORT_PREFIXES), module
        assert module.split(".")[0] in {
            "__future__",
            "collections",
            "dataclasses",
            "json",
            "re",
            "typing",
            "uuid",
        }, module


def test_owner_module_defines_every_moved_symbol() -> None:
    nodes = _top_level_nodes(MODULE_PATH)
    for name in (*MOVED_NAMES, *OWNER_ONLY_NAMES):
        assert name in nodes, name


@pytest.mark.parametrize(("name", "expected"), sorted(AST_HASHES.items()))
def test_owner_ast_matches_pre_move_api_ast(name: str, expected: str) -> None:
    node = _top_level_nodes(MODULE_PATH)[name]
    assert _ast_hash(node) == expected


def test_api_imports_the_parser_names_its_application_uses() -> None:
    tree = ast.parse(API_PATH.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "hipengine.server.chat_output":
            imported.update(alias.name for alias in node.names)
    assert imported == set(API_APPLICATION_NAMES)
    assert imported <= set(MOVED_NAMES)

    defined = {
        name
        for name, node in _top_level_nodes(API_PATH).items()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Assign))
    }
    assert defined.isdisjoint(set(MOVED_NAMES) | set(OWNER_ONLY_NAMES))


@pytest.mark.parametrize("name", API_APPLICATION_NAMES)
def test_api_reexports_the_identical_object(name: str) -> None:
    import hipengine.server.api as api

    assert getattr(api, name) is getattr(chat_output, name)


def test_result_policy_imports_the_parser_names_its_application_uses() -> None:
    from hipengine.server import chat_result_policy

    for name in RESULT_POLICY_NAMES:
        assert getattr(chat_result_policy, name) is getattr(chat_output, name)


def test_owner_loads_without_http_or_engine_modules() -> None:
    script = f"""
import importlib.util
import sys
spec = importlib.util.spec_from_file_location("_chat_output_standalone", {str(MODULE_PATH)!r})
module = importlib.util.module_from_spec(spec)
sys.modules["_chat_output_standalone"] = module
spec.loader.exec_module(module)
blocked = [name for name in sys.modules if name.startswith({BLOCKED_IMPORT_PREFIXES!r})]
assert not blocked, blocked
assert "hipengine" not in sys.modules
print(module._TOOL_CALL_START_MARKER)
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "<tool_call>"


def test_owner_package_import_pulls_no_http_application() -> None:
    script = """
import sys
import hipengine.server.chat_output
blocked = [name for name in sys.modules if name.startswith(
    ("fastapi", "starlette", "torch", "hipengine.server.api", "hipengine.generation")
)]
assert not blocked, blocked
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )
    assert result.returncode == 0, result.stderr


# --- reasoning splitting ---------------------------------------------------


@pytest.mark.parametrize(
    ("text", "initially_open", "content", "reasoning"),
    [
        ("hello", False, "hello", ""),
        (f"a{THINK_OPEN}why{THINK_CLOSE}b", False, "ab", "why"),
        ("why</think>yes", True, "yes", "why"),
        ("why", True, "", "why"),
        (f"a{THINK_OPEN}why", False, "a", "why"),
        ("<|channel>thought\nwhy<channel|>yes", False, "yes", "why"),
        ("", False, "", ""),
        ("answer<|im_end|>", False, "answer", ""),
        ("\n\nOK<|im_end|>", False, "\n\nOK", ""),
    ],
)
def test_reasoning_split_matches_pre_move_oracle(
    text: str, initially_open: bool, content: str, reasoning: str
) -> None:
    split = chat_output._split_reasoning(text, initially_open=initially_open)
    assert (split.content, split.reasoning_content) == (content, reasoning)


@pytest.mark.parametrize(
    ("chunks", "expected_events", "expected_finish"),
    [
        (
            ["a<th", "ink>b", "</think>c"],
            [
                [["content", "a", 0, 1]],
                [["reasoning_content", "b", 8, 9]],
                [["content", "c", 17, 18]],
            ],
            [],
        ),
        (
            ["<|channel>tho", "ught\nwhy<chan", "nel|>yes"],
            [
                [],
                [["reasoning_content", "why", 18, 21]],
                [["content", "yes", 31, 34]],
            ],
            [],
        ),
        (
            ["a</thi", "nk>b"],
            [
                [["content", "a", 0, 1]],
                [["content", "b", 9, 10]],
            ],
            [],
        ),
        (
            ["", "", "x", ""],
            [[], [], [["content", "x", 0, 1]], []],
            [],
        ),
    ],
)
def test_splitter_preserves_source_spans_across_chunk_splits(
    chunks, expected_events, expected_finish
) -> None:
    splitter = chat_output._ReasoningSplitter()
    assert [_parts(splitter.feed_parts(chunk)) for chunk in chunks] == expected_events
    assert _parts(splitter.finish_parts()) == expected_finish
    assert splitter.pending_text_length == 0


def test_splitter_prompt_open_reasoning_keeps_offsets() -> None:
    splitter = chat_output._ReasoningSplitter(initially_open=True)
    assert _parts(splitter.feed_parts("why</thi")) == [["reasoning_content", "why", 0, 3]]
    assert _parts(splitter.feed_parts("nk>yes")) == [["content", "yes", 11, 14]]
    assert _parts(splitter.finish_parts()) == []


@pytest.mark.parametrize(
    ("chunks", "held_length", "held_start"),
    [
        (["answer<|im_", "end|>"], 10, 6),
        (["answer", "<|im_end|>"], 10, 6),
        (["x<tu", "rn|>"], 7, 1),
    ],
)
def test_splitter_holds_split_terminal_marker_until_finish(chunks, held_length, held_start) -> None:
    splitter = chat_output._ReasoningSplitter()
    for chunk in chunks:
        splitter.feed_parts(chunk)
    assert splitter.pending_text_length == held_length
    assert splitter.pending_source_start == held_start
    # finish_parts strips the held marker and emits an empty content part at
    # the marker offset, exactly as the pre-move splitter did.
    assert _parts(splitter.finish_parts()) == [["content", "", held_start, held_start]]


def test_splitter_legacy_tuple_surface_is_unchanged() -> None:
    assert chat_output._ReasoningSplitter().feed("a") == [("content", "a")]
    splitter = chat_output._ReasoningSplitter()
    splitter.feed("a")
    splitter.feed("b")
    assert splitter.finish() == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("OK<|im_end|>", "OK"),
        ("\n\nOK<|im_end|>", "\n\nOK"),
        ("x<turn|><|im_end|>", "x"),
        ("plain", "plain"),
        ("x<|im_end|> ", "x"),
        ("<|im_end|>", ""),
        ("x<|im_end|>\n", "x"),
        ("a<turn|>b", "a<turn|>b"),
    ],
)
def test_strip_chat_terminal_markers_oracle(text: str, expected: str) -> None:
    assert chat_output._strip_chat_terminal_markers(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("<|im_end|>body<|im_end|>", "body"),
        (" <|im_start|>assistant\nbody\n<|im_end|> ", "<|im_start|>assistant\nbody"),
        ("<|im_start|>assistant", ""),
        ("plain", "plain"),
        ("", ""),
        ("<|endoftext|>", ""),
        ("<|im_end|>a<|im_end|>b", "a<|im_end|>b"),
    ],
)
def test_strip_chat_template_terminal_edges_oracle(text: str, expected: str) -> None:
    assert chat_output._strip_chat_template_terminal_edges(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("<|im_start|>assistant", True),
        ("<|im_start|>assistant\n", True),
        ("<|im_end|>", True),
        ("<|endoftext|>", True),
        ("plain", False),
        ("<|im_start|>", True),
        ("<|im_start|>user", True),
        ("<|im_start|>assistantx", False),
        (" <|im_end|> ", True),
    ],
)
def test_control_residue_oracle(text: str, expected: bool) -> None:
    assert chat_output._is_chat_template_control_residue(text) is expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("answer", "answer"),
        ("a<thi", "reasoning"),
        (f"a{THINK_OPEN}why", "reasoning"),
        (f"a{THINK_OPEN}why</thi", "closing_think"),
        ('<tool_call>{"name":"x"', "tool_call"),
        ('<tool_call>{"name":"x"}</tool_call>', "answer"),
        ('{"a":', "structured"),
        ("[1,", "structured"),
        ('{"a":1}', "answer"),
        ("<|channel>thought\nCheck.", "reasoning"),
        ("<|channel>thought\nCheck.<chan", "closing_think"),
        ("", "answer"),
    ],
)
def test_classify_chat_length_phase_oracle(text: str, expected: str) -> None:
    assert chat_output._classify_chat_length_phase(text) == expected


@pytest.mark.parametrize(
    ("text", "tag", "expected"),
    [
        ("<th", THINK_OPEN, 3),
        ("<thi", THINK_OPEN, 4),
        ("think", THINK_OPEN, 0),
        ("x", THINK_OPEN, 0),
        ("", THINK_OPEN, 0),
        ("<|im_", "<|im_end|>", 5),
        ("<turn|>", "<turn|>", 0),
        ("turn|", "<turn|>", 0),
        ("a", "</tool_call>", 0),
    ],
)
def test_tag_suffix_len_oracle(text: str, tag: str, expected: int) -> None:
    assert chat_output._tag_suffix_len(text, tag) == expected


# --- tool output parsing ---------------------------------------------------

PLAIN_PAYLOAD = {
    "text": "hello",
    "reasoning_initially_open": False,
    "tool_parser_name": None,
    "invalid_tool_call_blocks": [],
    "tool_calls": [],
}


@pytest.mark.parametrize(
    ("text", "tools", "expected"),
    [
        ("hello", None, PLAIN_PAYLOAD),
        (
            '<tool_call>{"name":"lookup","arguments":{"q":"x"}}</tool_call>',
            None,
            {
                "text": "",
                "reasoning_initially_open": False,
                "tool_parser_name": None,
                "invalid_tool_call_blocks": [],
                "tool_calls": [
                    {
                        "id": "call_ID",
                        "name": "lookup",
                        "arguments": '{"q":"x"}',
                        "raw_text": '<tool_call>{"name":"lookup","arguments":{"q":"x"}}</tool_call>',
                    }
                ],
            },
        ),
        (
            '{"name":"lookup","arguments":{"q":"x"}}',
            None,
            {
                "text": "",
                "reasoning_initially_open": False,
                "tool_parser_name": None,
                "invalid_tool_call_blocks": [],
                "tool_calls": [
                    {
                        "id": "call_ID",
                        "name": "lookup",
                        "arguments": '{"q":"x"}',
                        "raw_text": '{"name":"lookup","arguments":{"q":"x"}}',
                    }
                ],
            },
        ),
        (
            "[1,2]",
            None,
            {**PLAIN_PAYLOAD, "text": "[1,2]"},
        ),
        (
            '<tool_call>{oops}</tool_call>',
            None,
            {**PLAIN_PAYLOAD, "text": "<tool_call>{oops}</tool_call>"},
        ),
        (
            "<tool_call>\n<function=lookup>\n<parameter=query>\nx\n</parameter>\n</tool_call>",
            TOOLS,
            {
                **PLAIN_PAYLOAD,
                "text": (
                    "<tool_call>\n<function=lookup>\n<parameter=query>\nx\n</parameter>\n"
                    "</tool_call>"
                ),
            },
        ),
        (
            "<tool_call>\n<function=lookup>\nnoise\n<parameter=query>\nx\n</parameter>\n"
            "</function>\n</tool_call>",
            TOOLS,
            {
                **PLAIN_PAYLOAD,
                "text": (
                    "<tool_call>\n<function=lookup>\nnoise\n<parameter=query>\nx\n</parameter>\n"
                    "</function>\n</tool_call>"
                ),
            },
        ),
        (
            '<tool_call><tool_call>{"name":"lookup","arguments":{}}</tool_call>',
            None,
            {
                "text": "",
                "reasoning_initially_open": False,
                "tool_parser_name": None,
                "invalid_tool_call_blocks": [],
                "tool_calls": [
                    {
                        "id": "call_ID",
                        "name": "lookup",
                        "arguments": "{}",
                        "raw_text": '<tool_call><tool_call>{"name":"lookup","arguments":{}}</tool_call>',
                    }
                ],
            },
        ),
        (
            '<tool_call>{"name":"lookup","arguments":<|im_start|>assistant'
            '<tool_call>{"name":"lookup","arguments":{}}</tool_call>',
            None,
            {
                "text": "",
                "reasoning_initially_open": False,
                "tool_parser_name": None,
                "invalid_tool_call_blocks": [],
                "tool_calls": [
                    {
                        "id": "call_ID",
                        "name": "lookup",
                        "arguments": "{}",
                        "raw_text": '<tool_call>{"name":"lookup","arguments":{}}</tool_call>',
                    }
                ],
            },
        ),
        (
            'before <tool_call>{"name":"a","arguments":{}}</tool_call> middle '
            '<tool_call>{"name":"b","arguments":{}}</tool_call> after',
            None,
            {
                "text": "before  middle  after",
                "reasoning_initially_open": False,
                "tool_parser_name": None,
                "invalid_tool_call_blocks": [],
                "tool_calls": [
                    {
                        "id": "call_ID",
                        "name": "a",
                        "arguments": "{}",
                        "raw_text": '<tool_call>{"name":"a","arguments":{}}</tool_call>',
                    },
                    {
                        "id": "call_ID",
                        "name": "b",
                        "arguments": "{}",
                        "raw_text": '<tool_call>{"name":"b","arguments":{}}</tool_call>',
                    },
                ],
            },
        ),
        (
            '<tool_call>{oops}</tool_call>mid<tool_call>{"name":"b","arguments":{}}</tool_call>',
            None,
            {
                "text": "<tool_call>{oops}</tool_call>mid",
                "reasoning_initially_open": False,
                "tool_parser_name": None,
                "invalid_tool_call_blocks": [],
                "tool_calls": [
                    {
                        "id": "call_ID",
                        "name": "b",
                        "arguments": "{}",
                        "raw_text": '<tool_call>{"name":"b","arguments":{}}</tool_call>',
                    }
                ],
            },
        ),
        (
            '<tool_call>{"name":"lookup","arguments":{}}</tool_call><tool_call>{"name":',
            None,
            {
                "text": '<tool_call>{"name":',
                "reasoning_initially_open": False,
                "tool_parser_name": None,
                "invalid_tool_call_blocks": [],
                "tool_calls": [
                    {
                        "id": "call_ID",
                        "name": "lookup",
                        "arguments": "{}",
                        "raw_text": '<tool_call>{"name":"lookup","arguments":{}}</tool_call>',
                    }
                ],
            },
        ),
        (
            "<|im_start|>assistant\n<tool_call>{\"name\":\"lookup\",\"arguments\":{}}"
            "</tool_call>\n<|im_end|>",
            None,
            {
                "text": "",
                "reasoning_initially_open": False,
                "tool_parser_name": None,
                "invalid_tool_call_blocks": [],
                "tool_calls": [
                    {
                        "id": "call_ID",
                        "name": "lookup",
                        "arguments": "{}",
                        "raw_text": '<tool_call>{"name":"lookup","arguments":{}}</tool_call>',
                    }
                ],
            },
        ),
    ],
)
def test_parse_chat_tool_calls_oracle(text, tools, expected) -> None:
    assert _payload(chat_output._parse_chat_tool_calls(text, tools=tools)) == expected


def test_xml_multiline_string_value_stays_verbatim() -> None:
    parsed = chat_output._parse_chat_tool_calls(XML_MULTILINE, tools=TOOLS)
    assert len(parsed.tool_calls) == 1
    call = parsed.tool_calls[0]
    assert call.name == "lookup"
    assert call.arguments == '{"query":"line one\\nline two","count":7}'
    assert call.raw_text == XML_MULTILINE


def test_xml_parsing_without_declared_tools_decodes_json_values() -> None:
    parsed = chat_output._parse_chat_tool_calls(XML_MULTILINE, tools=None)
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].arguments == '{"query":"line one\\nline two","count":7}'


def test_xml_untyped_values_decode_as_json_with_raw_string_fallback() -> None:
    raw = (
        "<tool_call>\n<function=lookup>\n<parameter=count>\n7\n</parameter>\n"
        "<parameter=flag>\ntrue\n</parameter>\n<parameter=payload>\n{\"a\": 1}\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    parsed = chat_output._parse_chat_tool_calls(raw, tools=TOOLS)
    assert parsed.tool_calls[0].arguments == '{"count":7,"flag":true,"payload":{"a":1}}'

    malformed = (
        "<tool_call>\n<function=lookup>\n<parameter=payload>\n{oops}\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    parsed = chat_output._parse_chat_tool_calls(malformed, tools=TOOLS)
    assert parsed.tool_calls[0].arguments == '{"payload":"{oops}"}'


@pytest.mark.parametrize(
    ("schema", "expected"),
    [
        ({"type": "string"}, '"7"'),
        ({"type": ["string", "null"]}, '"7"'),
        ({"type": ["integer", "null"]}, "7"),
        ({"enum": ["a", "b"]}, '"7"'),
        ({"oneOf": [{"type": "string"}, {"type": "integer"}]}, '"7"'),
        ({"anyOf": [{"type": "integer"}]}, "7"),
        ({"allOf": [{"type": "string"}]}, '"7"'),
        ({"allOf": [{"type": "string"}, {"type": "integer"}]}, "7"),
        ({}, "7"),
    ],
)
def test_xml_argument_typing_follows_tool_schema(schema, expected) -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "parameters": {"type": "object", "properties": {"k": schema}},
            },
        }
    ]
    raw = "<tool_call>\n<function=lookup>\n<parameter=k>\n7\n</parameter>\n</function>\n</tool_call>"
    parsed = chat_output._parse_chat_tool_calls(raw, tools=tools)
    assert parsed.tool_calls[0].arguments == f'{{"k":{expected}}}'


@pytest.mark.parametrize("value", ["7", "true", '{"a":1}', "multi\nline"])
def test_string_typed_xml_values_are_not_json_decoded(value: str) -> None:
    raw = f"<tool_call>\n<function=lookup>\n<parameter=query>\n{value}\n</parameter>\n</function>\n</tool_call>"
    parsed = chat_output._parse_chat_tool_calls(raw, tools=TOOLS)
    assert parsed.tool_calls[0].arguments == json.dumps(
        {"query": value}, ensure_ascii=False, separators=(",", ":")
    )


@pytest.mark.parametrize(
    ("text", "tools"),
    [
        (
            "<tool_call>\n<function=lookup>\n<parameter=>\nx\n</parameter>\n</function>\n"
            "</tool_call>",
            TOOLS,
        ),
        (
            "<tool_call>\n<function=>\n<parameter=query>\nx\n</parameter>\n</function>\n"
            "</tool_call>",
            TOOLS,
        ),
        ('<tool_call>{"name":"","arguments":{}}</tool_call>', None),
        ('<tool_call>{"arguments":{}}</tool_call>', None),
        ('<tool_call>{"name":"x","arguments":</tool_call>', None),
    ],
)
def test_malformed_or_incomplete_envelopes_stay_visible_text(text, tools) -> None:
    parsed = chat_output._parse_chat_tool_calls(text, tools=tools)
    assert parsed.tool_calls == ()
    assert parsed.text == text


@pytest.mark.parametrize(
    ("text", "tools", "expected"),
    [
        (
            'before <tool_call>{"name":"a","arguments":{}}</tool_call> middle '
            '<tool_call>{"name":"b","arguments":{}}</tool_call> after',
            None,
            [[7, 57, "a"], [65, 115, "b"]],
        ),
        (
            '<tool_call>{oops}</tool_call>mid<tool_call>{"name":"b","arguments":{}}</tool_call>',
            None,
            [[32, 82, "b"]],
        ),
        (
            "<|im_start|>assistant\n<tool_call>{\"name\":\"lookup\",\"arguments\":{}}"
            "</tool_call>\n<|im_end|>",
            None,
            [[22, 77, "lookup"]],
        ),
        (XML_MULTILINE, TOOLS, [[0, 136, "lookup"]]),
        ('<tool_call>{"name":"x"', None, []),
    ],
)
def test_valid_tool_call_blocks_report_exact_source_offsets(text, tools, expected) -> None:
    blocks = chat_output._valid_tool_call_blocks(text, tools=tools)
    assert [[block.start, block.end, block.parsed.name] for block in blocks] == expected
    for block in blocks:
        assert text[block.start : block.end] == block.parsed.raw_text


def test_parsed_tool_call_identity_and_frozen_contract() -> None:
    call = chat_output._ParsedToolCall(
        id="call_x", name="lookup", arguments="{}", raw_text="RAW"
    )
    assert dataclasses.is_dataclass(call)
    assert call == chat_output._ParsedToolCall(
        id="call_x", name="lookup", arguments="{}", raw_text="RAW"
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        call.name = "other"  # type: ignore[misc]

    parsed = chat_output._ParsedChatOutput(text="t", tool_calls=(call,))
    assert parsed.reasoning_initially_open is False
    assert parsed.tool_parser_name is None
    assert parsed.invalid_tool_call_blocks == ()


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ('{"a": 1}', '{"a":1}'),
        ("hello", '"hello"'),
        ("42", "42"),
        ({"b": 2, "a": 1}, '{"b":2,"a":1}'),
        (None, "{}"),
        ([1, "x"], '[1,"x"]'),
        (True, "true"),
        ("", '""'),
    ],
)
def test_tool_arguments_json_oracle(value, expected) -> None:
    assert chat_output._tool_arguments_json(value) == expected


@pytest.mark.parametrize(
    ("raw", "raw_text", "expected"),
    [
        ('{"name":"x","arguments":{"a":1}}', "RAW", {"name": "x", "arguments": '{"a":1}', "raw_text": "RAW"}),
        ('{"function":{"name":"x","arguments":"{\\"a\\":1}"}}', "RAW", {"name": "x", "arguments": '{"a":1}', "raw_text": "RAW"}),
        ('{"name":"x"}', "", {"name": "x", "arguments": "{}", "raw_text": ""}),
        ("", "RAW", None),
        ("[1]", "RAW", None),
        ("{oops}", "RAW", None),
        ('{"name":"","arguments":{}}', "RAW", None),
    ],
)
def test_parsed_tool_call_from_json_oracle(raw, raw_text, expected) -> None:
    call = chat_output._parsed_tool_call_from_json(raw, raw_text=raw_text)
    payload = _call_payload(call)
    if expected is not None:
        expected = {"id": "call_ID", **expected}
    assert payload == expected


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"name": "x", "arguments": {"a": 1}}, {"name": "x", "arguments": '{"a":1}', "raw_text": "R"}),
        (
            {"name": "outer", "function": {"name": "inner", "arguments": {}}},
            {"name": "inner", "arguments": "{}", "raw_text": "R"},
        ),
        ({"name": "x", "arguments": "plain"}, {"name": "x", "arguments": '"plain"', "raw_text": "R"}),
        ({"name": "x", "arguments": [1]}, {"name": "x", "arguments": "[1]", "raw_text": "R"}),
        ({"name": 5}, None),
        ({"name": "x", "function": 5}, {"name": "x", "arguments": "{}", "raw_text": "R"}),
    ],
)
def test_parsed_tool_call_from_mapping_oracle(payload, expected) -> None:
    call = chat_output._parsed_tool_call_from_mapping(payload, raw_text="R")
    payload_out = _call_payload(call)
    if expected is not None:
        expected = {"id": "call_ID", **expected}
    assert payload_out == expected


@pytest.mark.parametrize(
    ("raw", "expected_arguments"),
    [
        ("nope", None),
        ("<function=>\nx\n</function>", None),
        (
            "<function=lookup>\n<parameter=q>\nv\n</parameter>\ntail\n</function>",
            None,
        ),
        (
            "<function=lookup>\n<parameter=query>\nmulti\nline\n</parameter>\n</function>",
            '{"query":"multi\\nline"}',
        ),
        (
            "<function=lookup>\n<parameter=count>\n7\n</parameter>\n</function>",
            '{"count":7}',
        ),
    ],
)
def test_parsed_tool_call_from_function_xml_oracle(raw, expected_arguments) -> None:
    call = chat_output._parsed_tool_call_from_function_xml(raw, raw_text="R", tools=TOOLS)
    if expected_arguments is None:
        assert call is None
    else:
        assert call is not None
        assert call.arguments == expected_arguments
        assert call.raw_text == "R"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('<tool_call>{"a":1}', '{"a":1}'),
        (' <TOOL_CALL>{"a":1}', '{"a":1}'),
        ("plain", "plain"),
        ("<tool_call>", ""),
        ('<tool_call>  {"a":1}', '{"a":1}'),
    ],
)
def test_strip_duplicate_tool_call_start_oracle(text: str, expected: str) -> None:
    assert chat_output._strip_duplicate_tool_call_start(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('<tool_call>{"name":"lookup","arguments":<|im_start|>assistant', True),
        ("plain", False),
        ('<tool_call>{"name":"other","arguments":', False),
        ('<tool_call>{"name":"lookup","arguments":{', False),
        ('<tool_call>{"name":"lookup","arguments":', False),
    ],
)
def test_incomplete_duplicate_prefix_residue_oracle(text: str, expected: bool) -> None:
    assert (
        chat_output._is_incomplete_duplicate_tool_prefix_control_residue(
            text, tool_name="lookup"
        )
        is expected
    )


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("lookup", '<tool_call>{"name":"lookup","arguments":'),
        ("a b", '<tool_call>{"name":"a b","arguments":'),
        ("", '<tool_call>{"name":"","arguments":'),
    ],
)
def test_tool_call_name_prefix_text_oracle(name: str, expected: str) -> None:
    assert chat_output._tool_call_name_prefix_text(name) == expected


@pytest.mark.parametrize(
    ("text", "partial", "unclosed"),
    [
        ("", False, False),
        ("x", False, False),
        ("{", True, False),
        ("{}", False, False),
        ("[1", True, False),
        ("[1]", False, False),
        ('{"a":1}', False, False),
        ("{oops", True, False),
        ("<tool_call>", False, True),
        ("</tool_call>", False, False),
        ("<tool_call>x</tool_call>", False, False),
        ("<tool_cal", False, True),
    ],
)
def test_partial_json_and_unclosed_tool_call_helpers(text, partial, unclosed) -> None:
    assert chat_output._looks_like_partial_json(text) is partial
    assert chat_output._has_unclosed_tool_call(text) is unclosed
