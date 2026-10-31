"""Boundary contract for the JSON Schema subset/value validators.

The validators were moved out of ``hipengine/server/api.py`` unchanged into
``hipengine/server/json_schema.py``. These checks pin the three properties the
move depends on: the module has no HTTP application or engine/model/runtime
dependency, the validator names referenced by the application and the
result-policy owner still resolve to the same objects, and the validators keep
byte-identical error tuples, messages,
numeric and boolean rules, local JSON Pointer resolution (including cycle
detection), JSON-value equality and regular-expression semantics.

The expected results below were captured from ``hipengine/server/api.py``
before the move (and re-checked against that pre-move implementation after the
move), so they are an oracle for behavior, not a restatement of the module.
"""

from __future__ import annotations

import ast
from decimal import Decimal
import json
from pathlib import Path
import subprocess
import sys

import pytest

from hipengine.server import json_schema

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "hipengine" / "server" / "json_schema.py"
API_PATH = REPO_ROOT / "hipengine" / "server" / "api.py"

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

# The subset ``hipengine.server.api`` imports for its own request handling.
# Result-policy consumers import the remaining validator from this module
# directly.
API_APPLICATION_NAMES = (
    "_JSON_SCHEMA_ANNOTATION_KEYWORDS",
    "_validate_json_schema_subset",
)

# The validator the extracted result-policy owner imports directly.
RESULT_POLICY_NAMES = ("_validate_json_schema_value",)

PRIVATE_NAMES = (
    "_JSON_SCHEMA_ANNOTATION_KEYWORDS",
    "_JSON_SCHEMA_SUPPORTED_KEYS",
    "_JSON_SCHEMA_SUPPORTED_TYPES",
    "_json_schema_pointer_token",
    "_json_schema_resolve_local_ref",
    "_json_schema_type_matches",
    "_primary_json_schema_type",
    "_schema_array_items_unique",
    "_schema_decimal_number",
    "_schema_finite_number",
    "_schema_json_value_kind",
    "_schema_json_values_equal",
    "_schema_nonnegative_int",
    "_schema_number_is_multiple_of",
    "_schema_positive_finite_number",
    "_validate_json_schema_subset",
    "_validate_json_schema_value",
)

SCHEMA_CASES = [
    ('empty', {}, None),
    (
        'object_basic',
        {
            'type': 'object',
            'properties': {'a': {'type': 'string'}},
            'required': ['a'],
        },
        None,
    ),
    ('type_list', {'type': ['string', 'null']}, None),
    (
        'type_list_bad_item',
        {
            'type': ['string', 5],
        },
        ('s.type', 's.type contains unsupported type 5'),
    ),
    ('type_not_str_or_list', {'type': 5}, ('s.type', 's.type must be a string or list of strings')),
    ('type_unsupported', {'type': 'wat'}, ('s.type', "s.type contains unsupported type 'wat'")),
    ('type_list_non_string', {'type': [None]}, ('s.type', 's.type contains unsupported type None')),
    ('enum_not_array', {'enum': 'x'}, ('s.enum', 's.enum must be an array')),
    ('enum_array', {'enum': [1, 2, 'a']}, None),
    (
        'unsupported_key',
        {
            'unsupported': True,
        },
        ('s.unsupported', 's.unsupported is not supported by hipEngine JSON schema subset'),
    ),
    (
        'annotation_only',
        {
            'title': 't',
            'description': 'd',
            'format': 'date',
            'examples': [1],
            'default': 1,
        },
        None,
    ),
    ('defs_ref', {'$defs': {'a': {'type': 'string'}}, '$ref': '#/$defs/a'}, None),
    (
        'definitions_ref',
        {
            'definitions': {'a': {'type': 'string'}},
            '$ref': '#/definitions/a',
        },
        None,
    ),
    (
        'ref_self_cycle',
        {
            '$defs': {'a': {'$ref': '#/$defs/a'}},
            '$ref': '#/$defs/a',
        },
        ('s.$defs.a.$ref', 's.$defs.a.$ref cycle is not supported'),
    ),
    (
        'ref_two_cycle',
        {
            '$defs': {'a': {'$ref': '#/$defs/b'}, 'b': {'$ref': '#/$defs/a'}},
            '$ref': '#/$defs/a',
        },
        ('s.$defs.a.$ref', 's.$defs.a.$ref cycle is not supported'),
    ),
    (
        'ref_remote',
        {
            '$ref': 'http://example.com/s.json',
        },
        ('s.$ref', 's.$ref only local JSON schema $ref values are supported'),
    ),
    ('ref_empty', {'$ref': ''}, ('s.$ref', 's.$ref $ref must be a non-empty string')),
    ('ref_blank', {'$ref': '   '}, ('s.$ref', 's.$ref $ref must be a non-empty string')),
    (
        'ref_non_pointer',
        {
            '$ref': '#foo',
        },
        ('s.$ref', 's.$ref $ref must be # or a JSON Pointer starting with #/'),
    ),
    (
        'ref_missing',
        {
            '$ref': '#/missing',
        },
        ('s.$ref', "s.$ref $ref target '#/missing' does not exist"),
    ),
    (
        'ref_escape_slash',
        {
            'a/b': {'type': 'string'},
            '$ref': '#/a~1b',
        },
        ('s.a/b', 's.a/b is not supported by hipEngine JSON schema subset'),
    ),
    (
        'ref_escape_tilde',
        {
            'a~b': {'type': 'string'},
            '$ref': '#/a~0b',
        },
        ('s.a~b', 's.a~b is not supported by hipEngine JSON schema subset'),
    ),
    (
        'ref_bad_escape',
        {
            '$ref': '#/a~2b',
        },
        ('s.$ref', "s.$ref invalid JSON Pointer escape in 'a~2b'"),
    ),
    ('ref_not_string', {'$ref': 5}, ('s.$ref', 's.$ref must be a string')),
    (
        'ref_percent_encoded',
        {
            'a b': {'type': 'string'},
            '$ref': '#/a%20b',
        },
        ('s.a b', 's.a b is not supported by hipEngine JSON schema subset'),
    ),
    (
        'ref_array_index',
        {
            'arr': [
                {'type': 'string'},
                {'type': 'integer'},
            ],
            '$ref': '#/arr/1',
        },
        ('s.arr', 's.arr is not supported by hipEngine JSON schema subset'),
    ),
    (
        'ref_array_index_oob',
        {
            'arr': [
                {'type': 'string'},
            ],
            '$ref': '#/arr/9',
        },
        ('s.arr', 's.arr is not supported by hipEngine JSON schema subset'),
    ),
    (
        'ref_array_index_non_decimal',
        {
            'arr': [
                {'type': 'string'},
            ],
            '$ref': '#/arr/-1',
        },
        ('s.arr', 's.arr is not supported by hipEngine JSON schema subset'),
    ),
    (
        'ref_array_index_leading_zero',
        {
            'arr': [
                {'type': 'string'},
                {'type': 'integer'},
            ],
            '$ref': '#/arr/01',
        },
        ('s.arr', 's.arr is not supported by hipEngine JSON schema subset'),
    ),
    (
        'ref_string_child',
        {
            's': 'abc',
            '$ref': '#/s/0',
        },
        ('s.s', 's.s is not supported by hipEngine JSON schema subset'),
    ),
    (
        'ref_target_not_schema',
        {
            'a': 5,
            '$ref': '#/a',
        },
        ('s.a', 's.a is not supported by hipEngine JSON schema subset'),
    ),
    ('ref_root', {'type': 'object', '$ref': '#'}, ('s.$ref', 's.$ref cycle is not supported')),
    (
        'allof_empty',
        {
            'allOf': [],
        },
        ('s.allOf', 's.allOf must be a non-empty array of schema objects'),
    ),
    (
        'allof_not_array',
        {
            'allOf': 'x',
        },
        ('s.allOf', 's.allOf must be a non-empty array of schema objects'),
    ),
    (
        'allof_bad_item',
        {
            'allOf': [
                {'type': 'string'},
                5,
            ],
        },
        ('s.allOf[1]', 's.allOf[1] must be an object'),
    ),
    ('anyof_ok', {'anyOf': [{'type': 'string'}]}, None),
    ('oneof_ok', {'oneOf': [{'type': 'string'}, {'type': 'integer'}]}, None),
    ('not_bad', {'not': 5}, ('s.not', 's.not must be an object')),
    (
        'conditional',
        {
            'if': {'type': 'string'},
            'then': {'minLength': 1},
            'else': {'maxLength': 3},
        },
        None,
    ),
    (
        'conditional_bad',
        {
            'if': {'type': 'string'},
            'then': 5,
        },
        ('s.then', 's.then must be an object'),
    ),
    (
        'properties_not_object',
        {
            'properties': [],
        },
        ('s.properties', 's.properties must be an object'),
    ),
    (
        'properties_bad_subschema',
        {
            'properties': {'a': 5},
        },
        ('s.properties.a', 's.properties.a must be an object'),
    ),
    (
        'pattern_properties_bad_regex',
        {
            'patternProperties': {'[': {}},
        },
        ('s.patternProperties.[', 's.patternProperties.[ must be a valid regular expression: unterminated character set at position 0'),
    ),
    ('pattern_properties_ok', {'patternProperties': {'^a': {'type': 'string'}}}, None),
    (
        'pattern_properties_non_string_key',
        {
            'patternProperties': {5: {}},
        },
        ('s.patternProperties', 's.patternProperties keys must be strings'),
    ),
    ('property_names', {'propertyNames': {'pattern': '^a'}}, None),
    (
        'property_names_bad',
        {
            'propertyNames': 5,
        },
        ('s.propertyNames', 's.propertyNames must be an object'),
    ),
    (
        'required_not_array',
        {
            'required': 'a',
        },
        ('s.required', 's.required must be an array of strings'),
    ),
    (
        'required_non_string',
        {
            'required': [1],
        },
        ('s.required', 's.required must contain only strings'),
    ),
    (
        'dependent_required_not_object',
        {
            'dependentRequired': [],
        },
        ('s.dependentRequired', 's.dependentRequired must be an object'),
    ),
    (
        'dependent_required_bad',
        {
            'dependentRequired': {'a': 'b'},
        },
        ('s.dependentRequired.a', 's.dependentRequired.a must be an array of strings'),
    ),
    (
        'dependent_required_non_string',
        {
            'dependentRequired': {'a': [1]},
        },
        ('s.dependentRequired.a', 's.dependentRequired.a must contain only strings'),
    ),
    (
        'dependent_required_key_non_string',
        {
            'dependentRequired': {5: ['a']},
        },
        ('s.dependentRequired', 's.dependentRequired keys must be strings'),
    ),
    (
        'dependent_schemas_bad',
        {
            'dependentSchemas': {'a': 5},
        },
        ('s.dependentSchemas.a', 's.dependentSchemas.a must be an object'),
    ),
    (
        'dependent_schemas_not_object',
        {
            'dependentSchemas': [],
        },
        ('s.dependentSchemas', 's.dependentSchemas must be an object'),
    ),
    ('additional_true', {'additionalProperties': True}, None),
    ('additional_schema', {'additionalProperties': {'type': 'string'}}, None),
    (
        'additional_bad',
        {
            'additionalProperties': 5,
        },
        ('s.additionalProperties', 's.additionalProperties must be a boolean or object in hipEngine JSON schema subset'),
    ),
    ('items_bad', {'items': 5}, ('s.items', 's.items must be an object')),
    ('contains_bad', {'contains': 5}, ('s.contains', 's.contains must be an object')),
    (
        'min_properties_negative',
        {
            'minProperties': -1,
        },
        ('s.minProperties', 's.minProperties must be a non-negative integer'),
    ),
    (
        'min_properties_bool',
        {
            'minProperties': True,
        },
        ('s.minProperties', 's.minProperties must be a non-negative integer'),
    ),
    (
        'min_properties_float',
        {
            'minProperties': 1.0,
        },
        ('s.minProperties', 's.minProperties must be a non-negative integer'),
    ),
    ('max_items_ok', {'maxItems': 2}, None),
    ('unique_items_bad', {'uniqueItems': 1}, ('s.uniqueItems', 's.uniqueItems must be a boolean')),
    ('pattern_not_string', {'pattern': 5}, ('s.pattern', 's.pattern must be a string')),
    (
        'pattern_bad_regex',
        {
            'pattern': '[',
        },
        ('s.pattern', 's.pattern must be a valid regular expression: unterminated character set at position 0'),
    ),
    ('minimum_not_number', {'minimum': '1'}, ('s.minimum', 's.minimum must be a finite number')),
    ('minimum_bool', {'minimum': True}, ('s.minimum', 's.minimum must be a finite number')),
    (
        'minimum_inf',
        {
            'minimum': float("inf"),
        },
        ('s.minimum', 's.minimum must be a finite number'),
    ),
    (
        'multiple_of_zero',
        {
            'multipleOf': 0,
        },
        ('s.multipleOf', 's.multipleOf must be a positive finite number'),
    ),
    (
        'multiple_of_negative',
        {
            'multipleOf': -1,
        },
        ('s.multipleOf', 's.multipleOf must be a positive finite number'),
    ),
    ('multiple_of_ok', {'multipleOf': 2}, None),
    ('defs_not_object', {'$defs': 5}, ('s.$defs', 's.$defs must be an object')),
    ('defs_bad_name', {'$defs': {5: {}}}, ('s.$defs', 's.$defs keys must be strings')),
    ('defs_bad_subschema', {'$defs': {'a': 5}}, ('s.$defs.a', 's.$defs.a must be an object')),
    (
        'deep_nested',
        {
            'properties': {'a': {'items': {'properties': {'b': {'pattern': '^x'}}}}},
        },
        None,
    ),
    ('ref_defs_escape_slash', {'$defs': {'a/b': {'type': 'string'}}, '$ref': '#/$defs/a~1b'}, None),
    ('ref_defs_escape_tilde', {'$defs': {'a~b': {'type': 'string'}}, '$ref': '#/$defs/a~0b'}, None),
    ('ref_defs_percent', {'$defs': {'a b': {'type': 'string'}}, '$ref': '#/$defs/a%20b'}, None),
    (
        'ref_defs_array',
        {
            '$defs': {'list': [{'type': 'string'}]},
            '$ref': '#/$defs/list/0',
        },
        ('s.$defs.list', 's.$defs.list must be an object'),
    ),
    ('type_empty_list', {'type': []}, None),
    ('enum_empty', {'enum': []}, None),
    ('additional_false', {'additionalProperties': False}, None),
    (
        'annotation_nested',
        {
            'properties': {'a': {'title': 't', 'format': 'date', 'default': 1}},
        },
        None,
    ),
    (
        'deep_nested_bad',
        {
            'properties': {'a': {'items': {'properties': {'b': {'pattern': 5}}}}},
        },
        ('s.properties.a.items.properties.b.pattern', 's.properties.a.items.properties.b.pattern must be a string'),
    ),
]

VALUE_CASES = [
    ('string_pattern_search', 'abc', {'type': 'string', 'pattern': 'b'}, None),
    (
        'string_pattern_search_fail',
        'ac',
        {
            'type': 'string',
            'pattern': 'b',
        },
        '$ does not match pattern',
    ),
    (
        'pattern_properties_search',
        {
            'abc': 'x',
        },
        {
            'type': 'object',
            'patternProperties': {'b': {'type': 'integer'}},
        },
        "$.abc does not match schema type 'integer'",
    ),
    (
        'oneof_overlap_numbers',
        1,
        {
            'oneOf': [
                {'type': 'number'},
                {'type': 'integer'},
            ],
        },
        '$ must match exactly one allowed schema',
    ),
    (
        'contains_min_zero',
        [
            1,
            2,
        ],
        {
            'type': 'array',
            'contains': {'type': 'string'},
            'minContains': 0,
        },
        None,
    ),
    ('max_contains_without_contains', [1, 2, 3], {'type': 'array', 'maxContains': 1}, None),
    (
        'ref_in_allof',
        'abc',
        {
            '$defs': {'a': {'type': 'string'}},
            'allOf': [
                {'$ref': '#/$defs/a'},
                {'minLength': 2},
            ],
        },
        None,
    ),
    (
        'ref_pointer_in_property',
        {
            'a': 5,
        },
        {
            '$defs': {'n': {'type': 'integer'}},
            'type': 'object',
            'properties': {'a': {'$ref': '#/$defs/n'}},
        },
        None,
    ),
    ('object_pass', {'a': 1}, {'type': 'object', 'properties': {'a': {'type': 'integer'}}}, None),
    ('object_required_missing', {}, {'type': 'object', 'required': ['a']}, '$.a is required'),
    (
        'object_property_type',
        {
            'a': 'x',
        },
        {
            'type': 'object',
            'properties': {'a': {'type': 'integer'}},
        },
        "$.a does not match schema type 'integer'",
    ),
    ('object_not_mapping', [1], {'type': 'object'}, "$ does not match schema type 'object'"),
    (
        'object_additional_false_extra',
        {
            'a': 1,
            'b': 2,
        },
        {
            'type': 'object',
            'properties': {'a': {}},
            'additionalProperties': False,
        },
        '$.b is not allowed',
    ),
    (
        'object_additional_false_pattern',
        {
            'abc': 1,
        },
        {
            'type': 'object',
            'patternProperties': {'^a': {}},
            'additionalProperties': False,
        },
        None,
    ),
    (
        'object_additional_schema',
        {
            'a': 1,
            'b': 'x',
        },
        {
            'type': 'object',
            'properties': {'a': {}},
            'additionalProperties': {'type': 'string'},
        },
        None,
    ),
    (
        'object_additional_schema_fail',
        {
            'a': 1,
            'b': 2,
        },
        {
            'type': 'object',
            'properties': {'a': {}},
            'additionalProperties': {'type': 'string'},
        },
        "$.b does not match schema type 'string'",
    ),
    (
        'object_property_names',
        {
            'ab': 1,
        },
        {
            'type': 'object',
            'propertyNames': {'pattern': '^a'},
        },
        None,
    ),
    (
        'object_property_names_fail',
        {
            'zb': 1,
        },
        {
            'type': 'object',
            'propertyNames': {'pattern': '^a'},
        },
        '$.zb property name is invalid',
    ),
    (
        'object_dependent_required',
        {
            'a': 1,
        },
        {
            'type': 'object',
            'dependentRequired': {'a': ['b']},
        },
        '$.b is required when a is present',
    ),
    (
        'object_dependent_required_ok',
        {
            'a': 1,
            'b': 2,
        },
        {
            'type': 'object',
            'dependentRequired': {'a': ['b']},
        },
        None,
    ),
    (
        'object_dependent_schema',
        {
            'a': 1,
            'b': 2,
        },
        {
            'type': 'object',
            'dependentSchemas': {'a': {'required': ['c']}},
        },
        '$.c is required',
    ),
    (
        'object_min_properties',
        {},
        {
            'type': 'object',
            'minProperties': 1,
        },
        '$ must have at least 1 properties',
    ),
    (
        'object_max_properties',
        {
            'a': 1,
            'b': 2,
        },
        {
            'type': 'object',
            'maxProperties': 1,
        },
        '$ must have at most 1 properties',
    ),
    ('array_pass', [1, 2], {'type': 'array', 'items': {'type': 'integer'}}, None),
    (
        'array_item_fail',
        [
            1,
            'x',
        ],
        {
            'type': 'array',
            'items': {'type': 'integer'},
        },
        "$[1] does not match schema type 'integer'",
    ),
    ('array_not_list', 'x', {'type': 'array'}, "$ does not match schema type 'array'"),
    ('array_min_items', [1], {'type': 'array', 'minItems': 2}, '$ must have at least 2 items'),
    ('array_max_items', [1, 2, 3], {'type': 'array', 'maxItems': 2}, '$ must have at most 2 items'),
    (
        'array_unique_number',
        [
            1,
            1.0,
        ],
        {
            'type': 'array',
            'uniqueItems': True,
        },
        '$ must contain unique items',
    ),
    ('array_unique_kinds', [1, True], {'type': 'array', 'uniqueItems': True}, None),
    (
        'array_unique_objects',
        [
            {'a': 1},
            {'a': 1},
        ],
        {
            'type': 'array',
            'uniqueItems': True,
        },
        '$ must contain unique items',
    ),
    ('array_contains_ok', [1, 'x'], {'type': 'array', 'contains': {'type': 'string'}}, None),
    (
        'array_contains_fail',
        [
            1,
            2,
        ],
        {
            'type': 'array',
            'contains': {'type': 'string'},
        },
        '$ must contain at least 1 matching items',
    ),
    (
        'array_min_contains',
        [
            1,
            'x',
        ],
        {
            'type': 'array',
            'contains': {'type': 'string'},
            'minContains': 2,
        },
        '$ must contain at least 2 matching items',
    ),
    (
        'array_max_contains',
        [
            1,
            'x',
            'y',
        ],
        {
            'type': 'array',
            'contains': {'type': 'string'},
            'maxContains': 1,
        },
        '$ must contain at most 1 matching items',
    ),
    (
        'string_min_length',
        'a',
        {
            'type': 'string',
            'minLength': 2,
        },
        '$ must have at least 2 characters',
    ),
    (
        'string_max_length',
        'abc',
        {
            'type': 'string',
            'maxLength': 2,
        },
        '$ must have at most 2 characters',
    ),
    ('string_pattern_ok', 'abc', {'type': 'string', 'pattern': '^a'}, None),
    ('string_pattern_fail', 'xbc', {'type': 'string', 'pattern': '^a'}, '$ does not match pattern'),
    ('string_wrong_type', 5, {'type': 'string'}, "$ does not match schema type 'string'"),
    ('integer_float', 1.0, {'type': 'integer'}, "$ does not match schema type 'integer'"),
    ('integer_bool', True, {'type': 'integer'}, "$ does not match schema type 'integer'"),
    ('integer_ok', 3, {'type': 'integer'}, None),
    ('number_bool', True, {'type': 'number'}, "$ does not match schema type 'number'"),
    ('number_inf', float("inf"), {'type': 'number'}, "$ does not match schema type 'number'"),
    ('number_nan', float("nan"), {'type': 'number'}, "$ does not match schema type 'number'"),
    ('minimum_fail', 0.5, {'type': 'number', 'minimum': 1}, '$ must be >= 1'),
    ('minimum_pass', 1, {'type': 'number', 'minimum': 1}, None),
    ('maximum_fail', 2.5, {'type': 'number', 'maximum': 2}, '$ must be <= 2'),
    ('exclusive_minimum_fail', 1, {'type': 'number', 'exclusiveMinimum': 1}, '$ must be > 1'),
    ('exclusive_maximum_fail', 2, {'type': 'number', 'exclusiveMaximum': 2}, '$ must be < 2'),
    ('exclusive_maximum_formatting', 1.25, {'type': 'number', 'exclusiveMaximum': 1.5}, None),
    ('multiple_of_decimal_ok', 0.3, {'type': 'number', 'multipleOf': 0.1}, None),
    (
        'multiple_of_decimal_fail',
        0.35,
        {
            'type': 'number',
            'multipleOf': 0.1,
        },
        '$ must be a multiple of 0.1',
    ),
    ('multiple_of_zero_ok', 7, {'type': 'number', 'multipleOf': 0}, None),
    ('multiple_of_int_fail', 7, {'type': 'number', 'multipleOf': 2}, '$ must be a multiple of 2'),
    ('number_min_with_bool_value', True, {'type': 'boolean', 'minimum': 1}, None),
    ('enum_bool_vs_int', True, {'enum': [1, 2]}, '$ is not one of the allowed enum values'),
    ('enum_number_equality', 1.0, {'enum': [1]}, None),
    ('enum_object_equality', {'a': 1}, {'enum': [{'a': 1.0}]}, None),
    ('enum_fail', 'z', {'enum': ['a', 'b']}, '$ is not one of the allowed enum values'),
    ('const_object_order', {'b': 2, 'a': [1, 2]}, {'const': {'a': [1, 2], 'b': 2}}, None),
    ('const_number', 1.0, {'const': 1}, None),
    ('const_fail', 2, {'const': 1}, '$ does not match const'),
    ('allof_pass', 'abc', {'allOf': [{'type': 'string'}, {'minLength': 2}]}, None),
    (
        'allof_fail',
        'a',
        {
            'allOf': [
                {'type': 'string'},
                {'minLength': 2},
            ],
        },
        '$ must have at least 2 characters',
    ),
    ('anyof_pass_second', 5, {'anyOf': [{'type': 'string'}, {'type': 'integer'}]}, None),
    (
        'anyof_fail',
        True,
        {
            'anyOf': [
                {'type': 'string'},
                {'type': 'integer'},
            ],
        },
        '$ does not match any allowed schema',
    ),
    ('oneof_pass', 'abc', {'oneOf': [{'type': 'string'}, {'type': 'integer'}]}, None),
    (
        'oneof_two_matches',
        'abc',
        {
            'oneOf': [
                {'type': 'string'},
                {'minLength': 1},
            ],
        },
        '$ must match exactly one allowed schema',
    ),
    (
        'oneof_zero',
        True,
        {
            'oneOf': [
                {'type': 'string'},
                {'type': 'integer'},
            ],
        },
        '$ must match exactly one allowed schema',
    ),
    ('not_pass', 'abc', {'not': {'type': 'integer'}}, None),
    ('not_fail', 5, {'not': {'type': 'integer'}}, '$ matches a disallowed schema'),
    (
        'conditional_then_fail',
        'abc',
        {
            'if': {'type': 'string'},
            'then': {'minLength': 5},
            'else': {'type': 'integer'},
        },
        '$ must have at least 5 characters',
    ),
    (
        'conditional_else_fail',
        1.5,
        {
            'if': {'type': 'string'},
            'then': {'minLength': 5},
            'else': {'type': 'integer'},
        },
        "$ does not match schema type 'integer'",
    ),
    (
        'conditional_then_pass',
        'abcdef',
        {
            'if': {'type': 'string'},
            'then': {'minLength': 5},
            'else': {'type': 'integer'},
        },
        None,
    ),
    ('ref_value_pass', 'abc', {'$defs': {'a': {'type': 'string'}}, '$ref': '#/$defs/a'}, None),
    (
        'ref_value_fail',
        5,
        {
            '$defs': {'a': {'type': 'string'}},
            '$ref': '#/$defs/a',
        },
        "$ does not match schema type 'string'",
    ),
    (
        'ref_value_cycle',
        'x',
        {
            '$defs': {'a': {'$ref': '#/$defs/a'}},
            '$ref': '#/$defs/a',
        },
        '$ has cyclic schema reference',
    ),
    ('ref_value_broken', 'x', {'$ref': '#/missing'}, '$ has invalid schema reference'),
    (
        'ref_value_root',
        {
            'a': 1,
        },
        {
            '$defs': {'a': {'type': 'string'}},
            '$ref': '#/$defs/a',
        },
        "$ does not match schema type 'string'",
    ),
    ('type_list_null', None, {'type': ['string', 'null']}, None),
    (
        'type_list_fail',
        5,
        {
            'type': ['string', 'null'],
        },
        "$ does not match schema type ['string', 'null']",
    ),
    ('implicit_object', {'a': 1}, {'properties': {'a': {'type': 'integer'}}}, None),
    ('implicit_array', [1], {'items': {'type': 'integer'}}, None),
    ('implicit_string', 'a', {'minLength': 2}, '$ must have at least 2 characters'),
    ('implicit_number', 1, {'minimum': 2}, '$ must be >= 2'),
    ('implicit_boolean', True, {'const': False}, '$ does not match const'),
    (
        'nested_path',
        {
            'a': [
                {'b': 'x'},
            ],
        },
        {
            'type': 'object',
            'properties': {
                'a': {
                    'type': 'array',
                    'items': {'type': 'object', 'properties': {'b': {'type': 'integer'}}},
                },
            },
        },
        "$.a[0].b does not match schema type 'integer'",
    ),
]

POINTER_CASES = [
    ('a', 'a', None),
    ('a~0b', 'a~b', None),
    ('a~1b', 'a/b', None),
    ('a~2b', None, "invalid JSON Pointer escape in 'a~2b'"),
    ('a~', None, "invalid JSON Pointer escape in 'a~'"),
    ('~1', '/', None),
    ('', '', None),
    ('~01', '~1', None),
    ('a/b', 'a/b', None),
    ('~~0', None, "invalid JSON Pointer escape in '~~0'"),
    ('0', '0', None),
]

RESOLVE_CASES = [
    (
        {'a': {'type': 'string'}},
        '#',
        {'a': {'type': 'string'}},
        '$',
        (),
        None,
    ),
    (
        {'a': {'type': 'string'}},
        '#/a',
        {'type': 'string'},
        '$.a',
        ('a',),
        None,
    ),
    (
        {'a': {'b': {'type': 'integer'}}},
        '#/a/b',
        {'type': 'integer'},
        '$.a.b',
        ('a', 'b'),
        None,
    ),
    (
        {'a/b': {'type': 'string'}},
        '#/a~1b',
        {'type': 'string'},
        '$.a/b',
        ('a/b',),
        None,
    ),
    (
        {'a~b': {'type': 'string'}},
        '#/a~0b',
        {'type': 'string'},
        '$.a~b',
        ('a~b',),
        None,
    ),
    (
        {'a b': {'type': 'string'}},
        '#/a%20b',
        {'type': 'string'},
        '$.a b',
        ('a b',),
        None,
    ),
    (
        {'arr': [{'type': 'string'}, {'type': 'integer'}]},
        '#/arr/1',
        {'type': 'integer'},
        '$.arr[1]',
        ('arr', '1'),
        None,
    ),
    (
        {'arr': [{'type': 'string'}]},
        '#/arr/9',
        None,
        '$.arr',
        ('arr', '9'),
        "$ref target '#/arr/9' does not exist",
    ),
    (
        {'arr': [{'type': 'string'}]},
        '#/arr/-1',
        None,
        '$.arr',
        ('arr', '-1'),
        "$ref target '#/arr/-1' does not exist",
    ),
    (
        {'arr': [{'type': 'string'}, {'type': 'integer'}]},
        '#/arr/01',
        {'type': 'integer'},
        '$.arr[1]',
        ('arr', '01'),
        None,
    ),
    (
        {'s': 'abc'},
        '#/s/0',
        None,
        '$.s',
        ('s', '0'),
        "$ref target '#/s/0' does not exist",
    ),
    (
        {'a': 5},
        '#/a',
        None,
        '$.a',
        ('a',),
        '$ref target must be a schema object',
    ),
    (
        {'a': {}},
        '#/a~2b',
        None,
        '$',
        (),
        "invalid JSON Pointer escape in 'a~2b'",
    ),
    (
        {},
        '',
        None,
        '$',
        (),
        '$ref must be a non-empty string',
    ),
    (
        {},
        '  ',
        None,
        '$',
        (),
        '$ref must be a non-empty string',
    ),
    (
        {},
        'http://x',
        None,
        '$',
        (),
        'only local JSON schema $ref values are supported',
    ),
    (
        {},
        '#foo',
        None,
        '$',
        (),
        '$ref must be # or a JSON Pointer starting with #/',
    ),
    (
        {},
        '#/missing',
        None,
        '$',
        ('missing',),
        "$ref target '#/missing' does not exist",
    ),
]

NONNEGATIVE_INT_CASES = [
    (0, 0),
    (1, 1),
    (-1, None),
    (True, None),
    (1.0, None),
    ('1', None),
    (None, None),
    (1180591620717411303424, 1180591620717411303424),
]

FINITE_NUMBER_CASES = [
    (0, 0.0),
    (1.5, 1.5),
    (-2, -2.0),
    (True, None),
    (float("nan"), None),
    (float("inf"), None),
    ('1', None),
    (None, None),
]

POSITIVE_FINITE_CASES = [
    (0, None),
    (0.5, 0.5),
    (-1, None),
    (float("inf"), None),
    (True, None),
    (2, 2.0),
    (None, None),
]

DECIMAL_CASES = [
    (1, Decimal('1')),
    (1.5, Decimal('1.5')),
    (True, None),
    ('x', None),
    (None, None),
    (float("nan"), None),
    (float("inf"), None),
    (Decimal('1.5'), None),
]

UNIQUE_CASES = [
    ([1, 2], True),
    ([1, 1.0], False),
    ([1, True], True),
    ([
        {'a': 1},
        {'a': 1.0},
    ], False),
    ([
        [1],
        [1.0],
    ], False),
    (['a', 'a'], False),
    ([None, None], False),
    ([], True),
]

KIND_CASES = [
    (None, 'null'),
    (True, 'boolean'),
    (False, 'boolean'),
    (0, 'number'),
    (1.5, 'number'),
    (float("nan"), 'unknown'),
    ('x', 'string'),
    ([], 'array'),
    ({}, 'object'),
    ((1,), 'unknown'),
    (Decimal('1.5'), 'unknown'),
]

MULTIPLE_CASES = [
    (0.3, 0.1, True),
    (0.35, 0.1, False),
    (7, 2, False),
    (7, 0, True),
    (7, -2, True),
    (True, 2, True),
    (2, True, True),
    (None, 2, True),
    (2, None, True),
    (1.0, 0.5, True),
]

EQUAL_CASES = [
    (1, 1.0, True),
    (True, 1, False),
    (True, True, True),
    (None, None, True),
    ('a', 'a', True),
    ('1', 1, False),
    ({'a': 1, 'b': 2}, {'b': 2.0, 'a': 1}, True),
    ({'a': 1}, {'a': 1, 'b': 2}, False),
    ([1, 2], [1.0, 2], True),
    ([1, 2], [1, 2, 3], False),
    (float("nan"), float("nan"), False),
    (0, -0.0, True),
]

TYPE_MATCH_CASES = [
    ('x', 'string', True),
    (1, 'string', False),
    (1, 'integer', True),
    (1.0, 'integer', False),
    (True, 'integer', False),
    (1, 'number', True),
    (True, 'number', False),
    (float("inf"), 'number', False),
    (1.5, 'number', True),
    (True, 'boolean', True),
    (0, 'boolean', False),
    ({'a': 1}, 'object', True),
    ([], 'array', True),
    (None, 'null', True),
    ('x', ['integer', 'string'], True),
    (5, ['integer', 'string'], True),
    ('x', ['integer'], False),
    ('x', 'unknown-type', True),
    ('x', None, True),
    ('x', 5, True),
]

PRIMARY_TYPE_CASES = [
    ('x', 'string', 'x'),
    (5, ['string', 'integer'], 'array'),
    (5, 'integer', 'string'),
    (5, ['string'], 'array'),
    ({'a': 1}, None, 'null'),
    ([], None, 'null'),
    ('s', None, 's'),
    (True, None, 'null'),
    (1, None, 'null'),
    (1.5, None, 'null'),
    (float("inf"), None, 'null'),
    (None, None, 'null'),
    (1.5, ['integer', 'number'], 'array'),
    (1.5, ['number', 'integer'], 'array'),
    (True, ['boolean'], 'array'),
    ((1,), None, '1'),
]


CONSTANTS = json.loads("""
{
 "annotations": [
  "title",
  "description",
  "default",
  "examples",
  "deprecated",
  "readOnly",
  "writeOnly",
  "format"
 ],
 "keys": [
  "$defs",
  "$ref",
  "additionalProperties",
  "allOf",
  "anyOf",
  "const",
  "contains",
  "default",
  "definitions",
  "dependentRequired",
  "dependentSchemas",
  "deprecated",
  "description",
  "else",
  "enum",
  "examples",
  "exclusiveMaximum",
  "exclusiveMinimum",
  "format",
  "if",
  "items",
  "maxContains",
  "maxItems",
  "maxLength",
  "maxProperties",
  "maximum",
  "minContains",
  "minItems",
  "minLength",
  "minProperties",
  "minimum",
  "multipleOf",
  "not",
  "oneOf",
  "pattern",
  "patternProperties",
  "properties",
  "propertyNames",
  "readOnly",
  "required",
  "then",
  "title",
  "type",
  "uniqueItems",
  "writeOnly"
 ],
 "types": [
  "array",
  "boolean",
  "integer",
  "null",
  "number",
  "object",
  "string"
 ]
}
""")


def test_module_depends_only_on_the_standard_library() -> None:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"), filename=str(MODULE_PATH))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            modules.add((node.module or "").split(".", 1)[0])
        elif isinstance(node, ast.Import):
            modules.update(alias.name.split(".", 1)[0] for alias in node.names)
    assert modules <= {"__future__", "collections", "decimal", "math", "re", "typing", "urllib"}, (
        modules
    )


def test_module_file_loads_without_the_http_app_or_model_runtime() -> None:
    probe = (
        "import importlib.util, sys\n"
        "spec = importlib.util.spec_from_file_location("
        f"'json_schema_probe', {str(MODULE_PATH)!r})\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        "assert not any(name == 'hipengine' or name.startswith('hipengine.') "
        "for name in sys.modules), [n for n in sys.modules if n.startswith('hipengine')]\n"
        "assert 'fastapi' not in sys.modules and 'starlette' not in sys.modules\n"
        "assert 'torch' not in sys.modules\n"
        "assert module._validate_json_schema_value("
        "{'a': 1}, "
        "{'type': 'object', 'properties': {'a': {'type': 'integer'}}}, "
        "path='$') is None\n"
        "assert module._validate_json_schema_value("
        "{'a': 'x'}, {'type': 'object', 'properties': {'a': {'type': 'integer'}}}, path='$') "
        "== \"$.a does not match schema type 'integer'\"\n"
        "assert module._validate_json_schema_subset("
        "{'type': 'object'}, path='s') is None\n"
        "print('JSON_SCHEMA_PROBE_OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "JSON_SCHEMA_PROBE_OK"


def test_package_import_of_the_module_stays_schema_only() -> None:
    blocked = "(" + ", ".join(repr(prefix) for prefix in BLOCKED_IMPORT_PREFIXES) + ",)"
    probe = (
        "import sys\n"
        "import hipengine.server.json_schema as json_schema\n"
        f"blocked = {blocked}\n"
        "loaded = [name for name in sys.modules if any(name == prefix or "
        "name.startswith(prefix + '.') for prefix in blocked)]\n"
        "assert loaded == [], loaded\n"
        "assert json_schema._validate_json_schema_value("
        "5, {'type': 'integer'}, path='$') is None\n"
        "print('JSON_SCHEMA_PACKAGE_IMPORT_OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "JSON_SCHEMA_PACKAGE_IMPORT_OK"


def test_api_application_names_are_the_json_schema_objects() -> None:
    from hipengine.server import api

    for name in API_APPLICATION_NAMES:
        assert getattr(api, name) is getattr(json_schema, name), name
        if callable(getattr(json_schema, name)):
            assert getattr(api, name).__module__ == "hipengine.server.json_schema", name


def test_result_policy_imports_the_validator_names_its_application_uses() -> None:
    from hipengine.server import chat_result_policy

    for name in RESULT_POLICY_NAMES:
        assert getattr(chat_result_policy, name) is getattr(json_schema, name)


def test_api_imports_the_validator_names_its_application_uses() -> None:
    tree = ast.parse(API_PATH.read_text(encoding="utf-8"), filename=str(API_PATH))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "hipengine.server.json_schema":
            imported.update(alias.name for alias in node.names)
    assert imported == set(API_APPLICATION_NAMES), sorted(set(API_APPLICATION_NAMES) - imported)
    assert imported <= set(PRIVATE_NAMES)
    api_text = API_PATH.read_text(encoding="utf-8")
    for name in PRIVATE_NAMES:
        assert f"def {name}(" not in api_text, name


def test_supported_constants_are_unchanged() -> None:
    assert json_schema._JSON_SCHEMA_ANNOTATION_KEYWORDS == tuple(CONSTANTS["annotations"])
    assert json_schema._JSON_SCHEMA_SUPPORTED_TYPES == frozenset(CONSTANTS["types"])
    assert json_schema._JSON_SCHEMA_SUPPORTED_KEYS == frozenset(CONSTANTS["keys"])
    assert json_schema._JSON_SCHEMA_SUPPORTED_KEYS >= set(
        json_schema._JSON_SCHEMA_ANNOTATION_KEYWORDS
    )


@pytest.mark.parametrize(
    ("name", "schema", "expected"), SCHEMA_CASES, ids=[case[0] for case in SCHEMA_CASES]
)
def test_schema_subset_validation_is_unchanged(name: str, schema, expected) -> None:
    del name
    result = json_schema._validate_json_schema_subset(schema, path="s")
    assert result == expected
    if expected is not None:
        assert isinstance(result, tuple) and len(result) == 2
        assert all(isinstance(item, str) for item in result)


@pytest.mark.parametrize(
    ("name", "value", "schema", "expected"),
    VALUE_CASES,
    ids=[case[0] for case in VALUE_CASES],
)
def test_value_validation_is_unchanged(name: str, value, schema, expected) -> None:
    del name
    result = json_schema._validate_json_schema_value(value, schema, path="$")
    assert result == expected
    if expected is not None:
        assert isinstance(result, str)


@pytest.mark.parametrize(("raw", "token", "error"), POINTER_CASES)
def test_pointer_token_escapes_are_unchanged(raw: str, token, error) -> None:
    assert json_schema._json_schema_pointer_token(raw) == (token, error)


@pytest.mark.parametrize(("root", "ref", "target", "path", "tokens", "error"), RESOLVE_CASES)
def test_local_ref_resolution_is_unchanged(root, ref, target, path, tokens, error) -> None:
    resolved, resolved_path, resolved_tokens, message = json_schema._json_schema_resolve_local_ref(
        root, ref, root_path="$"
    )
    assert (dict(resolved) if resolved is not None else None) == target
    assert resolved_path == path
    assert resolved_tokens == tokens
    assert message == error


@pytest.mark.parametrize(("value", "expected"), NONNEGATIVE_INT_CASES)
def test_nonnegative_int_helper_is_unchanged(value, expected) -> None:
    assert json_schema._schema_nonnegative_int(value) == expected


@pytest.mark.parametrize(("value", "expected"), FINITE_NUMBER_CASES)
def test_finite_number_helper_is_unchanged(value, expected) -> None:
    result = json_schema._schema_finite_number(value)
    assert result == expected
    assert type(result) is type(expected)


@pytest.mark.parametrize(("value", "expected"), POSITIVE_FINITE_CASES)
def test_positive_finite_number_helper_is_unchanged(value, expected) -> None:
    assert json_schema._schema_positive_finite_number(value) == expected


@pytest.mark.parametrize(("value", "expected"), DECIMAL_CASES)
def test_decimal_number_helper_is_unchanged(value, expected) -> None:
    assert json_schema._schema_decimal_number(value) == expected


@pytest.mark.parametrize(("value", "multiple_of", "expected"), MULTIPLE_CASES)
def test_multiple_of_helper_is_unchanged(value, multiple_of, expected) -> None:
    assert json_schema._schema_number_is_multiple_of(value, multiple_of) is expected


@pytest.mark.parametrize(("values", "expected"), UNIQUE_CASES)
def test_array_uniqueness_helper_is_unchanged(values, expected) -> None:
    assert json_schema._schema_array_items_unique(values) is expected


@pytest.mark.parametrize(("left", "right", "expected"), EQUAL_CASES)
def test_json_value_equality_helper_is_unchanged(left, right, expected) -> None:
    assert json_schema._schema_json_values_equal(left, right) is expected


@pytest.mark.parametrize(("value", "expected"), KIND_CASES)
def test_json_value_kind_helper_is_unchanged(value, expected) -> None:
    assert json_schema._schema_json_value_kind(value) == expected


@pytest.mark.parametrize(("value", "expected_type", "expected"), TYPE_MATCH_CASES)
def test_type_matching_helper_is_unchanged(value, expected_type, expected) -> None:
    assert json_schema._json_schema_type_matches(value, expected_type) is expected


@pytest.mark.parametrize(("expected_type", "value", "expected"), PRIMARY_TYPE_CASES)
def test_primary_type_helper_is_unchanged(expected_type, value, expected) -> None:
    assert json_schema._primary_json_schema_type(expected_type, value) == expected


def test_regex_errors_keep_the_engine_message_and_path() -> None:
    param, message = json_schema._validate_json_schema_subset({"pattern": "["}, path="s")
    assert param == "s.pattern"
    assert message.startswith("s.pattern must be a valid regular expression: ")
    assert message.endswith("unterminated character set at position 0")

    param, message = json_schema._validate_json_schema_subset(
        {"patternProperties": {"(": {}}}, path="s"
    )
    assert param == "s.patternProperties.("
    assert message.startswith("s.patternProperties.( must be a valid regular expression: ")


def test_patterns_use_search_semantics_and_value_paths() -> None:
    assert json_schema._validate_json_schema_value("abc", {"pattern": "b"}, path="$") is None
    assert (
        json_schema._validate_json_schema_value("ac", {"pattern": "b"}, path="$")
        == "$ does not match pattern"
    )
    assert (
        json_schema._validate_json_schema_value(
            {"abc": "x"}, {"patternProperties": {"b": {"type": "integer"}}}, path="$"
        )
        == "$.abc does not match schema type 'integer'"
    )


def test_reference_cycles_are_detected_at_both_layers() -> None:
    cyclic = {"$defs": {"a": {"$ref": "#/$defs/a"}}, "$ref": "#/$defs/a"}
    assert json_schema._validate_json_schema_subset(cyclic, path="s") == (
        "s.$defs.a.$ref",
        "s.$defs.a.$ref cycle is not supported",
    )
    assert (
        json_schema._validate_json_schema_value("x", cyclic, path="$")
        == "$ has cyclic schema reference"
    )


def test_boolean_is_never_a_number() -> None:
    assert json_schema._validate_json_schema_value(True, {"type": "integer"}, path="$") is not None
    assert json_schema._validate_json_schema_value(True, {"type": "number"}, path="$") is not None
    assert json_schema._validate_json_schema_value(True, {"type": "boolean"}, path="$") is None
    assert json_schema._validate_json_schema_value(True, {"enum": [1]}, path="$") is not None
    assert json_schema._schema_array_items_unique([1, True]) is True


def test_numeric_messages_keep_general_format() -> None:
    assert (
        json_schema._validate_json_schema_value(0.5, {"minimum": 1.0}, path="$")
        == "$ must be >= 1"
    )
    assert (
        json_schema._validate_json_schema_value(1.25, {"exclusiveMaximum": 1.5}, path="$") is None
    )
    assert (
        json_schema._validate_json_schema_value(1.5, {"exclusiveMaximum": 1.5}, path="$")
        == "$ must be < 1.5"
    )
    assert (
        json_schema._validate_json_schema_value(0.35, {"multipleOf": 0.1}, path="$")
        == "$ must be a multiple of 0.1"
    )
