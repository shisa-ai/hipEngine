"""JSON Schema subset validation owned by the HTTP boundary.

This module owns the pure JSON Schema subset/value validators that the
OpenAI-compatible request surface applies to ``response_format``,
``guided_json`` and tool parameter schemas. It depends only on the standard
library: it imports neither FastAPI nor the HTTP application, and no engine,
model or runtime module.

The validators keep the pre-extraction error tuples, messages, numeric and
boolean rules, local JSON Pointer resolution (including cycle detection),
JSON-value equality and regular-expression semantics unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
import math
import re
from typing import Any
from urllib.parse import unquote


_JSON_SCHEMA_ANNOTATION_KEYWORDS = (
    "title",
    "description",
    "default",
    "examples",
    "deprecated",
    "readOnly",
    "writeOnly",
    "format",
)
_JSON_SCHEMA_SUPPORTED_TYPES = frozenset(
    {
        "array",
        "boolean",
        "integer",
        "null",
        "number",
        "object",
        "string",
    }
)
_JSON_SCHEMA_SUPPORTED_KEYS = frozenset(
    {
        "type",
        "enum",
        "const",
        "$ref",
        "$defs",
        "definitions",
        "allOf",
        "anyOf",
        "oneOf",
        "not",
        "if",
        "then",
        "else",
        "properties",
        "patternProperties",
        "propertyNames",
        "required",
        "dependentRequired",
        "dependentSchemas",
        "additionalProperties",
        "minProperties",
        "maxProperties",
        "items",
        "contains",
        "minItems",
        "maxItems",
        "minContains",
        "maxContains",
        "uniqueItems",
        "minLength",
        "maxLength",
        "pattern",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        *_JSON_SCHEMA_ANNOTATION_KEYWORDS,
    }
)


def _json_schema_pointer_token(raw_token: str) -> tuple[str | None, str | None]:
    token: list[str] = []
    index = 0
    while index < len(raw_token):
        char = raw_token[index]
        if char != "~":
            token.append(char)
            index += 1
            continue
        if index + 1 >= len(raw_token) or raw_token[index + 1] not in {"0", "1"}:
            return None, f"invalid JSON Pointer escape in {raw_token!r}"
        token.append("~" if raw_token[index + 1] == "0" else "/")
        index += 2
    return "".join(token), None


def _json_schema_resolve_local_ref(
    root: Mapping[str, Any],
    ref: str,
    *,
    root_path: str,
) -> tuple[Mapping[str, Any] | None, str, tuple[str, ...], str | None]:
    text = ref.strip()
    if not text:
        return None, root_path, (), "$ref must be a non-empty string"
    if not text.startswith("#"):
        return None, root_path, (), "only local JSON schema $ref values are supported"
    fragment = unquote(text[1:])
    if fragment == "":
        return root, root_path, (), None
    if not fragment.startswith("/"):
        return None, root_path, (), "$ref must be # or a JSON Pointer starting with #/"

    current: Any = root
    current_path = root_path
    tokens: list[str] = []
    for raw_token in fragment[1:].split("/"):
        token, error = _json_schema_pointer_token(raw_token)
        if error is not None or token is None:
            return None, current_path, tuple(tokens), error
        tokens.append(token)
        if isinstance(current, Mapping):
            if token not in current:
                return None, current_path, tuple(tokens), f"$ref target {text!r} does not exist"
            current = current[token]
            current_path = f"{current_path}.{token}"
        elif isinstance(current, Sequence) and not isinstance(current, (str, bytes)):
            if not token.isdecimal():
                return None, current_path, tuple(tokens), f"$ref target {text!r} does not exist"
            item_index = int(token)
            if item_index >= len(current):
                return None, current_path, tuple(tokens), f"$ref target {text!r} does not exist"
            current = current[item_index]
            current_path = f"{current_path}[{item_index}]"
        else:
            return None, current_path, tuple(tokens), f"$ref target {text!r} does not exist"
    if not isinstance(current, Mapping):
        return None, current_path, tuple(tokens), "$ref target must be a schema object"
    return current, current_path, tuple(tokens), None


def _validate_json_schema_subset(
    schema: Mapping[str, Any],
    *,
    path: str,
    root: Mapping[str, Any] | None = None,
    root_path: str | None = None,
    ref_stack: tuple[tuple[str, ...], ...] = (),
) -> tuple[str, str] | None:
    root_schema = schema if root is None else root
    root_param = path if root_path is None else root_path
    for raw_key in schema:
        key = str(raw_key)
        if key not in _JSON_SCHEMA_SUPPORTED_KEYS:
            param = f"{path}.{key}"
            return (param, f"{param} is not supported by hipEngine JSON schema subset")

    for key in ("$defs", "definitions"):
        definitions = schema.get(key)
        if definitions is None:
            continue
        if not isinstance(definitions, Mapping):
            return (f"{path}.{key}", f"{path}.{key} must be an object")
        for raw_name, subschema in definitions.items():
            definition_path = f"{path}.{key}.{raw_name}"
            if not isinstance(raw_name, str):
                return (f"{path}.{key}", f"{path}.{key} keys must be strings")
            if not isinstance(subschema, Mapping):
                return (definition_path, f"{definition_path} must be an object")
            error = _validate_json_schema_subset(
                subschema,
                path=definition_path,
                root=root_schema,
                root_path=root_param,
                ref_stack=ref_stack,
            )
            if error is not None:
                return error

    ref = schema.get("$ref")
    if "$ref" in schema:
        if not isinstance(ref, str):
            return (f"{path}.$ref", f"{path}.$ref must be a string")
        target, target_path, pointer, error_message = _json_schema_resolve_local_ref(
            root_schema,
            ref,
            root_path=root_param,
        )
        if error_message is not None or target is None:
            return (f"{path}.$ref", f"{path}.$ref {error_message}")
        if pointer in ref_stack:
            return (f"{path}.$ref", f"{path}.$ref cycle is not supported")
        error = _validate_json_schema_subset(
            target,
            path=target_path,
            root=root_schema,
            root_path=root_param,
            ref_stack=(*ref_stack, pointer),
        )
        if error is not None:
            return error

    expected = schema.get("type")
    if expected is not None:
        if isinstance(expected, str):
            expected_types = (expected,)
        elif isinstance(expected, Sequence) and not isinstance(expected, (str, bytes)):
            expected_types = tuple(expected)
        else:
            return (f"{path}.type", f"{path}.type must be a string or list of strings")
        for item in expected_types:
            if not isinstance(item, str) or item not in _JSON_SCHEMA_SUPPORTED_TYPES:
                return (f"{path}.type", f"{path}.type contains unsupported type {item!r}")

    enum = schema.get("enum")
    if enum is not None and (not isinstance(enum, Sequence) or isinstance(enum, (str, bytes))):
        return (f"{path}.enum", f"{path}.enum must be an array")

    for key in ("allOf", "anyOf", "oneOf"):
        subschemas = schema.get(key)
        if subschemas is None:
            continue
        if not isinstance(subschemas, Sequence) or isinstance(subschemas, (str, bytes)) or not subschemas:
            return (f"{path}.{key}", f"{path}.{key} must be a non-empty array of schema objects")
        for index, subschema in enumerate(subschemas):
            subschema_path = f"{path}.{key}[{index}]"
            if not isinstance(subschema, Mapping):
                return (subschema_path, f"{subschema_path} must be an object")
            error = _validate_json_schema_subset(
                subschema,
                path=subschema_path,
                root=root_schema,
                root_path=root_param,
                ref_stack=ref_stack,
            )
            if error is not None:
                return error

    not_schema = schema.get("not")
    if "not" in schema:
        if not isinstance(not_schema, Mapping):
            return (f"{path}.not", f"{path}.not must be an object")
        error = _validate_json_schema_subset(
            not_schema,
            path=f"{path}.not",
            root=root_schema,
            root_path=root_param,
            ref_stack=ref_stack,
        )
        if error is not None:
            return error

    for key in ("if", "then", "else"):
        conditional_schema = schema.get(key)
        if key not in schema:
            continue
        if not isinstance(conditional_schema, Mapping):
            return (f"{path}.{key}", f"{path}.{key} must be an object")
        error = _validate_json_schema_subset(
            conditional_schema,
            path=f"{path}.{key}",
            root=root_schema,
            root_path=root_param,
            ref_stack=ref_stack,
        )
        if error is not None:
            return error

    properties = schema.get("properties")
    if properties is not None:
        if not isinstance(properties, Mapping):
            return (f"{path}.properties", f"{path}.properties must be an object")
        for raw_key, subschema in properties.items():
            property_path = f"{path}.properties.{raw_key}"
            if not isinstance(subschema, Mapping):
                return (property_path, f"{property_path} must be an object")
            error = _validate_json_schema_subset(
                subschema,
                path=property_path,
                root=root_schema,
                root_path=root_param,
                ref_stack=ref_stack,
            )
            if error is not None:
                return error

    pattern_properties = schema.get("patternProperties")
    if pattern_properties is not None:
        if not isinstance(pattern_properties, Mapping):
            return (f"{path}.patternProperties", f"{path}.patternProperties must be an object")
        for raw_pattern, subschema in pattern_properties.items():
            pattern_path = f"{path}.patternProperties.{raw_pattern}"
            if not isinstance(raw_pattern, str):
                return (f"{path}.patternProperties", f"{path}.patternProperties keys must be strings")
            try:
                re.compile(raw_pattern)
            except re.error as exc:
                return (pattern_path, f"{pattern_path} must be a valid regular expression: {exc}")
            if not isinstance(subschema, Mapping):
                return (pattern_path, f"{pattern_path} must be an object")
            error = _validate_json_schema_subset(
                subschema,
                path=pattern_path,
                root=root_schema,
                root_path=root_param,
                ref_stack=ref_stack,
            )
            if error is not None:
                return error

    property_names = schema.get("propertyNames")
    if property_names is not None:
        if not isinstance(property_names, Mapping):
            return (f"{path}.propertyNames", f"{path}.propertyNames must be an object")
        error = _validate_json_schema_subset(
            property_names,
            path=f"{path}.propertyNames",
            root=root_schema,
            root_path=root_param,
            ref_stack=ref_stack,
        )
        if error is not None:
            return error

    required = schema.get("required")
    if required is not None:
        if not isinstance(required, Sequence) or isinstance(required, (str, bytes)):
            return (f"{path}.required", f"{path}.required must be an array of strings")
        if any(not isinstance(item, str) for item in required):
            return (f"{path}.required", f"{path}.required must contain only strings")

    dependent_required = schema.get("dependentRequired")
    if dependent_required is not None:
        if not isinstance(dependent_required, Mapping):
            return (f"{path}.dependentRequired", f"{path}.dependentRequired must be an object")
        for raw_key, dependencies in dependent_required.items():
            dependency_path = f"{path}.dependentRequired.{raw_key}"
            if not isinstance(raw_key, str):
                return (f"{path}.dependentRequired", f"{path}.dependentRequired keys must be strings")
            if not isinstance(dependencies, Sequence) or isinstance(dependencies, (str, bytes)):
                return (dependency_path, f"{dependency_path} must be an array of strings")
            if any(not isinstance(item, str) for item in dependencies):
                return (dependency_path, f"{dependency_path} must contain only strings")

    dependent_schemas = schema.get("dependentSchemas")
    if dependent_schemas is not None:
        if not isinstance(dependent_schemas, Mapping):
            return (f"{path}.dependentSchemas", f"{path}.dependentSchemas must be an object")
        for raw_key, subschema in dependent_schemas.items():
            dependency_path = f"{path}.dependentSchemas.{raw_key}"
            if not isinstance(raw_key, str):
                return (f"{path}.dependentSchemas", f"{path}.dependentSchemas keys must be strings")
            if not isinstance(subschema, Mapping):
                return (dependency_path, f"{dependency_path} must be an object")
            error = _validate_json_schema_subset(
                subschema,
                path=dependency_path,
                root=root_schema,
                root_path=root_param,
                ref_stack=ref_stack,
            )
            if error is not None:
                return error

    additional_properties = schema.get("additionalProperties")
    if "additionalProperties" in schema:
        if isinstance(additional_properties, bool):
            pass
        elif isinstance(additional_properties, Mapping):
            error = _validate_json_schema_subset(
                additional_properties,
                path=f"{path}.additionalProperties",
                root=root_schema,
                root_path=root_param,
                ref_stack=ref_stack,
            )
            if error is not None:
                return error
        else:
            return (
                f"{path}.additionalProperties",
                f"{path}.additionalProperties must be a boolean or object in hipEngine JSON schema subset",
            )

    items = schema.get("items")
    if items is not None:
        if not isinstance(items, Mapping):
            return (f"{path}.items", f"{path}.items must be an object")
        error = _validate_json_schema_subset(
            items,
            path=f"{path}.items",
            root=root_schema,
            root_path=root_param,
            ref_stack=ref_stack,
        )
        if error is not None:
            return error

    contains = schema.get("contains")
    if contains is not None:
        if not isinstance(contains, Mapping):
            return (f"{path}.contains", f"{path}.contains must be an object")
        error = _validate_json_schema_subset(
            contains,
            path=f"{path}.contains",
            root=root_schema,
            root_path=root_param,
            ref_stack=ref_stack,
        )
        if error is not None:
            return error

    for key in (
        "minProperties",
        "maxProperties",
        "minItems",
        "maxItems",
        "minContains",
        "maxContains",
        "minLength",
        "maxLength",
    ):
        if key in schema and _schema_nonnegative_int(schema.get(key)) is None:
            return (f"{path}.{key}", f"{path}.{key} must be a non-negative integer")
    if "uniqueItems" in schema and not isinstance(schema.get("uniqueItems"), bool):
        return (f"{path}.uniqueItems", f"{path}.uniqueItems must be a boolean")
    if "pattern" in schema:
        pattern = schema.get("pattern")
        if not isinstance(pattern, str):
            return (f"{path}.pattern", f"{path}.pattern must be a string")
        try:
            re.compile(pattern)
        except re.error as exc:
            return (f"{path}.pattern", f"{path}.pattern must be a valid regular expression: {exc}")
    for key in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
        if key in schema and _schema_finite_number(schema.get(key)) is None:
            return (f"{path}.{key}", f"{path}.{key} must be a finite number")
    if "multipleOf" in schema and _schema_positive_finite_number(schema.get("multipleOf")) is None:
        return (f"{path}.multipleOf", f"{path}.multipleOf must be a positive finite number")
    return None


def _validate_json_schema_value(
    value: Any,
    schema: Mapping[str, Any],
    *,
    path: str,
    root: Mapping[str, Any] | None = None,
    ref_stack: tuple[tuple[str, ...], ...] = (),
) -> str | None:
    root_schema = schema if root is None else root
    ref = schema.get("$ref")
    if isinstance(ref, str):
        target, _, pointer, error_message = _json_schema_resolve_local_ref(
            root_schema,
            ref,
            root_path="$",
        )
        if error_message is not None or target is None:
            return f"{path} has invalid schema reference"
        if pointer in ref_stack:
            return f"{path} has cyclic schema reference"
        error = _validate_json_schema_value(
            value,
            target,
            path=path,
            root=root_schema,
            ref_stack=(*ref_stack, pointer),
        )
        if error is not None:
            return error
    enum = schema.get("enum")
    if (
        isinstance(enum, Sequence)
        and not isinstance(enum, (str, bytes))
        and not any(_schema_json_values_equal(value, item) for item in enum)
    ):
        return f"{path} is not one of the allowed enum values"
    if "const" in schema and not _schema_json_values_equal(value, schema.get("const")):
        return f"{path} does not match const"
    expected = schema.get("type")
    if expected is not None and not _json_schema_type_matches(value, expected):
        return f"{path} does not match schema type {expected!r}"
    all_of = schema.get("allOf")
    if isinstance(all_of, Sequence) and not isinstance(all_of, (str, bytes)):
        for subschema in all_of:
            if not isinstance(subschema, Mapping):
                continue
            error = _validate_json_schema_value(
                value,
                subschema,
                path=path,
                root=root_schema,
                ref_stack=ref_stack,
            )
            if error is not None:
                return error
    any_of = schema.get("anyOf")
    if isinstance(any_of, Sequence) and not isinstance(any_of, (str, bytes)):
        matches = 0
        for subschema in any_of:
            if (
                isinstance(subschema, Mapping)
                and _validate_json_schema_value(
                    value,
                    subschema,
                    path=path,
                    root=root_schema,
                    ref_stack=ref_stack,
                )
                is None
            ):
                matches += 1
                break
        if matches == 0:
            return f"{path} does not match any allowed schema"
    one_of = schema.get("oneOf")
    if isinstance(one_of, Sequence) and not isinstance(one_of, (str, bytes)):
        matches = sum(
            1
            for subschema in one_of
            if isinstance(subschema, Mapping)
            and _validate_json_schema_value(
                value,
                subschema,
                path=path,
                root=root_schema,
                ref_stack=ref_stack,
            )
            is None
        )
        if matches != 1:
            return f"{path} must match exactly one allowed schema"
    not_schema = schema.get("not")
    if (
        isinstance(not_schema, Mapping)
        and _validate_json_schema_value(
            value,
            not_schema,
            path=path,
            root=root_schema,
            ref_stack=ref_stack,
        )
        is None
    ):
        return f"{path} matches a disallowed schema"
    if_schema = schema.get("if")
    if isinstance(if_schema, Mapping):
        branch_key = (
            "then"
            if _validate_json_schema_value(
                value,
                if_schema,
                path=path,
                root=root_schema,
                ref_stack=ref_stack,
            )
            is None
            else "else"
        )
        branch_schema = schema.get(branch_key)
        if isinstance(branch_schema, Mapping):
            error = _validate_json_schema_value(
                value,
                branch_schema,
                path=path,
                root=root_schema,
                ref_stack=ref_stack,
            )
            if error is not None:
                return error
    schema_type = _primary_json_schema_type(expected, value)
    if schema_type == "object":
        if not isinstance(value, Mapping):
            return f"{path} must be an object"
        min_properties = _schema_nonnegative_int(schema.get("minProperties"))
        if min_properties is not None and len(value) < min_properties:
            return f"{path} must have at least {min_properties} properties"
        max_properties = _schema_nonnegative_int(schema.get("maxProperties"))
        if max_properties is not None and len(value) > max_properties:
            return f"{path} must have at most {max_properties} properties"
        required = schema.get("required", ())
        if isinstance(required, Sequence) and not isinstance(required, (str, bytes)):
            for key in required:
                if isinstance(key, str) and key not in value:
                    return f"{path}.{key} is required"
        property_names = schema.get("propertyNames")
        if isinstance(property_names, Mapping):
            for raw_key in value:
                key = str(raw_key)
                error = _validate_json_schema_value(
                    key,
                    property_names,
                    path=f"{path}.{key}",
                    root=root_schema,
                    ref_stack=ref_stack,
                )
                if error is not None:
                    return f"{path}.{key} property name is invalid"
        dependent_required = schema.get("dependentRequired")
        if isinstance(dependent_required, Mapping):
            for raw_key, dependencies in dependent_required.items():
                if not isinstance(raw_key, str) or raw_key not in value:
                    continue
                if isinstance(dependencies, Sequence) and not isinstance(dependencies, (str, bytes)):
                    for dependency in dependencies:
                        if isinstance(dependency, str) and dependency not in value:
                            return f"{path}.{dependency} is required when {raw_key} is present"
        dependent_schemas = schema.get("dependentSchemas")
        if isinstance(dependent_schemas, Mapping):
            for raw_key, subschema in dependent_schemas.items():
                if not isinstance(raw_key, str) or raw_key not in value or not isinstance(subschema, Mapping):
                    continue
                error = _validate_json_schema_value(
                    value,
                    subschema,
                    path=path,
                    root=root_schema,
                    ref_stack=ref_stack,
                )
                if error is not None:
                    return error
        properties = schema.get("properties")
        property_map = properties if isinstance(properties, Mapping) else {}
        for key, subschema in property_map.items():
            if key in value and isinstance(key, str) and isinstance(subschema, Mapping):
                error = _validate_json_schema_value(
                    value[key],
                    subschema,
                    path=f"{path}.{key}",
                    root=root_schema,
                    ref_stack=ref_stack,
                )
                if error is not None:
                    return error
        pattern_properties = schema.get("patternProperties")
        pattern_map = pattern_properties if isinstance(pattern_properties, Mapping) else {}
        matched_by_pattern: set[str] = set()
        for raw_pattern, subschema in pattern_map.items():
            if not isinstance(raw_pattern, str) or not isinstance(subschema, Mapping):
                continue
            for raw_key, item in value.items():
                key = str(raw_key)
                try:
                    matches = re.search(raw_pattern, key) is not None
                except re.error:
                    matches = False
                if not matches:
                    continue
                matched_by_pattern.add(key)
                error = _validate_json_schema_value(
                    item,
                    subschema,
                    path=f"{path}.{key}",
                    root=root_schema,
                    ref_stack=ref_stack,
                )
                if error is not None:
                    return error
        additional_properties = schema.get("additionalProperties")
        if additional_properties is False:
            allowed = {str(key) for key in property_map}
            extra = sorted(
                str(key)
                for key in value.keys()
                if str(key) not in allowed and str(key) not in matched_by_pattern
            )
            if extra:
                return f"{path}.{extra[0]} is not allowed"
        elif isinstance(additional_properties, Mapping):
            allowed = {str(key) for key in property_map}
            extra_items = sorted(
                (str(key), key)
                for key in value.keys()
                if str(key) not in allowed and str(key) not in matched_by_pattern
            )
            for key, raw_key in extra_items:
                error = _validate_json_schema_value(
                    value[raw_key],
                    additional_properties,
                    path=f"{path}.{key}",
                    root=root_schema,
                    ref_stack=ref_stack,
                )
                if error is not None:
                    return error
    elif schema_type == "array":
        if not isinstance(value, list):
            return f"{path} must be an array"
        min_items = _schema_nonnegative_int(schema.get("minItems"))
        if min_items is not None and len(value) < min_items:
            return f"{path} must have at least {min_items} items"
        max_items = _schema_nonnegative_int(schema.get("maxItems"))
        if max_items is not None and len(value) > max_items:
            return f"{path} must have at most {max_items} items"
        if schema.get("uniqueItems") is True and not _schema_array_items_unique(value):
            return f"{path} must contain unique items"
        contains = schema.get("contains")
        if isinstance(contains, Mapping):
            match_count = sum(
                1
                for item in value
                if _validate_json_schema_value(
                    item,
                    contains,
                    path=path,
                    root=root_schema,
                    ref_stack=ref_stack,
                )
                is None
            )
            min_contains = _schema_nonnegative_int(schema.get("minContains"))
            min_contains = 1 if min_contains is None else min_contains
            if match_count < min_contains:
                return f"{path} must contain at least {min_contains} matching items"
            max_contains = _schema_nonnegative_int(schema.get("maxContains"))
            if max_contains is not None and match_count > max_contains:
                return f"{path} must contain at most {max_contains} matching items"
        items = schema.get("items")
        if isinstance(items, Mapping):
            for index, item in enumerate(value):
                error = _validate_json_schema_value(
                    item,
                    items,
                    path=f"{path}[{index}]",
                    root=root_schema,
                    ref_stack=ref_stack,
                )
                if error is not None:
                    return error
    elif schema_type == "string":
        if isinstance(value, str):
            min_length = _schema_nonnegative_int(schema.get("minLength"))
            if min_length is not None and len(value) < min_length:
                return f"{path} must have at least {min_length} characters"
            max_length = _schema_nonnegative_int(schema.get("maxLength"))
            if max_length is not None and len(value) > max_length:
                return f"{path} must have at most {max_length} characters"
            pattern = schema.get("pattern")
            if isinstance(pattern, str) and re.search(pattern, value) is None:
                return f"{path} does not match pattern"
    elif schema_type in {"integer", "number"}:
        numeric = float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None
        if numeric is not None:
            minimum = _schema_finite_number(schema.get("minimum"))
            if minimum is not None and numeric < minimum:
                return f"{path} must be >= {minimum:g}"
            maximum = _schema_finite_number(schema.get("maximum"))
            if maximum is not None and numeric > maximum:
                return f"{path} must be <= {maximum:g}"
            exclusive_minimum = _schema_finite_number(schema.get("exclusiveMinimum"))
            if exclusive_minimum is not None and numeric <= exclusive_minimum:
                return f"{path} must be > {exclusive_minimum:g}"
            exclusive_maximum = _schema_finite_number(schema.get("exclusiveMaximum"))
            if exclusive_maximum is not None and numeric >= exclusive_maximum:
                return f"{path} must be < {exclusive_maximum:g}"
            multiple_of = schema.get("multipleOf")
            if not _schema_number_is_multiple_of(value, multiple_of):
                assert isinstance(multiple_of, (int, float))
                return f"{path} must be a multiple of {float(multiple_of):g}"
    return None


def _schema_nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value >= 0 else None


def _schema_finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _schema_positive_finite_number(value: Any) -> float | None:
    numeric = _schema_finite_number(value)
    if numeric is None or numeric <= 0:
        return None
    return numeric


def _schema_decimal_number(value: Any) -> Decimal | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        numeric = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return numeric if numeric.is_finite() else None


def _schema_number_is_multiple_of(value: Any, multiple_of: Any) -> bool:
    divisor = _schema_decimal_number(multiple_of)
    if divisor is None or divisor <= 0:
        return True
    numeric = _schema_decimal_number(value)
    if numeric is None:
        return True
    return numeric % divisor == 0


def _schema_array_items_unique(values: Sequence[Any]) -> bool:
    for index, item in enumerate(values):
        for other in values[index + 1 :]:
            if _schema_json_values_equal(item, other):
                return False
    return True


def _schema_json_values_equal(left: Any, right: Any) -> bool:
    left_kind = _schema_json_value_kind(left)
    right_kind = _schema_json_value_kind(right)
    if left_kind != right_kind:
        return False
    if left_kind == "number":
        left_number = _schema_decimal_number(left)
        right_number = _schema_decimal_number(right)
        return left_number is not None and right_number is not None and left_number == right_number
    if left_kind == "array":
        return len(left) == len(right) and all(
            _schema_json_values_equal(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    if left_kind == "object":
        if set(left.keys()) != set(right.keys()):
            return False
        return all(_schema_json_values_equal(left[key], right[key]) for key in left)
    return left == right


def _schema_json_value_kind(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if _schema_decimal_number(value) is not None:
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, Mapping):
        return "object"
    return "unknown"


def _json_schema_type_matches(value: Any, expected: Any) -> bool:
    if isinstance(expected, Sequence) and not isinstance(expected, (str, bytes)):
        return any(_json_schema_type_matches(value, item) for item in expected)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "object":
        return isinstance(value, Mapping)
    if expected == "array":
        return isinstance(value, list)
    if expected == "null":
        return value is None
    return True


def _primary_json_schema_type(expected: Any, value: Any) -> str | None:
    if isinstance(expected, str):
        return expected
    if isinstance(expected, Sequence) and not isinstance(expected, (str, bytes)):
        for item in expected:
            if _json_schema_type_matches(value, item):
                return str(item)
    if isinstance(value, Mapping):
        return "object"
    if isinstance(value, list):
        return "array"
    if isinstance(value, str):
        return "string"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float) and math.isfinite(value):
        return "number"
    if value is None:
        return "null"
    return None
