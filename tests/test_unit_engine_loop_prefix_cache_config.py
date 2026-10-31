"""Prefix-cache operator controls share one default and preserve explicit intent."""
import argparse

import pytest

from hipengine.generation.engine_loop import (
    add_engine_loop_config_args,
    engine_loop_config_from_args,
)
from hipengine.kvcache import PREFIX_CACHE_DEFAULT


def test_prefix_cache_cli_help_reports_actual_default():
    parser = argparse.ArgumentParser()
    add_engine_loop_config_args(parser, environ={})
    assert f"default: {PREFIX_CACHE_DEFAULT}" in parser.format_help()


@pytest.mark.parametrize("environ,args,expected", [
    ({}, [], PREFIX_CACHE_DEFAULT),
    ({"HIPENGINE_PREFIX_CACHE": "off"}, [], "off"),
    ({"HIPENGINE_PREFIX_CACHE": "radix"}, ["--prefix-cache", "off"], "off"),
    ({"HIPENGINE_PREFIX_CACHE": "off"}, ["--prefix-cache", "radix"], "radix"),
])
def test_prefix_cache_cli_and_environment_preserve_intent(environ, args, expected):
    parser = argparse.ArgumentParser()
    add_engine_loop_config_args(parser, environ=environ)
    assert engine_loop_config_from_args(parser.parse_args(args)).prefix_cache == expected
