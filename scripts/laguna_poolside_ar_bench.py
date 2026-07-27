#!/usr/bin/env python3
"""Compatibility entry point for the matched llama.cpp Laguna AR harness.

Use :mod:`scripts.laguna_llamacpp_ar_bench` for new commands.
"""

from scripts import laguna_llamacpp_ar_bench as _matched

_aggregate = _matched._aggregate
_post_json = _matched._post_json
_process_rss_bytes = _matched._process_rss_bytes
_read_optional_int = _matched._read_optional_int
_response_row = _matched._response_row
_terminate = _matched._terminate
_wait_for_server = _matched._wait_for_server
run = _matched.run
main = _matched.main


if __name__ == "__main__":
    raise SystemExit(main())
