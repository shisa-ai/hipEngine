#!/usr/bin/env python3
"""Run the resident sweep under scoped cumulative-prefill comparisons."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hipengine.runtime.gguf_linear import cumulative_prefill_diagnostic
from scripts import qwen35_readme_sweep


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    for field in ('dense-bulk', 'dense-residual', 'q8-pair', 'pack8-bulk', 'pack8-pair'):
        parser.add_argument(f'--diagnostic-{field}', choices=('on', 'off'), default='on')
    args, remaining = parser.parse_known_args()
    previous = sys.argv
    try:
        sys.argv = [previous[0], *remaining]
        with cumulative_prefill_diagnostic(
            dense_bulk=args.diagnostic_dense_bulk == 'on',
            dense_residual=args.diagnostic_dense_residual == 'on',
            q8_pair=args.diagnostic_q8_pair == 'on',
            pack8_bulk=args.diagnostic_pack8_bulk == 'on',
            pack8_pair=args.diagnostic_pack8_pair == 'on',
        ):
            return qwen35_readme_sweep.main()
    finally:
        sys.argv = previous


if __name__ == '__main__':
    raise SystemExit(main())
