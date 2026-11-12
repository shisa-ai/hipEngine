"""Collect filtered public-request projection counters using cached kernels."""
import os
import subprocess
import sys
from pathlib import Path
from hipengine.benchmark.roctx_discovery import default_sdk, sdk_dependency_paths, prepare_override
sdk = default_sdk(profiler_first=True)
override = prepare_override(sdk, Path('/tmp/qwen38-roctx'))
env = dict(os.environ)
env['HIPENGINE_COMPILER_VERSION_FILE'] = '/tmp/qwen38-hipcc-version.txt'
env['HIPENGINE_REQUIRE_CACHED_BUILD'] = '1'
env['LD_LIBRARY_PATH'] = ':'.join(map(str, (override, *sdk_dependency_paths(sdk)))) + ':' + env.get('LD_LIBRARY_PATH', '')
cmd = ['rocprofv3', '--pmc', *sys.argv[2].split(','), '--kernel-include-regex', 'q5_k_t16_dense_rowtile_gemv_kernel|gguf_iq4_xs_local32_gemv_kernel', '--selected-regions', '--output-format', 'csv', '-d', sys.argv[1], '--', sys.executable, 'benchmarks/results/2026-11-12-gfx1151-qwen38-dense-followup/attribution.py', '--profile', '--out', sys.argv[1] + '-request.json']
print(cmd, flush=True)
raise SystemExit(subprocess.run(cmd, env=env).returncode)
