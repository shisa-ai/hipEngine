import os,sys,subprocess
from pathlib import Path
from hipengine.benchmark.roctx_discovery import default_sdk,sdk_dependency_paths,prepare_override
sdk=default_sdk(profiler_first=True);override=prepare_override(sdk,Path('/tmp/qwen38-roctx'))
env=dict(os.environ)
env['HIPENGINE_COMPILER_VERSION_FILE']='/tmp/qwen38-hipcc-version.txt'
env['HIPENGINE_REQUIRE_CACHED_BUILD']='1'
env['LD_LIBRARY_PATH']=':'.join(map(str,(override,*sdk_dependency_paths(sdk))))+':'+env.get('LD_LIBRARY_PATH','')
cmd=['rocprofv3','--kernel-trace','--hip-runtime-trace','--selected-regions','-d',sys.argv[1],'--',sys.executable,*sys.argv[2:]]
print(cmd,flush=True)
raise SystemExit(subprocess.run(cmd,env=env).returncode)
