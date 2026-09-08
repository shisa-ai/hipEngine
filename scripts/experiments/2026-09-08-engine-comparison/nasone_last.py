import os
import subprocess
import time

while os.path.exists("/proc/3412938"):
    time.sleep(2)
raise SystemExit(subprocess.run(["python3", "/tmp/nasone_pure.py"]).returncode)
