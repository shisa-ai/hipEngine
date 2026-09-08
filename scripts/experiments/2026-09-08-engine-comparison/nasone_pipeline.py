import os
import subprocess
import time

while os.path.exists("/proc/3350762"):
    time.sleep(2)
for script in ("/tmp/nasone_hip_remaining.py", "/tmp/nasone_clean.py"):
    result = subprocess.run(["python3", script])
    if result.returncode:
        raise SystemExit(result.returncode)
