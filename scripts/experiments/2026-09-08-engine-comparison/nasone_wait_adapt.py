import os
import subprocess
import time

while os.path.exists("/proc/3419824"):
    time.sleep(2)
time.sleep(2)
raise SystemExit(subprocess.run(["python3", "/tmp/nasone_adapt_real.py"]).returncode)
