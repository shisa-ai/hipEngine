import json
from pathlib import Path
import time

previous = None
try:
    with open("/tmp/nasone-ownership.jsonl", "a", buffering=1) as log:
        while True:
            owners = []
            for path in Path("/sys/class/kfd/kfd/proc").glob("*/vram_33912"):
                try:
                    size = int(path.read_text())
                    if size > 1 << 20:
                        pid = int(path.parent.name)
                        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
                        owners.append(dict(pid=pid, vram=size, command=cmd))
                except (OSError, ValueError):
                    pass
            key = tuple((o["pid"], o["command"]) for o in owners)
            if key != previous:
                log.write(json.dumps(dict(time=time.time(), owners=owners)) + "\n")
                previous = key
            time.sleep(0.2)
except KeyboardInterrupt:
    pass
