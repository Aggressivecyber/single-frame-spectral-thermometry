"""Run the frozen simulation once and persist the real process exit status."""
import runtime
from pathlib import Path
import json
import os
import subprocess
import sys
import time

root = Path(__file__).resolve().parents[1]
protocol = json.loads((root/"research_filter_decode/protocol.json").read_text())
out = root/"reports"/protocol["run_id"]
out.mkdir(parents=True, exist_ok=True)
start = time.monotonic()
with (out/"stdout.txt").open("x") as stdout, (out/"stderr.txt").open("x") as stderr:
    child = subprocess.Popen([sys.executable, "-B", "-m", "research_filter_decode.run", "--device", "cpu"],
                             cwd=root, stdout=stdout, stderr=stderr)
    runtime.write_json(out/"PROCESS.json", {"time": runtime.now(), "supervisor_pid": os.getpid(),
                                           "child_pid": child.pid, "cwd": str(root)})
    try:
        code = child.wait(timeout=protocol["resources"]["maximum_wall_seconds"])
        reason = "child_exit"
    except subprocess.TimeoutExpired:
        child.terminate()
        try:
            code = child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            child.kill()
            code = child.wait()
        reason = "registered_wall_time_limit"
runtime.write_json(out/"EXIT.json", {"time": runtime.now(), "returncode": code, "reason": reason,
                                    "wall_s": time.monotonic()-start,
                                    "completed": code == 0 and (out/"COMPLETION.json").exists()})
sys.exit(code)
