"""Run one registered command with preserved stdout, stderr and exit status."""
import runtime
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

parser=argparse.ArgumentParser()
parser.add_argument("--output",type=Path,required=True)
parser.add_argument("--maximum-wall-seconds",type=int,required=True)
parser.add_argument("--software-only",action="store_true")
parser.add_argument("--protocol",type=Path,default=Path(__file__).with_name("protocol.json"))
args=parser.parse_args()
root=Path(__file__).resolve().parents[1]
out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
command=[sys.executable,"-B","-m","research_joint_imaging_v2.pipeline","--protocol",str(args.protocol.resolve()),"--output",str(out)]
if args.software_only:command.append("--software-only")
start=time.monotonic()
with (out/"stdout.txt").open("x") as stdout,(out/"stderr.txt").open("x") as stderr:
    child=subprocess.Popen(command,cwd=root,stdout=stdout,stderr=stderr)
    runtime.write_json(out/"PROCESS.json",{"time":runtime.now(),"supervisor_pid":os.getpid(),"child_pid":child.pid,"command":command,"source_root":str(root)})
    try:
        code=child.wait(timeout=args.maximum_wall_seconds);reason="child_exit"
    except subprocess.TimeoutExpired:
        child.terminate()
        try:code=child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            child.kill();code=child.wait()
        reason="registered_wall_time_limit"
runtime.write_json(out/"EXIT.json",{"time":runtime.now(),"returncode":code,"reason":reason,"wall_seconds":time.monotonic()-start,"completed":code==0 and (out/"COMPLETION.json").exists()})
sys.exit(code)
