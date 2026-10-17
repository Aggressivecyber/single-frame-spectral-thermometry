"""Recover the interrupted development study without regenerating raw/TV caches."""
import runtime
import argparse
import json
from pathlib import Path
import shutil
import torch
from research_joint_imaging.core import Imager,Radiometry
from research_joint_imaging_v2.pipeline import load_mothers,train_method


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--protocol",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--resume-from",type=Path,required=True)
    args=parser.parse_args()
    protocol=json.loads(args.protocol.read_text())
    registration=json.loads(args.protocol.with_name("REGISTRATION.json").read_text())
    root=Path(__file__).resolve().parents[1]
    for name,expected in registration["source_sha256"].items():
        if runtime.sha256(root/name)!=expected:raise RuntimeError("Source hash changed: "+name)
    output=args.output.resolve();source=args.resume_from.resolve()
    output.mkdir(parents=True,exist_ok=True)
    if (output/"PIPELINE_START.json").exists():raise FileExistsError("Recovery output already started")
    if (source/"COMPLETION.json").exists():raise RuntimeError("Source study is already complete")
    runtime.write_json(output/"PIPELINE_START.json",{"time":runtime.now(),"source_run":str(source),"protocol_sha256":runtime.sha256(args.protocol),"source_hashes_verified":True,"raw_and_TV_regenerated":False,"completed_methods_retrained":False,"heldout_evaluation_started":False})
    (output/"cache").symlink_to(source/"cache",target_is_directory=True)
    for name in ("DATA_RECEIPT.json","CACHE_PROGRESS.json","CUDA_CONTRACTS.json","FLOAT32_PROX_GRADIENT.json","NATIVE_TRAINING_PROBE.json"):
        shutil.copy2(source/name,output/name)
    receipt=json.loads((output/"DATA_RECEIPT.json").read_text())
    for item in receipt["mothers"]:
        if runtime.sha256(output/item["file"])!=item["sha256"]:raise RuntimeError("Frozen cache changed: "+item["file"])
    runtime.write_json(output/"CACHE_REUSE_VERIFIED.json",{"time":runtime.now(),"mothers":len(receipt["mothers"]),"raw_observations":receipt["total_raw_observations"],"all_cache_hashes_match":True,"cache_target":str((output/"cache").resolve()),"duplicated_cache_bytes":0})
    torch.set_num_threads(protocol["resources"]["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    device="cuda"
    torch.cuda.set_per_process_memory_fraction(protocol["resources"]["maximum_gpu_memory_MiB"]*1024**2/torch.cuda.get_device_properties(0).total_memory)
    op=Imager(protocol["model"]).to(device);rad=Radiometry(protocol,device)
    train=load_mothers(receipt,output,"train");val=load_mothers(receipt,output,"validation")
    completions=[]
    for method in protocol["learning"]["methods"]:
        completed=source/(method+"_TRAINING_COMPLETION.json")
        if completed.exists():
            result=json.loads(completed.read_text())
            for kind in ("best","last"):
                path=source/(method+"_"+kind+".pt")
                if runtime.sha256(path)!=result[kind+"_checkpoint_sha256"]:raise RuntimeError("Completed checkpoint changed: "+str(path))
            rows=[json.loads(line) for line in (source/(method+"_training.jsonl")).read_text().splitlines()]
            if result["updates"]!=sum(protocol["learning"]["phases"]) or [row["step"] for row in rows]!=list(range(1,result["updates"]+1)):raise RuntimeError("Completed training update count differs")
            for suffix in ("_best.pt","_last.pt","_training.jsonl","_validation.json","_TRAINING_COMPLETION.json"):
                shutil.copy2(source/(method+suffix),output/(method+suffix))
            print({"stage":"completed_method_reused","method":method,"updates":result["updates"]},flush=True)
        else:
            interrupted=(source/(method+"_training.jsonl")).exists()
            result=train_method(protocol,op,rad,train,val,output,device,method,source if interrupted else None)
        completions.append(result)
    runtime.write_json(output/"SELECTION_FROZEN.json",{"time":runtime.now(),"methods":completions,"heldout_loaded_for_evaluation":False,"recovery_source":str(source)})
    del train,val
    from research_joint_imaging_v2.evaluate import evaluate
    evaluate(protocol,op,rad,receipt,output,device)


if __name__=="__main__":main()
