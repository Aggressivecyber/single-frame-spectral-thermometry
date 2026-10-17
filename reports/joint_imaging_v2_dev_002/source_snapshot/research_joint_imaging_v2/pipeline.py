"""Registered server development study: software gates, cache, matched training."""
import runtime
import argparse
import copy
import json
from pathlib import Path
import time
import numpy as np
import torch
from research_joint_imaging.core import Imager,Radiometry
from research_joint_imaging_v2.check_contracts import run as check_contracts
from research_joint_imaging_v2.data import build_cache
from research_joint_imaging_v2.model import TVRRU,MODEL_KEYS
from research_joint_imaging_v2.losses import training_loss,readouts
from research_joint_imaging_v2.solver import anchored_tv_prox


def load_mothers(receipt,output,split):
    result={}
    for item in receipt["mothers"]:
        if item["split"]==split:
            path=output/item["file"]
            if runtime.sha256(path)!=item["sha256"]:
                raise RuntimeError("Frozen cache hash changed: "+str(path))
            result[item["mother"]]=torch.load(path,map_location="cpu",weights_only=False)
    return result


def batch_from_pairs(mothers,rng,device):
    selected=[]
    keys=list(mothers)
    for _ in range(2):
        mother=mothers[keys[int(rng.integers(len(keys)))]]
        level,repeat,condition=int(rng.integers(2)),int(rng.integers(2)),int(rng.integers(1,6))
        for index in (level*12+repeat,level*12+condition*2+repeat):
            selected.append({k:v[index] for k,v in mother["tensors"].items()})
    return {k:torch.stack([row[k] for row in selected]).to(device) for k in selected[0]}


def predict(model,op,tensors,device,chunk=4):
    outputs=[]
    for start in range(0,len(tensors["anchor"]),chunk):
        batch={k:v[start:start+chunk].to(device) for k,v in tensors.items()}
        pred=batch["anchor"] if model is None else model(op,{k:batch[k] for k in MODEL_KEYS})
        outputs.append(pred.cpu())
    return torch.cat(outputs)


@torch.no_grad()
def validation(model,op,rad,mothers,device):
    values=[]
    model.eval()
    for mother in mothers.values():
        ids=[0,8,12,20]
        batch={k:v[ids].to(device) for k,v in mother["tensors"].items()}
        pred=model(op,{k:batch[k] for k in MODEL_KEYS})
        loss,terms=training_loss(rad,pred,batch["target"],batch["scale"],pair_weight=0)
        values.append({"loss":float(loss),"terms":terms,"nonpositive_thermal_pixels":int((pred[:,3:]<=0).sum())})
    model.train()
    return {"selection_score":float(np.mean([row["loss"] for row in values])),"nonpositive_thermal_pixels":sum(row["nonpositive_thermal_pixels"] for row in values),"mothers":len(values),"per_mother":values}


def probe(protocol,op,rad,output,device):
    cfg=copy.deepcopy(protocol)
    cfg["design"]["mother_seeds"]={"train":[480001,480002]}
    cfg["design"]["noise_seed"]=8820241
    folder=output/"software_probe"
    folder.mkdir()
    receipt=build_cache(cfg,op,rad,folder)
    data=load_mothers(receipt,folder,"train")
    batch=batch_from_pairs(data,np.random.default_rng(811),device)
    direction=op.project(torch.randn_like(batch["anchor"]))*.01
    measurement=op.project(torch.randn_like(batch["anchor"]))
    op64=Imager(protocol["model"],torch.float64).to(device)
    original_buffers=dict(op.named_buffers())
    for name,buffer in op64.named_buffers():buffer.copy_(original_buffers[name].to(buffer.dtype))
    gradient_rows=[]
    for point in (0.,.3):
        row={"proposal_point":point,"dtypes":{}}
        for dtype,operator in ((torch.float32,op),(torch.float64,op64)):
            converted={k:batch[k].to(dtype) for k in MODEL_KEYS}
            state=tuple(converted[k] for k in ("anchor","d","u"))
            delta=direction.to(dtype);test=measurement.to(dtype)
            theta=torch.tensor(point,device=device,dtype=dtype,requires_grad=True)
            px=anchored_tv_prox(operator,converted["y"],converted["weight"],converted["anchor"]+theta*delta,state,protocol["solver"])[0]
            ad=torch.autograd.grad((px*test).sum(),theta)[0]
            steps=(.002,.05,.1) if dtype==torch.float32 else (.0001,)
            differences=[]
            for h in steps:
                with torch.no_grad():
                    plus=anchored_tv_prox(operator,converted["y"],converted["weight"],converted["anchor"]+(point+h)*delta,state,protocol["solver"])[0]
                    minus=anchored_tv_prox(operator,converted["y"],converted["weight"],converted["anchor"]+(point-h)*delta,state,protocol["solver"])[0]
                    fd=((plus.double()-minus.double())*test.double()).sum()/(2*h)
                    relative=float(abs(ad-fd)/torch.maximum(abs(ad),abs(fd)))
                differences.append({"epsilon":h,"finite_difference":float(fd),"relative_error":relative,"gate":dtype==torch.float64 or h>=.05})
            row["dtypes"][str(dtype)]={"autograd":float(ad),"finite_differences":differences}
        gradient_rows.append(row)
    native_fd_error=max(item["relative_error"] for row in gradient_rows for item in row["dtypes"]["torch.float32"]["finite_differences"] if item["gate"])
    lifted_error=max(abs(row["dtypes"]["torch.float32"]["autograd"]-row["dtypes"]["torch.float64"]["autograd"])/abs(row["dtypes"]["torch.float64"]["autograd"]) for row in gradient_rows)
    double_fd_error=max(row["dtypes"]["torch.float64"]["finite_differences"][0]["relative_error"] for row in gradient_rows)
    gradient_pass=native_fd_error<.03 and lifted_error<.0001 and double_fd_error<.002
    runtime.write_json(output/"FLOAT32_PROX_GRADIENT.json",{"time":runtime.now(),"native_float32_max_gated_relative_error":native_fd_error,"float32_vs_identical_lifted_operator_float64_AD_max_relative":lifted_error,"lifted_float64_FD_max_relative":double_fd_error,"rows":gradient_rows,"passed":gradient_pass,"small_float32_step":"0.002 retained diagnostically; roundoff cancellation established in fd_diagnostic_001"})
    if not gradient_pass:raise RuntimeError("Native multiscale float32/float64 proximal gradient gate failed")
    del px,theta,op64,original_buffers
    torch.manual_seed(811)
    model=TVRRU(protocol["solver"]).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=protocol["learning"]["learning_rate"],weight_decay=protocol["learning"]["weight_decay"])
    rows=[]
    start=time.monotonic()
    updates=protocol["software_gates"]["fixed_batch_probe_updates"]
    for step in range(updates):
        model.train(); optimizer.zero_grad(set_to_none=True)
        pred=model(op,{k:batch[k] for k in MODEL_KEYS})
        loss,terms=training_loss(rad,pred,batch["target"],batch["scale"])
        if not bool(torch.isfinite(loss)) or bool((pred[:,3:]<=0).any()):
            raise RuntimeError("Invalid native float32 probe output")
        loss.backward()
        if not all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None):
            raise RuntimeError("Nonfinite native float32 probe gradient")
        norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
        optimizer.step()
        row={"step":step+1,"loss":float(loss.detach()),"unclipped_gradient_norm":norm,"terms":terms}
        rows.append(row)
        runtime.write_json(output/"NATIVE_TRAINING_PROBE_PROGRESS.json",{"time":runtime.now(),"latest":row,"wall_seconds":time.monotonic()-start})
    result={"time":runtime.now(),"updates":updates,"fixed_batch_frames":4,"diagnostic_mothers":[480001,480002],"inherited_by_formal_models":False,"first_loss":rows[0]["loss"],"last_loss":rows[-1]["loss"],"loss_ratio":rows[-1]["loss"]/rows[0]["loss"],"float32_prox_FD_relative_error":native_fd_error,"all_gradients_finite":True,"rows":rows,"wall_seconds":time.monotonic()-start,"peak_GPU_MiB":torch.cuda.max_memory_allocated()/1024**2}
    result["passed"]=bool(result["loss_ratio"]<=protocol["software_gates"]["fixed_batch_loss_ratio_max"])
    runtime.write_json(output/"NATIVE_TRAINING_PROBE.json",result)
    if not result["passed"]:
        raise RuntimeError("Fixed-batch overfit probe did not improve")
    del model,optimizer,data,batch
    torch.cuda.empty_cache()


def train_method(protocol,op,rad,train,val,output,device,method):
    torch.manual_seed(protocol["learning"]["seed"])
    rng=np.random.default_rng(protocol["learning"]["seed"])
    model=TVRRU(protocol["solver"],method).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=protocol["learning"]["learning_rate"],weight_decay=protocol["learning"]["weight_decay"])
    best=float("inf"); best_step=None; start=time.monotonic()
    journal=(output/(method+"_training.jsonl")).open("x",encoding="utf-8")
    a,b,c=protocol["learning"]["phases"]
    validation_rows=[]
    for step in range(1,a+b+c+1):
        phase="thermal" if step<=a else "visible" if step<=a+b else "joint"
        for p in model.prior.thermal.parameters(): p.requires_grad_(phase!="visible")
        for p in model.prior.visible.parameters(): p.requires_grad_(phase!="thermal")
        for group in optimizer.param_groups:
            group["lr"]=protocol["learning"]["joint_learning_rate"] if phase=="joint" else protocol["learning"]["learning_rate"]
        batch=batch_from_pairs(train,rng,device)
        model.prior.capture_stats=step==1 or step%20==0
        model.train(); optimizer.zero_grad(set_to_none=True)
        pred=model(op,{k:batch[k] for k in MODEL_KEYS},thermal_only=phase=="thermal")
        loss,terms=training_loss(rad,pred,batch["target"],batch["scale"],pair_weight=0. if method=="no_pair" else 1.,thermal_only=phase=="thermal")
        if not bool(torch.isfinite(loss)) or bool((pred[:,3:]<=0).any()):
            raise RuntimeError(f"Invalid training output: {method} {step}")
        loss.backward()
        if not all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None):
            raise RuntimeError(f"Nonfinite training gradients: {method} {step}")
        norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),protocol["learning"]["gradient_clip"]))
        optimizer.step()
        row={"time":runtime.now(),"method":method,"step":step,"phase":phase,"loss":float(loss.detach()),"gradient_norm_before_clip":norm,"terms":terms,"wall_seconds":time.monotonic()-start}
        if model.prior.capture_stats:row["last_stage_prior_statistics"]=model.prior.statistics
        journal.write(json.dumps(row)+"\n");journal.flush()
        if step==1 or step%20==0:
            runtime.write_json(output/"TRAINING_PROGRESS.json",row)
        if step%protocol["learning"]["validation_every"]==0 or step==a+b+c:
            result=validation(model,op,rad,val,device)
            result.update(method=method,step=step,phase=phase,time=runtime.now())
            validation_rows.append(result)
            if result["nonpositive_thermal_pixels"]:
                raise RuntimeError("Invalid thermal validation pixels")
            if result["selection_score"]<best:
                best=result["selection_score"];best_step=step
                torch.save({"model":model.state_dict(),"step":step,"method":method,"seed":protocol["learning"]["seed"],"validation":result},output/(method+"_best.pt"))
            runtime.write_json(output/(method+"_validation.json"),{"rows":validation_rows,"best_step":best_step,"best_score":best})
            print({"stage":"validation","method":method,"step":step,"score":result["selection_score"],"best_step":best_step},flush=True)
    torch.save({"model":model.state_dict(),"step":a+b+c,"method":method,"seed":protocol["learning"]["seed"]},output/(method+"_last.pt"))
    journal.close()
    result={"method":method,"updates":a+b+c,"best_step":best_step,"best_validation_score":best,"wall_seconds":time.monotonic()-start,"parameters":sum(p.numel() for p in model.parameters()),"best_checkpoint_sha256":runtime.sha256(output/(method+"_best.pt")),"last_checkpoint_sha256":runtime.sha256(output/(method+"_last.pt"))}
    runtime.write_json(output/(method+"_TRAINING_COMPLETION.json"),result)
    del model,optimizer
    torch.cuda.empty_cache()
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--protocol",type=Path,default=Path(__file__).with_name("protocol.json"))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--software-only",action="store_true")
    args=parser.parse_args()
    protocol=json.loads(args.protocol.read_text())
    registration=json.loads(args.protocol.with_name("REGISTRATION.json").read_text())
    root=Path(__file__).resolve().parents[1]
    for name,expected in registration["source_sha256"].items():
        if runtime.sha256(root/name)!=expected:
            raise RuntimeError("Source hash changed: "+name)
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    if (output/"PIPELINE_START.json").exists():
        raise FileExistsError("Run directory already started")
    runtime.write_json(output/"PIPELINE_START.json",{"time":runtime.now(),"protocol_sha256":runtime.sha256(args.protocol),"source_hashes_verified":True,"training_model_seed":protocol["learning"]["seed"],"software_only":args.software_only})
    torch.set_num_threads(protocol["resources"]["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    device="cuda"
    torch.cuda.set_per_process_memory_fraction(protocol["resources"]["GPU_allocator_limit_MiB"]*1024**2/torch.cuda.get_device_properties(0).total_memory)
    contract=check_contracts(protocol,device)
    runtime.write_json(output/"CUDA_CONTRACTS.json",contract)
    if not contract["passed"]:
        raise RuntimeError("Native CUDA contracts failed: "+str(contract["checks"]))
    op=Imager(protocol["model"]).to(device)
    rad=Radiometry(protocol,device)
    probe(protocol,op,rad,output,device)
    if args.software_only:
        runtime.write_json(output/"COMPLETION.json",{"time":runtime.now(),"software_complete":True,"formal_training_started":False})
        return
    receipt=build_cache(protocol,op,rad,output)
    train=load_mothers(receipt,output,"train")
    val=load_mothers(receipt,output,"validation")
    completions=[]
    for method in protocol["learning"]["methods"]:
        completions.append(train_method(protocol,op,rad,train,val,output,device,method))
    runtime.write_json(output/"SELECTION_FROZEN.json",{"time":runtime.now(),"methods":completions,"heldout_loaded_for_evaluation":False})
    del train,val
    from research_joint_imaging_v2.evaluate import evaluate
    evaluate(protocol,op,rad,receipt,output,device)


if __name__=="__main__": main()
