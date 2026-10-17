import sys,json,pathlib,torch,numpy as np
root=pathlib.Path("/home/rylan/projects/thermometry-neural-20261011/joint_imaging_v2_dev_001")
sys.path.insert(0,str(root))
import runtime
from research_joint_imaging.core import Imager,Radiometry
from research_joint_imaging_v2.check_contracts import run
from research_joint_imaging_v2.pipeline import load_mothers,batch_from_pairs
from research_joint_imaging_v2.solver import anchored_tv_prox
p=json.loads((root/"research_joint_imaging_v2/protocol.json").read_text())
torch.set_num_threads(2)
run(p,"cuda")
op=Imager(p["model"]).cuda()
folder=root/"reports/joint_imaging_v2_dev_001/software_probe"
receipt=json.loads((folder/"DATA_RECEIPT.json").read_text())
batch=batch_from_pairs(load_mothers(receipt,folder,"train"),np.random.default_rng(811),"cuda")
direction=op.project(torch.randn_like(batch["anchor"]))*0.01
measurement=op.project(torch.randn_like(batch["anchor"]))
op64=Imager(p["model"],torch.float64).cuda()
buffers=dict(op.named_buffers())
for name,buffer in op64.named_buffers():buffer.copy_(buffers[name].to(buffer.dtype))
rows=[]
for dtype,operator in ((torch.float32,op),(torch.float64,op64)):
 b={k:v.to(dtype) for k,v in batch.items()}
 state=tuple(b[k] for k in ("anchor","d","u"))
 d=direction.to(dtype);probe=measurement.to(dtype)
 for point in (0.,0.3):
  theta=torch.tensor(point,device="cuda",dtype=dtype,requires_grad=True)
  pred=anchored_tv_prox(operator,b["y"],b["weight"],b["anchor"]+theta*d,state,p["solver"])[0]
  ad=torch.autograd.grad((pred*probe).sum(),theta)[0]
  differences=[]
  for h in (0.0001,0.0005,0.001,0.002,0.005,0.01,0.02,0.05,0.1):
   with torch.no_grad():
    plus=anchored_tv_prox(operator,b["y"],b["weight"],b["anchor"]+(point+h)*d,state,p["solver"])[0]
    minus=anchored_tv_prox(operator,b["y"],b["weight"],b["anchor"]+(point-h)*d,state,p["solver"])[0]
    fd=((plus.double()-minus.double())*probe.double()).sum()/(2*h)
    err=float(abs(ad-fd)/torch.maximum(abs(ad),abs(fd)))
   differences.append({"h":h,"finite_difference":float(fd),"relative_error":err})
  row={"dtype":str(dtype),"point":point,"autograd":float(ad),"differences":differences}
  rows.append(row);print(json.dumps(row),flush=True)
runtime.write_json(root.parent/"joint_imaging_v2_fd_diagnostic_001.json",{"time":runtime.now(),"source_root":str(root),"operator_float64_buffers_lifted_from_float32":True,"same_frozen_raw_and_ADMM_states":True,"rows":rows})
