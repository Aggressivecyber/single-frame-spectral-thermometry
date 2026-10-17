"""Single-raw deployment path; no targets, injections or blank partner required."""
import runtime
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from research_joint_imaging.core import Imager,Radiometry
from research_joint_imaging_v2.model import TVRRU
from research_joint_imaging_v2.solver import tv_anchor
from research_joint_imaging_v2.losses import readouts


parser=argparse.ArgumentParser()
parser.add_argument("--raw",type=Path,required=True)
parser.add_argument("--checkpoint",type=Path,required=True)
parser.add_argument("--output",type=Path,required=True)
args=parser.parse_args()
protocol=json.loads(Path(__file__).with_name("protocol.json").read_text())
raw=np.load(args.raw,allow_pickle=False)
if raw.shape!=tuple(protocol["model"]["shape"]) or not np.issubdtype(raw.dtype,np.integer):
    raise ValueError("Expected one native128x128 integer ADC frame")
if raw.min()<0 or raw.max()>=2**protocol["model"]["adc_bits"]-1:
    raise ValueError("Raw ADC outside the unsaturated calibrated range")
args.output.mkdir(parents=True,exist_ok=False)
torch.set_num_threads(2)
device="cuda"
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
op=Imager(protocol["model"]).to(device)
rad=Radiometry(protocol,device)
checkpoint=torch.load(args.checkpoint,map_location=device,weights_only=False)
model=TVRRU(protocol["solver"],checkpoint["method"]).to(device)
model.load_state_dict(checkpoint["model"]);model.eval()
torch.cuda.synchronize();start=time.monotonic()
with torch.no_grad():
    q=protocol["model"]["full_well_e"]/((2**protocol["model"]["adc_bits"]-1)-protocol["model"]["black_level_DN"])
    initial,y,scale,weight,noise=op.initialize((torch.as_tensor(raw.astype(np.float32),device=device)[None]-protocol["model"]["black_level_DN"])*q)
    state,history,_=tv_anchor(op,y,weight,initial,protocol["solver"])
    if not history[-1]["converged"]:raise RuntimeError("TV initialization did not converge")
    prediction=model(op,{"anchor":state[0],"d":state[1],"u":state[2],"initial":initial,"y":y,"weight":weight,"noise":noise})*scale
    if bool((prediction[:,3:]<=0).any()):raise RuntimeError("Nonpositive reconstructed thermal band")
    temperature,excess=readouts(rad,prediction)
    torch.cuda.synchronize();elapsed=time.monotonic()-start
np.savez_compressed(args.output/"prediction.npz",images_e=prediction[0].cpu().numpy(),effective_color_temperature_K=temperature[0].cpu().numpy(),band_excess_e=excess[0].cpu().numpy())
runtime.write_json(args.output/"INFERENCE.json",{"time":runtime.now(),"raw_sha256":runtime.sha256(args.raw),"checkpoint_sha256":runtime.sha256(args.checkpoint),"method":checkpoint["method"],"device":str(torch.cuda.get_device_name()),"wall_seconds_including_TV_initialization":elapsed,"TV_solver":history,"single_input_frame":True,"label_or_blank_partner_used":False,"calibration":"hypothetical frozen six-band software calibration"})
