"""Native geometry, finite-iteration derivatives and live zero-head gradients."""
import runtime
import argparse
import copy
import json
import time
from pathlib import Path
import torch
from research_joint_imaging.core import Imager,Radiometry
from research_joint_imaging_v2.solver import tv_anchor,anchored_tv_prox,residuals
from research_joint_imaging_v2.model import TVRRU,thermal_coordinates,coordinates_to_logs,sideband_residual
from research_joint_imaging_v2.losses import readouts


def run(protocol,device,quick=False):
    start=time.monotonic()
    torch.manual_seed(78101)
    op=Imager(protocol["model"],torch.float64).to(device)
    rad=Radiometry(protocol,device,torch.float64)
    shape=protocol["model"]["shape"]
    spectrum,_=rad.curve(torch.full(shape,3000.,dtype=torch.float64,device=device))
    yy,xx=torch.meshgrid(torch.arange(shape[0],device=device),torch.arange(shape[1],device=device),indexing="ij")
    target=op.object_blur(spectrum*(1+.05*torch.cos(2*torch.pi*xx/shape[1])))
    raw=op.encode_images(target)[None]*4000
    if not quick:
        raw=raw+raw.sqrt()*torch.randn_like(raw)
    initial,y,scale,weight,noise=op.initialize(raw)
    cfg=copy.deepcopy(protocol["solver"])
    if quick:
        cfg.update(maximum_iterations=40,prox_admm_iterations=2,prox_pcg_iterations=6)
    state,history,_=tv_anchor(op,y,weight,initial,cfg)
    anchor=state[0]
    double_cfg={**cfg,"prox_admm_iterations":2*cfg["prox_admm_iterations"]}
    zero_state=anchored_tv_prox(op,y,weight,anchor,state,cfg)
    zero_change=float((zero_state[0]-anchor).norm()/anchor.norm())
    direction=op.project(torch.randn_like(anchor))*.01
    probe=op.project(torch.randn_like(anchor))
    epsilon=1e-4
    theta=torch.zeros((),device=device,dtype=torch.float64,requires_grad=True)
    predicted=anchored_tv_prox(op,y,weight,anchor+theta*direction,state,cfg)[0]
    derivative=torch.autograd.grad((predicted*probe).sum(),theta)[0]
    with torch.no_grad():
        plus=anchored_tv_prox(op,y,weight,anchor+epsilon*direction,state,cfg)[0]
        minus=anchored_tv_prox(op,y,weight,anchor-epsilon*direction,state,cfg)[0]
        finite=((plus-minus)*probe).sum()/(2*epsilon)
    derivative_relative=float(abs(derivative-finite)/torch.maximum(abs(derivative),abs(finite)))
    positive=torch.rand(1,3,*shape,device=device,dtype=torch.float64)+.5
    coordinate_error=float((coordinates_to_logs(thermal_coordinates(positive)).exp()-positive).abs().max())
    batch={"anchor":anchor,"d":state[1],"u":state[2],"initial":initial,"y":y,"weight":weight,"noise":noise}
    model=TVRRU(cfg).to(device,dtype=torch.float64)
    model.stages=1 if quick else 3
    model.train()
    output=model(op,batch)
    loss=(output-target[None]/(scale/4000)).square().mean()
    loss.backward()
    head_gradient=float(model.prior.thermal.head.weight.grad.norm()+model.prior.visible.head.weight.grad.norm())
    gradient_finite=all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
    model.zero_grad()
    with torch.no_grad():
        model.prior.visible.head.weight.normal_(0,.001)
        model.prior.thermal.head.weight.normal_(0,.001)
        residual=sideband_residual(op,y-op.encode_images(anchor))
        p1=model.prior(op,anchor,anchor,initial,noise,residual,0)
        changed=anchor.clone(); changed[:,1]+=.2
        p2=model.prior(op,changed,anchor,initial,noise,residual,0)
        isolation=float((p1[:,0]-p2[:,0]).abs().max())
        proposal=anchor+p1
        short=anchored_tv_prox(op,y,weight,proposal,state,cfg)
        longer=anchored_tv_prox(op,y,weight,proposal,state,double_cfg)
        doubled_change=float((short[0]-longer[0]).norm()/longer[0].norm())
        low=op.project(short[0])-short[0]
        support_error=float(low.norm()/short[0].norm())
    # The unchanged readout must have a correct derivative, including batch reshape.
    radiometric_images=torch.stack((target,target*1.01)).detach().requires_grad_()
    rt,_=readouts(rad,radiometric_images)
    rt.mean().backward()
    rd_direction=torch.randn_like(radiometric_images)*radiometric_images.detach()*.01
    ad=(radiometric_images.grad*rd_direction).sum()
    with torch.no_grad():
        tp,_=readouts(rad,radiometric_images+epsilon*rd_direction)
        tm,_=readouts(rad,radiometric_images-epsilon*rd_direction)
        fd=(tp.mean()-tm.mean())/(2*epsilon)
    readout_relative=float(abs(ad-fd)/torch.maximum(abs(ad),abs(fd)))
    result={"time":runtime.now(),"device":device,"dtype":"float64","quick":quick,"native_shape":shape,"parameters_grouped":sum(p.numel() for p in model.parameters()),"parameters_ordinary":sum(p.numel() for p in TVRRU(cfg,"ordinary_post").parameters()),"zero_proposal_relative_change":zero_change,"zero_heads_output_relative_change":float((output-anchor).detach().norm()/anchor.norm()),"zero_head_gradient_norm":head_gradient,"all_parameter_gradients_finite":gradient_finite,"prox_directional_derivative_relative_error":derivative_relative,"readout_directional_derivative_relative_error":readout_relative,"thermal_coordinates_max_error":coordinate_error,"visible_proposal_cross_input_max_change":isolation,"doubled_prox_output_relative_change":doubled_change,"support_relative_error":support_error,"anchor_solver":history,"prox_residuals":residuals(op,y,weight,short,cfg["strength"],cfg["rho"],proposal,op.prior_mu[:,None,None]*cfg["mu_multiplier"]),"wall_seconds":time.monotonic()-start}
    checks={"zero_proposal":zero_change<1e-4,"zero_heads":result["zero_heads_output_relative_change"]<1e-4,"live_gradient":head_gradient>0 and gradient_finite,"prox_gradient":derivative_relative<.002,"readout_gradient":readout_relative<.002,"coordinates":coordinate_error<1e-12,"visible_isolation":isolation==0,"support":support_error<1e-10,"parameters":result["parameters_grouped"]<=1_000_000,"prox_budget":doubled_change<.001}
    if not quick:checks["anchor_converged"]=history[-1]["converged"]
    result.update(checks=checks,passed=all(checks.values()))
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--device",choices=["cpu","cuda"],required=True)
    parser.add_argument("--quick",action="store_true")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    protocol=json.loads((Path(__file__).with_name("protocol.json")).read_text())
    torch.set_num_threads(2)
    result=run(protocol,args.device,args.quick)
    runtime.write_json(args.output,result)
    print(json.dumps(result,ensure_ascii=False))
    assert result["passed"],result["checks"]
