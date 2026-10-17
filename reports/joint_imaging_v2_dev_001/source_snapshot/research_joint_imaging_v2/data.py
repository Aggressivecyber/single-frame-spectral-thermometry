"""New independent morphology streams and one WTV cache per acquired raw."""
import runtime
from pathlib import Path
import numpy as np
import torch
from research_joint_imaging_v2.solver import tv_anchor,anchored_tv_prox


def mother(seed,shape,device):
    streams=[np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(5)]
    yy,xx=np.meshgrid(np.linspace(-1,1,shape[0],endpoint=False),np.linspace(-1,1,shape[1],endpoint=False),indexing="ij")
    rng=streams[0]
    angle,phase=rng.uniform(0,2*np.pi,2)
    front=np.tanh((xx*np.cos(angle)+yy*np.sin(angle)-rng.uniform(-.4,.4))/.12)
    temperature=3000+600*np.tanh((rng.uniform(-230,230)+130*np.sin(np.pi*(xx+yy)+phase)+150*front)/600)
    rng=streams[1]
    amplitude=.6+.2*np.cos(np.pi*xx+rng.uniform(0,6))*np.cos(np.pi*yy+rng.uniform(0,6))
    features=[]
    families=[]
    for rng in streams[2:]:
        kind=int(rng.integers(6)); angle=rng.uniform(0,2*np.pi)
        u=xx*np.cos(angle)+yy*np.sin(angle)
        v=-xx*np.sin(angle)+yy*np.cos(angle)
        cx,cy=rng.uniform(-.5,.5,2)
        field=np.full(shape,.02)
        if kind==0:
            for _ in range(5):
                a,b=rng.uniform(-.65,.65,2)
                field+=rng.uniform(.3,1)*np.exp(-.5*(((u-a)/rng.uniform(.1,.35))**2+((v-b)/rng.uniform(.035,.10))**2))
        elif kind==1:
            radius=np.sqrt((xx-cx)**2+(yy-cy)**2)
            field+=np.exp(-.5*((radius-rng.uniform(.15,.55))/.06)**2)
        elif kind==2:
            field+=(.5+.5*np.cos(rng.integers(1,4)*np.pi*u+rng.uniform(0,6)))**4
        elif kind==3:
            field+=np.exp(-.5*((v-.2*np.sin(2*np.pi*u))/.045)**2)*np.exp(-.5*(u/.65)**2)
        elif kind==4:
            field+=.5+.5*np.tanh((u-rng.uniform(-.3,.3))/.08)
        else:
            field+=1
        features.append(field/field.mean()); families.append(kind)
    return tuple(torch.tensor(x,dtype=torch.float32,device=device) for x in (temperature,amplitude,np.stack(features))),families


@torch.no_grad()
def build_cache(protocol,op,rad,output):
    folder=output/"cache"
    folder.mkdir()
    receipt={"time":runtime.now(),"mothers":[],"total_raw_observations":0,"bytes":0,"source":"new generator with five independent morphology streams per mother","old_heldout_used":False}
    q=protocol["model"]["full_well_e"]/((2**protocol["model"]["adc_bits"]-1)-protocol["model"]["black_level_DN"])
    global_sample=0
    for split,(first,last) in protocol["design"]["mother_seeds"].items():
        for seed in range(first,last+1):
            (temperature,amplitude,features),families=mother(seed,protocol["model"]["shape"],op.patterns.device)
            spectrum,_=rad.curve(temperature)
            base=spectrum*amplitude
            records={k:[] for k in ("raw_adc","target_e","emission_e")}
            metadata=[]
            for level in protocol["design"]["counts_800nm_mean_e"]:
                continuum=base*(level/base[4].mean())
                for condition,amounts in protocol["design"]["conditions"].items():
                    emission=features*torch.tensor(amounts,device=features.device)[:,None,None]*level
                    pre=continuum.clone(); pre[:3]+=emission
                    target=op.object_blur(pre)
                    expectation=op.encode_images(target)
                    if float(expectation.min())<=0:
                        raise ValueError("Nonpositive acquisition expectation")
                    for repeat in range(protocol["design"]["noise_repeats"]):
                        noise_seed=protocol["design"]["noise_seed"]+global_sample*1009
                        generator=torch.Generator(device=features.device).manual_seed(noise_seed)
                        electrons=torch.poisson(expectation,generator=generator)+protocol["model"]["read_noise_e"]*torch.randn(expectation.shape,device=features.device,generator=generator)
                        dn=torch.round(electrons/q+protocol["model"]["black_level_DN"])
                        if bool(((dn<0)|(dn>2**protocol["model"]["adc_bits"]-1)).any()):
                            raise ValueError("ADC saturation in new cache")
                        records["raw_adc"].append(dn.to(torch.uint16))
                        records["target_e"].append(target)
                        records["emission_e"].append(op.object_blur(emission))
                        metadata.append({"mother":seed,"condition":condition,"level_e":level,"repeat":repeat,"noise_seed":noise_seed,"feature_families":families})
                        global_sample+=1
            data={k:torch.stack(v) for k,v in records.items()}
            initial,y,scale,weight,noise=op.initialize((data["raw_adc"].float()-protocol["model"]["black_level_DN"])*q)
            cached={"initial":initial,"y":y,"scale":scale,"weight":weight,"noise":noise,"target":data["target_e"]/scale,**data}
            states=[]; histories=[];references=[]
            for start in range(0,len(metadata),6):
                state,history,reference=tv_anchor(op,y[start:start+6],weight[start:start+6],initial[start:start+6],protocol["solver"])
                if not history[-1]["converged"]:
                    raise RuntimeError(f"WTV anchor failed numerical convergence: {seed} batch {start}: {history[-1]}")
                states.append(state); histories.append(history);references.append(reference)
            cached.update(anchor=torch.cat([s[0] for s in states]),d=torch.cat([s[1] for s in states]),u=torch.cat([s[2] for s in states]),reference_400=torch.cat(references))
            if bool((cached["anchor"][:,3:]<=0).any()):
                raise ValueError("Nonpositive thermal TV anchor")
            audit_ids=[0,20]
            audit_state=tuple(cached[k][audit_ids] for k in ("anchor","d","u"))
            continued=anchored_tv_prox(op,y[audit_ids],weight[audit_ids],audit_state[0],audit_state,protocol["solver"])
            identity_change=float((continued[0]-audit_state[0]).norm()/audit_state[0].norm())
            if identity_change>protocol["software_gates"]["anchor_identity_relative_max"]:
                raise RuntimeError(f"TV anchor identity not converged: {seed}: {identity_change}")
            path=folder/f"{split}_{seed}.pt"
            torch.save({"tensors":{k:v.cpu() for k,v in cached.items()},"metadata":metadata,"solver_history":histories},path)
            item={"split":split,"mother":seed,"file":str(path.relative_to(output)),"sha256":runtime.sha256(path),"bytes":path.stat().st_size,"observations":len(metadata),"identity_audit_relative_change":identity_change}
            receipt["mothers"].append(item)
            receipt["total_raw_observations"]+=len(metadata)
            receipt["bytes"]+=item["bytes"]
            if receipt["bytes"]>protocol["resources"]["maximum_new_artifact_bytes"]:
                raise RuntimeError("Registered cache allocation exceeded")
            runtime.write_json(output/"CACHE_PROGRESS.json",{"time":runtime.now(),"completed_mothers":len(receipt["mothers"]),"latest":item,"raw_observations":receipt["total_raw_observations"],"bytes":receipt["bytes"]})
            print({"stage":"TV_cache_saved",**item},flush=True)
    runtime.write_json(output/"DATA_RECEIPT.json",receipt)
    return receipt
