"""Frozen calibration, new heldout evaluation, response matrices and mother CIs."""
import runtime
import json
import time
from pathlib import Path
import numpy as np
import torch
from research_joint_imaging_v2.model import TVRRU,MODEL_KEYS
from research_joint_imaging_v2.losses import readouts
from research_joint_imaging_v2.solver import tv_anchor


def model_for(method,protocol,output,device):
    if method in ("tv","tv_400"):return None
    model=TVRRU(protocol["solver"],method).to(device)
    checkpoint=torch.load(output/(method+"_best.pt"),map_location=device,weights_only=False)
    model.load_state_dict(checkpoint["model"]);model.eval()
    return model


@torch.no_grad()
def inference(model,op,tensors,device,anchor_key="anchor"):
    images=[];tick=time.monotonic()
    for start in range(0,len(tensors["anchor"]),4):
        batch={k:v[start:start+4].to(device) for k,v in tensors.items()}
        pred=batch[anchor_key] if model is None else model(op,{k:batch[k] for k in MODEL_KEYS})
        images.append((pred*batch["scale"]).cpu())
    torch.cuda.synchronize()
    return torch.cat(images),time.monotonic()-tick


@torch.no_grad()
def rows_for(rad,images,tensors,metadata,device,thresholds):
    rows=[]
    temperature,excess=readouts(rad,images.to(device))
    true_t,clean_excess=readouts(rad,tensors["target_e"].to(device))
    temperature=temperature.cpu();excess=excess.cpu()
    true_t=true_t.cpu();clean_excess=clean_excess.cpu()
    scale=tensors["scale"]
    sd=(tensors["noise"][:,:3]*scale)[:,:,0,0]
    detected=excess/sd[:,:,None,None]>torch.tensor(thresholds)[None,:,None,None]
    pairing={(r["level_e"],r["repeat"],r["condition"]):i for i,r in enumerate(metadata)}
    for i,meta in enumerate(metadata):
        target=tensors["target_e"][i];pred=images[i];error=pred-target
        centered_target=target-target.mean((-2,-1),keepdim=True)
        centered_error=error-error.mean((-2,-1),keepdim=True)
        centered_den=centered_target.square().sum((-2,-1))
        band_l2=(error.square().sum((-2,-1))/target.square().sum((-2,-1))).sqrt()
        # A declared radiance-scale floor makes flat-field detail metrics finite.
        detail=(centered_error.square().sum((-2,-1))/(centered_den+.0001*target.square().sum((-2,-1)))).sqrt()
        emission=tensors["emission_e"][i]
        active=[k for k in range(3) if bool(emission[k].abs().max()>0)]
        row={**meta,"band_relative_L2":band_l2.tolist(),"band_mean_bias_fraction":(error.mean((-2,-1))/target.mean((-2,-1))).tolist(),"band_centered_relative_L2":detail.tolist(),"detail_denominator":"centered truth energy + 1e-4 total truth energy (1% radiance norm floor, shared by all methods)","temperature_MSE_K2":float((temperature[i]-true_t[i]).square().mean()),"emission_actual_MSE_e2":(excess[i]-emission).square().mean((-2,-1)).tolist(),"emission_clean_readout_MSE_e2":(excess[i]-clean_excess[i]).square().mean((-2,-1)).tolist(),"noiseless_mixing_MSE_e2":(clean_excess[i]-emission).square().mean((-2,-1)).tolist(),"detected_fraction":detected[i].float().mean((-2,-1)).tolist(),"nonpositive_thermal_pixels":int((pred[3:]<=0).any(0).sum()),"temperature_bound_pixels":int(((temperature[i]<=1800)|(temperature[i]>=4500)).sum())}
        if active:
            blank=pairing[(meta["level_e"],meta["repeat"],"blank")]
            response=excess[i]-excess[blank]
            row["injection_response_columns"]={str(j):[(float((response[k]*emission[j]).sum()/emission[j].square().sum())) for k in range(3)] for j in active}
            row["diagonal_response"]={str(j):row["injection_response_columns"][str(j)][j] for j in active}
            row["sensitivity"]={str(j):float(detected[i,j][emission[j]>.2*emission[j].mean()].float().mean()) for j in active}
        rows.append(row)
    return rows,temperature,excess


def aggregate(rows,protocol):
    result={}
    for level in protocol["design"]["counts_800nm_mean_e"]:
        for condition in protocol["design"]["conditions"]:
            group=[r for r in rows if r["level_e"]==level and r["condition"]==condition]
            item={"mothers":len({r["mother"] for r in group}),"frames":len(group),"temperature_RMSE_K":float(np.sqrt(np.mean([r["temperature_MSE_K2"] for r in group]))),"nonpositive_thermal_pixels":sum(r["nonpositive_thermal_pixels"] for r in group),"temperature_bound_pixels":sum(r["temperature_bound_pixels"] for r in group)}
            for key in ("band_relative_L2","band_mean_bias_fraction","band_centered_relative_L2","detected_fraction"):
                item[key+"_mean"]=np.mean([r[key] for r in group],axis=0).tolist()
            for key in ("emission_actual_MSE_e2","emission_clean_readout_MSE_e2","noiseless_mixing_MSE_e2"):
                item[key.replace("MSE_e2","RMSE_e")]=np.sqrt(np.mean([r[key] for r in group],axis=0)).tolist()
            if condition!="blank":
                active=list(group[0]["diagonal_response"])
                item["diagonal_response"]={key:float(np.mean([r["diagonal_response"][key] for r in group])) for key in active}
            result[f"{level}/{condition}"]=item
        matrix=[]
        for condition,j in (("CH_weak",0),("AlO_weak",1),("C2_weak",2)):
            group=[r for r in rows if r["level_e"]==level and r["condition"]==condition]
            matrix.append(np.mean([r["injection_response_columns"][str(j)] for r in group],axis=0))
        result[f"{level}/single_injection_response_matrix"]=np.stack(matrix,axis=1).tolist()
    return result


def mother_bootstrap(all_rows,methods,draws=2000):
    base_rows=all_rows["tv"]
    mothers=sorted({r["mother"] for r in base_rows})
    indices=np.random.default_rng(73191).integers(0,len(mothers),(draws,len(mothers)))
    result={}
    base_values=None
    for method in methods:
        rows=all_rows[method]
        per_mother=[]
        for mother in mothers:
            group=[r for r in rows if r["mother"]==mother and r["level_e"]==4000 and r["condition"]=="all_weak"]
            per_mother.append([np.mean([r["temperature_MSE_K2"] for r in group]),*np.mean([r["emission_actual_MSE_e2"] for r in group],axis=0),*np.mean([r["emission_clean_readout_MSE_e2"] for r in group],axis=0)])
        values=np.asarray(per_mother)
        sampled=np.sqrt(values[indices].mean(axis=1))
        item={"quantity_order":["temperature_RMSE_K","CH_actual_RMSE_e","AlO_actual_RMSE_e","C2_actual_RMSE_e","CH_clean_RMSE_e","AlO_clean_RMSE_e","C2_clean_RMSE_e"],"percentile_95_CI":np.quantile(sampled,[.025,.975],axis=0).tolist(),"draws":draws,"resampling_unit":"independent mother, carrying all noise repeats together"}
        if method=="tv":base_values=sampled
        else:item["paired_RMSE_ratio_to_TV_95_CI"]=np.quantile(sampled/base_values,[.025,.975],axis=0).tolist()
        result[method]=item
    return result


def figures(summaries,example,methods,output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
    t=[summaries[m]["4000/all_weak"]["temperature_RMSE_K"] for m in methods]
    axes[0].bar(methods,t);axes[0].axhline(50,color="r",linestyle="--");axes[0].set_ylabel("Effective color-T RMSE (K)")
    for c,label in enumerate(["431.5","486","516.5"]):
        axes[1].plot(methods,[summaries[m]["4000/all_weak"]["emission_actual_RMSE_e"][c] for m in methods],"o-",label=label+" nm")
    axes[1].set_ylabel("Excess vs actual injection RMSE (e)");axes[1].legend()
    im=axes[2].imshow(summaries["full"]["4000/single_injection_response_matrix"],vmin=-.1,vmax=1.1,cmap="coolwarm")
    axes[2].set_xlabel("Injected channel");axes[2].set_ylabel("Readout channel");axes[2].set_title("Full model: 3x3 weak response")
    axes[2].set_xticks(range(3),["431.5","486","516.5"])
    axes[2].set_yticks(range(3),["431.5","486","516.5"])
    for row in range(3):
        for col in range(3):axes[2].text(col,row,f"{summaries['full']['4000/single_injection_response_matrix'][row][col]:.3f}",ha="center",va="center",color="white")
    fig.colorbar(im,ax=axes[2])
    for ax in axes[:2]:ax.tick_params(axis="x",rotation=30)
    fig.savefig(output/"method_comparison.png",dpi=160);plt.close(fig)
    fig,axes=plt.subplots(6,3,figsize=(10,17),constrained_layout=True)
    for c,wl in enumerate([431.5,486,516.5,700,800,905]):
        target=example["target_e"][c]
        lo,hi=float(target.min()),float(target.max())
        for j,key in enumerate(["target_e","tv","full"]):
            axes[c,j].imshow(example[key][c],vmin=lo,vmax=hi,cmap="gray");axes[c,j].set_title(f"{wl} nm / {key}");axes[c,j].axis("off")
    fig.savefig(output/"channel_reconstruction.png",dpi=140);plt.close(fig)


@torch.no_grad()
def evaluate(protocol,op,rad,receipt,output,device):
    methods=["tv","tv_400",*protocol["learning"]["methods"]]
    thresholds={};model_hashes={}
    for method in methods:
        model=model_for(method,protocol,output,device)
        scores=[]
        for item in receipt["mothers"]:
            if item["split"]!="calibration":continue
            data=torch.load(output/item["file"],map_location="cpu",weights_only=False)
            ids=[i for i,r in enumerate(data["metadata"]) if r["condition"]=="blank"]
            tensors={k:v[ids] for k,v in data["tensors"].items()}
            images,_=inference(model,op,tensors,device,"reference_400" if method=="tv_400" else "anchor")
            _,excess=readouts(rad,images.to(device))
            sd=(tensors["noise"][:,:3]*tensors["scale"])[:,:,0,0].to(device)
            scores.append((excess/sd[:,:,None,None]).cpu())
        scores=torch.cat(scores)
        thresholds[method]=[float(torch.quantile(scores[:,c].flatten(),.99)) for c in range(3)]
        if method not in ("tv","tv_400"):model_hashes[method]=runtime.sha256(output/(method+"_best.pt"))
        del model
    runtime.write_json(output/"THRESHOLDS_FROZEN.json",{"time":runtime.now(),"thresholds":thresholds,"checkpoint_sha256":model_hashes,"source":"independent blank calibration mothers, 99th percentile of excess/upstream noise","heldout_loaded_for_evaluation":False})
    all_rows={};summaries={};example={};timing={}
    online_timing={}
    for method in methods:
        model=model_for(method,protocol,output,device)
        rows=[];predictions=[];elapsed=0.
        for item in receipt["mothers"]:
            if item["split"]!="heldout":continue
            path=output/item["file"]
            if runtime.sha256(path)!=item["sha256"]:raise RuntimeError("Heldout cache changed")
            data=torch.load(path,map_location="cpu",weights_only=False)
            images,seconds=inference(model,op,data["tensors"],device,"reference_400" if method=="tv_400" else "anchor")
            elapsed+=seconds
            if item["mother"]==protocol["design"]["mother_seeds"]["heldout"][0]:
                torch.cuda.synchronize();online_start=time.monotonic()
                raw=data["tensors"]["raw_adc"][20:21].to(device).float()
                q=protocol["model"]["full_well_e"]/((2**protocol["model"]["adc_bits"]-1)-protocol["model"]["black_level_DN"])
                ix,yy,ss,ww,nn=op.initialize((raw-protocol["model"]["black_level_DN"])*q)
                state,history,_=tv_anchor(op,yy,ww,ix,protocol["solver"],iterations=400 if method=="tv_400" else None)
                if method!="tv_400" and not history[-1]["converged"]:raise RuntimeError("Online TV probe not converged")
                fresh=state[0] if model is None else model(op,{"anchor":state[0],"d":state[1],"u":state[2],"initial":ix,"y":yy,"weight":ww,"noise":nn})
                fresh=fresh*ss
                readouts(rad,fresh)
                torch.cuda.synchronize()
                discrepancy=float((fresh.cpu()-images[20:21]).norm()/images[20:21].norm())
                online_timing[method]={"seconds_one_raw_to_readout_including_TV":time.monotonic()-online_start,"mother":item["mother"],"frame_index":20,"batch_size":1,"GPU":torch.cuda.get_device_name(),"cached_batch_vs_single_relative_output_change":discrepancy,"TV_iterations":history[-1]["iteration"],"software_latency_only":True}
                if discrepancy>.001:raise RuntimeError("Single-raw and cached-batch reconstructions disagree")
            mother_rows,_,_=rows_for(rad,images,data["tensors"],data["metadata"],device,thresholds[method])
            rows.extend(mother_rows);predictions.append(images)
            if item["mother"]==protocol["design"]["mother_seeds"]["heldout"][0] and method in ("tv","full"):
                example[method]=images[20].numpy()
                example["target_e"]=data["tensors"]["target_e"][20].numpy()
        all_rows[method]=rows
        summaries[method]=aggregate(rows,protocol)
        timing[method]={"cached_TV_extra_inference_seconds":elapsed,"frames":len(rows),"online_TV_initialization_excluded":True}
        runtime.write_json(output/(method+"_METRICS.json"),{"summary":summaries[method],"per_frame":rows,"timing":timing[method]})
        path=output/(method+"_heldout_predictions.npz")
        np.savez_compressed(path,images=torch.cat(predictions).numpy())
        print({"stage":"heldout_evaluated","method":method,"frames":len(rows)},flush=True)
        del model,predictions
    baseline=summaries["tv"]["4000/all_weak"]
    gates={}
    for method in protocol["learning"]["methods"]:
        row=summaries[method]["4000/all_weak"]
        matrix=np.asarray(summaries[method]["4000/single_injection_response_matrix"])
        off=matrix.copy();np.fill_diagonal(off,0)
        checks={"temperature_50K":row["temperature_RMSE_K"]<=50,"all_band_errors_5percent":max(row["band_relative_L2_mean"])<=.05,"weak_emission_each_improves_15percent":bool(np.all(np.asarray(row["emission_actual_RMSE_e"])/baseline["emission_actual_RMSE_e"]<=.85)),"temperature_no_5percent_regression":row["temperature_RMSE_K"]<=baseline["temperature_RMSE_K"]*1.05,"band_error_no_5percent_regression":bool(np.all(np.asarray(row["band_relative_L2_mean"])/baseline["band_relative_L2_mean"]<=1.05)),"detail_no_5percent_regression":bool(np.all(np.asarray(row["band_centered_relative_L2_mean"])/baseline["band_centered_relative_L2_mean"]<=1.05)),"mean_bias_under_2percent":max(abs(v) for v in row["band_mean_bias_fraction_mean"])<=.02,"blank_FPR_under_2percent":max(max(summaries[method][f"{level}/blank"]["detected_fraction_mean"]) for level in (1000,4000))<=.02,"single_diagonal_0_9_to_1_1":bool(np.all((np.diag(matrix)>=.9)&(np.diag(matrix)<=1.1))),"single_offdiagonal_under_0_05":float(abs(off).max())<=.05,"valid_thermal":sum(s["nonpositive_thermal_pixels"]+s["temperature_bound_pixels"] for k,s in summaries[method].items() if not k.endswith("response_matrix"))==0}
        gates[method]={"checks":checks,"passed":all(checks.values())}
    bootstrap=mother_bootstrap(all_rows,methods)
    runtime.write_json(output/"MOTHER_BOOTSTRAP.json",bootstrap)
    runtime.write_json(output/"ONLINE_TIMING.json",online_timing)
    runtime.write_json(output/"SUMMARY.json",{"time":runtime.now(),"scope":protocol["development_scope"],"seed":protocol["learning"]["seed"],"summaries":summaries,"development_gates":gates,"primary_condition":"4000/all_weak","single_injection_matrix_rows":"readout channel","single_injection_matrix_columns":"injected channel","formal_three_seed_acceptance_established":False,"end_to_end_latency_note":"cached inference timing excludes TV; cache histories and wall time must be included for online cost"})
    figures(summaries,example,methods,output)
    header="# TV-RRU 六通道开发对照\n\n"+protocol["development_scope"]+"；四组神经方法各一个种子；TV强度统一为0.003，数值收敛锚点与固定400轮参考同时保留。\n\n|方法|色温RMSE/K|CH波段总误差/e|AlO波段总误差/e|C2波段总误差/e|开发门槛|\n|---|---:|---:|---:|---:|---|\n"
    for method in methods:
        row=summaries[method]["4000/all_weak"];e=row["emission_actual_RMSE_e"]
        header+=f"|{method}|{row['temperature_RMSE_K']:.3f}|{e[0]:.3f}|{e[1]:.3f}|{e[2]:.3f}|{gates.get(method,{}).get('passed','reference')}|\n"
    header+="\n发光总误差以实际注入为参照；相对无噪读出的算法误差另见各方法METRICS.json。单通道3×3响应矩阵按注入波段列、读出波段行记录。区间按独立母场景成组bootstrap，噪声副本和像素不增加物理样本数。\n\n![方法比较](method_comparison.png)\n\n![通道重建](channel_reconstruction.png)\n\n缓存后的推理时间不等于端到端时延；在线使用须计入TV初始化。当前为假设滤片与相机下的合成开发试验。\n"
    (output/"RESULTS_zh.md").write_text(header,encoding="utf-8")
    artifacts={p.name:{"sha256":runtime.sha256(p),"bytes":p.stat().st_size} for p in output.iterdir() if p.is_file() and p.name not in ("stdout.txt","stderr.txt","PROCESS.json")}
    runtime.write_json(output/"OUTPUT_MANIFEST.json",artifacts)
    runtime.write_json(output/"COMPLETION.json",{"time":runtime.now(),"execution_completed":True,"formal_models":4,"optimizer_updates":4*sum(protocol["learning"]["phases"]),"diagnostic_probe_updates":protocol["software_gates"]["fixed_batch_probe_updates"],"raw_observations":receipt["total_raw_observations"],"full_model_development_gates_pass":gates["full"]["passed"],"formal_three_seed_acceptance":False,"GPU_peak_allocated_MiB":torch.cuda.max_memory_allocated()/1024**2})
