"""Render saved scientific arrays; run in an environment with matplotlib."""
import runtime
from pathlib import Path
import sys,json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

results=[]
for argument in sys.argv[1:]:
    out=Path(argument)
    run=out.name
    protocol=json.loads((out.parents[1]/"research_filter_decode/protocol.json").read_text())
    result=json.loads((out/"RESULTS.json").read_text())
    results.append(result)
    for profile in ["balanced", "AlO_weak"]:
        data = np.load(out/f"example_{profile}.npz")
        fig = plt.figure(figsize=(16, 9), constrained_layout=True)
        grid = fig.add_gridspec(3, 5, width_ratios=[1.25, 1, 1, 1, 1])
        ax = fig.add_subplot(grid[:, 0])
        ax.imshow(data["raw_e"], cmap="gray", vmin=0)
        ax.set_title("ONE sensor image\n3 channels overlap", fontsize=14)
        ax.set_axis_off()
        narrow = run.endswith("001")
        for k, wavelength in enumerate([431.5, 486, 516.5]):
            target = data["target"][k]
            vmax = float(target.max())
            error = data["decoded"][k]-target
            fields = [data["pre_optics"][k], target, data["decoded"][k], error]
            titles = ["Before optical limit", "Optical target", "Decoded", "Error / target mean"]
            for j, field in enumerate(fields):
                ax = fig.add_subplot(grid[k, j+1])
                if j == 3:
                    limit = .05 if profile == "balanced" or k != 1 else .40
                    im = ax.imshow(field/target.mean(), cmap="RdBu_r", vmin=-limit, vmax=limit)
                    fig.colorbar(im, ax=ax, fraction=.046, pad=.02)
                else:
                    ax.imshow(field, cmap="gray", vmin=0, vmax=vmax)
                if k == 0:
                    ax.set_title(titles[j], fontsize=12)
                if j == 0:
                    ax.set_ylabel(f"{wavelength:g} nm\nmean {target.mean():.0f} e", fontsize=12)
                if j == 2:
                    rel = np.linalg.norm(error)/np.linalg.norm(target)
                    ax.set_xlabel(f"relative L2 = {rel*100:.2f}%", fontsize=11)
                ax.set_xticks([])
                ax.set_yticks([])
        bandwidth = protocol["model"]["object_cutoff_cpp"]
        fig.suptitle(f"Simulated filter-channel recovery | {profile} | optical cutoff {bandwidth:g} cycles/pixel",
                     fontsize=17)
        fig.savefig(out/f"decoded_{profile}.png", dpi=160)
        plt.close(fig)
        data.close()

fig,axes=plt.subplots(1,3,figsize=(15,4.8),constrained_layout=True)
labels=["431.5 nm","486 nm","516.5 nm"]
names=["Narrow .075","Wide .16","Intermediate .13"]
colors=["#2463a5","#ca5525","#16856b"]
for i,(result,name,color) in enumerate(zip(results,names,colors)):
    balanced=[r["relative_L2_mean"]*100 for r in result["summary"] if r["bright_mean_e"]==4000 and r["profile"]=="balanced"]
    weak=[next(r for r in result["summary"] if r["bright_mean_e"]==4000 and r["profile"]==p and r["channel"]==k)["relative_L2_mean"]*100 for k,p in enumerate(["CH_weak","AlO_weak","C2_weak"])]
    x=np.arange(3)+(i-1)*.25
    axes[0].bar(x,balanced,width=.25,label=name,color=color)
    axes[1].bar(x,weak,width=.25,label=name,color=color)
    axes[2].bar(i,result["optical_detail_loss_relative_L2"]["mean"]*100,width=.6,color=color)
for ax in axes[:2]:
    ax.set_xticks(range(3),labels)
    ax.set_ylabel("Mean relative L2 error (%)")
    ax.legend(fontsize=9)
axes[0].set_title("Balanced: each channel mean 4000 e")
axes[1].set_title("Weak channel: 200 e; others: 4000 e")
axes[2].set_xticks([0,1,2],["Narrow","Wide","Intermediate"])
axes[2].set_ylabel("Mean relative L2 difference (%)")
axes[2].set_title("Optical target vs. pre-optics image")
fig.suptitle("Three-channel simulation: spatial detail and noise trade-off",fontsize=15)
fig.savefig(Path(sys.argv[-1])/"bandwidth_tradeoff.png",dpi=180)
plt.close(fig)
print("Seven scientific figures rendered from saved arrays.")
