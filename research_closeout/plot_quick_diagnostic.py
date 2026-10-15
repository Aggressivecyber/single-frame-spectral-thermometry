"""Plot the completed native smoke without rerunning any inverse."""
from . import storage as s
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    folder=s.REPORTS/'repair_001/quick_v001'
    result=s.read(folder/'QUICK_RESULT.json')
    rows=[r for r in result['summary']['cells'] if r['family']=='joint']
    fig,axes=plt.subplots(1,2,figsize=(10.4,4.2),layout='constrained')
    colors=('#2166ac','#67a9cf','#b2182b','#ef8a62')
    for row,color in zip(rows,colors):
        info=row['info'];stages=info['stages']
        sign='positive' if 'positive' in row['case_id'] else 'negative'
        init='v3' if row['initializer']=='v3' else 'demod'
        label=sign+' / '+init
        axes[0].plot([r['stage'] for r in stages],
            [r['gradient']['scaled_projected_linf'] for r in stages],marker='o',color=color,label=label)
        comparisons=info['output_comparisons']
        x=[r['to_stage'] for r in comparisons]
        axes[1].plot(x,[r['color_RMS_difference_K']/.2 for r in comparisons],color=color,marker='o')
        axes[1].plot(x,[r['template_RMS_difference_e']/.17 for r in comparisons],color=color,marker='x',linestyle=':')
    axes[0].plot([1,2,3,4],[1e-4,1e-5,1e-6,1e-7],color='black',linestyle='--',label='required tolerance')
    axes[0].set(yscale='log',xlabel='Tightening stage',ylabel='Projected gradient (fixed optimizer coordinates)',
                title='Stationarity remains unqualified',xticks=[1,2,3,4])
    axes[0].legend(fontsize=8,loc='lower left')
    axes[1].axhline(1,color='black',linestyle='--')
    axes[1].set(yscale='log',xlabel='Tighter stage (vs previous stage)',ylabel='Output RMS change / registered budget',
                title='Output differences are within budgets',xticks=[2,3,4])
    axes[1].text(.48,.96,'solid: color / 0.2 K\ndotted: template / 0.17 e',transform=axes[1].transAxes,
                 va='top',fontsize=9,bbox={'facecolor':'white','alpha':.95,'edgecolor':'none'})
    for ax in axes:
        ax.grid(alpha=.2,which='both')
    fig.suptitle('Native256 smoke: output stability and numerical stationarity',fontsize=13)
    fig.supxlabel('Four profile cells shown; two v3-refinement cells reached 180 s. Total certified: 0 / 6.',fontsize=9)
    target=s.REPORTS/'repair_001/quick_v001_analysis'
    target.mkdir(parents=True,exist_ok=True)
    for ext in ('png','svg'):
        output=target/('NATIVE_SMOKE_DIAGNOSTIC_002.'+ext)
        if output.exists():raise FileExistsError(output)
        fig.savefig(output,dpi=200)
    plt.close(fig)
    s.write(target/'FIGURE_RECEIPT_002.json',{'time':s.now(),'source_sha256':s.sha(__file__),
        'data_sha256':s.sha(folder/'QUICK_RESULT.json'),'new_inverse_calls':0,
        'files':[{'path':str(p),'sha256':s.sha(p)} for p in (target/'NATIVE_SMOKE_DIAGNOSTIC_002.png',target/'NATIVE_SMOKE_DIAGNOSTIC_002.svg')],
        'visual_verification':'pending'})
    print(str(target/'NATIVE_SMOKE_DIAGNOSTIC_002.png'))


if __name__=='__main__':main()
