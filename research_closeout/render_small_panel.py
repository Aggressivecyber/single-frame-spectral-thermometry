"""Supplementary development diagnostic; no fitting or method selection."""
from pathlib import Path
import json
import os
import hashlib


def run(revision=1):
    from . import storage as s
    folder=s.ROOT/'paper/figures/supplement_quick_v003'
    folder.mkdir(parents=True,exist_ok=True)
    cache=s.DATA/'repair_001/quick_v003/runtime_cache/mpl'
    cache.mkdir(parents=True,exist_ok=True)
    os.environ['MPLCONFIGDIR']=str(cache)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    source=s.REPORTS/'repair_001/quick_v003/NATIVE_PANEL_ANALYSIS.json'
    data=s.read(source)
    methods=('v3_physics','B1_lambda100','old_B4','C1v3__v3')
    names=('v3','Tikhonov 100','Frozen B4','C1 (v3 init)')
    colors=('#0072B2','#E69F00','#009E73','#CC79A7')
    cases={row['case_id'] for row in data['rows']}
    per_method={method:[] for method in methods}
    for filename in ('development_baseline_reuse.jsonl','development_results.jsonl'):
        for row in s.rows(s.REPORTS/filename):
            if row.get('case_id') in cases and row.get('method') in per_method:
                per_method[row['method']].append(row['metrics'])
    per_method['C1v3__v3']=[row['metrics'] for row in data['rows'] if row['initializer']=='v3']
    assert all(len(rows)==6 for rows in per_method.values())
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
        'axes.titlesize':11,'axes.labelsize':10,'savefig.dpi':180,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(10.8,4.4))
    for ax,field,label,title in zip(axes,
        ('temperature_RMSE_K','template_RMSE_e'),
        ('Color-temperature RMSE / K','Electron-template RMSE / e_ref'),
        ('a  Color temperature','b  Signed template')):
        values=[data['pooled_groups'][method]['statistics']['all'][field] for method in methods]
        ax.bar(range(4),values,color=colors,width=.64,alpha=.8,zorder=2)
        for index,method in enumerate(methods):
            individual=[row[field] for row in per_method[method]]
            ax.scatter(index+np.linspace(-.19,.19,6),individual,s=19,c='black',zorder=3)
        ax.set_xticks(range(4),[f'{name}\n{value:.2f}' for name,value in zip(names,values)],rotation=0)
        ax.set_ylabel(label);ax.set_title(title,loc='left')
        ax.grid(axis='y',alpha=.2,zorder=0)
        ax.set_ylim(0,max(max(row[field] for rows in per_method.values() for row in rows),max(values))*1.18)
    fig.suptitle('Development diagnostic: 6 raw cases, 5 parent scenes; fixed c1s1',y=.98,fontsize=12)
    fig.text(.04,.02,'Bars and tick values: pooled full-FOV RMSE. Dots: individual raw cases.\n'
        'Baselines: frozen cached metrics. C1: native byte reconstruction and rescoring; no method selected.',fontsize=9)
    fig.tight_layout(rect=(0,.12,1,.94))
    outputs=[]
    for suffix in ('png','pdf'):
        stem='C1_DEVELOPMENT_TRADEOFF'+('' if revision==1 else f'_{revision:03d}')
        path=folder/(stem+'.'+suffix)
        assert not path.exists(),'Retain the first rendered diagnostic'
        fig.savefig(path,bbox_inches='tight')
        outputs.append({'path':str(path),'sha256':s.sha(path),'bytes':path.stat().st_size})
    plt.close(fig)
    receipt={'time':s.now(),'source':{'path':str(source),'sha256':s.sha(source)},
        'plot_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'outputs':outputs,'revision':revision,'visual_QA':'pending','supplement_only':True,
        'seven_final_figure_groups_completed':False,'complete_inverse_calls':0}
    receipt_name='TRADEOFF_FIGURE'+('' if revision==1 else f'_{revision:03d}')+'.json'
    s.write(s.REPORTS/'repair_001/quick_v003'/receipt_name,receipt)
    print(json.dumps({'rendered':len(outputs),'supplement_only':True}),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision',type=int,default=1)
    args=parser.parse_args()
    if not 1<=args.revision<=999: parser.error('Use a retained revision between1 and999')
    run(args.revision)
