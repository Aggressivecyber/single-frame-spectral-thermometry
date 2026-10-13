"""Reconstruct actual four-sample gradients at saved update boundaries, without stepping."""
import runtime
import torch
from training_v4.common import read,write
from training_v4_1.common import setup,OUT as OLD,OLD_OUT
from training_v4_1.cache import load
from training_v4_1.model import new_model,protective_loss
from evaluation_v2.common import Journal
from training_v4_2.experiment import OUT,WEIGHTS,INITIAL


def main():
    setup(); selection=read(OLD_OUT/'backbone_selection.json')
    by_id={r['case_id']:r for r in read(OLD/'data_manifest.json')['rows']}
    draws=read(OLD/'sampling_sequence.json')['draws']
    records=[]
    for method in ('warm_safe1','identity_restart'):
        curve=Journal((OLD/'safe1' if method=='warm_safe1' else OUT)/'training_curve.jsonl').rows()
        for update in (0,100,300):
            if method=='identity_restart': path=WEIGHTS/('initial.pt' if not update else f'update_{update:04d}.pt')
            else: path=INITIAL if not update else runtime.ROOT/f'training_v4_1/checkpoints/safe1/update_{update:04d}.pt'
            model=new_model(selection); ck=torch.load(path,map_location='cpu',weights_only=False)
            model.load_cnn(ck['cnn_state_dict']); model.train(); params=tuple(model.parameters())
            total_main=torch.zeros(5797,device='cuda'); total_safe=torch.zeros_like(total_main)
            ordinary_main_safe=torch.zeros_like(total_main)
            cases=[]
            for case in draws[update]:
                row=by_id[case]; inputs,truth,base=load(row,selection)
                out=model(*inputs); loss,terms=protective_loss(model,inputs[0],out,truth,base,1.)
                ga=torch.autograd.grad(terms['main']/4,params,retain_graph=True,allow_unused=True)
                gb=torch.autograd.grad(terms['safe_total']/4,params,retain_graph=True,allow_unused=True)
                gc=torch.autograd.grad(loss/4,params,allow_unused=True)
                flatten=lambda gs:torch.cat([(g if g is not None else torch.zeros_like(p)).flatten() for p,g in zip(params,gs)])
                total_main+=flatten(ga); total_safe+=flatten(gb); ordinary_main_safe+=flatten(gc)
                cases.append({'case_id':case,'input_kind':row['input_kind'],'condition':row['condition'],
                              'raw_hash':row['raw_hash'],'cache_key':base['cache_key']})
            a,b=total_main,total_safe
            an,bn,cn=float(a.norm()),float(b.norm()),float(ordinary_main_safe.norm())
            expected=curve[update]['gradient_norm_before_clip']
            relative=abs(cn-expected)/max(expected,1e-12)
            # norm ordering can differ slightly from clip_grad_norm_'s per-tensor reduction.
            assert relative<1e-4,(method,update,cn,expected)
            assert torch.allclose(a+b,ordinary_main_safe,atol=2e-4,rtol=2e-4)
            records.append({'method':method,'checkpoint_update':update,'actual_gradient_update':update+1,'cases':cases,
                'checkpoint_sha256':runtime.sha256(path),'main_gradient_norm':an,'safe_gradient_norm':bn,
                'combined_gradient_norm':cn,'logged_combined_norm':expected,'norm_relative_difference':relative,
                'cosine':float(torch.dot(a,b)/(a.norm()*b.norm())) if an>0 and bn>0 else None,
                'clip_factor':min(1.,1./cn) if cn else 1.,'optimizer_steps':0})
            del model; torch.cuda.empty_cache()
    write(OUT/'actual_batch_gradients.json',{'records':records,'source_sha256':runtime.sha256(__file__),
        'actual_training_norms_reproduced':True,'optimizer_steps':0})
    print({'actual_batch_gradient_reconstruction':'passed','records':len(records)},flush=True)


if __name__=='__main__': main()
