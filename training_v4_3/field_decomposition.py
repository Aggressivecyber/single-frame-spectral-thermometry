"""Fixed continuum/signed state substitutions, with no optimizer or selection."""
import time
import runtime
import numpy as np
import torch
from training_v4.common import read,write,parameters_hash
from training_v4.model import physical_output
from training_v4_1.common import setup,OUT as OLD,OLD_OUT,rel
from training_v4_1.cache import load
from training_v4_1.evaluate import score_arrays
from training_v4_2.analyze import strata
from training_v4_3.residual_scaling import ResidualScaledCNN
from evaluation_v2.common import Journal,canonical_hash

OUT=runtime.ROOT/'reports/training_v4_3/run_001/field_decomposition'
ARMS=('residual_scaled','negative_balanced')
COMBOS=(('physical','physical'),('physical','CNN'),('CNN','physical'),('CNN','CNN'))


def run():
    setup(); OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'per_case.jsonl').exists(), 'Never overwrite diagnostic records'
    assert read(runtime.ROOT/'reports/training_v4_3/run_001/gradient_diagnostic/status.json')['stage']=='COMPLETE'
    plan=read(runtime.ROOT/'reports/local_session_20261013/field_decomposition_plan.json')
    ids=read(runtime.ROOT/'reports/training_v4_2/run_001/protocol.json')['probes']['stage_validation_ids']
    rows=read(OLD/'data_manifest.json')['rows']; by_id={r['case_id']:r for r in rows}
    probes=[by_id[i] for i in ids]
    assert len(probes)==144 and all(r['split']=='validation' for r in probes)
    protocol={**plan,'case_ids':ids,'case_ids_sha256':canonical_hash(ids),
        'source_sha256':runtime.sha256(__file__),'physics_source_sha256':runtime.sha256(runtime.ROOT/'physics.py'),
        'physical_output_source_sha256':runtime.sha256(runtime.ROOT/'training_v4/model.py'),'started':runtime.now()}
    write(OUT/'protocol.json',protocol)
    selection=read(OLD_OUT/'backbone_selection.json'); journal=Journal(OUT/'per_case.jsonl')
    started=time.monotonic(); checks=[]; endpoints=[]
    try:
        with torch.no_grad():
            for arm in ARMS:
                folder=runtime.ROOT/f'reports/training_v4_3/run_001/{arm}'
                assert read(folder/'status.json')['stage']=='COMPLETE'
                target={r['case_id']:r for r in Journal(folder/'validation_update_0500.jsonl').rows()}
                path=runtime.ROOT/f'training_v4_3/checkpoints/{arm}/update_0500.pt'
                ck=torch.load(path,map_location='cpu',weights_only=False)
                model=ResidualScaledCNN(ck['physics_config'],ck['coefficient_calibration']).cuda().eval()
                model.load_cnn(ck['cnn_state_dict']); before=parameters_hash(model)
                assert before==ck['parameters_sha256']
                buffers={k:v.clone() for k,v in model.named_buffers()}
                for index,row in enumerate(probes):
                    if time.monotonic()-started>plan['maximum_gpu_seconds']: raise TimeoutError('120-second diagnostic budget reached')
                    inputs,truth,base=load(row,selection); raw_e=model.physics.raw_to_electrons(inputs[0])
                    complete=model(*inputs); states={'physical':inputs[1],'CNN':complete['state']}
                    truths={k:v[0].cpu().numpy() for k,v in truth.items()}
                    results={}
                    for continuum,signed in COMBOS:
                        state=torch.cat([states[continuum][:,:2],states[signed][:,2:]],1)
                        pred=physical_output(model.physics,state,inputs[2],raw_e)
                        metrics=score_arrays(pred['template_e'][0].cpu().numpy(),
                            pred['temperature_color_K'][0].cpu().numpy(),truths)
                        rho=((pred['expected_e']-raw_e).square()/(raw_e.clamp_min(0)+model.physics.extra_variance)).mean().sqrt()
                        rec={k:row[k] for k in ('case_id','condition','geometry','input_kind','signed_aux','raw_hash','label_sha256')}
                        rec.update(arm=arm,continuum_source=continuum,signed_source=signed,
                            combination=continuum+'/'+signed,cache_key=base['cache_key'],metrics=metrics,
                            weighted_raw_RMS=float(rho),invalid_output_fraction=float((~pred['valid']).float().mean()),
                            minimum_expected_e=float(pred['expected_e'].min()),diagnostic_only=True,optimizer_steps=0)
                        journal.append(rec); results[(continuum,signed)]=pred
                    native=results[('CNN','CNN')]
                    native_metrics=score_arrays(native['template_e'][0].cpu().numpy(),native['temperature_color_K'][0].cpu().numpy(),truths)
                    metric_delta={k:abs(native_metrics[k]-target[row['case_id']]['metrics'][k])
                                  for k in ('template_RMSE_e','color_RMSE_K')}
                    forward_delta=max(float((native[k]-complete[k]).abs().max())
                                      for k in ('template_e','temperature_color_K','expected_e'))
                    assert max(metric_delta.values())<1e-5 and forward_delta<1e-5
                    color_block_delta=max(float((results[(c,'physical')]['temperature_color_K']-
                        results[(c,'CNN')]['temperature_color_K']).abs().max()) for c in ('physical','CNN'))
                    signed_block_delta=max(float((results[('physical',s)]['template_e']-
                        results[('CNN',s)]['template_e']).abs().max()) for s in ('physical','CNN'))
                    assert color_block_delta==0 and signed_block_delta==0
                    checks.append({'arm':arm,'case_id':row['case_id'],'native_metric_delta':metric_delta,
                        'native_forward_max_delta':forward_delta,'color_independent_of_signed_output_delta':color_block_delta,
                        'signed_independent_of_continuum_output_delta':signed_block_delta})
                    if (index+1)%48==0:
                        write(OUT/'status.json',{'stage':'RUNNING','arm':arm,'cases_in_arm':index+1,'time':runtime.now()})
                        print({'arm':arm,'fixed_cases':index+1},flush=True)
                assert before==parameters_hash(model) and all(torch.equal(v,buffers[k]) for k,v in model.named_buffers())
                endpoints.append({'arm':arm,'checkpoint':rel(path),'checkpoint_sha256':runtime.sha256(path),
                    'parameters_sha256_before_and_after':before,'physics_buffers_unchanged':True})
                del model
        write(OUT/'native_checks.json',checks)
        write(OUT/'computation_receipt.json',{'stage':'COMPUTE_COMPLETE','endpoint_frames':len(checks),
            'combination_records':len(journal.rows()),'optimizer_steps':0,'elapsed_s':time.monotonic()-started,
            'endpoints':endpoints,'source_sha256':runtime.sha256(__file__),'local_only':True,'final_test_used':False})
        write(OUT/'status.json',{'stage':'COMPUTE_COMPLETE','time':runtime.now()})
    except BaseException as exc:
        write(OUT/'failure_record.json',{'type':type(exc).__name__,'message':str(exc),
            'completed_combination_records':len(journal.rows()),'optimizer_steps':0,'failed_records_retained':True})
        write(OUT/'status.json',{'stage':'FAILED','time':runtime.now()})
        raise


def report():
    receipt=read(OUT/'computation_receipt.json'); assert receipt['endpoint_frames']==288
    records=Journal(OUT/'per_case.jsonl').rows(); assert len(records)==1152
    summary=[]
    for arm in ARMS:
        for continuum,signed in COMBOS:
            subset=[r for r in records if r['arm']==arm and r['continuum_source']==continuum and r['signed_source']==signed]
            for s in strata(subset):
                rs=[r for r in subset if r['condition']==s['condition'] and r['input_kind']==s['input_kind']]
                energy=sum(r['metrics']['truth_energy'] for r in rs)
                projection=sum((r['metrics']['signed_projection'] or 0)*r['metrics']['truth_energy'] for r in rs)/energy if energy else None
                summary.append({'arm':arm,'combination':continuum+'/'+signed,**s,
                    'energy_weighted_signed_projection':projection,
                    'pooled_weighted_raw_RMS':float(np.sqrt(np.mean([r['weighted_raw_RMS']**2 for r in rs]))),
                    'maximum_invalid_fraction':max(r['invalid_output_fraction'] for r in rs)})
    write(OUT/'summary.json',summary)
    lines=['# 固定字段分解：连续谱与有符号模板','',
        '对两组500次末点，严格使用既定144个开发诊断帧，每帧列出四种固定组合，共1152条记录；beta不变，optimizer更新0次。P=缓存固定物理，N=CNN实际最终输出；表中组合顺序为“连续谱字段/有符号模板字段”。truth只用于评分，没有用于选择组合。','',
        '这是开发子集上的机制诊断，不是新的模型选择，也不代表776帧全量验收。physics.constrain允许signed字段在[-20,20]范围，负值没有显式截成0；训练后的压零现象应单独解释。','',
        '代码中的色温输出只由连续谱字段计算，模板输出只由signed字段计算。本次所有288个端点帧验证了这两个输出层面的独立性（最大差0）；完整N/N与同一checkpoint原生forward及已保存逐例指标均复现。raw重投影与有效性仍依赖两类字段，因此不能仅凭一类truth误差挑选混合结果。','',
        '|权重|含噪条件|C/S组合|模板RMSE e|L2|色温RMSE K|模板投影|raw RMS|最大无效比例|',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for s in summary:
        if s['input_kind']!='noise': continue
        l2='N/A' if s['template_L2'] is None else f"{s['template_L2']:.5f}"
        projection='N/A' if s['energy_weighted_signed_projection'] is None else f"{s['energy_weighted_signed_projection']:.5f}"
        combo=s['combination'].replace('physical','P').replace('CNN','N')
        lines.append(f"|{s['arm']}|{s['condition']}|{combo}|{s['template_RMSE_e']:.4f}|{l2}|{s['color_RMSE_K']:.4f}|{projection}|{s['pooled_weighted_raw_RMS']:.5f}|{s['maximum_invalid_fraction']:.6f}|")
    lines+=['',f"GPU计算耗时{receipt['elapsed_s']:.2f}秒；288个完整N/N重放、模型参数哈希和物理缓冲区检查通过。连续、量化和含噪全部192个分层组合汇总在summary.json，完整逐例raw、投影、固定召回及有效性在per_case.jsonl。",'',
        '本分解定位的是当前输出字段贡献；校正过程中共享主干、物理残差特征和多步反馈的因果关系不由这项静态替换单独确定。没有创建新候选权重、按truth路由输入或打开最终测试。','']
    path=OUT/'FIELD_DECOMPOSITION_REPORT.md'; path.write_text('\n'.join(lines),encoding='utf-8')
    write(OUT/'receipt.json',{**receipt,'stage':'COMPLETE','report_sha256':runtime.sha256(path)})
    write(OUT/'status.json',{'stage':'COMPLETE','time':runtime.now(),'optimizer_steps':0})
    print({'field_decomposition':'COMPLETE','records':len(records),'optimizer_steps':0},flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument('--report',action='store_true')
    args=parser.parse_args(); report() if args.report else run()
