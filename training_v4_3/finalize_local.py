"""Close the bounded local study with byte/hash, checkpoint and record receipts."""
import ast
import hashlib
import subprocess
import xml.etree.ElementTree as ET
import runtime
import torch
from training_v4.common import read,write
from evaluation_v2.common import Journal,tensor_digest,canonical_hash

ROOT=runtime.ROOT
OUT=ROOT/'reports/local_session_20261013'
RUNS=ROOT/'reports/training_v4_3/run_001'
BASE='cca766c65814f89d6f8228f3e511974fa603455e'
ARMS=('fresh_trunk','residual_scaled','negative_balanced')
SOURCES={'fresh_trunk':'fresh_trunk.py','residual_scaled':'scaled_train.py','negative_balanced':'balanced_train.py'}


def main():
    assert ROOT.drive.upper() in ('E:','D:')
    summaries=[]; checkpoints=[]; all_used=set(); total_frames=0
    by_id={r['case_id']:r for r in read(ROOT/'reports/training_v4_1/run_001/data_manifest.json')['rows']}
    for arm in ARMS:
        folder=RUNS/arm
        receipt=read(folder/'training_receipt.json')
        assert receipt['updates']==500 and receipt['microbatches']==2000 and not receipt['final_test_used']
        assert read(folder/'status.json')['stage']=='COMPLETE'
        assert receipt['source_sha256']==runtime.sha256(ROOT/'training_v4_3'/SOURCES[arm])
        train=Journal(folder/'training_curve.jsonl').rows()
        assert len(train)==500 and all(r['optimizer_update']==i for i,r in enumerate(train,1))
        all_used.update(s['case_id'] for r in train for s in r['samples'])
        for cp in receipt['checkpoints']:
            path=ROOT/cp['path']; assert runtime.sha256(path)==cp['sha256']
            obj=torch.load(path,map_location='cpu',weights_only=False)
            assert sum(t.numel() for t in obj['cnn_state_dict'].values())==5797
            assert tensor_digest(obj['cnn_state_dict'].items())==obj['parameters_sha256']
            steps=sorted({int(s['step']) for s in obj['optimizer_state']['state'].values()})
            assert steps==([] if cp['update']==0 else [cp['update']])
            assert obj['optimizer_updates']==cp['update']
            if 'model_source_sha256' in obj:
                assert obj['model_source_sha256']==runtime.sha256(ROOT/'training_v4_3/residual_scaling.py')
            checkpoints.append({'arm':arm,'update':cp['update'],'path':cp['path'],'sha256':cp['sha256'],
                'Adam_steps':steps,'parameters_sha256':obj['parameters_sha256'],'CPU_reopened':True})
        assert len(receipt['native_replays'])==3
        assert all(max(r['differences'].values())<1e-5 for r in receipt['native_replays'])
        for update in range(0,501,100):
            rows=Journal(folder/f'validation_update_{update:04d}.jsonl').rows()
            assert len(rows)==776 and len({r['case_id'] for r in rows})==776
            assert all(r['status']=='SUCCESS' and r['invalid_output_fraction']==0 for r in rows)
            all_used.update(r['case_id'] for r in rows); total_frames+=len(rows)
            decision=read(folder/f'validation_summary_{update:04d}.json')
            assert len(decision['protection_groups'])==96 and decision['final_test_used'] is False
            summaries.append({'arm':arm,'update':update,'decision':decision})
    assert len(checkpoints)==18 and total_frames==13968
    original=Journal(RUNS/'residual_scaled/training_curve.jsonl').rows()
    balanced=Journal(RUNS/'negative_balanced/training_curve.jsonl').rows()
    assert all(a['samples'][j]==b['samples'][j] for a,b in zip(original,balanced) for j in (0,2,3))
    assert read(RUNS/'residual_scaled/validation_summary_0000.json')==read(RUNS/'negative_balanced/validation_summary_0000.json')
    gradients=read(RUNS/'gradient_diagnostic/receipt.json'); fields=read(RUNS/'field_decomposition/receipt.json')
    assert gradients['stage']==fields['stage']=='COMPLETE' and gradients['optimizer_steps']==fields['optimizer_steps']==0
    assert gradients['cases']==96 and fields['endpoint_frames']==288 and fields['combination_records']==1152
    assert fields['elapsed_s']<=120
    all_used.update(read(RUNS/'gradient_diagnostic/protocol.json')['case_ids'])
    all_used.update(read(ROOT/'reports/training_v4_2/run_001/protocol.json')['probes']['gradient_train_ids'])
    tests=ET.parse(RUNS/'residual_scaled/tests.xml').getroot().find('testsuite')
    assert tests is not None and tests.attrib['tests']=='3' and tests.attrib['failures']==tests.attrib['errors']=='0'

    # Older working copies have pre-existing CRLF despite Git eol=lf attributes.
    # Their frozen pre-training byte hashes are the correct preservation baseline.
    # The 80 v4.2 additions are checked against exact published blob bytes.
    tree=subprocess.check_output(['git','ls-tree','-rz','--full-tree',BASE],cwd=ROOT)
    old_working=read(ROOT/'reports/training_v4_2/run_001/preserved_files.json')
    preserved=[]
    for entry in tree.split(b'\0'):
        if not entry: continue
        attrs,name=entry.split(b'\t',1); mode,kind,expected=attrs.split()
        assert kind==b'blob' and mode in (b'100644',b'100755')
        path=ROOT/name.decode('utf-8'); size=path.stat().st_size
        sha1=hashlib.sha1(b'blob '+str(size).encode()+b'\0'); sha256=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''): sha1.update(block); sha256.update(block)
        if name.decode() in old_working:
            assert sha256.hexdigest()==old_working[name.decode()],str(path)
            basis='frozen_pre_training_working_file_SHA256'
        else:
            assert sha1.hexdigest()==expected.decode(),str(path)
            basis='exact_published_Git_blob_bytes'
        preserved.append({'path':name.decode(),'git_blob':expected.decode(),'sha256':sha256.hexdigest(),
                          'byte_comparison_basis':basis})
    write(OUT/'preserved_public_tree.json',{'base':BASE,'files':preserved,'bytewise_unchanged':True,
        'older_files_using_frozen_working_SHA256':len(old_working),
        'newer_files_using_published_blob_bytes':len(preserved)-len(old_working),
        'line_ending_changes_made':False})

    # Recheck every raw/label/cache used by these runs and the new training probes.
    old={r['case_id']:r for r in read(ROOT/'reports/training_v4_2/run_001/input_cache_hashes.json')}
    checked={}; inputs=[]
    for index,case in enumerate(sorted(all_used)):
        row=by_id[case]
        score_path=ROOT/'outputs/training_v4_1/run_001/physical_scores'/(case+'.json')
        score=read(score_path)
        assert score['raw_hash']==row['raw_hash'] and score['label_sha256']==row['label_sha256']
        cache_meta=str((ROOT/score['cache_path']).with_suffix('.json').relative_to(ROOT)).replace('\\','/')
        files={row['raw_path']:row['raw_file_sha256'],row['label_path']:row['label_sha256'],
            score['cache_path']:score['cache_sha256'],cache_meta:score['cache_metadata_sha256']}
        for path,expected in files.items():
            if path not in checked: checked[path]=runtime.sha256(ROOT/path)
            assert checked[path]==expected,path
        score_hash=runtime.sha256(score_path)
        if case in old:
            assert files==old[case]['files'] and score_hash==old[case]['physical_score_sha256']
        inputs.append({'case_id':case,'raw_hash':row['raw_hash'],'cache_key':score['cache_key'],
            'files':files,'score_sha256':score_hash,'previously_verified_v4_2':case in old})
        if (index+1)%500==0: print({'input_cases_verified':index+1,'total':len(all_used)},flush=True)
    write(OUT/'input_hash_receipt.json',{'used_cases':len(inputs),'unique_input_files':len(checked),
        'old_verified_cases_reused':sum(r['previously_verified_v4_2'] for r in inputs),'cases':inputs})
    sources=[]
    for path in sorted((ROOT/'training_v4_3').glob('*.py')):
        ast.parse(path.read_text(encoding='utf-8'))
        sources.append({'path':str(path.relative_to(ROOT)).replace('\\','/'),'sha256':runtime.sha256(path)})
    write(OUT/'source_receipt.json',{'python_sources':sources,'all_parsed':True})
    write(OUT/'checkpoint_receipt.json',{'checkpoints':checkpoints,'native_replays_in_training_receipts':9})
    final_report(summaries,len(preserved),len(inputs),len(checked))
    value={'status':'closed_not_qualified','active_local_training_jobs':0,'automatic_training_extension':False,
        'training_arms':list(ARMS),'actual_optimizer_updates':1500,'microbatches':6000,
        'checkpoints':18,'validation_records':13968,'native_replays':9,'same_slot_comparisons':1500,
        'zero_training_scaled_balanced_exactly_equal':True,'gradient_diagnostic_cases':96,
        'field_combinations':1152,'field_native_replays':288,'CUDA_tests_passed':3,
        'original_public_files_bytewise_unchanged':len(preserved),'used_input_cases_rehashed':len(inputs),
        'unique_input_files_rehashed':len(checked),'qualified_checkpoint_count':sum(s['decision']['eligible'] for s in summaries),
        'final_test_used':False,'local_only':True,'new_public_push':False,
        'completed':runtime.now(),'base_commit':BASE,'report_sha256':runtime.sha256(OUT/'LOCAL_STUDY_REPORT.md')}
    write(OUT/'final_receipt.json',value)
    print(value,flush=True)


def final_report(summaries,preserved,inputs,files):
    lines=['# 2026-10-13 测温有限研究结果','',
        '三组本地对照各完成500次更新，18个checkpoint均没有通过联合验收。训练已收口；新增工作保存在E盘本地分支，最终测试仍锁定。','',
        '本研究始终采用同一视场、各通道在同一传感器区域重叠后的算法分离。当前网络是固定物理反演后的5797参数CNN校正器，不是反传LM/PCG的完整展开网络。','',
        '|500次末点|原检查/16|保护/96|主弱L2|主正常RMSE e|含噪温度通过|联合合格|',
        '|---|---:|---:|---:|---:|---|---|']
    for s in summaries:
        if s['update']!=500: continue
        d=s['decision']; lines.append(f"|{s['arm']}|{d['original_gates_pass_count']}|{sum(g['passed'] for g in d['protection_groups'])}|{d['weak_L2']:.6f}|{d['normal_RMSE_e']:.4f}|{d['noisy_temperature_pass']}|{d['eligible']}|")
    lines+=['','## 有效发现与失败边界','',
        'fresh_trunk仅重置主干，保持零残差头及原训练设置。其输出更保守，但主弱L2=1.536909，没有得到足够降噪；一个种子500次不能证明更长训练无效。','',
        'residual_scaled只引入由固定物理raw残差决定的冻结系数；tau=0.3124117966、power=9在训练输入上定标。它在每个checkpoint都保持16/16与96/96，并保留主弱信号降噪，但末点含噪负组色温23.480891K仍超过15.9479K门槛。设计受已查看开发验证的残差间隔启发，不能称独立验证或真实相机保证。','',
        'negative_balanced只改变每批第二个含噪抽样：已有12个训练母体的24条含噪负模板反复暴露，三个其他位置共1500项与控制逐例相同；新增训练数据0。末点负组色温降至16.6774K，仍未通过；主正常模板误差由9.5767e升至13.6784e。不能用局部改善替代同一权重的联合验收。','',
        '|500次同权重有符号含噪指标|residual_scaled|negative_balanced|',
        '|---|---:|---:|',
        '|负正常模板投影|0.042094|0.457063|',
        '|负正常模板固定−60e召回率|0.000000|0.237787|',
        '|负弱模板L2（6帧）|0.965024|1.316166|',
        '|正弱模板L2（6帧）|0.691125|1.339713|','',
        '投影改善意味着部分负结构重新出现；弱组误差和低召回率表明恢复仍不完整。上述各列均来自该列同一个500次权重，没有拼接不同checkpoint的优势。','',
        '## 零更新机制诊断','',
        '96个训练探针的梯度诊断完成，模型参数与物理缓冲区不变。主弱与负正常的总梯度平均余弦在残差权重/平衡抽样两臂分别约−0.007/−0.107，并非普遍强烈对抗；平衡抽样末点负正常与辅助正弱组约−0.79，显示局部关系随组别变化。这些是clip和Adam之前的梯度，不是唯一根因证明。','',
        '字段分解严格沿用既定144开发帧，对两个末点列出四种连续谱/signed组合，共1152条记录、288次完整N/N原生复现，GPU耗时27.734秒，optimizer更新0次。色温由连续谱字段决定，模板由signed字段决定，这一输出层面的独立性逐帧验证差0。signed约束允许[-20,20]；负输出趋零属于学习结果，代码没有显式非负截断。','',
        '以residual_scaled的6帧含噪负正常模板为例，物理结果的色温/模板RMSE为15.7248K/19.9208e，CNN完整结果为29.5715K/51.9962e；raw RMS却只由0.9793变为1.0221。只保留CNN连续谱或signed时，raw RMS分别为1.3890和1.2958。这个固定分解与两类错误校正相互补偿的解释一致；raw拟合接近噪声量级没有保证正确分离。混合结果没有被选为新模型，子集也没有代替776帧验收。','',
        '## 保存与校验','',
        f'本地新增三臂共1500次optimizer更新、6000个微批次、18个checkpoint、13968条验证记录，9个训练回执中的原生预测复现与18个Adam步数/参数哈希重开均通过。{preserved}个原文件保留核对通过：770个旧工作副本按已冻结原始SHA256比较，80个v4.2新增文件按公开Git blob原始字节比较。旧工作副本既存换行未修改。{inputs}个已使用输入及{files}个去重raw/标签/缓存文件已核对冻结哈希。3个CUDA缩放机制检查通过。',
        '', '梯度计算后绘图曾因GPU解释器缺少matplotlib失败；完整96条梯度与不变性断言已完成并保存，失败记录保留，随后在既有CPU环境生成报告，没有重跑GPU计算。四张最终图均已重新打开检查。',
        '', '09:39:49已核验的v4.2公开提交为 `cca766c65814f89d6f8228f3e511974fa603455e`。此后本地v4.3结果未推送，不改公开默认模型。全部数据、权重、日志与临时产物在E/D盘；C盘路径只读取既有解释器。','',
        '## 下阶段决策入口','',
        '本轮停止追加训练。下阶段需要先审查连续谱校正与signed修正的补偿关系，再决定是否登记一个固定连续谱输出的有限对照；它尚未实现或训练。本轮的小signed训练池重复暴露不能替代更多独立母场景或独立采集验证，最终测试继续锁定。','',
        '详细结果：','',
        '- [全新主干](../training_v4_3/run_001/fresh_trunk/TRAIN_V4_3_LOCAL_REPORT.md)',
        '- [冻结残差系数](../training_v4_3/run_001/residual_scaled/TRAIN_V4_3_RESIDUAL_SCALED_REPORT.md)',
        '- [负模板抽样对照](../training_v4_3/run_001/negative_balanced/TRAIN_V4_3_NEGATIVE_BALANCED_REPORT.md)',
        '- [训练梯度](../training_v4_3/run_001/gradient_diagnostic/GRADIENT_DIAGNOSTIC_REPORT.md)',
        '- [固定字段分解](../training_v4_3/run_001/field_decomposition/FIELD_DECOMPOSITION_REPORT.md)',
        '- [完整回执](final_receipt.json)','']
    (OUT/'LOCAL_STUDY_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__': main()
