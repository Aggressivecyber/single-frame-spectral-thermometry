"""Package saved v5 data, upload a draft GitHub release, then publish once.

Root-owned delivery only. No scientific reconstruction or training. Durable
per-command receipts prevent treating an incomplete upload as delivery.
If this script fails, inspect the receipts; do not rerun from the start.
"""
from research_v5 import common
common.configure_runtime()
from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess
import sys
import traceback
import zipfile

REPO='Aggressivecyber/single-frame-spectral-thermometry'
TAG='research-20261014-v5-decision'
R=common.REPORTS/'delivery_001'
B=common.ROOT/'outputs/research_v5/delivery_001'
GH=Path('C:/Program Files/GitHub CLI/gh.exe')
STATE={}


def state(**values):
    STATE.update(values,time=common.now())
    common.write_json(R/'STATE.json',STATE,replace=True)
    print(json.dumps({k:STATE[k] for k in ['time','status','current_asset'] if k in STATE}),flush=True)


def command(argv,label):
    record={'label':label,'argv':[str(x) for x in argv],'started_at':common.now()}
    result=subprocess.run([str(x) for x in argv],cwd=common.ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace',env={**os.environ,'GH_PROMPT_DISABLED':'1'})
    record.update(finished_at=common.now(),exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr)
    common.append_jsonl(R/'COMMAND_RECEIPTS.jsonl',record)
    if result.returncode:
        raise RuntimeError(f'{label} failed with exit code {result.returncode}; inspect COMMAND_RECEIPTS.jsonl; do not restart delivery blindly')
    return result.stdout.strip()


def main():
    p=argparse.ArgumentParser();p.add_argument('--target',required=True);args=p.parse_args()
    assert len(args.target)==40
    assert not (R/'STARTED.json').exists(), 'Delivery has already started; inspect existing receipts.'
    R.mkdir(parents=True,exist_ok=True);B.mkdir(parents=True,exist_ok=True)
    assert not list(B.iterdir()), 'Packaging destination must be new.'
    actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=common.ROOT,text=True).strip()
    assert actual==args.target
    assert subprocess.check_output(['git','branch','--show-current'],cwd=common.ROOT,text=True).strip()=='research/v5-decision-20261013'
    audit=common.read_json(common.REPORTS/'COMPLETION_AUDIT.json')
    assert audit['all_applicable_scientific_work_accounted_for'] and not audit['estimator_scientific_acceptance_passed']
    push=common.read_json(R/'GIT_PUSH_RECEIPT.json')
    assert push['exit_code']==0 and push['commit']==actual
    inputs=common.read_json(common.REPORTS/'root_review_001/DATA_FILE_INVENTORY.json')
    required=sum(v['bytes'] for v in inputs['files'])+1000000000
    assert shutil.disk_usage(B).free>required
    common.write_json(R/'STARTED.json',{'time':common.now(),'pid':os.getpid(),'target_commit':actual,'repository':REPO,'tag':TAG,'data_files':len(inputs['files']),'data_bytes':inputs['total_bytes'],'scientific_work_complete':True,'scientific_acceptance_passed':False,'source_script_sha256':common.sha(Path(__file__)),'data_inventory_sha256':common.sha(common.REPORTS/'root_review_001/DATA_FILE_INVENTORY.json'),'download_verification_requested':False})
    try:
        state(status='PACKAGING',target_commit=actual,pid=os.getpid(),repository=REPO,tag=TAG)
        groups=[];group=[];size=0
        for f in inputs['files']:
            if group and size+f['bytes']>1000000000:
                groups.append(group);group=[];size=0
            group.append(f);size+=f['bytes']
        if group:groups.append(group)
        archives=[]
        for i,group in enumerate(groups,1):
            path=B/f'v5_scientific_data_{i:03d}of{len(groups):03d}.zip'
            state(status='PACKAGING',current_asset=path.name)
            with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_STORED,allowZip64=True) as z:
                for f in group:
                    src=common.ROOT/f['path'];assert src.stat().st_size==f['bytes']
                    z.write(src,arcname=f['path'])
            archives.append({'name':path.name,'bytes':path.stat().st_size,'sha256':common.sha(path),'members':[f['path'] for f in group]})
        source=B/'v5_source_and_reports.zip'
        command(['git','archive','--format=zip','--output='+str(source),actual,'research_v5','reports/research_v5'],'archive_source_and_reports')
        archives.append({'name':source.name,'bytes':source.stat().st_size,'sha256':common.sha(source),'git_commit':actual,'scope':['research_v5','reports/research_v5']})
        manifest={'schema':'v5-release-package-manifest-v1','created_at':common.now(),'repository':REPO,'tag':TAG,'source_commit':actual,'data_files':len(inputs['files']),'data_bytes':inputs['total_bytes'],'archives':archives,'extraction':'Extract all independent ZIPs into the repository root; original relative paths retained. Scientific arrays are complete; runtime dependencies/caches and implementation fixtures excluded.','estimator_scientific_acceptance_passed':False,'training_updates':0,'final_evaluation_opened':False,'final_accuracy':None,'checksums_scope':'Locally generated ZIP SHA256 and scientific member inventory; no remote redownload required.'}
        common.write_json(B/'V5_PACKAGE_MANIFEST.json',manifest)
        common.write_json(R/'PACKAGE_MANIFEST.json',manifest)
        checksums=''.join(f"{a['sha256']}  {a['name']}\n" for a in archives)
        (B/'SHA256SUMS.txt').write_text(checksums,encoding='utf-8')
        notes=R/'RELEASE_NOTES.md'
        notes.write_text(f'''# v5 决策性研究：物理负结果与完整科学证据

已执行任务书全部适用的预登记诊断；当前 v5 两步物理估计未通过，按严格停止规则关闭 CNN、候选和最终验证支线。

- Oracle：180输入、720四级结果、84×4基线；无噪O0正常±RMSE0.005612e。
- 开发：新36母场景、540输入、4320网格行；独立null96+96、MC12×8、Hessian8探针、压力36×5全部记录。
- 名义色温RMSE28.540K（目标20K）；正常正负模板RMSE64.629/63.761e，对照17.195/17.223e；两符号投影也失败。
- 色温MC覆盖89.729%未达90%；Hessian0/8收敛；增益/发射率失配造成大误差但当前零拒绝。weak阈值交叉不能作为正确弱结构检出。
- 原float32六项失败、高精度补充、原作业超时与单独预算补充、三例MC重复日志缺失均保留。
- CNN更新0，无v5 checkpoint；64原始最终与36signed确认母场景继续封存，最终性能未知。

[一页六问题决策](https://github.com/{REPO}/blob/research/v5-decision-20261013/reports/research_v5/run_001/V5_SCIENTIFIC_DECISION.md) · [完整报告](https://github.com/{REPO}/blob/research/v5-decision-20261013/reports/research_v5/run_001/ROOT_FINAL_REPORT.md) · [入口与完成审计](https://github.com/{REPO}/tree/research/v5-decision-20261013/research_v5/delivery)

源码与研究报告提交：`{actual}`。原平台和冻结实现保持不变。数据分卷含全部{len(inputs['files'])}个实际v5科学载荷文件，总原始文件大小{inputs['total_bytes']}bytes；各ZIP可独立解压至仓库根目录，路径原样保留。`V5_PACKAGE_MANIFEST.json`列明每卷内容和哈希，`SHA256SUMS.txt`提供本地生成包哈希。

这是合成平台上的模糊后连续谱色温与signed模板偏离研究；不证明真实气体温度、实测AlO浓度、可靠部署或10kHz性能。上传以成功命令回执为准，不进行重复下载核对。
''',encoding='utf-8')
        state(status='CREATE_DRAFT',current_asset=None)
        url=command([GH,'release','create',TAG,'--repo',REPO,'--target',actual,'--title','v5 decision study: complete negative-result evidence','--notes-file',notes,'--draft','--latest=false'],'create_draft_release')
        state(status='UPLOADING',release_url=url)
        assets=[B/a['name'] for a in archives]+[B/'V5_PACKAGE_MANIFEST.json',B/'SHA256SUMS.txt',common.REPORTS/'V5_SCIENTIFIC_DECISION.md',common.REPORTS/'ROOT_FINAL_REPORT.md',common.REPORTS/'root_review_001/V5_DECISION_EVIDENCE.png']
        archive_hashes={a['name']:a['sha256'] for a in archives}
        for i,path in enumerate(assets,1):
            state(status='UPLOADING',current_asset=path.name,asset_index=i,asset_count=len(assets))
            command([GH,'release','upload',TAG,path,'--repo',REPO],'upload_'+path.name)
            common.append_jsonl(R/'UPLOAD_RECEIPTS.jsonl',{'time':common.now(),'asset_name':path.name,'bytes':path.stat().st_size,'sha256':archive_hashes.get(path.name) or common.sha(path),'exit_code':0,'uploaded':True})
        state(status='PUBLISHING',current_asset=None)
        command([GH,'release','edit',TAG,'--repo',REPO,'--draft=false','--latest=false'],'publish_release')
        finish={'schema':'v5-delivery-completion-v1','status':'COMPLETED_GITHUB_RELEASE','time':common.now(),'repository':REPO,'tag':TAG,'release_url':f'https://github.com/{REPO}/releases/tag/{TAG}','source_commit':actual,'git_push_succeeded':True,'all_assets_uploaded':True,'published':True,'asset_count':len(assets),'data_files':len(inputs['files']),'data_bytes':inputs['total_bytes'],'package_manifest':{'path':common.relative(R/'PACKAGE_MANIFEST.json'),'sha256':common.sha(R/'PACKAGE_MANIFEST.json')},'no_remote_redownload_performed':True,'no_new_scientific_execution':True,'root_final_receipt_commit_pending':True}
        common.write_json(R/'END.json',finish)
        state(status='COMPLETED_GITHUB_RELEASE',current_asset=None)
    except BaseException as e:
        failure={'status':'FAILED_DELIVERY','time':common.now(),'exception':type(e).__name__,'message':str(e),'traceback':traceback.format_exc(),'state':STATE,'already_completed_uploads_preserved':True,'instruction':'Inspect receipts and continue only missing delivery; no scientific phase rerun and no blind delivery restart.'}
        common.write_json(R/'FAILED.json',failure)
        state(status='FAILED_DELIVERY')
        raise


if __name__=='__main__':main()
