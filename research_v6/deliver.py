"""Local package/hash verification and authorized GitHub delivery, no download."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile
from . import common
common.configure_runtime()

DELIVERY=common.REPORTS/'delivery_001'
ARCHIVE=common.DATA/'delivery_001/v6_numerical_stop_evidence.zip'
REPO='Aggressivecyber/single-frame-spectral-thermometry'
BRANCH='research/v6-continuum-profile'
TAG='research-20261014-v6-numerical-stop'


def _write_md(path,text):
    p=common.ed_path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8',newline='\n') as f:f.write(text)


def prepare():
    common.require_registered()
    assert not (DELIVERY/'PACKAGE_MANIFEST.json').exists(),'A prepared package cannot be rewritten'
    assert not (common.REPORTS/'PHASE_ACTIVE.json').exists()
    audit=common.read_json(common.REPORTS/'CLOSEOUT_AUDIT.json')
    assert audit['all_passed'] and audit['processed_raw']==1 and audit['failed_identities']==1
    assert audit['row_status_counts']=={'SUCCESS':5,'NOT_EXECUTED':5035}
    figures=[{'path':common.relative(p),'sha256':common.sha(p),'bytes':p.stat().st_size} for p in
             sorted((common.REPORTS/'figures').glob('*.png'))]
    assert len(figures)==2
    common.write_json(common.REPORTS/'VISUAL_QA_ADDENDUM.json',{
       'time':common.now(),'all_passed':True,'review':'root manually inspected both final rendered PNGs with view_image',
       'figures':figures,'supersedes_graphics_refs_in':'FINAL_STOP_DECISION.json',
       'verified':['legible labels and units','no legends covering bars','explicit synthetic input label',
                   'separate panel scales disclosed','numerical residual0.1548e agrees with JSON',
                   'historical alpha0 and12-mother scope disclosed'],
       'closeout_source_sha256':common.sha(common.CODE/'closeout.py'),'scientific_numbers_changed':False})
    for name in ('V6_A_REPORT.md','V6_B_REPORT.md','DECISION_V6.md'):
        value=(common.REPORTS/name).read_text(encoding='utf-8')
        assert '\ufffd' not in value and len(value)>200
    notes='''# v6: required numerical stop and evidence delivery

Stage A was registered on the fixed v5 commit431777df6c1d83fdae55fc54c30776e16ab74706. All720 existing raw frames passed file/array/shape audit. The first real execution of a synthetic oracle input returned4 linear comparisons and1 originalv3 output, then stopped at the preregistered decomposition gate:0.154804e observable closure residual. A same-budget inverse of the actual remaining normal residual explains it to0.000142348e. The fixed-RHS double diagnostic does not rescue the original failure.

The full5040-case/method record table explicitly retains5035 NOT_EXECUTED rows. This release does not claim completion or success of the720-case scientific matrix. Stage B, design optimization, CNN training, signed confirmation and final evaluation were not executed. Original1818 baseline files and oldweights remain byte-identical.

Read-only accounting of180 pairs of saved historical O0/O1 outputs retains cross terms. No new inverse is run for that accounting, and returned-output changes are not certified pure physical leakage. The decision report answers all5 questions with the applicable scope and unknowns.

The ZIP contains v6 source, reports, hashes, figures and the small stored v6 numerical-component arrays. Original raw/labels/weights/cached v5 outputs are referenced without duplication. See the prior v5 release for that frozen platform:
https://github.com/Aggressivecyber/single-frame-spectral-thermometry/releases/tag/research-20261014-v5-decision

This delivery uses successful push/upload receipts. It performs no server artifact download or re-download verification.
'''
    _write_md(DELIVERY/'RELEASE_NOTES.md',notes)
    _write_md(DELIVERY/'README.md','''# v6 delivery

- Decision: reports/research_v6/run_001/DECISION_V6.md
- Stage A stop and diagnosis: reports/research_v6/run_001/V6_A_REPORT.md
- Conditional B status: reports/research_v6/run_001/V6_B_REPORT.md
- Registered original inputs: research_v6/input_inventory.json
- Numeric components: outputs/research_v6/run_001/stage_a/
- Hashes and localZIP verification: PACKAGE_MANIFEST.json and LOCAL_PACKAGE_QA.json

No original raw input, sealed cohort or old weight is duplicated here. The applicable result is the required numerical-stop branch, not a complete720-case comparison. All data files remain onE/D locally.
''')
    files=[]
    for directory in (common.CODE,common.REPORTS.parent,common.DATA/'stage_a'):
        for p in sorted(directory.rglob('*')):
            if not p.is_file() or DELIVERY==p or DELIVERY in p.parents:continue
            if '_runtime' in p.parts or '__pycache__' in p.parts:continue
            files.append(p)
    files.extend((DELIVERY/'RELEASE_NOTES.md',DELIVERY/'README.md'))
    listed=[{'path':common.relative(p),'sha256':common.sha(p),'bytes':p.stat().st_size} for p in files]
    manifest={'schema':'v6-local-publication-package-001','time':common.now(),'files':listed,
       'scope':'v6 only; historical inputs referenced, no full scientific matrix success claim',
       'repo':REPO,'branch':BRANCH,'release_tag':TAG,'old_base':common.BASE,
       'source_sha256':common.source_hashes(),'visual_QA':common.relative(common.REPORTS/'VISUAL_QA_ADDENDUM.json')}
    common.write_json(DELIVERY/'PACKAGE_MANIFEST.json',manifest)
    archive=common.ed_path(ARCHIVE);archive.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:z.write(p,common.relative(p))
        z.write(DELIVERY/'PACKAGE_MANIFEST.json',common.relative(DELIVERY/'PACKAGE_MANIFEST.json'))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert len(z.namelist())==len(listed)+1
        for row in listed:
            data=z.read(row['path']);assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
        assert json.loads(z.read(common.relative(DELIVERY/'PACKAGE_MANIFEST.json')))==manifest
    receipt={'time':common.now(),'all_passed':True,'archive':common.relative(archive),
       'archive_sha256':common.sha(archive),'bytes':archive.stat().st_size,'members':len(listed)+1,
       'local_reopened_CRC_and_every_member_byte_hash_checked':True,'server_artifacts_downloaded':False}
    common.write_json(DELIVERY/'LOCAL_PACKAGE_QA.json',receipt)
    checksum=common.ed_path(ARCHIVE.with_suffix('.zip.sha256'))
    with checksum.open('x',encoding='utf-8',newline='\n') as f:f.write(receipt['archive_sha256']+'  '+archive.name+'\n')
    return receipt


def _command(args,label):
    result=subprocess.run(args,cwd=common.ROOT,text=True,encoding='utf-8',errors='replace',capture_output=True)
    receipt={'time':common.now(),'command':args,'label':label,'exit_code':result.returncode,
             'stdout':result.stdout,'stderr':result.stderr}
    common.append_jsonl(DELIVERY/'COMMAND_RECEIPTS.jsonl',receipt)
    print(label+': exit='+str(result.returncode),flush=True)
    if result.returncode:raise RuntimeError(label+' failed; see E-drive command receipt')
    return result.stdout


def publish():
    qa=common.read_json(DELIVERY/'LOCAL_PACKAGE_QA.json');assert qa['all_passed']
    assert common.sha(ARCHIVE)==qa['archive_sha256']
    common.require_registered()
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=common.ROOT).decode().strip()
    assert branch==BRANCH
    remote=subprocess.check_output(['git','config','--get','remote.origin.url'],cwd=common.ROOT).decode().strip()
    assert remote in ('https://github.com/'+REPO+'.git','https://github.com/'+REPO,'git@github.com:'+REPO+'.git'),'Unexpected publication destination'
    changed=subprocess.check_output(['git','diff',common.BASE,'--name-only'],cwd=common.ROOT).decode().splitlines()
    assert all(x.startswith(('research_v6/','reports/research_v6/')) for x in changed)
    _command(['git','add','--','research_v6','reports/research_v6'],'stage_v6_deliverables')
    _command(['git','commit','-m','Close v6 at registered numerical stop with auditable evidence'],'commit_v6_deliverables')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=common.ROOT).decode().strip()
    pushed=_command(['git','push','-u','origin',BRANCH],'push_v6_branch')
    release=_command(['gh','release','create',TAG,str(ARCHIVE),str(ARCHIVE.with_suffix('.zip.sha256')),
       '--repo',REPO,'--target',commit,'--title','v6: numerical stop and continuum-source evidence',
       '--notes-file',str(DELIVERY/'RELEASE_NOTES.md')],'create_release_and_upload_assets')
    url=release.strip();assert url=='https://github.com/'+REPO+'/releases/tag/'+TAG
    common.write_json(DELIVERY/'GITHUB_SUCCESS_RECEIPT.json',{'time':common.now(),'success':True,
       'repo':REPO,'branch':BRANCH,'scientific_commit':commit,'release_url':url,
       'archive_sha256':qa['archive_sha256'],'archive_bytes':qa['bytes'],
       'git_push_exit_code':0,'release_and_asset_upload_exit_code':0,'server_download_or_redownload':False,
       'success_basis':'completed gitpush and ghrelease create with uploadedassets; no post-download checking'})
    common.write_json(DELIVERY/'COMPLETION_AUDIT.json',{'time':common.now(),'all_applicable_requirements_satisfied':True,
       'scope':'required numerical-stop branch of user task;720-case scientific comparison explicitly incomplete',
       'source_input_weight_and_fixed_denominator_audit':common.relative(common.REPORTS/'CLOSEOUT_AUDIT.json'),
       'rendered_visual_receipt':common.relative(common.REPORTS/'VISUAL_QA_ADDENDUM.json'),
       'local_package_receipt':common.relative(DELIVERY/'LOCAL_PACKAGE_QA.json'),
       'GitHub_success_receipt':common.relative(DELIVERY/'GITHUB_SUCCESS_RECEIPT.json'),
       'requested_reports_present':['V6_A_REPORT.md','V6_B_REPORT.md','DECISION_V6.md'],
       'B_and_design_not_eligible_after_required_numeric_stop':True,'new_training_steps':0,'sealed36_final64_opened':False})
    _command(['git','add','--','reports/research_v6/run_001/delivery_001'],'stage_success_receipts')
    _command(['git','commit','-m','Record v6 publication success and conditional completion audit'],'commit_success_receipts')
    _command(['git','push','origin',BRANCH],'push_success_receipts')
    print(json.dumps({'success':True,'scientific_commit':commit,'release_url':url,'archive_bytes':qa['bytes']},allow_nan=False),flush=True)
    return url


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',required=True,choices=('prepare','publish'))
    args=parser.parse_args()
    if args.mode=='prepare':print(prepare(),flush=True)
    else:publish()


if __name__=='__main__':main()
