"""Native full TV-RRU state restoration and numerically close continuation."""
import runtime
import json
from pathlib import Path
import torch
from research_joint_imaging.core import Imager,Radiometry
from research_joint_imaging_v2.model import TVRRU
import research_joint_imaging_v2.pipeline as pipeline

root=Path.cwd();output=root/'reports/joint_imaging_v2_recovery_software_002'
output.mkdir(parents=True)
protocol=json.loads((root/'research_joint_imaging_v2/protocol.json').read_text())
protocol['learning']['phases']=[1,1,2];protocol['learning']['validation_every']=1
runtime.write_json(output/'SOFTWARE_PROTOCOL.json',protocol)
torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.backends.cudnn.benchmark=False
torch.cuda.set_per_process_memory_fraction(9500*1024**2/torch.cuda.get_device_properties(0).total_memory)
source=Path(protocol['recovery']['source_output'])
train={51001:torch.load(source/'cache/train_51001.pt',map_location='cpu',weights_only=False)}
val={61001:torch.load(source/'cache/validation_61001.pt',map_location='cpu',weights_only=False)}
op=Imager(protocol['model']).cuda();rad=Radiometry(protocol,'cuda')
for name in ('control','interrupted','resumed'):(output/name).mkdir()
pipeline.train_method(protocol,op,rad,train,val,output/'control','cuda','full')

class SimulatedReboot(Exception):pass

original_save=torch.save
def interrupt_save(value,path,*args,**kwargs):
    original_save(value,path,*args,**kwargs)
    if isinstance(value,dict) and 'optimizer' in value and value['step']==2:
        Path(path).replace(output/'interrupted/full_latest.pt')
        raise SimulatedReboot('forced stop after atomic checkpoint step2')

torch.save=interrupt_save
try:
    pipeline.train_method(protocol,op,rad,train,val,output/'interrupted','cuda','full')
except SimulatedReboot:
    pass
finally:
    torch.save=original_save
saved=torch.load(output/'interrupted/full_latest.pt',map_location='cpu',weights_only=False)
audit={}
original_load=TVRRU.load_state_dict
def audit_model_load(self,state,*args,**kwargs):
    result=original_load(self,state,*args,**kwargs)
    audit['model_restored_bitwise']=all(torch.equal(value.cpu(),self.state_dict()[key].cpu()) for key,value in state.items())
    return result

original_optimizer_load=torch.optim.AdamW.load_state_dict
def audit_optimizer_load(self,state):
    result=original_optimizer_load(self,state)
    actual=self.state_dict()
    audit['optimizer_restored_bitwise']=state['param_groups']==actual['param_groups'] and all(torch.equal(value.cpu(),actual['state'][index][key].cpu()) if torch.is_tensor(value) else value==actual['state'][index][key] for index,entry in state['state'].items() for key,value in entry.items())
    return result

original_batch=pipeline.batch_from_pairs
def audit_batch(mothers,rng,device):
    if 'numpy_rng_restored' not in audit:
        audit['numpy_rng_restored']=rng.bit_generator.state==saved['numpy_rng']
        audit['torch_rng_restored']=torch.equal(torch.get_rng_state(),saved['torch_rng'])
        audit['cuda_rng_restored']=all(torch.equal(a,b) for a,b in zip(torch.cuda.get_rng_state_all(),saved['cuda_rng']))
    return original_batch(mothers,rng,device)

TVRRU.load_state_dict=audit_model_load
torch.optim.AdamW.load_state_dict=audit_optimizer_load
pipeline.batch_from_pairs=audit_batch
try:
    pipeline.train_method(protocol,op,rad,train,val,output/'resumed','cuda','full',output/'interrupted')
finally:
    TVRRU.load_state_dict=original_load
    torch.optim.AdamW.load_state_dict=original_optimizer_load
    pipeline.batch_from_pairs=original_batch
control=torch.load(output/'control/full_last.pt',map_location='cpu',weights_only=False)['model']
resumed=torch.load(output/'resumed/full_last.pt',map_location='cpu',weights_only=False)['model']
parameter_relative=float(torch.sqrt(sum((control[key].double()-resumed[key].double()).square().sum() for key in control)/sum(value.double().square().sum() for value in control.values())))
rows={name:[json.loads(line) for line in (output/name/'full_training.jsonl').read_text().splitlines()] for name in ('control','resumed')}
loss_relative=abs(rows['control'][-1]['loss']-rows['resumed'][-1]['loss'])/abs(rows['control'][-1]['loss'])
result={'time':runtime.now(),'GPU':torch.cuda.get_device_name(),'torch':torch.__version__,'interrupted_step':2,'completed_steps':4,'full_three_phase_solver_used':True,'state_restoration_checks':audit,'continuation_relative_parameter_difference':parameter_relative,'parameter_relative_tolerance':1e-6,'continuation_relative_final_loss_difference':loss_relative,'final_loss_relative_tolerance':1e-4,'bitwise_future_GPU_trajectory_claimed':False,'formal_study_updates':0}
result['passed']=all(audit.values()) and len(audit)==5 and parameter_relative<=1e-6 and loss_relative<=1e-4
runtime.write_json(output/'RESUME_TEST.json',result)
print(json.dumps(result),flush=True)
if not result['passed']:raise RuntimeError('Native full-state restoration test failed')
