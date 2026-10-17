"""Native GPU check: full TV-RRU uninterrupted vs forced stop and exact resume."""
import runtime
import copy
import json
from pathlib import Path
import shutil
import torch
from research_joint_imaging.core import Imager,Radiometry
from research_joint_imaging_v2.pipeline import train_method

root=Path.cwd()
output=root/'reports/joint_imaging_v2_recovery_software_001'
output.mkdir(parents=True)
protocol=json.loads((root/'research_joint_imaging_v2/protocol.json').read_text())
protocol['learning']['phases']=[1,1,1]
protocol['learning']['validation_every']=1
runtime.write_json(output/'SOFTWARE_PROTOCOL.json',protocol)
torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
torch.backends.cudnn.benchmark=False
torch.cuda.set_per_process_memory_fraction(9500*1024**2/torch.cuda.get_device_properties(0).total_memory)
source=Path(protocol['recovery']['source_output'])
train={51001:torch.load(source/'cache/train_51001.pt',map_location='cpu',weights_only=False)}
val={61001:torch.load(source/'cache/validation_61001.pt',map_location='cpu',weights_only=False)}
op=Imager(protocol['model']).cuda();rad=Radiometry(protocol,'cuda')
for name in ('control','interrupted','resumed','legacy','replayed'):(output/name).mkdir()
train_method(protocol,op,rad,train,val,output/'control','cuda','full')

class SimulatedReboot(Exception):pass

original_save=torch.save
def interrupt_save(value,path,*args,**kwargs):
    original_save(value,path,*args,**kwargs)
    if isinstance(value,dict) and 'optimizer' in value and value['step']==2:
        Path(path).replace(output/'interrupted/full_latest.pt')
        raise SimulatedReboot('forced stop after atomic checkpoint step2')

torch.save=interrupt_save
try:
    train_method(protocol,op,rad,train,val,output/'interrupted','cuda','full')
except SimulatedReboot:
    pass
finally:
    torch.save=original_save
train_method(protocol,op,rad,train,val,output/'resumed','cuda','full',output/'interrupted')
for suffix in ('_best.pt','_training.jsonl'):
    shutil.copy2(output/('control/full'+suffix),output/('legacy/full'+suffix))
train_method(protocol,op,rad,train,val,output/'replayed','cuda','full',output/'legacy')
control=torch.load(output/'control/full_latest.pt',map_location='cpu',weights_only=False)
resumed=torch.load(output/'resumed/full_latest.pt',map_location='cpu',weights_only=False)
replayed=torch.load(output/'replayed/full_latest.pt',map_location='cpu',weights_only=False)
model_equal=all(torch.equal(value,resumed['model'][name]) for name,value in control['model'].items())
replay_equal=all(torch.equal(value,replayed['model'][name]) for name,value in control['model'].items())
optimizer_equal=all(torch.equal(value,resumed['optimizer']['state'][index][name]) if torch.is_tensor(value) else value==resumed['optimizer']['state'][index][name] for index,state in control['optimizer']['state'].items() for name,value in state.items())
numpy_equal=control['numpy_rng']==resumed['numpy_rng']
torch_equal=torch.equal(control['torch_rng'],resumed['torch_rng']) and all(torch.equal(a,b) for a,b in zip(control['cuda_rng'],resumed['cuda_rng']))
rows={name:[json.loads(line) for line in (output/name/'full_training.jsonl').read_text().splitlines()] for name in ('control','resumed','replayed')}
loss_equal=rows['control'][-1]['loss']==rows['resumed'][-1]['loss']==rows['replayed'][-1]['loss']
result={'time':runtime.now(),'GPU':torch.cuda.get_device_name(),'torch':torch.__version__,'full_TV_RRU_three_phases':True,'interruption_after_step':2,'resumed_from_step':2,'model_tensors_bitwise_equal':model_equal,'optimizer_state_bitwise_equal':optimizer_equal,'numpy_rng_equal':numpy_equal,'torch_and_cuda_rng_equal':torch_equal,'final_loss_equal':loss_equal,'legacy_prefix_replay_tensors_bitwise_equal':replay_equal,'final_loss':rows['control'][-1]['loss'],'formal_study_updates':0}
result['passed']=all((model_equal,optimizer_equal,numpy_equal,torch_equal,loss_equal,replay_equal))
runtime.write_json(output/'RESUME_TEST.json',result)
print(json.dumps(result),flush=True)
if not result['passed']:raise RuntimeError('Native optimizer/RNG resume test failed')
