"""Channel supervision, deterministic clean readouts and paired interventions."""
import runtime
import torch


def readouts(rad,images):
    batch,_,height,width = images.shape
    temperature,_,excess,_ = rad.fit(images.transpose(0,1).reshape(6,batch*height,width))
    return temperature.reshape(batch,height,width),excess.reshape(3,batch,height,width).transpose(0,1)


def training_loss(rad,pred,target,scale,pair_weight=1.0,thermal_only=False):
    means = target.mean((-2,-1),keepdim=True)
    reference,_=rad.curve(torch.full((1,1),3000.,device=pred.device,dtype=pred.dtype))
    fixed_scale=(reference/reference[3:].mean()).reshape(1,6,1,1)
    image = ((pred-target)/fixed_scale).square().mean()
    bias = ((pred.mean((-2,-1),keepdim=True)-means)/fixed_scale).square().mean()
    pt,pe = readouts(rad,pred*scale)
    with torch.no_grad():
        tt,te = readouts(rad,target*scale)
    temperature = ((pt-tt)/50).square().mean()
    pr,tr = pred[:,3:].clamp_min(1e-6).log(),target[:,3:].log()
    ratios = (((pr-pr[:,1:2])-(tr-tr[:,1:2]))/.02).square().mean()
    emission = ((pe-te)/(scale*.05)).square().mean()
    physical_error = (pred-target)*scale
    pair = ((physical_error[1::2]-physical_error[0::2])/(.025*(scale[1::2]+scale[0::2]))).square().mean()
    pair_readout = (((pe[1::2]-pe[0::2])-(te[1::2]-te[0::2]))/(scale[1::2]*.05)).square().mean()
    if thermal_only:
        loss = image+temperature+ratios+.5*bias
    else:
        loss = image+temperature+emission+pair_weight*(pair+pair_readout)+.5*bias+ratios
    return loss,{"channel":float(image.detach()),"temperature":float(temperature.detach()),"log_ratio":float(ratios.detach()),"clean_readout":float(emission.detach()),"pair":float(pair.detach()),"paired_readout":float(pair_readout.detach()),"mean":float(bias.detach())}
