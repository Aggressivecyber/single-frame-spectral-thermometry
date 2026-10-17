"""TV-RRU: role-separated NAF-style priors, shared across three stages."""
import runtime
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from research_joint_imaging.core import gradient
from research_joint_imaging_v2.solver import anchored_tv_prox

MODEL_KEYS=("anchor","d","u","initial","y","weight","noise")


def thermal_coordinates(images):
    log_i = images.clamp_min(1e-6).log()
    return torch.cat((log_i.mean(1,keepdim=True), log_i[:,0:1]-log_i[:,1:2], log_i[:,2:3]-log_i[:,1:2]),1)


def coordinates_to_logs(q):
    b,r1,r2 = q.chunk(3,1)
    return torch.cat((b+(2*r1-r2)/3, b-(r1+r2)/3, b+(-r1+2*r2)/3),1)


def sideband_residual(op,residual):
    spectrum=torch.fft.fft2(residual)
    return torch.stack([torch.fft.ifft2(torch.roll(spectrum,(-ky,-kx),(-2,-1))*op.decode_transfers[c]*op.support) for c,(kx,ky) in enumerate(op.bins)],1)


class ChannelNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1,width,1,1))
        self.bias = nn.Parameter(torch.zeros(1,width,1,1))

    def forward(self,x):
        mean = x.mean(1,keepdim=True)
        return (x-mean)*torch.rsqrt((x-mean).square().mean(1,keepdim=True)+1e-6)*self.weight+self.bias


class NAFBlock(nn.Module):
    # NAFNet mechanism: https://github.com/megvii-research/NAFNet (MIT).
    # This implementation uses periodic depthwise padding and native autograd.
    def __init__(self,width):
        super().__init__()
        self.norm1,self.norm2 = ChannelNorm(width),ChannelNorm(width)
        self.expand = nn.Conv2d(width,2*width,1)
        self.depthwise = nn.Conv2d(2*width,2*width,3,padding=1,groups=2*width,padding_mode="circular")
        self.attention = nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Conv2d(width,width,1))
        self.reduce = nn.Conv2d(width,width,1)
        self.ffn1,self.ffn2 = nn.Conv2d(width,2*width,1),nn.Conv2d(width,width,1)
        self.beta = nn.Parameter(torch.zeros(1,width,1,1))
        self.gamma = nn.Parameter(torch.zeros(1,width,1,1))

    def forward(self,x):
        a,b = self.depthwise(self.expand(self.norm1(x))).chunk(2,1)
        gated = a*b
        y = x+self.beta*self.reduce(gated*self.attention(gated))
        a,b = self.ffn1(self.norm2(y)).chunk(2,1)
        return y+self.gamma*self.ffn2(a*b)


class NAFUNet(nn.Module):
    def __init__(self,inputs,outputs,width=24):
        super().__init__()
        self.intro = nn.Conv2d(inputs,width,3,padding=1,padding_mode="circular")
        self.enc = nn.ModuleList([nn.Sequential(NAFBlock(width),NAFBlock(width)), nn.Sequential(NAFBlock(2*width),NAFBlock(2*width))])
        self.down = nn.ModuleList([nn.Conv2d(width,2*width,2,2),nn.Conv2d(2*width,4*width,2,2)])
        self.middle = nn.Sequential(*[NAFBlock(4*width) for _ in range(4)])
        self.up = nn.ModuleList([nn.Sequential(nn.Conv2d(4*width,8*width,1,bias=False),nn.PixelShuffle(2)), nn.Sequential(nn.Conv2d(2*width,4*width,1,bias=False),nn.PixelShuffle(2))])
        self.dec = nn.ModuleList([nn.Sequential(NAFBlock(2*width),NAFBlock(2*width)),nn.Sequential(NAFBlock(width),NAFBlock(width))])
        self.head = nn.Conv2d(width,outputs,3,padding=1,padding_mode="circular")
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self,features):
        x = self.intro(features)
        skips = []
        for enc,down in zip(self.enc,self.down):
            x = enc(x)
            skips.append(x)
            x = down(x)
        x = self.middle(x)
        for up,dec,skip in zip(self.up,self.dec,reversed(skips)):
            x = dec(up(x)+skip)
        return self.head(x)


class GroupedPrior(nn.Module):
    def __init__(self,width=24):
        super().__init__()
        self.thermal = NAFUNet(25,3,width)
        self.visible = NAFUNet(13,1,width)
        self.capture_stats=False
        self.statistics={}

    def forward(self,op,current,anchor,initial,noise,residual,stage):
        b,_,h,w = current.shape
        stage_map = torch.full_like(current[:,0:1],stage/2)
        t0 = anchor[:,3:]
        # First-order diagonal noise propagation; not a posterior uncertainty.
        relative = noise[:,3:]/t0.clamp_min(1e-6)
        sigma_b = relative.square().sum(1,keepdim=True).sqrt()/3
        sigma_q = torch.cat((sigma_b,(relative[:,0:1].square()+relative[:,1:2].square()).sqrt(),(relative[:,2:3].square()+relative[:,1:2].square()).sqrt()),1)
        tf = torch.cat((thermal_coordinates(t0),thermal_coordinates(current[:,3:]),initial[:,3:]-t0,noise[:,3:],residual[:,3:].real,residual[:,3:].imag,gradient(current[:,3:]).flatten(1,2),stage_map),1)
        thermal_head=self.thermal(tf)
        tq = 2*sigma_q*torch.tanh(thermal_head)
        dt = t0*torch.expm1(coordinates_to_logs(tq))
        vf = torch.stack([torch.cat((anchor[:,c:c+1],current[:,c:c+1],initial[:,c:c+1]-anchor[:,c:c+1],residual[:,c:c+1].real,residual[:,c:c+1].imag,noise[:,c:c+1],gradient(current[:,c:c+1]).flatten(1,2),torch.full_like(stage_map,[431.5,486,516.5][c]/905),torch.full_like(stage_map,float(op.prior_mu[c])),noise[:,c:c+1].clamp_min(1e-6).log(),stage_map,torch.ones_like(stage_map)),1) for c in range(3)],1)
        visible_head = self.visible(vf.flatten(0,1)).reshape(b,3,h,w)
        dv = 3*noise[:,:3]*torch.tanh(visible_head)
        if self.capture_stats:
            self.statistics={"thermal_tanh_over_0_98_fraction":float((thermal_head.detach().tanh().abs()>.98).float().mean()),"visible_tanh_over_0_98_fraction":float((visible_head.detach().tanh().abs()>.98).float().mean()),"normalized_delta_RMS":torch.cat((dv,dt),1).detach().square().mean((-2,-1)).sqrt().mean(0).tolist()}
        return op.project(torch.cat((dv,dt),1))


class OrdinaryPrior(nn.Module):
    """Both role outputs see all six images; separate trunks match freezing."""
    def __init__(self,width=24):
        super().__init__()
        self.thermal=NAFUNet(25,3,width)
        self.visible=NAFUNet(25,3,width)
        self.capture_stats=False
        self.statistics={}

    def forward(self,op,current,anchor,initial,noise,residual,stage):
        features=torch.cat((current,anchor,initial-anchor,noise,torch.full_like(current[:,:1],stage/2)),1)
        head=torch.cat((self.visible(features),self.thermal(features)),1)
        if self.capture_stats:self.statistics={"thermal_tanh_over_0_98_fraction":float((head[:,3:].detach().tanh().abs()>.98).float().mean()),"visible_tanh_over_0_98_fraction":float((head[:,:3].detach().tanh().abs()>.98).float().mean())}
        return op.project(3*noise*torch.tanh(head))


class TVRRU(nn.Module):
    def __init__(self,cfg,method="full",width=24):
        super().__init__()
        self.method,self.cfg = method,cfg
        if method=="ordinary_post":
            self.prior = OrdinaryPrior(width)
        else:
            self.prior = GroupedPrior(width)
        self.stages = 3

    def forward(self,op,batch,thermal_only=False):
        anchor,initial = batch["anchor"],batch["initial"]
        x,d,u = anchor,batch["d"],batch["u"]
        for stage in range(self.stages):
            if bool((x[:,3:]<=0).any()):
                raise RuntimeError(f"Nonpositive thermal input before stage {stage}")
            residual = sideband_residual(op,batch["y"]-op.encode_images(x))
            if self.training and torch.is_grad_enabled():
                delta=checkpoint(self.prior,op,x,anchor,initial,batch["noise"],residual,stage,use_reentrant=False)
            else:
                delta=self.prior(op,x,anchor,initial,batch["noise"],residual,stage)
            if thermal_only:
                delta = torch.cat((torch.zeros_like(delta[:,:3]),delta[:,3:]),1)
            proposal = anchor+delta
            if self.method in ("full","no_pair"):
                x,d,u = anchored_tv_prox(op,batch["y"],batch["weight"],proposal,(x,d,u),self.cfg,checkpoint_steps=self.training)
            else:
                x = proposal
        return x
