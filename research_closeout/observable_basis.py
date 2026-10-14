"""Real orthonormal Fourier coordinates, with one coordinate per real degree of freedom."""
from . import storage
import torch

class RealFourierBasis:
    def __init__(self,otf):
        self.shape=tuple(otf.shape);h,w=self.shape
        mask=otf!=0
        yy,xx=torch.meshgrid(torch.arange(h,device=otf.device),torch.arange(w,device=otf.device),indexing='ij')
        ids=yy*w+xx; neg=((-yy)%h)*w+((-xx)%w)
        if not torch.equal(mask.flatten(),mask.flatten()[neg.flatten()]): raise ValueError('OTF support is not Hermitian')
        self.self_ids=ids[mask & (ids==neg)].flatten()
        self.pair_ids=ids[mask & (ids<neg)].flatten()
        self.neg_ids=neg[mask & (ids<neg)].flatten()
        self.dimension=len(self.self_ids)+2*len(self.pair_ids)
        self.mask=mask;self.dtype=otf.dtype;self.device=otf.device
        fy,fx=torch.meshgrid(torch.fft.fftfreq(h,device=otf.device,dtype=otf.dtype),torch.fft.fftfreq(w,device=otf.device,dtype=otf.dtype),indexing='ij')
        lap=4*torch.sin(torch.pi*fx).square()+4*torch.sin(torch.pi*fy).square()
        self.laplacian=self.symbol(lap)
        self.otf_values=self.symbol(otf)
        if self.dimension!=int(mask.sum()): raise ValueError('Conjugate pairs were double-counted')
    def symbol(self,value):
        flat=value.flatten()
        return torch.cat([flat[self.self_ids],flat[self.pair_ids],flat[self.pair_ids]])
    def unpack(self,xi):
        a=len(self.self_ids);b=len(self.pair_ids)
        spectrum=torch.zeros(self.shape[0]*self.shape[1],dtype=torch.complex128 if xi.dtype==torch.float64 else torch.complex64,device=xi.device)
        spectrum[self.self_ids]=xi[:a].to(spectrum.dtype)
        values=(xi[a:a+b]+1j*xi[a+b:])/2**.5
        spectrum[self.pair_ids]=values;spectrum[self.neg_ids]=values.conj()
        return torch.fft.ifft2(spectrum.reshape(self.shape),norm='ortho').real
    def pack(self,image):
        X=torch.fft.fft2(image,norm='ortho').flatten()
        return torch.cat([X[self.self_ids].real,2**.5*X[self.pair_ids].real,2**.5*X[self.pair_ids].imag])
    def project(self,image): return self.unpack(self.pack(image))
    def report(self):
        return {'shape':list(self.shape),'nonzero_frequency_grid_points':int(self.mask.sum()),
            'real_degrees_of_freedom':self.dimension,'self_conjugate_modes':len(self.self_ids),
            'conjugate_pairs':len(self.pair_ids),'exact_zero_grid_points':int((~self.mask).sum()),
            'OTF_division':False,'norm':'orthonormal FFT'}
