"""Execute the original v3 scalar algorithm, memoizing constant linearizations."""
import copy
import time
import torch
from repair_v3.solver import DEFAULTS, consistent_initialize, joint_data_step


class LinearizationProxy:
    """Read-only original physics; a per-frame Jacobian cache, never shared."""
    def __init__(self, physics):
        self.physics, self._state, self._beta, self._jac = physics, None, None, None

    def __getattr__(self, name):
        return getattr(self.physics, name)

    def __call__(self, state, beta):
        return self.physics(state, beta)

    def spectral_jacobian(self, state, beta):
        if self._state is not state or self._beta is not beta:
            self._jac = self.physics.spectral_jacobian(state, beta)
            self._state, self._beta = state, beta
        return self._jac

    def jvp(self, state, beta, direction, beta_direction):
        jac = self.spectral_jacobian(state, beta)
        bands = (jac*direction[:, :, None]).sum(1)
        bg = self.Bscale*torch.einsum('bl,lhw->bhw', beta_direction, self.background_basis)
        return self.op(bands+self.q[None, :, None, None]*bg[:, None])

    def vjp(self, state, beta, residual):
        back = self.op.adjoint(residual)
        gradient = (self.spectral_jacobian(state, beta)*back[:, None]).sum(2)
        bgmap = (back*self.q[None, :, None, None]).sum(1)
        beta_gradient = self.Bscale*torch.einsum('bhw,lhw->bl', bgmap, self.background_basis)
        return gradient, beta_gradient

    def preconditioner_diagonal(self, state, beta, weights):
        wm = weights.mean((-2, -1))
        jac = self.spectral_jacobian(state, beta)
        diagonal = wm[:, None, None, None]*(jac.square()*self.op.pattern_energy[None, None, :, None, None]).sum(2)
        qb = (self.q.square()*self.op.pattern_energy).sum()*self.Bscale**2
        beta_diag = wm[:, None]*qb*self.background_basis.square().sum((-2, -1))[None]
        return diagonal.clamp_min(1e-7), beta_diag.clamp_min(1e-7)


def invert_single(physics, raw_e, method, config=None):
    assert len(raw_e) == 1
    config = copy.deepcopy(DEFAULTS if config is None else config)
    p = LinearizationProxy(physics)
    started = time.perf_counter()
    with torch.no_grad():
        if method.startswith('consistent_init'):
            state, beta, warm = consistent_initialize(p, raw_e, config)
        else:
            state, beta = p.initialize(raw_e)
            warm = []
        joint, damping = [], config['LM_initial']
        for i in range(config['outer_iterations']):
            state, beta, info = joint_data_step(p, raw_e, state, beta, config, damping)
            damping = info['next_damping']
            joint.append({'iteration': i+1, **info})
            if not info['accepted']:
                break
        assert torch.isfinite(state).all() and torch.isfinite(beta).all()
    return state, beta, [{'warm': warm, 'joint': joint}], time.perf_counter()-started
