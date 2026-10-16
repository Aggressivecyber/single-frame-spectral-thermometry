"""Raw-only preparation and an implicit, signed observable-space proximal step."""
import runtime
import torch
from torch import nn
from .linear_profile import LinearProfile


def cg(matvec, rhs, *, initial=None, tolerance=1e-8, limit=256):
    x = torch.zeros_like(rhs) if initial is None else initial.clone()
    norm = rhs.norm()
    if float(norm) == 0:
        return torch.zeros_like(rhs), {'iterations': 0, 'relative_residual': 0.}
    residual = rhs-matvec(x)
    direction = residual.clone()
    rr = residual.square().sum()
    for iteration in range(limit+1):
        if float(rr.sqrt()/norm) <= tolerance:
            actual = float((rhs-matvec(x)).norm()/norm)
            if actual <= tolerance:
                return x, {'iterations': iteration, 'relative_residual': actual}
        if iteration == limit:
            break
        product = matvec(direction)
        denominator = (direction*product).sum()
        if not bool(torch.isfinite(denominator)) or float(denominator) <= 0:
            raise ArithmeticError('Nonpositive or nonfinite CG curvature')
        alpha = rr/denominator
        x = x+alpha*direction
        residual = residual-alpha*product
        next_rr = residual.square().sum()
        direction = residual+(next_rr/rr)*direction
        rr = next_rr
    raise ArithmeticError(f'CG failed: relative residual {float((rhs-matvec(x)).norm()/norm):.3g}')


class ConditionalProblem:
    def __init__(self, physics, weights, target, x0, *, basis=None):
        self.linear = LinearProfile(physics, weights, 1., basis=basis)
        self.target, self.x0 = target, x0
        self.mu = .05*self.linear.contract['template_reference_sensitivity']
        self.penalty = self.linear.reg.square().clone()
        self.penalty[:-3] += self.mu
        self.scale = self.linear.column_scale
        self.data_rhs = self.linear.adjoint(weights*target)
        self.last_solve = {}

    def normal(self, x):
        return self.linear.adjoint(self.linear.weights*self.linear.apply(x))+self.penalty*x

    def solve_rhs(self, rhs, initial=None):
        scale = self.scale
        scaled_initial = None if initial is None else initial/scale
        y, info = cg(lambda v: scale*self.normal(scale*v), scale*rhs, initial=scaled_initial)
        return scale*y, info

    def solve(self, proposal):
        rhs = self.data_rhs+torch.cat([self.mu*proposal, proposal.new_zeros(3)])
        answer, self.last_solve = self.solve_rhs(rhs, self.x0)
        return answer


class Proximal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, proposal, problem):
        ctx.problem = problem
        return problem.solve(proposal)

    @staticmethod
    def backward(ctx, gradient):
        answer, _ = ctx.problem.solve_rhs(gradient)
        return ctx.problem.mu*answer[:-3], None


class ResidualNet(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [nn.Conv2d(8, 16, 1), nn.SiLU()]
        for dilation in (1, 2, 4):
            layers.extend([nn.Conv2d(16, 16, 3, padding=dilation,
                                    dilation=dilation, groups=16, padding_mode='circular'),
                           nn.SiLU(), nn.Conv2d(16, 16, 1), nn.SiLU()])
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv2d(16, 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, features):
        return 60.*self.head(self.body(features))


def tangent_features(p, z, residual, weights):
    """Ridge-regularized local tangent fit; not an exact nuisance projection."""
    C, slope = p.continuum(p.T0+p.Tscale*z[:, 0])
    jac = torch.stack([slope*p.Tscale*z[:, 1:2].exp(), C*z[:, 1:2].exp()], 1)
    def jvp(delta):
        return p.op((jac*delta[:, :, None]).sum(1))
    def vjp(raw):
        return (jac*p.op.adjoint(raw)[:, None]).sum(2)
    diagonal = (jac.square()*p.op.pattern_energy[None, None, :, None, None]).sum(2)*weights.mean()
    # A unit ridge in normalized local coordinates prevents the full-resolution
    # continuum fit from absorbing the entire measurement by construction.
    scale = diagonal.rsqrt()
    rhs = scale*vjp(weights*residual)
    delta, info = cg(lambda v: scale*vjp(weights*jvp(scale*v))+v, rhs,
                     tolerance=1e-7, limit=128)
    return residual-jvp(scale*delta), info


def prepare(p, raw_e, state):
    """Accept only a raw frame and its frozen traditional state, without labels."""
    z = state[:, :2].to(torch.float64)
    C, _ = p.continuum(p.T0+p.Tscale*z[:, 0])
    C = C*z[:, 1:2].exp()
    weights = (raw_e.clamp_min(0)+p.extra_variance).reciprocal()[0]
    target = (raw_e-p.op(C))[0]
    linear = LinearProfile(p, weights, 1.)
    x0, info = linear.solve(target, tolerance=1e-11, maxiter=1200)
    if not info['numerical_certified']:
        raise ArithmeticError(f'Uncertified fixed-continuum profile: {info}')
    residual = target-linear.apply(x0)
    effective, tangent_info = tangent_features(p, z, residual[None], weights)
    sensitivity = linear.contract['template_reference_sensitivity']
    back = linear.U.unpack(linear.adjoint(weights*residual)[:-3])/sensitivity
    effective_back = linear.U.unpack(linear.adjoint(weights*effective[0])[:-3])/sensitivity
    u0 = linear.U.unpack(x0[:-3])
    features = torch.stack([u0/60., z[0, 0], z[0, 1], residual*weights.sqrt(),
                            back/60., effective_back/60., weights.rsqrt()/60.,
                            effective[0]*weights.sqrt()]).float()[None]
    temperature, _ = p.color_temperature(p.op.object_blur(C))
    return {'weights': weights, 'target': target, 'x0': x0, 'features': features,
            'temperature': temperature[0],
            'v3_template': p.op.object_blur(state[:, 4].double()*p.Sscale)[0],
            'profile_info': info, 'tangent_info': tangent_info}


def infer(model, problem, features):
    proposal = problem.x0[:-3]+problem.linear.U.pack(model(features)[0, 0].double())
    x = Proximal.apply(proposal, problem)
    return problem.linear.U.unpack(x[:-3]), x, proposal
