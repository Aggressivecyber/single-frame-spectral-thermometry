"""Original isotropic TV, with saved ADMM states and differentiable anchored steps."""
import runtime
import torch
from torch.utils.checkpoint import checkpoint
from research_joint_imaging.core import gradient, gradient_adjoint


def shrink(v, threshold):
    norm = v.square().sum(-3, keepdim=True).clamp_min(torch.finfo(v.dtype).eps**2).sqrt()
    return v * (1-threshold/norm).clamp_min(0)


def residuals(op, y, weight, state, strength, rho, proposal=None, mu=0.0):
    x, d, u = state
    dx = gradient(x)
    stationarity = op.project(op.images_adjoint(weight*(op.encode_images(x)-y)) + rho*gradient_adjoint(u))
    if proposal is not None:
        stationarity = stationarity + mu*(x-proposal)
    objective = 0.5*(weight*(op.encode_images(x)-y).square()).sum((-2, -1)) + strength*dx.square().sum(-3).sqrt().sum((-3, -2, -1))
    if proposal is not None:
        objective = objective + 0.5*(mu*(x-proposal).square()).sum((-3, -2, -1))
    return {"objective_mean": float(objective.detach().mean()), "stationarity_rms": float(stationarity.detach().square().mean().sqrt()), "primal_relative": float((dx-d).detach().norm()/dx.detach().norm()),"primal_rms":float((dx-d).detach().square().mean().sqrt()),"gradient_rms":float(dx.detach().square().mean().sqrt())}


@torch.no_grad()
def tv_anchor(op, y, weight, initial, cfg, iterations=None):
    x = initial.clone()
    d, u = gradient(x), torch.zeros_like(gradient(x))
    data_rhs = op.project(op.images_adjoint(weight*y))
    history = []
    limit = cfg["maximum_iterations"] if iterations is None else iterations
    for step in range(limit):
        before = x
        x = op.solve(data_rhs+cfg["rho"]*gradient_adjoint(d-u), x, weight, 0.0, cfg["rho"], cfg["pcg_iterations"], cfg["pcg_tolerance"])
        dx = gradient(x)
        d = shrink(dx+u, cfg["strength"]/cfg["rho"])
        u = u+dx-d
        if step+1==min(400,limit):reference=x.clone()
        if (step+1) % 40 == 0 or step+1 == limit:
            item = residuals(op, y, weight, (x,d,u), cfg["strength"], cfg["rho"])
            item.update(iteration=step+1, relative_change=float((x-before).norm()/x.norm()))
            item["primal_rms_tolerance"]=cfg["primal_absolute_tolerance"]+cfg["primal_relative_tolerance"]*item["gradient_rms"]
            item["converged"]=bool(item["primal_rms"]<=item["primal_rms_tolerance"] and item["stationarity_rms"]<=cfg["stationarity_rms_tolerance"] and item["relative_change"]<cfg["relative_iteration_change_target"])
            history.append(item)
            if iterations is None and step+1>=min(400,limit) and item["converged"]:
                break
    return (x,d,u), history,reference


def fixed_pcg(op, rhs, initial, weight, mu, rho, steps):
    # A 95% warm start keeps the first Krylov vector active at a TV fixed point.
    # A relative arithmetic floor regularizes only roundoff after convergence.
    x = .95*initial
    r = rhs-op.normal(x, weight, mu, rho)
    q = op.precondition(r, weight, mu, rho)
    p = q
    rq = (r*q).sum((-3,-2,-1), keepdim=True)
    arithmetic_floor = rq.detach().abs()*torch.finfo(x.dtype).eps**2 + torch.finfo(x.dtype).tiny
    for _ in range(steps):
        ap = op.normal(p, weight, mu, rho)
        alpha = rq/((p*ap).sum((-3,-2,-1), keepdim=True)+arithmetic_floor)
        x = x+alpha*p
        r = r-alpha*ap
        q = op.precondition(r, weight, mu, rho)
        next_rq = (r*q).sum((-3,-2,-1), keepdim=True)
        p = q+(next_rq/(rq+arithmetic_floor))*p
        rq = next_rq
    return op.project(x)


def anchored_tv_prox(op, y, weight, proposal, state, cfg, checkpoint_steps=False):
    mu = op.prior_mu[:,None,None]*cfg["mu_multiplier"]
    rhs = op.project(op.images_adjoint(weight*y))+mu*proposal
    rho, threshold = cfg["rho"], cfg["strength"]/cfg["rho"]

    def iteration(x, d, u, data_rhs):
        x = fixed_pcg(op, data_rhs+rho*gradient_adjoint(d-u), x, weight, mu, rho, cfg["prox_pcg_iterations"])
        dx = gradient(x)
        d = shrink(dx+u, threshold)
        return x, d, u+dx-d

    x, d, u = state
    for _ in range(cfg["prox_admm_iterations"]):
        if checkpoint_steps and torch.is_grad_enabled():
            x,d,u = checkpoint(iteration, x,d,u,rhs, use_reentrant=False)
        else:
            x,d,u = iteration(x,d,u,rhs)
    return x,d,u
