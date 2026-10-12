"""Independent-frame vectorization of v3 LM; no physical operator replacement.

Linearization is reused inside each CG solve. Every frame retains its own CG
residual, damping, line search, regularization objective and acceptance ratio.
The engine fingerprint is part of cache identity; it never impersonates a
byte-identical old audit result.
"""
import copy
import time
import torch
from solver import pack, unpack
from repair_v3.solver import DEFAULTS, laplacian


def regularizer(state, strength):
    q = strength[None, :, None, None]
    cost = .5*(q*((state-torch.roll(state, 1, -1)).square()+
                  (state-torch.roll(state, 1, -2)).square())).sum((1, 2, 3))
    return cost, q*laplacian(state)


def pcg_independent(matvec, rhs, precondition, config):
    x, r = torch.zeros_like(rhs), rhs.clone()
    norm = torch.linalg.vector_norm(rhs, dim=1).clamp_min(1e-30)
    alive = torch.linalg.vector_norm(rhs, dim=1) > 0
    z = precondition(r)
    direction, rz = z.clone(), (r*z).sum(1)
    iterations = torch.zeros(len(rhs), device=rhs.device, dtype=torch.long)
    stops = ['MAX_ITERATIONS']*len(rhs)
    history = []
    for _ in range(config['cg_max_iterations']):
        if not bool(alive.any()):
            break
        product = matvec(direction)
        denominator = (direction*product).sum(1)
        bad = alive & (~torch.isfinite(denominator) | (denominator <= 0))
        for i in bad.nonzero().flatten().tolist():
            stops[i] = 'NON_POSITIVE_CURVATURE'
        alive = alive & ~bad
        alpha = torch.where(alive, rz/denominator.clamp_min(1e-30), 0.)
        x = x+alpha[:, None]*direction
        r = r-alpha[:, None]*product
        iterations += alive
        relative = torch.linalg.vector_norm(r, dim=1)/norm
        history.append(relative.detach())
        candidate = alive & (relative <= config['cg_relative_tolerance'])
        if bool(candidate.any()):
            true = rhs-matvec(x)
            r = torch.where(candidate[:, None], true, r)
            done = candidate & (torch.linalg.vector_norm(true, dim=1)/norm <= config['cg_relative_tolerance'])
            for i in done.nonzero().flatten().tolist():
                stops[i] = 'RELATIVE_TRUE_RESIDUAL'
            alive = alive & ~done
        z = precondition(r)
        next_rz = (r*z).sum(1)
        bad = alive & (~torch.isfinite(next_rz) | (next_rz <= 0))
        for i in bad.nonzero().flatten().tolist():
            stops[i] = 'PRECONDITIONER_BREAKDOWN'
        alive = alive & ~bad
        ratio = torch.where(alive, next_rz/rz.clamp_min(1e-30), 0.)
        direction = torch.where(alive[:, None], z+ratio[:, None]*direction, 0.)
        rz = next_rz
    relative = torch.linalg.vector_norm(rhs-matvec(x), dim=1)/norm
    reports = []
    values, counts = relative.tolist(), iterations.tolist()
    for i, (v, count) in enumerate(zip(values, counts)):
        reports.append({'iterations': count, 'relative_residual': v,
                        'converged': v <= config['cg_relative_tolerance'],
                        'stop': 'ZERO_RHS' if count == 0 and v == 0 else stops[i]})
    return x, reports


def lm_batch(p, observed, state, beta, weights, config, damping, *, warm=False):
    strength = state.new_tensor(config['warm_regularization'] if warm else config['regularization'])
    active = torch.cat([state.new_ones(2), state.new_zeros(3) if warm else p.active])
    indices = active.nonzero().flatten()
    jac = p.spectral_jacobian(state, beta)
    if warm:
        jac = jac[:, :, 5:7]
    def jvp(vector):
        dz, db = unpack(vector, state.shape[-2:])
        bands = (jac*(dz*active[None, :, None, None])[:, :, None]).sum(1)
        if warm:
            return p.op.object_blur(bands)
        bg = p.Bscale*torch.einsum('bl,lhw->bhw', db, p.background_basis)
        return p.op(bands+p.q[None, :, None, None]*bg[:, None])
    def vjp(residual):
        back = p.op.object_blur(residual) if warm else p.op.adjoint(residual)
        dz = (jac*back[:, None]).sum(2)*active[None, :, None, None]
        db = torch.zeros_like(beta) if warm else p.Bscale*torch.einsum('bhw,lhw->bl',
            (back*p.q[None, :, None, None]).sum(1), p.background_basis)
        return pack(dz, db)
    def objective(z, b):
        expected = p.op.object_blur(p.components(z, b)[0])[:, 5:7] if warm else p(z, b)
        raw = .5*((expected-observed).square()*weights).flatten(1).sum(1)
        reg, _ = regularizer(z, strength)
        return raw+reg, raw, reg, expected
    base, raw_base, reg_base, expected = objective(state, beta)
    _, reg_gradient = regularizer(state, strength)
    gradient = vjp(weights*(expected-observed))+pack(reg_gradient, torch.zeros_like(beta))
    def curvature(vector):
        dz, _ = unpack(vector, state.shape[-2:])
        return vjp(weights*jvp(vector))+pack(strength[None, :, None, None]*laplacian(dz), torch.zeros_like(beta))
    selected_jac = jac[:, indices]
    factors = weights.mean((-2, -1))[:, None, :, None, None] if warm else weights.mean((-2, -1))[:, None, None, None, None]*p.op.pattern_energy[None, None, :, None, None]
    gram = torch.einsum('bikhw,bjkhw->bhwij', selected_jac*factors, selected_jac)
    diagonal = gram.diagonal(dim1=-2, dim2=-1).clamp_min(1e-7)
    fields = torch.zeros_like(state)
    fields[:, indices] = diagonal.permute(0, 3, 1, 2)
    _, beta_diagonal = p.preconditioner_diagonal(state, beta, weights.mean(1) if warm else weights)
    scaling = pack(fields.clamp_min(1e-7), beta_diagonal)
    accepted = torch.zeros(len(state), dtype=torch.bool, device=state.device)
    chosen, chosen_beta = state, beta
    next_damping = damping.clone()
    reports = [None]*len(state)
    trial_records = [[] for _ in state]
    for trial in range(config['LM_trials']):
        if bool(accepted.all()):
            break
        correction = damping[:, None, None, None]*diagonal+4*strength[indices]
        inverse = torch.linalg.inv(gram+torch.diag_embed(correction))
        def precondition(vector):
            dz, db = unpack(vector, state.shape[-2:])
            answer = torch.zeros_like(dz)
            answer[:, indices] = torch.einsum('bhwij,bhwj->bhwi', inverse,
                dz[:, indices].permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            return pack(answer, torch.zeros_like(db) if warm else db/((1+damping[:, None])*beta_diagonal))
        normal = lambda v: curvature(v)+damping[:, None]*scaling*v
        direction, cgs = pcg_independent(normal, -gradient, precondition, config)
        dz, db = unpack(direction, state.shape[-2:])
        for alpha in config['line_search']:
            candidate, cbeta = p.constrain(state+alpha*dz, beta+alpha*db)
            projected = pack(candidate-state, cbeta-beta)
            predicted = -(gradient*projected).sum(1)-.5*(projected*curvature(projected)).sum(1)
            cost, raw_cost, reg_cost, expectation = objective(candidate, cbeta)
            actual = base-cost
            ratio = torch.where(predicted > 0, actual/predicted, float('-inf'))
            valid = torch.isfinite(cost) & torch.isfinite(expectation).flatten(1).all(1) & (expectation.flatten(1).amin(1) >= 0)
            take = ~accepted & valid & (predicted > 0) & (actual > 0) & (ratio >= config['acceptance_ratio_minimum'])
            for i in (~accepted).nonzero().flatten().tolist():
                trial_records[i].append({'trial': trial, 'alpha': alpha, 'damping': float(damping[i]),
                    'actual_reduction': float(actual[i]), 'predicted_reduction': float(predicted[i]),
                    'accepted': bool(take[i]), 'CG': cgs[i]})
            for i in take.nonzero().flatten().tolist():
                reports[i] = {'accepted': True, 'alpha': alpha, 'damping': float(damping[i]),
                    'actual_reduction': float(actual[i]), 'predicted_reduction': float(predicted[i]),
                    'ratio': float(ratio[i]), 'objective_before': float(base[i]), 'objective_after': float(cost[i]),
                    'raw_before': float(raw_base[i]), 'raw_after': float(raw_cost[i]),
                    'regularization_before': float(reg_base[i]), 'regularization_after': float(reg_cost[i]), 'CG': cgs[i]}
            chosen = torch.where(take[:, None, None, None], candidate, chosen)
            chosen_beta = torch.where(take[:, None], cbeta, chosen_beta)
            next_damping = torch.where(take, torch.where(ratio > .75,
                (damping*.3).clamp_min(config['LM_minimum']), damping), next_damping)
            accepted |= take
            if bool(accepted.all()):
                break
        damping = torch.where(accepted, damping, (damping*10).clamp_max(config['LM_maximum']))
    for i, report in enumerate(reports):
        if report is None:
            reports[i] = {'accepted': False, 'alpha': 0., 'damping': float(damping[i]),
                          'objective_before': float(base[i]), 'objective_after': float(base[i]),
                          'CG': cgs[i]}
            next_damping[i] = damping[i]
        reports[i]['trials'] = trial_records[i]
    return chosen, chosen_beta, next_damping, reports


def invert_batch(p, raw_e, method, config=None):
    """Raw electron tensors only; truth/labels are absent from this interface."""
    config = copy.deepcopy(DEFAULTS if config is None else config)
    started = time.perf_counter()
    histories = [{'warm': [], 'joint': []} for _ in raw_e]
    with torch.no_grad():
        state, beta = p.initialize(raw_e)
        if method.startswith('consistent_init'):
            decoded = p.op.decode(raw_e)[:, 5:7]
            weights = (decoded.clamp_min(0)+p.extra_variance).reciprocal()
            damping = raw_e.new_full((len(raw_e),), config['LM_initial'])
            alive = torch.ones(len(raw_e), device=raw_e.device, dtype=torch.bool)
            for iteration in range(config['warm_iterations']):
                z, b, damping, info = lm_batch(p, decoded, state, beta, weights, config, damping, warm=True)
                state = torch.where(alive[:, None, None, None], z, state)
                beta = torch.where(alive[:, None], b, beta)
                for i in alive.nonzero().flatten().tolist():
                    histories[i]['warm'].append({'iteration': iteration+1, **info[i]})
                alive &= torch.tensor([r['accepted'] for r in info], device=raw_e.device)
                if not bool(alive.any()):
                    break
        weights = (raw_e.clamp_min(0)+p.extra_variance).reciprocal()
        damping = raw_e.new_full((len(raw_e),), config['LM_initial'])
        alive = torch.ones(len(raw_e), device=raw_e.device, dtype=torch.bool)
        for iteration in range(config['outer_iterations']):
            z, b, damping, info = lm_batch(p, raw_e, state, beta, weights, config, damping)
            state = torch.where(alive[:, None, None, None], z, state)
            beta = torch.where(alive[:, None], b, beta)
            for i in alive.nonzero().flatten().tolist():
                histories[i]['joint'].append({'iteration': iteration+1, **info[i]})
            alive &= torch.tensor([r['accepted'] for r in info], device=raw_e.device)
            if not bool(alive.any()):
                break
        assert torch.isfinite(state).all() and torch.isfinite(beta).all()
    return state.detach(), beta.detach(), histories, time.perf_counter()-started
