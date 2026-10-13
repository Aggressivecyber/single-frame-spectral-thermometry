"""Constrained warm start and joint LM with raw-only acceptance objectives."""
import time
import torch
from solver import pack, unpack, normal_system, batched_cg, accept_step


DEFAULTS = {
    'warm_iterations': 30, 'outer_iterations': 12, 'cg_max_iterations': 120,
    'cg_relative_tolerance': 1e-4, 'LM_initial': 1e-3, 'LM_minimum': 1e-8,
    'LM_maximum': 1e8, 'LM_trials': 6, 'acceptance_ratio_minimum': .05,
    'line_search': [1., .5, .25, .125, .0625],
    'regularization': [1e-5, 1e-5, 0., 0., 1e-3],
    'warm_regularization': [1e-6, 1e-6, 0., 0., 0.],
    'per_frame_wall_s': 180., 'max_GPU_allocated_MiB': 2500.,
    'data_weights': 'fixed reciprocal(max(observed_e,0)+read_noise_squared+ADC_variance)',
    'regularizer': 'periodic latent-field nearest-neighbour quadratic; no template positivity',
    'damping': 'LM diagonal scaling in step system only; absent from acceptance objective',
}


def laplacian(x):
    return 4*x - torch.roll(x, 1, -1) - torch.roll(x, -1, -1) - torch.roll(x, 1, -2) - torch.roll(x, -1, -2)


def regularization(state, strength):
    penalty = strength[None, :, None, None]
    cost = .5 * (penalty*((state-torch.roll(state, 1, -1)).square()+
                         (state-torch.roll(state, 1, -2)).square())).sum()
    return cost, penalty*laplacian(state)


def pcg(matvec, target, precondition, *, rtol, max_iterations):
    """Single-frame PCG, relative true residual stopping, explicit failure reasons."""
    x = torch.zeros_like(target)
    r = target.clone()
    norm = torch.linalg.vector_norm(target)
    if float(norm) == 0:
        return x, {'iterations': 0, 'relative_residual': 0., 'stop': 'ZERO_RHS', 'converged': True}
    z = precondition(r)
    direction, rz = z.clone(), (r*z).sum()
    stop = 'MAX_ITERATIONS'
    history = []
    for iteration in range(1, max_iterations+1):
        product = matvec(direction)
        denominator = (direction*product).sum()
        if not torch.isfinite(denominator) or float(denominator) <= 0:
            stop = 'NON_POSITIVE_CURVATURE'
            break
        alpha = rz/denominator
        x = x+alpha*direction
        r = r-alpha*product
        relative = float(torch.linalg.vector_norm(r)/norm)
        history.append(relative)
        if relative <= rtol:
            r = target-matvec(x)
            if float(torch.linalg.vector_norm(r)/norm) <= rtol:
                stop = 'RELATIVE_TRUE_RESIDUAL'
                break
        z = precondition(r)
        next_rz = (r*z).sum()
        if not torch.isfinite(next_rz) or float(next_rz) <= 0:
            stop = 'PRECONDITIONER_BREAKDOWN'
            break
        direction = z+(next_rz/rz)*direction
        rz = next_rz
    relative = float(torch.linalg.vector_norm(target-matvec(x))/norm)
    return x, {'iterations': len(history), 'relative_residual': relative,
               'recursive_residual_history': history, 'stop': stop, 'converged': relative <= rtol}


def delta_statistics(p, old, new, old_beta, new_beta):
    delta = new-old
    def describe(x):
        return {'rms': float(x.square().mean().sqrt()), 'maximum_abs': float(x.abs().max()), 'signed_mean': float(x.mean())}
    limits = p.cfg['temperature_range_K']
    T = p.T0+p.Tscale*new[:, 0]
    active = new[:, 2:][:, p.active.bool()]
    return {'T_increment_K': describe(delta[:, 0]*p.Tscale),
            'loga_increment': describe(delta[:, 1]),
            'template_increment_e': describe(delta[:, 2:][:, p.active.bool()]*p.Sscale),
            'beta_increment': describe(new_beta-old_beta),
            'constraint_boundary_fraction': {
                'T': float(((T <= limits[0]+1e-4)|(T >= limits[1]-1e-4)).float().mean()),
                'loga': float(((new[:, 1] <= -4+1e-6)|(new[:, 1] >= .3-1e-6)).float().mean()),
                'active_template': float((active.abs() >= 20-1e-6).float().mean()),
                'beta': float((new_beta.abs() >= 20-1e-6).float().mean())}}


def linearization(p, state, beta, *, warm=False):
    """Use the existing JVP/VJP and object-blur operator, with no OTF division."""
    active = torch.cat([torch.ones(2, device=state.device, dtype=state.dtype),
                        torch.zeros_like(p.active) if warm else p.active])
    def jvp(vector):
        dz, db = unpack(vector, state.shape[-2:])
        dz = dz*active[None, :, None, None]
        if warm:
            jac = p.spectral_jacobian(state, beta)
            return p.op.object_blur((jac*dz[:, :, None]).sum(1))[:, 5:7]
        return p.jvp(state, beta, dz, db)
    def vjp(residual):
        if warm:
            back = p.op.object_blur(residual)
            jac = p.spectral_jacobian(state, beta)[:, :, 5:7]
            dz = (jac*back[:, None]).sum(2)
            db = torch.zeros_like(beta)
        else:
            dz, db = p.vjp(state, beta, residual)
        return pack(dz*active[None, :, None, None], db)
    return jvp, vjp, active


def objective(p, observed, state, beta, weights, strength, *, warm=False):
    prediction = p.op.object_blur(p.components(state, beta)[0])[:, 5:7] if warm else p(state, beta)
    raw = .5*((prediction-observed).square()*weights).sum()
    reg, _ = regularization(state, strength)
    return raw+reg, raw, reg, prediction


def block_preconditioner(p, state, beta, weights, active, strength, damping, *, warm=False):
    """Local coupled parameter blocks, retaining T/loga correlations and scales."""
    indices = active.nonzero().flatten()
    jac = p.spectral_jacobian(state, beta)[:, indices]
    if warm:
        jac = jac[:, :, 5:7]
        factors = weights.mean((-2, -1))[:, None, :, None, None]
    else:
        factors = weights.mean((-2, -1))[:, None, None, None, None]*p.op.pattern_energy[None, None, :, None, None]
    gram = torch.einsum('bikhw,bjkhw->bhwij', jac*factors, jac)
    diagonal = gram.diagonal(dim1=-2, dim2=-1).clamp_min(1e-7)
    correction = damping*diagonal+4*strength[indices]
    inverse = torch.linalg.inv(gram+torch.diag_embed(correction))
    _, beta_diagonal = p.preconditioner_diagonal(state, beta, weights.mean(1) if warm else weights)
    fields = torch.zeros_like(state)
    fields[:, indices] = diagonal.permute(0, 3, 1, 2)
    fields = fields.clamp_min(1e-7)
    scaling = pack(fields, beta_diagonal)
    def precondition(vector):
        dz, db = unpack(vector, state.shape[-2:])
        answer = torch.zeros_like(dz)
        answer[:, indices] = torch.einsum('bhwij,bhwj->bhwi', inverse, dz[:, indices].permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return pack(answer, torch.zeros_like(db) if warm else db/((1+damping)*beta_diagonal))
    return precondition, scaling


def lm_step(p, observed, state, beta, weights, config, damping, *, warm=False):
    strength = state.new_tensor(config['warm_regularization'] if warm else config['regularization'])
    base, raw_base, reg_base, prediction = objective(p, observed, state, beta, weights, strength, warm=warm)
    jvp, vjp, active = linearization(p, state, beta, warm=warm)
    _, reg_gradient = regularization(state, strength)
    gradient = vjp(weights*(prediction-observed))+pack(reg_gradient, torch.zeros_like(beta))
    def curvature(vector):
        dz, db = unpack(vector, state.shape[-2:])
        return vjp(weights*jvp(vector))+pack(strength[None, :, None, None]*laplacian(dz), torch.zeros_like(db))
    trials, accepted = [], False
    chosen_state, chosen_beta = state, beta
    last_cg = None
    used_damping, step_size, ratio, predicted, actual = damping, 0., None, 0., 0.
    for _ in range(config['LM_trials']):
        precondition, scaling = block_preconditioner(p, state, beta, weights, active, strength, damping, warm=warm)
        normal = lambda v: curvature(v)+damping*scaling*v
        direction, cg = pcg(normal, -gradient, precondition, rtol=config['cg_relative_tolerance'], max_iterations=config['cg_max_iterations'])
        last_cg, used_damping = cg, damping
        dz, db = unpack(direction, state.shape[-2:])
        for alpha in config['line_search']:
            candidate, cbeta = p.constrain(state+alpha*dz, beta+alpha*db)
            projected = pack(candidate-state, cbeta-beta)
            pred_drop = -(gradient*projected).sum()-.5*(projected*curvature(projected)).sum()
            cost, raw_cost, reg_cost, expectation = objective(p, observed, candidate, cbeta, weights, strength, warm=warm)
            actual_drop = base-cost
            rho = float(actual_drop/pred_drop) if float(pred_drop) > 0 else None
            valid = bool(torch.isfinite(cost) and torch.isfinite(expectation).all() and expectation.min() >= 0)
            accept = valid and rho is not None and float(actual_drop) > 0 and rho >= config['acceptance_ratio_minimum']
            trials.append({'damping': damping, 'alpha': alpha, 'actual_reduction': float(actual_drop),
                           'predicted_reduction': float(pred_drop), 'ratio': rho, 'accepted': accept,
                           'CG_relative_residual': cg['relative_residual'], 'CG_converged': cg['converged']})
            if accept:
                chosen_state, chosen_beta, accepted = candidate, cbeta, True
                step_size, ratio, predicted, actual = alpha, rho, float(pred_drop), float(actual_drop)
                damping = max(config['LM_minimum'], damping*.3) if rho > .75 else damping
                break
        if accepted:
            break
        damping = min(config['LM_maximum'], damping*10)
    final, raw_final, reg_final, _ = objective(p, observed, chosen_state, chosen_beta, weights, strength, warm=warm)
    report = {'accepted': accepted, 'accepted_alpha': step_size, 'damping_used': used_damping,
              'next_damping': damping, 'actual_predicted_ratio': ratio,
              'actual_reduction': actual, 'predicted_reduction': predicted,
              'objective_before': float(base), 'objective_after': float(final),
              'raw_objective_before': float(raw_base), 'raw_objective_after': float(raw_final),
              'regularization_before': float(reg_base), 'regularization_after': float(reg_final),
              'CG': last_cg, 'trials': trials,
              **delta_statistics(p, state, chosen_state, beta, chosen_beta)}
    return chosen_state, chosen_beta, report


def joint_data_step(p, raw_e, state, beta, config=None, damping=None):
    config = dict(DEFAULTS if config is None else config)
    weights = (raw_e.clamp_min(0)+p.extra_variance).reciprocal()
    return lm_step(p, raw_e, state, beta, weights, config,
                   config['LM_initial'] if damping is None else damping)


def consistent_initialize(p, raw_e, config=None, *, deadline=None):
    config = DEFAULTS if config is None else config
    state, beta = p.initialize(raw_e)
    decoded = p.op.decode(raw_e)[:, 5:7].detach()
    weights = (decoded.clamp_min(0)+p.extra_variance).reciprocal()
    trace, damping = [], config['LM_initial']
    for i in range(config['warm_iterations']):
        if deadline is not None and time.perf_counter() > deadline:
            raise TimeoutError('Continuum warm start exceeded frame budget')
        state, beta, info = lm_step(p, decoded, state, beta, weights, config, damping, warm=True)
        info['iteration'] = i+1
        trace.append(info)
        damping = info['next_damping']
        if not info['accepted']:
            break
    return state, beta, trace


def old_step_with_subblocks(p, raw_e, state, beta):
    """Instrument the exact old spectral and single-CG temperature blocks."""
    weights = (p(state, beta).detach().clamp_min(0)+p.extra_variance).reciprocal()
    mask = torch.cat([state.new_zeros(1), state.new_ones(1), p.active])
    normal, vjp, diagonal = normal_system(p, state, beta, weights, mask)
    target = -vjp(weights*(p(state, beta)-raw_e))
    step = batched_cg(normal, target, diagonal, p.cfg['CG_steps_per_spectral_block'])
    dz, db = unpack(step, state.shape[-2:])
    spectral, spectral_beta = accept_step(p, raw_e, state, beta, dz, db, weights)
    def accepted_alpha(before, before_beta, after, after_beta, direction, beta_direction):
        if torch.equal(before, after) and torch.equal(before_beta, after_beta):
            return 0.
        for alpha in (1., .5, .25, .125):
            candidate, candidate_beta = p.constrain(before+alpha*direction, before_beta+alpha*beta_direction)
            if torch.equal(candidate, after) and torch.equal(candidate_beta, after_beta):
                return alpha
        raise RuntimeError('Old accepted step could not be accounted for')
    report_s = {'CG_iterations': p.cfg['CG_steps_per_spectral_block'],
                'CG_relative_residual': float(torch.linalg.vector_norm(target-normal(step))/torch.linalg.vector_norm(target).clamp_min(1e-30)),
                'damping': p.cfg['GN_damping'],
                'accepted_alpha': accepted_alpha(state, beta, spectral, spectral_beta, dz, db),
                **delta_statistics(p, state, spectral, beta, spectral_beta)}
    mask_t = state.new_tensor([1, 0, 0, 0, 0])
    nt, vt, dt = normal_system(p, spectral, spectral_beta, weights, mask_t, beta_active=False)
    target_t = -vt(weights*(p(spectral, spectral_beta)-raw_e))
    step_t = batched_cg(nt, target_t, dt, 1)
    dz_t, db_t = unpack(step_t, state.shape[-2:])
    final, final_beta = accept_step(p, raw_e, spectral, spectral_beta, dz_t, db_t, weights)
    report_t = {'CG_iterations': 1,
                'CG_relative_residual': float(torch.linalg.vector_norm(target_t-nt(step_t))/torch.linalg.vector_norm(target_t).clamp_min(1e-30)),
                'damping': p.cfg['GN_damping'],
                'accepted_alpha': accepted_alpha(spectral, spectral_beta, final, final_beta, dz_t, db_t),
                **delta_statistics(p, spectral, final, spectral_beta, final_beta)}
    return final, final_beta, [(spectral, spectral_beta, 'old_spectral_subblock', report_s),
                              (final, final_beta, 'old_temperature_subblock', report_t)]
