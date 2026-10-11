"""Matrix-free damped Gauss-Newton blocks using the actual coded Jacobian."""
import runtime
import torch


def pack(fields, beta):
    return torch.cat([fields.flatten(1), beta], dim=1)


def unpack(vector, shape):
    n = 5 * shape[0] * shape[1]
    return vector[:, :n].reshape(-1, 5, *shape), vector[:, n:]


def batched_cg(matvec, target, diagonal, iterations=4):
    x = torch.zeros_like(target)
    r = target
    z = r / diagonal
    p = z
    rz = (r*z).sum(1, keepdim=True)
    for _ in range(iterations):
        ap = matvec(p)
        denominator = (p*ap).sum(1, keepdim=True).clamp_min(1e-30)
        alpha = rz / denominator
        x = x + alpha * p
        r = r - alpha * ap
        z = r / diagonal
        next_rz = (r*z).sum(1, keepdim=True)
        p = z + (next_rz/rz.clamp_min(1e-30)) * p
        rz = next_rz
    return x


def normal_system(physics, state, beta, weights, field_mask, beta_active=True, damping=None):
    jac = physics.spectral_jacobian(state, beta)
    # Reuse this linearization in CG; no CASSI diagonal data-step assumption.
    mask = field_mask[None, :, None, None]
    def jvp(vector):
        direction, bdir = unpack(vector, state.shape[-2:])
        bands = (jac * (direction*mask)[:, :, None]).sum(1)
        if beta_active:
            bg = physics.Bscale * torch.einsum("bl,lhw->bhw", bdir, physics.background_basis)
            bands = bands + physics.q[None, :, None, None] * bg[:, None]
        return physics.op(bands)
    def vjp(raw):
        back = physics.op.adjoint(raw)
        g = (jac * back[:, None]).sum(2) * mask
        bg = (back * physics.q[None, :, None, None]).sum(1)
        gb = physics.Bscale * torch.einsum("bhw,lhw->bl", bg, physics.background_basis)
        if not beta_active:
            gb = gb * 0
        return pack(g, gb)
    diagonal_fields, diagonal_beta = physics.preconditioner_diagonal(state, beta, weights)
    diagonal = pack(diagonal_fields, diagonal_beta)
    damping = physics.cfg["GN_damping"] if damping is None else damping
    def matvec(vector):
        return vjp(weights*jvp(vector)) + damping * diagonal * vector
    return matvec, vjp, diagonal * (1+damping)


def accept_step(physics, raw_e, state, beta, direction, beta_direction, weights):
    base = ((physics(state, beta)-raw_e).square()*weights).sum((-2, -1))
    chosen_state, chosen_beta = state, beta
    accepted = torch.zeros(len(state), dtype=torch.bool, device=state.device)
    for alpha in (1., .5, .25, .125):
        candidate, candidate_beta = physics.constrain(state + alpha*direction, beta + alpha*beta_direction)
        expectation = physics(candidate, candidate_beta)
        cost = ((expectation-raw_e).square()*weights).sum((-2, -1))
        accept = (~accepted) & torch.isfinite(cost) & (expectation.amin((-2, -1)) >= 0) & (cost <= base*(1+1e-7)+1e-10)
        chosen_state = torch.where(accept[:, None, None, None], candidate, chosen_state)
        chosen_beta = torch.where(accept[:, None], candidate_beta, chosen_beta)
        accepted = accepted | accept
    return chosen_state, chosen_beta


def data_step(physics, raw_e, state, beta, cg_steps=None, learnable=False):
    del learnable  # Both paths use the same physical update.
    cg_steps = cg_steps or physics.cfg["CG_steps_per_spectral_block"]
    variance = physics(state, beta).detach().clamp_min(0) + physics.extra_variance
    weights = variance.reciprocal()
    # Block one: log-amplitude linearization, species and structured background.
    mask = torch.cat([torch.zeros(1, device=state.device, dtype=state.dtype), torch.ones(1, device=state.device, dtype=state.dtype), physics.active])
    normal, vjp, diagonal = normal_system(physics, state, beta, weights, mask)
    target = -vjp(weights*(physics(state, beta)-raw_e))
    direction = batched_cg(normal, target, diagonal, cg_steps)
    dz, db = unpack(direction, state.shape[-2:])
    state, beta = accept_step(physics, raw_e, state, beta, dz, db, weights)
    # Block two: a damped temperature-Jacobian step, with the full coded operator.
    mask_t = torch.tensor([1, 0, 0, 0, 0], device=state.device, dtype=state.dtype)
    normal_t, vjp_t, diagonal_t = normal_system(physics, state, beta, weights, mask_t, beta_active=False)
    target_t = -vjp_t(weights*(physics(state, beta)-raw_e))
    dtemp = batched_cg(normal_t, target_t, diagonal_t, 1)
    dt, dbt = unpack(dtemp, state.shape[-2:])
    return accept_step(physics, raw_e, state, beta, dt, dbt, weights)


def reconstruct(physics, raw_e, outer_steps=60, cg_steps=12, tolerance=1e-6):
    """Conventional nonlinear reference; convergence is reported, not assumed."""
    state, beta = physics.initialize(raw_e)
    previous = None
    converged = False
    iterations = 0
    with torch.no_grad():
        for iterations in range(1, outer_steps+1):
            state, beta = data_step(physics, raw_e, state, beta, cg_steps=cg_steps)
            objective = ((physics(state, beta)-raw_e).square() / (physics(state, beta).clamp_min(0)+physics.extra_variance)).mean().item()
            if previous is not None and abs(previous-objective) <= tolerance*max(previous, 1e-12):
                converged = True
                break
            previous = objective
    return state, beta, {"outer_steps": iterations, "stagnation_tolerance_met": converged, "objective": objective, "stationarity_certified": False, "note": "Objective stagnation is not a KKT or global-convergence certificate."}

