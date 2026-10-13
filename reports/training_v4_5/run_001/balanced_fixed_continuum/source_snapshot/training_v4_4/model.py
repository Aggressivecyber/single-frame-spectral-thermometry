"""Fix only the final continuum output of the residual-scaled CNN.

The original six-stage proposal, its physical features, and its frozen residual
coefficient remain unchanged.  Internal proposal stages still move the continuum
and can affect later signed corrections; this is a final-output constraint, not
a fully unrolled inverse or a removal of all internal compensation.
"""
import runtime
import torch

from training_v4.model import physical_output
from training_v4_3.residual_scaling import ResidualScaledCNN


class FixedContinuumCNN(ResidualScaledCNN):
    """Keep cached continuum fields and the original scaled signed correction.

    Construction and CNN loading use the inherited ``(cfg, calibration)`` and
    ``load_cnn(state)`` APIs.  No parameters, buffers, thresholds, or inference
    inputs are added.  The continuum head remains trainable through its effect
    on the internal proposal and subsequent signed corrections.
    """

    def forward_electrons(self, raw_e, cached_state, cached_beta):
        scaled = super().forward_electrons(raw_e, cached_state, cached_beta)
        # Copy the physical continuum exactly; do not re-constrain or mutate the
        # cached inverse or either diagnostic state.  Keep the signed graph live.
        state = torch.cat([cached_state.detach()[:, :2],
                           scaled['state'][:, 2:]], dim=1)
        result = physical_output(self.physics, state, scaled['beta'], raw_e)
        result.update(
            correction_coefficient=scaled['correction_coefficient'],
            initial_weighted_raw_RMS=scaled['initial_weighted_raw_RMS'],
            unrestricted_proposal_state=scaled['unrestricted_proposal_state'],
            unrestricted_scaled_state=scaled['state'],
        )
        return result
