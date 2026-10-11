# Model card

| Item | Value |
|---|---|
| Model | ParameterUnfoldedNet, parameter-unfold-v004 |
| Trainable parameters | 5,797 |
| Stages | Six, CNN weights shared |
| Input | 256×256 single coded DN frame, fixed synthetic calibration |
| Active template | Third slot only; synthetic AlO-named numerical fixture |
| Weight selection | Validation-selected low_lr_control, fine-tuning update 160 |
| Training data | Generated synthetic continuum/template/noise scenes |
| Measured training/evaluation data | None |
| Intended use | Numerical experiments and calibration/architecture research |
| Scientific acceptance | Not accepted; relative spatial L2 still fails its prototype screen |

The spectral channels and nominal widths are specified in `configs/synthetic_256.json`. The checkpoint also contains the full configuration and physics buffers. Loading is strict; changing the calibration while keeping the same physics buffers is unsupported.

The five physical fields parameterize continuum temperature, amplitude and signed emission-template coefficients. A three-term background is fitted jointly. Only one emission-template slot is active. The two NIR continuum bands determine effective post-blur color temperature, not gas temperature.

The original selected weights arose from a 300-update local CUDA pilot followed by a three-candidate server fine-tuning comparison (160 updates per candidate). All fine-tuning candidates used the same input stream. Validation selected the low learning-rate control. The published bytes retain its optimizer state for provenance; inference uses only the configuration and state dictionary.

Original fresh test IDs were 7000–7007, with three conditions per ID. They follow the original single-blob generator distribution. The 24 cases and their aggregate scores provide a small synthetic diagnostic, not evidence for robust out-of-distribution behavior.

Known limitations: sensitivity to unmodelled background and spectral response; limited spatial detail at the specified optical bandwidth; signed false positives on null scenes; periodic edges; fixed encoding/calibration; no independently measured input data; no physical throughput measurement. Later module experiments are not adopted in this release. Expanded scenario evaluation must keep this checkpoint fixed and disclose saturation, valid coverage and each perturbation separately.

Code and weights are released under the MIT license. No private acquisition data is included.
