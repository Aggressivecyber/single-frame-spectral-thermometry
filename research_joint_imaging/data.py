"""Independent mother scenes and saved raw ADC observations for the pilot."""
import runtime
from pathlib import Path
import numpy as np
import torch


def mother(seed, shape, device):
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.linspace(-1, 1, shape[0], endpoint=False), np.linspace(-1, 1, shape[1], endpoint=False), indexing="ij")
    temperature = rng.uniform(2750, 3250)+150*np.sin(np.pi*(xx+yy)+rng.uniform(0, 6))+120*np.cos(2*np.pi*yy+rng.uniform(0, 6))
    amplitude = 0.6+0.2*np.cos(np.pi*xx+rng.uniform(0, 6))*np.cos(np.pi*yy+rng.uniform(0, 6))
    features = []
    for _ in range(3):
        field = np.full(shape, 0.015)
        for _ in range(5):
            cx, cy = rng.uniform(-0.65, 0.65, 2)
            angle = rng.uniform(0, np.pi)
            u, v = (xx-cx)*np.cos(angle)+(yy-cy)*np.sin(angle), -(xx-cx)*np.sin(angle)+(yy-cy)*np.cos(angle)
            field += rng.uniform(0.3, 1)*np.exp(-0.5*((u/rng.uniform(.10, .35))**2+(v/rng.uniform(.035, .10))**2))
        features.append(field/field.mean())
    return (torch.tensor(temperature, dtype=torch.float32, device=device), torch.tensor(amplitude, dtype=torch.float32, device=device), torch.tensor(np.stack(features), dtype=torch.float32, device=device))


@torch.no_grad()
def prepare(protocol, op, rad, output):
    folder = output/"data"
    folder.mkdir()
    q = protocol["model"]["full_well_e"]/((2**protocol["model"]["adc_bits"]-1)-protocol["model"]["black_level_DN"])
    receipt = {"time": runtime.now(), "splits": {}, "total_raw_observations": 0, "saturated_samples": 0}
    sample_id = 0
    for split, (first, last) in protocol["design"]["mother_seeds"].items():
        collected = {key: [] for key in ("raw_adc", "target", "emission", "color_temperature")}
        metadata = []
        for seed in range(first, last+1):
            temperature, amplitude, features = mother(seed, protocol["model"]["shape"], op.patterns.device)
            spectrum, _ = rad.curve(temperature)
            base = spectrum*amplitude
            for level in protocol["design"]["counts_800nm_mean_e"]:
                continuum_pre = base*(level/base[4].mean())
                continuum = op.object_blur(continuum_pre)
                reference_t, fitted_continuum, _, fit_residual = rad.fit(continuum)
                mixing_bias = continuum[:3]-fitted_continuum[:3]
                for condition, amounts in protocol["design"]["conditions"].items():
                    feature_pre = features*torch.tensor(amounts, device=features.device)[:, None, None]*level
                    emission = op.object_blur(feature_pre)
                    pre = continuum_pre.clone()
                    pre[:3] += feature_pre
                    target = op.object_blur(pre)
                    expectation = op.encode_images(target)
                    if float(expectation.min()) <= 0:
                        raise ValueError("Synthetic sensor expectation must be positive")
                    for repeat in range(protocol["design"]["noise_repeats"]):
                        noise_seed = protocol["design"]["noise_seed"]+sample_id*1009
                        generator = torch.Generator(device=features.device).manual_seed(noise_seed)
                        electrons = torch.poisson(expectation, generator=generator)+protocol["model"]["read_noise_e"]*torch.randn(expectation.shape, device=features.device, generator=generator)
                        dn = torch.round(electrons/q+protocol["model"]["black_level_DN"])
                        saturated = int(((dn < 0) | (dn > 2**protocol["model"]["adc_bits"]-1)).sum())
                        if saturated:
                            raise ValueError("Registered pilot contains ADC saturation")
                        collected["raw_adc"].append(dn.to(torch.uint16).cpu())
                        collected["target"].append(target.cpu())
                        collected["emission"].append(emission.cpu())
                        collected["color_temperature"].append(reference_t.cpu())
                        metadata.append({"sample": sample_id, "mother": seed, "condition": condition, "level_e": level, "repeat": repeat, "noise_seed": noise_seed,
                                         "optical_image_L2": ((target-pre).square().sum((-2, -1))/pre.square().sum((-2, -1))).sqrt().cpu().tolist(),
                                         "color_vs_latent_temperature_RMSE_K": float((reference_t-temperature).square().mean().sqrt()),
                                         "noiseless_thermal_fit_residual_mean": float(fit_residual.mean()),
                                         "noiseless_continuum_extrapolation_RMSE_e": mixing_bias.square().mean((-2, -1)).sqrt().cpu().tolist()})
                        sample_id += 1
        data = {key: torch.stack(value) for key, value in collected.items()}
        data["metadata"] = metadata
        path = folder/(split+".pt")
        torch.save(data, path)
        receipt["splits"][split] = {"mothers": last-first+1, "observations": len(metadata), "file": str(path.relative_to(output)), "sha256": runtime.sha256(path), "bytes": path.stat().st_size}
        receipt["total_raw_observations"] += len(metadata)
        print({"stage": "data_saved", "split": split, "observations": len(metadata)}, flush=True)
    runtime.write_json(output/"DATA_RECEIPT.json", receipt)
    return receipt
