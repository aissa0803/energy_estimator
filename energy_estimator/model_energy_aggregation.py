import torch
import torch.nn as nn

from .layer_energy_interpolation import estimate_energy
from .power_lookup_tables import resolve_layer_key


def estimate_model_energy(model, board="CoralDevBoard", masks=None):
    if masks is None:
        masks = {}

    results = {
        "layers": {},
        "total_energy": torch.tensor(0.0, dtype=torch.float64),
    }

    total = torch.tensor(0.0, dtype=torch.float64)
    prev_eff_out = None

    for name, module in model.named_children():

        layer_key = resolve_layer_key(module)
        if layer_key is None:
            continue

        if isinstance(module, nn.Linear):
            in_dim = prev_eff_out if prev_eff_out is not None else module.in_features
            out_dim = module.out_features

        elif isinstance(module, nn.Conv2d):
            in_dim = prev_eff_out if prev_eff_out is not None else module.in_channels
            out_dim = module.out_channels

        else:
            continue

        if name in masks:
            eff_out = masks[name].sum()
            masked = True
        else:
            eff_out = out_dim
            masked = False

        energy = estimate_energy(board, layer_key, in_dim, eff_out)
        prev_eff_out = eff_out

        energy_mj = energy * 1000
        total = total + energy

        results["layers"][name] = {
            "type": type(module).__name__,
            "layer_key": layer_key,
            "in": float(in_dim),
            "out": eff_out.item() if isinstance(eff_out, torch.Tensor) else float(eff_out),
            "masked": masked,
            "energy": float(energy),
            "energy_mj": float(energy_mj),
        }

    results["total_energy"] = total
    return results
