import torch
import torch.nn as nn

from .layer_energy_interpolation import estimate_energy
from .power_lookup_tables import resolve_layer_type


def estimate_model_energy(model, board="CoralDevBoard", masks=None):
    if masks is None:
        masks = {}

    results = {
        "layers": {},
        "total_energy": torch.tensor(0.0, dtype=torch.float64),
    }

    total = torch.tensor(0.0, dtype=torch.float64)

    for name, module in model.named_children():

        layer_type = resolve_layer_type(type(module).__name__)
        if layer_type is None:
            continue

        # dimensions
        if isinstance(module, nn.Linear):
            in_features = module.in_features
            out_features = module.out_features

        elif isinstance(module, nn.Conv2d):
            in_features = module.in_channels
            out_features = module.out_channels

        else:
            continue

        # prunning mask 
        if name in masks:    
            eff_out = masks[name].sum()
            masked = True
            print(f"Layer {name} is masked. Effective output features: {eff_out}")
        else:
            eff_out = out_features
            masked = False
            

        # energy computation 
        energy = estimate_energy(
            board,
            layer_type,
            in_features,
            eff_out,
        )

        # total energy
        energy_mj = energy * 1000
        total = total + energy

        results["layers"][name] = {
            "type": type(module).__name__,
            "in": float(in_features),
            "out": float(eff_out) if not isinstance(eff_out, torch.Tensor) else float(eff_out),
            "masked": masked,
            "energy": float(energy),
            "energy_mj": float(energy_mj),
        }

    results["total_energy"] = total
    return results