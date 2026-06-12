import torch
from .power_lookup_tables import load_table
from .grid_interpolation import clamp_index


def estimate_energy(board, layer, in_dim, out_dim):

    grid_x, grid_y, values = load_table(board, layer)

    if not isinstance(in_dim, torch.Tensor):
        in_dim = torch.tensor(float(in_dim), dtype=torch.float64)
    if not isinstance(out_dim, torch.Tensor):
        out_dim = torch.tensor(float(out_dim), dtype=torch.float64)


    lo_x, hi_x, tx = clamp_index(grid_x, in_dim)
    lo_y, hi_y, ty = clamp_index(grid_y, out_dim)

    Q11 = values[lo_x, lo_y]
    Q12 = values[lo_x, hi_y]
    Q21 = values[hi_x, lo_y]
    Q22 = values[hi_x, hi_y]

    return (
        Q11 * (1 - tx) * (1 - ty)
        + Q12 * (1 - tx) * ty
        + Q21 * tx * (1 - ty)
        + Q22 * tx * ty
    )