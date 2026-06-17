import torch


def clamp_index(grid: torch.Tensor, x: torch.Tensor):

    with torch.no_grad():
        x_lookup = x.detach().clamp(grid[0], grid[-1])
        hi = torch.searchsorted(grid, x_lookup, right=False)
        hi = hi.clamp(1, len(grid) - 1)
        lo = hi - 1

    lo_val = grid[lo]
    hi_val = grid[hi]
    span = hi_val - lo_val

    t = torch.where(
        span == 0,
        torch.zeros_like(x),
        (x - lo_val) / span,
    )

    return lo, hi, t
