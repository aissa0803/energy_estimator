import torch


def clamp_index(grid: torch.Tensor, x: torch.Tensor):
    x = x.clamp(grid[0], grid[-1])

    with torch.no_grad():
        hi = torch.searchsorted(grid, x.detach(), right=False)
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
    print(f"lo: {lo}, hi: {hi}, t: {t}")

    return lo, hi, t