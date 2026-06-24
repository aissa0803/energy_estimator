# Algorithm: Energy Estimation via Bilinear Interpolation

## Overview

`energy_estimator` predicts the energy consumption of a PyTorch model on a target
hardware board without running inference on the device. It does so by interpolating
over pre-measured power tables — one table per (board, layer-type) pair — and summing
across all layers of the model.

The result is a differentiable scalar with respect to the mask tensors when provided.

---

## Pipeline

```
nn.Module
    │
    ▼
1. resolve_layer_key          ← identify layer type
    │
    ▼
2. apply mask (optional)      ← compute effective output dimension
    │
    ▼
3. load_table                 ← read & cache the Excel power grid
    │
    ▼
4. clamp_index (×2)           ← locate (in_dim, out_dim) in the grid
    │
    ▼
5. bilinear interpolation     ← estimate energy in Joules
    │
    ▼
6. accumulate total_energy    ← sum across all layers
```

---

## Step-by-step

### 1. Layer key resolution — `resolve_layer_key`

Each `nn.Module` is mapped to a string key that identifies which lookup table to use.

| Module | Key example |
|---|---|
| `nn.Linear` | `"linear"` |
| `nn.Conv2d(kernel=3, stride=1, padding=0, dilation=1)` | `"conv_k3_s1_p0_d1"` |
| Anything else | `None` (skipped) |

Conv2d parameters are read directly from the module attributes, so a single table
file name encodes the full convolution configuration.

---

### 2. Effective dimension with soft masks

When `masks` is provided, the **effective output dimension** of layer `name` is:

```
eff_out = masks[name].sum()
```

This is a soft count — a differentiable scalar — rather than a hard integer. When no
mask is supplied, `eff_out = out_features` (or `out_channels` for Conv2d).

The effective output of layer *i* becomes the **input dimension** of layer *i+1*,
propagating the pruning signal through the depth of the network.

---

### 3. Power lookup table — `load_table`

Tables are stored as Excel files under `energy_estimator/data/<board>/`:

```
data/
└── JetsonNano/
    ├── linear_power_report.xlsx
    ├── conv_k3_s1_p0_d1_power_report.xlsx
    └── ...
```

Each file contains a sheet named `"Average Power Matrix"` where:

- **rows** = input dimension breakpoints (e.g. 64, 128, 256, …)
- **columns** = output dimension breakpoints
- **cells** = measured average energy in Joules

On first access the table is read once and cached in `_TABLE_CACHE`. Subsequent
calls for the same `(board, layer_key)` pair hit the cache directly.

#### Zero anchor

A row of zeros and a column of zeros are prepended to the grid before caching:

```
grid_x = [0, 64, 128, ...]
grid_y = [0, 64, 128, ...]
values[0, :] = 0      ← 0 input channels → 0 energy
values[:, 0] = 0      ← 0 output channels → 0 energy
```

This provides a physical anchor that prevents extrapolation from diverging when a
mask approaches zero (fully pruned layer).

---

### 4. Grid index lookup — `clamp_index`

Given a 1-D sorted grid and a query value `x`, `clamp_index` returns:

- `lo` — index of the left neighbor
- `hi` — index of the right neighbor (`hi = lo + 1`)
- `t` — fractional position of `x` between `grid[lo]` and `grid[hi]`

```
t = (x - grid[lo]) / (grid[hi] - grid[lo])     ∈ [0, 1]
```

The index search (`torch.searchsorted`) runs inside `torch.no_grad()` so that
discrete index selection does not block gradient flow. Only `t` is differentiable
with respect to `x`.

If `x` falls outside the measured range it is **clamped** to the boundary for index
selection, but `t` is computed from the true (unclamped) value, giving linear
extrapolation at the edges.

---

### 5. Bilinear interpolation — `estimate_energy`

The four corners of the cell that brackets `(in_dim, out_dim)` are read from the
values grid:

```
Q11 = values[lo_x, lo_y]    Q12 = values[lo_x, hi_y]
Q21 = values[hi_x, lo_y]    Q22 = values[hi_x, hi_y]
```

Energy is then computed as the standard bilinear formula:

```
E = Q11·(1−tx)·(1−ty)
  + Q12·(1−tx)·  ty
  + Q21·  tx  ·(1−ty)
  + Q22·  tx  ·  ty
```

where `tx` and `ty` are the fractional positions along the input and output
dimension axes respectively.

If any corner value is `NaN` (no measurement near that point), a `ValueError` is
raised immediately with the exact coordinates and corner values.

---

### 6. Accumulation — `estimate_model_energy`

The function iterates over `model.named_children()` (direct children only, not
recursive) and accumulates:

```python
total_energy += estimate_energy(board, layer_key, in_dim, eff_out)
```

The returned dict has the shape:

```python
{
    "layers": {
        "layer_name": {
            "type":       "Linear" | "Conv2d",
            "layer_key":  "linear" | "conv_k3_s1_p0_d1" | ...,
            "in":         float,          # input dim used
            "out":        float,          # effective output dim
            "masked":     bool,           # True if a mask was applied
            "energy":     float,          # Joules
            "energy_mj":  float,          # millijoules
        },
        ...
    },
    "total_energy": torch.Tensor,         # scalar float64, differentiable
}
```

---

## Differentiability

The full pipeline is differentiable with respect to `masks`:

```
masks[name].sum()  →  eff_out  →  ty  →  bilinear formula  →  total_energy
```

Calling `total_energy.backward()` propagates gradients back to each mask tensor,
enabling energy-aware pruning without any device-side execution.

---

## Supported layer types

| Layer | Key pattern | Parameters encoded |
|---|---|---|
| `nn.Linear` | `linear` | — |
| `nn.Conv2d` | `conv_k{k}_s{s}_p{p}_d{d}` | kernel, stride, padding, dilation |

Layers of any other type are silently skipped.

---

## Module structure

Each file in `energy_estimator/` has a single responsibility. They form a strict
dependency chain — higher layers call lower ones, never the reverse.

```
__init__.py                  ← public API (re-exports only)
    │
    └── main.py              ← orchestration
            │
            ├── power_lookup_tables.py   ← file I/O & caching
            │
            └── layer_energy_interpolation.py  ← interpolation
                        │
                        └── grid_interpolation.py   ← index math
```

---

### `__init__.py`

**Role:** defines what `from energy_estimator import ...` exposes.

Contains only re-exports — no logic. If you add a new public function anywhere in
the package, register it here.

```python
from .main import estimate_model_energy
from .power_lookup_tables import list_boards, list_layer_keys, load_all_tables, load_table

__all__ = ["estimate_model_energy", "list_boards", "list_layer_keys", "load_all_tables", "load_table"]
```

---

### `main.py`

**Role:** top-level orchestration — the only file most users ever call directly.

| Function | What it does |
|---|---|
| `estimate_model_energy(model, board, masks)` | Iterates `model.named_children()`, calls `resolve_layer_key` and `estimate_energy` for each supported layer, accumulates `total_energy`. |

Call order inside the function:

```
for name, module in model.named_children():
    layer_key  = resolve_layer_key(module)          # power_lookup_tables.py
    energy     = estimate_energy(board, layer_key,  # layer_energy_interpolation.py
                                 in_dim, eff_out)
    total     += energy
```

This file imports from `power_lookup_tables` and `layer_energy_interpolation` only.

---

### `power_lookup_tables.py`

**Role:** everything related to finding, loading, and caching the Excel measurement files.

| Function | What it does |
|---|---|
| `resolve_layer_key(module)` | Maps an `nn.Module` to its table key string (e.g. `"conv_k3_s1_p0_d1"`), or `None` if unsupported. |
| `list_boards()` | Returns all board names by listing subdirectories of `data/`. |
| `list_layer_keys(board)` | Returns all layer keys available for a board by globbing `*_power_report.xlsx`. |
| `xlsx_path(board, layer_key)` | Builds the path `data/<board>/<layer_key>_power_report.xlsx`. |
| `load_table(board, layer_key)` | Reads the Excel file, prepends the zero anchor row/column, caches the result, returns `(grid_x, grid_y, values)`. |
| `load_all_tables(board)` | Convenience wrapper — calls `load_table` for every key on a board. |

`_TABLE_CACHE` is a module-level dict that persists for the lifetime of the Python
process. It prevents re-reading the same Excel file on repeated calls.

This file has **no dependency** on any other file in the package.

---

### `layer_energy_interpolation.py`

**Role:** given a loaded table, perform bilinear interpolation for one layer.

| Function | What it does |
|---|---|
| `estimate_energy(board, layer, in_dim, out_dim)` | Loads the table via `load_table`, calls `clamp_index` twice (once per axis), applies the bilinear formula, returns a scalar `float64` tensor. |

This is the only file that combines the grid lookup (`grid_interpolation.py`) with
the measurement data (`power_lookup_tables.py`).

Accepts `out_dim` as either a plain `float` or a `torch.Tensor` (e.g. `mask.sum()`)
so the output remains differentiable when masks are used.

---

### `grid_interpolation.py`

**Role:** pure index math — locating a value in a 1-D sorted grid.

| Function | What it does |
|---|---|
| `clamp_index(grid, x)` | Returns `(lo, hi, t)`: the two neighboring grid indices and the fractional weight between them. Index search runs inside `torch.no_grad()`; only `t` carries a gradient. |

This file has **no imports from the package** — it only uses `torch`. It can be
tested and reasoned about in complete isolation from the rest of the codebase.

---

## Extending the package

### Adding a new board

A board is just a folder inside `energy_estimator/data/`. No code changes are needed —
`list_boards()` and `load_table()` discover boards by scanning that directory.

**Steps:**

1. Create the folder:
   ```
   energy_estimator/data/<BoardName>/
   ```
   The folder name becomes the `board` argument passed to `estimate_model_energy`.

2. Add one Excel file per layer type you have measurements for (see format below).

3. Verify discovery:
   ```python
   from energy_estimator import list_boards
   print(list_boards())   # should include "<BoardName>"
   ```

---

### Adding measurements for a new layer type

A new layer type requires two things: an Excel file in the board folder, and a
branch in `resolve_layer_key` so the package knows which key to assign to that
module.

#### 1. Excel file format

File name: `<layer_key>_power_report.xlsx`
Sheet name: `Average Power Matrix` (exact string, case-sensitive)

Layout:

```
                 64      128      256      512     ...   ← output dim (columns)
          64   0.0012   0.0024   0.0048   0.0095
         128   0.0023   0.0047   0.0093   0.0187
         256   0.0045   0.0091   0.0182   0.0364
         512   0.0089   0.0179   0.0358   0.0715
         ...
          ↑
     input dim (index / rows)
```

- The index column and header row must be **numeric** (integers or floats).
- Values are **average energy in Joules** measured on the target board.
- Rows and columns do not need to be evenly spaced, but both axes must be
  strictly increasing.
- Leave a cell as `NaN` only if you have no measurement there. Any query that
  lands on a `NaN` corner will raise a `ValueError` at runtime.
- The more breakpoints you provide, the more accurate the interpolation. As a
  minimum, cover the range of dimensions you expect to encounter in practice.

**Example** — existing files for reference:

```
energy_estimator/data/JetsonNano/
├── linear_power_report.xlsx
├── conv_k3_s1_p0_d1_power_report.xlsx
├── conv_k3_s1_p1_d1_power_report.xlsx
├── conv_k5_s1_p0_d1_power_report.xlsx
└── conv_k5_s1_p1_d1_power_report.xlsx
```

#### 2. Register the layer key in `resolve_layer_key`

Open [energy_estimator/power_lookup_tables.py](../energy_estimator/power_lookup_tables.py)
and add a branch to `resolve_layer_key`:

```python
def resolve_layer_key(module: nn.Module) -> str | None:
    if isinstance(module, nn.Linear):
        return "linear"

    if isinstance(module, nn.Conv2d):
        k = module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size
        s = module.stride[0]      if isinstance(module.stride,      tuple) else module.stride
        p = module.padding[0]     if isinstance(module.padding,     tuple) else module.padding
        d = module.dilation[0]    if isinstance(module.dilation,    tuple) else module.dilation
        return f"conv_k{k}_s{s}_p{p}_d{d}"

    # ← add your layer here, returning a key that matches the Excel file name
    if isinstance(module, nn.ConvTranspose2d):
        k = module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size
        s = module.stride[0]      if isinstance(module.stride,      tuple) else module.stride
        return f"convtranspose_k{k}_s{s}"

    return None
```

The returned string must match the file name prefix exactly:
`<returned_key>_power_report.xlsx`.

#### 3. Verify

```python
from energy_estimator import list_layer_keys
print(list_layer_keys("JetsonNano"))   # should include your new key
```

---

### Checklist

| Task | What to do |
|---|---|
| New board, existing layer types | Add `data/<Board>/` folder with Excel files |
| Existing board, new layer type | Add Excel file + branch in `resolve_layer_key` |
| New board + new layer type | Both of the above |
| Denser measurements for accuracy | Replace the Excel file; cache clears on next import |

---

## Contact

| | |
|---|---|
| **Name** | Aissa ABDELAZIZ |
| **Role** | Machine Learning Research Engineer |
| **Institution** | Hi! Paris |
| **Email** | [aissa.abdelaziz@ip-paris.fr](mailto:aissa.abdelaziz@ip-paris.fr) |

**Supervisor**

| | |
|---|---|
| **Name** | Prof. Enzo Tartaglione |
| **Institution** | Telecom Paris |
