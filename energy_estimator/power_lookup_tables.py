from pathlib import Path
import torch
import torch.nn as nn
import pandas as pd

_TABLE_CACHE = {}

BASE_DIR = Path(__file__).parent / "data"


def resolve_layer_key(module: nn.Module) -> str | None:
    if isinstance(module, nn.Linear):
        return "linear"

    if isinstance(module, nn.Conv2d):
        k = module.kernel_size
        p = module.padding
        k = k[0] if isinstance(k, tuple) else k
        p = p[0] if isinstance(p, tuple) else p
        return f"conv_k{k}_p{p}"

    return None


def list_boards() -> list[str]:
    return [d.name for d in BASE_DIR.iterdir() if d.is_dir()]


def list_layer_keys(board: str) -> list[str]:
    board_dir = BASE_DIR / board
    return [
        f.stem.replace("_power_report", "")
        for f in sorted(board_dir.glob("*_power_report.xlsx"))
    ]


def xlsx_path(board: str, layer_key: str) -> Path:
    return BASE_DIR / board / f"{layer_key}_power_report.xlsx"


def load_table(board: str, layer_key: str):
    key = (board, layer_key)

    if key in _TABLE_CACHE:
        return _TABLE_CACHE[key]

    path = xlsx_path(board, layer_key)

    if not path.exists():
        available = list_layer_keys(board)
        raise FileNotFoundError(
            f"No table for layer '{layer_key}' on board '{board}'. "
            f"Available: {available}"
        )

    df = pd.read_excel(path, sheet_name="Average Power Matrix", index_col=0)
    df.index = df.index.astype(float)
    df.columns = df.columns.astype(float)
    df = df.sort_index(axis=0).sort_index(axis=1)

    grid_x = torch.tensor(df.index.values, dtype=torch.float64)
    grid_y = torch.tensor(df.columns.values, dtype=torch.float64)
    values = torch.tensor(df.values, dtype=torch.float64)
    
    
    # Prepend 0 to both axes: 0 channels → 0 energy (physical anchor for pruned layers)
    zero_col = torch.zeros(values.shape[0], 1, dtype=torch.float64)
    values = torch.cat([zero_col, values], dim=1)
    grid_y = torch.cat([torch.zeros(1, dtype=torch.float64), grid_y])

    zero_row = torch.zeros(1, values.shape[1], dtype=torch.float64)
    values = torch.cat([zero_row, values], dim=0)
    grid_x = torch.cat([torch.zeros(1, dtype=torch.float64), grid_x])
    
    
    
    
    

    _TABLE_CACHE[key] = (grid_x, grid_y, values)
    return grid_x, grid_y, values


def load_all_tables(board: str) -> dict[str, tuple]:
    return {key: load_table(board, key) for key in list_layer_keys(board)}
