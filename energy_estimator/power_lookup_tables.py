from pathlib import Path
import pandas as pd
import torch

_TABLE_CACHE = {}

BASE_DIR = Path(__file__).parent / "data"

_LAYER_MAP = {
    "linear": "linear",
    "conv2d": "conv",
    "conv": "conv",
}


def resolve_layer_type(name: str):
    return _LAYER_MAP.get(name.lower())


def xlsx_path(board, layer):
    return BASE_DIR / board / f"{layer}_power_report.xlsx"


def load_table(board_type: str, layer_type: str):
    key = (board_type, layer_type)

    if key in _TABLE_CACHE:
        return _TABLE_CACHE[key]

    path = xlsx_path(board_type, layer_type)

    df = pd.read_excel(path, sheet_name="Average Power Matrix", index_col=0)
    df.index = df.index.astype(float)
    df.columns = df.columns.astype(float)

    df = df.sort_index(axis=0).sort_index(axis=1)

    grid_x = torch.tensor(df.index.values, dtype=torch.float64)
    grid_y = torch.tensor(df.columns.values, dtype=torch.float64)
    values = torch.tensor(df.values, dtype=torch.float64)

    _TABLE_CACHE[key] = (grid_x, grid_y, values)
    return grid_x, grid_y, values