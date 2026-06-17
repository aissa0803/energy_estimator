# Energy Estimator

A Python package for estimating the energy consumption of PyTorch neural network layers on embedded hardware boards (JetsonNano, CoralDevBoard).

Designed for **neural network pruning workflows**: the energy estimate is fully differentiable, so it can be used directly as a loss term inside a training or optimization loop — gradients flow through the energy value back to the pruning masks.

## Supported Hardware

| Board | Supported Layers |
|-------|-----------------|
| JetsonNano | `linear`, `conv_k3_p0`, `conv_k3_p1`, `conv_k5_p0`, `conv_k5_p1` |
| CoralDevBoard | `linear` |

## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/aissa0803/energy_estimator.git
```

Or clone and install in editable mode for development:

```bash
git clone https://github.com/aissa0803/energy_estimator.git
cd energy_estimator
pip install -e .
```

## Usage

### Estimate energy of a model

```python
import torch.nn as nn
import energy_estimator as ee

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 10)

model = MyModel()
results = ee.estimate_model_energy(model, board="CoralDevBoard")

for name, info in results["layers"].items():
    print(f"{name}: {info['energy_mj']:.4f} mJ")

print(f"Total: {float(results['total_energy']):.6f} J")
```

### Differentiable pruning (gradient through energy)

```python
import torch
import torch.nn as nn
import energy_estimator as ee

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 10)

model = MyModel()
scores = torch.randn(256, dtype=torch.float64, requires_grad=True)
mask = torch.sigmoid(scores)  # soft mask in (0, 1)

results = ee.estimate_model_energy(model, board="JetsonNano", masks={"fc1": mask})
results["total_energy"].backward()  # gradients flow back to scores
print(scores.grad)
```

### Discover available boards and layers

```python
import energy_estimator as ee

boards = ee.list_boards()
for board in boards:
    print(board, ee.list_layer_keys(board))
```

## API Reference

| Function | Description |
|----------|-------------|
| `estimate_model_energy(model, board, masks)` | Estimate total energy of a PyTorch model |
| `list_boards()` | List available hardware boards |
| `list_layer_keys(board)` | List supported layer types for a board |
| `load_table(board, layer_key)` | Load raw power lookup table |
| `load_all_tables(board)` | Load all tables for a board |
