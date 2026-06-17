import torch
import torch.nn as nn
import energy_estimator as ee


# ── 1. Discovery ─────────────────────────────────────────────────────────────

print("=" * 60)
print("TEST 1 — available boards and layers")
print("=" * 60)

boards = ee.list_boards()
print(f"boards : {boards}")

for board in boards:
    keys = ee.list_layer_keys(board)
    print(f"  {board} → {keys}")


# ── 2. Linear-only model (CoralDevBoard) ─────────────────────────────────────

print("\n" + "=" * 60)
print("TEST 2 — linear model on CoralDevBoard")
print("=" * 60)

class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

model_linear = LinearModel()

results = ee.estimate_model_energy(model_linear, board="CoralDevBoard")

for name, info in results["layers"].items():
    print(f"  {name:6s} | {info['layer_key']:10s} | in={info['in']:.0f} out={info['out']:.0f} | energy={info['energy']:.6f} J  ({info['energy_mj']:.4f} mJ)")

print(f"\n  TOTAL energy : {float(results['total_energy']):.6f} J")


# ── 3. Mixed conv + linear model (JetsonNano) ────────────────────────────────

print("\n" + "=" * 60)
print("TEST 3 — mixed Conv2d + Linear model on JetsonNano")
print("=" * 60)

class MixedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,  32, kernel_size=3, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=0)
        self.fc1   = nn.Linear(64, 128)
        self.fc2   = nn.Linear(128, 10)

model_mixed = MixedModel()

results = ee.estimate_model_energy(model_mixed, board="JetsonNano")

for name, info in results["layers"].items():
    print(f"  {name:6s} | {info['layer_key']:12s} | in={info['in']:.0f} out={info['out']:.0f} | energy={info['energy']:.6f} J  ({info['energy_mj']:.4f} mJ)")

print(f"\n  TOTAL energy : {float(results['total_energy']):.6f} J")


# ── 4. Differentiable pruning (soft mask + backward) ─────────────────────────

print("\n" + "=" * 60)
print("TEST 4 — differentiable pruning with soft mask")
print("=" * 60)

scores = torch.randn(64, dtype=torch.float64, requires_grad=True)
mask   = torch.sigmoid(scores)   # soft mask : values in (0, 1)

masks = {"fc1": mask}

results = ee.estimate_model_energy(model_mixed, board="JetsonNano", masks=masks)

results["total_energy"].backward()

print(f"  total energy         : {float(results['total_energy']):.6f} J")
print(f"  scores.grad is None  : {scores.grad is None}")
print(f"  scores.grad shape    : {scores.grad.shape}")
print(f"  scores.grad sample   : {scores.grad[:5]}")
print("\n  gradient flows correctly through the interpolation.")
