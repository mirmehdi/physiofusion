"""
TCN (Temporal Convolutional Network) for 30-min glucose forecasting.
Causal + dilated + residual 1D convolutions.

Problem: 24 past CGM readings (2h) -> glucose 30 min ahead.
Evaluated on held-out subjects (leakage-proof split).
"""
##########
# %% data prepration (similar to FCN)
#########
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from physiofusion.windowing import build_dataset
from physiofusion.splits import subject_grouped_split

# --- data + one leakage-proof split ---
ROOT = Path(__file__).resolve().parents[2]
BIG = ROOT / "Data" / "big-ideas-lab-glycemic-variability-and-wearable-device-data-1.1.2"
SUBJECTS = [f"{i:03d}" for i in range(1, 17) if i != 15]

X, y, groups = build_dataset(BIG, SUBJECTS)
train_idx, test_idx = subject_grouped_split(groups, n_splits=5, fold=0)
X_train, y_train = X[train_idx], y[train_idx]
X_test,  y_test  = X[test_idx],  y[test_idx]

# --- z-score normalize (train stats only) ---
mean, std = X_train.mean(), X_train.std()
X_train = (X_train - mean) / std
X_test  = (X_test  - mean) / std

# --- to tensors, shape (batch, channels=1, length=24) ---
def to_tensor(X, y):
    return (torch.tensor(X, dtype=torch.float32).unsqueeze(1),
            torch.tensor(y, dtype=torch.float32).unsqueeze(1))

X_train_t, y_train_t = to_tensor(X_train, y_train)
X_test_t,  y_test_t  = to_tensor(X_test,  y_test)


# %% The TCN 
# ---------------------------------------------------------------------------
# A causal, dilated residual block.
#   - Causal: we pad ONLY on the left by (kernel-1)*dilation, then crop the
#     extra tail off the right, so each output sees only past inputs.
#   - Dilated: dilation spreads the kernel to reach further back.
#   - Residual: we add the block's input back to its output (a shortcut).
#     If channel counts differ, a 1x1 conv matches shapes first.
# ---------------------------------------------------------------------------
class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation      # left padding for causality

        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size,
                               padding=self.pad, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size,
                               padding=self.pad, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()

        # 1x1 conv to match channels for the shortcut, only if needed
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def _causal(self, x, conv):
        out = conv(x)
        if self.pad:                 # drop the extra padded tail on the right
            out = out[..., :-self.pad]
        return out

    def forward(self, x):
        res = x if self.downsample is None else self.downsample(x)

        out = self.relu(self.bn1(self._causal(x, self.conv1)))
        out = self.relu(self.bn2(self._causal(out, self.conv2)))

        return self.relu(out + res)   # residual add


# ---------------------------------------------------------------------------
# The full TCN: stack blocks with doubling dilation (1,2,4,8) so the
# receptive field grows exponentially and covers the whole 24-step history.
# Then take the LAST timestep (it has seen everything) -> Linear -> 1 output.
# ---------------------------------------------------------------------------
class TCN(nn.Module):
    def __init__(self, channels=64, kernel_size=3, dilations=(1, 2, 4, 8)):
        super().__init__()
        layers = []
        in_ch = 1                              # one signal (glucose)
        for d in dilations:
            layers.append(TCNBlock(in_ch, channels, kernel_size, d))
            in_ch = channels
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Linear(channels, 1)

    def forward(self, x):                      # x: (batch, 1, 24)
        x = self.tcn(x)                        # -> (batch, channels, 24)
        x = x[..., -1]                         # last timestep -> (batch, channels)
        return self.head(x)                    # -> (batch, 1)

model = TCN()
n_params = sum(p.numel() for p in model.parameters())
print(model)
print(f"parameters: {n_params:,}")


# %% Training loop (same rhythm as FCN)
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t),
                          batch_size=256, shuffle=True)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
EPOCHS = 30

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)   # forward + loss
        loss.backward()                   # backprop
        optimizer.step()                  # update
        epoch_loss += loss.item() * len(xb)
    epoch_loss /= len(X_train_t)

    model.eval()
    with torch.no_grad():
        test_rmse = torch.sqrt(criterion(model(X_test_t), y_test_t)).item()
    print(f"epoch {epoch+1:2d}  train_MSE {epoch_loss:7.2f}  test_RMSE {test_rmse:6.2f}")


# %% final part to evaluate
model.eval()
with torch.no_grad():
    pred = model(X_test_t).squeeze(1).numpy()

rmse = np.sqrt(np.mean((y_test - pred) ** 2))
mae  = np.mean(np.abs(y_test - pred))
print(f"\nTCN  ->  RMSE {rmse:.2f}   MAE {mae:.2f}")
print("(bar: Linear 13.90, FCN 14.67)")