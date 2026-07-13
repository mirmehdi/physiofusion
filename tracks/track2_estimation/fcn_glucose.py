"""
1D Fully Convolutional Network (FCN) for 30-min glucose forecasting.
Architecture from Wang et al. 2016, adapted from classification to regression.

Problem: given 24 past CGM readings (2 hours), predict glucose 30 min ahead.
Evaluated on held-out subjects (leakage-proof split).
"""


from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from physiofusion.windowing import build_dataset
from physiofusion.splits import subject_grouped_split
'''
Data preparation
section 1 - 3
'''
# ---------------------------------------------------------------------------
# 1. Load windows and make one leakage-proof split (12 train / 3 test subjects)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
BIG = ROOT / "Data" / "big-ideas-lab-glycemic-variability-and-wearable-device-data-1.1.2"
SUBJECTS = [f"{i:03d}" for i in range(1, 17) if i != 15]

X, y, groups = build_dataset(BIG, SUBJECTS)
train_idx, test_idx = subject_grouped_split(groups, n_splits=5, fold=0)

X_train, y_train = X[train_idx], y[train_idx]
X_test,  y_test  = X[test_idx],  y[test_idx]

# ---------------------------------------------------------------------------
# 2. Normalize (z-score). Fit mean/std on TRAIN ONLY, then apply to both.
#    Neural nets train poorly on raw 40-260 values; standardizing to mean 0,
#    std 1 makes gradients well-behaved.
# ---------------------------------------------------------------------------
mean = X_train.mean()
std  = X_train.std()
X_train = (X_train - mean) / std
X_test  = (X_test  - mean) / std

# ---------------------------------------------------------------------------
# 3. To tensors, and reshape to (batch, channels, length).
#    A Conv1D expects a channel dimension. We have ONE signal (glucose),
#    so channels = 1 and length = 24.
# ---------------------------------------------------------------------------
def to_tensor(X, y):
    Xt = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (N, 24) -> (N, 1, 24)
    yt = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # (N,)    -> (N, 1)
    # .unsqueeze(1) is the important line — it inserts the channel axis, turning 
    #(N, 24) into (N, 1, 24), which is what Conv1D needs.
    return Xt, yt

X_train_t, y_train_t = to_tensor(X_train, y_train)
X_test_t,  y_test_t  = to_tensor(X_test,  y_test)

print("input shape:", X_train_t.shape)   # (n_train, 1, 24)

# ---------------------------------------------------------------------------
# 4. The FCN. Three conv blocks (Conv1d -> BatchNorm -> ReLU), then Global
#    Average Pooling, then one linear output.
#
#    - Conv1d(in, out, kernel, padding='same'): slides 'out' filters of the
#      given width over the signal, detecting local shapes. padding='same'
#      keeps length = 24 so blocks stack cleanly.
#    - BatchNorm1d: normalizes each layer's output -> stable, faster training.
#    - ReLU: nonlinearity, lets the net learn non-straight-line patterns.
#    - AdaptiveAvgPool1d(1): Global Average Pooling -> averages each filter
#      across time into a single number. Far fewer params than flatten.
#    - Linear(128, 1): maps the 128 pooled features to one prediction.
# ---------------------------------------------------------------------------
class FCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(1,   128, kernel_size=7, padding="same"),
            nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=5, padding="same"),
            nn.BatchNorm1d(256), nn.ReLU(),
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=3, padding="same"),
            nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)     # Global Average Pooling
        self.head = nn.Linear(128, 1)          # regression output: 1 number

    def forward(self, x):                      # x: (batch, 1, 24)
        x = self.block1(x)                     # -> (batch, 128, 24)
        x = self.block2(x)                     # -> (batch, 256, 24)
        x = self.block3(x)                     # -> (batch, 128, 24)
        x = self.gap(x).squeeze(-1)            # -> (batch, 128)
        return self.head(x)                    # -> (batch, 1)

model = FCN()
print(model)

# ---------------------------------------------------------------------------
# 5. Training setup: loss, optimizer, batching.
#    - MSELoss: mean squared error, the regression loss.
#    - Adam: the optimizer that updates weights from gradients.
#    - DataLoader: serves the data in shuffled mini-batches (256 at a time).
# ---------------------------------------------------------------------------
from torch.utils.data import TensorDataset, DataLoader

train_ds = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 30

# ---------------------------------------------------------------------------
# 6. The training loop. Each epoch = one full pass over the training data.
#    For each batch: predict -> measure loss -> backprop -> update weights.
# ---------------------------------------------------------------------------
for epoch in range(EPOCHS):
    model.train()                     # training mode (BatchNorm uses batch stats)
    epoch_loss = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()         # clear old gradients
        pred = model(xb)              # forward pass: predict
        loss = criterion(pred, yb)    # how wrong were we?
        loss.backward()               # backprop: compute gradients
        optimizer.step()              # update weights
        epoch_loss += loss.item() * len(xb)
    epoch_loss /= len(train_ds)

    # quick check on held-out test each epoch (eval mode: BatchNorm frozen)
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test_t)
        test_rmse = torch.sqrt(criterion(test_pred, y_test_t)).item()

    print(f"epoch {epoch+1:2d}  train_MSE {epoch_loss:7.2f}  test_RMSE {test_rmse:6.2f}")

# ---------------------------------------------------------------------------
# 7. Final scores on held-out subjects, same RMSE/MAE as your other models.
# ---------------------------------------------------------------------------
model.eval()
with torch.no_grad():
    pred = model(X_test_t).squeeze(1).numpy()

rmse = np.sqrt(np.mean((y_test - pred) ** 2))
mae  = np.mean(np.abs(y_test - pred))
print(f"\nFCN  ->  RMSE {rmse:.2f}   MAE {mae:.2f}")
print(f"(bar to beat: Linear 13.90)")