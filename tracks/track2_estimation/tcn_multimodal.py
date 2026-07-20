"""Early-fusion TCN: 5-channel sequences (glucose + wristband history) -> glucose +30min.
Same TCN as before; only in_channels changes 1 -> 5. That IS early fusion:
all channels stacked at the input, mixed from the first convolution.
"""
from pathlib import Path                                      # paths
import numpy as np                                            # arrays
import torch                                                  # pytorch
import torch.nn as nn                                         # layers
from torch.utils.data import TensorDataset, DataLoader        # batching
from physiofusion.splits import subject_grouped_split         # leakage-proof split

ROOT = Path(__file__).resolve().parents[2]                    # project root
P = ROOT / "Data" / "processed"                               # where .npy files live

# --- load the sequence dataset (instant — no rebuilding) ---
X = np.load(P / "multimodal_seq_X.npy")                       # (24348, 5, 24)
y = np.load(P / "multimodal_seq_y.npy")                       # (24348,)
groups = np.load(P / "multimodal_seq_groups.npy", allow_pickle=True)  # (24348,) subject tags
print("loaded:", X.shape, y.shape)                            # confirm shapes

# --- one leakage-proof split (12 train / 3 test subjects) ---
tr, te = subject_grouped_split(groups, n_splits=5, fold=0)    # split by subject
X_train, y_train = X[tr], y[tr]                               # training subjects' windows
X_test,  y_test  = X[te], y[te]                               # unseen subjects' windows

# --- normalize PER CHANNEL, using TRAIN stats only (no leakage) ---
# each channel has its own scale (glucose ~100, eda ~0.5, temp ~33), so we
# standardize each channel separately: mean/std over (windows, timesteps)
mean = X_train.mean(axis=(0, 2), keepdims=True)               # per-channel mean, shape (1,5,1)
std  = X_train.std(axis=(0, 2), keepdims=True)                # per-channel std,  shape (1,5,1)
X_train = (X_train - mean) / std                              # standardize train
X_test  = (X_test  - mean) / std                              # apply TRAIN stats to test

# --- to tensors (already in (batch, channels, length) layout — no reshape needed) ---
X_train_t = torch.tensor(X_train, dtype=torch.float32)        # (n, 5, 24)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)  # (n, 1)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32)        # (n, 5, 24)
y_test_t  = torch.tensor(y_test,  dtype=torch.float32).unsqueeze(1)  # (n, 1)

# ---------------------------------------------------------------------------
# The TCN — IDENTICAL to your glucose-only version except in_ch starts at 5.
# ---------------------------------------------------------------------------
class TCNBlock(nn.Module):                                    # causal + dilated + residual block
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation               # left padding for causality
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=self.pad, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)                     # stabilize training
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=self.pad, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)                     # stabilize training
        self.relu = nn.ReLU()                                 # nonlinearity
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None  # shortcut match

    def _causal(self, x, conv):                               # conv then drop the future tail
        out = conv(x)                                         # convolve
        if self.pad:                                          # remove right-side padding
            out = out[..., :-self.pad]                        # keeps output causal (past only)
        return out

    def forward(self, x):
        res = x if self.downsample is None else self.downsample(x)   # residual path
        out = self.relu(self.bn1(self._causal(x, self.conv1)))       # conv -> bn -> relu
        out = self.relu(self.bn2(self._causal(out, self.conv2)))     # conv -> bn -> relu
        return self.relu(out + res)                                  # residual add


class TCN(nn.Module):                                         # the full network
    def __init__(self, in_channels=5, channels=64, kernel_size=3, dilations=(1, 2, 4, 8)):
        super().__init__()
        layers = []                                           # collect blocks
        in_ch = in_channels                                   # ← 5 channels now (was 1) = EARLY FUSION
        for d in dilations:                                   # doubling dilation = exponential reach
            layers.append(TCNBlock(in_ch, channels, kernel_size, d))  # add a block
            in_ch = channels                                  # next block takes 'channels' inputs
        self.tcn = nn.Sequential(*layers)                     # stack the blocks
        self.head = nn.Linear(channels, 1)                    # regression output

    def forward(self, x):                                     # x: (batch, 5, 24)
        x = self.tcn(x)                                       # -> (batch, channels, 24)
        x = x[..., -1]                                        # last timestep (has seen everything)
        return self.head(x)                                   # -> (batch, 1)

model = TCN(in_channels=5)                                    # ← the ONLY real change
print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")  # model size

# --- training setup ---
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=256, shuffle=True)  # batches
criterion = nn.MSELoss()                                      # regression loss
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)     # optimizer
EPOCHS = 30                                                   # passes over the data

# --- training loop ---
for epoch in range(EPOCHS):                                   # each epoch = one full pass
    model.train()                                             # training mode
    epoch_loss = 0.0                                          # accumulate loss
    for xb, yb in train_loader:                               # each mini-batch
        optimizer.zero_grad()                                 # clear old gradients
        loss = criterion(model(xb), yb)                       # forward + loss
        loss.backward()                                       # backprop
        optimizer.step()                                      # update weights
        epoch_loss += loss.item() * len(xb)                   # weighted by batch size
    epoch_loss /= len(X_train_t)                              # average train loss

    model.eval()                                              # eval mode (BatchNorm frozen)
    with torch.no_grad():                                     # no gradients needed
        test_rmse = torch.sqrt(criterion(model(X_test_t), y_test_t)).item()  # held-out RMSE
    print(f"epoch {epoch+1:2d}  train_MSE {epoch_loss:7.2f}  test_RMSE {test_rmse:6.2f}")  # progress

# --- final score ---
model.eval()                                                  # eval mode
with torch.no_grad():                                         # no gradients
    pred = model(X_test_t).squeeze(1).numpy()                 # predictions on held-out subjects

rmse = np.sqrt(np.mean((y_test - pred) ** 2))                 # final RMSE
mae  = np.mean(np.abs(y_test - pred))                         # final MAE
print(f"\nEarly-fusion TCN (5ch) -> RMSE {rmse:.2f}   MAE {mae:.2f}")  # the answer
print("(bars: glucose-only linear 13.56, glucose-only TCN 14.66)")     # comparison