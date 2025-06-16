# mc_dropout_mlp_fullgrid.py

import json
import pickle
import numpy as np
import pandas as pd
import os
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import scipy.stats
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression

# ---------- Parameters ----------
SHOT = "-1"
FOLD = "0"
FEATURE_PATH = "/hpc/group/engelhardlab/xg97/EHRSHOT_ASSETS/features/clmbr_features.pkl"
BENCHMARK_ROOT = "/hpc/group/engelhardlab/xg97/EHRSHOT_ASSETS/benchmark"
OUTPUT_CSV = "/hpc/group/engelhardlab/xg97/EHRSHOT_ASSETS/results/mc_dropout_grid_results.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Load CLMBR features ----------
with open(FEATURE_PATH, "rb") as f:
    feats = pickle.load(f)

X_all = feats["data_matrix"]
pat_global = feats["patient_ids"]
time_global = feats["labeling_time"]
y_global = feats["label_values"]
time_global_str = np.array([t.isoformat() for t in time_global])
key_global = pd.Series(np.arange(len(pat_global)),
                       index=pd.MultiIndex.from_arrays([pat_global, time_global_str]))

# ---------- Define MLP model ----------
class MLP(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return torch.sigmoid(self.fc2(x)).view(-1)

    def enable_mc_dropout(self):
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

# ---------- MC Dropout Inference ----------
def mc_dropout_predict(model, X_tensor, T=30):
    model.eval()
    model.enable_mc_dropout()
    preds = []
    with torch.no_grad():
        for _ in range(T):
            out = model(X_tensor).cpu().numpy()
            preds.append(out)
    preds = np.stack(preds, axis=0)  # [T, N]
    mean_pred = np.mean(preds, axis=0)
    entropy_total = scipy.stats.entropy([mean_pred, 1 - mean_pred], base=2, axis=0)
    entropy_each = scipy.stats.entropy(np.stack([preds, 1 - preds], axis=2), base=2, axis=2)
    entropy_expected = np.mean(entropy_each, axis=0)
    epistemic = entropy_total - entropy_expected
    return mean_pred, entropy_total, entropy_expected, epistemic

# ---------- Safe lookup ----------
def lookup_rows_safe(pids, times, labels):
    pairs = list(zip(pids, times))
    valid_pairs = [pair for pair in pairs if pair in key_global]
    valid_indices = [i for i, pair in enumerate(pairs) if pair in key_global]
    if not valid_pairs:
        raise ValueError("No valid (pid, time) pairs found in features.")
    X_idx = key_global.loc[valid_pairs].values.astype(int)
    y_filtered = labels[valid_indices]
    return X_idx, y_filtered

# ---------- Grid Evaluation ----------
dropout_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
T_values = [10, 20, 30, 40, 50, 100]
results = []

task_dirs = sorted([
    name for name in os.listdir(BENCHMARK_ROOT)
    if os.path.isdir(os.path.join(BENCHMARK_ROOT, name)) and name != "chexpert"
])

for dropout in dropout_rates:
    for T in T_values:
        for task in tqdm(task_dirs, desc=f"Dropout={dropout}, T={T}"):
            try:
                with open(os.path.join(BENCHMARK_ROOT, task, "all_shots_data.json"), "r") as f:
                    fold = json.load(f)[task][SHOT][FOLD]

                train_idx, y_train = lookup_rows_safe(
                    np.array(fold["patient_ids_train_k"], dtype=int),
                    np.array(fold["label_times_train_k"]),
                    np.array(fold["label_values_train_k"]))

                val_idx, y_val = lookup_rows_safe(
                    np.array(fold["patient_ids_val_k"], dtype=int),
                    np.array(fold["label_times_val_k"]),
                    np.array(fold["label_values_val_k"]))

                X_train, X_val = X_all[train_idx], X_all[val_idx]

                model = MLP(dropout=dropout).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                loss_fn = nn.BCELoss()

                train_loader = DataLoader(
                    TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(y_train, dtype=torch.float32)),
                    batch_size=64, shuffle=True)

                model.train()
                for epoch in range(10):
                    for xb, yb in train_loader:
                        xb, yb = xb.to(device), yb.to(device)
                        optimizer.zero_grad()
                        loss = loss_fn(model(xb), yb)
                        loss.backward()
                        optimizer.step()

                # Inference with MC Dropout
                X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
                mean_pred, H_total, H_alea, H_epi = mc_dropout_predict(model, X_val_tensor, T=T)

                # Calibration Metrics
                ece = np.abs(np.array(calibration_curve(y_val, mean_pred, n_bins=10)).diff()).mean()
                brier = np.mean((mean_pred - y_val) ** 2)
                lr_cal = LogisticRegression(solver='lbfgs')
                lr_cal.fit(mean_pred.reshape(-1, 1), y_val)
                slope = lr_cal.coef_[0][0]
                intercept = lr_cal.intercept_[0]

                results.append({
                    "Task": task,
                    "Dropout Rate": dropout,
                    "MC Samples": T,
                    "Val AUROC": roc_auc_score(y_val, mean_pred),
                    "Val AUPRC": average_precision_score(y_val, mean_pred),
                    "ECE": ece,
                    "Brier": brier,
                    "Slope": slope,
                    "Intercept": intercept,
                    "Mean Epistemic": float(np.mean(H_epi)),
                    "Mean Aleatoric": float(np.mean(H_alea)),
                    "Total Uncertainty": float(np.mean(H_total)),
                })

            except Exception as e:
                print(f"❌ Failed: {task} @ Dropout={dropout}, T={T} — {e}")

# ---------- Save all results ----------
df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Saved to {OUTPUT_CSV}")
