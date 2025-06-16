
import json
import pickle
import numpy as np
import pandas as pd
import os
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import scipy.stats

SHOT = "-1"
FOLD = "0"
FEATURE_PATH = "/hpc/group/engelhardlab/xg97/EHRSHOT_ASSETS/features/clmbr_features.pkl"
BENCHMARK_ROOT = "/hpc/group/engelhardlab/xg97/EHRSHOT_ASSETS/benchmark"
OUTPUT_CSV = "/hpc/group/engelhardlab/xg97/EHRSHOT_ASSETS/results/mc_dropout_with_uncertainty_metrics_n_included.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(FEATURE_PATH, "rb") as f:
    feats = pickle.load(f)

X_all = feats["data_matrix"]
pat_global = feats["patient_ids"]
time_global = feats["labeling_time"]
y_global = feats["label_values"]
time_global_str = np.array([t.isoformat() for t in time_global])
key_global = pd.Series(np.arange(len(pat_global)),
                       index=pd.MultiIndex.from_arrays([pat_global, time_global_str]))

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

def mc_dropout_predict(model, X_tensor, T=30):
    model.eval()
    model.enable_mc_dropout()
    preds = []
    with torch.no_grad():
        for _ in range(T):
            out = model(X_tensor).cpu().numpy()
            preds.append(out)
    preds = np.stack(preds, axis=0)
    mean_pred = np.mean(preds, axis=0)
    entropy_total = scipy.stats.entropy([mean_pred, 1 - mean_pred], base=2, axis=0)
    entropy_each = scipy.stats.entropy(np.stack([preds, 1 - preds], axis=2), base=2, axis=2)
    entropy_expected = np.mean(entropy_each, axis=0)
    epistemic = entropy_total - entropy_expected
    return mean_pred, entropy_total, entropy_expected, epistemic

def compute_calibration(y_true, y_prob):
    ece_bins = 10
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=ece_bins, strategy='uniform')
    ece = np.sum(np.abs(prob_pred - prob_true) * np.histogram(y_prob, bins=ece_bins)[0] / len(y_true))
    brier = brier_score_loss(y_true, y_prob)
    clf = LogisticRegression().fit(y_prob.reshape(-1, 1), y_true)
    slope = clf.coef_[0][0]
    intercept = clf.intercept_[0]
    return ece, brier, slope, intercept

results = []
drop_percentages = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55,60, 65,70,75,80.85,90,95,100]
task_dirs = sorted([
    name for name in os.listdir(BENCHMARK_ROOT)
    if os.path.isdir(os.path.join(BENCHMARK_ROOT, name)) and name != "chexpert"
])

for task in tqdm(task_dirs, desc="Evaluating Tasks"):
    try:
        with open(os.path.join(BENCHMARK_ROOT, task, "all_shots_data.json"), "r") as f:
            fold = json.load(f)[task][SHOT][FOLD]

        def lookup_rows_safe(pids, times, labels):
            pairs = list(zip(pids, times))
            valid_pairs = [pair for pair in pairs if pair in key_global]
            valid_indices = [i for i, pair in enumerate(pairs) if pair in key_global]
            if not valid_pairs:
                raise ValueError("No valid (pid, time) pairs found in features.")
            X_idx = key_global.loc[valid_pairs].values.astype(int)
            y_filtered = labels[valid_indices]
            return X_idx, y_filtered

        train_idx, y_train = lookup_rows_safe(
            np.array(fold["patient_ids_train_k"], dtype=int),
            np.array(fold["label_times_train_k"]),
            np.array(fold["label_values_train_k"]))

        val_idx, y_val = lookup_rows_safe(
            np.array(fold["patient_ids_val_k"], dtype=int),
            np.array(fold["label_times_val_k"]),
            np.array(fold["label_values_val_k"]))

        X_train, X_val = X_all[train_idx], X_all[val_idx]

        train_n = len(y_train)
        val_n = len(y_val)

        model = MLP().to(device)
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

        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
        mean_pred, H_total, H_alea, H_epi = mc_dropout_predict(model, X_val_tensor)

        for drop_pct in drop_percentages:
            k = int(len(H_epi) * drop_pct / 100)
            keep_idx = np.argsort(H_epi)[k:]
            y_keep = y_val[keep_idx]
            pred_keep = mean_pred[keep_idx]

            ece, brier, slope, intercept = compute_calibration(y_keep, pred_keep)

            results.append({
                "Task": task,
                "Drop %": drop_pct,
                "Val AUROC": roc_auc_score(y_keep, pred_keep),
                "Val AUPRC": average_precision_score(y_keep, pred_keep),
                "ECE": ece,
                "Brier": brier,
                "Slope": slope,
                "Intercept": intercept,
                "Mean Epistemic": float(np.mean(H_epi[keep_idx])),
                "Mean Aleatoric": float(np.mean(H_alea[keep_idx])),
                "Dropout Rate": 0.3,
                "MC Samples": 30,
                "Train N": train_n,
                "Val N": val_n
            })
    except Exception as e:
        print(f"❌ Failed: {task} — {e}")

df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Results saved to {OUTPUT_CSV}")
