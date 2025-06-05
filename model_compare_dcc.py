# 执行模型比较框架：MLP, Residual MLP, Transformer，记录训练与验证集 AUROC/AUPRC 以评估过拟合

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

# ---------- 参数 ----------
SHOT = "-1"
FOLD = "0"
FEATURE_PATH = "/mnt/data/clmbr_features.pkl"
BENCHMARK_ROOT = "/mnt/data/benchmark"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- 1. 读取 CLMBR 特征 ----------
with open(FEATURE_PATH, "rb") as f:
    feats = pickle.load(f)

X_all = feats["data_matrix"]
pat_global = feats["patient_ids"]
time_global = feats["labeling_time"]
y_global = feats["label_values"]
time_global_str = np.array([t.isoformat() for t in time_global])
key_global = pd.Series(np.arange(len(pat_global)),
                       index=pd.MultiIndex.from_arrays([pat_global, time_global_str]))

# ---------- 2. 网络结构定义 ----------
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

class ResidualMLP(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(input_dim, 1)

    def forward(self, x):
        residual = x
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = residual + x
        return torch.sigmoid(self.out(x)).view(-1)

class TransformerClassifier(nn.Module):
    def __init__(self, seq_len=12, d_model=64, nhead=8, num_layers=2, dropout=0.3):
        super().__init__()
        assert seq_len * d_model == 768
        self.seq_len = seq_len
        self.d_model = d_model
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        B = x.size(0)
        x = x.view(B, self.seq_len, self.d_model)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.transformer(x)
        cls_out = x[:, 0]
        return torch.sigmoid(self.fc(cls_out)).view(-1)

# ---------- 3. 主评估流程 ----------
results = []

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

        def train_and_eval(model, name):
            model.to(device)
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

            model.eval()
            with torch.no_grad():
                train_probs = model(torch.tensor(X_train, dtype=torch.float32).to(device)).cpu().numpy()
                val_probs = model(torch.tensor(X_val, dtype=torch.float32).to(device)).cpu().numpy()

            results.append({
                "Task": task, "Model": name,
                "Train AUROC": roc_auc_score(y_train, train_probs),
                "Train AUPRC": average_precision_score(y_train, train_probs),
                "Val AUROC": roc_auc_score(y_val, val_probs),
                "Val AUPRC": average_precision_score(y_val, val_probs),
            })

        train_and_eval(MLP(), "MLP")
        train_and_eval(ResidualMLP(), "ResidualMLP")
        train_and_eval(TransformerClassifier(), "Transformer")

    except Exception as e:
        print(f"❌ Failed: {task} — {e}")

# ---------- 4. 输出结果 ----------
df_eval = pd.DataFrame(results)
df_eval.to_csv("nn_model_eval_results.csv", index=False)
print("✅ Evaluation results saved to nn_model_eval_results.csv")