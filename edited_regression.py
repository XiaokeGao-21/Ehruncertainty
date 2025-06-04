import json, pickle
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# ---------- 参数 ----------
TASK  = "lab_thrombocytopenia"
SHOT  = "-1"          # -1 = full‑shot    | 1/2/4/... = k‑shot
FOLD  = "0"

FEAT_PATH = "D:/DUKE/LAB/Matt/femr_env/EHRSHOT_ASSETS/features/clmbr_features.pkl"
JSON_PATH = f"D:/DUKE/LAB/Matt/femr_env/EHRSHOT_ASSETS/benchmark/{TASK}/all_shots_data.json"

# ---------- 1. 读特征 ----------
with open(FEAT_PATH, "rb") as f:
    feats = pickle.load(f)
X_all      = feats["data_matrix"]          # (N,768)
pat_global = feats["patient_ids"]          # (N,)
time_global= feats["labeling_time"]        # (N,) Python datetime objects
y_global   = feats["label_values"]         # (N,) bool / 0‑1

# 把时间统一成 ISO 字符串，便于 hash 做映射
time_global_str = np.array([t.isoformat() for t in time_global])

# ---------- 2. 构造 (patient_id, time_iso) → 行号 的字典 ----------
key_global = pd.Series(np.arange(len(pat_global)),
                       index=pd.MultiIndex.from_arrays([pat_global, time_global_str]))
# 现在 key_global.loc[(pid, time)] 就能直接得到行号

# ---------- 3. 读 fold ----------
with open(JSON_PATH, "r") as f:
    fold = json.load(f)[TASK][SHOT][FOLD]

pids_train = np.array(fold["patient_ids_train_k"], dtype=int)
times_train= np.array(fold["label_times_train_k"])         # ISO‑8601 字符串
y_train    = np.array(fold["label_values_train_k"])

pids_val   = np.array(fold["patient_ids_val_k"], dtype=int)
times_val  = np.array(fold["label_times_val_k"])
y_val      = np.array(fold["label_values_val_k"])

# ---------- 4. 把 fold 行号映射到 global 行号 ----------
def lookup_rows(pids, times):
    idx = key_global.loc[list(zip(pids, times))].values
    assert not np.isnan(idx).any(), "某些 (pid,time) 在特征矩阵里找不到！"
    return idx.astype(int)

train_rows = lookup_rows(pids_train, times_train)
val_rows   = lookup_rows(pids_val,   times_val)

# ---------- 5. 取子矩阵 ----------
X_train = X_all[train_rows]
X_val   = X_all[val_rows]

# ---------- 6. 训练 & 评估 ----------
clf = LogisticRegression(
        max_iter=10000,
        class_weight="balanced",   # 实验里正例往往只有 5% - 15%，不加容易全预测 0
        penalty="l2",
        C=1.0,
        solver="lbfgs",
    ).fit(X_train, y_train)

y_pred_prob = clf.predict_proba(X_val)[:, 1]
print("val AUROC =", roc_auc_score(y_val, y_pred_prob))
print(classification_report(y_val, y_pred_prob > 0.5, digits=3))
