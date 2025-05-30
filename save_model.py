import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.optimize import linear_sum_assignment

# === 参数配置 ===
LABEL_CSV = 'visual_output_simple/rawdata_with_label_pca6_k8.csv'
PCA_MEAN_PATH = 'npyfile/pca_mean.npy'
PCA_COMPONENTS_PATH = 'npyfile/pca_components.npy'
CENTROIDS_PATH = 'visual_output_simple/kmeans_centroids_pca6_k8.npy'
OUTPUT_MODEL_DIR = 'npyfile'
os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)

# === 读取 CSV 并跳过第一行 ===
df = pd.read_csv(LABEL_CSV, skiprows=1, header=None)

# 提取特征与标签
X_raw = df.iloc[:, :18].values.astype(np.float32)
labels = df.iloc[:, -1].values.astype(int)

# === 特殊归一化：每行和为 512 ===
row_sums = X_raw.sum(axis=1, keepdims=True)
X = (X_raw / row_sums) * 512

# === 加载 PCA 参数和质心 ===
pca_mean = np.load(PCA_MEAN_PATH)            # shape: (18,)
pca_components = np.load(PCA_COMPONENTS_PATH)  # shape: (6, 18)
centroids = np.load(CENTROIDS_PATH)          # shape: (k, 6)

# === 手动 PCA 投影 ===
X_centered = X - pca_mean                    # shape: (n, 18)
X_pca = X_centered @ pca_components.T        # shape: (n, 6)
print(X_pca[20,:])

# === 手动预测：使用距离投影公式 ‖x‖² + ‖c‖² − 2xᵀc
x_norms = np.sum(X_pca ** 2, axis=1, keepdims=True)           # shape: (n, 1)
c_norms = np.sum(centroids ** 2, axis=1, keepdims=True).T     # shape: (1, k)
dot_product = X_pca @ centroids.T                             # shape: (n, k)
dists_squared = x_norms + c_norms - 2 * dot_product
pred_labels = np.argmin(dists_squared, axis=1)

# === 匈牙利算法对齐标签 ===
def match_labels(true_labels, pred_labels):
    cm = confusion_matrix(true_labels, pred_labels)
    cost_matrix = -cm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    mapping = dict(zip(col_ind, row_ind))
    aligned_pred = np.vectorize(mapping.get)(pred_labels)
    return aligned_pred

aligned_pred = match_labels(labels, pred_labels)

# === 输出准确率 ===
acc = accuracy_score(labels, aligned_pred)
print(f"Clustering accuracy (Hungarian-aligned): {acc * 100:.2f}%")

# === 保存预测结果 ===
pd.DataFrame({'true_label': labels, 'predicted_label': aligned_pred}) \
  .to_csv(os.path.join(OUTPUT_MODEL_DIR, 'predictions.csv'), index=False)
