import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from joblib import Parallel, delayed
from tqdm import tqdm

# === 参数设置 ===
feature_npy = 'normalized_features_deduped.npy'
label_dir = 'visual_output'
dims = [6, 8]
ks = [4, 5, 6, 7, 8]
sample_threshold = 200000000  # 超过这个数量时启用采样
sample_size = 500000        # 采样数量

# === 加载特征数据 ===
X = np.load(feature_npy)

# === 计算函数（支持采样） ===
def compute_silhouette_score(X_pca, labels, dim, k):
    if len(np.unique(labels)) < 2:
        return None
    try:
        if len(X_pca) > sample_threshold:
            idx = np.random.choice(len(X_pca), size=sample_size, replace=False)
            score = silhouette_score(X_pca[idx], labels[idx])
        else:
            score = silhouette_score(X_pca, labels)
        return (dim, k, score)
    except:
        return None

# === 主程序 ===
best_score = -1
best_config = None
all_scores = []

for dim in dims:
    print(f"\n🌀 正在执行 PCA 降维至 {dim} 维...")
    pca = PCA(n_components=dim)
    X_pca = pca.fit_transform(X)

    print(f"🔎 分析 K 值组合: {ks}")
    results = Parallel(n_jobs=-1)(
        delayed(
            lambda k: compute_silhouette_score(
                X_pca,
                pd.read_csv(os.path.join(label_dir, f'labels_pca{dim}_k{k}.csv'))['label'].values
                if os.path.exists(os.path.join(label_dir, f'labels_pca{dim}_k{k}.csv')) else np.array([]),
                dim,
                k
            )
        )(k) for k in tqdm(ks, desc=f"PCA-{dim}")
    )

    for result in results:
        if result:
            dim, k, score = result
            all_scores.append(result)
            if score > best_score:
                best_score = score
                best_config = (dim, k)

# === 输出最终结果 ===
print("\n📊 所有组合的 Silhouette 分数：")
for dim, k, score in all_scores:
    print(f"  PCA-{dim}, K={k} -> Silhouette Score = {score:.4f}")

if best_config:
    print(f"\n✅ 最佳组合：PCA-{best_config[0]}, K={best_config[1]}，轮廓系数 = {best_score:.4f}")
else:
    print("\n⚠️ 没有找到有效的组合，请检查标签文件是否存在且聚类数大于1。")
