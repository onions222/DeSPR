import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import faiss
from tqdm import tqdm

# === 配置参数 ===
input_npy = 'normalized_features_deduped.npy'  # 归一化后的特征文件
output_dir = 'visual_output'
#os.makedirs(output_dir, exist_ok=True)

pca_dims = [15,  16, 17]
cluster_range = list(range(4, 11))
batch_size = 100000  # 按批处理样本数

# === 工具函数 ===
def apply_pca(X, n_components):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(X), pca

def gpu_kmeans(X, k):
    d = X.shape[1]

    res = faiss.StandardGpuResources()
    res.setTempMemory(2048 * 1024 * 1024)

    opts = faiss.GpuClonerOptions()
    opts.useFloat16 = True
    opts.usePrecomputedTables = True
    opts.verbose = False

    kmeans = faiss.Kmeans(
        d=d,
        k=k,
        niter=20,
        verbose=False,
        gpu=True,
        nredo=5,
        max_points_per_centroid=200000,
        min_points_per_centroid=1
    )
    kmeans.gpu_res = res
    kmeans.gpu_cloner_opts = opts

    kmeans.train(X)
    D, I = kmeans.index.search(X, 1)
    return I.flatten(), D.flatten(), kmeans

def process_config(dim, k, X_raw):
    X_pca, _ = apply_pca(X_raw, dim)
    labels, dists, model = gpu_kmeans(X_pca, k)
    inertia = np.mean(dists)

    # 保存图像
    if dim >= 2:
         X_vis = X_pca[:, :2]
          plt.figure(figsize=(6, 5))
          plt.scatter(X_vis[:, 0], X_vis[:, 1], c=labels, cmap='tab10', s=5)
          plt.title(f"PCA-{dim} + KMeans-{k}, Inertia={inertia:.2f}")
          plt.xlabel("PC1")
          plt.ylabel("PC2")
          plt.tight_layout()
          plt.savefig(os.path.join(output_dir, f'pca{dim}_k{k}.png'))
          plt.close()

      # 保存CSV
      csv_path = os.path.join(output_dir, f'labels_pca{dim}_k{k}.csv')
      df = pd.DataFrame({
          'label': labels
      })
      df.to_csv(csv_path, index=False)

      return dim, k, inertia, labels

  def process_in_batches(X, batch_size):
      num_batches = int(np.ceil(len(X) / batch_size))
      return [X[i * batch_size: (i + 1) * batch_size] for i in range(num_batches)]

  # === 主流程 ===
  X_raw = np.load(input_npy)
  # X_raw = np.unique(X_raw)
  # print(f"-- Data shape: {X_raw.shape[0]}")
  # unique_data_path = 'normalized_features_deduped.npy'

  # np.save(unique_data_path, X_raw)
  results = {dim: {} for dim in pca_dims}

  all_results = []

  for dim in tqdm(pca_dims, desc="Processing PCA Dimensions"):
      for k in tqdm(cluster_range, leave=False, desc=f"KMeans for PCA-{dim}"):
          result = process_config(dim, k, X_raw)
          all_results.append(result)

  for dim, k, inertia, labels in all_results:
      results[dim][k] = {
          'inertia': inertia,
          'labels': labels
      }

# === 自动选择最佳配置 ===
  best_config = None
  lowest_inertia = float('inf')
  for dim in pca_dims:
      for k in cluster_range:
          inertia = results[dim][k]['inertia']
          if inertia < lowest_inertia:
              lowest_inertia = inertia
              best_config = (dim, k)

>>print(f"✅ Best config by inertia: PCA-{best_config[0]}, K={best_config[1]}, Inertia={lowest_inertia:.4f}")

  # t-SNE 可视化
>>best_dim, best_k = best_config
  X_pca_best, _ = apply_pca(X_raw, best_dim)
  labels = results[best_dim][best_k]['labels']

  print("Running t-SNE (may take time)...")
  X_tsne = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(X_pca_best)

  plt.figure(figsize=(6, 5))
  plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='tab10', s=5)
  plt.title(f"t-SNE Projection (PCA-{best_dim}, K={best_k})")
  plt.xlabel("t-SNE 1")
  plt.ylabel("t-SNE 2")
  plt.tight_layout()
  plt.savefig(os.path.join(output_dir, f'tsne_pca{best_dim}_k{best_k}.png'))
  plt.close()
