 import os
  import glob
  import numpy as np
  import cupy as cp
  from cuml.preprocessing import StandardScaler as cuStandardScaler
  from cuml.decomposition import PCA as cuPCA
  from cuml.cluster import KMeans as cuKMeans
>>import matplotlib.pyplot as plt
⚠ import tqdm as tqdm


  def load_features(input_dir):
      files = glob.glob(os.path.join(input_dir, '*_RG.npy'))
      # glob.glob(os.path.join(input_dir, '*_BG.npy'))
      arrays = []
      for f in files:
          data = np.load(f)
          features = data[:, :-3]  # 前18列是特征
          arrays.append(features)
      return np.vstack(arrays)

  # 1. 加载并转到 GPU
  input_dir = 'dataset/rg'
  F_cpu = load_features(input_dir)
  F_gpu = cp.asarray(F_cpu)

  # 2. GPU 上标准化
  scaler = cuStandardScaler()
  F_std_gpu = scaler.fit_transform(F_gpu)

  # 3. GPU 上 PCA 降维
  pca = cuPCA(n_components=8)
  F_pca_gpu = pca.fit_transform(F_std_gpu)

  # 4. GPU 上 K-means (Elbow 计算)
  Ks = list(range(2, 11))
  inertias = []
  for k in Ks:
      print(f"Processing {k} clusters!")
      km = cuKMeans(n_clusters=k, random_state=0)
      km.fit(F_pca_gpu)
      inertias.append(km.inertia_)

  # 5. 绘制 Elbow 图
  inertias_cpu = cp.asnumpy(cp.array(inertias))
  plt.figure()
  plt.plot(Ks, inertias_cpu, marker='o')
  plt.xlabel('Number of clusters K')
  plt.ylabel('WCSS (Inertia)')
  plt.title('Elbow Method (GPU Acceleration)')
  plt.savefig("clusters.png")
