import os, glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import faiss
from tqdm import tqdm
from sklearn.cluster import KMeans as SKKMeans


def normalize_to_sum512_batch(X):
    """按行将矩阵归一化，使每行元素和为 512"""
    sums = np.sum(X, axis=1, keepdims=True)
    scale = np.where(sums == 0, 1.0, 512.0 / sums)
    X_scaled = X * scale
    return np.round(X_scaled)


def preprocess_all(input_dir):
    files = glob.glob(os.path.join(input_dir, 'rg', '*_RG.npy')) + \
            glob.glob(os.path.join(input_dir, 'bg', '*_BG.npy'))
    all_features_raw = []
    for f in files:
        data = np.load(f)
        if data.shape[1] < 18:
            print(f"⚠️ 文件 {f} 维度异常，跳过")
            continue
        # print(data.shape)
        features = data[:, :]
        all_features_raw.append(features)
    if not all_features_raw:
        raise ValueError("❌ 未读取到任何有效特征")
    
    all_features_raw = np.array(np.vstack(all_features_raw))
    print(f'✅  Raw data 总样本数： {all_features_raw.shape[0]}')
    X_unique = np.unique(all_features_raw, axis=0)
    print(f"👓  去重后总样本数: {X_unique.shape[0]}")
    features_raw = X_unique[:,:-3]
    
    # 特殊归一化
    features_normed = normalize_to_sum512_batch(features_raw)
    
    return features_normed.astype(np.float32), X_unique

def apply_pca(X, n_components):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(X), pca

def cpu_kmeans(X, k):
    kmeans = SKKMeans(n_clusters=k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(X)
    dists = np.linalg.norm(X - kmeans.cluster_centers_[labels], axis=1)
    return labels, dists, kmeans

def process_config(dim, k, X_raw, rawdata, output_dir):
    X_pca, _ = apply_pca(X_raw, dim)
    labels, dists, model = cpu_kmeans(X_pca, k)
    inertia = np.mean(dists)

    # 拼接原始数据 + label
    raw_with_label = np.hstack([rawdata, labels.reshape(-1, 1)])
    cols = [f'f{i+1}' for i in range(rawdata.shape[1])] + ['label']
    csv_path = os.path.join(output_dir, f'rawdata_with_label_pca{dim}_k{k}.csv')
    df = pd.DataFrame(raw_with_label, columns=cols)
    df.to_csv(csv_path, index=False)

    # === 保存质心 ===
    centroids_path = os.path.join(output_dir, f'kmeans_centroids_pca{dim}_k{k}.npy')
    np.save(centroids_path, model.cluster_centers_)
    print(f"✅ 已保存质心: {centroids_path}")

    return dim, k, inertia, labels

def main():
    # === 配置参数 ===
    pic_dir = 'dataset'
    # input_npy = 'mini_normalized_features_deduped.npy'  # 归一化后的特征文件
    output_dir = 'visual_output_simple'

    # 归一化特征X，和rawdata
    X, rawdata = preprocess_all(pic_dir)
    
    pca_dims = [6]
    cluster_range = [4, 5, 6, 7, 8]
    # batch_size = 100000  # 按批处理样本数
    
    # === 主流程 ===
    # X_raw = np.load(input_npy)
    results = {dim: {} for dim in pca_dims}
    
    all_results = []
    
    for dim in tqdm(pca_dims, desc="Processing PCA Dimensions"):
        for k in tqdm(cluster_range, leave=False, desc=f"KMeans for PCA-{dim}"):
            result = process_config(dim, k, X, rawdata, output_dir)
            all_results.append(result)
    
    for dim, k, inertia, labels in all_results:
        results[dim][k] = {
            'inertia': inertia,
            'labels': labels
        }
    

if __name__ == "__main__":
    main()

