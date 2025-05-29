import os
import glob
import numpy as np

# 配置路径
input_dir = 'dataset'
output_path = 'normalized_features.npy'
#os.makedirs(os.path.dirname(output_path), exist_ok=True)

def normalize_to_sum512_batch(X):
    """按行将矩阵归一化，使每行元素和为 512"""
    sums = np.sum(X, axis=1, keepdims=True)
    scale = np.where(sums == 0, 1.0, 512.0 / sums)
    X_scaled = X * scale
    return np.round(X_scaled).astype(np.uint16)

def preprocess_all(input_dir):
    files = glob.glob(os.path.join(input_dir, 'rg', '*_RG.npy')) + \
            glob.glob(os.path.join(input_dir, 'bg', '*_BG.npy'))
    all_features = []
    for f in files:
        data = np.load(f)
        if data.shape[1] < 18:
            print(f"⚠️ 文件 {f} 维度异常，跳过")
            continue
        features = data[:, :-3]
        normed = normalize_to_sum512_batch(features)
        all_features.append(normed)
    if not all_features:
        raise ValueError("❌ 未读取到任何有效特征")
    X_all = np.vstack(all_features)
    print(f"✅ 总样本数：{X_all.shape[0]}")
    return X_all.astype(np.float32)

# 主执行逻辑
if __name__ == '__main__':
    X = preprocess_all(input_dir)
    np.save(output_path, X)
    print(f"✅ 已保存归一化特征至 {output_path}")

