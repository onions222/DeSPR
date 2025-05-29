import numpy as np
import os

def deduplicate_npy(input_path, output_path=None):
    # 加载数据
    data = np.load(input_path)
    print(f"原始数据形状: {data.shape}")

    # 去重（基于整行）
    data_unique = np.unique(data, axis=0)
    print(f"去重后数据形状: {data_unique.shape}")

    # 保存结果
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = base + "_deduped.npy"
    np.save(output_path, data_unique)
    print(f"去重后的数据已保存为: {output_path}")

    return data_unique

input_npy = 'normalized_features.npy'
deduplicated_data = deduplicate_npy(input_npy)
