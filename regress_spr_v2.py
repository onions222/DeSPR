import numpy as np
from itertools import islice, product
import time
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import heapq
import pickle
import os

# 样本数据（部分省略，如需完整请替换）
samples = [
    ((24, 56, 57, 55), (0, 56, 56)), ((11, 31, 33, 32), (0, 32, 32)),
    ((28, 1, 31, 1), (32, 0, 32)), ((42, 3, 47, 3), (48, 0, 48)),
    ((65, 7, 72, 7), (72, 0, 72)), ((72, 72, 17, 72), (72, 72, 0)),
    ((56, 56, 11, 56), (56, 56, 0)), ((24, 24, 2, 24), (24, 24, 0)),
    ((214, 213, 73, 214), (216, 216, 0)), ((245, 244, 85, 244), (248, 248, 0)),
    ((137, 244, 246, 244), (0, 248, 248)), ((123, 221, 223, 221), (0, 224, 224)),
    ((99, 182, 181, 181), (0, 184, 184)), ((67, 128, 130, 129),(0, 128, 128)),
    ((29, 63, 65, 64), (0, 64, 64)), ((199, 46, 217, 47),(224, 0, 224)),
    ((130, 26, 141, 26),(144, 0, 144)),((58, 5, 64, 6),(64, 0, 64)),
    ((22, 230, 80, 230),(232, 232, 0)),((150, 150, 48, 150),(152, 152, 0)),
    ((96, 27, 96, 96),(96 ,96 ,0)),
    ((74, 63, 66, 62), (77, 62, 66)), ((89, 143, 92, 142), (49, 149, 90)), 
    ((108, 109, 206, 109), (97, 103, 219)), ((77, 96, 116, 95), (77, 96, 116)),
    ((56, 77, 186, 76), (30, 73, 198)), ((201, 127, 62, 128), (230, 120, 40)),
    ((209, 53, 80, 54), (239, 19, 70)), ((165, 37, 70, 38), (193, 0, 64)),
    ((67, 131, 121, 132), (0, 131, 117)), ((130, 150, 47, 149), (121, 153, 0)),
    ((128, 234, 135, 233), (0, 240, 118)), ((139, 219, 169, 218), (87, 223, 160)),
    ((241, 239, 116, 238), (245, 240, 113)), ((216, 70, 82, 69), (248, 53, 71)),
    ((169, 35, 70, 36), (193, 0, 64)), ((213, 162, 77, 161), (230, 160, 50)),
    ((241, 205, 96, 204), (255, 105, 64)), ((130, 134, 246, 135), (120, 130, 255)),
    ((81, 147, 194, 146), (11, 147, 197)), ((96, 136, 236, 135), (66, 134, 244)),
    ((82, 153, 97, 154), (15, 157, 88)), ((85, 157, 103, 158), (22, 160, 93))
]

def pentile_to_rgb(R, G1, B, G2, params):
    low_bound, mid_bound, close_threshold, enhance_coeffs, reduce_coeffs, thresholds, \
    R_params, G_params, B_params = params

    G = (G1 + G2) >> 1
    R_prime, G_prime, B_prime = R, G, B
    values = [R, G, B]
    sorted_vals = sorted(values, reverse=True)
    max_val, second_val, _ = sorted_vals
    idx = np.argsort(values)[::-1]

    if max_val - second_val < close_threshold:
        if max_val <= low_bound:
            enhance_coeff, reduce_coeff = enhance_coeffs[0], reduce_coeffs[0]
        elif max_val <= mid_bound:
            enhance_coeff, reduce_coeff = enhance_coeffs[1], reduce_coeffs[1]
        else:
            enhance_coeff, reduce_coeff = enhance_coeffs[2], reduce_coeffs[2]

        if (idx[0], idx[1]) in [(0,1), (1,0)]:
            R_prime = min(R + int(enhance_coeff * R), 255)
            G_prime = min(G + int(enhance_coeff * G), 255)
            B_prime = max(B - int(reduce_coeff * (max(R, G) - B)), 0)
        elif (idx[0], idx[1]) in [(0,2), (2,0)]:
            R_prime = min(R + int(enhance_coeff * R), 255)
            B_prime = min(B + int(enhance_coeff * B), 255)
            G_prime = max(G - int(reduce_coeff * (max(R, B) - G)), 0)
        elif (idx[0], idx[1]) in [(1,2), (2,1)]:
            G_prime = min(G + int(enhance_coeff * G), 255)
            B_prime = min(B + int(enhance_coeff * B), 255)
            R_prime = max(R - int(reduce_coeff * (max(G, B) - R)), 0)
    else:
        if idx[0] == 0:
            if R < 48:
                R_prime = min(R + R_params[0], 255)
                G_prime = max(G - R_params[4]*(R-G), 0)
                B_prime = max(B - R_params[4]*(R-B), 0)
            elif R < 96:
                R_prime = min(R + int(R_params[1]*R), 255)
                G_prime = max(G - int(R_params[5]*(R-G)), 0)
                B_prime = max(B - int(R_params[5]*(R-B)), 0)
            elif R < 160:
                R_prime = min(R + int(R_params[2]*R), 255)
                G_prime = max(G - int(R_params[6]*(R-G)), 0)
                B_prime = max(B - int(R_params[6]*(R-B)), 0)
            else:
                R_prime = min(R + int(R_params[3]*R), 255)
                G_prime = max(G - int(R_params[7]*(R-G)), 0)
                B_prime = max(B - int(R_params[7]*(R-B)), 0)
        elif idx[0] == 1:
            if G < 48:
                G_prime = min(G + G_params[0], 255)
                R_prime = max(R - G_params[4]*(G-R), 0)
                B_prime = max(B - G_params[4]*(G-B), 0)
            elif G < 112:
                G_prime = min(G + int(G_params[1]*G), 255)
                R_prime = max(R - int(G_params[5]*(G-R)), 0)
                B_prime = max(B - int(G_params[5]*(G-B)), 0)
            elif G < 176:
                G_prime = min(G + int(G_params[2]*G), 255)
                R_prime = max(R - int(G_params[6]*(G-R)), 0)
                B_prime = max(B - int(G_params[6]*(G-B)), 0)
            else:
                G_prime = min(G + int(G_params[3]*G), 255)
                R_prime = max(R - int(G_params[7]*(G-R)), 0)
                B_prime = max(B - int(G_params[7]*(G-B)), 0)
        else:
            if B < 48:
                B_prime = min(B + B_params[0], 255)
                R_prime = max(R - B_params[4]*(B-R), 0)
                G_prime = max(G - B_params[4]*(B-G), 0)
            elif B < 96:
                B_prime = min(B + int(B_params[1]*B), 255)
                R_prime = max(R - int(B_params[5]*(B-R)), 0)
                G_prime = max(G - int(B_params[5]*(B-G)), 0)
            elif B < 160:
                B_prime = min(B + int(B_params[2]*B), 255)
                R_prime = max(R - int(B_params[6]*(B-R)), 0)
                G_prime = max(G - int(B_params[6]*(B-G)), 0)
            else:
                B_prime = min(B + int(B_params[3]*B), 255)
                R_prime = max(R - int(B_params[7]*(B-R)), 0)
                G_prime = max(G - int(B_params[7]*(B-G)), 0)
    return R_prime, G_prime, B_prime

def compute_mae(params):
    total_mae = 0
    for (R, G1, B, G2), (R_target, G_target, B_target) in samples:
        R_pred, G_pred, B_pred = pentile_to_rgb(R, G1, B, G2, params)
        mae = abs(R_pred - R_target) + abs(G_pred - G_target) + abs(B_pred - B_target)
        total_mae += mae
    return total_mae / len(samples)

def evaluate_param_combo(combo):
    combo_dict = dict(zip(keys, combo))
    if combo_dict['low_bound'] >= combo_dict['mid_bound']:
        return float('inf'), None
    R_params = tuple(combo_dict[k] for k in R_keys)
    G_params = tuple(combo_dict[k] for k in G_keys)
    B_params = tuple(combo_dict[k] for k in B_keys)
    params = (
        combo_dict['low_bound'], combo_dict['mid_bound'], combo_dict['close_threshold'],
        combo_dict['enhance_coeffs'], combo_dict['reduce_coeffs'], combo_dict['thresholds'],
        R_params, G_params, B_params
    )
    mae = compute_mae(params)
    return mae, params

# 构造搜索网格（这里只列举较小范围，完整搜索请自行扩展）
param_grid = {
    'low_bound': [96],
    'mid_bound': [217],
    'close_threshold': [2],
    'enhance_coeffs': [[0.03125, 0.0625, 0.09375]],
    'reduce_coeffs': [[0.125, 0.25, 0.5]],
    'thresholds': [[10, 20, 30]],
    'R_add0': [1], 'R_enh1': np.arange(0.02, 0.08, 0.01), 'R_enh2': np.arange(0.07, 0.13, 0.01), 'R_enh3': np.arange(0.05, 0.11, 0.01),
    'R_red1': np.arange(0.01, 0.06, 0.01), 'R_red2': np.arange(0.05, 0.11, 0.01), 'R_red3': np.arange(0.12, 0.18, 0.01), 'R_red4': np.arange(0.22, 0.28, 0.01),
    'G_add0': [1], 'G_enh1': np.arange(0.01, 0.08, 0.01), 'G_enh2': np.arange(0.05, 0.1, 0.01), 'G_enh3': np.arange(0.06, 0.11, 0.01),
    'G_red1': np.arange(0.22, 0.28, 0.01), 'G_red2': np.arange(0.42, 0.48, 0.01), 'G_red3': np.arange(0.52, 0.58, 0.01), 'G_red4': np.arange(0.78, 0.83, 0.01),
    'B_add0': [1], 'B_enh1': np.arange(0.02, 0.08, 0.01), 'B_enh2': np.arange(0.06, 0.12, 0.01), 'B_enh3': np.arange(0.01, 0.06, 0.01),
    'B_red1': np.arange(0.01, 0.06, 0.01), 'B_red2': np.arange(0.08, 0.13, 0.01), 'B_red3': np.arange(0.12, 0.18, 0.01), 'B_red4': np.arange(0.02, 0.08, 0.01),
}

# 扩展 key 列表以便解包组合
keys = list(param_grid.keys())
values = list(param_grid.values())
R_keys = ['R_add0', 'R_enh1', 'R_enh2', 'R_enh3', 'R_red1', 'R_red2', 'R_red3', 'R_red4']
G_keys = ['G_add0', 'G_enh1', 'G_enh2', 'G_enh3', 'G_red1', 'G_red2', 'G_red3', 'G_red4']
B_keys = ['B_add0', 'B_enh1', 'B_enh2', 'B_enh3', 'B_red1', 'B_red2', 'B_red3', 'B_red4']

# 中断恢复文件
RECOVERY_FILE = "search_progress.pkl"

# 设置保留的 Top-N 最优组合
TOP_N = 5

# 恢复已保存进度
if os.path.exists(RECOVERY_FILE):
    with open(RECOVERY_FILE, 'rb') as f:
        checked_combos, top_results = pickle.load(f)
else:
    checked_combos = set()
    top_results = []

    
def batch_generator(it, size):
    while True:
        batch = list(islice(it, size))
        if not batch:
            break
        yield batch



if __name__ == '__main__':
    start_time = time.time()
    chunk_size = 5000
    pool = Pool(cpu_count())
    try:
        for batch in tqdm(batch_generator(product(*values), chunk_size), desc="Searching", unit="batch"):
            # 跳过已检查组合
            batch = [tuple(tuple(x) if isinstance(x, list) else x for x in combo) for combo in batch]
            batch = [combo for combo in batch if combo not in checked_combos]
            if not batch:
                continue

            results = pool.map(evaluate_param_combo, batch)
            for combo, (mae, p) in zip(batch, results):
                checked_combos.add(combo)
                if p is not None:
                    if len(top_results) < TOP_N:
                        heapq.heappush(top_results, (-mae, p))
                    else:
                        heapq.heappushpop(top_results, (-mae, p))

            # 每批保存进度
            with open(RECOVERY_FILE, 'wb') as f:
                pickle.dump((checked_combos, top_results), f)

    finally:
        pool.close()
        pool.join()

    # 输出前 N 名
    final_results = sorted([(-mae, p) for (mae, p) in top_results])
    print("\nTop 10 Results:")
    for i, (mae, p) in enumerate(final_results, 1):
        print(f"#{i}: MAE = {mae:.4f}, Params = {p}")

    print(f"\nTotal combinations checked: {len(checked_combos)}")
    print(f"Time taken: {time.time() - start_time:.2f} seconds")
