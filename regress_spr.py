import numpy as np
from itertools import product
import time

# ###
    # ((74, 63, 66, 62), (77, 62, 66)), ((89, 143, 92, 142), (49, 149, 90)), 
    # ((108, 109, 206, 109), (97, 103, 219)), ((77, 96, 116, 95), (77, 96, 116)),
    # ((56, 77, 186, 76), (30, 73, 198)), ((201, 127, 62, 128), (230, 120, 40)),
    # ((209, 53, 80, 54), (239, 19, 70)), ((165, 37, 70, 38), (193, 0, 64)),
    # ((67, 131, 121, 132), (0, 131, 117)), ((130, 150, 47, 149), (121, 153, 0)),
    # ((128, 234, 135, 233), (0, 240, 118)), ((139, 219, 169, 218), (87, 223, 160)),
    # ((241, 239, 116, 238), (245, 240, 113)), ((216, 70, 82, 69), (248, 53, 71)),
    # ((169, 35, 70, 36), (193, 0, 64)), ((213, 162, 77, 161), (230, 160, 50)),
    # ((241, 205, 96, 204), (255, 105, 64)), ((130, 134, 246, 135), (120, 130, 255)),
    # ((81, 147, 194, 146), (11, 147, 197)), ((96, 136, 236, 135), (66, 134, 244)),
    # ((82, 153, 97, 154), (15, 157, 88)), ((85, 157, 103, 158), (22, 160, 93)),
#     ((155, 193, 100, 192), (142, 196, 79)), ((39, 80, 82, 80), (0, 80, 80)),
# ###

# 样本数据（36个样本）
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

# Pentile到RGB转换函数
def pentile_to_rgb(R, G1, B, G2, params):
    low_bound, mid_bound, close_threshold, enhance_coeffs, reduce_coeffs, thresholds = params
    G = (G1 + G2) >> 1  # 位移替代除法
    R_prime, G_prime, B_prime = R, G, B
    sum_val = R + G + B
    
    # 确定最大值和次大值
    values = [R, G, B]
    sorted_values = sorted(values, reverse=True)
    max_val, second_val, min_val = sorted_values[0], sorted_values[1], sorted_values[2]
    idx = np.argsort(values)[::-1]
    
    # 两个接近的较大值
    if max_val - second_val < close_threshold:
        if max_val <= low_bound:
            enhance_coeff, reduce_coeff = enhance_coeffs[0], reduce_coeffs[0]
        elif max_val <= mid_bound:
            enhance_coeff, reduce_coeff = enhance_coeffs[1], reduce_coeffs[1]
        else:
            enhance_coeff, reduce_coeff = enhance_coeffs[2], reduce_coeffs[2]
        
        if idx[0] == 0 and idx[1] == 1 or idx[0] == 1 and idx[1] == 0:      # R和G接近
            R_prime = min(R + int(enhance_coeff * R), 255)
            G_prime = min(G + int(enhance_coeff * G), 255)
            B_prime = max(B - int(reduce_coeff * (max(R, G) - B)), 0)
        elif idx[0] == 0 and idx[1] == 2 or idx[0] == 2 and idx[1] == 0:  # R和B接近
            R_prime = min(R + int(enhance_coeff * R), 255)
            B_prime = min(B + int(enhance_coeff * B), 255)
            G_prime = max(G - int(reduce_coeff * (max(R, B) - G)), 0)
        elif idx[0] == 1 and idx[1] == 2 or idx[0] == 2 and idx[1] == 1:  # G和B接近
            G_prime = min(G + int(enhance_coeff * G), 255)
            B_prime = min(B + int(enhance_coeff * B), 255)
            R_prime = max(R - int(reduce_coeff * (max(G, B) - R)), 0)
    else:
        if max_val == R:

            if R < 48:
                R_prime = min(R + 1, 255)
                G_prime = max(G - 0.03*(R-G), 0)
                B_prime = max(B - 0.03*(R-B), 0)
            elif R < 96 :
                R_prime = min(R + 0.05*R, 255)
                G_prime = max(G - 0.08*(R-G), 0)
                B_prime = max(B - 0.08*(R-B), 0)
            elif R < 160:
                R_prime = min(R + 0.1*R, 255)
                G_prime = max(G - 0.15*(R-G), 0)
                B_prime = max(B - 0.15*(R-B), 0)
            else:
                R_prime = min(R + 0.08*R, 255)
                G_prime = max(G - 0.25*(R-G), 0)
                B_prime = max(B - 0.25*(R-B), 0)

            if R_prime < 32 and R <40 and G> 40:
                R_prime = 0

        elif max_val == B:
            if B < 48:
                B_prime = min(B + 1, 255)
                R_prime = max(R - 0.03*(B-R), 0)
                G_prime = max(G - 0.03*(B-G), 0)
            elif B < 96:
                B_prime = min(B + 0.05*B, 255)
                R_prime = max(R - 0.1*(B-R), 0)
                G_prime = max(G - 0.1*(B-G), 0)
            elif B < 160:
                B_prime = min(B + 0.09*B, 255)
                R_prime = max(R - 0.15*(B-R), 0)
                G_prime = max(G - 0.15*(B-G), 0)
            else:
                B_prime = min(B + 0.03*B, 255)
                R_prime = max(R - 0.05*(B-R), 0)
                G_prime = max(G - 0.05*(B-G), 0)

            if B_prime < 32 and B <40 and G > 50:
                R_prime = 0
            
        elif max_val == G:
            if G < 48:
                G_prime = min(G + 1, 255)
                R_prime = max(R - 0.25*(G-R), 0)
                B_prime = max(B - 0.15*(G-B), 0)
            elif G < 112:
                G_prime = min(G + 0.04*G, 255)
                R_prime = max(R - 0.45*(G-R), 0)
                B_prime = max(B - 0.25*(G-B), 0)
            elif G < 176:
                G_prime = min(G + 0.07*G, 255)
                R_prime = max(R - 0.55*(G-R), 0)
                B_prime = max(B - 0.3*(G-B), 0)
            else:
                G_prime = min(G + 0.08*G, 255)
                R_prime = max(R - 0.8*(G-R), 0)
                B_prime = max(B - 0.35*(G-B), 0)

            if G_prime < 32 and R < 40 and R > 50:
                G_prime = 0
    
    return R_prime, G_prime, B_prime

# 计算MAE
def compute_mae(params):
    total_mae = 0
    for (R, G1, B, G2), (R_target, G_target, B_target) in samples:
        R_pred, G_pred, B_pred = pentile_to_rgb(R, G1, B, G2, params)
        mae = abs(R_pred - R_target) + abs(G_pred - G_target) + abs(B_pred - B_target)
        total_mae += mae
    return total_mae / len(samples)

# 扩展参数网格
param_grid = {
    'low_bound': [96],  # 7个值
    'mid_bound': [217],  # 7个值
    'close_threshold': [10, 7, 2, 3, 4, 5, 6, 8, 9],  # 7个值
    'enhance_coeffs': [
        [0.03125, 0.0625, 0.09375],   # 1/32, 1/16, 3/32
        [0.0625, 0.125, 0.1875],     # 1/16, 1/8, 3/16 (当前值)
        [0.09375, 0.1875, 0.28125],  # 3/32, 3/16, 9/32
        [0.125, 0.25, 0.375],        # 1/8, 1/4, 3/8
        [0.1875, 0.375, 0.5625],     # 3/16, 3/8, 9/16
        [0.25, 0.5, 0.75],           # 1/4, 1/2, 3/4
        [0.3125, 0.625, 0.9375]      # 5/16, 5/8, 15/16
    ],
    'reduce_coeffs': [
        [0.0625, 0.125, 0.25],       # 1/16, 1/8, 1/4
        [0.125, 0.25, 0.5],          # 1/8, 1/4, 1/2 (当前值)
        [0.1875, 0.375, 0.5625],     # 3/16, 3/8, 9/16
        [0.25, 0.5, 0.75],           # 1/4, 1/2, 3/4
        [0.3125, 0.625, 0.9375],     # 5/16, 5/8, 15/16
        [0.375, 0.75, 1.125],        # 3/8, 3/4, 9/8 (稍超1)
        [0.5, 1.0, 1.5]              # 1/2, 1, 3/2
    ],
    'thresholds': [
        [5, 10, 15],                 # 极小值
        [10, 20, 30],                # 较小值
        [15, 25, 35],                # 当前值
        [20, 30, 40],                # 中等值
        [25, 35, 45],                # 较大值
        [30, 40, 50],                # 更大值
        [35, 45, 55]                 # 最大值
    ]
}

# 参数搜索
start_time = time.time()
best_mae = float('inf')
best_params = None
total_combinations = 0

for low_bound, mid_bound, close_threshold, enhance_coeffs, reduce_coeffs, thresholds in product(
    param_grid['low_bound'], param_grid['mid_bound'], param_grid['close_threshold'],
    param_grid['enhance_coeffs'], param_grid['reduce_coeffs'], param_grid['thresholds']
):
    if low_bound >= mid_bound:  # 确保区间有效
        continue
    total_combinations += 1
    params = (low_bound, mid_bound, close_threshold, enhance_coeffs, reduce_coeffs, thresholds)
    mae = compute_mae(params)
    if mae < best_mae:
        best_mae = mae
        best_params = params
    if total_combinations % 1000 == 0:  # 每1000次打印进度
        print(f"Progress: {total_combinations} combinations, Current MAE: {round(mae,2)}, Best MAE: {round(best_mae,2)}")

print(f"Total combinations tested: {total_combinations}")
print(f"Best MAE: {best_mae}")
print(f"Best Params: {best_params}")
print(f"Time taken: {time.time() - start_time:.2f} seconds")

### Best Params: (72, 217, 2, [0.03125, 0.0625, 0.09375], [0.375, 0.75, 1.125], [20, 30, 40])