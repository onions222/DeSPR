# regress_spr_v4.py
# 分阶段 RGBG-Pentile 转 RGB-Stripe 参数优化

import numpy as np
import time
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from datetime import datetime
from collections import defaultdict

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
    ((229, 230, 80, 230),(232, 232, 0)),((150, 150, 48, 150),(152, 152, 0)),
    ((96, 96, 27, 96),(96 ,96 ,0)),
    ((74, 63, 66, 62), (77, 62, 66)), ((89, 143, 92, 142), (49, 149, 90)), 
    ((108, 109, 206, 109), (97, 103, 219)), ((77, 96, 116, 95), (77, 96, 116)),
    ((56, 77, 186, 76), (30, 73, 198)), ((201, 127, 62, 128), (230, 120, 40)),
    ((209, 53, 80, 54), (239, 19, 70)), ((165, 37, 70, 38), (193, 0, 64)),
    ((67, 131, 121, 132), (0, 131, 117)), ((130, 150, 47, 149), (121, 153, 0)),
    ((128, 234, 135, 233), (0, 240, 118)), ((139, 219, 169, 218), (87, 223, 160)),
    ((241, 239, 116, 238), (245, 240, 113)), ((216, 70, 82, 69), (248, 53, 71)),
    ((169, 35, 70, 36), (193, 0, 64)), ((213, 162, 77, 161), (230, 160, 50)),
    ((241, 205, 96, 204), (255, 205, 64)), ((130, 134, 246, 135), (120, 130, 255)),
    ((81, 147, 194, 146), (11, 147, 197)), ((96, 136, 236, 135), (66, 134, 244)),
    ((82, 153, 97, 154), (15, 157, 88)), ((85, 157, 103, 158), (22, 160, 93)),
    ((198, 50, 144, 49),(225, 20, 145)), ((228,128,147,128),(253,118,144)),
    ((207, 46, 141, 46),(236, 0, 140)), ((239, 195, 70, 194),(255, 193, 0)),
    ((230, 149, 59, 148),(253, 145, 12)),((194, 79, 67, 79),(219, 68, 55)),
    ((182, 39, 74, 38),(207, 0, 69)), ((106, 24, 72, 24),(117, 11, 70)),
    ((135, 187, 239, 186), (101, 187, 244)), ((118, 187, 239, 186), (101, 187, 244)),
    ((133, 164, 225, 163), (112, 164, 230)), ((149, 127, 218, 125),(152, 121, 224)),
    ((80, 110, 208, 110),(56, 107, 215)), ((23, 49, 82, 49),(0, 49, 82)),
    ((110, 200, 129, 200),(11, 205, 116)), ((93, 163, 182, 163),(31, 165, 183)),
    ((89, 160, 108, 160), (30, 163, 98)), ((157, 194, 99, 192),(143, 197, 80))
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
        major = idx[0]
        val = [R, G, B][major]
        t = thresholds
        p = [R_params, G_params, B_params][major]
        out = [R, G, B]
        if val < t[0]:
            out[major] = min(val + p[0], 255)
            for i in range(3):
                if i != major:
                    out[i] = max(out[i] - p[4]*(val - out[i]), 0)
        elif val < t[1]:
            out[major] = min(val + int(p[1]*val), 255)
            for i in range(3):
                if i != major:
                    out[i] = max(out[i] - int(p[5]*(val - out[i])), 0)
        elif val < t[2]:
            out[major] = min(val + int(p[2]*val), 255)
            for i in range(3):
                if i != major:
                    out[i] = max(out[i] - int(p[6]*(val - out[i])), 0)
        else:
            out[major] = min(val + int(p[3]*val), 255)
            for i in range(3):
                if i != major:
                    out[i] = max(out[i] - int(p[7]*(val - out[i])), 0)
        R_prime, G_prime, B_prime = out

    return R_prime, G_prime, B_prime

def compute_mae_and_smoothness(params):
    total_mae = 0
    smoothness_score = 0
    errors = []
    for (R, G1, B, G2), (RT, GT, BT) in samples:
        R_pred, G_pred, B_pred = pentile_to_rgb(R, G1, B, G2, params)
        err = abs(R_pred - RT) + abs(G_pred - GT) + abs(B_pred - BT)
        smooth = -abs(R_pred - G_pred) - abs(G_pred - B_pred) - abs(B_pred - R_pred)
        total_mae += err
        smoothness_score += smooth
        errors.append(err)
    n = len(samples)
    return total_mae / n, -smoothness_score / n

def run_stage_optimization(stage_id, base_params=None, n_trials=100):
    base_params = base_params or {}
    include = stages[stage_id]['params']

    def suggest(trial, name, low, high):
        if name in include:
            return trial.suggest_float(name, low, high)
        elif name in base_params:
            return base_params[name]
        else:
            return 0.0

    def suggest_int(trial, name, low, high):
        if name in include:
            return trial.suggest_int(name, low, high)
        elif name in base_params:
            return base_params[name]
        else:
            return low

    def objective(trial):
        threshold0 = suggest_int(trial, "threshold0", 5, 100)
        threshold1 = suggest_int(trial, "threshold1", threshold0 + 1, 200)
        threshold2 = suggest_int(trial, "threshold2", threshold1 + 1, 255)

        low_bound = suggest_int(trial, "low_bound", 40, 150)
        mid_bound = suggest_int(trial, "mid_bound", low_bound + 1, 255)
        close_threshold = suggest_int(trial, "close_threshold", 1, 10)

        enhance = [suggest(trial, f"enhance{i}", 0.01, 0.8) for i in range(3)]
        reduce = [suggest(trial, f"reduce{i}", 0.01, 0.8) for i in range(3)]

        def build_params(prefix, enh_range, red_range):
            enh = [suggest(trial, f"{prefix}_enh{i+1}", *enh_range) for i in range(3)]
            red = [suggest(trial, f"{prefix}_red{i+1}", *red_range) for i in range(4)]
            return [1.0] + enh + red

        R_params = build_params("R", (0.01, 0.5), (0.01, 0.5))
        G_params = build_params("G", (0.01, 0.1), (0.01, 0.1))
        B_params = build_params("B", (0.01, 0.5), (0.01, 0.5))

        params = (
            low_bound, mid_bound, close_threshold,
            enhance, reduce, [threshold0, threshold1, threshold2],
            tuple(R_params), tuple(G_params), tuple(B_params)
        )
        return compute_mae_and_smoothness(params)

    study = optuna.create_study(
        directions=["minimize", "minimize"],
        sampler=TPESampler(),
        pruner=MedianPruner(n_warmup_steps=20)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best = min(study.best_trials, key=lambda t: t.values[0] * abs(t.values[1]))
    return best.params, study

stages = {
    1: {"desc": "仅调 enhance/reduce", "params": {
        "enhance0", "enhance1", "enhance2", "reduce0", "reduce1", "reduce2"
    }},
    2: {"desc": "加入 thresholds 和 close_threshold", "params": {
        "enhance0", "enhance1", "enhance2", "reduce0", "reduce1", "reduce2",
        "threshold0", "threshold1", "threshold2", "low_bound", "mid_bound", "close_threshold"
    }},
    3: {"desc": "加入 R/G/B 通道参数", "params": {
        "enhance0", "enhance1", "enhance2", "reduce0", "reduce1", "reduce2",
        "threshold0", "threshold1", "threshold2", "low_bound", "mid_bound", "close_threshold",
        *[f"R_enh{i+1}" for i in range(3)], *[f"R_red{i+1}" for i in range(4)],
        *[f"G_enh{i+1}" for i in range(3)], *[f"G_red{i+1}" for i in range(4)],
        *[f"B_enh{i+1}" for i in range(3)], *[f"B_red{i+1}" for i in range(4)]
    }},
    4: {"desc": "全参数微调", "params": set()}
}

if __name__ == "__main__":
    base_params = {}
    for stage in [1, 2, 3, 4]:
        print(f"\n===== 随階调优 阶段 {stage}: {stages[stage]['desc']} =====")
        base_params, _ = run_stage_optimization(stage, base_params, n_trials=100)

    print("\n最终最优参数:")
    for k, v in base_params.items():
        print(f"  {k}: {v}")
