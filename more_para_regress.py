import numpy as np
from datetime import datetime
import time
import optuna
import os
import joblib
import matplotlib.pyplot as plt
import csv
from optuna.visualization.matplotlib import plot_optimization_history, plot_pareto_front
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

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

CHECKPOINT_PATH = f"pkls/spr_study_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
RESULTS_CSV = f"params/best_params_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
PLOT_PATH = "optimization_history.png"
PARETO_PATH = "pareto_front.png"


def pentile_to_rgb(R, G1, B, G2, params):
    (low_bound, mid_bound, close_threshold, thresholds,
     RG_enh, RG_red, RB_enh, RB_red, GB_enh, GB_red,
     R_params, G_params, B_params) = params

    G = (G1 + G2) >> 1
    R_prime, G_prime, B_prime = R, G, B
    values = [R, G, B]
    sorted_vals = sorted(values, reverse=True)
    max_val, second_val, _ = sorted_vals
    idx = np.argsort(values)[::-1]

    if max_val - second_val < close_threshold:
        if (idx[0], idx[1]) in [(0, 1), (1, 0)]:
            if max_val <= low_bound:
                enh, red = RG_enh[0], RG_red[0]
            elif max_val <= mid_bound:
                enh, red = RG_enh[1], RG_red[1]
            else:
                enh, red = RG_enh[2], RG_red[2]
            R_prime = min(R + int(enh * R), 255)
            G_prime = min(G + int(enh * G), 255)
            B_prime = max(B - int(red * (max(R, G) - B)), 0)
        elif (idx[0], idx[1]) in [(0, 2), (2, 0)]:
            if max_val <= low_bound:
                enh, red = RB_enh[0], RB_red[0]
            elif max_val <= mid_bound:
                enh, red = RB_enh[1], RB_red[1]
            else:
                enh, red = RB_enh[2], RB_red[2]
            R_prime = min(R + int(enh * R), 255)
            B_prime = min(B + int(enh * B), 255)
            G_prime = max(G - int(red * (max(R, B) - G)), 0)
        elif (idx[0], idx[1]) in [(1, 2), (2, 1)]:
            if max_val <= low_bound:
                enh, red = GB_enh[0], GB_red[0]
            elif max_val <= mid_bound:
                enh, red = GB_enh[1], GB_red[1]
            else:
                enh, red = GB_enh[2], GB_red[2]
            G_prime = min(G + int(enh * G), 255)
            B_prime = min(B + int(enh * B), 255)
            R_prime = max(R - int(red * (max(G, B) - R)), 0)
    else:
        if idx[0] == 0:
            if R < thresholds[0]:
                R_prime = min(R + R_params[0], 255)
                G_prime = max(G - R_params[4]*(R-G), 0)
                B_prime = max(B - R_params[4]*(R-B), 0)
            elif R < thresholds[1]:
                R_prime = min(R + int(R_params[1]*R), 255)
                G_prime = max(G - int(R_params[5]*(R-G)), 0)
                B_prime = max(B - int(R_params[5]*(R-B)), 0)
            elif R < thresholds[2]:
                R_prime = min(R + int(R_params[2]*R), 255)
                G_prime = max(G - int(R_params[6]*(R-G)), 0)
                B_prime = max(B - int(R_params[6]*(R-B)), 0)
            else:
                R_prime = min(R + int(R_params[3]*R), 255)
                G_prime = max(G - int(R_params[7]*(R-G)), 0)
                B_prime = max(B - int(R_params[7]*(R-B)), 0)
        elif idx[0] == 1:
            if G < thresholds[0]:
                G_prime = min(G + G_params[0], 255)
                R_prime = max(R - G_params[4]*(G-R), 0)
                B_prime = max(B - G_params[4]*(G-B), 0)
            elif G < thresholds[1]:
                G_prime = min(G + int(G_params[1]*G), 255)
                R_prime = max(R - int(G_params[5]*(G-R)), 0)
                B_prime = max(B - int(G_params[5]*(G-B)), 0)
            elif G < thresholds[2]:
                G_prime = min(G + int(G_params[2]*G), 255)
                R_prime = max(R - int(G_params[6]*(G-R)), 0)
                B_prime = max(B - int(G_params[6]*(G-B)), 0)
            else:
                G_prime = min(G + int(G_params[3]*G), 255)
                R_prime = max(R - int(G_params[7]*(G-R)), 0)
                B_prime = max(B - int(G_params[7]*(G-B)), 0)
        else:
            if B < thresholds[0]:
                B_prime = min(B + B_params[0], 255)
                R_prime = max(R - B_params[4]*(B-R), 0)
                G_prime = max(G - B_params[4]*(B-G), 0)
            elif B < thresholds[1]:
                B_prime = min(B + int(B_params[1]*B), 255)
                R_prime = max(R - int(B_params[5]*(B-R)), 0)
                G_prime = max(G - int(B_params[5]*(B-G)), 0)
            elif B < thresholds[2]:
                B_prime = min(B + int(B_params[2]*B), 255)
                R_prime = max(R - int(B_params[6]*(B-R)), 0)
                G_prime = max(G - int(B_params[6]*(B-G)), 0)
            else:
                B_prime = min(B + int(B_params[3]*B), 255)
                R_prime = max(R - int(B_params[7]*(B-R)), 0)
                G_prime = max(G - int(B_params[7]*(B-G)), 0)

    return R_prime, G_prime, B_prime


def compute_mae_and_smoothness(params):
    total_mae = 0
    smoothness_score = 0
    weighted_mae = 0
    errors = []
    max_weight = 8.0

    for i, ((R, G1, B, G2), (R_target, G_target, B_target)) in enumerate(samples):
        R_pred, G_pred, B_pred = pentile_to_rgb(R, G1, B, G2, params)
        r_err = abs(R_pred - R_target)
        g_err = abs(G_pred - G_target)
        b_err = abs(B_pred - B_target)
        mae = r_err + g_err + b_err
        base_error = r_err + g_err + b_err
        important_indices = [20, 37, 39, 31, 10, 11]
        if i in important_indices:
            base_error *= max_weight

        errors.append(base_error)
        smooth = -abs(R_pred - G_pred) - abs(G_pred - B_pred) - abs(B_pred - R_pred)
        total_mae += mae
        smoothness_score += smooth
    n = len(samples)
    max_error = max(errors)
    weights = [np.exp(e / max_error) for e in errors]

    for i, weight in enumerate(weights):
        weighted_mae += errors[i] * weight

    return weighted_mae / n, -smoothness_score / n


def print_predictions(best_params):
    thresholds = [best_params[f"threshold{i}"] for i in range(3)]
    RG_enh = [best_params[f"RG_enh{i}"] for i in range(3)]
    RG_red = [best_params[f"RG_red{i}"] for i in range(3)]
    RB_enh = [best_params[f"RB_enh{i}"] for i in range(3)]
    RB_red = [best_params[f"RB_red{i}"] for i in range(3)]
    GB_enh = [best_params[f"GB_enh{i}"] for i in range(3)]
    GB_red = [best_params[f"GB_red{i}"] for i in range(3)]

    R_params = [1] + [best_params[f"R_enh{i+1}"] for i in range(3)] + [best_params[f"R_red{i+1}"] for i in range(4)]
    G_params = [1] + [best_params[f"G_enh{i+1}"] for i in range(3)] + [best_params[f"G_red{i+1}"] for i in range(4)]
    B_params = [1] + [best_params[f"B_enh{i+1}"] for i in range(3)] + [best_params[f"B_red{i+1}"] for i in range(4)]

    params = (
        best_params["low_bound"],
        best_params["mid_bound"],
        best_params["close_threshold"],
        thresholds,
        RG_enh, RG_red,
        RB_enh, RB_red,
        GB_enh, GB_red,
        tuple(R_params), tuple(G_params), tuple(B_params)
    )

    print("\nSample predictions:")
    for i, ((R, G1, B, G2), (RT, GT, BT)) in enumerate(samples):
        R_pred, G_pred, B_pred = pentile_to_rgb(R, G1, B, G2, params)
        print(f"Sample {i+1}: input = ({(R, G1, B, G2)}), Predicted = ({R_pred}, {G_pred}, {B_pred}), Target = ({RT}, {GT}, {BT}), Δ = ({abs(R_pred - RT)}, {abs(G_pred - GT)}, {abs(B_pred - BT)})")


if __name__ == '__main__':
    start_time = time.time()

    if os.path.exists(CHECKPOINT_PATH):
        study = joblib.load(CHECKPOINT_PATH)
        print("✅ 恢复已有优化状态。")
    else:
        sampler = TPESampler(multivariate=True)
        pruner = MedianPruner(n_warmup_steps=50)
        study = optuna.create_study(
        directions=["minimize", "minimize"],
            sampler=sampler,
            pruner=pruner
        )

    # ✅ 定义完整的优化目标函数，包括 thresholds
    def objective(trial):
        threshold0 = trial.suggest_int("threshold0", 5, 100)
        threshold1 = trial.suggest_int("threshold1", threshold0 + 1, 200)
        threshold2 = trial.suggest_int("threshold2", threshold1 + 1, 255)

        low_bound = trial.suggest_int("low_bound", 40, 150)
        mid_bound = trial.suggest_int("mid_bound", low_bound + 1, 255)
        close_threshold = trial.suggest_int("close_threshold", 1, 10)

        RG_enh = [trial.suggest_float(f"RG_enh{i}", 0.01, 0.8) for i in range(3)]
        RG_red = [trial.suggest_float(f"RG_red{i}", 0.01, 0.8) for i in range(3)]

        RB_enh = [trial.suggest_float(f"RB_enh{i}", 0.01, 0.8) for i in range(3)]
        RB_red = [trial.suggest_float(f"RB_red{i}", 0.01, 0.8) for i in range(3)]

        GB_enh = [trial.suggest_float(f"GB_enh{i}", 0.01, 0.8) for i in range(3)]
        GB_red = [trial.suggest_float(f"GB_red{i}", 0.01, 0.8) for i in range(3)]

        thresholds = [threshold0, threshold1, threshold2]

        R_params = [1,
                    trial.suggest_float("R_enh1", 0.01, 0.5),
                    trial.suggest_float("R_enh2", 0.01, 0.5),
                    trial.suggest_float("R_enh3", 0.01, 0.5),
                    trial.suggest_float("R_red1", 0.01, 0.5),
                    trial.suggest_float("R_red2", 0.01, 0.5),
                    trial.suggest_float("R_red3", 0.01, 0.5),
                    trial.suggest_float("R_red4", 0.01, 0.5)]

        G_params = [1,
                    trial.suggest_float("G_enh1", 0.01, 0.1),
                    trial.suggest_float("G_enh2", 0.01, 0.1),
                    trial.suggest_float("G_enh3", 0.01, 0.1),
                    trial.suggest_float("G_red1", 0.01, 0.1),
                    trial.suggest_float("G_red2", 0.01, 0.1),
                    trial.suggest_float("G_red3", 0.01, 0.1),
                    trial.suggest_float("G_red4", 0.01, 0.1)]

        B_params = [1,
                    trial.suggest_float("B_enh1", 0.01, 0.5),
                    trial.suggest_float("B_enh2", 0.01, 0.5),
                    trial.suggest_float("B_enh3", 0.01, 0.5),
                    trial.suggest_float("B_red1", 0.01, 0.5),
                    trial.suggest_float("B_red2", 0.01, 0.5),
                    trial.suggest_float("B_red3", 0.01, 0.5),
                    trial.suggest_float("B_red4", 0.01, 0.5)]

        params = (low_bound, mid_bound, close_threshold, thresholds,
            RG_enh, RG_red, RB_enh, RB_red, GB_enh, GB_red,
            tuple(R_params), tuple(G_params), tuple(B_params))

        return compute_mae_and_smoothness(params)


    study.optimize(objective, n_trials=300, n_jobs=-1, show_progress_bar=True)
    # joblib.dump(study, CHECKPOINT_PATH)

    # 自动筛选平衡解（最小化 MAE 和 Smoothness 乘积）
    best_trial = min(
        study.best_trials,
        key=lambda t: t.values[0] * abs(t.values[1])  # 平衡误差与平滑
    )

    with open(RESULTS_CSV, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["parameter", "value"])
        for k, v in best_trial.params.items():
            writer.writerow([k, v])

    fig_ax = plot_optimization_history(study, target=lambda t: t.values[0], target_name="MAE")
    fig_ax.figure.tight_layout()
    fig_ax.figure.savefig(PLOT_PATH)

    # pareto_ax = plot_pareto_front(study, target_names=["MAE", "Smoothness"])
    # pareto_ax.figure.tight_layout()
    # pareto_ax.figure.savefig(PARETO_PATH)

    print("\nBest Trial:")
    print("  MAE:", best_trial.values[0])
    print("  Smoothness:", best_trial.values[1])
    print("  Parameters:")
    for k, v in best_trial.params.items():
        print(f"    {k}: {v}")

    print_predictions(best_trial.params)

    print(f"\n总耗时: {time.time() - start_time:.2f} 秒")
    print(f"最优参数已保存到 {RESULTS_CSV}")
    print(f"优化过程图像已保存为 {PLOT_PATH}")
    print(f"Pareto 前沿图已保存为 {PARETO_PATH}")
