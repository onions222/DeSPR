import numpy as np
import xgboost as xgb
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, classification_report
from scipy.io import savemat, loadmat
import json

def train_and_export_xgb(good_path, bad_path, out_dir='xgb_out', use_gpu=False):
    """
    使用 18 维特征做二分类，训练 XGBoost 模型，并将所有树以 MATLAB 可读格式导出，
    包括：
      - scaler.mat    (mu, sigma)
      - sel_idx.txt   (0..17)
      - xgb_trees.mat (trees 结构体)
    """
    Path(out_dir).mkdir(exist_ok=True)

    # === 1. 加载数据 ===
    #good = np.load(good_path)   # shape: (N1, 18)
    #bad  = np.load(bad_path)    # shape: (N2, 18)
    # X = np.vstack([good, bad])  # shape: (N1+N2, 18)
    # y = np.r_[np.ones(len(good)), np.zeros(len(bad))]
    
    X = np.vstack([bad, good])  # shape: (N1+N2, 18)
    y = np.r_[np.zeros(len(bad)), np.ones(len(good))]
    
    G   = X[:, 0:9]
    Maj = X[:, 9:14]
    Min = X[:, 14:18]

    Gx = (G[:,2]+2*G[:,5]+G[:,8] - G[:,0]+2*G[:,3]+G[:,7]).reshape(-1, 1)
    Gy = (G[:,6]+2*G[:,7]+G[:,8] - G[:,0]+2*G[:,1]+G[:,2]).reshape(-1, 1)
    maxG  = np.max(G, axis=1).reshape(-1, 1)
    minG  = np.min(G, axis=1).reshape(-1, 1)
    Gdiff = maxG - minG
    Gmean = np.mean(G, axis=1).reshape(-1, 1)

    Majx = (Maj[:,1]+Maj[:,4]-Maj[:,0]-Maj[:,3]).reshape(-1, 1)
    Majy = (Maj[:,3]+Maj[:,4]-Maj[:,0]-Maj[:,1]).reshape(-1, 1)
    maxMaj = np.max(Maj, axis=1).reshape(-1, 1)
    minMaj = np.min(Maj, axis=1).reshape(-1, 1)
    Majdiff = maxMaj - minMaj
    Majmean = np.mean(Maj, axis=1).reshape(-1, 1)

    Minx = (Min[:, 2] - Min[:, 1]).reshape(-1, 1)
    Miny = (Min[:, 3] - Min[:, 0]).reshape(-1, 1)
    maxMin = np.max(Min, axis=1).reshape(-1, 1)
    minMin = np.min(Min, axis=1).reshape(-1, 1)
    Mindiff = maxMin - minMin
    Minmean = np.mean(Min, axis=1).reshape(-1, 1)
    
    # color
    temp = np.hstack([maxG, maxMaj, maxMin])

    maxV  = np.max(np.hstack([maxG, maxMaj, maxMin]),axis=1).reshape(-1, 1).astype(np.float32)
    minV  = np.max(np.hstack([minG, minMaj, minMin]),axis=1).reshape(-1, 1).astype(np.float32)
    cross = (maxV-minV)/(maxV+1)
    


    X = np.hstack([Gx, Gy, Gdiff, Gmean, Majx, Majy, Majdiff, Majmean, Minx, Miny, Mindiff, Minmean, cross])
        
    # === 2. 标准化 ===
    scaler = StandardScaler().fit(X)
    X_std = scaler.transform(X)

    # === 3. 处理样本不平衡 ===
    pos = np.sum(y == 1)
    neg = np.sum(y == 0)
    scale_pos_weight = neg / pos

    # === 4. XGBoost 训练 ===
    params = dict(
        objective='binary:logistic',
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=1.0,
        colsample_bytree=0.8,
        eval_metric='aucpr',
        n_jobs=0,
        random_state=42,
        tree_method='hist',
        scale_pos_weight=scale_pos_weight
    )
    if use_gpu:
        params['device'] = 'cuda'
        #params['gpu_id'] = 0
    
    #weights = np.ones(len(y))
    #idx = [0, 1, 3, 4, 6, 12, 13, 14, 15, 17, 19, 20, 22, 27]
    #idx1 = [0, 20, 27]
    #weights[idx] = 5.0
    #weights[idx1] = 15.0
    #weights[14:16] = 20.0
    #weights[15] = 35.0
    #print(X[0:5,:])
    # weights[22] = 20.0
    clf = xgb.XGBClassifier(**params).fit(X_std, y)
    #clf = xgb.XGBClassifier(**params).fit(X_std, y, sample_weight = weights)
    importances = clf.feature_importances_
    print(importances)

    pred_bad = np.zeros(len(bad))


    # === 5. 训练集 ACC & AUC 打印 ===
    y_prob = clf.predict_proba(X_std)[:, 1]
    y_pred = clf.predict(X_std)
    #print(f'--Pred: {y_pred[0:25]}')
    #print(f'--Sample: yprob:{y_prob[0:25]}')
    #print(f'--Sample: label:{y_pred[-24:]}')

    auc = roc_auc_score(y, y_prob)
    acc = accuracy_score(y, y_pred)
    rec = recall_score(y, y_pred)
    print(f"✅ Train Accuracy: {acc:.4f}")
    #print(f"✅ Train AUC:      {auc:.4f}")
    #print(f"✅ Recall score:   {rec:.4f}")
    print(classification_report(y, y_pred, digits=4))

    # 打印bad分类情况
    bad_idx = []
    for i in range(len(bad)):
        if y_pred[i] != 0:
            #print(f'sample: {bad[i]}')
            bad_idx.append(i)
    #print(bad[bad_idx[:20]])
    #print(bad_idx[:20])
    

    # === 6. 保存标准化参数 ===
    savemat(Path(out_dir) / 'scaler.mat', {
        'mu': scaler.mean_,       # (18,)
        'sigma': scaler.scale_    # (18,)
    })

    # === 7. 保存特征索引 0..17 ===
    sel_idx = np.arange(X.shape[1], dtype=np.int32)
    np.savetxt(Path(out_dir) / 'sel_idx.txt', sel_idx, fmt='%d')

    # === 8. 导出每棵树的结构到 MATLAB ===
    export_xgb_model_to_mat(clf, Path(out_dir) / 'xgb_trees.mat')

def export_xgb_model_to_mat(clf, save_path):
    """
    将 XGBClassifier 导出的 Booster，按 JSON 中的 nodeid 建立数组，
    最终保存为 {'trees': tree_structs} 的 .mat 文件。
    """
    booster = clf.get_booster()
    # 获取 JSON 格式的 dump
    model_dump = booster.get_dump(with_stats=True, dump_format='json')

    # 保存所有树的列表，后面变成 MATLAB 结构体数组
    tree_structs = []

    # 对于每一棵树
    for tree_json in model_dump:
        tree_dict = json.loads(tree_json)

        # 先用递归把每个节点信息塞到字典里，key = nodeid（0-based）
        nodes_map = {}
        def traverse(node, parent_id):
            """
            node: JSON dict 节点
            parent_id: int, 父节点的 nodeid；根节点 parent_id = -1
            """
            nid = node['nodeid']  # XGBoost JSON 提供的节点 id
            entry = {
                'parent': parent_id,
                'is_leaf': ('leaf' in node),
                'split_feature': -1,
                'split_condition': 0.0,
                'yes': -1,
                'no': -1,
                'leaf_value': 0.0
            }

            if 'leaf' in node:
                # 叶子节点
                entry['leaf_value'] = node['leaf']
            else:
                # 非叶子节点：记录分裂特征 idx (去掉 f 前缀)，阈值，以及 yes/no 子节点 id
                entry['split_feature'] = int(node['split'][1:])  # “f8” -> 8
                entry['split_condition'] = float(node['split_condition'])
                entry['yes'] = node['yes']
                entry['no']  = node['no']
                # 继续遍历左右孩子
                traverse(node['children'][0], nid)
                traverse(node['children'][1], nid)

            nodes_map[nid] = entry

        # 从根节点开始（parent = -1）
        traverse(tree_dict, -1)

        # 得到这棵树中最大的 nodeid
        max_nid = max(nodes_map.keys())
        num_nodes = max_nid + 1  # 节点总数

        # 申请固定长度的数组（num_nodes x 1）
        is_leaf_arr      = np.zeros(num_nodes, dtype=np.uint8)
        parent_arr       = np.full(num_nodes, -1, dtype=np.int32)
        split_feat_arr   = np.zeros(num_nodes, dtype=np.int32)
        split_cond_arr   = np.zeros(num_nodes, dtype=np.float64)
        yes_arr          = np.full(num_nodes, -1, dtype=np.int32)
        no_arr           = np.full(num_nodes, -1, dtype=np.int32)
        leaf_value_arr   = np.zeros(num_nodes, dtype=np.float64)

        # 按 nodeid 填充
        for nid, ent in nodes_map.items():
            is_leaf_arr[nid]       = 1 if ent['is_leaf'] else 0
            parent_arr[nid]        = ent['parent']
            split_feat_arr[nid]    = ent['split_feature']
            split_cond_arr[nid]    = ent['split_condition']
            yes_arr[nid]           = ent['yes']
            no_arr[nid]            = ent['no']
            leaf_value_arr[nid]    = ent['leaf_value']

        # 将这一棵树的各数组保存到一个结构体
        tree_structs.append({
            'nid'            : np.arange(num_nodes, dtype=np.int32),
            'parent'         : parent_arr,
            'is_leaf'        : is_leaf_arr,
            'split_feature'  : split_feat_arr,
            'split_condition': split_cond_arr,
            'yes'            : yes_arr,
            'no'             : no_arr,
            'leaf_value'     : leaf_value_arr
        })

    # 将所有树打包成 MATLAB 结构体数组保存
    savemat(save_path, {'trees': tree_structs})

# === 主入口 ===
if __name__ == '__main__':
    
    #output_path = 'eval/split4/'
    #bad_data = loadmat("/work/hwc/matfile/split4/triplebads1.mat", struct_as_record=False, squeeze_me=True)
    #good_data = loadmat("/work/hwc/matfile/split4/triplebads2.mat", struct_as_record=False, squeeze_me=True)
    #bad = bad_data['triplebads1'][:,:18]
    #good = good_data['triplebads2'][:,:18]

    #output_path = 'eval/split3/'
    #bad_data = loadmat("/work/hwc/matfile/split3/bad_bads2.mat", struct_as_record=False, squeeze_me=True)
    #good_data = loadmat("/work/hwc/matfile/split3/bad_goods2.mat", struct_as_record=False, squeeze_me=True)
    ##print(bad_data.keys())
    ##print(good_data.keys())
    #bad = bad_data['bad_bads2'][:,:18]
    #good = good_data['bad_goods2'][:,:18]



    output_path = 'eval/split1/'
    bad_data = loadmat("/work/hwc/matfile/split1/features.mat", struct_as_record=False, squeeze_me=True)
    good_data = loadmat("/work/hwc/matfile/split1/normals.mat", struct_as_record=False, squeeze_me=True)
    bad = bad_data['features'][:,:18]
    good = good_data['normals'][:,:18]

    #output_path = 'eval/split2/good_split/'
    #bad_data = loadmat("/work/hwc/matfile/split2/good_bads.mat", struct_as_record=False, squeeze_me=True)
    #good_data = loadmat("/work/hwc/matfile/split2/good_goods.mat", struct_as_record=False, squeeze_me=True)
    #bad = bad_data['good_bads'][:,:18]
    #good = good_data['good_goods'][:,:18]


    #output_path = 'eval/split1v2/'
    #bad_data = loadmat("/work/hwc/matfile/sub_feature.mat", struct_as_record=False, squeeze_me=True)
    #good_data = loadmat("/work/hwc/matfile/sub_normal.mat", struct_as_record=False, squeeze_me=True)
    #bad = bad_data['sub_feature'][:,:18]
    #good = good_data['sub_normal'][:,:18]

    #bad = bad_data['features'][:,:18]
    #good = good_data['normals'][:,:18]
    #print(bad[-1,:])
    train_and_export_xgb(good, bad, out_dir=output_path, use_gpu=False)
