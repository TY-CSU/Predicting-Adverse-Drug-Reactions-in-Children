import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    f1_score, matthews_corrcoef, balanced_accuracy_score,
    recall_score
)
from sklearn.model_selection import StratifiedKFold, train_test_split

import xgboost as xgb

# =========================
# 可配置参数
# =========================

FEATURES_PATH = '/public/home/tianyao/biosignature/features.csv'
DATA_PATH     = '/public/home/tianyao/xgboost/433_labeled_results_with_smiles.csv'

# 从第几个任务开始（1-based），用于断点续跑
START_TASK_ID = 1

# 分层 K 折交叉验证设置
N_SPLITS = 10
SHUFFLE = True
RANDOM_STATE = 42

# XGBoost 基础超参（不使用特征选择，也不做 Optuna）
XGB_PARAMS = dict(
    tree_method='hist',        
    objective='binary:logistic',
    eval_metric='auc',
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    n_estimators=500,
    reg_alpha=0.0,
    reg_lambda=1.0
)
USE_GPU = False  

# 输出目录
OUTPUT_ROOT = 'xgb_cv_biofeat'
os.makedirs(OUTPUT_ROOT, exist_ok=True)

USE_EXTERNAL_HOLDOUT = True
EXTERNAL_TEST_SIZE = 0.1

# ==============
# 工具函数
# ==============

def log_message(message: str):
    """带时间戳的日志输出"""
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{ts}] {message}")
    sys.stdout.flush()

def parse_biofeat(biofeat_str):
    """将 BioFeat 的 JSON 字符串解析为一维 numpy 数组"""
    try:
        arr = np.array(json.loads(biofeat_str))
        return arr.flatten()
    except Exception as e:
        log_message(f"BioFeat 解析失败: {str(e)}; 样例片段: {str(biofeat_str)[:120]} ...")
        return None

def evaluate_model(y_true, y_pred, y_pred_proba):
    """稳健计算各类评估指标（异常时返回 NaN）"""
    metrics = {}
    try:
        metrics['ROC AUC'] = roc_auc_score(y_true, y_pred_proba)
    except Exception:
        metrics['ROC AUC'] = np.nan
    try:
        metrics['PR AUC'] = average_precision_score(y_true, y_pred_proba)
    except Exception:
        metrics['PR AUC'] = np.nan
    try:
        metrics['Accuracy'] = accuracy_score(y_true, y_pred)
    except Exception:
        metrics['Accuracy'] = np.nan
    try:
        metrics['F1'] = f1_score(y_true, y_pred)
    except Exception:
        metrics['F1'] = np.nan
    try:
        metrics['MCC'] = matthews_corrcoef(y_true, y_pred)
    except Exception:
        metrics['MCC'] = np.nan
    try:
        metrics['Balanced Accuracy'] = balanced_accuracy_score(y_true, y_pred)
    except Exception:
        metrics['Balanced Accuracy'] = np.nan
    try:
        metrics['Sensitivity'] = recall_score(y_true, y_pred, pos_label=1)
    except Exception:
        metrics['Sensitivity'] = np.nan
    try:
        metrics['Specificity'] = recall_score(y_true, y_pred, pos_label=0)
    except Exception:
        metrics['Specificity'] = np.nan
    return metrics

# =====
# 主流程
# =====

def main():
    try:
        log_message("启动：XGBoost（无特征选择）+ 分层10折交叉验证（BioFeat）")

        # 读取 CSV
        log_message("读取输入 CSV ...")
        df_labels = pd.read_csv(FEATURES_PATH)  # 含标签与 Smiles
        df_feats  = pd.read_csv(DATA_PATH)      # 含 Smiles 与 BioFeat

        # 依据 'Smiles' 合并
        log_message("按 'Smiles' 进行合并 ...")
        df = pd.merge(df_labels, df_feats, on='Smiles', how='inner')
        log_message(f"合并后形状: {df.shape}")

        # 解析 BioFeat -> Fingerprints（数值特征向量）
        log_message("解析 BioFeat 到数值数组 ...")
        if 'BioFeat' not in df.columns:
            raise KeyError("合并结果中不包含 'BioFeat' 列，请确认 DATA_PATH 文件中含有该列。")
        df['Fingerprints'] = df['BioFeat'].apply(parse_biofeat)
        df = df.dropna(subset=['Fingerprints'])

        # 组装特征矩阵 X
        X_all = np.vstack(df['Fingerprints'].values)

        # 识别任务列（标签列）：排除特征及辅助列
        exclude_cols = {'Smiles', 'BioFeat', 'Fingerprints'}
        task_cols = [c for c in df.columns if c not in exclude_cols]

        if len(task_cols) == 0:
            raise ValueError("未找到标签列（任务列）。请确认 FEATURES_PATH 文件中包含 ADR 标签列。")

        # 为数据集创建输出目录
        dataset_name = os.path.splitext(os.path.basename(DATA_PATH))[0]
        dataset_dir = os.path.join(OUTPUT_ROOT, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)

        # 全局汇总日志
        log_path = os.path.join(OUTPUT_ROOT, f'cv_summary_{dataset_name}.txt')
        mode = 'a' if os.path.exists(log_path) and START_TASK_ID > 1 else 'w'
        with open(log_path, mode) as logf:
            if mode == 'a':
                logf.write("\n" + "=" * 80 + "\n")
                logf.write(f"恢复执行，从任务 {START_TASK_ID} 开始\n")
                logf.write("=" * 80 + "\n\n")

            # 遍历每个任务（ADR 端点）
            for task_idx, task_name in enumerate(task_cols, start=1):
                if task_idx < START_TASK_ID:
                    continue

                log_message(f"处理任务 {task_idx}/{len(task_cols)}: {task_name}")
                task_dir = os.path.join(dataset_dir, f'task_{task_idx:04d}_{task_name}')
                os.makedirs(task_dir, exist_ok=True)

                # 取出该任务的标签，并过滤 NaN
                y_all = df[task_name].values
                mask = ~np.isnan(y_all)
                X = X_all[mask]
                y = y_all[mask]

                # 有效性检查（至少包含两个类别）
                if len(y) == 0 or len(np.unique(y)) < 2:
                    log_message(f"任务 {task_idx}（{task_name}）无效（样本为空或仅单一类别），跳过。")
                    with open(os.path.join(dataset_dir, 'last_completed_task.txt'), 'w') as ck:
                        ck.write(str(task_idx))
                    continue

                # 可选：外部留出集（先划分出一部分样本，剩余做 CV）
                if USE_EXTERNAL_HOLDOUT:
                    X_train_dev, X_holdout, y_train_dev, y_holdout = train_test_split(
                        X, y,
                        test_size=EXTERNAL_TEST_SIZE,
                        stratify=y,
                        random_state=RANDOM_STATE
                    )
                else:
                    X_train_dev, y_train_dev = X, y
                    X_holdout, y_holdout = None, None

                # 分层 10 折交叉验证
                kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=SHUFFLE, random_state=RANDOM_STATE)
                fold_metrics_list = []
                fold_pred_records = []

                for fold_id, (tr_idx, te_idx) in enumerate(kf.split(X_train_dev, y_train_dev), start=1):
                    X_tr = X_train_dev[tr_idx]
                    y_tr = y_train_dev[tr_idx]
                    X_te = X_train_dev[te_idx]
                    y_te = y_train_dev[te_idx]

                    # 不做特征选择，直接训练 XGBoost
                    params = XGB_PARAMS.copy()
                    if USE_GPU:
                        params['tree_method'] = 'gpu_hist'
                    model = xgb.XGBClassifier(**params)

                    # 使用早停避免过拟合（验证折做监控）
                    model.fit(
                        X_tr, y_tr,
                        eval_set=[(X_te, y_te)],
                        verbose=False,
                        early_stopping_rounds=20
                    )

                    # 验证折预测
                    y_te_proba = model.predict_proba(X_te)[:, 1]
                    y_te_pred = (y_te_proba > 0.5).astype(int)

                    # 计算评估指标
                    m = evaluate_model(y_te, y_te_pred, y_te_proba)
                    fold_metrics_list.append({'fold': fold_id, **m})

                    # 记录每折预测
                    fold_pred_records.append(pd.DataFrame({
                        'fold': fold_id,
                        'y_true': y_te,
                        'y_pred': y_te_pred,
                        'y_proba': y_te_proba
                    }))

                    log_message(f"任务 {task_idx} | 折 {fold_id} 指标: {m}")

                # 保存每折指标
                cv_df = pd.DataFrame(fold_metrics_list)
                cv_df.to_csv(os.path.join(task_dir, 'cv_fold_metrics.csv'), index=False)

                # 保存每折预测
                preds_df = pd.concat(fold_pred_records, ignore_index=True)
                preds_df.to_csv(os.path.join(task_dir, 'cv_fold_predictions.csv'), index=False)

                # 写入任务级汇总（均值 ± 标准差）
                with open(os.path.join(task_dir, 'cv_summary.txt'), 'w') as tf:
                    tf.write(f"Task: {task_name}\n")
                    tf.write("Stratified 10-Fold CV（均值 ± 标准差）:\n")
                    for metric in ['ROC AUC', 'PR AUC', 'Accuracy', 'F1', 'MCC',
                                   'Balanced Accuracy', 'Sensitivity', 'Specificity']:
                        mean_val = cv_df[metric].mean()
                        std_val = cv_df[metric].std()
                        tf.write(f"- {metric}: {mean_val:.6f} ± {std_val:.6f}\n")

                # 如启用外部留出集：在全部 train_dev 上重训并评估
                if USE_EXTERNAL_HOLDOUT:
                    params = XGB_PARAMS.copy()
                    if USE_GPU:
                        params['tree_method'] = 'gpu_hist'
                    final_model = xgb.XGBClassifier(**params)
                    final_model.fit(X_train_dev, y_train_dev)

                    y_hold_proba = final_model.predict_proba(X_holdout)[:, 1]
                    y_hold_pred = (y_hold_proba > 0.5).astype(int)
                    hold_metrics = evaluate_model(y_holdout, y_hold_pred, y_hold_proba)

                    # 保存外部留出集结果
                    pd.DataFrame([hold_metrics]).to_csv(os.path.join(task_dir, 'external_holdout_metrics.csv'), index=False)
                    pd.DataFrame({
                        'y_true': y_holdout,
                        'y_pred': y_hold_pred,
                        'y_proba': y_hold_proba
                    }).to_csv(os.path.join(task_dir, 'external_holdout_predictions.csv'), index=False)

                # 写入全局汇总日志
                with open(log_path, 'a' if mode == 'a' else 'w') as lg:
                    if mode != 'a':
                        lg.write("")
                    lg.write(f"\nTask {task_idx} ({task_name}) CV metrics (mean ± std):\n")
                    for metric in ['ROC AUC', 'PR AUC', 'Accuracy', 'F1', 'MCC',
                                   'Balanced Accuracy', 'Sensitivity', 'Specificity']:
                        mean_val = cv_df[metric].mean()
                        std_val = cv_df[metric].std()
                        lg.write(f"{metric}: {mean_val:.6f} ± {std_val:.6f}\n")
                    lg.write("-" * 60 + "\n")

                # 记录断点
                with open(os.path.join(dataset_dir, 'last_completed_task.txt'), 'w') as ck:
                    ck.write(str(task_idx))

                log_message(f"完成任务 {task_idx}: {task_name}")

        log_message("全部任务处理完成。")

    except Exception as e:
        log_message(f"程序异常终止: {str(e)}")
        import traceback
        log_message(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
