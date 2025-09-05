import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import optuna
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

# Optuna 设置
N_TRIALS = 100           # Optuna 搜索次数
USE_GPU = False          # 是否使用 GPU 加速
OPTUNA_TIMEOUT = 7200    # 单个任务的超参搜索时间限制（秒），None 为无限制
# Optuna 优化目标，0.6*ROC_AUC + 0.4*PR_AUC
OPTUNA_OBJECTIVE_WEIGHTS = {
    'roc_auc': 0.6,
    'pr_auc': 0.4
}

# 输出目录
OUTPUT_ROOT = 'xgb_optuna_cv_biofeat'
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# 是否先划出外部留出集（CV 之前）
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

def objective(trial, X, y):
    """Optuna 优化目标函数，基于分层 10 折交叉验证"""
    # 定义超参数搜索空间
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'tree_method': 'gpu_hist' if USE_GPU else 'hist',
        'verbosity': 0,
        
        # 超参数搜索范围
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 10.0),
    }
    
    # 分层 10 折交叉验证
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    roc_scores = []
    pr_scores = []
    
    for train_idx, val_idx in kf.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        # 检查验证集中是否至少有两个类别
        if len(np.unique(y_val)) < 2:
            continue
            
        # 训练模型
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=25,
            verbose=False
        )
        
        # 预测并评估
        y_val_proba = model.predict_proba(X_val)[:, 1]
        try:
            roc_auc = roc_auc_score(y_val, y_val_proba)
            pr_auc = average_precision_score(y_val, y_val_proba)
            roc_scores.append(roc_auc)
            pr_scores.append(pr_auc)
        except Exception:
            pass
    
    # 如果所有折都无法评估，返回一个很低的分数
    if len(roc_scores) == 0:
        return -1.0
    
    # 计算加权平均分数
    mean_roc_auc = np.mean(roc_scores)
    mean_pr_auc = np.mean(pr_scores)
    
    # 加权组合分数作为优化目标
    weighted_score = (
        OPTUNA_OBJECTIVE_WEIGHTS['roc_auc'] * mean_roc_auc +
        OPTUNA_OBJECTIVE_WEIGHTS['pr_auc'] * mean_pr_auc
    )
    
    return weighted_score

# =====
# 主流程
# =====

def main():
    try:
        log_message("启动：XGBoost + Optuna 超参优化 + 分层10折交叉验证（BioFeat）")

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

            # 遍历每个任务
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

                # ============ Optuna 超参优化 ============
                log_message(f"启动 Optuna 超参优化 (任务 {task_idx}: {task_name}), 共 {N_TRIALS} 次尝试...")
                
                study = optuna.create_study(
                    direction='maximize',
                    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
                )
                
                study.optimize(
                    lambda trial: objective(trial, X_train_dev, y_train_dev),
                    n_trials=N_TRIALS,
                    timeout=OPTUNA_TIMEOUT
                )
                
                best_params = study.best_params
                best_value = study.best_value
                
                log_message(f"Optuna 优化完成: 最佳分数 = {best_value:.6f}")
                log_message(f"最佳超参: {best_params}")
                
                # 保存优化结果
                with open(os.path.join(task_dir, 'optuna_best_params.json'), 'w') as f:
                    json.dump({
                        'best_params': best_params,
                        'best_score': best_value,
                        'n_trials': N_TRIALS,
                        'weights': OPTUNA_OBJECTIVE_WEIGHTS
                    }, f, indent=2)

                # 把最佳超参加入到 XGB 配置
                best_xgb_params = {
                    'objective': 'binary:logistic',
                    'eval_metric': 'auc',
                    'tree_method': 'gpu_hist' if USE_GPU else 'hist',
                    'verbosity': 0,
                    **best_params
                }

                # ============ 分层 10 折交叉验证 ============
                log_message(f"使用最佳超参进行 {N_SPLITS} 折交叉验证评估...")
                
                kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=SHUFFLE, random_state=RANDOM_STATE)
                fold_metrics_list = []
                fold_pred_records = []

                for fold_id, (tr_idx, te_idx) in enumerate(kf.split(X_train_dev, y_train_dev), start=1):
                    X_tr = X_train_dev[tr_idx]
                    y_tr = y_train_dev[tr_idx]
                    X_te = X_train_dev[te_idx]
                    y_te = y_train_dev[te_idx]
                    
                    # 使用最佳超参训练模型
                    model = xgb.XGBClassifier(**best_xgb_params)
                    
                    # 使用早停避免过拟合
                    model.fit(
                        X_tr, y_tr,
                        eval_set=[(X_te, y_te)],
                        verbose=False,
                        early_stopping_rounds=25
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
                    tf.write(f"超参优化: Optuna ({N_TRIALS} 次尝试), 最佳分数: {best_value:.6f}\n")
                    tf.write(f"最佳超参: {json.dumps(best_params, indent=2)}\n\n")
                    tf.write(f"Stratified 10-Fold CV 评估结果（均值 ± 标准差）:\n")
                    for metric in ['ROC AUC', 'PR AUC', 'Accuracy', 'F1', 'MCC',
                                   'Balanced Accuracy', 'Sensitivity', 'Specificity']:
                        mean_val = cv_df[metric].mean()
                        std_val = cv_df[metric].std()
                        tf.write(f"- {metric}: {mean_val:.6f} ± {std_val:.6f}\n")

                if USE_EXTERNAL_HOLDOUT:
                    # 在全部训练开发集上训练最终模型
                    final_model = xgb.XGBClassifier(**best_xgb_params)
                    final_model.fit(X_train_dev, y_train_dev)

                    # 在外部留出测试集上评估
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
                    
                    # 保存最终模型
                    model_path = os.path.join(task_dir, 'final_model.joblib')
                    joblib.dump(final_model, model_path)
                    log_message(f"保存最终模型到: {model_path}")

                # 写入全局汇总日志
                with open(log_path, 'a' if mode == 'a' else 'w') as lg:
                    if mode != 'a':
                        lg.write("")
                    lg.write(f"\nTask {task_idx} ({task_name}):\n")
                    lg.write(f"Optuna 最佳分数: {best_value:.6f}\n")
                    lg.write(f"CV metrics (mean ± std):\n")
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
