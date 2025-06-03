import os  
import joblib  
import pandas as pd  
import numpy as np  
import cupy as cp  
import json  
from sklearn.model_selection import StratifiedKFold  
from sklearn.metrics import (  
    roc_auc_score,  
    average_precision_score,  
    accuracy_score,  
    f1_score,  
    matthews_corrcoef,  
    roc_curve  
)  
import xgboost as xgb  
import optuna  
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split  
from scipy.interpolate import interp1d  

 

# 定义数据集路径  
data_path = '/public/home/tianyao/biosignature/features.csv'  
features_path = '/public/home/tianyao/xgboost/433_labeled_results_with_smiles.csv'  

def parse_biofeat(biofeat_str):  
    """解析 BioFeat 字符串为 NumPy 数组。支持 JSON 格式的字符串"""  
    try:  
        return np.array(json.loads(biofeat_str))  
    except (ValueError, json.JSONDecodeError):  
        print(f"无法解析的特征: {biofeat_str}")  
        return None  

def auc_eval(preds, dtrain):  
    labels = dtrain.get_label()  
    preds = 1.0 / (1.0 + np.exp(-preds))  
    auc = roc_auc_score(labels, preds)  
    return 'auc', auc  

def objective(trial, X, y):  
    param = {  
        'tree_method': 'hist',  
        'device': 'gpu',  
        'objective': 'binary:logistic',  
        'predictor': 'gpu_predictor',  
        'eval_metric': 'auc',  
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),  
        'max_depth': trial.suggest_int('max_depth', 3, 8),  
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),  
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),  
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),  
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),  
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 10)  
    }  

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)  
    scores = []  

    for train_index, val_index in skf.split(cp.asnumpy(X), cp.asnumpy(y)):  
        X_train_kf, X_val_kf = X[train_index], X[val_index]  
        y_train_kf, y_val_kf = y[train_index], y[val_index]  

        if len(cp.unique(y_val_kf)) < 2:  
            continue  

        X_train_kf_np = cp.asnumpy(X_train_kf)  
        y_train_kf_np = cp.asnumpy(y_train_kf)  
        X_val_kf_np = cp.asnumpy(X_val_kf)  
        y_val_kf_np = cp.asnumpy(y_val_kf)  

        model = xgb.XGBClassifier(  
            **param,  
            use_label_encoder=False,  
            early_stopping_rounds=10,  
            verbosity=0  
        )  

        model.fit(  
            X_train_kf_np, y_train_kf_np,  
            eval_set=[(X_val_kf_np, y_val_kf_np)],  
            verbose=False  
        )  

        y_val_pred_proba = model.predict_proba(X_val_kf_np)[:, 1]  

        try:  
            roc_auc = roc_auc_score(y_val_kf_np, y_val_pred_proba)  
        except ValueError:  
            roc_auc = 0  

        prc = average_precision_score(y_val_kf_np, y_val_pred_proba)  
        score = 0.4 * prc + 0.6 * roc_auc  
        scores.append(score)  

    return np.mean(scores) if scores else -1  

# 读取特征文件  
features_df = pd.read_csv(features_path)  
print("特征文件列名:", features_df.columns)  

# 读取数据集  
data = pd.read_csv(data_path)  
print("数据集列名:", data.columns)  

# 合并数据  
data = pd.merge(data, features_df, on='Smiles', how='inner')  

# 解析特征，并去除无效行  
data['Fingerprints'] = data['BioFeat'].apply(parse_biofeat)  
data = data.dropna(subset=['Fingerprints'])  

# 将每个 FingerprintsList 转换为二维数组  
data['Fingerprints'] = data['Fingerprints'].apply(lambda x: x.flatten() if x is not None else None)  

# 找出 ADR 终点列（排除 Smiles 和 Fingerprints）  
adr_columns = [col for col in data.columns if col not in ['Smiles', 'BioFeat', 'Fingerprints']]  

# 将 Fingerprints 列转换为二维数组  
X = np.vstack(data['Fingerprints'].values)  # 从 DataFrame 提取特征并转为二维数组  
y = cp.array(data[adr_columns].values)  

# 确保数据形状符合要求  
print("特征形状:", X.shape)  
print("标签形状:", y.shape)  

# 设置测试集比例  
TEST_SIZE = 0.1  

# 数据集名称  
dataset_name = os.path.splitext(os.path.basename(data_path))[0]  
dataset_dir = os.path.join('xg_biofeat_ablation', dataset_name)  
os.makedirs(dataset_dir, exist_ok=True)  

output_file_name = os.path.join('xg_biofeat_ablation', f'xb_biofeat_ablation{dataset_name}.txt')  

with open(output_file_name, 'w') as f:  
    for task_id in range(y.shape[1]):  
        task_dir = os.path.join(dataset_dir, f'task_{task_id + 1}')  
        os.makedirs(task_dir, exist_ok=True)  

        current_y = y[:, task_id]  
        print(  
            f"数据集 {dataset_name} 的任务 {task_id + 1} 的标签分布：{np.unique(cp.asnumpy(current_y), return_counts=True)}")  

        mask = ~cp.isnan(current_y)  

        # 将 mask 转换为 NumPy 数组  
        mask_np = cp.asnumpy(mask)  

        # 使用 NumPy 数组进行索引  
        X_task = X[mask_np]  # 这里直接用 NumPy 的数组  
        y_task = current_y[mask]  

        if len(y_task) == 0 or len(cp.unique(y_task)) < 2:  
            print(f"数据集 {dataset_name} 的任务 {task_id + 1} 数据无效，跳过")  
            continue  

        # Use .get() to convert CuPy to NumPy  
        X_task_np = X_task  # 这里 X_task 已经是 NumPy 数组  
        y_task_np = y_task.get()  # 创建 NumPy 数组  

        # 划分训练集和测试集  
        X_train_dev, X_test, y_train_dev, y_test = train_test_split(  
            X_task_np, y_task_np,  
            test_size=TEST_SIZE,  
            stratify=y_task_np,  
            random_state=42  
        )  

        # 保存测试集数据  
        test_data_path = os.path.join(task_dir, 'test_data.npz')  
        np.savez(test_data_path, X_test=X_test, y_test=y_test)  

        # 保持张量属性  
        X_train_dev = cp.array(X_train_dev)  
        X_test = cp.array(X_test)  
        y_train_dev = cp.array(y_train_dev)  
        y_test = cp.array(y_test)  

        study = optuna.create_study(direction='maximize')  
        study.optimize(lambda trial: objective(trial, X_train_dev, y_train_dev), n_trials=200)  

        best_params = study.best_params  
        best_value = study.best_value  

        best_params.update({  
            'tree_method': 'hist',  
            'device': 'cuda',  
            'objective': 'binary:logistic',  
            'predictor': 'gpu_predictor',  
            'eval_metric': 'auc',  
            'verbosity': 0  
        })  

        final_model = xgb.XGBClassifier(**best_params)  

        X_train_dev_np = cp.asnumpy(X_train_dev)  
        y_train_dev_np = cp.asnumpy(y_train_dev)  
        X_test_np = cp.asnumpy(X_test)  
        y_test_np = cp.asnumpy(y_test)  

        final_model.fit(X_train_dev_np, y_train_dev_np)  

        y_test_pred_proba = final_model.predict_proba(X_test_np)[:, 1]  
        y_test_pred = y_test_pred_proba > 0.5  

        test_roc_auc = roc_auc_score(y_test_np, y_test_pred_proba)  
        test_prc = average_precision_score(y_test_np, y_test_pred_proba)  
        test_accuracy = accuracy_score(y_test_np, y_test_pred)  
        test_f1 = f1_score(y_test_np, y_test_pred)  
        test_mcc = matthews_corrcoef(y_test_np, y_test_pred)  

        test_metrics = {  
            'ROC AUC': test_roc_auc,  
            'Precision-Recall AUC': test_prc,  
            'Accuracy': test_accuracy,  
            'F1 Score': test_f1,  
            'MCC': test_mcc  
        }  

        # 保存测试集性能指标  
        test_metrics_path = os.path.join(task_dir, 'test_metrics.csv')  
        pd.DataFrame([test_metrics]).to_csv(test_metrics_path, index=False)  

        # 记录测试集性能指标  
        f.write(f"\n任务 {task_id + 1} 的测试集性能指标：\n")  
        for metric_name, value in test_metrics.items():  
            f.write(f"{metric_name}: {value:.4f}\n")  
        f.write('-' * 50 + '\n')  

        # 保存最终模型  
        model_path = os.path.join(task_dir, 'final_model.joblib')  
        joblib.dump(final_model, model_path)  

        # 绘制测试集的 ROC 曲线  
        plt.figure(figsize=(8, 6))  
        fpr, tpr, _ = roc_curve(y_test_np, y_test_pred_proba)  

        # 绘制 ROC 曲线  
        plt.figure(figsize=(8, 6))  
        plt.plot(fpr, tpr, label=f'Test ROC (AUC = {test_roc_auc:.2f})')  
        plt.plot([0, 1], [0, 1], linestyle='--', color='r')  # 参考线  
        plt.xlabel('False Positive Rate')  
        plt.ylabel('True Positive Rate')  
        plt.title(f'Test ROC Curve for {dataset_name} Task {task_id + 1}')  
        plt.legend()  
        plt.grid(True)  
        plt.tight_layout()  

        # 保存图像  
        test_roc_plot_path = os.path.join(task_dir, 'test_roc_curve.png')  
        plt.savefig(test_roc_plot_path, dpi=300, bbox_inches='tight')  
        plt.close()  

        print(f"任务 {task_id + 1} 完成，所有文件已保存到: {task_dir}")   

print("所有任务完成！")
