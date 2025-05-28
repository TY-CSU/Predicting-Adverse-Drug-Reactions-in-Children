import os  
import sys  
import time  
import joblib  
import pandas as pd  
import numpy as np  
import json  
from datetime import datetime  
from sklearn.metrics import (  
    roc_auc_score, average_precision_score, accuracy_score,  
    f1_score, matthews_corrcoef, balanced_accuracy_score,  
    recall_score  
)  
from sklearn.model_selection import train_test_split  
from sklearn.feature_selection import SelectKBest, mutual_info_classif  
from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier  

# 在运行主程序之前，添加这段测试代码  
os.environ['TABPFN_MODEL_PATH'] = "/public/home/tianyao/.conda/envs/tabpfn/lib/python3.11/site-packages/tabpfn/models/tabpfn-v2-classifier-od3j1g5m.ckpt"  

# 定义数据集路径   
data_path = '/public/home/tianyao/biosignature/features.csv'  
features_path = '/public/home/tianyao/xgboost/433_labeled_results_with_smiles.csv'  

# 定义从第几个任务开始  
START_TASK_ID = 1  # 从任务270开始  

def log_message(message):  
    """带时间戳的日志输出"""  
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  
    print(f"[{timestamp}] {message}")  
    sys.stdout.flush()  

def parse_biofeat(biofeat_str):  
    """解析 BioFeat 字符串为 NumPy 数组"""  
    try:  
        return np.array(json.loads(biofeat_str))  
    except (ValueError, json.JSONDecodeError) as e:  
        log_message(f"无法解析的特征: {biofeat_str[:100]}... 错误: {str(e)}")  
        return None  

def evaluate_model(y_true, y_pred, y_pred_proba):  
    """评估模型性能"""  
    metrics = {}  
    try:  
        metrics['ROC AUC'] = roc_auc_score(y_true, y_pred_proba)  
        metrics['Precision-Recall AUC'] = average_precision_score(y_true, y_pred_proba)  
        metrics['Accuracy'] = accuracy_score(y_true, y_pred)  
        metrics['F1 Score'] = f1_score(y_true, y_pred)  
        metrics['MCC'] = matthews_corrcoef(y_true, y_pred)  
        metrics['Balanced Accuracy'] = balanced_accuracy_score(y_true, y_pred)  
        metrics['Sensitivity'] = recall_score(y_true, y_pred, pos_label=1)  
        metrics['Specificity'] = recall_score(y_true, y_pred, pos_label=0)  
    except Exception as e:  
        log_message(f"评估指标计算出错: {str(e)}")  
        metrics = {k: None for k in ['ROC AUC', 'Precision-Recall AUC', 'Accuracy',  
                                   'F1 Score', 'MCC', 'Balanced Accuracy',  
                                   'Sensitivity', 'Specificity']}  
    return metrics  

def select_features(X, y, n_features=500):  
    """使用互信息进行特征选择"""  
    log_message(f"开始特征选择，原始特征数量: {X.shape[1]}")  
    selector = SelectKBest(score_func=mutual_info_classif, k=n_features)  
    X_selected = selector.fit_transform(X, y)  
    return X_selected, selector  

def main():  
    try:  
        log_message("程序开始执行，从任务 %d 开始" % START_TASK_ID)  

        # 读取数据  
        log_message("开始读取数据文件...")  
        features_df = pd.read_csv(features_path)  
        data = pd.read_csv(data_path)  

        # 合并数据  
        data = pd.merge(data, features_df, on='Smiles', how='inner')  
        log_message(f"数据合并完成，形状: {data.shape}")  

        # 处理特征  
        data['Fingerprints'] = data['BioFeat'].apply(parse_biofeat)  
        data = data.dropna(subset=['Fingerprints'])  
        data['Fingerprints'] = data['Fingerprints'].apply(lambda x: x.flatten() if x is not None else None)  

        # 准备特征和标签  
        X = np.vstack(data['Fingerprints'].values)  
        adr_columns = [col for col in data.columns if col not in ['Smiles', 'BioFeat', 'Fingerprints']]  
        y = data[adr_columns].values  

        # 创建输出目录  
        dataset_name = os.path.splitext(os.path.basename(data_path))[0]  
        dataset_dir = os.path.join('tabpfn_auto_biofeat_hpt', dataset_name)  
        os.makedirs(dataset_dir, exist_ok=True)  

        # 创建或追加输出文件  
        output_file_name = os.path.join('tabpfn_auto_biofeat_hpt', f'tabpfn_auto_biofeat_hpt{dataset_name}.txt')  
        
        # 如果文件已存在且我们是恢复模式，就使用追加模式打开  
        file_mode = 'a' if os.path.exists(output_file_name) and START_TASK_ID > 0 else 'w'  
        
        with open(output_file_name, file_mode) as f:  
            # 如果是追加模式，添加分隔符  
            if file_mode == 'a':  
                f.write("\n" + "="*70 + "\n")  
                f.write(f"恢复执行，从任务 {START_TASK_ID} 开始\n")  
                f.write("="*70 + "\n\n")  
            
            for task_id in range(y.shape[1]):  
                # 跳过已经完成的任务  
                if task_id < START_TASK_ID - 1:  # -1 是因为任务ID从0开始，但显示从1开始  
                    continue  
                    
                log_message(f"\n开始处理任务 {task_id + 1}/{y.shape[1]}")  

                task_dir = os.path.join(dataset_dir, f'task_{task_id + 1}')  
                os.makedirs(task_dir, exist_ok=True)  

                # 获取当前任务的数据  
                current_y = y[:, task_id]  
                mask = ~np.isnan(current_y)  
                X_task = X[mask]  
                y_task = current_y[mask]  

                # 检查数据有效性  
                if len(y_task) == 0 or len(np.unique(y_task)) < 2:  
                    log_message(f"任务 {task_id + 1} ({adr_columns[task_id]}) 数据无效，跳过")  
                    continue  

                # 数据集划分  
                X_train_dev, X_test, y_train_dev, y_test = train_test_split(  
                    X_task, y_task,  
                    test_size=0.1,  
                    stratify=y_task,  
                    random_state=42  
                )  

                # 特征选择  
                X_train_selected, selector = select_features(X_train_dev, y_train_dev)  
                X_test_selected = selector.transform(X_test)  

                # 训练模型  
                log_message("开始训练模型...")  
                model = AutoTabPFNClassifier(device='auto', max_time=600)  
                model.fit(X_train_selected, y_train_dev)  

                # 模型评估  
                y_test_pred_proba = model.predict_proba(X_test_selected)[:, 1]  
                y_test_pred = y_test_pred_proba > 0.5  

                # 计算评估指标  
                test_metrics = evaluate_model(y_test, y_test_pred, y_test_pred_proba)  

                # 注释掉模型保存相关代码，因为模型文件过大（约1GB/任务）  
                # joblib.dump(model, os.path.join(task_dir, 'final_model.joblib'))  

                # 保存预测结果和测试集数据  
                np.savez(os.path.join(task_dir, 'test_data.npz'),  
                        X_test=X_test_selected,  
                        y_test=y_test,  
                        y_pred=y_test_pred,          # 添加预测标签  
                        y_pred_proba=y_test_pred_proba  # 添加预测概率  
                        )  

                # 保存测试集性能指标为 CSV 文件  
                test_metrics_path = os.path.join(task_dir, 'test_metrics.csv')  
                pd.DataFrame([test_metrics]).to_csv(test_metrics_path, index=False)  

                # 记录结果到主日志文件  
                f.write(f"\n任务 {task_id + 1} ({adr_columns[task_id]}) 的测试集性能指标：\n")  
                for metric_name, value in test_metrics.items():  
                    f.write(f"{metric_name}: {value if value is not None else '无法计算'}\n")  
                f.write('-' * 50 + '\n')  

                log_message(f"任务 {task_id + 1} 完成")  
                
                # 保存一个检查点标记这个任务已完成  
                with open(os.path.join(dataset_dir, 'last_completed_task.txt'), 'w') as checkpoint:  
                    checkpoint.write(str(task_id + 1))  

        log_message("所有任务处理完成")  

    except Exception as e:  
        log_message(f"程序执行出错: {str(e)}")  
        import traceback  
        log_message(f"错误详情:\n{traceback.format_exc()}")  
        sys.exit(1)  

if __name__ == "__main__":  
    main()  


# 使用TabPFN的特征选择方法
# import os  
# import sys  
# import time  
# import joblib  
# import pandas as pd  
# import numpy as np  
# import json  
# from datetime import datetime  
# from sklearn.metrics import (  
#     roc_auc_score, average_precision_score, accuracy_score,  
#     f1_score, matthews_corrcoef, balanced_accuracy_score,  
#     recall_score  
# )  
# from sklearn.model_selection import train_test_split  
# from tabpfn_extensions import interpretability  # 导入TabPFN的interpretability模块  
# from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier  
# from tabpfn import TabPFNClassifier
# # 在运行主程序之前，添加这段测试代码  
# os.environ['TABPFN_MODEL_PATH'] = "/public/home/tianyao/.conda/envs/tabpfn/lib/python3.11/site-packages/tabpfn/models/tabpfn-v2-classifier-od3j1g5m.ckpt"  

# # 定义数据集路径   
# data_path = '/public/home/tianyao/biosignature/features.csv'  
# features_path = '/public/home/tianyao/xgboost/433_labeled_results_with_smiles.csv'  

# # 定义从第几个任务开始  
# START_TASK_ID = 1  # 从任务346开始  

# def log_message(message):  
#     """带时间戳的日志输出"""  
#     timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  
#     print(f"[{timestamp}] {message}")  
#     sys.stdout.flush()  

# def parse_biofeat(biofeat_str):  
#     """解析 BioFeat 字符串为 NumPy 数组"""  
#     try:  
#         return np.array(json.loads(biofeat_str))  
#     except (ValueError, json.JSONDecodeError) as e:  
#         log_message(f"无法解析的特征: {biofeat_str[:100]}... 错误: {str(e)}")  
#         return None  

# def evaluate_model(y_true, y_pred, y_pred_proba):  
#     """评估模型性能"""  
#     metrics = {}  
#     try:  
#         metrics['ROC AUC'] = roc_auc_score(y_true, y_pred_proba)  
#         metrics['Precision-Recall AUC'] = average_precision_score(y_true, y_pred_proba)  
#         metrics['Accuracy'] = accuracy_score(y_true, y_pred)  
#         metrics['F1 Score'] = f1_score(y_true, y_pred)  
#         metrics['MCC'] = matthews_corrcoef(y_true, y_pred)  
#         metrics['Balanced Accuracy'] = balanced_accuracy_score(y_true, y_pred)  
#         metrics['Sensitivity'] = recall_score(y_true, y_pred, pos_label=1)  
#         metrics['Specificity'] = recall_score(y_true, y_pred, pos_label=0)  
#     except Exception as e:  
#         log_message(f"评估指标计算出错: {str(e)}")  
#         metrics = {k: None for k in ['ROC AUC', 'Precision-Recall AUC', 'Accuracy',  
#                                    'F1 Score', 'MCC', 'Balanced Accuracy',  
#                                    'Sensitivity', 'Specificity']}  
#     return metrics  
# def select_features_tabpfn(X, y, n_features=500):  
#     """先用互信息降维，然后使用TabPFN interpretability模块进行特征选择"""  
#     log_message(f"开始特征选择，原始特征数量: {X.shape[1]}")  
    
#     # 第一步：如果特征数量超过500，先用互信息方法将特征降至500  
#     if X.shape[1] > 500:  
#         log_message("特征数量超过TabPFN限制，先使用互信息方法降至500个特征")  
#         from sklearn.feature_selection import SelectKBest, mutual_info_classif  
#         pre_selector = SelectKBest(mutual_info_classif, k=500)  
#         X_reduced = pre_selector.fit_transform(X, y)  
#         log_message(f"互信息降维后的特征数量: {X_reduced.shape[1]}")  
#     else:  
#         X_reduced = X  
#         pre_selector = None  
    
#     # 第二步：使用TabPFN的特征选择方法  
#     try:  
#         # 初始化TabPFN模型  
#         clf = TabPFNClassifier(device='cuda',n_estimators=3)  
        
#         # 生成特征名称  
#         feature_names = [f"feature_{i}" for i in range(X_reduced.shape[1])]  
        
#         # 关键修改：确保n_features_to_select小于当前特征数量  
#         # 选择当前特征数量的80%或最多400个特征，以确保不会触发错误  
#         n_features_to_select = min(256, int(X_reduced.shape[1] * 0.8))  
        
#         log_message(f"开始使用TabPFN进行特征选择，目标选择 {n_features_to_select} 个特征...")  
#         feature_selector = interpretability.feature_selection.feature_selection(  
#             estimator=clf,  
#             X=X_reduced,  
#             y=y,  
#             n_features_to_select=n_features_to_select,  
#             feature_names=feature_names  
#         )  
        
#         # 构建完整的特征选择管道  
#         if pre_selector is not None:  
#             from sklearn.pipeline import Pipeline  
#             # 创建一个包含两个步骤的特征选择管道  
#             pipeline = Pipeline([  
#                 ('pre_selection', pre_selector),  
#                 ('tabpfn_selection', feature_selector)  
#             ])  
            
#             # 获取选择的特征  
#             X_selected = pipeline.transform(X)  
#             log_message(f"最终选择了 {X_selected.shape[1]} 个特征")  
            
#             return X_selected, pipeline  
#         else:  
#             # 如果没有预选择步骤，直接返回TabPFN的特征选择结果  
#             X_selected = feature_selector.transform(X)  
#             log_message(f"TabPFN特征选择完成，选择了 {X_selected.shape[1]} 个特征")  
#             return X_selected, feature_selector  
    
#     except Exception as e:  
#         log_message(f"TabPFN特征选择失败: {str(e)}，回退到互信息特征选择")  
#         # 如果TabPFN特征选择失败，使用互信息特征选择作为备选方案  
#         from sklearn.feature_selection import SelectKBest, mutual_info_classif  
#         selector = SelectKBest(mutual_info_classif, k=min(n_features, X.shape[1]))  
#         X_selected = selector.fit_transform(X, y)  
#         log_message(f"使用互信息方法选择了 {X_selected.shape[1]} 个特征")  
#         return X_selected, selector  

# def main():  
#     try:  
#         log_message("程序开始执行，从任务 %d 开始" % START_TASK_ID)  

#         # 读取数据  
#         log_message("开始读取数据文件...")  
#         features_df = pd.read_csv(features_path)  
#         data = pd.read_csv(data_path)  

#         # 合并数据  
#         data = pd.merge(data, features_df, on='Smiles', how='inner')  
#         log_message(f"数据合并完成，形状: {data.shape}")  

#         # 处理特征  
#         data['Fingerprints'] = data['BioFeat'].apply(parse_biofeat)  
#         data = data.dropna(subset=['Fingerprints'])  
#         data['Fingerprints'] = data['Fingerprints'].apply(lambda x: x.flatten() if x is not None else None)  

#         # 准备特征和标签  
#         X = np.vstack(data['Fingerprints'].values)  
#         adr_columns = [col for col in data.columns if col not in ['Smiles', 'BioFeat', 'Fingerprints']]  
#         y = data[adr_columns].values  

#         # 创建输出目录  
#         dataset_name = os.path.splitext(os.path.basename(data_path))[0]  
#         dataset_dir = os.path.join('tabpfn_auto_biofeat_selection', dataset_name)  
#         os.makedirs(dataset_dir, exist_ok=True)  

#         # 创建或追加输出文件  
#         output_file_name = os.path.join('tabpfn_auto_biofeat_selection', f'tabpfn_auto_biofeat_selection{dataset_name}.txt')  
        
#         # 如果文件已存在且我们是恢复模式，就使用追加模式打开  
#         file_mode = 'a' if os.path.exists(output_file_name) and START_TASK_ID > 0 else 'w'  
        
#         with open(output_file_name, file_mode) as f:  
#             # 如果是追加模式，添加分隔符  
#             if file_mode == 'a':  
#                 f.write("\n" + "="*70 + "\n")  
#                 f.write(f"恢复执行，从任务 {START_TASK_ID} 开始\n")  
#                 f.write("="*70 + "\n\n")  
            
#             for task_id in range(y.shape[1]):  
#                 # 跳过已经完成的任务  
#                 if task_id < START_TASK_ID - 1:  # -1 是因为任务ID从0开始，但显示从1开始  
#                     continue  
                    
#                 log_message(f"\n开始处理任务 {task_id + 1}/{y.shape[1]}")  

#                 task_dir = os.path.join(dataset_dir, f'task_{task_id + 1}')  
#                 os.makedirs(task_dir, exist_ok=True)  

#                 # 获取当前任务的数据  
#                 current_y = y[:, task_id]  
#                 mask = ~np.isnan(current_y)  
#                 X_task = X[mask]  
#                 y_task = current_y[mask]  

#                 # 检查数据有效性  
#                 if len(y_task) == 0 or len(np.unique(y_task)) < 2:  
#                     log_message(f"任务 {task_id + 1} ({adr_columns[task_id]}) 数据无效，跳过")  
#                     continue  

#                 # 数据集划分  
#                 X_train_dev, X_test, y_train_dev, y_test = train_test_split(  
#                     X_task, y_task,  
#                     test_size=0.1,  
#                     stratify=y_task,  
#                     random_state=42  
#                 )  

#                 # 特征选择 - 使用TabPFN interpretability模块  
#                 X_train_selected, selector = select_features_tabpfn(X_train_dev, y_train_dev, n_features=256)  
#                 X_test_selected = selector.transform(X_test)  

#                 # 训练模型  
#                 log_message("开始训练模型...")  
#                 model = AutoTabPFNClassifier(device='auto', max_time=120)  
#                 model.fit(X_train_selected, y_train_dev)  

#                 # 模型评估  
#                 y_test_pred_proba = model.predict_proba(X_test_selected)[:, 1]  
#                 y_test_pred = y_test_pred_proba > 0.5  

#                 # 计算评估指标  
#                 test_metrics = evaluate_model(y_test, y_test_pred, y_test_pred_proba)  

#                 # 保存预测结果和测试集数据  
#                 np.savez(os.path.join(task_dir, 'test_data.npz'),  
#                         X_test=X_test_selected,  
#                         y_test=y_test,  
#                         y_pred=y_test_pred,          # 添加预测标签  
#                         y_pred_proba=y_test_pred_proba  # 添加预测概率  
#                         )  

#                 # 保存测试集性能指标为 CSV 文件  
#                 test_metrics_path = os.path.join(task_dir, 'test_metrics.csv')  
#                 pd.DataFrame([test_metrics]).to_csv(test_metrics_path, index=False)  

#                 # 记录结果到主日志文件  
#                 f.write(f"\n任务 {task_id + 1} ({adr_columns[task_id]}) 的测试集性能指标：\n")  
#                 for metric_name, value in test_metrics.items():  
#                     f.write(f"{metric_name}: {value if value is not None else '无法计算'}\n")  
#                 f.write('-' * 50 + '\n')  

#                 log_message(f"任务 {task_id + 1} 完成")  
                
#                 # 保存一个检查点标记这个任务已完成  
#                 with open(os.path.join(dataset_dir, 'last_completed_task.txt'), 'w') as checkpoint:  
#                     checkpoint.write(str(task_id + 1))  

#         log_message("所有任务处理完成")  

#     except Exception as e:  
#         log_message(f"程序执行出错: {str(e)}")  
#         import traceback  
#         log_message(f"错误详情:\n{traceback.format_exc()}")  
#         sys.exit(1)  

# if __name__ == "__main__":  
#     main()  