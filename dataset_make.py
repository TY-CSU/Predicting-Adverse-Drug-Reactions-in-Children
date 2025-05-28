#进行黑框匹配，分情况进行投票，最后给出最终ADRlabel
import pandas as pd
import numpy as np


def create_labels_with_majority_vote(
        input_filepath,
        output_filepath,
        bw_adr_filepath
):
    """
    读取信号检测结果CSV，按照多数投票规则计算最终标签，并生成新的CSV数据集。
    同时读取bw_adr.xls文件，进行FDA黑框ADR的分类，匹配时忽略大小写。
    """
    # 读取CSV文件
    df = pd.read_csv(input_filepath)

    # 读取FDA黑框ADR列表
    bw_adr_df = pd.read_csv(bw_adr_filepath)

    # 确保 'bw_adr' 列存在
    if 'bw_adr' not in bw_adr_df.columns:
        raise ValueError("FDA黑框ADR文件缺少 'bw_adr' 列。")

        # 将bw_adr列表转换为小写，去除空值
    bw_adr_list = bw_adr_df['bw_adr'].dropna().str.lower().unique().tolist()

    # 确保相关列存在
    required_columns = [
        'Drug', 'Adverse Event', 'Reports_Children',
        '95% CI Lower ROR', '95% CI Lower PRR',
        'IC Lower 95% CI', '95% CI Lower EBGM'
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"输入文件缺少必要的列: {missing_columns}")

        # 进行小写转换以确保匹配不区分大小写
    df['Adverse Event_lower'] = df['Adverse Event'].str.lower()

    # 添加 'FDA_Black_Box' 列，根据 'Adverse Event' 是否在 bw_adr_list 中
    df['FDA_Black_Box'] = df['Adverse Event_lower'].isin(bw_adr_list).astype(int)

    # 定义标签条件
    # 灵敏度高的条件（不考虑a）
    df['ROR_Label_No_A'] = np.where(
        df['95% CI Lower ROR'].notnull(),
        np.where(df['95% CI Lower ROR'] > 1, 1, 0),
        np.nan
    )
    df['PRR_Label_No_A'] = np.where(
        df['95% CI Lower PRR'].notnull(),
        np.where(df['95% CI Lower PRR'] > 1, 1, 0),
        np.nan
    )
    df['BCPNN_Label_GT0'] = np.where(
        df['IC Lower 95% CI'].notnull(),
        np.where(df['IC Lower 95% CI'] > 0, 1, 0),
        np.nan
    )

    # 特异度高的条件（考虑a）
    df['ROR_Label'] = np.where(
        df[['Reports_Children', '95% CI Lower ROR']].notnull().all(axis=1),
        np.where((df['Reports_Children'] >= 3) & (df['95% CI Lower ROR'] > 1), 1, 0),
        np.nan
    )
    df['PRR_Label'] = np.where(
        df[['Reports_Children', '95% CI Lower PRR']].notnull().all(axis=1),
        np.where((df['Reports_Children'] >= 3) & (df['95% CI Lower PRR'] > 1), 1, 0),
        np.nan
    )
    df['BCPNN_Label_GT3'] = np.where(
        df['IC Lower 95% CI'].notnull(),
        np.where(df['IC Lower 95% CI'] > 3, 1, 0),
        np.nan
    )

    # EBGM_Label
    df['EBGM_Label'] = np.where(
        df['95% CI Lower EBGM'].notnull(),
        np.where(df['95% CI Lower EBGM'] > 2, 1, 0),
        np.nan
    )

    # Final_Label初始化为空
    df['Final_Label'] = np.nan

    # 应用决策规则
    for idx, row in df.iterrows():
        if row['FDA_Black_Box'] == 1:
            # 灵敏度高的方法 + EBGM05
            signal_methods = [
                row['ROR_Label_No_A'],
                row['PRR_Label_No_A'],
                row['BCPNN_Label_GT0'],
                row['EBGM_Label']
            ]
            # 去除缺失值
            valid_signals = [s for s in signal_methods if not pd.isna(s)]
            # 计算信号数量
            signal_count = sum(valid_signals)
            # 判断是否达到多数
            if signal_count >= 2:
                df.at[idx, 'Final_Label'] = 1
            else:
                df.at[idx, 'Final_Label'] = 0
        elif row['FDA_Black_Box'] == 0:
            # 特异度高的方法 + EBGM05
            signal_methods = [
                row['ROR_Label'],
                row['PRR_Label'],
                row['BCPNN_Label_GT3'],
                row['EBGM_Label']
            ]
            # 去除缺失值
            valid_signals = [s for s in signal_methods if not pd.isna(s)]
            # 计算信号数量
            signal_count = sum(valid_signals)
            # 判断是否达到多数（至少3个信号）
            if signal_count >= 2:
                df.at[idx, 'Final_Label'] = 1
            else:
                df.at[idx, 'Final_Label'] = 0
        else:
            # 未分类的ADR
            df.at[idx, 'Final_Label'] = np.nan

            # 保留所需的列
    output_df = df[['Drug', 'Adverse Event', 'FDA_Black_Box',
                    'ROR_Label_No_A', 'PRR_Label_No_A', 'BCPNN_Label_GT0',
                    'ROR_Label', 'PRR_Label', 'BCPNN_Label_GT3',
                    'EBGM_Label', 'Final_Label']]

    #将结果保存到新的CSV文件
    output_df.to_csv(output_filepath, index=False)
    print(f"标签已计算完成并保存到 {output_filepath}")


if __name__ == "__main__":
    # 输入和输出文件路径
    input_filepath = 'child_14/F_sd_child.csv'
    output_filepath = 'child_14/balenced_F_labeled_results_child14.csv'
    bw_adr_filepath = '14-18/bw_adr.csv'  # FDA黑框ADR文件路径
    # input_filepath = '14-18/F_signal_detection_results_14-18.csv'
    # output_filepath = '14-18/balenced_F_labeled_results_child14-18.csv'
    # bw_adr_filepath = '14-18/bw_adr.csv'  # FDA黑框ADR文件路径
    # 生成带标签的数据集
    create_labels_with_majority_vote(input_filepath, output_filepath, bw_adr_filepath)

#对终点进行过滤选择，得到最终建模的ADR和dataset
import pandas as pd


# def reduce_adrs(
#     input_csv, output_csv, stats_csv, filter_adr_csv,
#     threshold_non_bw=0.8, threshold_bw=0.85, balance_threshold=0.2
# ):
#     """
#     精简ADR终点：加入Balance Ratio的保留逻辑
#     """
#     # 1. 读取数据
#     print("读取主数据集...")
#     df = pd.read_csv(input_csv)
#     print(f"数据包含 {df.shape[0]} 行，{df.shape[1]} 列。")
#
#     # 2. 读取需要过滤删除的ADR终点
#     print("读取需要过滤的ADR终点...")
#     filter_df = pd.read_csv(filter_adr_csv)
#     filter_adrs = filter_df['filter_adr'].dropna().unique().tolist()
#     print(f"共找到 {len(filter_adrs)} 个需要过滤的ADR终点。")
#
#     # 3. 移除需要过滤的ADR终点
#     print("移除过滤的ADR终点...")
#     initial_adr_count = df['Adverse Event'].nunique()
#     df = df[~df['Adverse Event'].isin(filter_adrs)]
#     final_adr_count = df['Adverse Event'].nunique()
#     removed_adr_count = initial_adr_count - final_adr_count
#     print(f"已移除 {removed_adr_count} 个ADR终点。")
#
#     # 4. 提取FDA黑框ADR列表
#     print("提取FDA黑框ADR列表...")
#     black_box_adrs = df[df['FDA_Black_Box'] == 1]['Adverse Event'].unique().tolist()
#     print(f"找到 {len(black_box_adrs)} 个FDA黑框ADR。")
#
#     # 5. 透视表转换数据格式：药物为行，ADR终点为列，值为Final_Label
#     print("转换数据格式为宽格式...")
#     pivot_df = df.pivot_table(index='Drug',
#                               columns='Adverse Event',
#                               values='Final_Label',
#                               aggfunc='last')
#     print(f"转换后的数据形状为：{pivot_df.shape}")
#
#     # 6. 计算每个ADR终点的缺失值比例
#     print("计算每个ADR终点的缺失值比例...")
#     na_ratio = pivot_df.isna().mean()
#
#     # 7. 计算每个ADR终点的 Balance Ratio
#     print("计算每个ADR终点的 Balance Ratio...")
#     balance_ratios = {}
#
#     for adr in pivot_df.columns:
#         adr_data = pivot_df[adr].dropna()
#         count_0 = (adr_data == 0).sum()
#         count_1 = (adr_data == 1).sum()
#         total_count = count_0 + count_1
#         if total_count > 0:
#             balance_ratio = abs(count_1 - count_0) / total_count
#         else:
#             balance_ratio = 1  # 如果当前ADR没有有效数据，设为1（完全不平衡）
#         balance_ratios[adr] = balance_ratio
#     balance_ratios = pd.Series(balance_ratios)
#     print("Balance Ratio 计算完成。")
#
#     # 8. 标识需要保留的ADR终点
#     print("确定需要保留的ADR终点...")
#
#     # 第一步：基于缺失值比例的初步筛选
#     # 黑框ADR筛选
#     keep_black_box_adrs_set = set()
#     if black_box_adrs:
#         valid_black_box_adrs = [adr for adr in black_box_adrs if adr in pivot_df.columns]
#         if valid_black_box_adrs:
#             black_box_na_ratio = na_ratio[valid_black_box_adrs]
#             # 第一轮筛选：基于缺失值比例
#             filtered_na_black_box = black_box_na_ratio <= threshold_bw
#             keep_black_box_adrs_set = set(filtered_na_black_box[filtered_na_black_box].index.tolist())
#             print(f"第一轮筛选后保留 {len(keep_black_box_adrs_set)} 个黑框ADR（基于缺失值比例）。")
#
#             # 非黑框ADR筛选
#     keep_non_black_box_adrs_set = set()
#     non_black_box_adrs = [adr for adr in pivot_df.columns if adr not in black_box_adrs]
#     if non_black_box_adrs:
#         non_black_box_na_ratio = na_ratio[non_black_box_adrs]
#         # 第一轮筛选：基于缺失值比例
#         filtered_na_non_black_box = non_black_box_na_ratio <= threshold_non_bw
#         keep_non_black_box_adrs_set = set(filtered_na_non_black_box[filtered_na_non_black_box].index.tolist())
#         print(f"第一轮筛选后保留 {len(keep_non_black_box_adrs_set)} 个非黑框ADR（基于缺失值比例）。")
#
#         # 第二步：对被筛掉的ADR进行平衡性检查
#     # 获取第一轮被筛掉的ADR
#     excluded_adrs = set(pivot_df.columns) - keep_black_box_adrs_set - keep_non_black_box_adrs_set
#     additional_adrs = set()
#
#     for adr in excluded_adrs:
#         na_ratio_value = na_ratio[adr]
#         balance_ratio_value = balance_ratios[adr]
#
#         # 检查是否满足第二轮筛选条件：平衡性好且缺失值不太高
#         if balance_ratio_value <= balance_threshold and na_ratio_value <= 0.9:
#             additional_adrs.add(adr)
#
#     print(f"第二轮筛选后额外保留 {len(additional_adrs)} 个高平衡性ADR。")
#
#     # 合并所有保留的ADR
#     adrs_to_keep = list(keep_black_box_adrs_set | keep_non_black_box_adrs_set | additional_adrs)
#
#     print(f"总共保留 {len(adrs_to_keep)} 个ADR终点（包括黑框ADR、非黑框ADR和高平衡性ADR）。")
#
#     # 10. 过滤透视表，保留指定的ADR终点
#     print("过滤数据，保留指定的ADR终点...")
#     reduced_pivot_df = pivot_df[adrs_to_keep]
#     print(f"精简后的数据形状为：{reduced_pivot_df.shape}")
#
#     # 11. 计算每个ADR的0,1,NaN比例
#     print("计算每个ADR的0,1,NaN比例...")
#     stats = []
#     for adr in adrs_to_keep:
#         total = reduced_pivot_df.shape[0]
#         count_1 = reduced_pivot_df[adr].sum()
#         count_0 = (reduced_pivot_df[adr] == 0).sum()
#         count_nan = reduced_pivot_df[adr].isna().sum()
#         proportion_1 = count_1 / total
#         proportion_0 = count_0 / total
#         proportion_nan = count_nan / total
#         adr_type = 'Black_Box' if adr in black_box_adrs else 'Normal'
#         balance_ratio = balance_ratios[adr]
#
#         stats.append({
#             'Adverse Event': adr,
#             'Type': adr_type,
#             'Count_1': count_1,
#             'Proportion_1': round(proportion_1, 4),
#             'Count_0': count_0,
#             'Proportion_0': round(proportion_0, 4),
#             'Count_NaN': count_nan,
#             'Proportion_NaN': round(proportion_nan, 4),
#             'Balance Ratio': round(balance_ratio, 4)
#         })
#
#     stats_df = pd.DataFrame(stats)
#     print("每个ADR的0,1,NaN比例计算完成。")
#
#     # 12. 保存统计结果到CSV
#     stats_df.to_csv(stats_csv, index=False)
#     print(f"保存统计结果到 {stats_csv} 完成。")
#
#     # 13. 重置索引，将Drug作为列保存最终精简后的数据
#     reduced_pivot_df.reset_index(inplace=True)
#     reduced_pivot_df.to_csv(output_csv, index=False)
#     print(f"保存精简后的数据到 {output_csv} 完成。\n")
#
#
# # 示例执行
# if __name__ == "__main__":
#     input_csv = '73adr/balenced_F_labeled_results_child14.csv'
#     output_csv = 'balenced_num01ratio_labeled_results_child14.csv'
#     stats_csv = '73adr/balenced_num01ratioADR_distribution_statistics_child.csv'
#     filter_adr_csv = 'child_14/filter_adr.csv'
#
#     reduce_adrs(input_csv, output_csv, stats_csv, filter_adr_csv, threshold_non_bw=0.85, threshold_bw=0.9, balance_threshold=0.3)

#给最终的数据集匹配上smiles列准备训练！
# import pandas as pd
# # Step 1: Load both datasets
# # Load the dataset containing the drugs and ADR endpoints
# dataset = pd.read_csv("balenced_num01ratio_labeled_results_child14.csv")
#
# # Load the drug-Smiles mapping table
# mapping = pd.read_excel("new_drug_listiso_with_info.xlsx", engine='openpyxl')
#
# # Step 2: Set up mapping for the "Smiles" field
# # Ensure column names are standardized
# mapping = mapping.rename(columns={"Drug": "Drug", "Smiles": "Smiles"})
#
# # Merge the "Smiles" information into the dataset
# merged_dataset = dataset.merge(mapping, on="Drug", how="left")
#
# # Step 3: Identify unmatched drugs
# unmatched_drugs = merged_dataset[merged_dataset["Smiles"].isnull()]["Drug"].unique()
#
# # Step 4: Save the new dataset
# # Save the updated dataset to a new CSV
# merged_dataset.to_csv("433_labeled_results_with_smiles.csv", index=False)
#
# # Step 5: Save unmatched drugs to a separate file (optional)
# if len(unmatched_drugs) > 0:
#     print(f"Number of unmatched drugs: {len(unmatched_drugs)}")
#     unmatched_df = pd.DataFrame(unmatched_drugs, columns=["Unmatched Drugs"])
#     unmatched_df.to_csv("unmatched_drugs.csv", index=False)
#     print("Unmatched drugs have been saved to 'unmatched_drugs.csv'.")
# else:
#     print("All drugs matched successfully!")


# 将处理好的带有smiles的数据集通过我自定义的adr分类表进行数据集划分，以方便多任务训练
# import os
# import pandas as pd
#
# # Step 1：加载原始数据集和分类文件
# raw_data = pd.read_csv("child_14/F_reduced_labeled_results_with_smiles.csv")  # 原始数据集
#
# # 加载ADR分类文件（Excel版本），处理列名和数据
# adr_categories = pd.read_excel("child_14/adr_category test.xlsx")  # 从Excel读取文件
# adr_categories.columns = adr_categories.columns.str.strip().str.lower()  # 列名全转为小写，去除多余空格
# adr_categories['adr_term'] = adr_categories['adr_term'].str.strip()  # 清洗ADR_Term列首尾空格
#
# # Step 2：生成一个包含所有 `ADR_Term` 的集合（用于快速匹配）
# defined_adrs = set(adr_categories['adr_term'])  # `adr_category.xlsx` 中的所有ADR终点
#
# # Step 3：在原始数据集中查找未匹配的 ADR 终点
# # 从原始数据中获取所有ADR列名（跳过第一列 Smiles）
# raw_adr_columns = set(raw_data.columns[1:])  # 除 Smiles 列外的列名
# unmatched_adrs = raw_adr_columns - defined_adrs  # 不在 adr_category 中的列名
#
# # Step 4：输出结果
# # 创建保存结果的文件夹
# output_folder = "adr_system_datasets"
# os.makedirs(output_folder, exist_ok=True)
#
# # 保存未匹配到的ADR终点
# if unmatched_adrs:
#     unmatched_file = os.path.join(output_folder, "unmatched_adr_in_raw_data.csv")
#     pd.DataFrame({"Unmatched_ADR_Term": list(unmatched_adrs)}).to_csv(unmatched_file, index=False)
#     print(f"未匹配的ADR终点（原始数据中无法在分类文件中找到的ADR终点）已保存至：{unmatched_file}")
# else:
#     print("所有原始数据的ADR终点均已在分类文件中匹配到。")
#
# # Step 5：执行分类并保存匹配的结果
# summary = []  # 记录分类匹配统计
#
# for category in adr_categories['category'].unique():
#     # 获取该系统下的所有ADR终点
#     category_adrs = adr_categories[adr_categories['category'] == category]['adr_term']
#
#     # 筛选匹配到的ADR终点
#     matched_adrs = [col for col in raw_data.columns if col in category_adrs.values]
#
#     # 筛选匹配到原始数据的列（包括Smiles这一列）
#     matched_data = raw_data[['Smiles'] + matched_adrs] if matched_adrs else None
#
#     # 如果有匹配的ADR终点，保存这个系统分类的数据集到文件
#     if matched_data is not None and len(matched_data.columns) > 1:  # 确保有Smiles和至少一个ADR列
#         output_filename = os.path.join(output_folder, f"{category.replace(' ', '_')}.csv")
#         matched_data.to_csv(output_filename, index=False)
#         print(f"分类 '{category}' 已生成文件：{output_filename}")
#         summary.append({"Category": category, "Matched_ADR_Count": len(matched_adrs)})
#     else:
#         print(f"分类 '{category}' 没有匹配成功，跳过。")
#         summary.append({"Category": category, "Matched_ADR_Count": 0})
#
# # Step 6：保存分类匹配统计汇总文件
# summary_file = os.path.join(output_folder, "adr_category_summary.csv")
# summary_df = pd.DataFrame(summary)
# summary_df.to_csv(summary_file, index=False)
# print(f"分类统计汇总已保存至：{summary_file}")

#去水
# import pandas as pd
# from rdkit import Chem
# from rdkit.Chem import RemoveHs
#
# # 文件路径
# input_file = r"D:\FAERS\analysis\pythonProject\433adr12_16\respiratory depression.csv"
# output_file = r"D:\FAERS\analysis\pythonProject\433adr12_16\respiratory_depression_processed.csv"
#
# # 读取文件
# df = pd.read_csv(input_file)
#
# # 确保列名一致
# df.columns = ["Smiles", "Endpoint"]  # 假设第一列是Smiles，第二列是终点值
#
# # 定义去氢函数
# def remove_h(smiles):
#     if pd.isna(smiles):  # 如果是空值，返回错误标记
#         return smiles, "Error"
#     try:
#         # 使用 RDKit 解析 SMILES
#         mol = Chem.MolFromSmiles(smiles)
#         if mol is None:
#             return smiles, "Error"
#
#         # 去除氢原子
#         mol_no_h = RemoveHs(mol)
#         processed_smiles = Chem.MolToSmiles(mol_no_h)
#
#         # 比较原始和去氢后的 SMILES
#         if processed_smiles != smiles:
#             return processed_smiles, "Dehydrated"
#         else:
#             return smiles, "Unchanged"
#     except Exception as e:
#         return smiles, "Error"
#
# # 应用去氢函数
# df[["Processed Smiles", "Status"]] = df["Smiles"].apply(lambda x: pd.Series(remove_h(x)))
#
# # 统计结果
# total_records = len(df)
# dehydrated_count = len(df[df["Status"] == "Dehydrated"])
# unchanged_count = len(df[df["Status"] == "Unchanged"])
# error_count = len(df[df["Status"] == "Error"])
#
# # 打印统计信息
# print(f"总记录数: {total_records}")
# print(f"进行了去氢处理的记录数: {dehydrated_count}")
# print(f"未进行去氢处理的记录数: {unchanged_count}")
# print(f"存在错误的记录数: {error_count}")
#
# # 保存处理后的文件
# df.to_csv(output_file, index=False, encoding="utf-8")
# print(f"处理后的文件已保存到 {output_file}")
