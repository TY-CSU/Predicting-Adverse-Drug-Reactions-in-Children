# import pandas as pd
# import requests
# from tqdm import tqdm
#
# # 读取原始数据
# print("Reading input data...")
# df = pd.read_excel('../fare-ror/test_smile.xlsx')
# print("Input data loaded successfully.")
#
# # 获取去重后的药物名称列表
# drug_names = df['Drug'].unique()
# total_drugs = len(drug_names)
# print(f"Found {total_drugs} unique drugs in the input data.")
#
# # 获取SMILE字符串
# smiles_list = []
# print(f"Fetching SMILE strings for {total_drugs} drugs...")
# for drug_name in tqdm(drug_names, desc="Progress", unit="drug"):
#     # 使用ChEMBL的标准化名称
#     url = f'https://www.ebi.ac.uk/chembl/api/data/molecule?pref_name={drug_name}&format=json'
#     response = requests.get(url)
#     print(response.json())
#     try:
#         response = requests.get(url)
#         if response.status_code == 200:
#             data = response.json()
#             if data['molecules']:
#                 smiles = data['molecules'][0].get('molecule_structures', {}).get('canonical_smiles', None)
#                 if smiles:
#                     print(f"Successfully fetched SMILE string for {drug_name}: {smiles}")
#                 else:
#                     print(f"No SMILE string found for {drug_name}")
#                 smiles_list.append(smiles)
#             else:
#                 print(f"No data found for {drug_name}")
#                 smiles_list.append(None)
#         else:
#             print(f"Error fetching SMILE string for {drug_name}. Status code: {response.status_code}")
#             smiles_list.append(None)
#     except Exception as e:
#         print(f"Exception occurred while fetching SMILE string for {drug_name}: {e}")
#         smiles_list.append(None)
# print("SMILE strings fetched.")
#
# # 创建新的DataFrame
# print("Creating new DataFrame with SMILE strings...")
# new_df = pd.DataFrame({'Drug': drug_names, 'Smiles': smiles_list})
#
# # 合并原始数据和新的SMILE数据
# print("Merging input data with SMILE strings...")
# merged_df = pd.merge(df, new_df, on='Drug', how='left')
#
# # 保存结果到新的Excel文件
# print("Saving results to 'drug_list_with_smiles.xlsx'...")
# merged_df.to_excel('drug_list_with_smiles.xlsx', index=False)
# print("Done!")


# import pandas as pd
# import requests
# from tqdm import tqdm
#
# # 读取原始数据
# print("Reading input data...")
# df = pd.read_excel('../fare-ror/Child_ROR_large3_total.xlsx')
# print("Input data loaded successfully.")
#
# # 获取去重后的药物名称列表
# drug_names = df['Drug'].unique()
# total_drugs = len(drug_names)
# print(f"Found {total_drugs} unique drugs in the input data.")
#
# # 获取药物信息
# drug_info = []
# print(f"Fetching drug information for {total_drugs} drugs...")
# for drug_name in tqdm(drug_names, desc="Progress", unit="drug"):
#     url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{drug_name}/property/MolecularFormula,MolecularWeight,IUPACName,CanonicalSMILES,InChIKey,InChI/JSON'
#     try:
#         response = requests.get(url)
#         if response.status_code == 200:
#             data = response.json()['PropertyTable']['Properties'][0]
#             smiles = data['CanonicalSMILES']
#             mol_formula = data['MolecularFormula']
#             mol_weight = data['MolecularWeight']
#             iupac_name = data['IUPACName']
#             inchikey = data['InChIKey']
#             inchi = data['InChI']
#             drug_info.append({
#                 'Drug': drug_name,
#                 'Smiles': smiles,
#                 'MolecularFormula': mol_formula,
#                 'MolecularWeight': mol_weight,
#                 'IUPACName': iupac_name,
#                 'InChIKey': inchikey,
#                 'InChI': inchi
#             })
#         else:
#             print(f"Error fetching information for {drug_name}. Status code: {response.status_code}")
#             drug_info.append({
#                 'Drug': drug_name,
#                 'Smiles': None,
#                 'MolecularFormula': None,
#                 'MolecularWeight': None,
#                 'IUPACName': None,
#                 'InChIKey': None,
#                 'InChI': None
#             })
#     except:
#         print(f"Exception occurred while fetching information for {drug_name}")
#         drug_info.append({
#             'Drug': drug_name,
#             'Smiles': None,
#             'MolecularFormula': None,
#             'MolecularWeight': None,
#             'IUPACName': None,
#             'InChIKey': None,
#             'InChI': None
#         })
# print("Drug information fetched.")
#
# # 创建新的DataFrame
# print("Creating new DataFrame with drug information...")
# new_df = pd.DataFrame(drug_info)
#
# # 合并原始数据和新的药物信息
# print("Merging input data with drug information...")
# merged_df = pd.merge(df, new_df, on='Drug', how='left')
#
# # 保存结果到新的Excel文件
# print("Saving results to 'drug_list_with_info.xlsx'...")
# merged_df.to_excel('drug_list_with_info.xlsx', index=False)
# print("Done!")


# import pandas as pd
# import requests
# from tqdm import tqdm
#
# # 读取原始数据
# print("Reading input data...")
# df = pd.read_excel('./unique_drugs.xlsx')
# print("Input data loaded successfully.")
#
# # 获取去重后的药物名称列表
# drug_names = df['Drug'].unique()
# total_drugs = len(drug_names)
# print(f"Found {total_drugs} unique drugs in the input data.")
#
# # 获取药物信息
# drug_info = []
# print(f"Fetching drug information for {total_drugs} drugs...")
# for drug_name in tqdm(drug_names, desc="Progress", unit="drug"):
#     url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{drug_name}/property/MolecularFormula,MolecularWeight,IUPACName,CanonicalSMILES,InChIKey,InChI/JSON'
#     try:
#         response = requests.get(url)
#         if response.status_code == 200:
#             data = response.json()['PropertyTable']['Properties'][0]
#             smiles = data['CanonicalSMILES']
#             mol_formula = data['MolecularFormula']
#             mol_weight = data['MolecularWeight']
#             iupac_name = data['IUPACName']
#             inchikey = data['InChIKey']
#             inchi = data['InChI']
#             drug_info.append({
#                 'Drug': drug_name,
#                 'Smiles': smiles,
#                 'MolecularFormula': mol_formula,
#                 'MolecularWeight': mol_weight,
#                 'IUPACName': iupac_name,
#                 'InChIKey': inchikey,
#                 'InChI': inchi
#             })
#         else:
#             print(f"Error fetching information for {drug_name}. Status code: {response.status_code}")
#             drug_info.append({
#                 'Drug': drug_name,
#                 'Smiles': None,
#                 'MolecularFormula': None,
#                 'MolecularWeight': None,
#                 'IUPACName': None,
#                 'InChIKey': None,
#                 'InChI': None
#             })
#     except Exception as e:
#         print(f"Exception occurred while fetching information for {drug_name}: {str(e)}")
#         drug_info.append({
#             'Drug': drug_name,
#             'Smiles': None,
#             'MolecularFormula': None,
#             'MolecularWeight': None,
#             'IUPACName': None,
#             'InChIKey': None,
#             'InChI': None
#         })
# print("Drug information fetched.")
#
# # 创建新的DataFrame
# print("Creating new DataFrame with drug information...")
# new_df = pd.DataFrame(drug_info)
#
# # 合并原始数据和新的药物信息
# print("Merging input data with drug information...")
# merged_df = pd.merge(df, new_df, on='Drug', how='left')
#
# # 保存结果到新的Excel文件
# print("Saving results to 'drug_list_with_info.xlsx'...")
# merged_df.to_excel('drug_list_with_info.xlsx', index=False)
# print("Done!")

import pandas as pd
import requests
from tqdm import tqdm

# 读取原始数据
print("Reading input data...")
df = pd.read_csv('unique_sider_drugname.csv')   #unique_drugs.xlsx
print("Input data loaded successfully.")

# 获取去重后的药物名称列表
drug_names = df['Drug'].unique()
total_drugs = len(drug_names)
print(f"Found {total_drugs} unique drugs in the input data.")

# 获取药物信息
drug_info = []
print(f"Fetching drug information for {total_drugs} drugs...")
for drug_name in tqdm(drug_names, desc="Progress", unit="drug"):
    url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{drug_name}/property/MolecularFormula,MolecularWeight,IUPACName,CanonicalSMILES,IsomericSMILES,InChIKey,InChI/JSON'
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()['PropertyTable']['Properties'][0]
            smiles = data.get('CanonicalSMILES', None)
            isomeric_smiles = data.get('IsomericSMILES', None)
            # mol_formula = data.get('MolecularFormula', None)
            # mol_weight = data.get('MolecularWeight', None)
            # iupac_name = data.get('IUPACName', None)
            # inchikey = data.get('InChIKey', None)
            # inchi = data.get('InChI', None)

            # 如果存在异构 SMILES，则使用异构 SMILES
            if isomeric_smiles:
                smiles = isomeric_smiles

            drug_info.append({
                'Drug': drug_name,
                'Smiles': smiles,
                # 'MolecularFormula': mol_formula,
                # 'MolecularWeight': mol_weight,
                # 'IUPACName': iupac_name,
                # 'InChIKey': inchikey,
                # 'InChI': inchi
            })
        else:
            print(f"Error fetching information for {drug_name}. Status code: {response.status_code}")
            drug_info.append({
                'Drug': drug_name,
                'Smiles': None,
                'MolecularFormula': None,
                'MolecularWeight': None,
                'IUPACName': None,
                'InChIKey': None,
                'InChI': None
            })
    except Exception as e:
        print(f"Exception occurred while fetching information for {drug_name}: {str(e)}")
        drug_info.append({
            'Drug': drug_name,
            'Smiles': None,
            'MolecularFormula': None,
            'MolecularWeight': None,
            'IUPACName': None,
            'InChIKey': None,
            'InChI': None
        })
print("Drug information fetched.")

# 创建新的DataFrame
print("Creating new DataFrame with drug information...")
new_df = pd.DataFrame(drug_info)

# 合并原始数据和新的药物信息
print("Merging input data with drug information...")
merged_df = pd.merge(df, new_df, on='Drug', how='left')

# 保存结果到新的Excel文件
print("Saving results to 'drug_list_with_info.xlsx'...")

merged_df.to_excel('siderdrug_listiso_with_info.xlsx', index=False)   #drug_list_with_info.xlsx
print("Done!")

# import pandas as pd
#
# # 读取原始数据
# print("Reading input data...")
# df = pd.read_excel('./unique_drugs.xlsx')
# print("Input data loaded successfully.")
#
# # 获取'Drug'列并去重
# unique_drugs = df['Drug'].drop_duplicates().reset_index(drop=True)
#
# # 创建新的DataFrame,只包含去重后的'Drug'列
# new_df = pd.DataFrame({'Drug': unique_drugs})
#
# # 保存结果到新的Excel文件
# output_file = 'unique_drugs.xlsx'
# print(f"Saving unique drugs to '{output_file}'...")
# new_df.to_excel(output_file, index=False)
# print("Done!")
#
# # 打印统计信息
# total_drugs = len(df['Drug'])
# unique_drug_count = len(unique_drugs)
# print(f"Total number of drugs in original file: {total_drugs}")
# print(f"Number of unique drugs: {unique_drug_count}")
# print(f"Removed {total_drugs - unique_drug_count} duplicate entries")