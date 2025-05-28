import os
import logging  
import subprocess  
import pandas as pd  
import numpy as np  
from signaturizer import Signaturizer  
from rdkit import Chem  
import json  

# 配置日志记录  
logging.basicConfig(level=logging.INFO,   
                    format='%(asctime)s - %(levelname)s: %(message)s')  

# Base directory configuration  
base_dir = "/public/home/tianyao/biosignature"  

def ensure_dir(directory):  
    """Ensure directory exists"""  
    if not os.path.exists(directory):  
        os.makedirs(directory)  

def extract_remaining_files():  
    """Extract any remaining compressed model files"""  
    for letter in 'ABCDE':  
        for number in '12345':  
            model_name = f"{letter}{number}"  
            model_path = os.path.join(base_dir, model_name)  
            tar_file = f"{model_path}.tar.gz"  
            
            # Check if tar file exists and directory doesn't  
            if os.path.exists(tar_file) and not os.path.exists(model_path):  
                logging.info(f"Extracting {tar_file} to {model_path}")  
                ensure_dir(model_path)  
                subprocess.run(['tar', '-xzf', tar_file, '-C', model_path], check=True)  

def get_model_paths():  
    """Get absolute paths for all available models"""  
    model_paths = []  
    for letter in 'ABCDE':  
        for number in '12345':  
            model_name = f"{letter}{number}"  
            model_path = os.path.join(base_dir, model_name)  
            if os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, 'saved_model.pb')):  
                model_paths.append(model_path)  
    return sorted(model_paths)  

def initialize_signaturizer(verbose=True):  
    """Initialize the Signaturizer with all available models"""  
    extract_remaining_files()  
    model_paths = get_model_paths()  

    signaturizer = Signaturizer(  
        model_name=model_paths,  
        local=True,  
        verbose=verbose  
    )  
    return signaturizer  

def is_valid_smiles(smiles):  
    """验证 SMILES 是否有效"""  
    try:  
        mol = Chem.MolFromSmiles(smiles)  
        return mol is not None  
    except Exception:  
        return False  

def signature_to_string(signature):  
    """  
    将 signature 转换为紧凑的 JSON 字符串  
    """  
    return json.dumps(signature.tolist())  

def string_to_signature(signature_str):  
    """  
    从 JSON 字符串恢复 signature  
    """  
    return np.array(json.loads(signature_str))  

def main():  
    # Initialize the Signaturizer  
    sign_G = initialize_signaturizer()  

    # Path to the input CSV file  
    input_csv_path = 'all_drugs_with_smiles.csv'   #respiratory_depression.csv
    # Path to the output CSV file  
    output_csv_path = 'features_all.csv'  #features.csv

    # Read SMILES from the CSV file  
    smiles_df = pd.read_csv(input_csv_path)  

    # Prepare a list to store the features  
    features = []  
    valid_smiles = []  
    invalid_smiles = []  

    # Extract SMILES and generate bio_feat  
    for idx, smiles in enumerate(smiles_df['Smiles'], 1):  
        try:  
            # 先验证 SMILES 的有效性  
            if not is_valid_smiles(smiles):  
                logging.warning(f"Invalid SMILES at index {idx}: {smiles}")  
                invalid_smiles.append(smiles)  
                features.append(None)  
                continue  

            # 生成 signature  
            bio_feat = sign_G.predict(smiles)  
            
            # 将 signature 转换为 JSON 字符串  
            signature_str = signature_to_string(bio_feat.signature)  
            
            features.append(signature_str)  
            valid_smiles.append(smiles)  
            
            # 每 100 个 SMILES 打印一次进度  
            if idx % 100 == 0:  
                logging.info(f"Processed {idx} SMILES")  

        except Exception as e:  
            logging.error(f"Error processing SMILES '{smiles}': {e}")  
            features.append(None)  
            invalid_smiles.append(smiles)  

    # 创建 DataFrame 存储 SMILES 和特征  
    features_df = pd.DataFrame({  
        'Smiles': valid_smiles,  
        'BioFeat': features  
    })  

    # 保存特征到 CSV 文件  
    features_df.to_csv(output_csv_path, index=False)  

    # 记录处理统计信息  
    logging.info(f"Total SMILES processed: {len(smiles_df)}")  
    logging.info(f"Valid SMILES: {len(valid_smiles)}")  
    logging.info(f"Invalid SMILES: {len(invalid_smiles)}")  
    logging.info(f"Successfully extracted features and saved to {output_csv_path}")  

    # 如果有无效 SMILES，可以将其记录到单独的文件中  
    if invalid_smiles:  
        with open('invalid_smiles.txt', 'w') as f:  
            for smiles in invalid_smiles:  
                f.write(f"{smiles}\n")  
        logging.warning(f"Invalid SMILES have been saved to 'invalid_smiles.txt'")  

if __name__ == "__main__":  
    main()  
    # 恢复特征的示例代码  
    def recover_features():  
        """恢复特征的示例方法"""  
        df = pd.read_csv('all_drugs_with_smiles.csv')  
        # 将 JSON 字符串转换回 NumPy 数组  
        df['BioFeat'] = df['BioFeat'].apply(string_to_signature)  
        return df
    