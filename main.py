#!/usr/bin/env python3
from xyz_maker import xyz_maker
from descriptors_maker_QM import QM_DFT,grep_qm
from descriptors_maker import descriptors_maker
import argparse
import sys
import re
import joblib
import pandas as pd

def parse_input_file(input_file):
    config = {}
    
    try:
        with open(input_file, 'r') as f:
            for line in f:
                # 移除行尾换行符
                line = line.strip()
                
                # 跳过空行和纯注释行
                if not line or line.startswith('#'):
                    continue
                
                # 处理行内注释（删除#后面的内容）
                if '#' in line:
                    line = line.split('#', 1)[0].strip()
                
                # 解析键值对
                if '=' in line:
                    # 分割键和值
                    key, value = line.split('=', 1)
                    
                    # 清理键和值
                    key = key.strip()
                    value = value.strip()
                    
                    # 添加到配置字典
                    config[key] = value
            
    
    except Exception as e:
        return {}
    
    return config

# 测试代码
if __name__ == "__main__":

# 解析输入文件
    config = parse_input_file(r"example\example.inp")

    # INPUT PARAMETERS
    smiles = config.get("SMILES", "")
    xc_functional = config.get("XC_FUNCTIONAL", "WB97XD")  # 默认
    machine_learning = config.get("MACHINE_LEARNING", "XGBOOST")
    basis_set = config.get("BASIS_SET")
    output_task = config.get("OUTPUT_TASK", "TDDFT")
    slurm_path = config.get(r"SLURM_PATH", r"descriptors_maker_QM\test_1.slurm")
    
    
    #Classcial Descriptors Calculation and molecule .xyz generation
    xyz_gen = xyz_maker.XYZGenerator(output_dir='xyz_molecules')
    desc_calc = descriptors_maker.MolecularDescriptorCalculator()
    classic_descriptors = desc_calc.calculate_descriptors(smiles)
    print(classic_descriptors)

    xyz_gen.generate_xyz(smiles)
    
   #QM Descriptors Calculation and Gaussian input file generation
    qm_cal = QM_DFT.GaussianInputGenerator()
    qm_descriptors = qm_cal.generate_gaussian_inputs(structure_folder='xyz_molecules')
    qm_slurm = qm_cal.generate_slurm_scripts(slurm_path)
    qm_cal.submit_slurm_jobs()
    grep_qm = grep_qm.MultipoleParser()
    qm_descriptors = grep_qm.process_directory()
    
    #descriptors merging
    descriptors = pd.concat([qm_descriptors, classic_descriptors], axis=1)
    
    #Machine_learning model
    if xc_functional == "WB97XD":
        scaler = joblib.load('SGM_wB97XD\Scaler\scaler_wb97xd.pkl')
        if machine_learning == "XGBOOST":
           model = joblib.load('SGM_wB97XD\Stacking_model\final_xgb_wb97xd.pkl')
        elif machine_learning == "RANDOMFOREST":
            model = joblib.load('SGM_wB97XD\Stacking_model\final_rf_wb97xd.pkl')
        elif machine_learning == "LIGHTGBM":
            model = joblib.load('SGM_wB97XD\Stacking_model\final_lgb_wb97xd.pkl')
        elif machine_learning == "CATBOOST":
            model = joblib.load('SGM_wB97XD\Stacking_model\final_catboost_wb97xd.pkl')
        elif machine_learning == "GBDT":
            model = joblib.load('SGM_wB97XD\Stacking_model\final_gbr_wb97xd.pkl')
        elif machine_learning == "RIDGE":
            model = joblib.load('SGM_wB97XD\Stacking_model\final_ridge_wb97xd.pkl')
        elif machine_learning == "LASSO":
            model = joblib.load('SGM_wB97XD\Stacking_model\final_lasso_wb97xd.pkl')
        elif machine_learning == "ELASTICNET":
            model = joblib.load('SGM_wB97XD\Stacking_model\final_elasticnet_wb97xd.pkl')
        elif machine_learning == "ADABOOST":
            model = joblib.load('SGM_wB97XD\Stacking_model\final_adaboost_wb97xd.pkl')
        elif machine_learning == "SGM":
            model = joblib.load('SGM_wB97XD\Stacking_model\final_sgm_wb97xd.pkl')
        else:
            print("YOU MAKE A MISTAKE IN MACHINE_LEARNING INPUT PARAMETERS")
            sys.exit(1)
    elif xc_functional == "LC-wPBE":
        scaler = joblib.load('SGM_LC-wPBE\Scaler\scaler_LC.pkl')
        if machine_learning == "XGBOOST":
           model = joblib.load('SGM_LC-wPBE\Stacking_model\final_xgb_LC.pkl')
        elif machine_learning == "RANDOMFOREST":
            model = joblib.load('SGM_LC-wPBE\Stacking_model\final_rf_LC.pkl')
        elif machine_learning == "LIGHTGBM":
            model = joblib.load('SGM_LC-wPBE\Stacking_model\final_lgb_LC.pkl')
        elif machine_learning == "CATBOOST":
            model = joblib.load('SGM_LC-wPBE\Stacking_model\final_catboost_LC.pkl')
        elif machine_learning == "GBDT":
            model = joblib.load('SGM_LC-wPBE\Stacking_model\final_gbr_LC.pkl')
        elif machine_learning == "RIDGE":
            model = joblib.load('SGM_LC-wPBE\Stacking_model\final_ridge_LC.pkl')
        elif machine_learning == "LASSO":
            model = joblib.load('SGM_LC-wPBE\Stacking_model\final_lasso_LC.pkl')
        elif machine_learning == "ELASTICNET":
            model = joblib.load('SGM_LC-wPBE\Stacking_model\final_elasticnet_LC.pkl')
        elif machine_learning == "ADABOOST":
            model = joblib.load('SGM_LC-wPBE\Stacking_model\final_adaboost_LC.pkl')
        elif machine_learning == "SGM":
            model = joblib.load('SGM_LC-wPBE\Stacking_model\final_sgm_LC.pkl')
        else:
            print("YOU MAKE A MISTAKE IN MACHINE_LEARNING INPUT PARAMETERS")
            sys.exit(1)
    else:
        print("YOU MAKE A MISTAKE IN XC_FUNCTIONAL INPUT PARAMETERS")
    # Scale descriptors
    scaled_descriptors = scaler.transform(descriptors)
    # Predict
    xc_functional_parameters = model.predict(scaled_descriptors)
    # Output
    print(xc_functional_parameters)
    
    # Output
    with open('output.out', 'w') as f:
        f.write(f'XC_FUNCTIONAL: {xc_functional}\n')
        f.write(f'MACHINE_LEARNING: {machine_learning}\n')
        f.write(f'XC_FUNCTIONAL_PARAMETERS: {xc_functional_parameters}\n')
        f.write(f'CLASSIC_DESCRIPTORS: {classic_descriptors}\n')
        f.write(f'QM_DESCRIPTORS: {qm_descriptors}\n')
        