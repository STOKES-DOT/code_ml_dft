#!/usr/bin/env python3
from xyz_maker import xyz_maker
from descriptors_maker_QM import QM_DFT,grep_qm
from descriptors_maker import descriptors_maker
import argparse
import sys
import re
import joblib
import pandas as pd
import time
import os

def parse_input_file(input_file):
    config = {}
    try:
        with open(input_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '#' in line:
                    line = line.split('#', 1)[0].strip()
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    config[key] = value
    except Exception as e:
        return {}
    return config

def check_gaussian_completion(log_file, timeout=36000, check_interval=10):
    start_time = time.time()
    
    while not os.path.exists(log_file):
        if time.time() - start_time > timeout:
            print(f"错误: 超时等待 {log_file} 生成")
            return False
        print(f"等待 {log_file} 生成...")
        time.sleep(check_interval)
    
    print(f"检测到文件: {log_file}")
    print("检查计算是否正常终止...")
    
    while True:
        if time.time() - start_time > timeout:
            print(f"错误: 超时等待 {log_file} 完成")
            return False
        
        # 检查文件是否包含"Normal termination"
        with open(log_file, 'r') as f:
            content = f.read()
            if "Normal termination" in content:
                print("Gaussian计算正常终止")
                return True
            elif "Error termination" in content:
                print(f"错误: Gaussian计算失败 - {log_file}")
                return False
        
        print(f"计算尚未完成，等待 {check_interval} 秒后重试...")
        time.sleep(check_interval)

def validate_input(config):
    """Validate input parameters and return errors/warnings"""
    errors = []
    warnings = []
    valid_params = ['SMILES', 'XC_FUNCTIONAL', 'MACHINE_LEARNING', 'BASIS_SET', 
                   'OUTPUT_TASK', 'SLURM_PATH']
    
    # Check for required parameters
    required_params = ['SMILES', 'BASIS_SET', 'SLURM_PATH']
    for param in required_params:
        if param not in config or not config[param]:
            errors.append(f"Missing required parameter: {param}")
    
    # Check for unrecognized parameters
    for param in config:
        if param not in valid_params:
            warnings.append(f"Unrecognized parameter: {param}")
    
    # Validate specific values
    if 'XC_FUNCTIONAL' in config and config['XC_FUNCTIONAL'] not in ['WB97XD', 'LC-wPBE']:
        errors.append(f"Invalid XC_FUNCTIONAL: {config['XC_FUNCTIONAL']}. Must be WB97XD or LC-wPBE")
    
    valid_ml_models = ['XGBOOST', 'RANDOMFOREST', 'LIGHTGBM', 'CATBOOST', 'GBDT', 
                       'RIDGE', 'LASSO', 'ELASTICNET', 'ADABOOST', 'SGM']
    if 'MACHINE_LEARNING' in config and config['MACHINE_LEARNING'] not in valid_ml_models:
        errors.append(f"Invalid MACHINE_LEARNING model: {config['MACHINE_LEARNING']}")
    
    return errors, warnings

if __name__ == "__main__":
    input_file = r"example\example.inp"
    config = parse_input_file(input_file)
    
    # Input validation
    input_errors, input_warnings = validate_input(config)
    
    # INPUT PARAMETERS
    smiles = config.get("SMILES", "")
    xc_functional = config.get("XC_FUNCTIONAL", "WB97XD")
    machine_learning = config.get("MACHINE_LEARNING", "XGBOOST")
    basis_set = config.get("BASIS_SET", "")
    output_task = config.get("OUTPUT_TASK", "TDDFT")
    slurm_path = config.get("SLURM_PATH", "")
    
    # Prepare output file
    with open(r'example\example.out', 'w') as f:
        # ASCII Art Header
        f.write("""
  ___  _____  _____  _   _   ___   __
 / _ \|  _  ||_   _|| | | | / _ \ / _|
| | | | | | |  | |  | |_| |/ /_\ \\\ \ 
| | | | | | |  | |  |  _  ||  _  | |_ |
| |_| \ \_/ /  | |  | | | || | | ||  _|
 \___/ \___/   \_/  \_| |_/\_| |_/|_|  
Exchange-Correlation Optimization Report
=======================================\n\n""")
        
        # Input validation report
        f.write("[ INPUT VALIDATION ]\n")
        f.write("-----------------------------------------\n")
        if input_errors or input_warnings:
            if input_errors:
                f.write("CRITICAL ERRORS FOUND:\n")
                for error in input_errors:
                    f.write(f"• {error}\n")
                f.write("\nCalculation cannot proceed. Please fix input errors.\n")
            if input_warnings:
                f.write("\nWARNINGS:\n")
                for warning in input_warnings:
                    f.write(f"• {warning}\n")
                f.write("\nProceeding with calculation, but verify input parameters.\n")
        else:
            f.write("All input parameters validated successfully.\n")
        f.write("\n")
        
        # Exit if critical errors found
        if input_errors:
            f.write("\n[ END OF REPORT ]\n")
            f.write("=======================================")
            print("Input validation errors found. Exiting.")
            sys.exit(1)
        
        # Basic Parameters
        f.write(f"[ INPUT PARAMETERS ]\n")
        f.write(f"• SMILES: {smiles}\n")
        f.write(f"• XC_FUNCTIONAL: {xc_functional}\n")
        f.write(f"• MACHINE_LEARNING_MODEL: {machine_learning}\n")
        f.write(f"• BASIS_SET: {basis_set}\n")
        f.write(f"• OUTPUT_TASK: {output_task}\n")
        f.write(f"• SLURM_PATH: {slurm_path}\n\n")
    
    # Continue with calculation if no critical errors
    # Classical Descriptors Calculation and molecule .xyz generation
    try:
        xyz_gen = xyz_maker.XYZGenerator(output_dir='xyz_molecules')
        desc_calc = descriptors_maker.MolecularDescriptorCalculator()
        classic_descriptors = desc_calc.calculate_descriptors(smiles)
        print(classic_descriptors)
        xyz_gen.generate_xyz(smiles)
    except Exception as e:
        with open(r'example\example.out', 'a') as f:
            f.write(f"\n[ CALCULATION ERROR ]\n")
            f.write(f"Error in classical descriptors calculation: {str(e)}\n")
        print(f"Error in classical descriptors calculation: {str(e)}")
        sys.exit(1)
    
    # QM Descriptors Calculation and Gaussian input file generation
    try:
        qm_cal = QM_DFT.GaussianInputGenerator()
        qm_descriptors = qm_cal.generate_gaussian_inputs(structure_folder='xyz_molecules')
        qm_slurm = qm_cal.generate_slurm_scripts(slurm_path)
        qm_cal.submit_slurm_jobs()
        
        # Determine log file path
        xyz_file = os.path.join(r'xyz_molecules')
        base_name = os.path.basename(xyz_file).replace('.xyz', '')
        log_file = os.path.join('gaussian_output', f'{base_name}_g16.log')
        print(f"等待Gaussian计算结果: {log_file}")
        
        # Check Gaussian completion
        if not check_gaussian_completion(log_file):
            raise Exception(f"Gaussian calculation failed for {log_file}")
        
        grep_qm_parser = grep_qm.MultipoleParser()
        qm_descriptors = grep_qm_parser.process_directory()
    except Exception as e:
        with open(r'example\example.out', 'a') as f:
            f.write(f"\n[ CALCULATION ERROR ]\n")
            f.write(f"Error in QM descriptors calculation: {str(e)}\n")
        print(f"Error in QM descriptors calculation: {str(e)}")
        sys.exit(1)
    
    # Descriptors merging
    try:
        descriptors = pd.concat([qm_descriptors, classic_descriptors], axis=1)
    except Exception as e:
        with open(r'example\example.out', 'a') as f:
            f.write(f"\n[ DATA PROCESSING ERROR ]\n")
            f.write(f"Error merging descriptors: {str(e)}\n")
        print(f"Error merging descriptors: {str(e)}")
        sys.exit(1)
    
    # Machine learning model
    try:
        if xc_functional == "WB97XD":
            scaler = joblib.load('SGM_wB97XD\Scaler\scaler_wb97xd.pkl')
            model_path = 'SGM_wB97XD\Stacking_model\\'
            if machine_learning == "XGBOOST":
                model = joblib.load(model_path + 'final_xgb_wb97xd.pkl')
            elif machine_learning == "RANDOMFOREST":
                model = joblib.load(model_path + 'final_rf_wb97xd.pkl')
            elif machine_learning == "LIGHTGBM":
                model = joblib.load(model_path + 'final_lgb_wb97xd.pkl')
            elif machine_learning == "CATBOOST":
                model = joblib.load(model_path + 'final_catboost_wb97xd.pkl')
            elif machine_learning == "GBDT":
                model = joblib.load(model_path + 'final_gbr_wb97xd.pkl')
            elif machine_learning == "RIDGE":
                model = joblib.load(model_path + 'final_ridge_wb97xd.pkl')
            elif machine_learning == "LASSO":
                model = joblib.load(model_path + 'final_lasso_wb97xd.pkl')
            elif machine_learning == "ELASTICNET":
                model = joblib.load(model_path + 'final_elasticnet_wb97xd.pkl')
            elif machine_learning == "ADABOOST":
                model = joblib.load(model_path + 'final_adaboost_wb97xd.pkl')
            elif machine_learning == "SGM":
                model = joblib.load(model_path + 'final_sgm_wb97xd.pkl')
            else:
                raise ValueError("Invalid MACHINE_LEARNING model specified")
        elif xc_functional == "LC-wPBE":
            scaler = joblib.load('SGM_LC-wPBE\Scaler\scaler_LC.pkl')
            model_path = 'SGM_LC-wPBE\Stacking_model\\'
            if machine_learning == "XGBOOST":
                model = joblib.load(model_path + 'final_xgb_LC.pkl')
            elif machine_learning == "RANDOMFOREST":
                model = joblib.load(model_path + 'final_rf_LC.pkl')
            elif machine_learning == "LIGHTGBM":
                model = joblib.load(model_path + 'final_lgb_LC.pkl')
            elif machine_learning == "CATBOOST":
                model = joblib.load(model_path + 'final_catboost_LC.pkl')
            elif machine_learning == "GBDT":
                model = joblib.load(model_path + 'final_gbr_LC.pkl')
            elif machine_learning == "RIDGE":
                model = joblib.load(model_path + 'final_ridge_LC.pkl')
            elif machine_learning == "LASSO":
                model = joblib.load(model_path + 'final_lasso_LC.pkl')
            elif machine_learning == "ELASTICNET":
                model = joblib.load(model_path + 'final_elasticnet_LC.pkl')
            elif machine_learning == "ADABOOST":
                model = joblib.load(model_path + 'final_adaboost_LC.pkl')
            elif machine_learning == "SGM":
                model = joblib.load(model_path + 'final_sgm_LC.pkl')
            else:
                raise ValueError("Invalid MACHINE_LEARNING model specified")
        else:
            raise ValueError("Invalid XC_FUNCTIONAL specified")
        
        # Scale descriptors and predict
        scaled_descriptors = scaler.transform(descriptors)
        xc_functional_parameters = model.predict(scaled_descriptors)
    except Exception as e:
        with open(r'example\example.out', 'a') as f:
            f.write(f"\n[ MODEL ERROR ]\n")
            f.write(f"Error in machine learning prediction: {str(e)}\n")
        print(f"Error in machine learning prediction: {str(e)}")
        sys.exit(1)
    
    # Final output
    with open(r'example\example.out', 'a') as f:
        # Predicted Parameters
        f.write(f"[ OPTIMIZED XC PARAMETERS ]\n")
        f.write(f"{xc_functional_parameters}\n\n")
        
        # Classical Descriptors Section
        f.write("[ CLASSICAL DESCRIPTORS ]\n")
        f.write("-----------------------------------------\n")
        f.write("A. GEOMETRIC STRUCTURE DESCRIPTORS:\n")
        f.write(f"  • PMI1: {classic_descriptors.get('PMI1', 'N/A')} (1st principal moment of inertia)\n")
        f.write(f"  • PMI2: {classic_descriptors.get('PMI2', 'N/A')} (2nd principal moment of inertia)\n")
        f.write(f"  • PMI3: {classic_descriptors.get('PMI3', 'N/A')} (3rd principal moment of inertia)\n")
        f.write(f"  • Planar_RMSD: {classic_descriptors.get('Planar_RMSD', 'N/A')} Å (Root mean square deviation from best-fit plane)\n")
        f.write(f"  • Inertia_Ratio: {classic_descriptors.get('Inertia_Ratio', 'N/A')} (Ratio of smallest to largest principal moment)\n")
        f.write(f"  • Max_Planar_Deviation: {classic_descriptors.get('Max_Planar_Deviation', 'N/A')} Å (Maximum deviation from molecular plane)\n")
        f.write(f"  • Conjugated_Ratio: {classic_descriptors.get('Conjugated_Ratio', 'N/A')} (Proportion of atoms in conjugated systems)\n\n")
        
        f.write("B. ELECTRONIC PROPERTY DESCRIPTORS:\n")
        f.write(f"  • Gasteiger_Charge_Mean: {classic_descriptors.get('Gasteiger_Charge_Mean', 'N/A')} (Average partial atomic charge)\n")
        f.write(f"  • Gasteiger_Charge_Std: {classic_descriptors.get('Gasteiger_Charge_Std', 'N/A')} (Standard deviation of partial charges)\n")
        f.write(f"  • EState_Mean: {classic_descriptors.get('EState_Mean', 'N/A')} (Average E-State index)\n")
        f.write(f"  • EState_Std: {classic_descriptors.get('EState_Std', 'N/A')} (Dispersion of E-State indices)\n")
        f.write(f"  • MolMR: {classic_descriptors.get('MolMR', 'N/A')} cm³/mol (Molar refractivity)\n\n")
        
        f.write("C. DIMENSIONALITY REDUCTION DESCRIPTORS:\n")
        for i in range(1, 4):  # Assuming 3 UMAP components
            key = f'UMAP{i}'
            if key in classic_descriptors:
                f.write(f"  • {key}: {classic_descriptors[key]} (Component {i} of fingerprint dimensionality reduction)\n")
        f.write("\n")
        
        # QM Descriptors Section
        f.write("[ QUANTUM MECHANICAL DESCRIPTORS ]\n")
        f.write("-----------------------------------------\n")
        f.write("A. MULTIPOLE MOMENTS:\n")
        f.write("  1. Dipole (Debye):\n")
        f.write(f"     • μ_x: {qm_descriptors.get('x', 'N/A')}\n")
        f.write(f"     • μ_y: {qm_descriptors.get('y', 'N/A')}\n")
        f.write(f"     • μ_z: {qm_descriptors.get('z', 'N/A')}\n")
        f.write(f"     • |μ|: {qm_descriptors.get('tot', 'N/A')}\n\n")
        
        f.write("  2. Quadrupole (Debye·Å):\n")
        f.write(f"     • Q_xx: {qm_descriptors.get('xx', 'N/A')}\n")
        f.write(f"     • Q_yy: {qm_descriptors.get('yy', 'N/A')}\n")
        f.write(f"     • Q_zz: {qm_descriptors.get('zz', 'N/A')}\n")
        f.write(f"     • Q_xy: {qm_descriptors.get('xy', 'N/A')}\n")
        f.write(f"     • Q_xz: {qm_descriptors.get('xz', 'N/A')}\n")
        f.write(f"     • Q_yz: {qm_descriptors.get('yz', 'N/A')}\n\n")
        
        f.write("  3. Octapole (Debye·Å²):\n")
        octapole_keys = ['xxx', 'yyy', 'zzz', 'xyy', 'xxy', 'xxz', 'xzz', 'yzz', 'yyz', 'xyz']
        for key in octapole_keys:
            if key in qm_descriptors:
                f.write(f"     • O_{key}: {qm_descriptors[key]}\n")
        f.write("\n")
        
        f.write("  4. Hexadecapole (Debye·Å³):\n")
        hexadecapole_keys = ['xxxx', 'yyyy', 'zzzz', 'xxxy', 'xxxz', 'yyyx', 'yyyz', 'zzzx', 'zzzy', 'xxyy', 'xxzz', 'yyzz']
        for key in hexadecapole_keys:
            if key in qm_descriptors:
                f.write(f"     • H_{key}: {qm_descriptors[key]}\n")
        
        # Footer
        f.write("\n[ END OF REPORT ]\n")
        f.write("\n The complete code is available at https://github.com/STOKES-DOT/code_ml_dft. You can edit the codes for your own needs. Good Luck! Don't forget to sent me 2.5$ for a cup of coffee! Just a joke, but I appreciate it (Doge). Have a great day!\n")
        f.write("=======================================")