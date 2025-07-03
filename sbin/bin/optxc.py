#!/usr/bin/env python3
import os
import sys
import re
import time
import argparse
import joblib
import pandas as pd
from glob import glob
from . import xyz_maker, descriptors_maker
from . import QM_DFT, grep_qm

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

def get_input_directory(input_file):
    """Return the directory containing the input file"""
    return os.path.dirname(os.path.abspath(input_file)) or os.getcwd()

def check_gaussian_completion(log_file, timeout=36000, check_interval=10):
    start_time = time.time()
    
    while not os.path.exists(log_file):
        if time.time() - start_time > timeout:
            print(f"Error: Timeout waiting for {log_file}")
            return False
        print(f"Waiting for {log_file}...")
        time.sleep(check_interval)
    
    print(f"File detected: {log_file}")
    print("Checking calculation status...")
    
    while True:
        if time.time() - start_time > timeout:
            print(f"Error: Timeout waiting for {log_file}")
            return False
        
        with open(log_file, 'r') as f:
            content = f.read()
            if "Normal termination" in content:
                print("Gaussian calculation completed successfully")
                return True
            elif "Error termination" in content:
                print(f"Error: Gaussian calculation failed - {log_file}")
                return False
        
        print(f"Calculation in progress, retrying in {check_interval} seconds...")
        time.sleep(check_interval)

def validate_input(config):
    errors = []
    warnings = []
    valid_params = ['SMILES', 'XC_FUNCTIONAL', 'MACHINE_LEARNING', 'BASIS_SET', 
                   'OUTPUT_TASK', 'SLURM_PATH']
    
    required_params = ['SMILES', 'BASIS_SET', 'SLURM_PATH']
    for param in required_params:
        if param not in config or not config[param]:
            errors.append(f"Missing required parameter: {param}")
    
    for param in config:
        if param not in valid_params:
            warnings.append(f"Unrecognized parameter: {param}")
    
    if 'XC_FUNCTIONAL' in config and config['XC_FUNCTIONAL'] not in ['WB97XD', 'LC-wPBE']:
        errors.append(f"Invalid XC_FUNCTIONAL: {config['XC_FUNCTIONAL']}. Must be WB97XD or LC-wPBE")
    
    valid_ml_models = ['XGBOOST', 'RANDOMFOREST', 'LIGHTGBM', 'CATBOOST', 'GBDT', 
                       'RIDGE', 'LASSO', 'ELASTICNET', 'ADABOOST', 'SGM']
    if 'MACHINE_LEARNING' in config and config['MACHINE_LEARNING'] not in valid_ml_models:
        errors.append(f"Invalid MACHINE_LEARNING model: {config['MACHINE_LEARNING']}")
    
    return errors, warnings

def get_model_path():
    """Get absolute path to model files"""
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, 'models')

def main():
    parser = argparse.ArgumentParser(description='Optimize XC functional parameters.')
    parser.add_argument('input_file', help='Input file (.inp format)')
    args = parser.parse_args()
    
    input_file = args.input_file
    input_dir = get_input_directory(input_file)
    output_file = os.path.splitext(input_file)[0] + '.out'
    config = parse_input_file(input_file)
    # Change working directory to input file location
    os.chdir(input_dir)
    # Input validation
    input_errors, input_warnings = validate_input(config)
    
    # Prepare output file
    with open(output_file, 'w') as f:
        f.write("""
    .::::     .:::::::  .::: .::::::.::      .::    .::   
  .::    .::  .::    .::     .::     .::   .::   .::   .::
.::        .::.::    .::     .::      .:: .::   .::       
.::        .::.:::::::       .::        .::     .::       
.::        .::.::            .::      .:: .::   .::       
  .::     .:: .::            .::     .::   .::   .::   .::
    .::::     .::            .::    .::      .::   .::::    .::
  
    
  \n Exchange-Correlation Optimization Output
=======================================\n\n""")
        
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
        
        if input_errors:
            f.write("\n[ END OF REPORT ]\n")
            f.write("=======================================")
            print("Input validation errors found. Exiting.")
            sys.exit(1)
        
        # Basic parameters
        smiles = config.get("SMILES", "")
        xc_functional = config.get("XC_FUNCTIONAL", "WB97XD")
        machine_learning = config.get("MACHINE_LEARNING", "XGBOOST")
        basis_set = config.get("BASIS_SET", "")
        output_task = config.get("OUTPUT_TASK", "TDDFT")
        slurm_path = config.get("SLURM_PATH", "")
        
        f.write(f"[ INPUT PARAMETERS ]\n")
        f.write(f"• SMILES: {smiles}\n")
        f.write(f"• XC_FUNCTIONAL: {xc_functional}\n")
        f.write(f"• MACHINE_LEARNING_MODEL: {machine_learning}\n")
        f.write(f"• BASIS_SET: {basis_set}\n")
        f.write(f"• OUTPUT_TASK: {output_task}\n")
        f.write(f"• SLURM_PATH: {slurm_path}\n\n")
    
    # Molecular descriptor calculation
    try:
        xyz_gen = xyz_maker.XYZGenerator(output_dir=input_dir)
        desc_calc = descriptors_maker.MolecularDescriptorCalculator()
        classic_descriptors = desc_calc.calculate_descriptors(smiles)
        xyz_gen.generate_xyz(smiles)
    except Exception as e:
        with open(output_file, 'a') as f:
            f.write(f"\n[ CALCULATION ERROR ]\n")
            f.write(f"Error in classical descriptors calculation: {str(e)}\n")
        print(f"Error in classical descriptors calculation: {str(e)}")
        sys.exit(1)
    
    # QM descriptor calculation
    try:
        qm_cal = QM_DFT.GaussianInputGenerator()
        qm_descriptors = qm_cal.generate_gaussian_inputs(structure_folder=input_dir)
        qm_slurm = qm_cal.generate_slurm_scripts(slurm_path)
        qm_cal.submit_slurm_jobs()
        
        xyz_file = os.path.join(input_dir)
        base_name = os.path.basename(xyz_file).replace('.xyz', '')
        log_file = os.path.join(input_dir, f'{base_name}_g16.log')
        
        if not check_gaussian_completion(log_file):
            raise Exception(f"Gaussian calculation failed for {log_file}")
        
        grep_qm_parser = grep_qm.MultipoleParser()
        qm_descriptors = grep_qm_parser.process_directory()
    except Exception as e:
        with open(output_file, 'a') as f:
            f.write(f"\n[ CALCULATION ERROR ]\n")
            f.write(f"Error in QM descriptors calculation: {str(e)}\n")
        print(f"Error in QM descriptors calculation: {str(e)}")
        sys.exit(1)

    # Clean temporary files
    patterns = ["job_*.xyz", "job_*.gjf", "job_*.slurm"]
    remove_files = []
    for pattern in patterns:
        remove_files.extend(glob(pattern))
    for file_path in remove_files:
        try:
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Deletion failed {file_path}: {str(e)}")
    
    # Merge descriptors
    try:
        descriptors = pd.concat([qm_descriptors, classic_descriptors], axis=1)
    except Exception as e:
        with open(output_file, 'a') as f:
            f.write(f"\n[ DATA PROCESSING ERROR ]\n")
            f.write(f"Error merging descriptors: {str(e)}\n")
        print(f"Error merging descriptors: {str(e)}")
        sys.exit(1)
    
    # Machine learning prediction
    try:
        model_base_path = get_model_path()
        
        if xc_functional == "WB97XD":
            scaler_path = os.path.join(model_base_path, 'SGM_wB97XD', 'Scaler', 'scaler_wb97xd.pkl')
            model_path = os.path.join(model_base_path, 'SGM_wB97XD', 'Stacking_model')
            model_files = {
                'XGBOOST': 'final_xgb_wb97xd.pkl',
                'RANDOMFOREST': 'final_rf_wb97xd.pkl',
                'LIGHTGBM': 'final_lgb_wb97xd.pkl',
                'CATBOOST': 'final_catboost_wb97xd.pkl',
                'GBDT': 'final_gbr_wb97xd.pkl',
                'RIDGE': 'final_ridge_wb97xd.pkl',
                'LASSO': 'final_lasso_wb97xd.pkl',
                'ELASTICNET': 'final_elasticnet_wb97xd.pkl',
                'ADABOOST': 'final_adaboost_wb97xd.pkl',
                'SGM': 'final_sgm_wb97xd.pkl'
            }
        elif xc_functional == "LC-wPBE":
            scaler_path = os.path.join(model_base_path, 'SGM_LC-wPBE', 'Scaler', 'scaler_LC.pkl')
            model_path = os.path.join(model_base_path, 'SGM_LC-wPBE', 'Stacking_model')
            model_files = {
                'XGBOOST': 'final_xgb_LC.pkl',
                'RANDOMFOREST': 'final_rf_LC.pkl',
                'LIGHTGBM': 'final_lgb_LC.pkl',
                'CATBOOST': 'final_catboost_LC.pkl',
                'GBDT': 'final_gbr_LC.pkl',
                'RIDGE': 'final_ridge_LC.pkl',
                'LASSO': 'final_lasso_LC.pkl',
                'ELASTICNET': 'final_elasticnet_LC.pkl',
                'ADABOOST': 'final_adaboost_LC.pkl',
                'SGM': 'final_sgm_LC.pkl'
            }
        else:
            raise ValueError("Invalid XC_FUNCTIONAL specified")
        
        scaler = joblib.load(scaler_path)
        model_file = model_files.get(machine_learning)
        if not model_file:
            raise ValueError(f"Invalid MACHINE_LEARNING model: {machine_learning}")
        
        model = joblib.load(os.path.join(model_path, model_file))
        scaled_descriptors = scaler.transform(descriptors)
        xc_functional_parameters = model.predict(scaled_descriptors)
    except Exception as e:
        with open(output_file, 'a') as f:
            f.write(f"\n[ MODEL ERROR ]\n")
            f.write(f"Error in machine learning prediction: {str(e)}\n")
        print(f"Error in machine learning prediction: {str(e)}")
        sys.exit(1)
    
    # Final output
    with open(output_file, 'a') as f:
        f.write(f"[ OPTIMIZED XC PARAMETERS ]\n")
        f.write(f"{xc_functional_parameters}\n\n")
        
        f.write("[ CLASSICAL DESCRIPTORS ]\n")
        f.write("-----------------------------------------\n")
        f.write("A. GEOMETRIC STRUCTURE DESCRIPTORS:\n")
        f.write(f"  • PMI1: {classic_descriptors.get('PMI1', 'N/A')}\n")
        f.write(f"  • PMI2: {classic_descriptors.get('PMI2', 'N/A')}\n")
        f.write(f"  • PMI3: {classic_descriptors.get('PMI3', 'N/A')}\n")
        f.write(f"  • Planar_RMSD: {classic_descriptors.get('Planar_RMSD', 'N/A')} Å\n")
        f.write(f"  • Inertia_Ratio: {classic_descriptors.get('Inertia_Ratio', 'N/A')}\n")
        f.write(f"  • Max_Planar_Deviation: {classic_descriptors.get('Max_Planar_Deviation', 'N/A')} Å\n")
        f.write(f"  • Conjugated_Ratio: {classic_descriptors.get('Conjugated_Ratio', 'N/A')}\n\n")
        
        f.write("B. ELECTRONIC PROPERTY DESCRIPTORS:\n")
        f.write(f"  • Gasteiger_Charge_Mean: {classic_descriptors.get('Gasteiger_Charge_Mean', 'N/A')}\n")
        f.write(f"  • Gasteiger_Charge_Std: {classic_descriptors.get('Gasteiger_Charge_Std', 'N/A')}\n")
        f.write(f"  • EState_Mean: {classic_descriptors.get('EState_Mean', 'N/A')}\n")
        f.write(f"  • EState_Std: {classic_descriptors.get('EState_Std', 'N/A')}\n")
        f.write(f"  • MolMR: {classic_descriptors.get('MolMR', 'N/A')} cm³/mol\n\n")
        
        f.write("C. DIMENSIONALITY REDUCTION DESCRIPTORS:\n")
        for i in range(1, 10):
            key = f'UMAP{i}'
            if key in classic_descriptors:
                f.write(f"  • {key}: {classic_descriptors[key]}\n")
        f.write("\n")
        
        f.write("[ QUANTUM MECHANICAL DESCRIPTORS ]\n")
        f.write("-----------------------------------------\n")
        f.write("A. MULTIPOLE MOMENTS:\n")
        f.write("  1. Dipole (Debye):\n")
        f.write(f"     • x: {qm_descriptors.get('x', 'N/A')}\n")
        f.write(f"     • y: {qm_descriptors.get('y', 'N/A')}\n")
        f.write(f"     • z: {qm_descriptors.get('z', 'N/A')}\n")
        f.write(f"     • tot: {qm_descriptors.get('tot', 'N/A')}\n\n")
        
        f.write("  2. Quadrupole (Debye·Å):\n")
        f.write(f"     • xx: {qm_descriptors.get('xx', 'N/A')}\n")
        f.write(f"     • yy: {qm_descriptors.get('yy', 'N/A')}\n")
        f.write(f"     • zz: {qm_descriptors.get('zz', 'N/A')}\n")
        f.write(f"     • xy: {qm_descriptors.get('xy', 'N/A')}\n")
        f.write(f"     • xz: {qm_descriptors.get('xz', 'N/A')}\n")
        f.write(f"     • yz: {qm_descriptors.get('yz', 'N/A')}\n\n")
        
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
        
        f.write("\n [ END OF REPORT ] \n")
        f.write("\n Complete code available at https://github.com/STOKES-DOT/code_ml_dft\n")
        f.write("\n Feel free to modify the code for your needs \n")
        f.write("\n Have a great day! \n")
        f.write("=======================================")

if __name__ == "__main__":
    main()
