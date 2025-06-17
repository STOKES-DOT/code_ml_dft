import rdkit as rd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
import pandas as pd
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image
import os
import io
import math

def read_excel_string(file_path, sheet_name=0, usecols=None):
    try:
        # 使用pandas的read_excel函数读取数据
        df = pd.read_excel(file_path, sheet_name=sheet_name, usecols=usecols)
        return df
    except Exception as e:
        # 如果发生异常，打印错误信息
        print(f"Error reading Excel file: {e}")
        return None
TADF_smi = pd.read_csv('xyz_maker\SMILES.csv', usecols=['SMILES'])

print(TADF_smi)

# 提取 'SMILES' 列
TADF_smi = TADF_smi['SMILES'].tolist()
result = []

for i, smi in enumerate(TADF_smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        mol = Chem.AddHs(mol)  # 添加显式氢原子
        state = AllChem.EmbedMolecule(mol, useRandomCoords=True)  # 为分子生成3D构象
        num_atoms = mol.GetNumAtoms()
        AllChem.MMFFOptimizeMolecule(mol)  # 优化分子
        mol_filename = fr"xyz_maker\xyz_molecules\job_{i+1}.xyz"
        Chem.MolToXYZFile(mol, mol_filename)
        print(f"Saved XYZ file: {mol_filename}")
        result.append({"SMILES": smi, "Mol_File": mol_filename})

# 将结果保存到Excel文件
df_result = pd.DataFrame(result)
excel_output_path = r"xyz_maker\xyz_molecules\molecule_info.csv"
df_result.to_csv(excel_output_path, index=False)

print(f"处理完成，结果已保存到 {excel_output_path}")
        
    