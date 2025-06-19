# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 22:09:40 2025

@author: 26607
"""

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors3D
from rdkit.Chem import Descriptors
from rdkit.Chem import EState
from sklearn.decomposition import PCA
import umap
import numpy as np
from scipy.spatial import ConvexHull
from rdkit.Chem import rdMolDescriptors
import seaborn as sns
import matplotlib.pyplot as plt
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
import py3Dmol
import joblib

# 读取 Excel 文件
excel_file = r'xyz_maker\SMILES.csv'  # 替换为你的 Excel 文件路径
df = pd.read_csv(excel_file)

# 假设 SMILES 字符串在列名为 'SMILES' 的列中
smiles_column = 'SMILES'
smiles_list = df[smiles_column].tolist()

# 创建一个空的 DataFrame 用于存储结果
results = pd.DataFrame(columns=[
    'SMILES', 'PMI1', 'PMI2', 'PMI3', 
    'Planar_RMSD', 'Inertia_Ratio', 'Max_Planar_Deviation', 'Conjugated_Ratio',
    'Principal_Axes_Cosines_X', 'Principal_Axes_Cosines_Y', 'Principal_Axes_Cosines_Z',
    'Projection_Area_Ratio',
    'Gasteiger_Charge_Mean', 'Gasteiger_Charge_Std',  # Gasteiger电荷
    'EState_Mean', 'EState_Std',  # 电拓扑状态（E-State）
    'MolMR'  # 摩尔折射率
])

# 定义平面性描述符计算函数
def compute_planar_rmsd(mol):
    conf = mol.GetConformer()
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    pca = PCA(n_components=2)
    pca.fit(coords)
    plane_normal = np.cross(pca.components_[0], pca.components_[1])
    plane_normal /= np.linalg.norm(plane_normal)
    distances = np.abs(np.dot(coords - pca.mean_, plane_normal))
    return np.sqrt(np.mean(distances**2))

def compute_inertia_ratio(mol):
    conf = mol.GetConformer()
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    inertia = np.cov(coords.T)
    eigvals = np.linalg.eigvalsh(inertia)
    return eigvals[0] / eigvals[-1]

def compute_max_planar_deviation(mol):
    conf = mol.GetConformer()
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    pca = PCA(n_components=2)
    pca.fit(coords)
    plane_normal = np.cross(pca.components_[0], pca.components_[1])
    plane_normal /= np.linalg.norm(plane_normal)
    distances = np.abs(np.dot(coords - pca.mean_, plane_normal))
    return np.max(distances)

def compute_conjugated_ratio(mol):
    conjugated_atoms = set()
    for bond in mol.GetBonds():
        if bond.GetIsConjugated():
            conjugated_atoms.add(bond.GetBeginAtomIdx())
            conjugated_atoms.add(bond.GetEndAtomIdx())
    return len(conjugated_atoms) / mol.GetNumAtoms()

# 定义三维取向描述符计算函数
def compute_principal_axes(mol):
    conf = mol.GetConformer()
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    inertia = np.cov(coords.T)
    eigvals, eigvecs = np.linalg.eigh(inertia)
    return eigvecs.T  # 返回三个主轴方向向量

def compute_projection_area(mol):
    conf = mol.GetConformer()
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    areas = []
    for i in [(0,1), (1,2), (0,2)]:  # XY, YZ, XZ平面
        proj = coords[:, i]
        hull = ConvexHull(proj)
        areas.append(hull.volume)  # 在2D中volume即为面积
    return np.argmax(areas)  # 返回最大投影平面的索引

# 遍历每一行，处理每个 SMILES 字符串
for index, row in df.iterrows():
    smiles = row[smiles_column]
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    if mol is not None:
        # 生成分子的 3D 坐标
        AllChem.EmbedMolecule(mol)
        
        # 计算 3D 描述符
        pmi1 = Descriptors3D.PMI1(mol)
        pmi2 = Descriptors3D.PMI2(mol)
        pmi3 = Descriptors3D.PMI3(mol)
        
        # 计算平面性描述符
        planar_rmsd = compute_planar_rmsd(mol)
        inertia_ratio = compute_inertia_ratio(mol)
        max_planar_deviation = compute_max_planar_deviation(mol)
        conjugated_ratio = compute_conjugated_ratio(mol)
        
        # 计算三维取向描述符
        principal_axes = compute_principal_axes(mol)
        projection_area_ratio = compute_projection_area(mol)

        # 计算 Gasteiger 电荷
        AllChem.ComputeGasteigerCharges(mol)
        gast_charges = [atom.GetDoubleProp('_GasteigerCharge') for atom in mol.GetAtoms()]
        gast_charge_mean = np.mean(gast_charges)
        gast_charge_std = np.std(gast_charges)

        # 计算电拓扑状态（E-State）
        estate_indices = EState.EStateIndices(mol)
        estate_mean = np.mean(estate_indices)
        estate_std = np.std(estate_indices)

        # 计算摩尔折射率（MolMR）
        mol_mr = Descriptors.MolMR(mol)
        #计算拓扑极性表面积TPSA
        tpsa=Descriptors.TPSA(mol)

        # 将结果存储到 DataFrame 中
        result_row = pd.DataFrame({
            'SMILES': [smiles],
            'PMI1': [pmi1],
            'PMI2': [pmi2],
            'PMI3': [pmi3],
            'Planar_RMSD': [planar_rmsd],
            'Inertia_Ratio': [inertia_ratio],
            'Max_Planar_Deviation': [max_planar_deviation],
            'Conjugated_Ratio': [conjugated_ratio],
            'Principal_Axes_Cosines_X': [principal_axes[0][0]],
            'Principal_Axes_Cosines_Y': [principal_axes[0][1]],
            'Principal_Axes_Cosines_Z': [principal_axes[0][2]],
            'Projection_Area_Ratio': [projection_area_ratio],
            'Gasteiger_Charge_Mean': [gast_charge_mean],
            'Gasteiger_Charge_Std': [gast_charge_std],
            'EState_Mean': [estate_mean],
            'EState_Std': [estate_std],
            'MolMR': [mol_mr],
            "TPSA":[tpsa]
        })
        
        # 使用 pd.concat 合并结果
        results = pd.concat([results, result_row], ignore_index=True)
    else:
        print(f"Invalid SMILES string at row {index}: {smiles}")

# 保存结果到新的 Excel 文件
results.to_excel(r'discrepors_maker\molecules_results.xlsx', index=False)

print("Processing complete. Results saved to 'molecules_results.xlsx'.")

# 分子指纹生成和降维部分保持不变
fingerprints = []

# 遍历每一行，处理每个 SMILES 字符串
for index, row in df.iterrows():
    smiles = row[smiles_column]
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is not None:
        # 生成分子指纹
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)  # 生成 2048 位的 Morgan 指纹
        fingerprints.append(fp.ToBitString())
    else:
        print(f"Invalid SMILES string at row {index}: {smiles}")

# 将指纹转换为 DataFrame
fingerprints_df = pd.DataFrame([list(map(int, fp)) for fp in fingerprints])

# 使用 PCA 进行降维
#pca = PCA(n_components=50)  # 首先降维到 50 维
loaded_pca = joblib.load('discrepors_maker\pca_model.joblib')
fingerprints_pca = loaded_pca.transform(fingerprints_df)

d = 10
# 使用 UMAP 进一步降维到 2 维
umap_model = umap.UMAP(n_components=d)
fingerprints_umap = umap_model.fit_transform(fingerprints_pca)

for i in range(d):
    df[f"UMAP{i+1}"] = fingerprints_umap[:, i]

# 保存结果到新的 Excel 文件
df.to_excel(r'discrepors_maker\molecules_results_1.xlsx', index=False)

print("Processing complete. Results saved to 'molecules_results_1.xlsx'.")
numeric_columns = results.select_dtypes(include='number')


