# descriptor_calculator.py
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors3D, Descriptors, EState
import numpy as np
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
import joblib
import umap
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

def calculate_descriptors(smiles):
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

    # 处理 SMILES 字符串
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Invalid SMILES string: {smiles}")
        return None

    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)

    # 计算描述符
    pmi1 = Descriptors3D.PMI1(mol)
    pmi2 = Descriptors3D.PMI2(mol)
    pmi3 = Descriptors3D.PMI3(mol)
    planar_rmsd = compute_planar_rmsd(mol)
    inertia_ratio = compute_inertia_ratio(mol)
    max_planar_deviation = compute_max_planar_deviation(mol)
    conjugated_ratio = compute_conjugated_ratio(mol)
    principal_axes = compute_principal_axes(mol)
    projection_area_ratio = compute_projection_area(mol)

    AllChem.ComputeGasteigerCharges(mol)
    gast_charges = [atom.GetDoubleProp('_GasteigerCharge') for atom in mol.GetAtoms()]
    gast_charge_mean = np.mean(gast_charges)
    gast_charge_std = np.std(gast_charges)

    estate_indices = EState.EStateIndices(mol)
    estate_mean = np.mean(estate_indices)
    estate_std = np.std(estate_indices)

    mol_mr = Descriptors.MolMR(mol)

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
        'MolMR': [mol_mr]
    })

    results = pd.concat([results, result_row], ignore_index=True)

    return results

def calculate_descriptors_umap(smiles, pca_model_path='descriptors_maker/pca_model.joblib', umap_model=joblib.load(r'descriptors_maker/umap_model.joblib')):

    # 加载PCA模型
    loaded_pca = joblib.load(pca_model_path)
    
    # 将SMILES字符串转换为分子对象
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"无法将SMILES字符串转换为有效的分子对象: {smiles}")
    
    # 生成分子指纹
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    
    # 将指纹转换为NumPy数组
    fp_array = np.array(fp)
    
    # 对指纹进行PCA降维
    fingerprints_pca = loaded_pca.transform(fp_array.reshape(1, -1))
    
    # 对PCA降维后的指纹进行UMAP降维
    if umap_model is None:
        raise ValueError("UMAP模型未提供")
    fingerprints_umap = umap_model.transform(fingerprints_pca)
    
    return fingerprints_umap.flatten()

# 示例用法：
if __name__ == "__main__":
    
    # 示例SMILES字符串
    smiles_list = ["CCO", "CCN", "CCC"]
    
    for smiles in smiles_list:
        try:
            result = calculate_descriptors_umap(smiles)
            print(f"SMILES: {smiles}, UMAP特征: {result}")
        except Exception as e:
            print(f"处理SMILES {smiles}时出错: {str(e)}")