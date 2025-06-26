import numpy as np
import pandas as pd
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Descriptors3D, EState
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
import umap
import os
import math

class MolecularDescriptorCalculator:
    def __init__(self, 
                 pca_model_path=r'descriptors_maker\pca_model.joblib', 
                 umap_model_path=r'descriptors_maker\umap_model.joblib',
                 xyz_output_dir='xyz_molecules'):
        """
        分子描述符计算器
        参数:
        pca_model_path: PCA模型文件路径
        umap_model_path: UMAP模型文件路径
        xyz_output_dir: XYZ文件输出目录
        """
        self.pca_model = joblib.load(pca_model_path)
        self.umap_model = joblib.load(umap_model_path)
        self.xyz_output_dir = xyz_output_dir
        
        # 确保输出目录存在
        os.makedirs(self.xyz_output_dir, exist_ok=True)
    
    @staticmethod
    def _compute_planar_rmsd(mol):
        conf = mol.GetConformer()
        coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
        pca = PCA(n_components=2)
        pca.fit(coords)
        plane_normal = np.cross(pca.components_[0], pca.components_[1])
        plane_normal /= np.linalg.norm(plane_normal)
        distances = np.abs(np.dot(coords - pca.mean_, plane_normal))
        return np.sqrt(np.mean(distances**2))
    
    @staticmethod
    def _compute_inertia_ratio(mol):
        conf = mol.GetConformer()
        coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
        inertia = np.cov(coords.T)
        eigvals = np.linalg.eigvalsh(inertia)
        return eigvals[0] / eigvals[-1]
    
    @staticmethod
    def _compute_max_planar_deviation(mol):
        conf = mol.GetConformer()
        coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
        pca = PCA(n_components=2)
        pca.fit(coords)
        plane_normal = np.cross(pca.components_[0], pca.components_[1])
        plane_normal /= np.linalg.norm(plane_normal)
        distances = np.abs(np.dot(coords - pca.mean_, plane_normal))
        return np.max(distances)
    
    @staticmethod
    def _compute_conjugated_ratio(mol):
        conjugated_atoms = set()
        for bond in mol.GetBonds():
            if bond.GetIsConjugated():
                conjugated_atoms.add(bond.GetBeginAtomIdx())
                conjugated_atoms.add(bond.GetEndAtomIdx())
        return len(conjugated_atoms) / mol.GetNumAtoms()
    
    @staticmethod
    def _compute_principal_axes(mol):
        conf = mol.GetConformer()
        coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
        inertia = np.cov(coords.T)
        eigvals, eigvecs = np.linalg.eigh(inertia)
        return eigvecs.T  # 返回三个主轴方向向量
    
    @staticmethod
    def _compute_projection_area(mol):
        conf = mol.GetConformer()
        coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
        areas = []
        for i in [(0,1), (1,2), (0,2)]:  # XY, YZ, XZ平面
            proj = coords[:, i]
            hull = ConvexHull(proj)
            areas.append(hull.volume)  # 在2D中volume即为面积
        return np.argmax(areas)  # 返回最大投影平面的索引
    
    def _generate_3d_structure(self, mol):
        """为分子生成3D结构并保存为XYZ文件"""
        # 添加氢原子并生成3D坐标
        mol = Chem.AddHs(mol)
        state = AllChem.EmbedMolecule(mol, useRandomCoords=True)
        if state == -1:
            raise RuntimeError("Failed to generate 3D coordinates for molecule")
        
        # 优化分子
        AllChem.MMFFOptimizeMolecule(mol)
        return mol
    
    def _save_xyz_file(self, mol, index):
        """将分子保存为XYZ文件"""
        filename = os.path.join(self.xyz_output_dir, f"job_{index+1}.xyz")
        Chem.MolToXYZFile(mol, filename)
        return filename
    
    def _calculate_3D_descriptors(self, mol):
        """计算3D结构描述符"""
        # 计算3D描述符
        pmi1 = Descriptors3D.PMI1(mol)
        pmi2 = Descriptors3D.PMI2(mol)
        pmi3 = Descriptors3D.PMI3(mol)
        planar_rmsd = self._compute_planar_rmsd(mol)
        inertia_ratio = self._compute_inertia_ratio(mol)
        max_planar_deviation = self._compute_max_planar_deviation(mol)
        conjugated_ratio = self._compute_conjugated_ratio(mol)
        principal_axes = self._compute_principal_axes(mol)
        projection_area_ratio = self._compute_projection_area(mol)
        
        # 计算电荷相关描述符
        AllChem.ComputeGasteigerCharges(mol)
        gast_charges = [atom.GetDoubleProp('_GasteigerCharge') for atom in mol.GetAtoms()]
        gast_charge_mean = np.mean(gast_charges)
        gast_charge_std = np.std(gast_charges)
        
        # 计算E-State描述符
        estate_indices = EState.EStateIndices(mol)
        estate_mean = np.mean(estate_indices)
        estate_std = np.std(estate_indices)
        
        # 计算摩尔折射率
        mol_mr = Descriptors.MolMR(mol)
        tpsa = Descriptors.TPSA(mol)
        
        return {
            'PMI1': pmi1,
            'PMI2': pmi2,
            'PMI3': pmi3,
            'Planar_RMSD': planar_rmsd,
            'Inertia_Ratio': inertia_ratio,
            'Max_Planar_Deviation': max_planar_deviation,
            'Conjugated_Ratio': conjugated_ratio,
            'Principal_Axes_Cosines_X': principal_axes[0][0],
            'Principal_Axes_Cosines_Y': principal_axes[0][1],
            'Principal_Axes_Cosines_Z': principal_axes[0][2],
            'Projection_Area_Ratio': projection_area_ratio,
            'Gasteiger_Charge_Mean': gast_charge_mean,
            'Gasteiger_Charge_Std': gast_charge_std,
            'EState_Mean': estate_mean,
            'EState_Std': estate_std,
            'MolMR': mol_mr,
            'TPSA': tpsa
        }
    
    def _calculate_umap_descriptors(self, smiles):
        """计算UMAP降维描述符"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"无法将SMILES字符串转换为有效的分子对象: {smiles}")
        
        # 生成分子指纹
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        fp_array = np.array(fp).reshape(1, -1)
        
        # 应用PCA和UMAP降维
        fingerprints_pca = self.pca_model.transform(fp_array)
        fingerprints_umap = self.umap_model.transform(fingerprints_pca)
        
        # 创建UMAP描述符字典
        umap_dict = {}
        for i in range(fingerprints_umap.shape[1]):
            umap_dict[f'UMAP{i+1}'] = fingerprints_umap[0, i]
        
        return umap_dict
    
    def calculate_descriptors(self, smiles, index=None, save_xyz=True):
        """
        计算所有分子描述符
        
        参数:
        smiles: SMILES字符串
        index: 分子索引（用于文件名）
        save_xyz: 是否保存XYZ文件
        
        返回:
        包含所有描述符的字典
        """
        # 转换SMILES为分子对象
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"无效的SMILES字符串: {smiles}")
        
        # 生成3D结构
        mol_3d = self._generate_3d_structure(mol)
        
        # 保存XYZ文件
        xyz_filename = None
        if save_xyz and index is not None:
            xyz_filename = self._save_xyz_file(mol_3d, index)
        
        # 计算3D描述符
        descriptors_3d = self._calculate_3D_descriptors(mol_3d)
        
        # 计算UMAP描述符
        descriptors_umap = self._calculate_umap_descriptors(smiles)
        
        # 合并所有描述符
        all_descriptors = {
            **descriptors_3d,
            **descriptors_umap
        }
        
        return all_descriptors
    
    def calculate_descriptors_batch(self, smiles_list, save_xyz=True, info_csv='molecule_info.csv'):
        """
        批量计算分子描述符并生成XYZ文件
        
        参数:
        smiles_list: SMILES字符串列表
        save_xyz: 是否保存XYZ文件
        info_csv: 分子信息CSV文件名
        
        返回:
        包含所有描述符的DataFrame
        """
        results = []
        xyz_info = []
        
        for i, smiles in enumerate(smiles_list):
            try:
                descriptors = self.calculate_descriptors(smiles, i, save_xyz)
                results.append(descriptors)
                
                if save_xyz:
                    xyz_info.append({
                        'SMILES': smiles,
                        'XYZ_Filename': descriptors['XYZ_Filename']
                    })
            except Exception as e:
                print(f"处理SMILES {smiles} 时出错: {str(e)}")
                results.append({'SMILES': smiles})  # 保留SMILES即使计算失败
        
        # 保存分子信息到CSV
        if save_xyz and xyz_info:
            df_info = pd.DataFrame(xyz_info)
            info_path = os.path.join(self.xyz_output_dir, info_csv)
            df_info.to_csv(info_path, index=False)
            print(f"分子信息已保存到: {info_path}")
        
        return pd.DataFrame(results)

    
