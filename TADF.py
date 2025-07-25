from xyz_maker import xyz_maker
from descriptors_maker_QM import QM_DFT,grep_qm
from descriptors_maker import descriptors_maker
import pandas as pd

smiles = pd.read_csv('smiles.csv')
smiles = smiles['SMILES'].tolist()
xyz_maker = xyz_maker.XYZGenerator(output_dir='xyz_files')
descriptors_maker = descriptors_maker.DescriptorMaker(output_dir='descriptors')
xyz_maker.generate_xyz_files(smiles)
descriptors_maker.generate_descriptors(smiles)
