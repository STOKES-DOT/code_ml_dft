from xyz_maker import xyz_maker
from descriptors_maker_QM import QM_DFT,grep_qm
from descriptors_maker import descriptors_maker
import pandas as pd

smiles = pd.read_csv('smiles.csv')
smiles = smiles['smiles'].tolist()
xyz_maker = xyz_maker.XYZGenerator(output_dir='xyz_files')
descriptors_maker = descriptors_maker.MolecularDescriptorCalculator()
xyz_maker.generate_xyz_batch(smiles)

df = [descriptors_maker.calculate_descriptors(smiles[i], i) for i in range(len(smiles))]
print(df)
df = pd.DataFrame(df)
df["smiles"] = smiles

df.to_csv('descriptors.csv', index=False)