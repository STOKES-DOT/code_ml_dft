# OPTXC

```
    .::::     .:::::::  .::: .::::::.::      .::    .::   
  .::    .::  .::    .::     .::     .::   .::   .::   .::
.::        .::.::    .::     .::      .:: .::   .::       
.::        .::.:::::::       .::        .::     .::       
.::        .::.::            .::      .:: .::   .::       
  .::     .:: .::            .::     .::   .::   .::   .::
    .::::     .::            .::    .::      .::   .::::    ::
```

**OPTXC** (Exchange-Correlation Optimization Tool) is a machine learning-powered framework for optimizing exchange-correlation (XC) functionals in Time-Dependent Density Functional Theory (TDDFT) calculations. It focuses on improving the accuracy of charge-transfer (CT) and local excited state predictions for single molecular systems.

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Basic XC Functional Optimization](#basic-xc-functional-optimization)
  - [Input File Configuration](#input-file-configuration)
  - [Batch Processing](#batch-processing)
- [Project Structure](#project-structure)
- [Supported Models and Functionals](#supported-models-and-functionals)
- [Examples](#examples)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [Support](#support)
- [License](#license)
- [Contact](#contact)

## Features

- **SMILES to 3D Structure Conversion**: Automatically generates optimized 3D molecular structures from SMILES strings
- **Descriptor Calculation**: Computes both classical molecular descriptors and quantum mechanical (QM) descriptors
- **ML-Based XC Optimization**: Uses ensemble machine learning models (XGBoost, Random Forest, LightGBM, CatBoost, etc.) to predict optimal XC functional parameters
- **Gaussian Integration**: Automated generation and submission of Gaussian input files for QM calculations
- **Supported Functionals**: LC-ωPBE and ωB97XD functionals
- **SLURM Compatibility**: Built-in support for HPC cluster job submission
- **Comprehensive Output**: Detailed reports with molecular descriptors, multipole moments, and optimized parameters

## Prerequisites

- **Python**: 3.6 or higher
- **Gaussian**: G16 or compatible version (for QM calculations)
- **Internet Connection**: Required during initial installation
- **Operating System**: Linux/Unix (recommended) or Windows

### Required Python Packages

- `pandas >= 1.3.0`
- `scikit-learn >= 1.3.0`
- `joblib >= 1.2.0`
- `rdkit >= 2023.9.2`
- `umap-learn >= 0.5.7`

Additional packages for model training (optional):
- `xgboost`
- `lightgbm`
- `catboost`
- `shap`
- `optuna`

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/STOKES-DOT/OPTXC.git
cd OPTXC
```

### 2. Install the OPTXC Package

```bash
cd sbin
python setup.py install
```

> **Note**: Installation requires an internet connection to download dependencies. The installation will create a command-line tool `optxc` that you can use from anywhere.

### 3. Verify Installation

```bash
optxc --help
```

## Quick Start

Here's a minimal example to optimize XC functionals for biphenyl:

1. Create an input file `example.inp`:

```bash
SMILES=C1=CC=C(C=C1)C2=CC=CC=C2
XC_FUNCTIONAL=LC-wPBE
MACHINE_LEARNING=XGBOOST
BASIS_SET=6-31G(d)
SLURM_PATH=descriptors_maker_QM/test_1.slurm
```

2. Run the optimization:

```bash
optxc example.inp
```

3. Check the output in `example.out` for optimized parameters and molecular descriptors.

## Usage

### Basic XC Functional Optimization

The main workflow consists of:

1. **Input Preparation**: Create an input file with molecular SMILES and configuration
2. **Execution**: Run `optxc` with your input file
3. **Analysis**: Review the output file containing optimized parameters

```bash
# Navigate to example directory
cd example

# Run optimization
optxc example.inp

# View results
cat example.out
```

### Input File Configuration

Create an input file with the following parameters:

| Parameter | Description | Options | Required |
|-----------|-------------|---------|----------|
| `SMILES` | Molecular structure in SMILES format | Any valid SMILES string | Yes |
| `XC_FUNCTIONAL` | XC functional to optimize | `LC-wPBE`, `WB97XD` | No (default: WB97XD) |
| `MACHINE_LEARNING` | ML model for optimization | `XGBOOST`, `RANDOMFOREST`, `LIGHTGBM`, `CATBOOST`, `GBDT`, `RIDGE`, `LASSO`, `ELASTICNET`, `ADABOOST`, `SGM` | No (default: XGBOOST) |
| `BASIS_SET` | Basis set for Gaussian calculations | Any valid Gaussian basis set | Yes |
| `OUTPUT_TASK` | Type of calculation | `TDDFT` (default) | No |
| `SLURM_PATH` | Path to SLURM template file | Valid file path | Yes |

**Example Input File:**

```bash
#INPUT_FILE=example.inp
SMILES=C1=CC=C(C=C1)C2=CC=CC=C2
XC_FUNCTIONAL=LC-wPBE
MACHINE_LEARNING=XGBOOST
BASIS_SET=6-31G(d)
OUTPUT_TASK=TDDFT
SLURM_PATH=descriptors_maker_QM/test_1.slurm
```

### Batch Processing

For processing multiple molecules, use the TADF example:

```python
from xyz_maker import xyz_maker
from descriptors_maker import descriptors_maker
import pandas as pd

# Load SMILES from CSV
smiles_df = pd.read_csv('smiles.csv')
smiles_list = smiles_df['smiles'].tolist()

# Generate XYZ files and calculate descriptors
xyz_gen = xyz_maker.XYZGenerator(output_dir='xyz_files')
desc_calc = descriptors_maker.MolecularDescriptorCalculator()

xyz_gen.generate_xyz_batch(smiles_list)
descriptors = [desc_calc.calculate_descriptors(s, i) for i, s in enumerate(smiles_list)]

# Save results
results_df = pd.DataFrame(descriptors)
results_df['smiles'] = smiles_list
results_df.to_csv('descriptors.csv', index=False)
```

## Project Structure

```
OPTXC/
├── sbin/                      # Main package installation
│   ├── bin/                   # Command-line executable
│   ├── src/                   # Source code for package
│   └── setup.py              # Installation script
├── descriptors_maker/         # Classical descriptor calculation
│   ├── descriptors_maker.py  # Main descriptor calculator
│   └── code_maker.py         # Helper utilities
├── descriptors_maker_QM/      # Quantum mechanical descriptors
│   ├── QM_DFT.py             # Gaussian input generator
│   └── grep_qm.py            # QM output parser
├── xyz_maker/                 # SMILES to XYZ converter
│   └── xyz_maker.py          # XYZ generation utilities
├── SGM_LC-wPBE/              # LC-ωPBE functional models
│   └── Stacking_model/       # Pre-trained ML models
├── SGM_wB97XD/               # ωB97XD functional models
│   └── Stacking_model/       # Pre-trained ML models
├── example/                   # Usage examples
│   ├── example.inp           # Sample input file
│   └── example.slurm         # Sample SLURM script
├── TADF_example/             # Batch processing example
├── model_vis/                # Model visualization and analysis
└── main.py                   # Main execution script
```

## Supported Models and Functionals

### XC Functionals

- **LC-ωPBE**: Long-range corrected ωPBE functional
- **ωB97XD**: Range-separated hybrid functional with empirical dispersion

### Machine Learning Models

| Model | Description | Performance |
|-------|-------------|-------------|
| `SGM` | Stacking ensemble model | Highest accuracy |
| `XGBOOST` | Extreme Gradient Boosting | High accuracy, fast |
| `RANDOMFOREST` | Random Forest | Good balance |
| `LIGHTGBM` | Light Gradient Boosting | Fast, memory efficient |
| `CATBOOST` | Categorical Boosting | Robust to overfitting |
| `GBDT` | Gradient Boosting Decision Tree | Good accuracy |
| `RIDGE` | Ridge Regression | Simple, interpretable |
| `LASSO` | Lasso Regression | Feature selection |
| `ELASTICNET` | Elastic Net | Combines L1/L2 |
| `ADABOOST` | Adaptive Boosting | Ensemble method |

> **Note**: The stacking models (`SGM`) are not included in the repository due to size constraints. Users can train them using the scripts in `SGM_LC-wPBE/` or `SGM_wB97XD/` directories. Individual base learners can be used directly with slightly reduced performance.

## Examples

### Example 1: Single Molecule Optimization

Optimize LC-ωPBE functional for naphthalene:

```bash
# Create input file
cat > naphthalene.inp << EOF
SMILES=C1=CC=C2C=CC=CC2=C1
XC_FUNCTIONAL=LC-wPBE
MACHINE_LEARNING=XGBOOST
BASIS_SET=6-31G(d)
SLURM_PATH=descriptors_maker_QM/test_1.slurm
EOF

# Run optimization
optxc naphthalene.inp
```

### Example 2: Using Python API

```python
from xyz_maker import xyz_maker
from descriptors_maker import descriptors_maker

# Initialize generators
xyz_gen = xyz_maker.XYZGenerator(output_dir='output_xyz')
desc_calc = descriptors_maker.MolecularDescriptorCalculator()

# Process molecule
smiles = "c1ccccc1"  # Benzene
xyz_gen.generate_xyz(smiles)
descriptors = desc_calc.calculate_descriptors(smiles)
print(descriptors)
```

See the [TADF_example](TADF_example/) directory for more comprehensive examples including batch processing and result analysis.

## Documentation

For detailed information about the methodology and theoretical background, please refer to our upcoming publication.

### Key Descriptor Types

**Classical Descriptors:**
- Geometric structure (PMI, planarity, conjugation)
- Electronic properties (Gasteiger charges, E-State indices)
- Dimensionality reduction (UMAP components)

**Quantum Mechanical Descriptors:**
- Multipole moments (dipole, quadrupole, octapole, hexadecapole)
- Derived from Gaussian calculations

### Workflow

1. **Structure Generation**: SMILES → 3D coordinates (RDKit)
2. **Classical Descriptors**: Fast molecular property calculation
3. **QM Descriptors**: Gaussian single-point calculations
4. **ML Prediction**: Trained models predict optimal XC parameters
5. **Output Generation**: Comprehensive report with all descriptors and parameters

## Contributing

We welcome contributions! If you'd like to contribute to this project:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Test your changes thoroughly
5. Commit your changes (`git commit -am 'Add new feature'`)
6. Push to the branch (`git push origin feature/improvement`)
7. Create a Pull Request

Please ensure your code follows the existing style and includes appropriate documentation.

## Support

If you encounter issues or have questions:

- **Email**: [jiaoyuan24@mails.ucas.ac.cn](mailto:jiaoyuan24@mails.ucas.ac.cn)
- **Issues**: Open an issue on [GitHub Issues](https://github.com/STOKES-DOT/OPTXC/issues)

We appreciate feedback and are happy to help with any questions about the code or theoretical details!

## License

UCAS, SAIS

## Contact

**Maintainer**: Yuan Jiao  
**Email**: [jiaoyuan24@mails.ucas.ac.cn](mailto:jiaoyuan24@mails.ucas.ac.cn)  
**Institution**: University of Chinese Academy of Sciences (UCAS), School of Artificial Intelligence (SAIS)

---

<div align="center">

**Note**: This project is actively being developed. We welcome feedback and contributions from the community!

*If you find this project useful, please consider citing our work (publication forthcoming).*

</div>

