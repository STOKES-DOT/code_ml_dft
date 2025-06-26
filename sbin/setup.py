from setuptools import setup, find_packages
import sys
import os
from sys import platform

# Package metadata
NAME = 'optxc'
VERSION = '1.0.0-beta'
DESCRIPTION = 'Exchange-Correlation Optimization Tool'
AUTHOR = 'Yuan Jiao'
EMAIL = 'jiaoyuan24@mails.ucas.ac.cn '
URL = 'https://github.com/STOKES-DOT/code_ml_dft'
LICENSE = 'UCAS,SAIS'

# Dependencies
INSTALL_REQUIRES = [
    'pandas>=1.3.0',
    'scikit-learn>=1.0.0',
    'joblib>=1.0.0',
    'rdkit-pypi>=2021.9.4'
]

# Platform-specific configurations
if platform == "win32":
    # Windows specific configuration
    try:
        from cx_Freeze import setup, Executable
        BUILD_EXE = True
        build_exe_options = {
            "packages": ["os", "sys", "re", "time", "argparse", "joblib", "pandas", "glob"],
            "excludes": ["tkinter"],
            "include_files": [
                ("models/", "models/"),
                ("src/", "src/")
            ]
        }
        executables = [Executable("src/main.py", base=None, target_name="optxc.exe")]
    except ImportError:
        BUILD_EXE = False
else:
    # Linux/Unix configuration
    BUILD_EXE = False

# Common setup configuration
setup_config = {
    'name': NAME,
    'version': VERSION,
    'description': DESCRIPTION,
    'author': AUTHOR,
    'author_email': EMAIL,
    'url': URL,
    'license': LICENSE,
    'packages': find_packages(),
    'install_requires': INSTALL_REQUIRES,
    'entry_points': {
        'console_scripts': [
            'optxc=src.main:main'
        ]
    },
    'package_data': {
        'optxc': ['models/*/*/*.pkl']
    },
    'classifiers': [
        'Programming Language :: Python :: 3',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows'
    ]
}

# Add platform-specific configurations
if BUILD_EXE:
    setup_config.update({
        'options': {"build_exe": build_exe_options},
        'executables': executables
    })

# Run setup
setup(**setup_config)