import os
import subprocess
import sys
from pathlib import Path

# Configuration
SBIN_DIR = Path(__file__).parent
REQUIREMENTS_FILE = SBIN_DIR / "requirements.txt"
PACKAGES = [
    "numpy",
    "pandas",
    "joblib",
    "rdkit-pypi",  # Official RDKit package
    "scipy",
    "scikit-learn",  # Includes PCA and model tools
    "umap-learn",
    "setuptools"    # Base installation tools
]

def main():
    # Create sbin directory
    SBIN_DIR.mkdir(parents=True, exist_ok=True)
    print(f"📦 Target directory: {SBIN_DIR.resolve()}")

    # Verify pip availability
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "--version"])
    except Exception:
        print("❌ Error: pip not available. Install pip before proceeding")
        sys.exit(1)

    # Generate requirements file
    with open(REQUIREMENTS_FILE, "w") as f:
        f.write("\n".join(PACKAGES))
    print(f"📝 Created requirements file: {REQUIREMENTS_FILE}")

    # Download packages
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "download",
        "-d", str(SBIN_DIR),
        "-r", str(REQUIREMENTS_FILE),
        "--no-cache-dir",
        "--prefer-binary"
    ]
    
    print("🚀 Downloading dependencies...")
    print("🔍 Packages:", ", ".join(PACKAGES))
    try:
        subprocess.check_call(cmd)
        print("✅ All dependencies downloaded successfully!")
        
        # Generate offline installation script
        create_install_script()
    except subprocess.CalledProcessError:
        print("❌ Download failed. Check network connection")
        sys.exit(1)

def create_install_script():
    """Create offline installation script"""
    script = f"""#!/bin/bash
echo "Installing Python dependencies from local repository"
pip install --no-index --find-links "{SBIN_DIR.resolve()}" -r "{REQUIREMENTS_FILE.resolve()}"
echo "✅ Installation completed"
"""
    (SBIN_DIR / "install_deps.sh").write_text(script)
    
    print("\nNEXT STEPS:")
    print(f"1. Transfer entire project to offline machine")
    print(f"2. Run installation script:")
   # print(f"   bash {SBIN_DIR}/install_deps.sh")

if __name__ == "__main__":
    main()