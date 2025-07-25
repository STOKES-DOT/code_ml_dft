#!/usr/bin/env python3
"""
unzip_wheels.py
Run inside the folder that contains *.whl files:
    python unzip_wheels.py
Effect:
- Extract every *.whl in the current directory
- Delete the original *.whl after extraction
"""
import zipfile
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent

def main() -> None:
    wheels = list(PKG_DIR.glob("*.whl"))
    if not wheels:
        print("No .whl files found.")
        return

    for whl in wheels:
        print(f"Extracting {whl.name} ...")
        with zipfile.ZipFile(whl) as zf:
            zf.extractall(PKG_DIR)
        whl.unlink()
        print(f"Removed {whl.name}")

    print(">>> All wheels extracted.")

if __name__ == "__main__":
    main()