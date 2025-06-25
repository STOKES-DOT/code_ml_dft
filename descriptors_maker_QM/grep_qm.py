import re
import pandas as pd
from collections import OrderedDict
from pathlib import Path
class MultipoleParser:
    # Define regex patterns as class constants for better performance
    DIPOLE_PATTERN = re.compile(
        r'Dipole moment \(field-independent basis, Debye\):\s*\n\s*'
        r'X=\s*([-\d.]+)\s+Y=\s*([-\d.]+)\s+Z=\s*([-\d.]+)\s+Tot=\s*([-\d.]+)'
    )
    QUADRUPOLE_PATTERN = re.compile(
        r'Quadrupole moment \(field-independent basis, Debye-Ang\):\s*\n\s*'
        r'XX=\s*([-\d.]+)\s+YY=\s*([-\d.]+)\s+ZZ=\s*([-\d.]+)\s*\n\s*'
        r'XY=\s*([-\d.]+)\s+XZ=\s*([-\d.]+)\s+YZ=\s*([-\d.]+)'
    )
    OCTAPOLE_PATTERN = re.compile(
        r'Octapole moment \(field-independent basis, Debye-Ang\*\*2\):\s*\n\s*'
        r'XXX=\s*([-\d.]+)\s+YYY=\s*([-\d.]+)\s+ZZZ=\s*([-\d.]+)\s+XYY=\s*([-\d.]+)\s*\n\s*'
        r'XXY=\s*([-\d.]+)\s+XXZ=\s*([-\d.]+)\s+XZZ=\s*([-\d.]+)\s+YZZ=\s*([-\d.]+)\s*\n\s*'
        r'YYZ=\s*([-\d.]+)\s+XYZ=\s*([-\d.]+)'
    )
    HEXADECAPOLE_PATTERN = re.compile(
        r'Hexadecapole moment \(field-independent basis, Debye-Ang\*\*3\):\s*\n\s*'
        r'XXXX=\s*([-\d.]+)\s+YYYY=\s*([-\d.]+)\s+ZZZZ=\s*([-\d.]+)\s+XXXY=\s*([-\d.]+)\s*\n\s*'
        r'XXXZ=\s*([-\d.]+)\s+YYYX=\s*([-\d.]+)\s+YYYZ=\s*([-\d.]+)\s+ZZZX=\s*([-\d.]+)\s*\n\s*'
        r'ZZZY=\s*([-\d.]+)\s+XXYY=\s*([-\d.]+)\s+XXZZ=\s*([-\d.]+)\s+YYZZ=\s*([-\d.]+)'
    )
    
    def __init__(self):
        """Initialize the multipole parser with pre-compiled regex patterns"""
        # Patterns are already defined as class constants
        pass
    
    def parse_multipole_moments(self, file_content):
        """
        Parse multipole moments from Gaussian output content
        
        Args:
            file_content (str): Content of Gaussian output file
            
        Returns:
            OrderedDict: Parsed multipole moment values
        """
        results = OrderedDict()
        
        # Parse dipole moments
        dipole_match = self.DIPOLE_PATTERN.search(file_content)
        if dipole_match:
            results['x'] = float(dipole_match.group(1))
            results['y'] = float(dipole_match.group(2))
            results['z'] = float(dipole_match.group(3))
            results['tot'] = float(dipole_match.group(4))
        
        # Parse quadrupole moments
        quad_match = self.QUADRUPOLE_PATTERN.search(file_content)
        if quad_match:
            results['xx'] = float(quad_match.group(1))
            results['yy'] = float(quad_match.group(2))
            results['zz'] = float(quad_match.group(3))
            results['xy'] = float(quad_match.group(4))
            results['xz'] = float(quad_match.group(5))
            results['yz'] = float(quad_match.group(6))
        
        # Parse octapole moments
        octa_match = self.OCTAPOLE_PATTERN.search(file_content)
        if octa_match:
            results['xxx'] = float(octa_match.group(1))
            results['yyy'] = float(octa_match.group(2))
            results['zzz'] = float(octa_match.group(3))
            results['xyy'] = float(octa_match.group(4))
            results['xxy'] = float(octa_match.group(5))
            results['xxz'] = float(octa_match.group(6))
            results['xzz'] = float(octa_match.group(7))
            results['yzz'] = float(octa_match.group(8))
            results['xyz'] = float(octa_match.group(9))
        
        # Parse hexadecapole moments
        hexa_match = self.HEXADECAPOLE_PATTERN.search(file_content)
        if hexa_match:
            results['xxxx'] = float(hexa_match.group(1))
            results['yyyy'] = float(hexa_match.group(2))
            results['zzzz'] = float(hexa_match.group(3))
            results['xxyy'] = float(hexa_match.group(4))
            results['xxyz'] = float(hexa_match.group(5))
            results['yyyx'] = float(hexa_match.group(6))
            results['yyyz'] = float(hexa_match.group(7))
            results['zzzx'] = float(hexa_match.group(8))
            results['zzzy'] = float(hexa_match.group(9))
            results['xxzz'] = float(hexa_match.group(10))
            results['yyzz'] = float(hexa_match.group(11))
        
        return results
    
    def process_files(self, file_list, include_filename=True):
        """
        Process multiple log files and return results as a DataFrame
        
        Args:
            file_list (list): List of file paths to process
            include_filename (bool): Whether to include filename in results
            
        Returns:
            pd.DataFrame: DataFrame containing parsed results
        """
        all_data = []
        
        for file_path in file_list:
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                    file_data = self.parse_multipole_moments(content)
                    
                    # Add filename if requested
                    if include_filename:
                        file_data['file_name'] = Path(file_path).name
                    
                    all_data.append(file_data)
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
        
        if not all_data:
            return pd.DataFrame()
        
        # Create DataFrame with consistent column order
        columns = list(all_data[0].keys())
        if include_filename:
            # Move filename to last column
            columns.remove('file_name')
            columns.append('file_name')
        
        return pd.DataFrame(all_data, columns=columns)
    
    def process_directory(self, directory_path, pattern="*.log", include_filename=True):
        """
        Process all matching files in a directory
        
        Args:
            directory_path (str): Path to directory with log files
            pattern (str): File pattern to match
            include_filename (bool): Whether to include filename in results
            
        Returns:
            pd.DataFrame: DataFrame containing parsed results
        """
        dir_path = Path(directory_path)
        if not dir_path.exists() or not dir_path.is_dir():
            raise ValueError(f"Invalid directory: {directory_path}")
        
        file_list = list(dir_path.glob(pattern))
        if not file_list:
            print(f"No files found matching pattern: {pattern}")
            return pd.DataFrame()
        
        return self.process_files(file_list, include_filename)