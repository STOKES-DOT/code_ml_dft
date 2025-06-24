import re
import pandas as pd
from collections import OrderedDict

def parse_multipole_moments(file_content):

    results = OrderedDict()
    
   
    patterns = {
        'dipole': re.compile(
            r'Dipole moment \(field-independent basis, Debye\):\s*\n\s*'
            r'X=\s*([-\d.]+)\s+Y=\s*([-\d.]+)\s+Z=\s*([-\d.]+)\s+Tot=\s*([-\d.]+)'
        ),
        'quadrupole': re.compile(
            r'Quadrupole moment \(field-independent basis, Debye-Ang\):\s*\n\s*'
            r'XX=\s*([-\d.]+)\s+YY=\s*([-\d.]+)\s+ZZ=\s*([-\d.]+)\s*\n\s*'
            r'XY=\s*([-\d.]+)\s+XZ=\s*([-\d.]+)\s+YZ=\s*([-\d.]+)'
        ),
        'octapole': re.compile(
            r'Octapole moment \(field-independent basis, Debye-Ang\*\*2\):\s*\n\s*'
            r'XXX=\s*([-\d.]+)\s+YYY=\s*([-\d.]+)\s+ZZZ=\s*([-\d.]+)\s+XYY=\s*([-\d.]+)\s*\n\s*'
            r'XXY=\s*([-\d.]+)\s+XXZ=\s*([-\d.]+)\s+XZZ=\s*([-\d.]+)\s+YZZ=\s*([-\d.]+)\s*\n\s*'
            r'YYZ=\s*([-\d.]+)\s+XYZ=\s*([-\d.]+)'
        ),
        'hexadecapole': re.compile(
            r'Hexadecapole moment \(field-independent basis, Debye-Ang\*\*3\):\s*\n\s*'
            r'XXXX=\s*([-\d.]+)\s+YYYY=\s*([-\d.]+)\s+ZZZZ=\s*([-\d.]+)\s+XXXY=\s*([-\d.]+)\s*\n\s*'
            r'XXXZ=\s*([-\d.]+)\s+YYYX=\s*([-\d.]+)\s+YYYZ=\s*([-\d.]+)\s+ZZZX=\s*([-\d.]+)\s*\n\s*'
            r'ZZZY=\s*([-\d.]+)\s+XXYY=\s*([-\d.]+)\s+XXZZ=\s*([-\d.]+)\s+YYZZ=\s*([-\d.]+)'
        )
    }
    
    
    dipole_match = patterns['dipole'].search(file_content)
    if dipole_match:
        results['x'] = float(dipole_match.group(1))
        results['y'] = float(dipole_match.group(2))
        results['z'] = float(dipole_match.group(3))
        results['tot'] = float(dipole_match.group(4))
    
   
    quad_match = patterns['quadrupole'].search(file_content)
    if quad_match:
        results['xx'] = float(quad_match.group(1))
        results['yy'] = float(quad_match.group(2))
        results['zz'] = float(quad_match.group(3))
        results['xy'] = float(quad_match.group(4))
        results['xz'] = float(quad_match.group(5))
        results['yz'] = float(quad_match.group(6))
    
    
    octa_match = patterns['octapole'].search(file_content)
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
    
  
    hexa_match = patterns['hexadecapole'].search(file_content)
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
        results['xxyy'] = float(hexa_match.group(10))
        results['xxzz'] = float(hexa_match.group(11))
        results['yyzz'] = float(hexa_match.group(12))
    
    return results

def process_log_files(file_list):

    all_data = []
    
    for file_path in file_list:
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                file_data = parse_multipole_moments(content)
                all_data.append(file_data)
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    

    if all_data:
      
        columns = list(all_data[0].keys()) if all_data else []
        
       
        if 'file_name' in columns:
            columns.remove('file_name')
            columns.append('file_name')
        
        df = pd.DataFrame(all_data, columns=columns)
        return df
    else:
        return pd.DataFrame()
