import os
import re
import shutil
from pathlib import Path

def generate_gaussian_inputs(template_path, structure_folder):
    """
    Generate Gaussian input files based on a template and XYZ structure files.
    
    Args:
        template_path (str): Path to the template GJF file
        structure_folder (str): Path to the folder containing XYZ structure files
        
    Returns:
        list: Paths to the generated GJF files
    """
    # Validate template file existence
    template_path = Path(template_path)
    if not template_path.exists():
        raise FileNotFoundError(f"Error: Template file {template_path} not found")
    
    # Validate structure folder existence
    structure_folder = Path(structure_folder)
    if not structure_folder.exists() or not structure_folder.is_dir():
        raise NotADirectoryError(f"Error: Structure folder {structure_folder} not found")
    
    # Find all XYZ structure files
    xyz_files = list(structure_folder.glob("job_*.xyz"))
    if not xyz_files:
        raise FileNotFoundError(f"Error: No XYZ files found in {structure_folder}")
    
    print(f"Found {len(xyz_files)} structure files. Processing...")
    
    generated_files = []
    
    for xyz_file in xyz_files:
        # Extract task number from filename
        match = re.search(r'job_(\d+)\.xyz', xyz_file.name)
        if not match:
            print(f"Warning: Skipping file with unexpected name: {xyz_file.name}")
            continue
            
        task_number = match.group(1)
        base_name = f"job_{task_number}_g16"
        gjf_file = base_name + ".gjf"
        
        # Copy template file
        shutil.copy(template_path, gjf_file)
        print(f"Created: {gjf_file}")
        
        # Read template content
        with open(gjf_file, 'r') as f:
            content = f.read()
        
        # Update CHK file paths
        updated_content = content.replace("%chk=_n.chk", f"%chk={base_name}_n.chk")
        updated_content = updated_content.replace("%chk=_o.chk", f"%chk={base_name}.chk")
        updated_content = updated_content.replace("%ochk=_o.chk", f"%ochk={base_name}.chk")
        
        # Read XYZ coordinates (skip first two lines)
        with open(xyz_file, 'r') as f:
            xyz_lines = f.readlines()[2:]
        
        # Split template content at the link1 separator
        parts = updated_content.split("--link1--")
        if len(parts) < 2:
            print(f"Warning: '--link1--' separator not found in {gjf_file}")
            continue
            
        # Rebuild file content with coordinates inserted
        new_content = (
            parts[0].rstrip() + "\n" +  # First section (before coordinates)
            "".join(xyz_lines).strip() + "\n\n" +  # XYZ coordinates
            "--link1--" + parts[1]  # Second section (after coordinates)
        )
        
        # Write updated content
        with open(gjf_file, 'w') as f:
            f.write(new_content)
        
        print(f"  - Optimization CHK: {base_name}.chk")
        print(f"  - TDDFT CHK: {base_name}_TDDFT.chk")
        generated_files.append(gjf_file)
    
    print(f"Operation completed. Successfully generated {len(generated_files)} Gaussian input files.")
    return generated_files

def generate_slurm_scripts(template_path):
    """
    Generate SLURM submission scripts based on a template.
    
    Args:
        template_path (str): Path to the template SLURM file
        
    Returns:
        list: Paths to the generated SLURM files
    """
    template_path = Path(template_path)
    if not template_path.exists():
        raise FileNotFoundError(f"Error: Template file {template_path} not found")
    
    # Find all GJF files in current directory
    gjf_files = list(Path.cwd().glob("*.gjf"))
    if not gjf_files:
        print("Warning: No GJF files found in current directory")
        return []
    
    generated_files = []
    
    # Read template content
    with open(template_path, 'r') as f:
        template_content = f.read()
    
    for gjf_file in gjf_files:
        # Derive job name from GJF filename
        job_name = gjf_file.stem
        slurm_file = job_name + ".slurm"
        
        # Replace job name placeholder
        new_content = template_content.replace("jobname1", job_name)
        
        # Write SLURM file
        with open(slurm_file, 'w') as f:
            f.write(new_content)
        
        print(f"Generated SLURM script: {slurm_file}")
        generated_files.append(slurm_file)
    
    print(f"Operation completed. Generated {len(generated_files)} SLURM scripts.")
    return generated_files

def main():
    """Main function demonstrating workflow execution"""
    try:
        # Step 1: Generate Gaussian input files
        gjf_files = generate_gaussian_inputs(
            template_path="test.gjf",
            structure_folder="../../main/structure"
        )
        
        # Step 2: Generate SLURM scripts (if GJF files were created)
        if gjf_files:
            slurm_files = generate_slurm_scripts(
                template_path="test_1.slurm"
            )
        else:
            print("Skipping SLURM script generation as no GJF files were created.")
        
        print("\nAll operations completed successfully.")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Operation aborted due to errors.")

if __name__ == "__main__":
    main()