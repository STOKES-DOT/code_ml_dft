# In your main application
from descriptors_maker_QM import grep_qm

try:
    # Generate input files for 50 molecular structures
    input_files = grep_qm.generate_gaussian_inputs(
        template_path="templates/calc_template.gjf",
        structure_folder="molecular_structures"
    )
    
    # Create submission scripts for all generated inputs
    if input_files:
        scripts = grep_qm.generate_slurm_scripts(
            template_path="templates/slurm_template.slurm"
        )
        print(f"Created {len(scripts)} submission scripts")
    
    # Submit jobs to cluster (pseudo-code)
    for script in scripts:
        os.system(f"sbatch {script}")
        
except Exception as e:
    print(f"Workflow failed: {str(e)}")