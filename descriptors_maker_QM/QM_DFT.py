import os
import re
import shutil
import subprocess
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict

class GaussianInputGenerator:
    def __init__(self, 
                 gjf_template: str = r"descriptors_maker_QM\template_QM.gjf",
                 slurm_template: str = r"descriptors_maker_QM\test_1.slurm",
                 structure_folder: str = r"xyz_molecules",
                 output_dir: str = ".",
                 overwrite: bool = False,
                 dry_run: bool = False):
        """
        Initialize Gaussian input file generator
        
        Args:
            gjf_template: Gaussian template file path
            slurm_template: SLURM template file path
            structure_folder: XYZ structure files directory
            output_dir: Directory to save generated files
            overwrite: Overwrite existing files
            dry_run: Simulate operations without writing files
        """
        self.gjf_template = Path(gjf_template).resolve()
        self.slurm_template = Path(slurm_template).resolve()
        self.structure_folder = Path(structure_folder).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.overwrite = overwrite
        self.dry_run = dry_run
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        self.logger = logging.getLogger('GaussianInputGenerator')
        
        # Create output directory if needed
        if not self.dry_run:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def validate_paths(self) -> None:
        """Validate all input paths exist"""
        if not self.gjf_template.exists():
            raise FileNotFoundError(f"GJF template not found: {self.gjf_template}")
        if not self.slurm_template.exists():
            raise FileNotFoundError(f"SLURM template not found: {self.slurm_template}")
        if not self.structure_folder.exists():
            raise NotADirectoryError(f"Structure folder not found: {self.structure_folder}")

    def generate_gaussian_inputs(self, 
                                template_path: Optional[str] = None, 
                                structure_folder: Optional[str] = None
                                ) -> List[str]:
        """
        Generate Gaussian input files
        
        Args:
            template_path: Override GJF template path
            structure_folder: Override XYZ structure folder
            
        Returns:
            Paths to generated GJF files
        """
        template = Path(template_path).resolve() if template_path else self.gjf_template
        structures = Path(structure_folder).resolve() if structure_folder else self.structure_folder
        
        # Validate paths
        for path in [template, structures]:
            if not path.exists():
                raise FileNotFoundError(f"Path not found: {path}")

        # Find XYZ files
        xyz_files = list(structures.glob("job_*.xyz"))
        if not xyz_files:
            raise FileNotFoundError(f"No XYZ files found in {structures}")
        self.logger.info(f"Found {len(xyz_files)} XYZ files")

        generated_files = []
        
        # Read template once
        with open(template, 'r') as f:
            template_content = f.read()

        for xyz_file in xyz_files:
            # Extract task number
            if not (match := re.search(r'job_(\d+)\.xyz', xyz_file.name)):
                self.logger.warning(f"Skipping invalid XYZ file: {xyz_file.name}")
                continue
                
            task_number = match.group(1)
            base_name = f"job_{task_number}_g16"
            gjf_file = self.output_dir / (base_name + ".gjf")
            
            # Skip existing files
            if gjf_file.exists() and not self.overwrite:
                self.logger.info(f"Skipping existing file: {gjf_file.name}")
                continue
                
            # Process template
            content = template_content
            content = content.replace("%chk=_n.chk", f"%chk={base_name}_TDDFT.chk")
            content = content.replace("%chk=_o.chk", f"%chk={base_name}.chk")
            content = content.replace("%ochk=_o.chk", f"%ochk={base_name}.chk")
            
            # Insert coordinates
            with open(xyz_file, 'r') as f:
                xyz_data = ''.join(f.readlines()[2:]).strip()
                
            # Use regex to find insertion point
            if "--link1--" not in content:
                self.logger.warning(f"Missing '--link1--' separator in template")
                continue
                
            parts = content.split("--link1--", 1)
            new_content = f"{parts[0].rstrip()}\n{xyz_data}\n\n--link1--{parts[1]}"
            
            # Write output
            if not self.dry_run:
                with open(gjf_file, 'w') as f:
                    f.write(new_content)
                self.logger.info(f"Created Gaussian input: {gjf_file.name}")
            else:
                self.logger.info(f"[DRY RUN] Would create: {gjf_file.name}")
                
            generated_files.append(str(gjf_file))
            
        self.logger.info(f"Generated {len(generated_files)} Gaussian inputs")
        return generated_files

    def generate_slurm_scripts(self, 
                              template_path: Optional[str] = None
                              ) -> List[str]:
        """
        Generate SLURM submission scripts
        
        Args:
            template_path: Override SLURM template path
            
        Returns:
            Paths to generated SLURM files
        """
        template = Path(template_path).resolve() if template_path else self.slurm_template
        
        if not template.exists():
            raise FileNotFoundError(f"SLURM template not found: {template}")
            
        gjf_files = list(self.output_dir.glob("*.gjf"))
        if not gjf_files:
            self.logger.warning("No GJF files found for SLURM generation")
            return []
            
        generated_files = []
        
        # Read template once
        with open(template, 'r') as f:
            template_content = f.read()

        for gjf_file in gjf_files:
            job_name = gjf_file.stem
            slurm_file = self.output_dir / (job_name + ".slurm")
            
            if slurm_file.exists() and not self.overwrite:
                self.logger.info(f"Skipping existing SLURM script: {slurm_file.name}")
                continue
                
            # Customize template
            content = template_content.replace("jobname1", job_name)
            
            if not self.dry_run:
                with open(slurm_file, 'w') as f:
                    f.write(content)
                self.logger.info(f"Created SLURM script: {slurm_file.name}")
            else:
                self.logger.info(f"[DRY RUN] Would create SLURM: {slurm_file.name}")
                
            generated_files.append(str(slurm_file))
            
        self.logger.info(f"Generated {len(generated_files)} SLURM scripts")
        return generated_files

    def submit_slurm_jobs(self, 
                         directory: Optional[str] = None,
                         pattern: str = "*.slurm"
                         ) -> Dict[str, str]:
        """
        Submit all SLURM scripts in the specified directory
        
        Args:
            directory: Directory containing SLURM scripts (default: output_dir)
            pattern: File pattern to match SLURM scripts
            
        Returns:
            Dictionary mapping script paths to submission outputs
        """
        target_dir = Path(directory) if directory else self.output_dir
        if not target_dir.exists():
            raise NotADirectoryError(f"Directory not found: {target_dir}")
            
        slurm_scripts = list(target_dir.glob(pattern))
        if not slurm_scripts:
            self.logger.warning(f"No SLURM scripts found in {target_dir} with pattern '{pattern}'")
            return {}
            
        results = {}
        self.logger.info(f"Submitting {len(slurm_scripts)} SLURM jobs...")
        
        for script in slurm_scripts:
            if self.dry_run:
                self.logger.info(f"[DRY RUN] Would submit: {script.name}")
                results[str(script)] = "Dry run - no submission"
                continue
                
            try:
                result = subprocess.run(
                    ["sbatch", str(script)],
                    capture_output=True,
                    text=True,
                    check=True
                )
                output = result.stdout.strip()
                self.logger.info(f"Submitted {script.name}: {output}")
                results[str(script)] = output
            except subprocess.CalledProcessError as e:
                error_msg = f"Submission failed for {script.name}: {e.stderr.strip()}"
                self.logger.error(error_msg)
                results[str(script)] = error_msg
            except Exception as e:
                error_msg = f"Unexpected error submitting {script.name}: {str(e)}"
                self.logger.error(error_msg)
                results[str(script)] = error_msg
                
        self.logger.info(f"Submitted {len(results)} jobs")
        return results

    def run_workflow(self, 
                    gjf_template: Optional[str] = None, 
                    structure_folder: Optional[str] = None, 
                    slurm_template: Optional[str] = None,
                    submit_jobs: bool = False
                    ) -> Tuple[List[str], List[str], Dict[str, str]]:
        """
        Execute complete workflow
        
        Args:
            submit_jobs: Automatically submit SLURM jobs after generation
            
        Returns:
            Tuple of (GJF paths, SLURM paths, submission results)
        """
        try:
            self.validate_paths()
            
            # Generate Gaussian inputs
            gjf_files = self.generate_gaussian_inputs(
                template_path=gjf_template,
                structure_folder=structure_folder
            )
            
            # Generate SLURM scripts
            slurm_files = self.generate_slurm_scripts(
                template_path=slurm_template
            ) if gjf_files else []
            
            # Submit jobs if requested
            submission_results = {}
            if submit_jobs and slurm_files:
                submission_results = self.submit_slurm_jobs()
            
            self.logger.info("Workflow completed successfully")
            return gjf_files, slurm_files, submission_results
            
        except Exception as e:
            self.logger.error(f"Workflow failed: {str(e)}", exc_info=True)
            return [], [], {}


# Example usage
if __name__ == "__main__":
    # Create generator with custom options
    generator = GaussianInputGenerator(
        gjf_template=r"descriptors_maker_QM\template_QM.gjf",
        slurm_template=r"descriptors_maker_QM\test_1.slurm",
        structure_folder=r"xyz_molecules",
        output_dir="gaussian_jobs",
        overwrite=False,
        dry_run=False
    )
    
    # Run full workflow with job submission
    gjf_files, slurm_files, submission_results = generator.run_workflow(
        submit_jobs=True
    )
    
    # Alternatively run individual steps
    # gjf_files = generator.generate_gaussian_inputs()
    # slurm_files = generator.generate_slurm_scripts()
    # submission_results = generator.submit_slurm_jobs()