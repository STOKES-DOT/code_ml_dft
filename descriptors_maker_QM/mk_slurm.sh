#!/bin/bash

template_file="test_1.slurm"

if [ ! -f "$template_file" ]; then
    echo "$template_file Wrong!"
    exit 1
fi


for gjf_file in *.gjf; do
   
    jobname1="${gjf_file%.gjf}"
    
    slurm_file="${jobname1}.slurm"
    
    sed "s/jobname1/$jobname1/g" "$template_file" > "$slurm_file"
    
    echo "Finished: $slurm_file"
done
