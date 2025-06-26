#!/bin/bash

# 提交当前目录下的所有 .slurm 文件
for slurm_script in *.slurm; do
  if [ -f "$slurm_script" ]; then
    echo "Submitting job: $slurm_script"
    sbatch "$slurm_script"
  fi
done

echo "All jobs submitted!"