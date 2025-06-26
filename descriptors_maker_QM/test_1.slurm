#!/bin/bash

#BATCH --time=0-1000:00:00
##SBATCH --output="job_1.err"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=96
#SBATCH -p X48Cv4

# Define variable "jobname".

jobname=jobname1
username=`whoami`

ulimit -s unlimited

# Make a directory in scr and copy .com and .g16 file to there.

GAUSS_RUNDIR=/scratchmem/${username}.$SLURM_JOB_ID
if [ ! -a $GAUSS_RUNDIR ]; then
   echo "Scratch directory $GAUSS_RUNDIR created."
   mkdir -p $GAUSS_RUNDIR
fi

cp $SLURM_SUBMIT_DIR/${jobname}.* $GAUSS_RUNDIR

ORIG_SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR}
SLURM_SUBMIT_DIR=${GAUSS_RUNDIR}
# Note: SLURM defaults to running jobs in the directory
# where they are submitted, no need for $PBS_O_WORKDIR

# Setup for Gaussian 16:
# =======================
# Make a scratch directory if it doesn't already exist.

GAUSS_SCRDIR=$GAUSS_RUNDIR/${jobname}
if [ ! -a $GAUSS_SCRDIR ]; then
   echo "Scratch directory $GAUSS_SCRDIR created."
   mkdir -p $GAUSS_SCRDIR
fi
export GAUSS_SCRDIR
echo "Using $GAUSS_SCRDIR for temporary Gaussian 16 files."


# Define the location where Gaussian was installed and run
# a setup script, g16.profile.

#export g16root=/export/home/gaussian/g16_A03_AVX
#source $g16root/g16/bsd/g16.profile

if grep 'avx2 ' /proc/cpuinfo > /dev/null 2>&1
then
  echo "g16.c01-avx2"
  module load  gaussian/g16.c01-avx2
elif grep 'avx ' /proc/cpuinfo > /dev/null 2>&1
then
  echo "g16.c01-avx"
  module load gaussian/g16.c01-avx
elif grep 'sse4_2 ' /proc/cpuinfo > /dev/null 2>&1
then
  echo "g16.c01-sse4_2"
  module load gaussian/g16.c01-sse4_2
fi 

G16_NODEFILE=${SLURM_JOB_NODELIST}

GAUSS_LFLAGS="-mp 4 -nodefile $SLURM_JOB_NODELIST"

# Export variable list

export PATH GAUSS_LFLAGS

echo slurm nodefile:

echo $SLURM_JOB_NODELIST

#Run a Gaussian command file, water03.com, redirecting output
#to a file, water16.log

cd $GAUSS_RUNDIR

echo "Starting Gaussian run at" `date`

time g16 < $GAUSS_RUNDIR/${jobname}.gjf >$GAUSS_RUNDIR/${jobname}.log

echo "Finished Gaussian run at" `date`

SLURM_SUBMIT_DIR=${ORIG_SLURM_SUBMIT_DIR}

echo $SLURM_SUBMIT_DIR

mv $GAUSS_RUNDIR/*.chk $SLURM_SUBMIT_DIR
mv $GAUSS_RUNDIR/${jobname}.* $SLURM_SUBMIT_DIR
echo "$GAUSS_SCRDIR"

rm -Rf $GAUSS_RUNDIR


