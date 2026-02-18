#!/bin/bash
#
#
#SBATCH --job-name=postprocessing
#SBATCH --output=postprocessing_-%j.log   
#            %j is a placeholder for the jobid
#
#SBATCH --reservation=vogelgrp_153
#SBATCH --nodelist=delta216
#
#Define the number of hours the job should run. 
#Maximum runtime is limited to 10 days, ie. 240 hours
#SBATCH --time=36:00:00
#
#Define the amount of RAM used by your job in GigaBytes
#SBATCH --mem=100G
#
#SBATCH --no-requeue
#
#Do not export the local environment to the compute nodes
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV
#
#for single-CPU jobs make sure that they use a single thread
export OMP_NUM_THREADS=1
#
#load the respective software module you intend to use
source /nfs/scistore23/vogelgrp/bconfavr/.bashrc
module load miniforge3
conda activate libdyn
#
#run the respective binary through SLURM's srun
srun --cpu_bind=verbose  python /nfs/scistore23/vogelgrp/bconfavr/synapsesbi/Process_sim_results.py