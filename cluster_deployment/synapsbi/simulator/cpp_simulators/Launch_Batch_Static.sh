#!/bin/bash        
#                                                                                                                         
#-------------------------------------------------------------                                        
#running a shared memory (multithreaded) job over multiple CPUs 
#-------------------------------------------------------------                 
#            
#SBATCH --job-name=seq_1s4hBreak_static
#SBATCH --reservation=vogelgrp_58
#
#SBATCH --output=sim_workdir/array_%A-%a.log
#
#SBATCH --array=1-100:1 #################################################### TO CHANGE EVERY TIME
#              
#Number of CPU cores to use within one node             
#SBATCH -c 1    
#               
#Define the number of hours the job should run.             
#Maximum runtime is limited to 10 days, ie. 240 hours            
#SBATCH --time=240:00:00              
#               
#Define the amount of RAM used by your job in GigaBytes           
#In shared memory applications this is shared among multiple CPUs            
#SBATCH --mem=2G                
#               
#Send emails when a job starts, it is finished or it exits             
#SBATCH --mail-user=basile.confavreux@ist.ac.at               
#SBATCH --mail-type=ALL              
#                
#Do not requeue the job in the case it fails.              
#SBATCH --no-requeue              
#                
#Do not export the local environment to the compute nodes              
#SBATCH --export=NONE             
unset SLURM_EXPORT_ENV
#               
#Set the number of threads to the SLURM internal variable             
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
#               
#load the respective software module you intend to use               
module purge
module load cmake/3.26.3
module load openmpi/4.1.4
module load boost/1.80.0

echo $HOSTNAME
echo $SLURM_ARRAY_TASK_ID

################## CHOOSE THE RIGHT ONE #######################
# config=/nfs/scistore23/vogelgrp/bconfavr/synapsesbi/synapsbi/simulator/cpp_simulators/bg_IF_EEEIIEII_6pPol_params.txt
# config=/nfs/scistore23/vogelgrp/bconfavr/synapsesbi/synapsbi/simulator/cpp_simulators/bg_CVAIF_EEIE_T4wvceciMLP_params.txt
# config=/nfs/scistore23/vogelgrp/bconfavr/synapsesbi/synapsbi/simulator/cpp_simulators/BND_CVAIF_EEIE_T4wvceciMLP_params.txt
# config=/nfs/scistore23/vogelgrp/bconfavr/synapsesbi/synapsbi/simulator/cpp_simulators/BND_IF_EEEIIEII_6pPol_params.txt
config=/nfs/scistore23/vogelgrp/bconfavr/synapsesbi/synapsbi/simulator/cpp_simulators/seq_IF_EEEIIEII_6pPol_params.txt
cls=`awk -v line=${SLURM_ARRAY_TASK_ID} 'NR==line' $config`
sleep $SLURM_ARRAY_TASK_ID
echo $cls
$cls #would actually run the C++ compiled file

duration=$(( SECONDS - start ))
echo "task lasted ${duration} seconds"
