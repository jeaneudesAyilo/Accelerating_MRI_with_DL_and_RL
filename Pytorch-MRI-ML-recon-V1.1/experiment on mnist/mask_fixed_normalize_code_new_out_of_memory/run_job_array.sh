#!/bin/bash

#SBATCH --job-name="run_oom_fixed_mask" 
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.out
#SBATCH --mail-user=your@mail
#SBATCH --mail-type=ALL
#SBATCH --time=96:00:00
#SBATCH --mem=64GB
#SBATCH --array=0-180%9
#SBATCH --exclude=sensei1,lifnode1,see4c1,adnvideo1,asfalda1,selexini-1
##SBATCH --dependency=afterany:84222
#***************************
#DO NOT MODIFY THESE OPTIONS
##SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#***************************


#bash "config_learn_general/hyperp_train_general_$(($SLURM_ARRAY_TASK_ID)).sh"
#echo "run_hps_$(($SLURM_ARRAY_TASK_ID+1000)).sh"

bash "config_learn_fixedmask/learn_fixedmask.$(($SLURM_ARRAY_TASK_ID)).sh"

