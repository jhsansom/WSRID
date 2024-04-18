#!/bin/bash

#SBATCH --job-name=test_job_eecs588
#SBATCH --mail-user=jhsansom@umich.edu
#SBATCH --mail-type=END
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=50g
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --account=chaijy2
#SBATCH --partition=spgpu
#SBATCH --output=./jobs/%u/%x-%j.log

module load cuda
source ../cleanrl-env/bin/activate

python run_experiment.py