#!/bin/bash
#SBATCH --job-name=isol24lno
##SBATCH --output=myjob.out
#SBATCH --partition=gpuH200x8
#SBATCH --mem=200G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --constraint="scratch"
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest   # select a cpu close to gpu on pci bus topology
#SBATCH --account=bdka-delta-gpu    # <- match to a "Project" returned by the "accounts" command
###SBATCH --exclusive  # dedicated node for this job
#SBATCH --no-requeue
#SBATCH -t 48:00:00
#SBATCH -e slurm.err
#SBATCH -o slurm.out

module purge
# module unload cudatoolkit
module load cuda/12.8

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

export TMPDIR=/u/yzhang65/myscratch/tmp
source /projects/bdka/yzhang65/software/miniconda3/etc/profile.d/conda.sh
conda activate pyscf

export PYTHONPATH=/u/yzhang65/myprojects/software/pyscf:/u/yzhang65/myprojects/software/afqmc:$PYTHONPATH
export PYSCF_EXT_PATH=/u/yzhang65/myprojects/software/pyscf-forge
export PYSCF_TMPDIR=/projects/bdka/yzhang65/scratch/pyscf
unset LD_LIBRARY_PATH
export JAX_ENABLE_X64=True
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python run_lnoafqmc.py
