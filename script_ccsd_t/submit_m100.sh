#!/bin/bash
#SBATCH --job-name=iso_ccsdt
#SBATCH --partition=gpuMI100x8
#SBATCH --account=bdka-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32         # half the node (128 cores total)
#SBATCH --mem=400g                 # ~half of the ~1.8-1.9 TB usable
#SBATCH --gpus-per-node=0          # CPU-only
#SBATCH --constraint="scratch"
#SBATCH --no-requeue
#SBATCH -t 48:00:00
#SBATCH -e slurm.err
#SBATCH -o slurm.out

module purge
module list

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
export JAX_PLATFORMS=cpu

echo "node: $(hostname)"
free -g
python run_rhf_ccsd_t.py
