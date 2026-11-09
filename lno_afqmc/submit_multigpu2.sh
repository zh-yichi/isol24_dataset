#!/bin/bash
#SBATCH -A jcap_g
#SBATCH -C gpu&hbm80g
##SBATCH -C gpu
##SBATCH -q regular
#SBATCH -q premium
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 128
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH -t 24:00:00
#SBATCH -J lno2isol
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err

# Resources: whole node = 64 physical cores (128 hardware threads),
# 256 GB host DDR4 (--mem=0 = all of it), 4x A100 80GB (hbm80g nodes).

# --- environment (self-contained; conda activation sets
#     PYSCF_EXT_PATH and LD_LIBRARY_PATH for you) ---
unset PYTHONPATH
source /global/common/software/jcap/zhyichi/miniconda3/bin/activate pyscf

# --- CPU threading: one thread per physical core (64 on a GPU node) ---
export OMP_NUM_THREADS=64
export OMP_PLACES=cores
export OMP_PROC_BIND=spread
export MKL_NUM_THREADS=64

export PYTHONPATH=/global/homes/z/zhyichi/software/pyscf:/global/homes/z/zhyichi/software/trot:$PYTHONPATH
export PYSCF_EXT_PATH=/global/homes/z/zhyichi/software/pyscf-forge

# --- PySCF scratch files: compute nodes have no local disk, /tmp is RAM ---
export PYSCF_TMPDIR=/tmp/pyscf_tmp/$SLURM_JOB_ID
mkdir -p $PYSCF_TMPDIR

# --- record what ran ---
echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
nvidia-smi -L
free -g
python -c "import jax; print(jax.devices())"

python -u run_lno_small.py 3e-6 --skip-done

rm -rf $PYSCF_TMPDIR
