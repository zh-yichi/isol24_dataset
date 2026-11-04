import os, time

os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false " + os.environ.get("XLA_FLAGS", "")

from pyscf import gto, scf, cc, lib, mp
import jax
from trot.afqmc import AfqmcMixed
from trot.sharding import make_data_mesh


path2chk = './mf.chk'
path2qmc = './pt2_stage.h5'

atoms = '''
C 1.40669300 1.59131300 1.55927800
C -1.40669300 -1.59131300 -1.55927800
O -2.53335500 -1.46263300 -1.60880400
O 2.53335500 1.46263300 1.60880400
O 0.00000000 2.92526600 -1.60880400
O -2.53335500 1.46263300 1.60880400
O 0.00000000 -2.92526600 1.60880400
O 2.53335500 -1.46263300 -1.60880400
C 0.67477000 -2.01388900 1.55927800
C 2.08146400 -0.42257600 -1.55927800
C -0.67477000 2.01388900 -1.55927800
C -2.08146400 0.42257600 1.55927800
'''

mol = gto.M(atom=atoms,
            basis='ccpvdz',
            verbose=4,
            unit='A',
            symmetry=0,
            charge=0,
            spin=0,
            max_memory=250000,
            )

mf = scf.UHF(mol).density_fit()
# mf.with_df._cderi_to_save = path2eri
# mf.with_df._cderi = path2eri
mf.chkfile = path2chk

for _ in range(10):
    mo_i, _, stable, _ = mf.stability(return_status=True)
    if stable:
        break
    mf.kernel(dm0=mf.make_rdm1(mo_i, mf.mo_occ))
# print(f"RHF energy: {mf.e_tot:.10f}")

mmp = mp.MP2(mf).set_frozen()
mmp.kernel()

mycc = cc.CCSD(mf).set_frozen()
mycc.conv_tol = 1e-6
mycc.conv_tol_normt = 1e-5
mycc.verbose = 5
mycc.kernel()

# one "data" mesh axis over every visible GPU: the walkers are split evenly across the
# devices (so n_walkers must be divisible by their number), the hamiltonian and the
# trial are replicated on each, and the stochastic reconfiguration resamples from the
# whole population. Restrict the devices with CUDA_VISIBLE_DEVICES if needed.
n_gpu = jax.local_device_count()
print(f"JAX sees {n_gpu} device(s): {jax.devices()}")
mesh = make_data_mesh()
n_walkers = 300
assert n_walkers % n_gpu == 0, f"n_walkers={n_walkers} is not divisible by {n_gpu} devices"

af = AfqmcMixed(
    mycc,
    guide="uhf",
    trial="upt2ccsd_bar",
    n_walkers=n_walkers,
    n_blocks=2000,
    tau_eql=50,
    seed=7,
    mixed_precision=True,
    )

path = af.save_staged(path2qmc)

af.mixed_outlier = 'none'
e_rhf, err_rhf = af.kernel(mesh=mesh)
print(f"AFQMC/pt2CCSD (UHF guide)  energy: {e_rhf:.6f} +/- {err_rhf:.6f}")
print(f"  guide energy: {af.guide_e_tot:.6f} +/- {af.guide_e_err:.6f}")