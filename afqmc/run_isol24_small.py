#!/usr/bin/env python
"""DF-HF -> DF-MP2 -> DF-CCSD -> AFQMC/pt2CCSD for the small ISOL24 subset (geometry/small).

For every .xyz file the script

  1. converges a density-fitted RHF and drives it to an internally stable solution
     (orbital rotation along the instability + the second-order Newton/CIAH solver),
  2. runs the RHF -> UHF external stability analysis; when the RHF is unstable, a DF-UHF
     is started from the rotated orbitals and driven to an internally stable solution,
  3. decides the reference: the UHF is used only when it is a different solution from the
     RHF (lower by more than E_TOL or with <S^2> above S2_TOL); otherwise the RHF is kept
     and the whole post-HF chain is restricted (RCCSD, pt2ccsd_bar trial, rhf guide),
     which is cheaper than the unrestricted one (UCCSD, upt2ccsd_bar, uhf guide),
  4. runs DF-MP2 and DF-CCSD with the chemical frozen core and stages the AFQMC inputs
     (cholesky vectors, guide, pt2CCSD amplitudes) on the host,
  5. runs AFQMC/pt2CCSD on every visible GPU (walkers split over the devices),

and appends one line per molecule to OUTDIR/summary.out, with the details in
OUTDIR/<tag>/result.json. The molecules run from small to large (ORDER = "size": basis
functions, then electrons), so the short ones finish first and the pipeline fills early.

CPU/GPU pipeline. Steps 1-4 are the host stage, step 5 the device stage. The host stage
of the next molecule runs on a background thread as soon as the current molecule's AFQMC
sampling starts (PREFETCH = 1 keeps one molecule ahead), so the SCF / CCSD time is hidden
behind the sampling. Two molecules' CC objects are then alive at once; MAX_MEMORY is per
molecule. PREFETCH = 0 (or --hf-only) runs everything in order on the main thread.

The pyscf log of each molecule (SCF, stability, CCSD iterations) goes to
OUTDIR/<tag>/<tag>.out so that the two stages do not interleave on stdout; the AFQMC
output and one-line progress messages go to stdout, as in example_afqmc.py.

Reactions 23 and 24 are cations (charge +1, see CHARGES); every other molecule is neutral.

    python run_isol24_small.py                      # every molecule in GEOM_DIR
    python run_isol24_small.py 03_educt 03_product  # a selection (with or without .xyz)
    python run_isol24_small.py --list               # what would run, in run order, with charges and sizes
    python run_isol24_small.py --skip-done          # resume: skip molecules with a result.json
    python run_isol24_small.py --hf-only            # HF / MP2 / CCSD only, no GPU needed

Everything else is configured in the SETTINGS block.
"""

import argparse
import gc
import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from functools import partial

# before any JAX import: the platform allocator returns device memory between molecules,
# cuBLAS instead of the Triton GEMM emitter for the float32 products of the unrestricted kernels
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false " + os.environ.get("XLA_FLAGS", "")

import numpy as np
from pyscf import cc, gto, mp, scf

# ----------------------------------------------------------------------------
# SETTINGS -- edit these
# ----------------------------------------------------------------------------
GEOM_DIR = "../geometry/small"   # relative paths are resolved against this script
OUTDIR = "./small"               # one sub-directory per molecule + summary.out

BASIS = "cc-pVDZ"
AUXBASIS = None                  # DF auxiliary basis; None -> the matching JK-fit set
UNIT = "Angstrom"
CHARGES = {"23": 1, "24": 1}     # by reaction number; anything else is neutral
MAX_MEMORY = 250000              # MB, for pyscf (per molecule; two are alive in the pipeline)
VERBOSE = 4
PYSCF_LOG = True                 # pyscf output to OUTDIR/<tag>/<tag>.out (False: stdout)
ORDER = "size"                   # run order: "size" (basis functions, then electrons) or "name"

# pipeline
PREFETCH = 1                     # host stages run ahead of the device stage; 0 = serial

# mean field
CONV_TOL = 1e-10
MAX_STAB_CYCLE = 10              # stability / restart iterations for RHF and for UHF
E_TOL = 1e-6                     # Ha: UHF counts as a different solution below E_RHF - E_TOL
S2_TOL = 1e-3                    # ... or with <S^2> above this

# coupled cluster
CC_CONV_TOL = 1e-6
CC_CONV_TOL_NORMT = 1e-5
CC_MAX_CYCLE = 100

# AFQMC/pt2CCSD
N_WALKERS = 300                  # must be divisible by the number of GPUs
N_BLOCKS = 2000
TAU_EQL = 50
SEED = 7
MIXED_PRECISION = True
MIXED_OUTLIER = "none"           # per-walker handling during the run (clamp / reject / none)
OUTLIER_ZETA = None              # post-sampling block rejection; None keeps every block
# ----------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
print = partial(print, flush=True)


def _resolve(path):
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(HERE, path))


def read_xyz(path):
    """(atom list for gto.Mole, comment line) from an .xyz file."""
    with open(path) as fh:
        lines = fh.read().splitlines()
    natm = int(lines[0].split()[0])
    comment = lines[1].strip() if len(lines) > 1 else ""
    atoms = []
    for line in lines[2:]:
        if line.strip():
            sym, x, y, z = line.split()[:4]
            atoms.append((sym, (float(x), float(y), float(z))))
    if len(atoms) != natm:
        raise ValueError(f"{path}: header says {natm} atoms, found {len(atoms)}")
    return atoms, comment


def charge_of(tag):
    """The charge from the reaction number in the file name, e.g. '23_educt' -> 1."""
    return CHARGES.get(tag.split("_")[0], 0)


def size_key(xyz):
    """(basis functions, electrons, name): the run order of ORDER = "size"."""
    tag = os.path.splitext(os.path.basename(xyz))[0]
    mol = gto.M(atom=read_xyz(xyz)[0], basis=BASIS, charge=charge_of(tag), unit=UNIT, verbose=0)
    return (mol.nao_nr(), mol.nelectron, tag)


def build_mol(xyz, output=None):
    atoms, comment = read_xyz(xyz)
    tag = os.path.splitext(os.path.basename(xyz))[0]
    mol = gto.M(atom=atoms, basis=BASIS, charge=charge_of(tag), spin=0, unit=UNIT,
                symmetry=0, max_memory=MAX_MEMORY, verbose=VERBOSE, output=output)
    mol.comment = comment
    return mol


# ----------------------------------------------------------------------------- mean field
def _drive_internal(mf, label):
    """Repeat SCF + internal stability until the solution is stable; Newton after the
    first instability. Returns (mf, stable, n_cycles)."""
    second_order = False
    stable = False
    cycle = 0
    for cycle in range(1, MAX_STAB_CYCLE + 1):
        mo_i, _, stable_i, _ = mf.stability(internal=True, external=False, return_status=True)
        if stable_i and mf.converged:
            stable = True
            print(f"[{label}] stable after {cycle} stability cycle(s), E = {mf.e_tot:.12f}")
            break
        print(f"[{label}] cycle {cycle}: internal instability or unconverged SCF; "
              "rotating the orbitals and restarting with the second-order solver")
        mo_occ = mf.mo_occ
        if not second_order:
            mf = mf.newton()
            second_order = True
        mf.kernel(mo_i, mo_occ)
    if not stable:
        print(f"[{label}] WARNING: no stable solution after {MAX_STAB_CYCLE} stability cycles")
    return mf, stable, cycle


def run_stable_rhf(mol, chkfile, tag):
    mf = scf.RHF(mol).density_fit(auxbasis=AUXBASIS)
    mf.conv_tol = CONV_TOL
    mf.chkfile = chkfile
    mf.kernel()
    return _drive_internal(mf, f"{tag} RHF")


def run_stable_uhf(mol, mo_rhf_to_uhf, mo_occ_rhf, chkfile, tag):
    """DF-UHF started from the orbitals of the RHF -> UHF instability, then driven to
    internal stability."""
    mf = scf.UHF(mol).density_fit(auxbasis=AUXBASIS)
    mf.conv_tol = CONV_TOL
    mf.chkfile = chkfile
    occ = np.asarray(mo_occ_rhf > 0, dtype=float)
    dm0 = mf.make_rdm1(mo_rhf_to_uhf, (occ, occ))
    mf.kernel(dm0=dm0)
    return _drive_internal(mf, f"{tag} UHF")


def run_mean_field(mol, workdir, tag):
    """RHF (internal stability) -> external RHF->UHF check -> UHF (internal stability) if
    needed. Returns (mf, info): mf is the RHF or the UHF the post-HF chain is built on."""
    t0 = time.time()
    mf_r, stable_r, ncyc_r = run_stable_rhf(mol, os.path.join(workdir, "rhf.chk"), tag)
    info = dict(e_rhf=float(mf_r.e_tot), rhf_converged=bool(mf_r.converged),
                rhf_stable=bool(stable_r), rhf_stab_cycles=int(ncyc_r))

    _, mo_e, _, stable_e = mf_r.stability(internal=False, external=True, return_status=True)
    info["rhf_external_stable"] = bool(stable_e)
    if stable_e:
        print(f"[{tag} HF] RHF is stable against RHF -> UHF: restricted reference, E = {mf_r.e_tot:.12f}")
        info.update(reference="RHF", e_uhf=None, s2=0.0, uhf_stable=None, uhf_stab_cycles=0)
        mf = mf_r
    else:
        print(f"[{tag} HF] RHF -> UHF instability found: running UHF from the rotated orbitals")
        mf_u, stable_u, ncyc_u = run_stable_uhf(mol, mo_e, mf_r.mo_occ, os.path.join(workdir, "uhf.chk"), tag)
        s2, mult = mf_u.spin_square()
        de = float(mf_u.e_tot - mf_r.e_tot)
        info.update(e_uhf=float(mf_u.e_tot), s2=float(s2), uhf_converged=bool(mf_u.converged),
                    uhf_stable=bool(stable_u), uhf_stab_cycles=int(ncyc_u), e_uhf_minus_rhf=de)
        print(f"[{tag} HF] E(UHF) - E(RHF) = {de:.3e} Ha, <S^2> = {s2:.3e}")
        # the UHF is the RHF solution only if both the energy and <S^2> agree with it
        same_solution = (de >= -E_TOL) and (s2 <= S2_TOL)
        if same_solution:
            print(f"[{tag} HF] the UHF collapsed onto the RHF solution: restricted reference")
            info["reference"] = "RHF"
            mf = mf_r
        else:
            print(f"[{tag} HF] the UHF is a different solution: unrestricted reference")
            info["reference"] = "UHF"
            mf = mf_u
    info["t_hf"] = time.time() - t0
    return mf, info


# ----------------------------------------------------------------------------- post-HF
def run_mp2_ccsd(mf, tag):
    t0 = time.time()
    mmp = mp.MP2(mf).set_frozen()
    mmp.kernel()
    mycc = cc.CCSD(mf).set_frozen()
    mycc.conv_tol = CC_CONV_TOL
    mycc.conv_tol_normt = CC_CONV_TOL_NORMT
    mycc.max_cycle = CC_MAX_CYCLE
    mycc.verbose = 5
    mycc.kernel()
    if not mycc.converged:
        print(f"[{tag} CC] WARNING: CCSD did not converge")
    print(f"[{tag} CC] E(MP2) = {mmp.e_tot:.10f}  E(CCSD) = {mycc.e_tot:.10f}  "
          f"({type(mycc).__name__}, {time.time() - t0:.0f} s)")
    info = dict(frozen=int(mycc.frozen or 0), e_mp2=float(mmp.e_tot), e_corr_mp2=float(mmp.e_corr),
                e_ccsd=float(mycc.e_tot), e_corr_ccsd=float(mycc.e_corr),
                ccsd_converged=bool(mycc.converged), t_cc=time.time() - t0)
    return mycc, info


def stage_afqmc(mycc, restricted, workdir, tag):
    """Build the mixed AFQMC object and stage its inputs: host work only (no device)."""
    from trot.afqmc import AfqmcMixed

    t0 = time.time()
    guide, trial = ("rhf", "pt2ccsd_bar") if restricted else ("uhf", "upt2ccsd_bar")
    af = AfqmcMixed(mycc, guide=guide, trial=trial, n_walkers=N_WALKERS, n_blocks=N_BLOCKS,
                    tau_eql=TAU_EQL, seed=SEED, mixed_precision=MIXED_PRECISION)
    af.save_staged(os.path.join(workdir, "pt2_stage.h5"))
    af.mixed_outlier = MIXED_OUTLIER
    print(f"[{tag} AFQMC] staged ({guide} guide, {trial} trial, {time.time() - t0:.0f} s)")
    info = dict(guide=guide, trial=trial, n_walkers=N_WALKERS, n_blocks=N_BLOCKS, tau_eql=TAU_EQL,
                seed=SEED, mixed_outlier=MIXED_OUTLIER, outlier_zeta=OUTLIER_ZETA, t_stage=time.time() - t0)
    return af, info


# ----------------------------------------------------------------------------- the two stages
def host_stage(xyz, outdir, hf_only):
    """HF -> MP2 -> CCSD -> AFQMC staging of one molecule. Returns the work item of the
    device stage: dict(tag, workdir, mol, af, result)."""
    tag = os.path.splitext(os.path.basename(xyz))[0]
    workdir = os.path.join(outdir, tag)
    os.makedirs(workdir, exist_ok=True)
    t0 = time.time()
    mol = build_mol(xyz, output=os.path.join(workdir, tag + ".out") if PYSCF_LOG else None)
    print(f"[{tag}] {mol.comment}: atoms {mol.natm}, charge {mol.charge}, electrons {mol.nelectron}, "
          f"basis {BASIS}, AOs {mol.nao_nr()}")
    result = dict(tag=tag, xyz=xyz, comment=mol.comment, natm=mol.natm, charge=mol.charge,
                  nelectron=mol.nelectron, basis=BASIS, nao=mol.nao_nr())
    mf, info = run_mean_field(mol, workdir, tag)
    result.update(info)
    mycc, info = run_mp2_ccsd(mf, tag)
    result.update(info)
    af = None
    if not hf_only:
        af, info = stage_afqmc(mycc, result["reference"] == "RHF", workdir, tag)
        result.update(info)
    result["t_host"] = time.time() - t0
    print(f"[{tag}] host stage done in {result['t_host']:.0f} s")
    return dict(tag=tag, workdir=workdir, mol=mol, af=af, result=result)


def device_stage(item, mesh):
    """AFQMC sampling of one staged molecule on the GPUs."""
    tag, af, result = item["tag"], item["af"], item["result"]
    t0 = time.time()
    e, err = af.kernel(mesh=mesh, outlier_zeta=OUTLIER_ZETA)
    print(f"[{tag} AFQMC] AFQMC/pt2CCSD ({result['guide']} guide, {result['trial']}) energy: {e:.6f} +/- {err:.6f}")
    print(f"[{tag} AFQMC]   guide energy: {af.guide_e_tot:.6f} +/- {af.guide_e_err:.6f}")
    result.update(e_afqmc=float(e), e_afqmc_err=float(err), e_guide=float(af.guide_e_tot),
                  e_guide_err=float(af.guide_e_err), t_afqmc=time.time() - t0)
    return result


def finish(item, t_wait):
    """Write result.json, close the pyscf log, release the molecule."""
    result = item["result"]
    result["t_wait_host"] = t_wait
    result["wall"] = result["t_host"] + result.get("t_afqmc", 0.0)
    with open(os.path.join(item["workdir"], "result.json"), "w") as fh:
        json.dump(result, fh, indent=1)
    if PYSCF_LOG and item["mol"].stdout is not sys.stdout:
        item["mol"].stdout.close()
    item["af"] = item["mol"] = None
    gc.collect()
    return result


# ----------------------------------------------------------------------------- summary
SUMMARY_HEADER = (f"{'molecule':<12} {'nao':>4} {'q':>2} {'ref':>3} {'E(RHF)':>18} {'E(UHF)':>18} {'<S^2>':>8} "
                  f"{'E(MP2)':>18} {'E(CCSD)':>18} {'E(AFQMC)':>14} {'+/-':>9} {'E(guide)':>14} {'+/-':>9} "
                  f"{'t_host[s]':>9} {'t_qmc[s]':>8}")


def summary_line(tag, r):
    f = lambda x, w=18: f"{x:>{w}.8f}" if x is not None else f"{'-':>{w}}"
    return (f"{tag:<12} {r['nao']:>4} {r['charge']:>2} {r['reference']:>3} {f(r['e_rhf'])} {f(r.get('e_uhf'))} "
            f"{r.get('s2', 0.0):>8.4f} {f(r.get('e_mp2'))} {f(r.get('e_ccsd'))} "
            f"{f(r.get('e_afqmc'), 14)} {f(r.get('e_afqmc_err'), 9)} {f(r.get('e_guide'), 14)} {f(r.get('e_guide_err'), 9)} "
            f"{r['t_host']:>9.0f} {r.get('t_afqmc', 0.0):>8.0f}")


# ----------------------------------------------------------------------------- driver
def run_all(xyzs, outdir, mesh, hf_only, prefetch):
    """The molecule loop with the host stage running `prefetch` molecules ahead of the
    device stage on one background thread. Returns (failed tags, summary file)."""
    summary = os.path.join(outdir, "summary.out")
    if not os.path.exists(summary):
        with open(summary, "w") as fh:
            fh.write(SUMMARY_HEADER + "\n")

    depth = 0 if hf_only else max(0, min(int(prefetch), len(xyzs) - 1))
    print(f"CPU/GPU pipeline: {'ON' if depth > 0 else 'OFF'} (prefetch depth {depth}), {len(xyzs)} molecule(s)")
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="host-stage") if depth > 0 else None
    futures = {}

    def schedule(i):
        if executor is not None and 0 <= i < len(xyzs) and i not in futures:
            futures[i] = executor.submit(host_stage, xyzs[i], outdir, hf_only)

    def fill(i):
        """Queue the host stages of molecules i ... i + depth - 1 (depth = 1: the next one)."""
        for k in range(i, min(i + depth, len(xyzs))):
            schedule(k)

    failed = []
    try:
        schedule(0)
        for i, xyz in enumerate(xyzs):
            tag = os.path.basename(xyz)[:-4]
            print("\n" + "=" * 100 + f"\n{tag}  ({i + 1}/{len(xyzs)})\n" + "=" * 100)
            # ---------------- host stage (possibly already finished on the thread)
            t0 = time.time()
            try:
                item = futures.pop(i).result() if executor is not None else host_stage(xyz, outdir, hf_only)
            except Exception:
                traceback.print_exc()
                failed.append(tag)
                fill(i + 1)
                continue
            t_wait = time.time() - t0
            t_host = item["result"]["t_host"]
            hidden = 100.0 * (1.0 - t_wait / t_host) if t_host > 0 else 0.0
            print(f"[{tag}] host stage wait {t_wait:.0f} s of {t_host:.0f} s ({hidden:.0f}% hidden)")
            # the next molecule's host stage starts now, behind this molecule's sampling
            fill(i + 1)
            # ---------------- device stage
            try:
                if not hf_only:
                    device_stage(item, mesh)
            except Exception:
                traceback.print_exc()
                failed.append(tag)
                finish(item, t_wait)
                continue
            r = finish(item, t_wait)
            line = summary_line(tag, r)
            print("\n" + SUMMARY_HEADER + "\n" + line)
            with open(summary, "a") as fh:
                fh.write(line + "\n")
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    return failed, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("molecules", nargs="*", help="file names in GEOM_DIR (default: all), with or without .xyz")
    ap.add_argument("--list", action="store_true", help="print the molecules with charge and size, then exit")
    ap.add_argument("--skip-done", action="store_true", help="skip molecules that already have a result.json")
    ap.add_argument("--hf-only", action="store_true", help="HF / MP2 / CCSD only (no jax, no GPU, no pipeline)")
    ap.add_argument("--prefetch", type=int, default=PREFETCH, help=f"host stages to run ahead (default {PREFETCH}; 0 = serial)")
    args = ap.parse_args()

    geom_dir = _resolve(GEOM_DIR)
    outdir = _resolve(OUTDIR)
    names = args.molecules or sorted(f for f in os.listdir(geom_dir) if f.endswith(".xyz"))
    xyzs = [os.path.join(geom_dir, n if n.endswith(".xyz") else n + ".xyz") for n in names]
    for xyz in xyzs:
        if not os.path.isfile(xyz):
            sys.exit(f"geometry not found: {xyz}")
    if ORDER == "size":  # small molecules first
        xyzs.sort(key=size_key)

    if args.list:
        print(f"{'molecule':<12} {'atoms':>5} {'charge':>6} {'nelec':>5} {'nao':>5}  comment")
        for xyz in xyzs:
            mol = gto.M(atom=read_xyz(xyz)[0], basis=BASIS, charge=charge_of(os.path.basename(xyz)[:-4]),
                        unit=UNIT, verbose=0)
            print(f"{os.path.basename(xyz)[:-4]:<12} {mol.natm:>5} {mol.charge:>6} {mol.nelectron:>5} {mol.nao_nr():>5}  {read_xyz(xyz)[1]}")
        return

    os.makedirs(outdir, exist_ok=True)
    if args.skip_done:
        todo = [x for x in xyzs if not os.path.exists(os.path.join(outdir, os.path.basename(x)[:-4], "result.json"))]
        for x in xyzs:
            if x not in todo:
                print(f"{os.path.basename(x)[:-4]}: result.json exists, skipped")
        xyzs = todo

    mesh = None
    if not args.hf_only:
        import jax
        from trot.afqmc import AfqmcMixed  # noqa: F401  imported on the main thread before the pipeline
        from trot.sharding import make_data_mesh

        n_gpu = jax.local_device_count()
        print(f"JAX sees {n_gpu} device(s): {jax.devices()}")
        assert N_WALKERS % n_gpu == 0, f"N_WALKERS={N_WALKERS} is not divisible by {n_gpu} devices"
        mesh = make_data_mesh()

    failed, summary = run_all(xyzs, outdir, mesh, args.hf_only, args.prefetch)
    print(f"\ndone: {len(xyzs) - len(failed)} of {len(xyzs)} molecules; summary in {summary}")
    if failed:
        print("failed: " + ", ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
