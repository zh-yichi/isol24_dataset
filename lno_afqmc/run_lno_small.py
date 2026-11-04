#!/usr/bin/env python
"""HF-guided LNO-AFQMC/pt2CCSD for the small ISOL24 subset, one LNO threshold per run.

The mean field is NOT recomputed: it is rebuilt from the converged SCF checkpoint of the
reference run (run_isol24_small.py), REF_DIR/<tag>/rhf.chk or uhf.chk, whichever the
reference chose (REF_DIR/<tag>/result.json, key "reference"). The rebuilt mean field is
checked against the stored SCF energy before anything else runs. A restricted reference
gives the restricted LNO chain (pt2ccsd trial, rhf guide); an unrestricted one the
unrestricted chain (upt2ccsd, uhf).

For every molecule:
  1. IAO fragments, one per heavy atom with its hydrogens ("h2heavy"), built once and
     stored in OUTDIR/<tag>/iao_fragments.h5 so that every threshold uses the same ones,
  2. LNO-MP2 / LNO-CCSD / LNO-AFQMC per fragment with LNO thresholds [10 eta, eta],
     the CPU stage of the next fragment pipelined behind the GPU stage,
  3. a summary.json in OUTDIR/<tag>/thresh_<eta>/ and one line in OUTDIR/summary_<eta>.out
     with the LNO energies next to the canonical ones of the reference run.

    python run_lno_small.py 1e-5                      # every molecule with a reference chk
    python run_lno_small.py 1e-5 03_educt 03_product  # a selection
    python run_lno_small.py 1e-5 --skip-done          # resume
    python run_lno_small.py 1e-5 --list               # what would run, no computation

Everything else is configured in the SETTINGS block.
"""

import argparse
import json
import os
import sys
import time
import traceback
from functools import partial

# XLA's Triton GEMM emitter fails on the float32 products of the unrestricted kernels on
# jaxlib 0.8.x; cuBLAS does not. Before any JAX import.
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false " + os.environ.get("XLA_FLAGS", "")

import numpy as np
from pyscf import gto, lib, scf
from pyscf.data.elements import chemcore

# ----------------------------------------------------------------------------
# SETTINGS -- edit these
# ----------------------------------------------------------------------------
REF_DIR = "./small"              # output directory of run_isol24_small.py (relative to this script)
OUTDIR = "./lno_small"           # OUTDIR/<tag>/thresh_<eta>/ per molecule and threshold

AUXBASIS = None                  # must match the reference run (None -> the matching JK-fit set)
MAX_MEMORY = 250000              # MB, for pyscf
VERBOSE = 4
E_CHECK_TOL = 1e-7               # Ha: rebuilt mean field vs the checkpoint energy

# fragments
FRAG_TYPE = "h2heavy"            # one IAO fragment per heavy atom, hydrogens attached
MORE_LOC = None                  # None, "boys" or "pm"

# AFQMC/pt2CCSD per fragment (as the (CO)6 sweep)
N_WALKERS = 300                  # must be divisible by the number of GPUs
N_EQL_BLOCKS = 200               # tau 50 at dt 0.005
N_BLOCKS = 2000                  # at most; a fragment stops early at the target error
TARGET_ERROR = 1e-3              # Ha, on the total; a fragment stops at 0.7 * target / sqrt(nfrag)
MIN_BLOCKS = 300
SEED = 7
MIXED_PRECISION = True
MIXED_OUTLIER = "none"           # per-walker handling during the run (clamp / reject / none)
OUTLIER_ZETA = 1e12              # post-sampling block rejection per fragment, |E_b - median| < zeta x MAD;
                                 # 1e12 keeps every finite block (the fragment class takes a float, not None);
                                 # 20 is the library default that the (CO)6 sweep used
# ----------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
print = partial(print, flush=True)
MH = 1e3


def _resolve(path):
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(HERE, path))


# ----------------------------------------------------------------------------- reference
def load_reference(ref_dir, tag):
    """The reference run's record and the checkpoint file of the chosen mean field."""
    rdir = os.path.join(ref_dir, tag)
    rfile = os.path.join(rdir, "result.json")
    if not os.path.isfile(rfile):
        raise FileNotFoundError(f"{rfile}: run run_isol24_small.py for {tag} first")
    with open(rfile) as fh:
        ref = json.load(fh)
    chk = os.path.join(rdir, "uhf.chk" if ref["reference"] == "UHF" else "rhf.chk")
    if not os.path.isfile(chk):
        raise FileNotFoundError(f"{chk}: the reference checkpoint is missing")
    return ref, chk


def rebuild_mean_field(chk, unrestricted):
    """The density-fitted RHF/UHF of the reference from its checkpoint, without an SCF."""
    mol = lib.chkfile.load_mol(chk)
    mol.max_memory = MAX_MEMORY
    mol.verbose = VERBOSE
    mol.build()
    mf = (scf.UHF if unrestricted else scf.RHF)(mol).density_fit(auxbasis=AUXBASIS)
    data = lib.chkfile.load(chk, "scf")
    mf.__dict__.update(data)
    e_chk = float(data["e_tot"])
    e_now = float(mf.energy_tot(mf.make_rdm1()))
    if abs(e_now - e_chk) > E_CHECK_TOL:
        raise RuntimeError(
            f"{chk}: the rebuilt mean field gives {e_now:.10f} but the checkpoint holds "
            f"{e_chk:.10f} (difference {e_now - e_chk:.2e}); AUXBASIS or the basis differ"
        )
    mf.converged = True
    print(f"mean field from {chk}: E = {e_chk:.10f} (rebuilt {e_now:.10f}), "
          f"{'UHF' if unrestricted else 'RHF'}, {mol.nao_nr()} AOs, {mol.nelectron} electrons")
    return mf


# ----------------------------------------------------------------------------- LNO
def run_lno(tag, thresh_str, ref, chk, outdir, mesh):
    import trot.lnoafqmc  # noqa: F401  before jax: the allocator that frees memory between fragments
    from trot.lnoafqmc import LnoAfqmcMixed, iao_fragment

    thresh = float(thresh_str)
    unrestricted = ref["reference"] == "UHF"
    moldir = os.path.join(outdir, tag)
    rundir = os.path.join(moldir, f"thresh_{thresh_str}")
    os.makedirs(rundir, exist_ok=True)

    t0 = time.time()
    mf = rebuild_mean_field(chk, unrestricted)
    mol = mf.mol
    nfrozen = int(chemcore(mol))
    if ref.get("frozen") is not None and int(ref["frozen"]) != nfrozen:
        print(f"WARNING: the reference CCSD froze {ref['frozen']} orbitals, chemcore gives {nfrozen}")

    # the same IAO fragments for every threshold of this molecule
    frag_file = os.path.join(moldir, "iao_fragments.h5")
    if os.path.isfile(frag_file):
        lo_coeff, frag_list, frag_name = iao_fragment(mf, nfrozen, frag_type=FRAG_TYPE, more_loc=MORE_LOC,
                                                      read_from=frag_file)
    else:
        lo_coeff, frag_list, frag_name = iao_fragment(mf, nfrozen, frag_type=FRAG_TYPE, more_loc=MORE_LOC,
                                                      save2=frag_file)
    nfrag = len(frag_list)
    print(f"{nfrag} IAO fragments ({FRAG_TYPE}): {frag_name}; nfrozen = {nfrozen}")

    trial, guide = ("upt2ccsd", "uhf") if unrestricted else ("pt2ccsd", "rhf")
    lno = LnoAfqmcMixed(
        mf,
        lo_coeff,
        frag_list,
        frag_name=frag_name,
        lno_thresh=thresh,  # -> [10 thresh, thresh] for occupied / virtual
        nfrozen=nfrozen,
        pipeline=True,  # the CPU stage of the next fragment runs behind the GPU stage
        prefetch=1,
        trial=trial,
        guide=guide,
        mesh=mesh,  # every fragment's walkers over all the GPUs
        target_error=TARGET_ERROR,
        min_blocks=MIN_BLOCKS,
        n_walkers=N_WALKERS,
        n_eql_blocks=N_EQL_BLOCKS,
        n_blocks=N_BLOCKS,
        seed=SEED,
        mixed_precision=MIXED_PRECISION,
        mixed_outlier=MIXED_OUTLIER,
        frag_output=os.path.join(rundir, "fragment.out"),
        lno_output=os.path.join(rundir, "lno_result.out"),
    )
    # LnoAfqmcMixed does not expose the post-sampling block rejection of its fragments;
    # its per-fragment keyword dictionary is passed on to every LnoFragMixed, which does
    lno.qmc_kwargs["outlier_zeta"] = float(OUTLIER_ZETA)
    e_qmc, e_qmc_err = lno.kernel()
    wall = time.time() - t0

    e_hf = float(ref["e_uhf"] if unrestricted else ref["e_rhf"])
    summary = dict(
        tag=tag,
        reference=ref["reference"],
        e_hf=e_hf,
        chkfile=chk,
        lno_thresh=thresh,
        lno_thresh_occ_vir=[float(x) for x in np.atleast_1d(lno.lno_thresh_in)],
        nfrozen=nfrozen,
        nao=int(mol.nao_nr()),
        nfrag=nfrag,
        frag_type=FRAG_TYPE,
        run_frag=list(lno.run_frag),
        frag_name=list(lno.frag_name),
        lno_size=[int(np.max(s)) for s in lno.lno_size],
        lno_nocc=[int(np.max(s)) for s in lno.lno_nocc],
        lno_emp=[float(x) for x in lno.lno_emp],
        lno_ecc=[float(x) for x in lno.lno_ecc],
        lno_eqmc=[float(x) for x in lno.lno_eqmc],
        lno_eqmc_err=[float(x) for x in lno.lno_eqmc_err],
        lno_cc_time=[float(x) for x in lno.lno_cc_time],
        lno_qmc_time=[float(x) for x in lno.lno_qmc_time],
        e_lno_mp2=float(lno.e_mp),
        e_lno_ccsd=float(lno.e_cc),
        e_lno_afqmc=float(e_qmc),
        e_lno_afqmc_err=float(e_qmc_err),
        # the canonical values of the reference run (correlation energies)
        ref_e_corr_mp2=ref.get("e_corr_mp2"),
        ref_e_corr_ccsd=ref.get("e_corr_ccsd"),
        ref_e_corr_afqmc=None if ref.get("e_afqmc") is None else float(ref["e_afqmc"]) - e_hf,
        ref_e_afqmc_err=ref.get("e_afqmc_err"),
        trial=trial,
        guide=guide,
        n_walkers=N_WALKERS,
        n_eql_blocks=N_EQL_BLOCKS,
        n_blocks=N_BLOCKS,
        target_error=TARGET_ERROR,
        seed=SEED,
        mixed_outlier=MIXED_OUTLIER,
        outlier_zeta=OUTLIER_ZETA,
        wall_s=wall,
    )
    with open(os.path.join(rundir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print(f"\n{tag} thresh {thresh:g}: LNO-MP2 {lno.e_mp:.6f}  LNO-CCSD {lno.e_cc:.6f}  "
          f"LNO-AFQMC {e_qmc:.6f} +/- {e_qmc_err:.6f}  ({wall:.0f} s)")
    return summary


# ----------------------------------------------------------------------------- summary
SUMMARY_HEADER = (f"{'molecule':<12} {'ref':>3} {'eta':>7} {'nfrag':>5} {'maxLAS':>6} | "
                  f"{'LNO-MP2':>12} {'REF-MP2':>12} | {'LNO-CCSD':>12} {'REF-CCSD':>12} | "
                  f"{'LNO-AFQMC':>12} {'+/-':>8} {'REF-AFQMC':>12} {'+/-':>8} {'dAFQMC[mHa]':>11} {'+/-':>5} | {'wall[s]':>7}")


def summary_line(s):
    f = lambda x, w=12, p=6: f"{x:>{w}.{p}f}" if x is not None else f"{'-':>{w}}"
    d = de = None
    if s["ref_e_corr_afqmc"] is not None:
        d = MH * (s["e_lno_afqmc"] - s["ref_e_corr_afqmc"])
        de = MH * np.sqrt(s["e_lno_afqmc_err"] ** 2 + (s["ref_e_afqmc_err"] or 0.0) ** 2)
    return (f"{s['tag']:<12} {s['reference']:>3} {s['lno_thresh']:>7.0e} {s['nfrag']:>5} {max(s['lno_size']):>6} | "
            f"{f(s['e_lno_mp2'])} {f(s['ref_e_corr_mp2'])} | {f(s['e_lno_ccsd'])} {f(s['ref_e_corr_ccsd'])} | "
            f"{f(s['e_lno_afqmc'])} {f(s['e_lno_afqmc_err'], 8)} {f(s['ref_e_corr_afqmc'])} {f(s['ref_e_afqmc_err'], 8)} "
            f"{f(d, 11, 2)} {f(de, 5, 2)} | {s['wall_s']:>7.0f}")


# ----------------------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("thresh", help="LNO threshold eta, e.g. 1e-5 (occupied 10 eta, virtual eta)")
    ap.add_argument("molecules", nargs="*", help="tags, e.g. 03_educt (default: every molecule with a reference)")
    ap.add_argument("--list", action="store_true", help="print the molecules and their reference, then exit")
    ap.add_argument("--skip-done", action="store_true", help="skip molecules with a summary.json at this threshold")
    args = ap.parse_args()
    float(args.thresh)  # validate early

    ref_dir = _resolve(REF_DIR)
    outdir = _resolve(OUTDIR)
    tags = args.molecules or sorted(
        d for d in os.listdir(ref_dir) if os.path.isfile(os.path.join(ref_dir, d, "result.json"))
    )
    if not tags:
        sys.exit(f"no reference results under {ref_dir}")

    if args.list:
        print(f"{'molecule':<12} {'ref':>3} {'nao':>4} {'E(HF)':>18} {'E(CCSD)':>18} {'E(AFQMC)':>14} {'+/-':>9}  chk")
        for tag in tags:
            ref, chk = load_reference(ref_dir, tag)
            e_hf = ref["e_uhf"] if ref["reference"] == "UHF" else ref["e_rhf"]
            ea = ref.get("e_afqmc")
            ea_s = "%14.6f" % ea if ea is not None else "%14s" % "-"
            err_s = "%9.6f" % ref["e_afqmc_err"] if ea is not None else "%9s" % "-"
            print(f"{tag:<12} {ref['reference']:>3} {ref['nao']:>4} {e_hf:>18.8f} {ref.get('e_ccsd', float('nan')):>18.8f} "
                  f"{ea_s} {err_s}  {chk}")
        return

    import jax
    from trot.sharding import make_data_mesh

    n_gpu = jax.local_device_count()
    print(f"JAX sees {n_gpu} device(s): {jax.devices()}")
    assert N_WALKERS % n_gpu == 0, f"N_WALKERS={N_WALKERS} is not divisible by {n_gpu} devices"
    mesh = make_data_mesh()

    os.makedirs(outdir, exist_ok=True)
    summary_file = os.path.join(outdir, f"summary_{args.thresh}.out")
    if not os.path.exists(summary_file):
        with open(summary_file, "w") as fh:
            fh.write(SUMMARY_HEADER + "\n")

    failed = []
    for tag in tags:
        done = os.path.join(outdir, tag, f"thresh_{args.thresh}", "summary.json")
        if args.skip_done and os.path.exists(done):
            print(f"{tag}: {done} exists, skipped")
            continue
        print("\n" + "=" * 100 + f"\n{tag}, LNO threshold {args.thresh}\n" + "=" * 100)
        try:
            ref, chk = load_reference(ref_dir, tag)
            s = run_lno(tag, args.thresh, ref, chk, outdir, mesh)
        except Exception:  # keep going with the next molecule, report at the end
            traceback.print_exc()
            failed.append(tag)
            continue
        line = summary_line(s)
        print("\n" + SUMMARY_HEADER + "\n" + line)
        with open(summary_file, "a") as fh:
            fh.write(line + "\n")

    print(f"\ndone: {len(tags) - len(failed)} of {len(tags)} molecules at eta = {args.thresh}; summary in {summary_file}")
    if failed:
        print("failed: " + ", ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
