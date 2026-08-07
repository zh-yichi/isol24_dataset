#!/usr/bin/env python
"""DF-RHF -> LNO-AFQMC for the ISOL24 geometries.

For every requested .xyz file the script

  1. converges a density-fitted RHF and drives it to an internally stable
     solution with the second-order (Newton/CIAH) solver,
  2. builds IAO local orbitals / atom-centred fragments and runs LNO-AFQMC,
     which also gives LNO-MP2 and LNO-CCSD correlation energies for free,

and collects everything into a single, human-readable `summary_lno.out`.

Each molecule runs inside its own working directory, because the LNO driver
reads/writes `options.bin`, `FCIDUMP_chol`, `amplitudes.npz`, `fragment.out<n>`
and `lno_result.out` in the current directory.

Everything is configured in the SETTINGS block below; just run

    python run_rhf_lnoafqmc.py
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import jax

jax.config.update("jax_enable_x64", True)

import glob
import re
import time
import traceback
from collections import Counter
from datetime import datetime

from pyscf import gto, scf, lib
from pyscf.data import elements

# ----------------------------------------------------------------------------
# SETTINGS -- edit these
# ----------------------------------------------------------------------------
# Directory holding the .xyz files.  Relative paths are resolved with respect to
# this script, so it works no matter where you launch it from.
GEOM_DIR = '../geometry/small'

# Which molecule(s) to run -- no '.xyz' needed:
#                            'all'  -> every .xyz in GEOM_DIR
#                            a name -> '03_educt'
#                            a list -> ['03_educt', '03_product']
MOLECULES = ['14_educt', '14_product', 
             '10_educt', '10_product',
             '20_educt', '20_product',
             '05_educt', '05_product',]

# Order used when MOLECULES == 'all':  'name' or 'size' (fewest heavy atoms
# first, so the cheap jobs finish first).
ORDER = 'size'

BASIS = 'ccpvdz'      # orbital basis
AUXBASIS = None        # DF auxiliary basis; None -> matching JK-fit set
CHARGE = 0             # default charge, overridden per molecule by CHARGES
UNIT = 'Angstrom'      # unit of the coordinates in the .xyz files

# Per-molecule charges.  A key is either a full name ('24_educt') or just the
# reaction number ('24'), which then covers both the educt and the product.
# Reaction 24 (C18H33Si) is a cation: neutral would have an odd electron count.
CHARGES = {
    '23': 1,
    '24': 1,
}

CONV_TOL = 1e-10       # SCF energy convergence
MAX_STAB_CYCLE = 10    # max stability-check / restart iterations
CHECK_EXTERNAL = False # also run the external (RHF -> UHF) stability check

# --- LNO settings -----------------------------------------------------------
DO_LNO_AFQMC = True
LNO_THRESH = 1e-6      # a float -> [10*thresh, thresh]; a pair -> used as is
FROZEN = 'auto'        # 'auto' -> elements.chemcore(mol); an int -> that many
                       # frozen core orbitals; 0 -> all-electron
CHOL_CUT = 1e-5        # Cholesky decomposition threshold for the integrals
TARGET_STO_ERROR = 5e-4  # target stochastic error of the *total* correlation
                         # energy; split over the fragments by the driver
AFQMC_OPTIONS = {
    'eql_time': 40,
    'n_prop_steps': 50,
    'n_blocks': 600,
    'n_walkers': 300,
    'mix_precision': True,
    'max_memory': 100000,
    'seed': 18,
    'walker_type': 'rhf',
    'trial': 'pt2ccsd',
}

MAX_MEMORY = 200000     # MB, for the pyscf part
VERBOSE = 4            # PySCF print level
WORKDIR = '../run_lno2' # one sub-directory per molecule is created here
SUMMARY = 'summary_lno.out'     # written inside WORKDIR
SKIP_EXISTING = True   # skip molecules whose work directory already holds a
                       # finished result (lets you resume an interrupted batch)
# ----------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
HARTREE2KCAL = 627.5094740631
LNO_OUT = 'lno_result.out'


def _resolve(path):
    """Resolve `path` relative to this script's directory."""
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(HERE, path))


def read_xyz(path):
    """Return (atom list for gto.Mole, comment line) from an .xyz file."""
    with open(path) as fh:
        lines = fh.read().splitlines()

    natm = int(lines[0].split()[0])
    comment = lines[1].strip() if len(lines) > 1 else ''

    atoms = []
    for line in lines[2:]:
        if not line.strip():
            continue
        sym, x, y, z = line.split()[:4]
        atoms.append((sym, (float(x), float(y), float(z))))

    if len(atoms) != natm:
        raise ValueError('%s: header says %d atoms, found %d'
                         % (path, natm, len(atoms)))
    return atoms, comment


def formula(atoms):
    """Molecular formula in Hill notation."""
    c = Counter(sym for sym, _ in atoms)
    keys = ['C', 'H'] + sorted(k for k in c if k not in ('C', 'H'))
    return ''.join(k + (str(c[k]) if c[k] > 1 else '') for k in keys if k in c)


def n_heavy(atoms):
    return sum(1 for sym, _ in atoms if sym != 'H')


def charge_of(tag):
    """Charge for molecule `tag`: an entry for the full name wins over one for
    the bare reaction number, which in turn wins over the global default."""
    if tag in CHARGES:
        return CHARGES[tag]
    return CHARGES.get(tag.split('_')[0], CHARGE)


def build_mol(path, tag, output=None):
    """Return (mol, comment, formula).  The metadata is kept outside the Mole
    object so that pyscf's check_sanity does not complain about it."""
    atoms, comment = read_xyz(path)
    charge = charge_of(tag)

    # catch a bad charge here, with a clearer message than pyscf's
    nelec = sum(gto.charge(sym) for sym, _ in atoms) - charge
    if nelec % 2:
        raise ValueError('%s: %d electrons at charge %+d is open shell; RHF '
                         'needs an even count -- set CHARGES[%r]'
                         % (tag, nelec, charge, tag))

    mol = gto.M(atom=atoms, basis=BASIS, charge=charge, spin=0, unit=UNIT,
                max_memory=MAX_MEMORY, verbose=VERBOSE, output=output)
    return mol, comment, formula(atoms)


def run_stable_rhf(mol, chkfile=None):
    """DF-RHF driven until the solution is internally stable.

    Returns (mf, stable, n_cycle) where `mf` is the final mean-field object.
    """
    mf = scf.RHF(mol).density_fit(auxbasis=AUXBASIS)
    mf.conv_tol = CONV_TOL
    if chkfile:
        mf.chkfile = chkfile
    mf.kernel()

    second_order = False
    stable = False
    for cycle in range(1, MAX_STAB_CYCLE + 1):
        mo_i, mo_e, stable_i, stable_e = mf.stability(
            internal=True, external=CHECK_EXTERNAL, return_status=True)

        if CHECK_EXTERNAL and not stable_e:
            lib.logger.warn(mf, 'External instability (RHF -> UHF/complex) '
                                'detected; an unrestricted reference may be '
                                'lower in energy.')

        if stable_i and mf.converged:
            stable = True
            lib.logger.note(mf, 'Stable RHF solution after %d stability '
                                'cycle(s), E = %.12f', cycle, mf.e_tot)
            break

        lib.logger.note(mf, 'Cycle %d: internal instability (or unconverged '
                            'SCF); rotating orbitals and restarting with the '
                            'second-order solver.', cycle)

        mo_occ = mf.mo_occ
        if not second_order:
            # Newton/CIAH on top of the DF-RHF object; density fitting is kept.
            mf = mf.newton()
            second_order = True
        mf.kernel(mo_i, mo_occ)
    else:
        lib.logger.warn(mf, 'No stable solution after %d stability cycles.',
                        MAX_STAB_CYCLE)

    return mf, stable, cycle


def n_frozen(mol):
    """Number of frozen core orbitals for this molecule."""
    return elements.chemcore(mol) if FROZEN == 'auto' else int(FROZEN or 0)


def run_lno_afqmc(mf):
    """IAO fragments + LNO-AFQMC in the current directory.

    Returns (parsed lno_result.out, number of fragments).
    """
    from afqmc.lno_afqmc import lno_afqmc, tools

    lo_coeff, frag_lolist, atm_center = tools.iao_localization(mf)

    lno_afqmc.run_afqmc(
        mf,
        lo_coeff=lo_coeff,
        frag_lolist=frag_lolist,
        nfrozen=n_frozen(mf.mol),
        thresh=LNO_THRESH,
        qmc_options=dict(AFQMC_OPTIONS),
        chol_cut=CHOL_CUT,
        target_sto_error=TARGET_STO_ERROR,
        run_frag_list=None,
        atom_group=atm_center,
    )
    return parse_lno_out(LNO_OUT), len(frag_lolist)


# 'LNO-Thresh' summary row, e.g.
#   [1.00e-04, 1.00e-05]        8  -0.23137630  -0.20887179  -0.21126  0.00122 ...
_SUMMARY_RE = re.compile(r'^\s*(\[[^\]]*\])\s+(\d+)\s+'
                         r'(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+'
                         r'(-?\d+\.\d+)\s+(\d+\.\d+)\s+'
                         r'(\d+\.\d+)\s+(\d+\.\d+)\s*$')
# per-fragment row: index, LAS centre, LAS size, then the six numbers
_FRAG_RE = re.compile(r'^\s*(\d+)\s+(\S+)\s+(\d+)\s+'
                      r'(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+'
                      r'(-?\d+\.\d+)\s+(\d+\.\d+)\s+'
                      r'(\d+\.\d+)\s+(\d+\.\d+)\s*$')


def parse_lno_out(path):
    """Pull the fragment-summed energies out of an lno_result.out.

    Returns a dict with the LNO *correlation* energies ('e_mp2', 'e_ccsd',
    'e_afqmc', 'e_afqmc_err'), the LNO thresholds, the largest active space and
    the accumulated timings -- empty if the file has no complete summary.
    """
    found = {}
    if not os.path.isfile(path):
        return found

    nfrag = 0
    with open(path) as fh:
        for line in fh:
            m = _SUMMARY_RE.match(line)
            if m:
                found.update(lno_thresh=m.group(1), max_las=int(m.group(2)),
                             e_mp2=float(m.group(3)), e_ccsd=float(m.group(4)),
                             e_afqmc=float(m.group(5)),
                             e_afqmc_err=float(m.group(6)),
                             t_cc=float(m.group(7)), t_qmc=float(m.group(8)))
            elif _FRAG_RE.match(line):
                nfrag += 1
    if found:
        found['nfrag'] = nfrag
    return found


def parse_scf_energy(path):
    """Converged SCF energy from a pyscf output file, or None."""
    energy = None
    if not os.path.isfile(path):
        return energy
    with open(path) as fh:
        for line in fh:
            m = re.search(r'converged SCF energy = (-?\d+\.\d+)', line)
            if m:
                energy = float(m.group(1))
    return energy


def select_molecules(geom_dir):
    """Turn the MOLECULES setting into an ordered list of tags (no extension).

    A trailing '.xyz' is accepted but never required.
    """
    if isinstance(MOLECULES, str) and MOLECULES.lower() == 'all':
        tags = [os.path.splitext(os.path.basename(p))[0]
                for p in glob.glob(os.path.join(geom_dir, '*.xyz'))]
        if ORDER == 'size':
            def key(tag):
                atoms, _ = read_xyz(os.path.join(geom_dir, tag + '.xyz'))
                return (n_heavy(atoms), len(atoms), tag)
            tags.sort(key=key)
        else:
            tags.sort()
        return tags

    names = [MOLECULES] if isinstance(MOLECULES, str) else list(MOLECULES)
    return [os.path.splitext(os.path.basename(n))[0] for n in names]


def _store_lno(res, energies, nfrag=None):
    """Copy a parsed lno_result.out into a result dict, adding total energies."""
    res.update(e_corr_mp2=energies['e_mp2'], e_corr_ccsd=energies['e_ccsd'],
               e_corr_afqmc=energies['e_afqmc'],
               e_afqmc_err=energies['e_afqmc_err'],
               lno_thresh=energies['lno_thresh'], max_las=energies['max_las'],
               nfrag=nfrag if nfrag is not None else energies.get('nfrag'),
               t_cc=energies['t_cc'], t_qmc=energies['t_qmc'])
    if 'e_rhf' in res:
        res['e_mp2'] = res['e_rhf'] + energies['e_mp2']
        res['e_ccsd'] = res['e_rhf'] + energies['e_ccsd']
        res['e_afqmc'] = res['e_rhf'] + energies['e_afqmc']
    return res


def run_one(tag, geom_dir, workroot):
    """Run RHF -> LNO-AFQMC for one geometry; returns a result dict."""
    xyz = os.path.join(geom_dir, tag + '.xyz')
    if not os.path.isfile(xyz):
        raise IOError('geometry not found: %s' % xyz)

    rundir = os.path.join(workroot, tag)
    os.makedirs(rundir, exist_ok=True)

    res = {'tag': tag, 'status': 'running'}
    cwd = os.getcwd()
    os.chdir(rundir)          # the LNO driver reads/writes in the current dir
    try:
        mol, comment, form = build_mol(xyz, tag, output=tag + '.out')
        res.update(formula=form, natm=mol.natm, nao=mol.nao_nr(),
                   nelec=mol.nelectron, charge=mol.charge,
                   frozen=n_frozen(mol))
        lib.logger.note(mol, 'Geometry : %s', xyz)
        lib.logger.note(mol, 'Comment  : %s', comment)
        lib.logger.note(mol, 'Atoms    : %d   charge: %+d   basis: %s   AOs: %d',
                        mol.natm, mol.charge, BASIS, mol.nao_nr())

        t0 = time.time()
        mf, stable, ncycle = run_stable_rhf(mol, chkfile=tag + '.chk')
        res.update(e_rhf=mf.e_tot, scf_converged=bool(mf.converged),
                   stable=bool(stable), stab_cycles=ncycle,
                   t_scf=time.time() - t0)

        if DO_LNO_AFQMC:
            t0 = time.time()
            energies, nfrag = run_lno_afqmc(mf)
            res['t_lno'] = time.time() - t0
            if not energies:
                raise RuntimeError('no LNO summary found in %s' % LNO_OUT)
            _store_lno(res, energies, nfrag)

        res['status'] = 'ok'
    except Exception as err:                       # keep the batch going
        res['status'] = 'FAILED: %s' % err
        with open('traceback.txt', 'w') as fh:
            fh.write(traceback.format_exc())
        print('%-16s FAILED: %s' % (tag, err), flush=True)
    finally:
        os.chdir(cwd)

    if res['status'] == 'ok':
        print_result(res)
    return res


def load_one(tag, geom_dir, workroot):
    """Rebuild the result dict of a finished run from the files it left behind.

    Returns None when the run directory does not hold a complete result, so
    that the caller can (re)run the molecule.
    """
    rundir = os.path.join(workroot, tag)
    e_rhf = parse_scf_energy(os.path.join(rundir, tag + '.out'))
    if e_rhf is None:
        return None
    energies = parse_lno_out(os.path.join(rundir, LNO_OUT))
    if DO_LNO_AFQMC and not energies:
        return None

    atoms, _ = read_xyz(os.path.join(geom_dir, tag + '.xyz'))
    res = {'tag': tag, 'status': 'ok', 'cached': True, 'e_rhf': e_rhf,
           'formula': formula(atoms), 'natm': len(atoms),
           'charge': charge_of(tag)}
    if energies:
        _store_lno(res, energies)
    return res


def print_result(res):
    print('%-16s %-12s nao=%4s  E(RHF)=%18.10f  E(LNO-CCSD)=%18.10f  '
          'E(LNO-AFQMC)=%s'
          % (res['tag'], res.get('formula', ''), res.get('nao', '-'),
             res.get('e_rhf', float('nan')),
             res.get('e_ccsd', float('nan')),
             ('%.5f +/- %.5f' % (res['e_afqmc'], res['e_afqmc_err']))
             if 'e_afqmc' in res else 'n/a'), flush=True)


# ----------------------------------------------------------------------------
# summary
# ----------------------------------------------------------------------------

def _f(value, width=18, prec=10):
    return ('%*.*f' % (width, prec, value)) if value is not None else \
        '%*s' % (width, 'n/a')


def write_summary(results, path, geom_dir):
    """(Re)write the summary file from the results collected so far."""
    cols = ('molecule', 'formula', 'chg', 'nao', 'nel', 'frz', 'stab', 'nfrg',
            'LAS', 'E(RHF)', 'E(LNO-MP2)', 'E(LNO-CCSD)', 'E(LNO-AFQMC)',
            'err', 't_scf', 't_cc', 't_qmc')
    head = ('%-16s %-12s %4s %5s %5s %4s %5s %5s %5s %18s %18s %18s %16s %9s '
            '%9s %9s %9s' % cols)

    lines = []
    lines.append('=' * len(head))
    lines.append(' ISOL24 : DF-RHF / LNO-AFQMC summary')
    lines.append('=' * len(head))
    lines.append(' generated : %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    lines.append(' geometry  : %s' % geom_dir)
    lines.append(' basis     : %-14s auxbasis : %-22s charge : %+d%s'
                 % (BASIS, AUXBASIS or 'default JK-fit', CHARGE,
                    ('  (overrides: %s)'
                     % ', '.join('%s %+d' % kv for kv in sorted(CHARGES.items())))
                    if CHARGES else ''))
    lines.append(' scf       : DF-RHF, conv_tol=%.1e, stability loop '
                 '(max %d cycles, second-order restart)'
                 % (CONV_TOL, MAX_STAB_CYCLE))
    if DO_LNO_AFQMC:
        lines.append(' lno       : IAO fragments (atom-centred), thresh=%s, '
                     'frozen : %s' % (LNO_THRESH, FROZEN))
        opt = ', '.join('%s=%s' % kv for kv in sorted(AFQMC_OPTIONS.items()))
        lines.append(' afqmc     : chol_cut=%.1e, target_sto_error=%.1e, %s'
                     % (CHOL_CUT, TARGET_STO_ERROR, opt))
    lines.append(' E(LNO-*) are total energies, E(RHF) + the fragment-summed')
    lines.append(' LNO correlation energy; t_cc / t_qmc are summed over the')
    lines.append(' fragments.  Energies in Hartree, times in seconds.')
    lines.append('')
    lines.append(head)
    lines.append('-' * len(head))

    for r in results:
        if r['status'] != 'ok':
            lines.append('%-16s %-12s %s' % (r['tag'], r.get('formula', ''),
                                             r['status']))
            continue
        lines.append(
            '%-16s %-12s %+4d %5s %5s %4s %5s %5s %5s %s %s %s %s %s %9s %9s %9s'
            % (r['tag'], r.get('formula', ''), r.get('charge', 0),
               r.get('nao', '-'), r.get('nelec', '-'), r.get('frozen', '-'),
               'yes' if r.get('stable') else ('-' if r.get('cached') else 'NO'),
               r.get('nfrag', '-'), r.get('max_las', '-'),
               _f(r.get('e_rhf')), _f(r.get('e_mp2')), _f(r.get('e_ccsd')),
               _f(r.get('e_afqmc'), 16, 5), _f(r.get('e_afqmc_err'), 9, 5),
               '%9.1f' % r['t_scf'] if 't_scf' in r else '%9s' % '-',
               '%9.1f' % r['t_cc'] if 't_cc' in r else '%9s' % '-',
               '%9.1f' % r['t_qmc'] if 't_qmc' in r else '%9s' % '-'))

    lines.append('-' * len(head))

    rxn = reaction_energies(results)
    if rxn:
        lines.append('')
        lines.append(' Isomerization energies, E(product) - E(educt), kcal/mol')
        rhead = ('%-10s %-12s %14s %14s %14s %14s %9s'
                 % ('reaction', 'formula', 'dE(RHF)', 'dE(LNO-MP2)',
                    'dE(LNO-CCSD)', 'dE(LNO-AFQMC)', 'err'))
        lines.append('')
        lines.append(rhead)
        lines.append('-' * len(rhead))
        for idx, form, d in rxn:
            lines.append('%-10s %-12s %s %s %s %s %s'
                         % (idx, form,
                            _f(d.get('rhf'), 14, 4), _f(d.get('mp2'), 14, 4),
                            _f(d.get('ccsd'), 14, 4),
                            _f(d.get('afqmc'), 14, 4),
                            _f(d.get('afqmc_err'), 9, 4)))
        lines.append('-' * len(rhead))

    with open(path, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')


def reaction_energies(results):
    """Pair up <n>_educt / <n>_product results and return the deltas."""
    done = {r['tag']: r for r in results if r['status'] == 'ok'}
    out = []
    for tag in sorted(done):
        if not tag.endswith('_educt'):
            continue
        idx = tag[:-len('_educt')]
        prod = done.get(idx + '_product')
        if prod is None:
            continue
        educt = done[tag]
        d = {}
        for key, name in (('e_rhf', 'rhf'), ('e_mp2', 'mp2'),
                          ('e_ccsd', 'ccsd'), ('e_afqmc', 'afqmc')):
            if key in educt and key in prod:
                d[name] = (prod[key] - educt[key]) * HARTREE2KCAL
        if 'e_afqmc_err' in educt and 'e_afqmc_err' in prod:
            d['afqmc_err'] = ((educt['e_afqmc_err'] ** 2 +
                               prod['e_afqmc_err'] ** 2) ** 0.5) * HARTREE2KCAL
        out.append((idx, educt.get('formula', ''), d))
    return out


def main():
    geom_dir = _resolve(GEOM_DIR)
    workroot = _resolve(WORKDIR)
    os.makedirs(workroot, exist_ok=True)
    summary = os.path.join(workroot, SUMMARY)

    tags = select_molecules(geom_dir)
    print('%d molecule(s) to run from %s' % (len(tags), geom_dir), flush=True)

    results = []
    for i, tag in enumerate(tags, 1):
        if SKIP_EXISTING:
            cached = load_one(tag, geom_dir, workroot)
            if cached is not None:
                print('[%d/%d] %-16s already done, skipping'
                      % (i, len(tags), tag), flush=True)
                results.append(cached)
                write_summary(results, summary, geom_dir)
                continue

        print('\n[%d/%d] %s' % (i, len(tags), tag), flush=True)
        results.append(run_one(tag, geom_dir, workroot))
        write_summary(results, summary, geom_dir)   # keep it up to date

    write_summary(results, summary, geom_dir)
    print('\nSummary written to %s' % summary, flush=True)
    return results


if __name__ == '__main__':
    main()
