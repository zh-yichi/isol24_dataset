#!/usr/bin/env python
"""DF-RHF -> DF-CCSD -> AFQMC for the ISOL24 geometries.

For every requested .xyz file the script

  1. converges a density-fitted RHF and drives it to an internally stable
     solution with the second-order (Newton/CIAH) solver,
  2. runs CCSD (optionally with the (T) correction) on top of it,
  3. prepares the Cholesky integrals / CC amplitudes and launches ph-AFQMC,

and collects everything into a single, human-readable `summary.out`.

Each molecule runs inside its own working directory, because the AFQMC driver
reads `options.bin`, `FCIDUMP_chol` and `amplitudes.npz` from the current
directory and writes `afqmc.out` there.

Everything is configured in the SETTINGS block below; just run

    python run_rhf_ccsd_afqmc.py
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

from pyscf import gto, scf, cc, lib

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
MOLECULES = 'all'

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

CC_ENE_TOL = 1e-6
CC_NORM_TOL = 1e-5

DO_CCSD = True
DO_CCSD_T = True      # add the perturbative (T) correction
FROZEN = 'auto'        # 'auto' -> mycc.set_frozen(); an int -> that many frozen
                       # core orbitals; 0 -> all-electron

DO_AFQMC = False
CHOL_CUT = 1e-5        # Cholesky decomposition threshold for the integrals
AFQMC_OPTIONS = {
    'eql_time': 40,
    'n_blocks': 1000,
    'n_walkers': 300,
    'max_memory': 100000,
    'seed': 17,
    'walker_type': 'rhf',
    'trial': 'pt2ccsd_bar',
}

MAX_MEMORY = 400000      # MB, for the pyscf part
VERBOSE = 4            # PySCF print level
WORKDIR = '../run_ccsd_t'     # one sub-directory per molecule is created here
SUMMARY = 'summary.out'         # written inside WORKDIR
SKIP_EXISTING = True   # skip molecules whose work directory already holds a
                       # finished result (lets you resume an interrupted batch)
# ----------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
HARTREE2KCAL = 627.5094740631


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


def run_ccsd(mf):
    """CCSD (optionally +(T)) on top of the stable DF-RHF reference."""
    mycc = cc.CCSD(mf)
    mycc.conv_tol = CC_ENE_TOL # 1e-6
    mycc.conv_tol_normt = CC_NORM_TOL # e-5
    mycc.max_memory = MAX_MEMORY
    if FROZEN == 'auto':
        mycc.set_frozen()
    elif FROZEN:
        mycc.frozen = FROZEN
    mycc.kernel()

    e_t = None
    if DO_CCSD_T:
        e_t = mycc.ccsd_t()
    return mycc, e_t


def run_afqmc(mycc):
    """Prepare the integrals and launch ph-AFQMC in the current directory."""
    from afqmc import integral, launch_afqmc

    integral.prep_integral(mycc, chol_cut=CHOL_CUT)
    launch_afqmc.ph_afqmc(dict(AFQMC_OPTIONS))
    return parse_afqmc_out('afqmc.out')


def parse_afqmc_out(path):
    """Pull the final energies out of an afqmc.out.

    Handles both the plain 'Energy (Ha)' line and the '[Guide]' / '[Trial]'
    lines printed by the pt2CCSD driver.  Returns a dict of
    {label: (energy, error)} with label in {'afqmc', 'guide', 'trial'}.
    """
    pattern = re.compile(r'^\s*(?:\[(\w+)\]\s+)?Energy \(Ha\)\s+'
                         r'(-?\d+\.\d+)\s*\+/-\s*(\d+\.\d+)')
    found = {}
    if not os.path.isfile(path):
        return found
    with open(path) as fh:
        for line in fh:
            m = pattern.match(line)
            if m:
                label = (m.group(1) or 'afqmc').lower()
                found[label] = (float(m.group(2)), float(m.group(3)))
    return found


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


def run_one(tag, geom_dir, workroot):
    """Run RHF -> CCSD -> AFQMC for one geometry; returns a result dict."""
    xyz = os.path.join(geom_dir, tag + '.xyz')
    if not os.path.isfile(xyz):
        raise IOError('geometry not found: %s' % xyz)

    rundir = os.path.join(workroot, tag)
    os.makedirs(rundir, exist_ok=True)

    res = {'tag': tag, 'status': 'running'}
    cwd = os.getcwd()
    os.chdir(rundir)          # AFQMC reads/writes in the current directory
    try:
        mol, comment, form = build_mol(xyz, tag, output=tag + '.out')
        res.update(formula=form, natm=mol.natm, nao=mol.nao_nr(),
                   nelec=mol.nelectron, charge=mol.charge)
        lib.logger.note(mol, 'Geometry : %s', xyz)
        lib.logger.note(mol, 'Comment  : %s', comment)
        lib.logger.note(mol, 'Atoms    : %d   charge: %+d   basis: %s   AOs: %d',
                        mol.natm, mol.charge, BASIS, mol.nao_nr())

        t0 = time.time()
        mf, stable, ncycle = run_stable_rhf(mol, chkfile=tag + '.chk')
        res.update(e_rhf=mf.e_tot, scf_converged=bool(mf.converged),
                   stable=bool(stable), stab_cycles=ncycle,
                   t_scf=time.time() - t0)

        if DO_CCSD:
            t0 = time.time()
            mycc, e_t = run_ccsd(mf)
            nfrz = mycc.frozen if isinstance(mycc.frozen, int) else \
                len(mycc.frozen or [])
            res.update(e_ccsd=mycc.e_tot, e_corr=mycc.e_corr,
                       cc_converged=bool(mycc.converged), frozen=nfrz,
                       t_cc=time.time() - t0)
            if e_t is not None:
                res['e_ccsd_t'] = mycc.e_tot + e_t

            if DO_AFQMC:
                t0 = time.time()
                energies = run_afqmc(mycc)
                res['t_qmc'] = time.time() - t0
                # the pt2CCSD driver reports [Guide] and [Trial]; the trial
                # (pt2-corrected) energy is the production number.
                for label in ('trial', 'afqmc', 'guide'):
                    if label in energies:
                        res['e_afqmc'], res['e_afqmc_err'] = energies[label]
                        res['afqmc_label'] = label
                        break
                else:
                    raise RuntimeError('no AFQMC energy found in afqmc.out')
                if 'guide' in energies:
                    res['e_guide'], res['e_guide_err'] = energies['guide']

        res['status'] = 'ok'
    except Exception as err:                       # keep the batch going
        res['status'] = 'FAILED: %s' % err
        with open('traceback.txt', 'w') as fh:
            fh.write(traceback.format_exc())
        print('%-16s FAILED: %s' % (tag, err), flush=True)
    finally:
        os.chdir(cwd)

    if res['status'] == 'ok':
        print('%-16s %-12s nao=%4d  E(RHF)=%18.10f  E(CCSD)=%18.10f  '
              'E(AFQMC)=%s'
              % (tag, res.get('formula', ''), res.get('nao', 0),
                 res.get('e_rhf', float('nan')),
                 res.get('e_ccsd', float('nan')),
                 ('%.5f +/- %.5f' % (res['e_afqmc'], res['e_afqmc_err']))
                 if 'e_afqmc' in res else 'n/a'), flush=True)
    return res


# ----------------------------------------------------------------------------
# summary
# ----------------------------------------------------------------------------

def _f(value, width=18, prec=10):
    return ('%*.*f' % (width, prec, value)) if value is not None else \
        '%*s' % (width, 'n/a')


def write_summary(results, path, geom_dir):
    """(Re)write the summary file from the results collected so far."""
    cols = ('molecule', 'formula', 'chg', 'nao', 'nel', 'frz', 'stab',
            'E(RHF)', 'E(CCSD)', 'E(CCSD(T))', 'E(AFQMC)', 'err',
            't_scf', 't_cc', 't_qmc')
    head = ('%-16s %-12s %4s %5s %5s %4s %5s %18s %18s %18s %16s %9s '
            '%9s %9s %9s' % cols)

    lines = []
    lines.append('=' * len(head))
    lines.append(' ISOL24 : DF-RHF / DF-CCSD / AFQMC summary')
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
    lines.append(' cc        : %s, conv_tol=%.1e, conv_tol_normt=%.1e   '
                 'frozen : %s'
                 % ('CCSD(T)' if DO_CCSD_T else 'CCSD', CC_ENE_TOL,
                    CC_NORM_TOL, FROZEN))
    if DO_AFQMC:
        opt = ', '.join('%s=%s' % kv for kv in sorted(AFQMC_OPTIONS.items()))
        lines.append(' afqmc     : chol_cut=%.1e, %s' % (CHOL_CUT, opt))
    lines.append(' energies in Hartree, times in seconds')
    lines.append('')
    lines.append(head)
    lines.append('-' * len(head))

    for r in results:
        if r['status'] != 'ok':
            lines.append('%-16s %-12s %s' % (r['tag'], r.get('formula', ''),
                                             r['status']))
            continue
        lines.append(
            '%-16s %-12s %+4d %5d %5d %4s %5s %s %s %s %s %s %9s %9s %9s'
            % (r['tag'], r.get('formula', ''), r.get('charge', 0),
               r.get('nao', 0), r.get('nelec', 0),
               r.get('frozen', '-'),
               'yes' if r.get('stable') else 'NO',
               _f(r.get('e_rhf')), _f(r.get('e_ccsd')), _f(r.get('e_ccsd_t')),
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
                 % ('reaction', 'formula', 'dE(RHF)', 'dE(CCSD)',
                    'dE(CCSD(T))', 'dE(AFQMC)', 'err'))
        lines.append('')
        lines.append(rhead)
        lines.append('-' * len(rhead))
        for idx, form, d in rxn:
            lines.append('%-10s %-12s %s %s %s %s %s'
                         % (idx, form,
                            _f(d.get('rhf'), 14, 4), _f(d.get('ccsd'), 14, 4),
                            _f(d.get('ccsd_t'), 14, 4),
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
        for key, name in (('e_rhf', 'rhf'), ('e_ccsd', 'ccsd'),
                          ('e_ccsd_t', 'ccsd_t'), ('e_afqmc', 'afqmc')):
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
        done = os.path.join(workroot, tag, 'afqmc.out' if DO_AFQMC
                            else tag + '.out')
        if SKIP_EXISTING and os.path.isfile(done):
            energies = parse_afqmc_out(done) if DO_AFQMC else {}
            if energies or not DO_AFQMC:
                print('[%d/%d] %-16s already done, skipping'
                      % (i, len(tags), tag), flush=True)
                continue

        print('\n[%d/%d] %s' % (i, len(tags), tag), flush=True)
        results.append(run_one(tag, geom_dir, workroot))
        write_summary(results, summary, geom_dir)   # keep it up to date

    write_summary(results, summary, geom_dir)
    print('\nSummary written to %s' % summary, flush=True)
    return results


if __name__ == '__main__':
    main()
