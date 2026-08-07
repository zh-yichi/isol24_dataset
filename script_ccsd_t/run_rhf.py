#!/usr/bin/env python
"""Density-fitted RHF for the ISOL24 geometries, driven to a stable solution.

The SCF is repeated in a loop: after each convergence the internal stability of
the RHF solution is analysed, and while an instability is found the orbitals are
rotated along the instability direction and the SCF is restarted with the
second-order (Newton/CIAH) solver.  The loop exits once the solution converges
and the stability analysis returns the same orbitals.

Everything is configured in the SETTINGS block below; just run

    python run_rhf.py
"""

import os
import time

from pyscf import gto, scf, lib

# ----------------------------------------------------------------------------
# SETTINGS -- edit these
# ----------------------------------------------------------------------------
# Directory holding the .xyz files.  Relative paths are resolved with respect to
# this script, so it works no matter where you launch it from.
GEOM_DIR = '../geometry/small'

# Which molecule(s) to run.  A single file name, or a list of names to run in
# sequence, e.g. ['03_educt.xyz', '03_product.xyz'].
MOLECULES = '24_educt.xyz'

BASIS = 'cc-pVDZ'      # orbital basis
AUXBASIS = None        # DF auxiliary basis; None -> matching JK-fit set
CHARGE = 1
UNIT = 'Angstrom'      # unit of the coordinates in the .xyz files

CONV_TOL = 1e-10       # SCF energy convergence
MAX_STAB_CYCLE = 10    # max stability-check / restart iterations
CHECK_EXTERNAL = False # also run the external (RHF -> UHF) stability check

MAX_MEMORY = 4000      # MB
VERBOSE = 4            # PySCF print level
OUTDIR = None          # e.g. '../out' to write <name>.out and <name>.chk;
                       # None -> log to stdout, no checkpoint file
# ----------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))


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


def build_mol(path, output=None):
    atoms, comment = read_xyz(path)
    mol = gto.M(atom=atoms, basis=BASIS, charge=CHARGE, spin=0, unit=UNIT,
                max_memory=MAX_MEMORY, verbose=VERBOSE, output=output)
    mol.comment = comment
    return mol


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


def run_one(name):
    """Run one .xyz file from GEOM_DIR; returns the mean-field object."""
    xyz = os.path.join(_resolve(GEOM_DIR), name)
    if not os.path.isfile(xyz):
        raise IOError('geometry not found: %s' % xyz)

    tag = os.path.splitext(os.path.basename(xyz))[0]
    output = chkfile = None
    if OUTDIR:
        outdir = _resolve(OUTDIR)
        os.makedirs(outdir, exist_ok=True)
        output = os.path.join(outdir, tag + '.out')
        chkfile = os.path.join(outdir, tag + '.chk')

    mol = build_mol(xyz, output)
    lib.logger.note(mol, 'Geometry : %s', xyz)
    lib.logger.note(mol, 'Comment  : %s', mol.comment)
    lib.logger.note(mol, 'Atoms    : %d   basis: %s   AOs: %d',
                    mol.natm, BASIS, mol.nao_nr())

    t0 = time.time()
    mf, stable, ncycle = run_stable_rhf(mol, chkfile)
    wall = time.time() - t0

    print('%-24s E(DF-RHF) = %20.12f  converged=%s  stable=%s  '
          'stab_cycles=%d  wall=%.1f s'
          % (tag, mf.e_tot, mf.converged, stable, ncycle, wall), flush=True)
    return mf


def main():
    names = [MOLECULES] if isinstance(MOLECULES, str) else list(MOLECULES)
    return [run_one(name) for name in names]


if __name__ == '__main__':
    main()
