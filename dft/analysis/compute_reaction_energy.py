"""
Compute Reaction Energy from Quantum ESPRESSO Output Files
==========================================================
Extracts total energies from QE pw.x output files and computes
the reaction energy for the cisplatin-thiolate substitution:

  cis-[PtCl2(NH3)2] + CH3S- -> cis-[PtCl(SCH3)(NH3)2] + Cl-

Usage:
  python compute_reaction_energy.py [output_dir]

  output_dir: directory containing *.out files (default: ../dft/)

Author: Thomas Adesoye Adeoye
"""

import sys
import os
import re

# ============================================================
# Configuration
# ============================================================
# Expected output files
OUTPUT_FILES = {
    'cisplatin': 'cisplatin.out',
    'thiolate':  'thiolate.out',
    'product':   'product.out',
    'chloride':  'chloride.out',
}

# Conversion factors
RY_TO_EV   = 13.6057
RY_TO_KCAL = 313.755
EV_TO_KCAL = 23.0609

# ============================================================
# Functions
# ============================================================
def extract_total_energy(filepath):
    """
    Extract the final total energy (Ry) from a QE pw.x output file.
    Looks for the last occurrence of '!    total energy'.
    """
    energy = None
    pattern = re.compile(r'!\s+total energy\s+=\s+([-\d.]+)\s+Ry')

    with open(filepath, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                energy = float(match.group(1))

    if energy is None:
        raise ValueError(f"Could not find total energy in {filepath}")

    return energy


def extract_relaxed_coordinates(filepath):
    """
    Extract the last set of ATOMIC_POSITIONS from a QE output file.
    Returns a list of (symbol, x, y, z) tuples.
    """
    coords = []
    current_block = []
    in_block = False

    with open(filepath, 'r') as f:
        for line in f:
            if 'ATOMIC_POSITIONS' in line:
                in_block = True
                current_block = []
                continue
            if in_block:
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        sym = parts[0]
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                        current_block.append((sym, x, y, z))
                    except (ValueError, IndexError):
                        in_block = False
                else:
                    in_block = False

    return current_block if current_block else coords


def compute_bond_length(pos1, pos2):
    """Euclidean distance between two 3D points."""
    return sum((a - b)**2 for a, b in zip(pos1, pos2)) ** 0.5


# ============================================================
# Main
# ============================================================
def main():
    # Determine output directory
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, '..', 'dft')

    print("=" * 60)
    print("  CISPLATIN-THIOLATE REACTION ENERGY ANALYSIS")
    print("  DFT (PBE) — Quantum ESPRESSO 7.4.1")
    print("=" * 60)
    print(f"\n  Output directory: {os.path.abspath(output_dir)}\n")

    # ----------------------------------------------------------
    # Extract energies
    # ----------------------------------------------------------
    energies = {}
    for name, filename in OUTPUT_FILES.items():
        filepath = os.path.join(output_dir, filename)
        if not os.path.exists(filepath):
            print(f"  WARNING: {filepath} not found — skipping.")
            continue
        e = extract_total_energy(filepath)
        energies[name] = e
        print(f"  E({name:12s}) = {e:.8f} Ry")

    if len(energies) < 4:
        print("\n  ERROR: Need all 4 output files to compute reaction energy.")
        print("  Missing:", [n for n in OUTPUT_FILES if n not in energies])
        sys.exit(1)

    # ----------------------------------------------------------
    # Compute reaction energy
    # ----------------------------------------------------------
    dE_ry = (energies['product'] + energies['chloride']
           - energies['cisplatin'] - energies['thiolate'])

    dE_ev   = dE_ry * RY_TO_EV
    dE_kcal = dE_ry * RY_TO_KCAL

    print(f"\n  Reaction: cis-[PtCl2(NH3)2] + CH3S-")
    print(f"         -> cis-[PtCl(SCH3)(NH3)2] + Cl-")
    print(f"\n  dE = {dE_ry:+.8f} Ry")
    print(f"     = {dE_ev:+.3f} eV")
    print(f"     = {dE_kcal:+.1f} kcal/mol")

    if dE_kcal < 0:
        print(f"\n  => Reaction is THERMODYNAMICALLY FAVOURABLE")
        print(f"  => Thiolate substitution deactivates cisplatin")
    else:
        print(f"\n  => Reaction is THERMODYNAMICALLY UNFAVOURABLE")

    # ----------------------------------------------------------
    # Extract and report bond lengths (if relax outputs exist)
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("  KEY BOND LENGTHS (Angstrom)")
    print("=" * 60)

    cis_path = os.path.join(output_dir, 'cisplatin.out')
    prod_path = os.path.join(output_dir, 'product.out')

    if os.path.exists(cis_path):
        cis_coords = extract_relaxed_coordinates(cis_path)
        if cis_coords:
            # Cisplatin: Pt(0), Cl(1), Cl(2), N(3), N(4), H...
            pt = cis_coords[0][1:]
            cl1 = cis_coords[1][1:]
            cl2 = cis_coords[2][1:]
            n1 = cis_coords[3][1:]
            n2 = cis_coords[4][1:]

            print(f"\n  Cisplatin:")
            print(f"    Pt-Cl(1):  {compute_bond_length(pt, cl1):.3f}")
            print(f"    Pt-Cl(2):  {compute_bond_length(pt, cl2):.3f}")
            print(f"    Pt-N(1):   {compute_bond_length(pt, n1):.3f}")
            print(f"    Pt-N(2):   {compute_bond_length(pt, n2):.3f}")

    if os.path.exists(prod_path):
        prod_coords = extract_relaxed_coordinates(prod_path)
        if prod_coords:
            # Product: Pt(0), Cl(1), S(2), C(3), N(4), N(5), H...
            pt = prod_coords[0][1:]
            cl = prod_coords[1][1:]
            s  = prod_coords[2][1:]
            c  = prod_coords[3][1:]
            n1 = prod_coords[4][1:]
            n2 = prod_coords[5][1:]

            print(f"\n  Product:")
            print(f"    Pt-Cl:     {compute_bond_length(pt, cl):.3f}")
            print(f"    Pt-S:      {compute_bond_length(pt, s):.3f}")
            print(f"    S-C:       {compute_bond_length(s, c):.3f}")
            print(f"    Pt-N(1):   {compute_bond_length(pt, n1):.3f}")
            print(f"    Pt-N(2):   {compute_bond_length(pt, n2):.3f}")

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"""
  Method:      DFT (PBE functional)
  Code:        Quantum ESPRESSO 7.4.1
  Pseudopots:  SSSP Efficiency (PSLibrary RRKJUS, PBE)
  Basis:       Plane waves, ecutwfc/ecutrho per SSSP recommendations
  Boundary:    30 bohr cubic box, Martyna-Tuckerman correction
  k-points:    Gamma only

  Reaction energy: {dE_kcal:+.1f} kcal/mol ({dE_ev:+.3f} eV)
  Interpretation:  Thiolate readily substitutes a chloride ligand
                   in cisplatin, consistent with the known drug
                   resistance mechanism via glutathione binding.
""")


if __name__ == '__main__':
    main()
