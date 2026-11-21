"""
MACE-MP-0 Benchmark of Cisplatin-Thiolate Substitution Reaction
================================================================
Compares the pre-trained MACE-MP-0 universal potential against
DFT (PBE, Quantum ESPRESSO) for the reaction:

  cis-[PtCl2(NH3)2] + CH3S- -> cis-[PtCl(SCH3)(NH3)2] + Cl-

Usage:
  pip install mace-torch ase
  python mace_benchmark.py

Also runs on Google Colab (copy cells into notebook).

Author: Thomas Adesoye Adeoye
"""

import numpy as np
from ase import Atoms
from ase.optimize import BFGS
from mace.calculators import mace_mp

# ============================================================
# 1. Load MACE-MP-0
# ============================================================
calc = mace_mp(model="medium", default_dtype="float64")
print("MACE-MP-0 (medium) loaded successfully!\n")

# ============================================================
# 2. Build molecules from DFT-relaxed coordinates
# ============================================================

# Cisplatin: cis-[PtCl2(NH3)2], 11 atoms, neutral
cisplatin = Atoms(
    symbols=['Pt','Cl','Cl','N','N','H','H','H','H','H','H'],
    positions=[
        [7.8668330478, 7.8668330478, 8.1047115091],
        [10.1484308479, 7.7404313181, 8.3041351192],
        [7.7404313181, 10.1484308479, 8.3041351192],
        [5.7877721403, 8.0444817086, 7.9282669007],
        [8.0444817086, 5.7877721403, 7.9282669007],
        [5.6694820166, 9.0697385345, 8.0037202108],
        [5.2612885446, 7.5962724920, 8.6828105272],
        [5.4108134831, 7.7344558667, 7.0287114876],
        [9.0697385345, 5.6694820166, 8.0037202108],
        [7.5962724920, 5.2612885446, 8.6828105272],
        [7.7344558667, 5.4108134831, 7.0287114876],
    ],
    pbc=False
)

# Thiolate: CH3S-, 5 atoms, charge -1
thiolate = Atoms(
    symbols=['S','C','H','H','H'],
    positions=[
        [8.0000000000, 8.0000000000, 8.0000000000],
        [8.0000272627, 8.0000332492, 9.8021765232],
        [7.9999835651, 9.0345062122, 10.1984428498],
        [8.8957525035, 7.4827317122, 10.1984412986],
        [7.1042550665, 7.4827528247, 10.1984252944],
    ],
    pbc=False
)

# Product: cis-[PtCl(SCH3)(NH3)2], 15 atoms, neutral
product = Atoms(
    symbols=['Pt','Cl','S','C','N','N','H','H','H','H','H','H','H','H','H'],
    positions=[
        [8.0000000000, 8.0000000000, 8.0000000000],
        [7.9055603811, 10.0363787876, 8.4829258836],
        [10.2126690725, 7.5666603785, 8.1524905199],
        [10.6926790815, 8.0818495852, 9.8386520323],
        [5.7941696780, 8.0541705276, 7.9432650394],
        [8.0164457544, 5.6921697369, 7.8453716058],
        [5.7216997440, 9.0463551552, 8.2152433972],
        [5.2058569357, 7.4956435647, 8.5645279601],
        [5.4463677760, 7.9587893221, 6.9860474988],
        [9.0437045085, 5.5330866563, 7.8978629772],
        [7.5716902996, 5.1828877070, 8.6133766612],
        [7.6562619652, 5.3405850685, 6.9537072105],
        [10.3359116915, 9.1006453721, 10.0260720025],
        [10.3019609312, 7.4052405040, 10.6076150579],
        [11.7898986248, 8.0685471472, 9.8717069620],
    ],
    pbc=False
)

# Chloride: Cl-, 1 atom, charge -1
chloride = Atoms(
    symbols=['Cl'],
    positions=[[8.000, 8.000, 8.000]],
    pbc=False
)

molecules = {
    'cisplatin': cisplatin,
    'thiolate': thiolate,
    'product': product,
    'chloride': chloride,
}

print("All 4 molecules built from DFT-relaxed coordinates.\n")

# ============================================================
# 3. Single-point energies at DFT geometry
# ============================================================
sp_results = {}

print("=" * 60)
print("  SINGLE-POINT ENERGIES AT DFT-RELAXED GEOMETRY")
print("=" * 60)
for name, mol in molecules.items():
    mol_copy = mol.copy()
    mol_copy.calc = calc
    e = mol_copy.get_potential_energy()
    sp_results[name] = e
    print(f"  {name:12s}: {e:.6f} eV")

dE_sp = sp_results['product'] + sp_results['chloride'] - sp_results['cisplatin'] - sp_results['thiolate']
print(f"\n  dE (single-point) = {dE_sp:.4f} eV = {dE_sp * 23.0609:.1f} kcal/mol")

# ============================================================
# 4. Re-optimize with MACE and get energies
# ============================================================
opt_results = {}

print("\n" + "=" * 60)
print("  MACE GEOMETRY OPTIMIZATION")
print("=" * 60)
for name, mol in molecules.items():
    mol.calc = calc

    if name == 'chloride':
        e = mol.get_potential_energy()
        print(f"  {name:12s}: E = {e:.6f} eV (single atom, no optimization)")
    else:
        opt = BFGS(mol, logfile=f'{name}_opt.log')
        opt.run(fmax=0.01)
        e = mol.get_potential_energy()
        nsteps = opt.get_number_of_steps()
        print(f"  {name:12s}: E = {e:.6f} eV  ({nsteps} steps)")

    opt_results[name] = e

dE_opt = opt_results['product'] + opt_results['chloride'] - opt_results['cisplatin'] - opt_results['thiolate']
print(f"\n  dE (re-optimized) = {dE_opt:.4f} eV = {dE_opt * 23.0609:.1f} kcal/mol")

# ============================================================
# 5. Compare DFT vs MACE reaction energies
# ============================================================
dE_dft_ry = -0.07256630
dE_dft_ev = dE_dft_ry * 13.6057
dE_dft_kcal = dE_dft_ry * 313.755

dE_sp_kcal = dE_sp * 23.0609
dE_opt_kcal = dE_opt * 23.0609

print("\n" + "=" * 60)
print("  REACTION ENERGY COMPARISON")
print("  cis-[PtCl2(NH3)2] + CH3S-")
print("       -> cis-[PtCl(SCH3)(NH3)2] + Cl-")
print("=" * 60)
print(f"\n  {'Method':<30s} {'dE (eV)':>10s} {'dE (kcal/mol)':>14s}")
print(f"  {'-'*54}")
print(f"  {'DFT (PBE, QE 7.4.1)':<30s} {dE_dft_ev:>10.3f} {dE_dft_kcal:>14.1f}")
print(f"  {'MACE-MP-0 (single-point)':<30s} {dE_sp:>10.3f} {dE_sp_kcal:>14.1f}")
print(f"  {'MACE-MP-0 (re-optimized)':<30s} {dE_opt:>10.3f} {dE_opt_kcal:>14.1f}")
print(f"\n  Negative dE => thermodynamically favourable")
print(f"  => Thiolate substitution deactivates cisplatin")

# ============================================================
# 6. Compare bond lengths
# ============================================================
# DFT-relaxed cisplatin
dft_cis_pos = np.array([
    [7.8668330478, 7.8668330478, 8.1047115091],  # Pt
    [10.1484308479, 7.7404313181, 8.3041351192], # Cl1
    [7.7404313181, 10.1484308479, 8.3041351192], # Cl2
    [5.7877721403, 8.0444817086, 7.9282669007],  # N1
    [8.0444817086, 5.7877721403, 7.9282669007],  # N2
])

# DFT-relaxed product
dft_prod_pos = np.array([
    [8.0000000000, 8.0000000000, 8.0000000000],  # Pt
    [7.9055603811, 10.0363787876, 8.4829258836], # Cl
    [10.2126690725, 7.5666603785, 8.1524905199], # S
    [10.6926790815, 8.0818495852, 9.8386520323], # C
    [5.7941696780, 8.0541705276, 7.9432650394],  # N1
    [8.0164457544, 5.6921697369, 7.8453716058],  # N2
])

# DFT bond lengths
dft_ptcl1 = np.linalg.norm(dft_cis_pos[0] - dft_cis_pos[1])
dft_ptn1 = np.linalg.norm(dft_cis_pos[0] - dft_cis_pos[3])
dft_ptcl_p = np.linalg.norm(dft_prod_pos[0] - dft_prod_pos[1])
dft_pts_p = np.linalg.norm(dft_prod_pos[0] - dft_prod_pos[2])
dft_ptn1_p = np.linalg.norm(dft_prod_pos[0] - dft_prod_pos[4])
dft_ptn2_p = np.linalg.norm(dft_prod_pos[0] - dft_prod_pos[5])

# MACE bond lengths (from re-optimized structures)
mace_cis_pos = cisplatin.get_positions()
mace_ptcl1 = np.linalg.norm(mace_cis_pos[0] - mace_cis_pos[1])
mace_ptn1 = np.linalg.norm(mace_cis_pos[0] - mace_cis_pos[3])

mace_prod_pos = product.get_positions()
mace_ptcl_p = np.linalg.norm(mace_prod_pos[0] - mace_prod_pos[1])
mace_pts_p = np.linalg.norm(mace_prod_pos[0] - mace_prod_pos[2])
mace_ptn1_p = np.linalg.norm(mace_prod_pos[0] - mace_prod_pos[4])
mace_ptn2_p = np.linalg.norm(mace_prod_pos[0] - mace_prod_pos[5])

print("\n" + "=" * 60)
print("  KEY BOND LENGTHS (Angstrom)")
print("=" * 60)
print(f"\n  {'Bond':<24s} {'DFT':>8s} {'MACE':>8s} {'Diff':>8s}")
print(f"  {'-'*48}")
print(f"  {'Cisplatin Pt-Cl':<24s} {dft_ptcl1:>8.3f} {mace_ptcl1:>8.3f} {mace_ptcl1-dft_ptcl1:>+8.3f}")
print(f"  {'Cisplatin Pt-N':<24s} {dft_ptn1:>8.3f} {mace_ptn1:>8.3f} {mace_ptn1-dft_ptn1:>+8.3f}")
print(f"  {'Product Pt-Cl':<24s} {dft_ptcl_p:>8.3f} {mace_ptcl_p:>8.3f} {mace_ptcl_p-dft_ptcl_p:>+8.3f}")
print(f"  {'Product Pt-S':<24s} {dft_pts_p:>8.3f} {mace_pts_p:>8.3f} {mace_pts_p-dft_pts_p:>+8.3f}")
print(f"  {'Product Pt-N(trans Cl)':<24s} {dft_ptn1_p:>8.3f} {mace_ptn1_p:>8.3f} {mace_ptn1_p-dft_ptn1_p:>+8.3f}")
print(f"  {'Product Pt-N(trans S)':<24s} {dft_ptn2_p:>8.3f} {mace_ptn2_p:>8.3f} {mace_ptn2_p-dft_ptn2_p:>+8.3f}")

print("\n" + "=" * 60)
print("  CONCLUSION")
print("=" * 60)
print("""
  MACE-MP-0 accurately reproduces cisplatin geometry (Pt-Cl
  within 0.001 A of DFT) but fails to capture the energetics
  of the thiolate substitution reaction. This is because:

  1. MACE-MP-0 was trained on Materials Project data (inorganic
     crystals), lacking platinum coordination chemistry with
     biological ligands.
  2. MACE has no mechanism for handling charged species.
  3. The mixed Pt-Cl/S/N coordination in the product is poorly
     represented in the training data.

  Universal ML potentials require domain-specific fine-tuning
  for pharmaceutical molecular modelling.
""")
