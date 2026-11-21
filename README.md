# DFT and ML Study of Cisplatin-Thiolate Substitution Reaction

A computational study of the thiolate ligand exchange reaction in cisplatin using Density Functional Theory (DFT) and machine-learning interatomic potentials (MACE-MP-0). This work investigates the thermodynamic driving force behind glutathione-mediated cisplatin deactivation — a key mechanism of chemotherapy drug resistance.

## Reaction Studied

```
cis-[PtCl₂(NH₃)₂] + CH₃S⁻ → cis-[PtCl(SCH₃)(NH₃)₂] + Cl⁻
     (cisplatin)    (thiolate)       (product)         (chloride)
```

Methanethiolate (CH₃S⁻, 5 atoms) is used as a computationally tractable model for glutathione, which is the biological thiol responsible for cisplatin deactivation in cancer cells.

## Methods

### DFT Calculations
- **Code:** Quantum ESPRESSO 7.4.1
- **Functional:** PBE (Perdew-Burke-Ernzerhof)
- **Pseudopotentials:** SSSP Efficiency (PSLibrary RRKJUS, PBE)
- **Kinetic energy cutoff:** 40 Ry (wavefunction), 320 Ry (charge density)
- **Box size:** 30 bohr (~15.9 Å) cubic cell with Martyna-Tuckerman isolated boundary correction
- **K-points:** Gamma-point only
- **Charged species:** `tot_charge = -1.0` for thiolate and chloride
- **Relaxation:** BFGS ionic relaxation (`calculation = 'relax'`); SCF only for chloride

### ML Benchmark
- **Model:** MACE-MP-0 (medium), a universal machine-learning interatomic potential pre-trained on ~150k structures from the Materials Project
- **Framework:** ASE + mace-torch
- **Purpose:** Benchmark the applicability of universal ML potentials to platinum coordination chemistry

## Results

### Reaction Energy

| Method | ΔE (eV) | ΔE (kcal/mol) |
|--------|---------|----------------|
| DFT (PBE) | −0.987 | −22.8 |
| MACE-MP-0 (single-point at DFT geometry) | +0.572 | +13.2 |
| MACE-MP-0 (re-optimized) | +0.030 | +0.7 |

The DFT result of **−22.8 kcal/mol** confirms that the thiolate substitution reaction is **thermodynamically favourable**, consistent with the known mechanism of glutathione-mediated cisplatin resistance.

### Bond Length Comparison (Å)

| Bond | DFT | MACE-MP-0 | Difference |
|------|-----|-----------|------------|
| Cisplatin Pt–Cl | 2.294 | 2.293 | −0.001 |
| Cisplatin Pt–N | 2.094 | 2.075 | −0.020 |
| Product Pt–Cl | 2.095 | 2.316 | +0.221 |
| Product Pt–S | 2.260 | 2.226 | −0.034 |
| Product Pt–N (trans Cl) | 2.207 | 2.096 | −0.111 |
| Product Pt–N (trans S) | 2.313 | 2.109 | −0.204 |

### Key Findings

1. **DFT confirms drug resistance mechanism:** The exothermic reaction energy (−22.8 kcal/mol) provides a quantitative basis for understanding why sulphur-containing biomolecules like glutathione readily displace chloride from cisplatin, deactivating the drug before it can bind to DNA.

2. **MACE-MP-0 fails for this chemistry:** While MACE accurately reproduces cisplatin geometry (Pt–Cl within 0.001 Å of DFT), it predicts a near-thermoneutral reaction (+0.7 kcal/mol) instead of the strongly exothermic DFT value. This arises because:
   - MACE-MP-0 was trained on Materials Project data (inorganic crystals), lacking platinum coordination chemistry with biological ligands
   - MACE has no mechanism for handling charged species (thiolate anion, chloride anion)
   - The mixed Pt–Cl/S/N coordination environment in the product is poorly represented in the training data

3. **Implication:** Universal ML potentials require domain-specific fine-tuning or retraining for pharmaceutical molecular modelling — highlighting the need for specialised computational approaches at the interface of physics, chemistry, and pharmacology.

## Repository Structure

```
├── README.md
├── dft/
│   ├── cisplatin.in          # QE input: cisplatin relaxation
│   ├── thiolate.in           # QE input: thiolate relaxation (charge -1)
│   ├── product.in            # QE input: product relaxation
│   ├── chloride.in           # QE input: chloride SCF (charge -1)
│   ├── run_cisplatin.sh      # Job script for cisplatin
│   ├── run_thiolate.sh       # Job script for thiolate
│   ├── run_product.sh        # Job script for product
│   └── run_chloride.sh       # Job script for chloride
├── mace/
│   └── mace_benchmark.py     # MACE-MP-0 benchmark script (Colab-ready)
└── analysis/
    └── compute_reaction_energy.py  # Energy extraction and comparison
```

## How to Run

### DFT (on HPC cluster with Quantum ESPRESSO)
```bash
cd dft/
# Download pseudopotentials to pseudo/ directory
# Submit jobs in order:
qsub run_chloride.sh    # ~1 min
qsub run_thiolate.sh    # ~10 min
qsub run_cisplatin.sh   # ~15 min
qsub run_product.sh     # ~30 min
```

### MACE Benchmark (Google Colab or local Python)
```bash
pip install mace-torch ase
cd mace/
python mace_benchmark.py
```

## Computational Resources

- **HPC:** ICTP-EAIFR twas1 cluster (CentOS 7, Grid Engine)
- **Cores:** 16 MPI processes per calculation
- **Software:** Quantum ESPRESSO 7.4.1, MACE-MP-0 (via mace-torch), ASE

## Author

**Thomas Adesoye Adeoye**
B.Sc. Physics (First Class Honours, 3.97/4.00)
University of Ibadan, Nigeria

## References

1. Rabik, C. A., & Dolan, M. E. (2007). Molecular mechanisms of resistance and toxicity associated with platinating agents. *Cancer Treatment Reviews*, 33(1), 9–23.
2. Batatia, I., et al. (2024). A foundation model for atomistic simulation: MACE-MP-0. *arXiv:2401.00096*.
3. Giannozzi, P., et al. (2009). QUANTUM ESPRESSO: a modular and open-source software project for quantum simulations of materials. *Journal of Physics: Condensed Matter*, 21(39), 395502.
