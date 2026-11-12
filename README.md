# Reproducible Data & Verification Suite: Q-BESC & Q-BECK

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24149207.svg)](https://doi.org/10.5281/zenodo.24149207)

**Title:** *Quantum Boltzmann Equation for Static Correlation with Cumulant Kernel Closure*  
**Author:** Romit Chakraborty (Point Reyes Sound, Inc.)  
**Corporate Copyright:** Copyright © 2026 Point Reyes Sound, Inc.  
**Zenodo DOI:** [`10.5281/zenodo.24149207`](https://doi.org/10.5281/zenodo.24149207)  
**GitHub Repository:** [`https://github.com/point-reyes-sound/qbeck-static-correlation-data`](https://github.com/point-reyes-sound/qbeck-static-correlation-data) (`v1.0.0`)  
**IP & Trade-Secret Notice:** The Q-BESC and Q-BECK solver kernels constitute proprietary trade secrets and patent-pending intellectual property of Point Reyes Sound, Inc. protected under Indian Convention Patent Application No. IN 202634135090 and U.S. Provisional Patent Application No. US 64/033,274.  
**Invariant Production Solver Parameters:** Markovian relaxation weight $w = \Delta t / \tau = 0.5$, susceptibility window width $\Delta_0 = 0.45\text{ }E_{\mathrm{h}}$, orthogonalized commutator stationarity threshold $\Vert [\mathbf{S}^{-1/2}\,{}^{1}\mathbf{F}_{\mathrm{eff}}\,\mathbf{S}^{-1/2}, \mathbf{S}^{1/2}\,{}^{1}\mathbf{P}\,\mathbf{S}^{1/2}] \Vert_F \le 10^{-9}\text{ }E_{\mathrm{h}}$ (strictly below $10^{-7}$), and `brentq` chemical-potential root-finding tolerance $10^{-12}\text{ }E_{\mathrm{h}}$.

---

## 1. Repository Layout

```text
qbeck-static-correlation-data/
├── .zenodo.json                           # Zenodo archival metadata (Point Reyes Sound, Inc.)
├── LICENSE                                # Corporate copyright & patent reservation (Point Reyes Sound, Inc.)
├── MANIFEST.sha256                        # Cryptographic SHA-256 digests of all datasets & OpenAPI schema
├── README.md                              # Documentation and reproduction guide
├── openapi.json                           # OpenAPI 3.1 schema for GET /v1/health, POST /v1/q-beck, POST /v1/q-besc
├── requirements.txt                       # Python dependencies (numpy, scipy, pyscf, httpx)
├── verify_reproducibility.py              # Unified standalone offline (PySCF) & live API (httpx) verifier
└── data/
    ├── converged_ao_1rdms.npz             # Converged AO 1-RDMs, S, T, H_core, and energies (H3, H4, BeH2, H2O, N2, C6H4)
    ├── h3_collinear_data.npz              # Collinear H3 stretch & Borland-Dennis audit (cc-pVDZ)
    ├── h3_mep_data.npz                    # H3 Jahn-Teller pseudorotation minimum-energy path (cc-pVDZ)
    ├── h3_mep_gpc_distances.{npz,json}    # H3 pure Borland-Dennis & ensemble Pauli polytope distances
    ├── h4_d4h_data.npz                    # H4 square D4h symmetric dissociation (cc-pVDZ)
    ├── h4_d4h_mott_transition.{npz,json}  # H4 configurational entropy & Mott fractionalization diagnostics
    ├── fig7a_beh2_data.npz                # BeH2 + H2 C2v insertion avoided crossing (6-31G)
    ├── beh2_c2v_avoided_crossing_diagnostics.{npz,json} # Symmetry-resolved 3a1/1b2 occupations & dE/dx
    ├── h2o_dissociation_data.npz          # Symmetric H2O double dissociation (cc-pVDZ)
    ├── h2o_dissociation_data_dense.npz    # Dense 73-point H2O double dissociation curve (cc-pVDZ)
    ├── h2o_dissociation_benchmark.json    # Complete H2O benchmark JSON table
    ├── n2_casci_registers_study.{npz,json}# N2 triple-bond dissociation across active spaces (cc-pVDZ)
    ├── n2_virial_defect.npz               # N2 quantum virial ratio -<V>/(2<T>) audit
    ├── n2_eq_table1_convergence.json      # N2 Dunning basis hierarchy (cc-pVDZ..aug-cc-pV5Z, Table 1)
    ├── n2_eq_virial_basis_convergence.{npz,json} # N2 basis-set virial convergence data
    ├── mr_mgn_be17_benchmark.json         # 11-molecule MR-MGN-BE17 strong-correlation benchmark
    └── benzyne_and_scaling_data.{npz,json}# Singlet o-, m-, p-C6H4 diradicals & 36-system O(N^3) scaling (N_AO<=300)
```

---

## 2. Offline PySCF Verification (`--mode offline`)

`verify_reproducibility.py --mode offline` runs completely offline using open-source `numpy`, `scipy`, and `pyscf` with zero proprietary solver dependencies. It validates:
1. **Cryptographic Integrity:** Verifies SHA-256 digests in `MANIFEST.sha256` and schema structure in `openapi.json`.
2. **AO 1-RDM Observable Recomputation (`< 10^{-9} E_h`):** Reconstructs exact 1-electron and 2-electron AO integrals via PySCF from `data/converged_ao_1rdms.npz` and recomputes:
   - Total electronic + nuclear energy $E_{\mathrm{tot}}$ ($|\Delta E| < 10^{-9}\text{ }E_{\mathrm{h}}$),
   - Quantum virial ratio $-\langle V\rangle / (2\langle T\rangle)$,
   - Configurational entropy $S_{\mathrm{config}} = -\sum_i [n_i \log_2 n_i + (1-n_i)\log_2(1-n_i)]$,
   - Generalized Pauli polytope distances ($\|\mathbf{n} - \mathbf{p}\|_2$, $\|\mathbf{n} - \mathbf{e}\|_2$) and Borland–Dennis slack $D_{\mathrm{BD}} = \max(0, n_5 + n_6 - n_4) \ge 0$ for $\mathrm{H}_3$.
3. **Full 8-Suite Manuscript Dataset Audit:** Verifies all trajectory curves, isomer splittings, `MR-MGN-BE17` MUEs, and $\mathcal{O}(N_{\mathrm{AO}}^3)$ scaling metrics across $N_{\mathrm{AO}} \in [15, 300]$.

```bash
python verify_reproducibility.py --mode offline
```

---

## 3. Live FastAPI Gateway Verification (`--mode api`)

`verify_reproducibility.py --mode api` queries a running FastAPI Quantum Boltzmann Gateway (`GET /v1/health`, `POST /v1/q-beck`, and `POST /v1/q-besc`, documented in `openapi.json`) over HTTP via `httpx` across all 8 manuscript benchmark suites:

```bash
python verify_reproducibility.py --mode api --url http://127.0.0.1:8000
```

Pass `--extended` to include the largest Dunning basis expansions on $\mathrm{N}_2$ (`aug-cc-pVQZ` through `aug-cc-pV5Z`, $N_{\mathrm{AO}} \le 254$) and hydrogen chains up to $\mathrm{H}_{60}$ ($N_{\mathrm{AO}} = 300$), or `--mode both` to execute both offline and live API audits sequentially.

---

## 4. Citation

```bibtex
@dataset{chakraborty2026qbeck_data,
  author       = {Chakraborty, Romit},
  title        = {{Reproducible Data and Verification Suite for: Quantum Boltzmann Equation for Static Correlation with Cumulant Kernel Closure}},
  year         = {2026},
  publisher    = {Point Reyes Sound, Inc. / Zenodo},
  version      = {v1.0.0},
  doi          = {10.5281/zenodo.24149207},
  url          = {https://doi.org/10.5281/zenodo.24149207}
}
```
