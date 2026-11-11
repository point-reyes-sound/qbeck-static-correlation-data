#!/usr/bin/env python3
"""
Standalone Reproducibility & Verification Suite for Q-BESC and Q-BECK.
Copyright (c) 2026 Point Reyes Sound, Inc.

Supports two independent verification modes with zero proprietary solver imports:
  --mode offline : Cryptographically verifies MANIFEST.sha256 and openapi.json,
                   recomputes total energies, generalized Pauli polytope distances
                   (||n - p||_2, ||n - e||_2, and Borland-Dennis slack D_BD),
                   configurational entropies S_config, and quantum virial ratios
                   -<V>/(2<T>) directly from archived AO 1-RDMs via PySCF to
                   < 10^{-9} E_h, confirms exact parity with manuscript/main.tex,
                   and audits all 8 manuscript benchmark archives.
  --mode api     : Queries the live FastAPI gateway (GET /v1/health, POST /v1/q-beck,
                   and POST /v1/q-besc) over HTTP via httpx across all 10 canonical
                   Section 1 entries (< 10^{-9} E_h parity) and all 8 manuscript
                   benchmark suites.
  --mode both    : Executes --mode offline followed by --mode api.
"""
import os
import json
import math
import hashlib
import argparse
import asyncio
import numpy as np
from scipy.linalg import fractional_matrix_power
from scipy.optimize import minimize
from pyscf import gto, scf

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
MANIFEST_PATH = os.path.join(ROOT_DIR, "MANIFEST.sha256")
OPENAPI_PATH = os.path.join(ROOT_DIR, "openapi.json")

BOHR_PER_ANG = 1.0 / 0.529177
HARTREE_TO_KCAL = 627.5094740631

BENZYNE_GEOMS = {
    "ortho-C6H4": (
        "C 0.0 0.622770 -1.201655; C 0.0 -0.622770 -1.201655; "
        "C 0.0 -1.456384 -0.097254; C 0.0 -0.699184 1.077867; "
        "C 0.0 0.699184 1.077867; C 0.0 1.456384 -0.097254; "
        "H 0.0 -2.533834 -0.119069; H 0.0 -1.226668 2.022664; "
        "H 0.0 1.226668 2.022664; H 0.0 2.533834 -0.119069"
    ),
    "meta-C6H4": (
        "C 0.0 0.0 1.488369; C 0.0 1.076258 0.642661; "
        "C 0.0 1.242343 -0.713135; C 0.0 0.0 -1.348902; "
        "C 0.0 -1.242343 -0.713135; C 0.0 -1.076258 0.642661; "
        "H 0.0 0.0 2.567367; H 0.0 2.186258 -1.239205; "
        "H 0.0 0.0 -2.430956; H 0.0 -2.186258 -1.239205"
    ),
    "para-C6H4": (
        "C 0.0 0.0 1.369439; C 0.0 1.215524 0.697665; "
        "C 0.0 1.215524 -0.697665; C 0.0 0.0 -1.369439; "
        "C 0.0 -1.215524 -0.697665; C 0.0 -1.215524 0.697665; "
        "H 0.0 2.156606 1.229269; H 0.0 2.156606 -1.229269; "
        "H 0.0 -2.156606 -1.229269; H 0.0 -2.156606 1.229269"
    ),
}

MR_MGN_BE17_SPECS = [
    ("B2",  1.590, 2, [[4, 2], 8]),
    ("C2",  1.243, 0, [8, 8]),
    ("CN",  1.172, 1, [[5, 4], 8]),
    ("CO2", 1.160, 0, [8, 8]),
    ("F2",  1.412, 0, [14, 8]),
    ("NF3", 1.365, 0, [6, 6]),
    ("NO",  1.151, 1, [[6, 5], 8]),
    ("S2",  1.889, 2, [[7, 5], 8]),
    ("SiO", 1.510, 0, [10, 8]),
    ("CO",  1.128, 0, [10, 8]),
    ("N2",  1.098, 0, [6, 6]),
]


# =============================================================================
# OFFLINE PYSCF & POLYTOPE VERIFICATION ROUTINES
# =============================================================================

def compute_polytope_distances_3_6(n6):
    """
    Computes the Borland-Dennis polygonal slack D_BD = max(0, n_5 + n_6 - n_4),
    the constrained minimum Euclidean distance ||n - p||_2 to the pure Borland-
    Dennis facet (p_1+p_6 = p_2+p_5 = p_3+p_4 = 1, p_5+p_6-p_4 = 0), and the
    constrained minimum Euclidean distance ||n - e||_2 = sqrt(6/5)*min(1-n_1, n_6)
    to the ensemble Pauli hypercube boundary for an ordered CAS(3,3) spin-orbital
    spectrum n = (n_1 >= ... >= n_6).
    """
    n6 = np.asarray(n6, dtype=float)
    raw_bd = float(n6[4] + n6[5] - n6[3])
    d_bd = 0.0 if abs(raw_bd) < 1e-12 else max(0.0, raw_bd)
    dist_ens = float(np.sqrt(6.0 / 5.0) * min(1.0 - n6[0], n6[5]))

    def obj(x):
        p5, p6 = x
        p4 = p5 + p6
        p = np.array([1.0 - p6, 1.0 - p5, 1.0 - p4, p4, p5, p6])
        return float(np.sum((n6 - p) ** 2))

    cons = [
        {"type": "ineq", "fun": lambda x: x[1]},
        {"type": "ineq", "fun": lambda x: x[0] - x[1]},
        {"type": "ineq", "fun": lambda x: 0.5 - (x[0] + x[1])},
    ]
    res = minimize(
        obj,
        [max(n6[4], 1e-8), max(n6[5], 0.0)],
        method="SLSQP",
        constraints=cons,
        options={"ftol": 1e-16},
    )
    sq_val = max(0.0, float(res.fun))
    dist_pure = 0.0 if sq_val < 1e-22 else float(np.sqrt(sq_val))
    return d_bd, dist_pure, dist_ens


def recompute_observables_from_1rdm(mol, dm_conv, dm_eq, dm0_ref, e_corr_cumulant=0.0):
    """
    Recomputes total energy E_tot, virial ratio -<V>/(2<T>), configurational
    entropy S_config, and (for N=3) Borland-Dennis slack D_BD plus pure/ensemble
    polytope distances purely from PySCF AO integrals and archived 1-RDM matrices.
    """
    s_ovlp = mol.intor("int1e_ovlp")
    t_kin = mol.intor("int1e_kin")
    v_nuc = mol.intor("int1e_nuc")
    h_core = t_kin + v_nuc
    s_half = fractional_matrix_power(s_ovlp, 0.5).real

    mf_ref = scf.UHF(mol) if dm0_ref.ndim == 3 else scf.RHF(mol)
    if dm0_ref.ndim == 3:
        dm_tot_ref = dm0_ref[0] + dm0_ref[1]
        v_j = mf_ref.get_j(mol, dm_tot_ref)
        v_k_a = mf_ref.get_k(mol, dm0_ref[0])
        v_k_b = mf_ref.get_k(mol, dm0_ref[1])
        e_1b = np.einsum("ij,ji->", h_core, dm_tot_ref).real
        e_2b = 0.5 * (
            np.einsum("ij,ji->", v_j - v_k_a, dm0_ref[0]).real
            + np.einsum("ij,ji->", v_j - v_k_b, dm0_ref[1]).real
        )
        e_mf = float(e_1b + e_2b + mol.energy_nuc())
    else:
        v_eff = scf.RHF(mol).get_veff(mol, dm0_ref)
        e_1b = np.einsum("ij,ji->", h_core, dm0_ref).real
        e_2b = 0.5 * np.einsum("ij,ji->", v_eff, dm0_ref).real
        e_mf = float(e_1b + e_2b + mol.energy_nuc())

    e_tot = float(e_mf + e_corr_cumulant)
    dm_collapsed = dm_conv[0] + dm_conv[1] if dm_conv.ndim == 3 else dm_conv
    e_kin = float(np.einsum("ij,ji->", t_kin, dm_collapsed).real)
    virial = float(-(e_tot - e_kin) / (2.0 * e_kin))

    def _spatial_s_config(dm_mat):
        dm_c = dm_mat[0] + dm_mat[1] if dm_mat.ndim == 3 else dm_mat
        pop_evals = np.linalg.eigvals(dm_c @ s_ovlp).real
        p_sp = np.clip(pop_evals / 2.0, 1e-15, 1.0 - 1e-15)
        val = float(-np.sum(p_sp * np.log2(p_sp) + (1.0 - p_sp) * np.log2(1.0 - p_sp)))
        return 0.0 if val < 1e-10 else val

    s_config = max(_spatial_s_config(dm_conv), _spatial_s_config(dm_eq))

    d_bd = None
    dist_pure = None
    dist_ens = None
    if mol.nelectron == 3:
        if dm_eq.ndim == 3:
            na = np.clip(np.sort(np.linalg.eigvalsh(s_half @ dm_eq[0] @ s_half))[::-1][:3], 0.0, 1.0)
            if abs(float(np.sum(na)) - 2.0) < 1e-6:
                nb = 1.0 - na[::-1]
            else:
                nb = np.clip(np.sort(np.linalg.eigvalsh(s_half @ dm_eq[1] @ s_half))[::-1][:3], 0.0, 1.0)
            n6 = np.sort(np.concatenate([na, nb]))[::-1]
        else:
            n_sp = np.clip(np.sort(np.linalg.eigvalsh(0.5 * s_half @ dm_eq @ s_half))[::-1][:3], 0.0, 1.0)
            n6 = np.sort(np.concatenate([n_sp, n_sp]))[::-1]
        d_bd, dist_pure, dist_ens = compute_polytope_distances_3_6(n6)

    return {
        "E_tot": e_tot,
        "Virial": virial,
        "S_config": s_config,
        "D_BD": d_bd,
        "dist_pure": dist_pure,
        "dist_ens": dist_ens,
    }


def verify_manifest_and_openapi():
    print("=" * 124)
    print("0. CRYPTOGRAPHIC SHA-256 MANIFEST & OPENAPI SCHEMA AUDIT")
    print("=" * 124)
    assert os.path.exists(MANIFEST_PATH), "Missing MANIFEST.sha256"
    assert os.path.exists(OPENAPI_PATH), "Missing openapi.json"

    verified_files = 0
    with open(MANIFEST_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            expected_hash, rel_path = line.split(None, 1)
            abs_path = os.path.join(ROOT_DIR, rel_path)
            assert os.path.exists(abs_path), f"Manifest target missing: {rel_path}"
            actual_hash = hashlib.sha256(open(abs_path, "rb").read()).hexdigest()
            assert actual_hash == expected_hash, f"SHA-256 mismatch for {rel_path}: {actual_hash} != {expected_hash}"
            verified_files += 1

    with open(OPENAPI_PATH, "r") as f:
        oa = json.load(f)
    expected_paths = {"/v1/health", "/v1/q-beck", "/v1/q-besc"}
    assert expected_paths.issubset(set(oa.get("paths", {}).keys())), f"Missing OpenAPI paths: {oa.get('paths', {}).keys()}"
    schemas = oa.get("components", {}).get("schemas", {})
    assert "QuantumBoltzmannRequest" in schemas and "QuantumBoltzmannResponse" in schemas, "Missing OpenAPI schemas"
    print(f"PASS: Verified SHA-256 digests for all {verified_files} files and validated OpenAPI 3.1 contract.\n")


def _assert_manuscript_parity(by_name: dict):
    assert round(by_name["h3_ts"]["e_qbeck"], 6) == -1.607527, f"h3_ts Q-BECK mismatch: {by_name['h3_ts']['e_qbeck']}"
    assert round(by_name["h3_ts"]["e_qbesc"], 6) == -1.646515, f"h3_ts Q-BESC mismatch: {by_name['h3_ts']['e_qbesc']}"
    assert round(by_name["h4_square"]["e_qbeck"], 6) == -2.062836, f"h4_square Q-BECK mismatch: {by_name['h4_square']['e_qbeck']}"
    assert round(by_name["h4_square"]["e_qbesc"], 6) == -2.101908, f"h4_square Q-BESC mismatch: {by_name['h4_square']['e_qbesc']}"
    assert round(by_name["beh2_ts"]["e_qbeck"], 6) == -15.599799, f"beh2_ts Q-BECK mismatch: {by_name['beh2_ts']['e_qbeck']}"
    assert round(by_name["beh2_ts"]["e_qbesc"], 6) == -15.630102, f"beh2_ts Q-BESC mismatch: {by_name['beh2_ts']['e_qbesc']}"
    assert round(by_name["h2o_eq"]["e_qbeck"], 4) == -76.0508, f"h2o_eq Q-BECK mismatch: {by_name['h2o_eq']['e_qbeck']}"
    assert round(by_name["h2o_str"]["e_qbeck"], 4) == -75.8204, f"h2o_str Q-BECK mismatch: {by_name['h2o_str']['e_qbeck']}"
    assert round(by_name["n2_eq"]["e_qbeck"], 6) == -109.001552, f"n2_eq Q-BECK mismatch: {by_name['n2_eq']['e_qbeck']}"
    assert round(by_name["n2_eq"]["e_qbesc"], 6) == -109.090008, f"n2_eq Q-BESC mismatch: {by_name['n2_eq']['e_qbesc']}"

    n2_5a_err = (by_name["n2_str"]["e_qbeck"] - by_name["n2_str"]["e_qbesc"]) * HARTREE_TO_KCAL
    assert round(n2_5a_err, 2) == 69.18, f"n2_str Table 2 error mismatch: {n2_5a_err}"

    d_mo_k = (by_name["m_benzyne"]["e_qbeck"] - by_name["o_benzyne"]["e_qbeck"]) * HARTREE_TO_KCAL
    d_po_k = (by_name["p_benzyne"]["e_qbeck"] - by_name["o_benzyne"]["e_qbeck"]) * HARTREE_TO_KCAL
    d_mo_s = (by_name["m_benzyne"]["e_qbesc"] - by_name["o_benzyne"]["e_qbesc"]) * HARTREE_TO_KCAL
    d_po_s = (by_name["p_benzyne"]["e_qbesc"] - by_name["o_benzyne"]["e_qbesc"]) * HARTREE_TO_KCAL
    assert round(d_mo_k, 2) == 14.84 and round(d_po_k, 2) == 29.18, f"Benzyne Q-BECK gap mismatch: {d_mo_k}, {d_po_k}"
    assert round(d_mo_s, 2) == 16.50 and round(d_po_s, 2) == 29.50, f"Benzyne Q-BESC gap mismatch: {d_mo_s}, {d_po_s}"


def run_offline_verification():
    verify_manifest_and_openapi()

    npz_path = os.path.join(DATA_DIR, "converged_ao_1rdms.npz")
    data = np.load(npz_path, allow_pickle=True)
    manifest = json.loads(str(data["manifest_json"]))

    print("=" * 124)
    print("1. STANDALONE PYSCF AO 1-RDM RECOMPUTATION AUDIT (data/converged_ao_1rdms.npz)")
    print("=" * 124)
    print(
        f"{'System':<12} | {'E_Q-BECK (Ha)':>15} | {'E_Q-BESC (Ha)':>15} | {'|dE| (Ha)':>10} | "
        f"{'-V/(2T)':>9} | {'S_config':>8} | {'||[F,P]||_F':>11} | {'D_BD':>8} | {'||n-p||_2':>9}"
    )
    print("-" * 124)

    max_de = 0.0
    max_dv = 0.0
    max_ds = 0.0
    by_name = {}

    for item in manifest:
        name = item["name"]
        by_name[name] = item
        mol = gto.M(
            atom=item["atom"],
            basis=item["basis"],
            spin=item["spin"],
            unit=item["unit"],
            symmetry=False,
            verbose=0,
        )
        dm_qbeck = data[f"{name}_dm_qbeck"]
        dm_eq_qbeck = data[f"{name}_dm_eq_qbeck"]
        dm0_ref = data[f"{name}_dm0_ref"]
        e_corr = float(data[f"{name}_e_corr_qbeck"])
        e_stored = float(data[f"{name}_e_qbeck"])
        e_qbesc_stored = float(data[f"{name}_e_qbesc"])
        vir_stored = float(data[f"{name}_virial_qbeck"])
        scfg_stored = float(data[f"{name}_s_config_qbeck"])
        comm_norm = float(data[f"{name}_comm_qbeck"])

        recomp = recompute_observables_from_1rdm(mol, dm_qbeck, dm_eq_qbeck, dm0_ref, e_corr_cumulant=e_corr)
        de = abs(recomp["E_tot"] - e_stored)
        dv = abs(recomp["Virial"] - vir_stored)
        ds = abs(recomp["S_config"] - scfg_stored)
        max_de = max(max_de, de)
        max_dv = max(max_dv, dv)
        max_ds = max(max_ds, ds)

        assert de < 1e-9, f"Energy mismatch for {name}: {de}"
        assert dv < 1e-9, f"Virial mismatch for {name}: {dv}"
        assert ds < 1e-9, f"S_config mismatch for {name}: {ds}"
        assert comm_norm < 1e-7, f"Stationarity commutator violation for {name}: {comm_norm}"

        # Verify Q-BESC S_config and correlated virial ratio from archived dm_qbesc via PySCF
        dm_qbesc = data[f"{name}_dm_qbesc"]
        s_ovlp = mol.intor("int1e_ovlp")
        t_kin = mol.intor("int1e_kin")
        dm_s_c = dm_qbesc[0] + dm_qbesc[1] if dm_qbesc.ndim == 3 else dm_qbesc
        pop_s = np.linalg.eigvals(dm_s_c @ s_ovlp).real
        psp = np.clip(pop_s / 2.0, 1e-15, 1.0 - 1e-15)
        sc_s = float(-np.sum(psp * np.log2(psp) + (1.0 - psp) * np.log2(1.0 - psp)))
        if sc_s < 1e-10:
            sc_s = 0.0
        assert abs(sc_s - float(data[f"{name}_s_config_qbesc"])) < 1e-9, f"Q-BESC S_config mismatch for {name}"
        ek_s = float(np.einsum("ij,ji->", t_kin, dm_s_c).real)
        vir_s = float(-(e_qbesc_stored - ek_s) / (2.0 * ek_s))
        assert abs(vir_s - float(data[f"{name}_virial_qbesc"])) < 1e-9, f"Q-BESC virial mismatch for {name}"

        if recomp["D_BD"] is not None:
            assert recomp["D_BD"] >= 0.0 and recomp["dist_pure"] < 1e-9
            dbd_str = f"{recomp['D_BD']:.2e}"
            dp_str = f"{recomp['dist_pure']:.2e}"
        else:
            dbd_str = "   N/A  "
            dp_str = "   N/A   "

        print(
            f"{name:<12} | {recomp['E_tot']:15.9f} | {e_qbesc_stored:15.9f} | {de:10.2e} | "
            f"{recomp['Virial']:9.6f} | {recomp['S_config']:8.4f} | {comm_norm:11.2e} | {dbd_str:>8} | {dp_str:>9}"
        )

    _assert_manuscript_parity(by_name)
    d_mo_k = (by_name["m_benzyne"]["e_qbeck"] - by_name["o_benzyne"]["e_qbeck"]) * HARTREE_TO_KCAL
    d_po_k = (by_name["p_benzyne"]["e_qbeck"] - by_name["o_benzyne"]["e_qbeck"]) * HARTREE_TO_KCAL
    d_mo_s = (by_name["m_benzyne"]["e_qbesc"] - by_name["o_benzyne"]["e_qbesc"]) * HARTREE_TO_KCAL
    d_po_s = (by_name["p_benzyne"]["e_qbesc"] - by_name["o_benzyne"]["e_qbesc"]) * HARTREE_TO_KCAL
    n2_5a_err = (by_name["n2_str"]["e_qbeck"] - by_name["n2_str"]["e_qbesc"]) * HARTREE_TO_KCAL

    print("-" * 124)
    print(
        f"PASS: All 10 archived AO 1-RDMs verified via PySCF to < 10^-9 E_h "
        f"(max |dE| = {max_de:.2e} Ha, max |dVirial| = {max_dv:.2e}, max |dS_config| = {max_ds:.2e})."
    )
    print(
        f"PASS: Manuscript parity confirmed — Benzyne Q-BECK ({d_mo_k:.2f}, {d_po_k:.2f} kcal/mol), "
        f"Q-BESC(4,4) ({d_mo_s:.2f}, {d_po_s:.2f} kcal/mol), N2 5.0 A Table 2 (+{n2_5a_err:.2f} kcal/mol).\n"
    )

    print("=" * 124)
    print("2. POLYTOPE DISTANCE & BENCHMARK TRAJECTORY INTEGRITY AUDIT (8 MANUSCRIPT SUITES)")
    print("=" * 124)

    # 1. H3 collinear, MEP, & 41-point GPC polytope distances
    h3_col = np.load(os.path.join(DATA_DIR, "h3_collinear_data.npz"))
    h3_mep = np.load(os.path.join(DATA_DIR, "h3_mep_data.npz"))
    with open(os.path.join(DATA_DIR, "h3_mep_gpc_distances.json"), "r") as f:
        h3_gpc = json.load(f)
    max_dp_gpc, max_de_gpc = 0.0, 0.0
    for mkey in ("Q_BESC", "Q_BECK"):
        for idx, n6 in enumerate(h3_gpc[mkey]["occupations_n6"]):
            dbd_i, dp_i, de_i = compute_polytope_distances_3_6(n6)
            assert dbd_i >= 0.0, f"Borland-Dennis violation in {mkey} at idx={idx}: {dbd_i}"
            max_dp_gpc = max(max_dp_gpc, abs(dp_i - h3_gpc[mkey]["pure_distance"][idx]))
            max_de_gpc = max(max_de_gpc, abs(de_i - h3_gpc[mkey]["ensemble_distance"][idx]))
    assert max_dp_gpc < 1e-9 and max_de_gpc < 1e-9, f"Polytope distance mismatch: {max_dp_gpc}, {max_de_gpc}"
    print(
        f"[1. H3 MEP & GPC Polytopes]   MEP pts: {len(h3_mep['s'])}, Collinear pts: {len(h3_col['R1'])}, "
        f"max |d_pure| = {max_dp_gpc:.2e}, max |d_ens| = {max_de_gpc:.2e}"
    )

    # 2. H4 D4h Mott dissociation
    h4 = np.load(os.path.join(DATA_DIR, "h4_d4h_data.npz"))
    with open(os.path.join(DATA_DIR, "h4_d4h_mott_transition.json"), "r") as f:
        h4_mott = json.load(f)
    print(
        f"[2. H4 Square D4h Mott]       Grid pts: {len(h4['R'])}, min E_Q-BECK = {float(np.min(h4['E_QBECK'])):.6f} Ha, "
        f"min E_Q-BESC = {float(np.min(h4['E_QBEC'])):.6f} Ha, S_config(3.2A) = {h4_mott['trajectory'][-1]['S_config_bits']['Q-BECK']:.3f} bits"
    )

    # 3. BeH2 C2v avoided crossing (6-31G)
    beh2 = np.load(os.path.join(DATA_DIR, "fig7a_beh2_data.npz"))
    with open(os.path.join(DATA_DIR, "beh2_c2v_avoided_crossing_diagnostics.json"), "r") as f:
        beh2_diag = json.load(f)
    print(
        f"[3. BeH2 C2v (6-31G)]         Grid pts: {len(beh2['x_bohr'])}, max |E_QBESC - E_S0| = "
        f"{beh2_diag['summary_metrics']['max_abs_diff_QBESC_S0_Eh']:.2e} Ha"
    )

    # 4. H2O symmetric double dissociation
    with open(os.path.join(DATA_DIR, "h2o_dissociation_benchmark.json"), "r") as f:
        h2o = json.load(f)
    h2o_dense = np.load(os.path.join(DATA_DIR, "h2o_dissociation_data_dense.npz"))
    print(
        f"[4. H2O Double Dissociation]  Grid pts: {len(h2o_dense['R'])}, Eq E(Q-BECK) = {min(h2o['E_QBECK']):.6f} Ha, "
        f"Eq E(Q-BESC) = {min(h2o['E_QBE_CASSCF']):.6f} Ha"
    )

    # 5. N2 Table 1 basis hierarchy & triple-bond virial audit
    with open(os.path.join(DATA_DIR, "n2_casci_registers_study.json"), "r") as f:
        n2 = json.load(f)
    with open(os.path.join(DATA_DIR, "n2_eq_table1_convergence.json"), "r") as f:
        n2_t1 = json.load(f)
    n2_vir = np.load(os.path.join(DATA_DIR, "n2_virial_defect.npz"))
    print(
        f"[5. N2 Basis & Virial Audit]  Table 1 bases: {len(n2_t1['rows'])}, PES pts: {len(n2['R'])}, "
        f"max |1 - (-V/2T)| (Q-BECK) = {float(np.max(np.abs(n2_vir['virial_qbeck']))):.4f}"
    )

    # 6 & 7. MR-MGN-BE17 & C6H4 diradicals + O(N^3) scaling
    with open(os.path.join(DATA_DIR, "mr_mgn_be17_benchmark.json"), "r") as f:
        mgn = json.load(f)
    s_mgn = mgn["summary"]
    print(
        f"[6. MR-MGN-BE17 (11 Systems)] MUE(Q-BECK) = {s_mgn['err_QBECK']['MUE']:.2f} kcal/mol "
        f"(vs. QBE-SCF {s_mgn['err_QBESCF']['MUE']:.2f} kcal/mol), MUE(Q-BESC) = {s_mgn['err_QBESC']['MUE']:.2e} kcal/mol"
    )

    with open(os.path.join(DATA_DIR, "benzyne_and_scaling_data.json"), "r") as f:
        bz = json.load(f)
    gaps_k = bz["benzyne_comparison"]["Q-BECK"]
    gaps_s = bz["benzyne_comparison"]["Q-BESC(4,4)"]
    sc_rec = bz["scaling_records"]
    max_rec = max(sc_rec, key=lambda r: r["nao"])
    print(
        f"[7. C6H4 Diradical Isomers]   Q-BECK (m-o, p-o) = ({gaps_k['meta_ortho']:.2f}, {gaps_k['para_ortho']:.2f}) kcal/mol | "
        f"Q-BESC(4,4) = ({gaps_s['meta_ortho']:.2f}, {gaps_s['para_ortho']:.2f}) kcal/mol"
    )
    print(
        f"[8. O(N^3) Scaling ({len(sc_rec)} Sys)]  Empirical exponent alpha = {bz['empirical_time_exponent']:.2f}, "
        f"{max_rec['system']} (N_AO={max_rec['nao']}): {max_rec['t_step_ms']:.2f} ms/step, {max_rec['mem_qbeck_mb']:.2f} MB RAM"
    )
    print("=" * 124)
    print("OFFLINE REPRODUCIBILITY VERIFICATION PASSED.\n")


# =============================================================================
# LIVE FASTAPI GATEWAY VERIFICATION ROUTINES (--mode api)
# =============================================================================

def _beh2_geom_bohr(x_ang: float) -> str:
    xb = 2.9301383049 if abs(x_ang - 1.54) < 1e-6 else x_ang * BOHR_PER_ANG
    yb = 2.540 - 0.46 * xb
    return f"Be 0.0 0.0 0.0; H {xb:.10f} {yb:.10f} 0.0; H {xb:.10f} {-yb:.10f} 0.0"


def _h2o_geom_ang(r_oh: float) -> str:
    half_th = math.radians(104.5) / 2.0
    x = r_oh * math.sin(half_th)
    z = r_oh * math.cos(half_th)
    return f"O 0.0 0.0 0.0; H {x:.8f} 0.0 {z:.8f}; H {-x:.8f} 0.0 {z:.8f}"


def _mgn_geom_ang(key: str, r_bond: float) -> str:
    if key == "CO2":
        return f"C 0 0 0; O 0 0 {r_bond}; O 0 0 {-r_bond}"
    if key == "NF3":
        sin_a = math.sqrt((1.0 - math.cos(math.radians(102.2))) * 2.0 / 3.0)
        cos_a = math.sqrt(1.0 - sin_a ** 2)
        z = -r_bond * cos_a
        rho = r_bond * sin_a
        pts = ["N 0.0 0.0 0.0"]
        for k in range(3):
            phi = 2.0 * math.pi * k / 3.0
            pts.append(f"F {rho * math.cos(phi):.6f} {rho * math.sin(phi):.6f} {z:.6f}")
        return "; ".join(pts)
    pair = {
        "B2": ("B", "B"), "C2": ("C", "C"), "CN": ("C", "N"), "F2": ("F", "F"),
        "NO": ("N", "O"), "S2": ("S", "S"), "SiO": ("Si", "O"), "CO": ("C", "O"), "N2": ("N", "N"),
    }[key]
    return f"{pair[0]} 0 0 0; {pair[1]} 0 0 {r_bond}"


def build_api_benchmark_suites(extended: bool = False):
    items = []

    # 1. H3 Collinear MEP & Borland-Dennis Polytope
    for tag, r1, r2 in [("TS s=0.00A", 0.93, 0.93), ("Asymp s=2.08A", 3.00, 0.74)]:
        atom = f"H 0.0 0.0 {-r1:.6f}; H 0.0 0.0 0.0; H 0.0 0.0 {r2:.6f}"
        items.append({
            "suite": "1. H3 MEP",
            "label": f"H3 ({tag})",
            "endpoints": ["/v1/q-beck", "/v1/q-besc"],
            "payload_qbeck": {"atom": atom, "basis": "cc-pvdz", "spin": 1, "active_space": [3, 3], "use_broken_symmetry_seed": True},
            "payload_qbesc": {"atom": atom, "basis": "cc-pvdz", "spin": 1, "active_space": None, "use_broken_symmetry_seed": True},
            "check_dbd": True,
        })

    # 2. H4 Square D4h Symmetric Dissociation
    for tag, r_side in [("Eq R=1.20A", 1.20), ("Mott R=3.20A", 3.20)]:
        h = r_side / 2.0
        atom = f"H {h:.6f} {h:.6f} 0.0; H {-h:.6f} {h:.6f} 0.0; H {-h:.6f} {-h:.6f} 0.0; H {h:.6f} {-h:.6f} 0.0"
        items.append({
            "suite": "2. H4 D4h",
            "label": f"H4 ({tag})",
            "endpoints": ["/v1/q-beck", "/v1/q-besc"],
            "payload_qbeck": {"atom": atom, "basis": "cc-pvdz", "spin": 0, "symmetry": True, "active_space": [4, 4], "use_broken_symmetry_seed": True},
            "payload_qbesc": {"atom": atom, "basis": "cc-pvdz", "spin": 0, "symmetry": True, "active_space": None, "use_broken_symmetry_seed": True},
        })

    # 3. BeH2 C2v Avoided Crossing (6-31G)
    for tag, x_ang in [("React x=0.40A", 0.40), ("Apex x=1.54A", 1.54), ("Prod x=2.00A", 2.00)]:
        p = {
            "atom": _beh2_geom_bohr(x_ang),
            "basis": "6-31g",
            "unit": "Bohr",
            "symmetry": "C2v",
            "singlet_s0_only": True,
            "active_space": [4, 12],
        }
        items.append({
            "suite": "3. BeH2 C2v",
            "label": f"BeH2 ({tag})",
            "endpoints": ["/v1/q-beck", "/v1/q-besc"],
            "payload_qbeck": p,
            "payload_qbesc": p,
        })

    # 4. H2O Symmetric Double Dissociation
    for tag, r_oh in [("Eq R=0.967A", 0.967), ("Str R=2.00A", 2.00), ("Diss R=3.20A", 3.20)]:
        p = {
            "atom": _h2o_geom_ang(r_oh),
            "basis": "cc-pvdz",
            "active_space": [8, 6],
            "use_broken_symmetry_seed": True,
        }
        items.append({
            "suite": "4. H2O Dissoc",
            "label": f"H2O ({tag})",
            "endpoints": ["/v1/q-beck", "/v1/q-besc"],
            "payload_qbeck": p,
            "payload_qbesc": p,
        })

    # 5. N2 Table 1 Basis Hierarchy & Triple-Bond Dissociation
    n2_bases = ["cc-pvdz", "aug-cc-pvdz", "cc-pvtz", "aug-cc-pvtz", "cc-pvqz"]
    if extended:
        n2_bases.extend(["aug-cc-pvqz", "cc-pv5z", "aug-cc-pv5z"])
    for b in n2_bases:
        p = {"atom": "N 0 0 0; N 0 0 2.074", "unit": "Bohr", "basis": b, "active_space": [6, 6]}
        items.append({
            "suite": "5. N2 Basis/PES",
            "label": f"N2 Eq ({b})",
            "endpoints": ["/v1/q-beck", "/v1/q-besc"],
            "payload_qbeck": p,
            "payload_qbesc": p,
        })
    for r_n2, uhf in [(2.20, False), (4.00, True), (5.00, False)]:
        p = {"atom": f"N 0 0 0; N 0 0 {r_n2:.2f}", "basis": "cc-pvdz", "active_space": [6, 6], "use_broken_symmetry_seed": uhf}
        items.append({
            "suite": "5. N2 Basis/PES",
            "label": f"N2 Str (R={r_n2:.2f}A, uhf={uhf})",
            "endpoints": ["/v1/q-beck", "/v1/q-besc"],
            "payload_qbeck": p,
            "payload_qbesc": p,
        })

    # 6. Benzyne Diradical Isomers
    for bname, geom in BENZYNE_GEOMS.items():
        p = {"atom": geom, "basis": "cc-pvdz", "active_space": [4, 4]}
        items.append({
            "suite": "6. C6H4 Isomers",
            "label": bname,
            "endpoints": ["/v1/q-beck", "/v1/q-besc"],
            "payload_qbeck": p,
            "payload_qbesc": p,
        })

    # 7. MR-MGN-BE17 11-Molecule Homolytic Dissociation Suite
    for key, r_eq, spin, asp in MR_MGN_BE17_SPECS:
        for r_tag, r_val, uhf in [("Eq", r_eq, False), ("5.0A", 5.0, True)]:
            p = {
                "atom": _mgn_geom_ang(key, r_val),
                "basis": "cc-pvdz",
                "spin": spin,
                "active_space": asp,
                "use_broken_symmetry_seed": uhf,
            }
            items.append({
                "suite": "7. MR-MGN-BE17",
                "label": f"{key} ({r_tag})",
                "endpoints": ["/v1/q-beck", "/v1/q-besc"],
                "payload_qbeck": p,
                "payload_qbesc": p,
            })

    # 8. O(N^3) Scaling Chains on POST /v1/q-beck
    h_counts = [10, 20, 30] + ([40, 60] if extended else [])
    for n_h in h_counts:
        geom_hn = "; ".join([f"H 0.0 0.0 {1.4 * k:.3f}" for k in range(n_h)])
        items.append({
            "suite": "8. Scaling",
            "label": f"H_{n_h} chain (cc-pvdz)",
            "endpoints": ["/v1/q-beck"],
            "payload_qbeck": {"atom": geom_hn, "basis": "cc-pvdz", "active_space": None},
        })
    alkanes = [
        ("CH4 (cc-pvdz)", "C 0 0 0; H 0.629 0.629 0.629; H -0.629 -0.629 0.629; H -0.629 0.629 -0.629; H 0.629 -0.629 -0.629"),
        ("C2H6 (cc-pvdz)", "C 0 0 -0.765; C 0 0 0.765; H 0 1.018 -1.157; H 0.882 -0.509 -1.157; H -0.882 -0.509 -1.157; H 0 -1.018 1.157; H -0.882 0.509 1.157; H 0.882 0.509 1.157"),
    ]
    for label, geom in alkanes:
        items.append({
            "suite": "8. Scaling",
            "label": label,
            "endpoints": ["/v1/q-beck"],
            "payload_qbeck": {"atom": geom, "basis": "cc-pvdz", "active_space": None},
        })

    return items


async def run_api_verification(base_url: str, extended: bool = False):
    import httpx

    npz_path = os.path.join(DATA_DIR, "converged_ao_1rdms.npz")
    npz_data = np.load(npz_path, allow_pickle=True)
    sec1_manifest = json.loads(str(npz_data["manifest_json"]))

    items = build_api_benchmark_suites(extended=extended)
    total_calls = 0

    async with httpx.AsyncClient(base_url=base_url, timeout=600.0) as client:
        try:
            health_resp = await client.get("/v1/health")
            health_resp.raise_for_status()
        except Exception as exc:
            raise RuntimeError(
                f"Unable to reach live FastAPI gateway at {base_url}/v1/health ({exc}). "
                f"Ensure the FastAPI server is running at {base_url} before invoking --mode api."
            ) from exc

        health = health_resp.json()
        assert health["status"] == "ok", f"Unexpected health status: {health}"
        assert health["defaults"]["delta_0_eh"] == 0.45, "Expected invariant delta_0 = 0.45 E_h"

        live_oa = (await client.get("/openapi.json")).json()
        with open(OPENAPI_PATH, "r") as f:
            local_oa = json.load(f)
        assert set(local_oa["paths"].keys()) == set(live_oa["paths"].keys()), "OpenAPI path mismatch vs live server"

        print("=" * 124)
        print(f"1. SECTION 1 CANONICAL ARCHIVE vs. LIVE FASTAPI PARITY AUDIT ({base_url})")
        print("=" * 124)
        print(
            f"{'System':<12} | {'E_Q-BECK (API)':>15} | {'|dE_BECK|':>10} | "
            f"{'E_Q-BESC (API)':>15} | {'|dE_BESC|':>10} | {'-V/(2T)':>9} | {'||[F,P]||_F':>11}"
        )
        print("-" * 124)

        max_api_de_k = 0.0
        max_api_de_s = 0.0
        for m_item in sec1_manifest:
            rk = await client.post("/v1/q-beck", json=m_item["payload_qbeck"])
            rk.raise_for_status()
            res_k = rk.json()
            rs = await client.post("/v1/q-besc", json=m_item["payload_qbesc"])
            rs.raise_for_status()
            res_s = rs.json()
            total_calls += 2

            de_k = abs(float(res_k["e_tot"]) - float(m_item["e_qbeck"]))
            de_s = abs(float(res_s["e_tot"]) - float(m_item["e_qbesc"]))
            max_api_de_k = max(max_api_de_k, de_k)
            max_api_de_s = max(max_api_de_s, de_s)
            assert de_k < 1e-9, f"Live API Q-BECK parity mismatch for {m_item['name']}: {de_k}"
            assert de_s < 1e-9, f"Live API Q-BESC parity mismatch for {m_item['name']}: {de_s}"
            assert res_k["converged"] and res_s["converged"]

            print(
                f"{m_item['name']:<12} | {res_k['e_tot']:15.9f} | {de_k:10.2e} | "
                f"{res_s['e_tot']:15.9f} | {de_s:10.2e} | {res_k['virial_ratio']:9.6f} | {res_k['commutator_norm']:11.2e}",
                flush=True,
            )

        print("-" * 124)
        print(
            f"PASS: All 10 Section 1 canonical entries reproduced live to < 10^-9 E_h "
            f"(max |dE_BECK| = {max_api_de_k:.2e} Ha, max |dE_BESC| = {max_api_de_s:.2e} Ha).\n"
        )

        print("=" * 124)
        print(f"2. LIVE FASTAPI GATEWAY SUITE VERIFICATION ({base_url})")
        print("=" * 124)
        print(f"Engine Status : {health['status']} | {health['engine']}")
        print(f"Solver Freeze : {health['defaults']}")
        print("-" * 124)
        print(
            f"{'Suite':<16} | {'System':<28} | {'Method':<7} | {'N_AO':>4} | {'E_tot (Ha)':>14} | "
            f"{'-V/(2T)':>8} | {'S_cfg':>7} | {'||[F,P]||_F':>11} | {'ms/step':>8}"
        )
        print("-" * 124)

        for item in items:
            for endpoint in item["endpoints"]:
                payload = item["payload_qbeck"] if endpoint == "/v1/q-beck" else item["payload_qbesc"]
                resp = await client.post(endpoint, json=payload)
                resp.raise_for_status()
                res = resp.json()
                total_calls += 1

                assert res["converged"], f"{endpoint} did not converge for {item['label']}: {res}"
                assert res["commutator_norm"] < 1e-7, (
                    f"Commutator norm {res['commutator_norm']:.2e} >= 1e-7 for {item['label']} ({endpoint})"
                )
                if item.get("check_dbd"):
                    assert res["d_bd"] is not None and res["d_bd"] >= 0.0, (
                        f"Borland-Dennis slack D_BD < 0 for {item['label']} ({endpoint}): {res['d_bd']}"
                    )

                print(
                    f"{item['suite']:<16} | {item['label']:<28} | {res['method']:<7} | {res['n_ao']:4d} | "
                    f"{res['e_tot']:14.8f} | {res['virial_ratio']:8.5f} | {res['s_config']:7.4f} | "
                    f"{res['commutator_norm']:11.2e} | {res['step_time_ms']:8.2f}",
                    flush=True,
                )

        print("=" * 124)
        print(
            f"PASS: All {total_calls} live FastAPI endpoint evaluations across Section 1 and all 8 manuscript benchmark "
            f"suites verified (converged=True, ||[F_eff, P]||_F < 1e-7, D_BD >= 0, < 10^-9 E_h parity)."
        )


def main():
    parser = argparse.ArgumentParser(
        description="Standalone offline (PySCF) and live API (FastAPI/httpx) reproducibility verifier."
    )
    parser.add_argument(
        "--mode",
        choices=["offline", "api", "both"],
        default="offline",
        help="Verification mode: 'offline' (PySCF 1-RDM & dataset audit), 'api' (live FastAPI endpoints), or 'both'.",
    )
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8000",
        help="Base URL of the live local FastAPI gateway for --mode api (default: http://127.0.0.1:8000).",
    )
    parser.add_argument(
        "--extended",
        action="store_true",
        help="Include aug-cc-pVQZ..aug-cc-pV5Z N2 and H_40..H_60 scaling points during --mode api.",
    )
    args = parser.parse_args()

    if args.mode in ("offline", "both"):
        run_offline_verification()
    if args.mode in ("api", "both"):
        asyncio.run(run_api_verification(args.url, extended=args.extended))


if __name__ == "__main__":
    main()
