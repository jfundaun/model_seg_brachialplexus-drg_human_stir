#!/usr/bin/env python3
"""
The R normative pipeline (bpseg_normative_v3.R) does its OWN multi-cohort ID
crosswalk and healthy/patient split from merged_demographics_2026.csv, so this
script no longer joins demographics or writes hc/patient splits. It just does
the raw consolidation + QC that R consumes:

    idps_all_long.csv   one row per subject per structure (the IDPs)
    normref_all.csv     per-subject vertebral reference
    qc_report.csv       per-subject completeness + plausibility flags
                        (complete_16, has_SI, n_zero_or_neg_vol, n_implausible_vol)

Run:
    python 04_consolidate.py \
        --idp-dir /scratch/users/jfundaun/bpseg/scripts/totalspineseg/idps \
        --out-dir /scratch/users/jfundaun/bpseg/derivatives/bp_totalspineseg_analysis
"""
import argparse, glob, os, sys
import pandas as pd

LEVELS, SIDES, TYPES = ["C5","C6","C7","C8"], ["L","R"], ["DRG","root"]
EXPECTED = {f"{s}_{lv}_{t}" for s in SIDES for lv in LEVELS for t in TYPES}   # 16
N_EXPECTED = len(EXPECTED)


def concat(idp_dir, suffix):
    files = sorted(glob.glob(os.path.join(idp_dir, f"*{suffix}")))
    frames = []
    for f in files:
        if os.path.getsize(f) == 0:
            continue
        try:
            d = pd.read_csv(f)
        except Exception as e:
            print(f"WARN: could not read {os.path.basename(f)}: {e}"); continue
        if not d.empty:
            frames.append(d)
    return (pd.concat(frames, ignore_index=True) if frames else None), len(files)


def qc_report(long):
    rows = []
    for sid, g in long.groupby("subject_id"):
        present = set(g["structure"])
        vols = g["Volume_mm3"] if "Volume_mm3" in g else pd.Series(dtype=float)
        rows.append({
            "subject_id": sid,
            "n_rows": len(g),
            "n_structures": len(present),
            "complete_16": len(present & EXPECTED) == N_EXPECTED,
            "n_missing": len(EXPECTED - present),
            "missing_structures": ";".join(sorted(EXPECTED - present)),
            "n_zero_or_neg_vol": int((vols.fillna(0) <= 0).sum()),
            "n_implausible_vol": int(((vols > 0) & ((vols < 5) | (vols > 5000))).sum()),
            "has_SI": int(g["SI_norm"].notna().any()) if "SI_norm" in g else 0,
        })
    return pd.DataFrame(rows).sort_values("subject_id")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--idp-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    long, n_idp = concat(args.idp_dir, "_idps.csv")
    if long is None:
        sys.exit(f"ERROR: no *_idps.csv found in {args.idp_dir}")
    long.to_csv(os.path.join(args.out_dir, "idps_all_long.csv"), index=False)
    print(f"idps_all_long.csv : {n_idp} files -> {len(long)} rows, "
          f"{long['subject_id'].nunique()} subjects")

    nr, n_nr = concat(args.idp_dir, "_normref.csv")
    if nr is not None:
        nr.to_csv(os.path.join(args.out_dir, "normref_all.csv"), index=False)
        print(f"normref_all.csv   : {n_nr} files -> {len(nr)} rows")
    else:
        print("normref_all.csv   : none found (skipped)")

    qc = qc_report(long)
    qc.to_csv(os.path.join(args.out_dir, "qc_report.csv"), index=False)
    print(f"qc_report.csv     : {len(qc)} subjects | "
          f"complete_16={int(qc['complete_16'].sum())} | "
          f"has_SI={int((qc['has_SI']==1).sum())}")
    print(f"\nWrote 3 tables to {args.out_dir}. Next: run bpseg_normative_v3.R "
          f"(it does the ID crosswalk, demographics join, and healthy/patient split).")


if __name__ == "__main__":
    main()
