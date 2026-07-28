#!/usr/bin/env python3
"""
extract_one_label.py  --  compute pyRadiomics Mean + JointEntropy for ONE label
================================================================================
Runs as an isolated subprocess so a native crash (segfault) inside pyRadiomics
for one structure cannot kill the whole subject's extraction -- the parent
(02_extract_idps.py) just sees a non-zero/negative return code and logs it.

Usage: python extract_one_label.py IMAGE SEG LABEL BIN_WIDTH
Prints one line of JSON to stdout on success: {"mean": ..., "entropy": ...}
Any failure (exception OR crash) -> non-zero exit, nothing useful on stdout.
"""
import sys
import json
import logging

import SimpleITK as sitk
from radiomics import featureextractor
logging.getLogger("radiomics").setLevel(logging.ERROR)


def main():
    img_path, seg_path, label, bin_width = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    img = sitk.ReadImage(img_path)
    seg = sitk.ReadImage(seg_path)
    seg.CopyInformation(img)                 # guard against float header noise

    ex = featureextractor.RadiomicsFeatureExtractor()
    ex.settings["binWidth"] = bin_width
    ex.disableAllFeatures()
    ex.enableFeaturesByName(firstorder=["Mean"], glcm=["JointEntropy"])
    f = ex.execute(img, seg, label=label)
    print(json.dumps({"mean": float(f["original_firstorder_Mean"]),
                      "entropy": float(f["original_glcm_JointEntropy"])}))


if __name__ == "__main__":
    main()
