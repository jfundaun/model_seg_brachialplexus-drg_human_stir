#!/usr/bin/env python3
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
