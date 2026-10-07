# BPSeg

Automatic segmentation of the bilateral C5 to C8 brachial plexus nerve roots and dorsal root ganglia (DRG) from T2 STIR MRI using a two-stage cascaded nnU-Net model.

Repository: `jfundaun/model_seg_brachialplexus-drg_human_stir`. Trained weights are distributed as GitHub release assets under tag `r20260723`.

## Table of contents
- [Model description](#model-description)
- [List of classes](#list-of-classes)
- [Datasets](#datasets)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Weights](#weights)
- [Inference](#inference)
- [Training](#training)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

## Model description
BPSeg targets small, bilateral neural structures that span a large field of view. Such segmentations can be challenging for a single network, so the model uses a two-stage cascade:
- **Stage 1 (Dataset201, coarse):** a 3-class segmentation (background, DRG, plexus) that localizes the target region.
- **Stage 2 (Dataset202, fine):** a 16-class bilateral segmentation. It takes two input channels, the image and the Stage 1 prediction, using the coarse map as a spatial prior.

Both stages use the nnU-Net v2 `3d_fullres` configuration with the ResEncUNet-M planner (`nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres`) and are released as 5-fold ensembles. Inference applies a post-processing step that splits any left/right merged structures by world-space (RAS) x-coordinate. 

## List of classes
The mapping is also in the `dataset.json` inside each Dataset202 archive.

| Label | Structure | Label | Structure |
|------:|-----------|------:|-----------|
| 1 | DRG_C5_R | 9  | BP_C5_R |
| 2 | DRG_C6_R | 10 | BP_C6_R |
| 3 | DRG_C7_R | 11 | BP_C7_R |
| 4 | DRG_C8_R | 12 | BP_C8_R |
| 5 | DRG_C5_L | 13 | BP_C5_L |
| 6 | DRG_C6_L | 14 | BP_C6_L |
| 7 | DRG_C7_L | 15 | BP_C7_L |
| 8 | DRG_C8_L | 16 | BP_C8_L |

## Datasets
Trained on approximately 400 T2 STIR scans from three sites (Oxford, UK; Brighton, UK; Stanford, CA), acquired on Siemens and GE scanners. The train/test split is in `participants/participant_ids_train_test.csv`.

## Dependencies
- Python 3.9 or later
- nnU-Net v2
- PyTorch (CUDA build for GPU use)
- nibabel, numpy, scipy

Tested with Python 3.9.21, `nnunetv2` 2.5.2, `torch` 2.6.0, `nibabel` 5.3.2, `scipy` 1.13.1. Pin `numpy<2`: numpy 2.x triggers a `blosc2` binary-incompatibility error when importing `nnunetv2`.

## Installation
```bash
git clone https://github.com/jfundaun/model_seg_brachialplexus-drg_human_stir.git
cd model_seg_brachialplexus-drg_human_stir

python3 -m venv venv && source venv/bin/activate
pip install "nnunetv2" "numpy<2" nibabel scipy
```

Install PyTorch following the instructions on the PyTorch website for your platform and CUDA version, then download the weights (see below).

## Weights
Weights are attached to the `r20260723` release: ten archives (5 folds per stage), each containing `checkpoint_final.pth`, `plans.json`, `dataset.json`, and `dataset_fingerprint.json`.

Full 5-fold ensemble:

```bash
gh release download r20260723 -R jfundaun/model_seg_brachialplexus-drg_human_stir --dir nnunet_results
cd nnunet_results && for z in *_r20260723.zip; do unzip -o "$z"; done
```

The fast preset needs only fold 0 of each stage:

```bash
gh release download r20260723 -R jfundaun/model_seg_brachialplexus-drg_human_stir \
  -p "Dataset201_DRGPlexusCoarse_fold0_r20260723.zip" \
  -p "Dataset202_DRGPlexusFine_fold0_r20260723.zip" --dir nnunet_results
cd nnunet_results && for z in *_r20260723.zip; do unzip -o "$z"; done
```

Both unzip to `model_seg_brachialplexus-drg_human_t2_r20260723/` in the standard nnU-Net results layout.

## Inference
Set `NNUNET_RESULTS` near the top of `inference/infer_clinical_single_fast.py` to the unzipped weights directory (`.../nnunet_results/model_seg_brachialplexus-drg_human_t2_r20260723`), then run on a single image:

```bash
python inference/infer_clinical_single_fast.py \
  --preset fast --device cpu \
  --input sub-001_t2_space_stir_cor.nii.gz \
  --output out/sub-001
```

The cascade runs Stage 1, feeds its output to Stage 2 as the second channel, and writes the 16-class segmentation to the output directory.

- `--preset fast` uses a single fold with no test-time augmentation (about 6 minutes per scan on CPU). See the script for the full-ensemble option.
- `--device cpu` works everywhere; use `--device cuda` on a GPU. `--device mps` is unsupported on Apple Silicon (no MPS kernel for `ConvTranspose3d`); fall back to `cpu`.

## Training
Set the nnU-Net environment variables and submit the pipeline on a Slurm cluster:

```bash
export nnUNet_raw=/path/to/raw
export nnUNet_preprocessed=/path/to/preprocessed
export nnUNet_results=/path/to/results
bash training/submit_pipeline_v2.sh
```

`prepare_nnunet_data.py` builds the raw dataset from a CSV data dictionary. `submit_pipeline_v2.sh` then chains, with `afterok` dependencies: Dataset201 preprocessing, Stage 1 5-fold training, Phase 2, and Stage 2 5-fold training.

Phase 2 (`phase2_v2.sh`) transitions between stages. `crossval_preds.py` generates a Stage 1 prediction for each training case from the folds that did not see it, and `update_channel1.py` writes those predictions into the Stage 2 channel-1 inputs, replacing the ground-truth coarse initialization so Stage 2 trains under inference-like conditions.

## Citation
Manuscript in preparation:

This work is built on nnU-Net; please also cite:
```bibtex
@article{isensee2021nnunet,
  title={nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation},
  author={Isensee, Fabian and Jaeger, Paul F and Kohl, Simon A A and Petersen, Jens and Maier-Hein, Klaus H},
  journal={Nature Methods},
  volume={18},
  number={2},
  pages={203--211},
  year={2021}
}
```
