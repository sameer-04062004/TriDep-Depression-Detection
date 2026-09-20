# TriDep — Tri-Modal Deep Learning Framework for Automated Depression Detection

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces%20Live%20Demo-blue?style=for-the-badge&logo=huggingface)](https://huggingface.co/spaces/sameer-04062004/TriDep-Depression-Detection)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

> 🚀 **Live Interactive Demo:** [Open TriDep on Hugging Face Spaces](https://huggingface.co/spaces/sameer-04062004/TriDep-Depression-Detection)  
> ⚠️ **Research prototype only — not a clinical diagnostic tool.** Always consult a qualified mental health professional.

A research prototype that detects depression from clinical interview recordings by combining three behavioural modalities — **text**, **audio**, and **video** — using pre-trained representation models and a model-level fusion network. Evaluated on the DAIC-WOZ dataset using both Leave-One-Subject-Out Cross-Validation and 5-Fold Stratified Cross-Validation.

## Overview

| Modality | Model | Output |
|---|---|---|
| Text | Sentence-BERT (`all-mpnet-base-v2`) | 768-d semantic embedding |
| Audio | Wav2Vec2 (`facebook/wav2vec2-base-960h`) | 1536-d acoustic embedding (mean+std pooled) |
| Video | OpenFace Facial Action Units | 20-d facial-behaviour vector |

The three modality vectors are fused through a **model-level (intermediate) fusion** architecture: audio and video are combined first, then text is introduced, followed by a shared dense network producing a binary depression classification.

## Model Evolution

This project developed in two iterations:

- **Baseline (Initial Approach):** Leave-One-Subject-Out Cross-Validation, simple dense fusion, binary cross-entropy loss, fixed 0.5 threshold.
- **Improved (Refined Approach):** 5-Fold Stratified Cross-Validation, BatchNorm + L2-regularised fusion, Focal Loss (handles class imbalance), SMOTE oversampling on training folds, and an optimal Youden-J decision threshold.

## Dataset

This project uses the **DAIC-WOZ** dataset (Distress Analysis Interview Corpus – Wizard of Oz), a licensed clinical dataset. **The dataset is NOT included in this repository** and must be obtained directly from the official source:

🔗 https://dcapswoz.ict.usc.edu/

Access requires signing an End User Licence Agreement (EULA). This repository contains only the processing and modelling code — no data, cached features, or model weights derived from the dataset are distributed here, in compliance with the dataset licence.

## Repository Structure

```
app.py                     # Standalone Gradio web application
run_demo.bat               # 1-click Windows launcher for local demo
notebooks/
  01_preprocess.ipynb      # Feature extraction (SBERT, Wav2Vec2, FAU) and caching
  02_train_evaluate.ipynb  # Model training + Baseline vs Improved evaluation
  03_tridep_gcn_demo.ipynb # Interactive demo notebook (TriDep + InducT-GCN)
  fusion.ipynb             # Decision-level ensemble experiments & evaluations
  Induct_gcn.ipynb         # InducT-GCN text graph neural network training
results/figures/           # Aggregate result plots (no patient data)
docs/                      # Project summary / abstract
```

## Quick Start — Run the Demo

### Online (Cloud)
Visit the live Hugging Face Space: **[TriDep Demo](https://huggingface.co/spaces/sameer-04062004/TriDep-Depression-Detection)**

### Locally (Windows)
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Double-click **`run_demo.bat`** (or run `python app.py`) to launch the local screening interface at `http://127.0.0.1:7860`.

Notebooks are designed for **Google Colab** with a mounted Google Drive containing your own licensed copy of DAIC-WOZ, organised as:

```
MyDrive/DAIC/
  {subject_id}_P/
    {subject_id}_AUDIO.wav
    {subject_id}_TRANSCRIPT.csv
    {subject_id}_CLNF_AUs.txt
  train_split_Depression_AVEC2017.csv
  dev_split_Depression_AVEC2017.csv
  full_test_split.csv
```

## Run Order

1. `01_preprocess.ipynb` — run once; extracts and caches features to Drive.
2. `02_train_evaluate.ipynb` — restores cache, trains both Baseline and Improved models, runs evaluation.
3. `03_demo.ipynb` — interactive demo using the trained model.

## Results Summary

| Model | Accuracy | F1 (depressed) | ROC-AUC |
|---|---|---|---|
| Baseline (LOSOCV) | ~0.70 | ~0.46 | ~0.62 |
| Improved (5-Fold CV) | see `results/figures/` | | |

Full evaluation, unimodal ablation, and discussion are provided in the project report (`docs/`).

## Acknowledgements

- DAIC-WOZ dataset: Gratch et al., "The Distress Analysis Interview Corpus of human and computer interviews," LREC 2014.
- Sentence-BERT: Reimers & Gurevych, 2019.
- Wav2Vec2: Baevski et al., Facebook AI, 2020.
- OpenFace: Baltrušaitis et al., 2016.

## License

Code in this repository is released under the MIT License (see `LICENSE`). This licence covers the code only — it does not grant any rights to the DAIC-WOZ dataset, which remains subject to its own licence terms.
