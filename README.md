# Multi-Class ALS Diagnosis (Few-Shot + SSL Layer Fusion)

Companion **code + summarized results** for the paper:
**“Multi-Class ALS Diagnosis from Speech via a Hybrid Few-Shot Learning Framework with Layer-Wise Fusion of Self-Supervised Models.”**

This repository intentionally **does not include any private/raw audio data** or the large extracted embedding matrices.

## What’s in this repo
- `Main/`: experiment notebooks/scripts used to produce the paper results
- `Results/`: **paper-ready summaries** (CSV/JSON/MD) for each experiment
- `Embeddings_Extraction/`: optional notebooks used to extract embeddings (requires access to audio data)

## What’s NOT in this repo
- Raw datasets (Korean corpus, VOC-ALS audio)
- Large extracted embeddings (`.npy`), intermediate caches, or model checkpoints

Embeddings are released separately (see “Embeddings release” below).

## Quick start (recommended)
1. Create an environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Download the released embeddings + split files from the separate embeddings repository (https://github.com/Alishah2046/ALS_Diagnosis_Meta-ALS_SSL_Fusion_Embeddings_Release.git).
3. Run the notebooks in `Main/` (paths are currently written for a local research workspace; adjust paths at the top of each notebook/script).

## Reproducing paper results
The **already-generated** summary files are under `Results/`:
- Layer fusion (main table): `Results/Layer_Analysis/experiment_summary_subject_fulltest.csv`
- Utterance-type analysis: `Results/Uttrance_Analysis/experiment_summary_utterance_types_wav2vec_stability.csv`
- Gender analysis: `Results/Gender_Analysis/experiment_summary_gender_fair_wav2vec_stability.csv`
- Feature comparison (OpenSMILE / MFCC): `Results/Feature-comparison/`
- Baselines (LR/SVM/RF + CNN): `Results/SOTA_models/`
- VOC-ALS external evaluation: `Results/VOC_ALS/OURS.json` and `Results/VOC_ALS/BASELINE.json`

## Embeddings release
The paper contribution includes releasing the extracted **multi-layer fusion embeddings** (HuBERT-XLarge, Wav2Vec2-Large, Data2vec-Audio-Large, WavLM-Large) together with **subject-disjoint split info (seed 42)**.

For public sharing, host embeddings in a separate repository/archive (Git LFS, GitHub Releases, Zenodo, OSF, HuggingFace, etc.) and link it here.

## Privacy note
Do **not** publish original filenames, subject identifiers, or raw recordings. This repo contains only aggregated/summarized outputs suitable for public release.

