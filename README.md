# LOVW-8.5K: handwriting decoding as a challenging motor task for EEG foundation models

This repository holds the code for the NeurIPS 2026 Evaluations & Datasets track paper *Handwriting decoding as a challenging motor task for EEG Foundation Models*. The dataset, LOVW-8.5K, is on Hugging Face: https://huggingface.co/datasets/Ouroboros/LOVW-8.5K

LOVW-8.5K has about 8.5K EEG trials from 4 participants. In each trial a participant either wrote one of the letters L, O, V or W (motor execution, ME) or imagined writing it (motor imagery, MI).

> **Status: first release.** Every number in the tables below was re-run from the public dataset with this code. Baselines not re-run yet are marked as such, and the remaining items are listed under [Not included yet](#not-included-yet).

## Setup

```bash
pip install -r requirements.txt               # Python 3.11, CUDA 12.4
python scripts/download_data.py               # paper sessions only, ~0.5 GB, into data/LOVW-8.5K
```

`--all` downloads the full dataset (~4.8 GB). `--data-root` changes the download location, and every script accepts the same `--data-root` flag.

## Reproducing the paper

| Paper result | Command |
|---|---|
| Table 1 and Fig. 3a: within-participant decoding | `python scripts/classify.py --model {eegnet,conformer,sparcnet,cbramod,reve} [--mode lp] --setting {me_movement,me_cue,mi_cue}` |
| Fig. 3b and the mixed-effects analysis: temporal jitter | `python scripts/jitter.py` |
| Fig. 4: training-set size vs. test-time SNR | `python scripts/snr_sample_complexity.py` |
| Fig. 5: MRCP at C1 and EEGNet Grad-CAM | `python scripts/interpretability.py` |
| Table 3 and the appendix MI table: leave-one-session-out | `python scripts/loso.py --setting {me_movement,mi_cue}` |
| Appendix, cross-subject transfer | `python scripts/cross_subject.py` |
| Appendix, permutation test | `python scripts/permutation_test.py <model A results> --vs <model B results>` |
| Summary tables, Figs. 3a, 3b and 4, reaction times | `python scripts/make_figures.py` |

Results are written to `results/` as JSON. The JSON files hold per-fold accuracies and per-trial predictions, so you can recompute the statistics.

### Re-run results

All runs use chronological 5-fold cross-validation. Accuracy is the mean over folds. In Table 1, ± is the SD across participants. Chance is 0.25.

**Table 1: ME movement-locked, participant average**

| Model | This code | Paper |
|---|---|---|
| EEGNet (3 seeds) | 0.403 ± 0.063 | 0.413 ± 0.041 |
| CBraMod, linear probe | 0.260 ± 0.019 | 0.266 ± 0.026 |
| CBraMod, full fine-tuning | 0.320 ± 0.036 | 0.308 ± 0.029 |
| SPaRCNet* | not re-run yet | 0.359 ± 0.052 |
| REVE (LP / FT) | not re-run, see below | 0.306 / 0.322 |

\*Reimplemented with braindecode's default architecture settings, because the original baseline configuration is not available.

**Fig. 4: P1, final 160 trials, all training data (3 seeds)**

| Test trials averaged | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| Accuracy | 0.47 | 0.52 | 0.65 | 0.80 |

The paper reports 45% on single trials and 78% when averaging 8 trials.

**Fig. 3b: P1, jitter half-width vs. accuracy without augmentation (5 seeds × 5 folds)**

| Jitter (s) | 0 | ±0.1 | ±0.2 | ±0.3 | ±0.5 |
|---|---|---|---|---|---|
| Accuracy | 0.47 | 0.37 | 0.30 | 0.27 | 0.25 |

Augmentation does not change the drop-off (condition × jitter interaction, p = 0.58).

**Table 3: leave-one-session-out, EEGNet, ME movement-locked (accuracy ± binomial SE)**

| | Session 1 | Session 2 | Session 3 | Session 4 |
|---|---|---|---|---|
| P1 | 0.311 ± 0.026 | 0.352 ± 0.018 | 0.326 ± 0.013 | 0.433 ± 0.045 |
| P2 | 0.289 ± 0.025 | 0.277 ± 0.024 | 0.302 ± 0.016 | 0.270 ± 0.025 |

For MI cue-locked, every held-out session scores 0.25–0.28, near chance, as in the paper's appendix table.

## What the code does

- **Data:** the paper uses the ICA-cleaned, 0.3–40 Hz, 100 Hz EEGLAB files of 17 sessions (`lovw/sessions.py`).
  1. The 3 tablet channels (`Vx`, `Vy`, `penDown`) are dropped.
  2. Cz, the online reference, is added back as an all-zero channel.
  3. The data are re-referenced to the common average, giving 32 channels.
- **Settings:** trials are centered on pen-down (ME movement-locked) or on the go cue (`X_fixation`, ME and MI cue-locked).
- **Input windows:**
  - Specialist models get [-200, +800] ms at 100 Hz, in µV.
  - Foundation models get [-1, +3] s at 200 Hz, in units of 100 µV.
- **Cross-validation:** trials stay in chronological order. Each of the 5 folds is a contiguous block of every session.
- **Training:**
  - Within the training folds, a random 20% split selects the best checkpoint.
  - Optimizer: Adam with a one-cycle learning-rate schedule and gradient clipping at 5.
  - Learning rates and epoch counts per model are set in `DEFAULTS` in `scripts/classify.py`.
- **CBraMod:** the model code is vendored in `lovw/third_party` under its MIT license. The official pretrained weights are downloaded from https://huggingface.co/weighting666/CBraMod.
- **REVE:** `--model reve` loads `brain-bzh/reve-base` from Hugging Face. This requires `trust_remote_code=True`, which runs the REVE authors' code. Its LP and FT modes use the same linear-head setup as CBraMod, not the two-stage PEFT procedure used in the paper. This path has not been re-run for this release.

## Not included yet

- **Conformer and SPaRCNet:** the paper's exact configurations; see the note under Table 1.
- **REVE:** the two-stage parameter-efficient fine-tuning used in the paper.
- **MIRepNet:** fine-tuned with the official MIRepNet repository and its own preprocessing.
- **Prior-dataset analyses:** everything on the 20-participant dataset of Crell et al. That covers the onset drop, REVE vs. EEGNet, and the subject-as-unit test, plus the confound analyses of prior work.

## Citation

```bibtex
@inproceedings{ravishankar2026handwriting,
  title={Handwriting decoding as a challenging motor task for {EEG} Foundation Models},
  author={Ravishankar, Srinivas and Ghosh, Ishayu and Zajzon, Nora and Fei, Teng and de Sa, Virginia},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS), Evaluations and Datasets Track},
  year={2026}
}
```

## License

The code is released under the MIT license (`LICENSE`). The vendored CBraMod code keeps its own MIT license (`lovw/third_party/LICENSE-CBraMod`). The dataset is released under CC BY 4.0.
