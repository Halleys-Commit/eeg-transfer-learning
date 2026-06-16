# EEG Transfer Learning — Population Priors & Meta-Learning for Motor Imagery BCI

Can a neural network trained on 101 people decode a new person's imagined hand movement
with almost no calibration data? This project tests two strategies for building that
transferable prior — standard population pre-training and Reptile meta-learning — and
quantifies how much per-subject calibration is actually needed.

Companion to [neuro-data-explorer](https://github.com/Halleys-Commit/neuro-data-explorer),
which established the cross-subject baseline this project extends.

Dataset: [PhysioNet EEGBCI](https://physionet.org/content/eegmmidb/1.0.0/) (Schalk et al., 2004) —
109 subjects, 64 channels, 160 Hz, Task 2 (imagined left vs right fist). 6 excluded after
quality screening; 102 retained throughout.

---

## The question

The neuro-data-explorer series ended with two findings that point directly here:

1. **65.1% LOSO** — the best cross-subject model (EA + EEGNet, N=102) is already strong.
2. **57% LORO** — within-subject fine-tuning on 2 calibration runs *underperforms* cross-subject LOSO.

The gap means a population prior exists and is useful, but naïve within-subject fine-tuning
doesn't exploit it well. Transfer learning is the proposed fix.

---

## Notebooks

| # | Notebook | Goal | Key result |
|---|---|---|---|
| 1 | [NB1 — Population pre-training](NB1_population_pretraining.ipynb) | Train EEGNet on 101 subjects LOSO with Euclidean Alignment | **73.1% ± 12.4%** mean LOSO (vs 65.1% baseline, +8 pp) |
| 2 | [NB2 — Fine-tuning & evaluation](NB2_finetune_evaluation.ipynb) | Fine-tune NB1 checkpoint on 2 calibration runs; compare to Riemannian MDM | FT **73.4%**, MDM 58.1% (+15.3 pp, p < 0.0001); FT vs zero-shot not significant |
| 3 | [NB3 — Reptile meta-training](NB3_reptile_metatraining.ipynb) | Replace standard pre-training with Reptile; compare all methods | Zero-shot **79.7% ± 13.5%**; fine-tuning the Reptile prior *hurts* (p = 0.002) |

---

## Results

```
Method                              Mean accuracy   vs standard FT
────────────────────────────────────────────────────────────────────
Reptile zero-shot (NB3)             79.7 ± 13.5%   +6.3 pp   ← best result
Standard fine-tune (NB2)            73.4 ± 16.1%   reference
Reptile fine-tuned (NB3)            75.7 ± 14.7%   +2.3 pp   p = 0.090
NB1 zero-shot (population prior)    72.6 ± 15.7%   −0.9 pp
MDM baseline (Riemannian, NB2)      58.1 ± 13.6%   −15.3 pp  p < 0.0001
────────────────────────────────────────────────────────────────────
Calibration budget (3–12 trials/class): accuracy flat for both fine-tuned methods
```

**Unexpected finding:** The Reptile zero-shot (79.7%) outperforms Reptile fine-tuning (75.7%,
p = 0.002). The meta-initialization is already well-positioned for motor imagery; 30 epochs
of fine-tuning on ~14 trials/class moves weights away from the optimal basin rather than
refining them. This is characteristic MAML-family behavior — the init is optimized for rapid
adaptation, so it sits close to a good solution for *any* subject before a single gradient step.

---

## Design

- **EEGNet** throughout (F1=8, D=2, F2=16, kernel=64, dropout=0.25) — compact, interpretable,
  well-validated for BCI.
- **Euclidean Alignment** per-subject, derived from calibration runs only (no leakage from test run
  into the alignment matrix).
- **Reptile** (Nichol et al., 2018) over MAML — no second-order gradients, ~3× cheaper, competitive
  on structured task domains where subjects are not wildly diverse.
- **Calibration budget sweep** (3–12 trials/class) maps the clinically actionable question: how much
  user burden is actually needed? Answer: essentially none for the Reptile zero-shot.

---

## Run

```bash
pip install mne braindecode pyriemann scikit-learn scipy matplotlib torch
# Data downloads automatically from PhysioNet on first run
jupyter notebook NB1_population_pretraining.ipynb
```

NB1 requires ~2–3 h and a GPU (CPU fallback is available but slow). NB2 and NB3 load cached
NB1 results and take 30–90 min each with a GPU.

---

## References

Schalk et al. (2004). BCI2000: A General-Purpose Brain-Computer Interface System.
*IEEE Trans. Biomed. Eng.* 51(6):1034–1043.

Goldberger et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet.
*Circulation* 101(23):e215–e220.

Nichol, Achiam & Schulman (2018). On First-Order Meta-Learning Algorithms. *arXiv:1803.02999*.

Lawhern et al. (2018). EEGNet: A Compact Convolutional Neural Network for EEG-Based
Brain–Computer Interfaces. *J. Neural Eng.* 15(5):056013.
