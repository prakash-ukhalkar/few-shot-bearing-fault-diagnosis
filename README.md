# Few-Shot Bearing Fault Diagnosis

A label-efficient fault diagnosis pipeline for rotating machinery. This project
compares standard supervised classifiers against few-shot and self-supervised
approaches across varying label-scarcity levels (5%, 10%, 25%, 50%, 100% of
available labels), targeting the empirical claim that few-shot / self-supervised
methods degrade more gracefully than fully-supervised baselines as labeled data
shrinks. Built as a research artifact for a submission to *Applied Intelligence*
(Springer).

## Project structure

```
few-shot-bearing-fault-diagnosis/
├── configs/                     # YAML experiment configs
│   ├── default.yaml             # main label-fraction sweep, same-condition split
│   └── cross_load.yaml          # cross-load generalization sweep
├── data/
│   ├── raw/                     # downloaded .mat files (gitignored)
│   └── processed/                # windowed .npz arrays (gitignored)
├── notebooks/                   # exploratory analysis notebooks
├── results/
│   ├── figures/                 # accuracy-vs-label-fraction plots, confusion matrices
│   ├── tables/                  # summary CSVs, significance test results
│   └── models/                  # trained model checkpoints (gitignored)
├── src/
│   ├── data/                    # CWRU loading, download, preprocessing, splitting
│   ├── features/                # hand-crafted time/spectral feature extraction
│   ├── models/                  # SVM/RF, 1D-CNN, LSTM, Prototypical Networks
│   ├── self_supervised/         # SimCLR-style contrastive + masked reconstruction
│   ├── evaluation/               # metrics, significance tests, plotting
│   └── utils/                   # seeding, config loading
├── tests/                       # unit tests (preprocessing, windowing, metrics, models)
├── train.py                     # train/evaluate one method @ one fraction/seed
├── run_all_experiments.py       # full grid sweep + aggregation + plots
├── requirements.txt             # pinned dependencies
└── .github/workflows/tests.yml  # CI: pytest on push/PR to main
```

## Quickstart

```bash
# 1. Create environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

# 2. Download the CWRU dataset subset (no credentials required)
python -m src.data.download_cwru --out data/raw/cwru

# 3. Build windowed train/test arrays
python -m src.data.build_dataset --raw data/raw/cwru --out data/processed \
    --window-size 1024 --stride 512 --channel DE

# 4. Run a single method/label-fraction/seed configuration
python train.py --config configs/default.yaml --method protonet \
    --label-fraction 0.1 --seed 0

# 5. Run the full experiment grid (all methods x label fractions x seeds)
python run_all_experiments.py --config configs/default.yaml

# 6. Cross-load generalization experiment (train on 0-2 HP, test on 3 HP)
python run_all_experiments.py --config configs/cross_load.yaml

# Run tests
pytest tests/ -v
```

All training runs on CPU by default (`training.device: cpu` in the configs);
backbones are kept small (<500K parameters) and pretraining/episodic training
budgets are sized to complete in minutes on a laptop CPU.

## Methods compared

- **SVM / Random Forest** on hand-crafted time-domain and spectral features
  (`src/features/handcrafted.py`) — classical supervised baseline.
- **1D-CNN** / **shallow LSTM** trained directly on raw windowed vibration
  signal — supervised deep baseline.
- **Prototypical Networks** (Snell et al., 2017) — metric-learning few-shot
  method; class prototypes are the mean embedding of the support set, query
  samples classified by nearest prototype. Main proposed method.
- **SimCLR-style contrastive pretraining** with signal-specific augmentations
  (jitter, scaling, time-warping) on unlabeled data, followed by a linear
  probe fine-tuned on the scarce labels. Second proposed method.
- **Masked-signal reconstruction** pretext task (`src/self_supervised/masked_reconstruction.py`)
  as an ablation against contrastive pretraining.

## Experimental design

- Each method is trained/evaluated at label fractions `[5%, 10%, 25%, 50%, 100%]`,
  5 seeds per configuration (`configs/default.yaml`), reporting mean ± std.
- Metrics: accuracy, macro-F1 (fault classes may be imbalanced), and per-class
  confusion matrices (`src/evaluation/metrics.py`, `src/evaluation/plots.py`).
- Statistical significance between methods at each label fraction is assessed
  with a paired t-test and Wilcoxon signed-rank test across seeds
  (`src/evaluation/significance.py`).
- Cross-load generalization: `configs/cross_load.yaml` trains on CWRU loads
  0-2 HP and evaluates on the held-out 3 HP condition, testing robustness
  beyond same-condition overfitting.
- The central figure is accuracy (and macro-F1) vs. label fraction, one curve
  per method with error bars across seeds
  (`src/evaluation/plots.py::plot_accuracy_vs_label_fraction`).

## Reproducibility & data availability

**Data sources (all free, no credentialed/paid access required):**

- **CWRU Bearing Dataset** — Case Western Reserve University Bearing Data
  Center (https://engineering.case.edu/bearingdatacenter). Files are served
  publicly with no login required. This project uses the **12 kHz Drive-End
  (DE) fault dataset**, fault diameters **0.007" / 0.014" / 0.021"**, fault
  location **OR@6:00** for outer-race faults, all four motor **load
  conditions (0-3 HP, approx. 1730-1797 RPM)**, plus the normal baseline —
  10 classes total. The exact file-number-to-label mapping is fixed and
  version-controlled in `src/data/cwru_manifest.py`, so re-running
  `src/data/download_cwru.py` deterministically reconstructs the identical
  raw dataset used in this project.
- **NASA C-MAPSS** turbofan degradation dataset (NASA Prognostics Data
  Repository) — optional secondary dataset for generalization experiments
  beyond CWRU; publicly downloadable with no credentials.
- **PRONOSTIA/FEMTO** bearing run-to-failure dataset — optional third dataset
  for cross-dataset generalization claims; publicly available.

**No raw data is committed to this repository** (`data/raw/`, `data/processed/`,
and `results/models/` are gitignored — raw CWRU files alone are tens of MB).
Instead, `src/data/download_cwru.py` and `src/data/build_dataset.py`
deterministically regenerate the exact processed dataset from the public
source, so a reviewer can reproduce the full pipeline from a clean checkout
with:

```bash
python -m src.data.download_cwru --out data/raw/cwru
python -m src.data.build_dataset --raw data/raw/cwru --out data/processed
python run_all_experiments.py --config configs/default.yaml
```

All dependency versions are pinned exactly in `requirements.txt` for
long-term reproducibility.

## License

MIT — see [LICENSE](LICENSE).
