"""One-off generator: builds notebooks/colab_run_experiments.ipynb by
embedding every project source file as a %%writefile cell, followed by
setup/run cells. Not part of the package; run manually to regenerate the
notebook after editing source files."""
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent

FILES = [
    "requirements.txt",
    "configs/default.yaml",
    "configs/cross_load.yaml",
    "src/__init__.py",
    "src/data/__init__.py",
    "src/data/preprocessing.py",
    "src/data/cwru.py",
    "src/data/cwru_manifest.py",
    "src/data/download_cwru.py",
    "src/data/build_dataset.py",
    "src/data/splits.py",
    "src/features/__init__.py",
    "src/features/handcrafted.py",
    "src/models/__init__.py",
    "src/models/classical.py",
    "src/models/cnn1d.py",
    "src/models/lstm.py",
    "src/models/prototypical.py",
    "src/models/train_utils.py",
    "src/models/episodic.py",
    "src/self_supervised/__init__.py",
    "src/self_supervised/augmentations.py",
    "src/self_supervised/simclr.py",
    "src/self_supervised/masked_reconstruction.py",
    "src/evaluation/__init__.py",
    "src/evaluation/metrics.py",
    "src/evaluation/significance.py",
    "src/evaluation/plots.py",
    "src/utils/__init__.py",
    "src/utils/config.py",
    "src/utils/seeding.py",
    "tests/__init__.py",
    "tests/test_preprocessing.py",
    "tests/test_metrics.py",
    "tests/test_splits.py",
    "tests/test_features_and_models.py",
    "train.py",
    "run_all_experiments.py",
]


def md_cell(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def code_cell(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


def writefile_cell(rel_path: str, content: str) -> dict:
    header = f"%%writefile {rel_path}\n"
    return code_cell(header + content)


def build():
    cells = []

    cells.append(md_cell(
        "# Few-Shot Bearing Fault Diagnosis — Colab Runner\n\n"
        "Self-contained notebook: writes out the full project source tree, installs pinned\n"
        "dependencies, downloads the public CWRU bearing dataset, builds windowed features,\n"
        "and runs the full label-fraction experiment grid (SVM/RF, 1D-CNN, LSTM, Prototypical\n"
        "Networks, SimCLR contrastive pretraining) — all CPU-only, no GPU required.\n\n"
        "**Recommended runtime:** Runtime > Change runtime type > CPU (GPU not required; the\n"
        "code hardcodes `device: cpu` in the configs, though you can edit that to `cuda` if a\n"
        "GPU runtime is available — it will speed up the CNN/ProtoNet/SimCLR methods).\n\n"
        "**Persisting results across disconnects:** Colab's local disk is ephemeral. Run the\n"
        "'Mount Google Drive' cell below and point `results/` there so a disconnect doesn't\n"
        "lose your progress — the experiment runner checkpoints every completed run to\n"
        "`results/tables/raw_results.jsonl` and **skips already-completed configs on rerun**,\n"
        "so you can safely restart this notebook after any disconnect."
    ))

    cells.append(md_cell("## 0. (Optional) Mount Google Drive for persistent results"))
    cells.append(code_cell(
        "# Uncomment to persist results/ across Colab disconnects.\n"
        "# from google.colab import drive\n"
        "# drive.mount('/content/drive')\n"
        "# import os\n"
        "# os.makedirs('/content/drive/MyDrive/few-shot-bearing-fault-diagnosis', exist_ok=True)\n"
        "# %cd /content/drive/MyDrive/few-shot-bearing-fault-diagnosis\n"
    ))

    cells.append(md_cell("## 1. Create project directory structure"))
    cells.append(code_cell(
        "import os\n\n"
        "dirs = [\n"
        "    'configs', 'data/raw', 'data/processed', 'results/figures',\n"
        "    'results/tables', 'results/models',\n"
        "    'src', 'src/data', 'src/features', 'src/models', 'src/self_supervised',\n"
        "    'src/evaluation', 'src/utils', 'tests',\n"
        "]\n"
        "for d in dirs:\n"
        "    os.makedirs(d, exist_ok=True)\n"
        "print('Directory structure created.')\n"
    ))

    cells.append(md_cell("## 2. Write out all project source files"))
    for rel_path in FILES:
        full = ROOT / rel_path
        content = full.read_text(encoding="utf-8")
        cells.append(writefile_cell(rel_path, content))

    cells.append(md_cell("## 3. Install pinned dependencies"))
    cells.append(code_cell("!pip install -q -r requirements.txt\n"))

    cells.append(md_cell(
        "## 4. Download the CWRU Bearing Dataset\n\n"
        "Public data, no credentials required — 12 kHz Drive-End fault subset (10 classes,\n"
        "4 load conditions), served directly by the Case Western Reserve University Bearing\n"
        "Data Center."
    ))
    cells.append(code_cell("!python -m src.data.download_cwru --out data/raw/cwru\n"))

    cells.append(md_cell("## 5. Build windowed dataset (preprocessing + segmentation)"))
    cells.append(code_cell(
        "!python -m src.data.build_dataset --raw data/raw/cwru --out data/processed "
        "--window-size 1024 --stride 512 --channel DE\n"
    ))

    cells.append(md_cell("## 6. Run unit tests (sanity check before the full sweep)"))
    cells.append(code_cell("!pytest tests/ -v\n"))

    cells.append(md_cell(
        "## 7. Run the full experiment grid\n\n"
        "6 methods x 5 label fractions x 5 seeds = 150 runs. This is the long-running cell —\n"
        "expect on the order of 1.5-3 hours on Colab's default CPU. Progress is checkpointed\n"
        "incrementally to `results/tables/raw_results.jsonl`; if this cell is interrupted\n"
        "(disconnect, runtime restart), just rerun it — completed configs are skipped\n"
        "automatically."
    ))
    cells.append(code_cell("!python run_all_experiments.py --config configs/default.yaml\n"))

    cells.append(md_cell(
        "## 8. (Optional) Cross-load generalization experiment\n\n"
        "Trains on CWRU loads 0-2 HP, evaluates on the held-out 3 HP condition."
    ))
    cells.append(code_cell("!python run_all_experiments.py --config configs/cross_load.yaml\n"))

    cells.append(md_cell("## 9. View results"))
    cells.append(code_cell(
        "import pandas as pd\n"
        "from IPython.display import Image, display\n\n"
        "acc_summary = pd.read_csv('results/tables/accuracy_summary.csv')\n"
        "display(acc_summary)\n\n"
        "f1_summary = pd.read_csv('results/tables/f1_macro_summary.csv')\n"
        "display(f1_summary)\n\n"
        "display(Image('results/figures/accuracy_vs_label_fraction.png'))\n"
        "display(Image('results/figures/f1_macro_vs_label_fraction.png'))\n"
    ))

    cells.append(code_cell(
        "# Paired significance tests between methods at each label fraction\n"
        "sig = pd.read_csv('results/tables/accuracy_significance.csv')\n"
        "display(sig)\n"
    ))

    cells.append(md_cell(
        "## 10. Download results locally\n\n"
        "If you didn't mount Drive in step 0, zip and download the results folder here."
    ))
    cells.append(code_cell(
        "!zip -rq results.zip results/\n"
        "from google.colab import files\n"
        "files.download('results.zip')\n"
    ))

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
            "colab": {"provenance": [], "name": "colab_run_experiments.ipynb"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    out_path = ROOT / "notebooks" / "colab_run_experiments.ipynb"
    out_path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"Wrote {out_path} ({len(cells)} cells)")


if __name__ == "__main__":
    build()
