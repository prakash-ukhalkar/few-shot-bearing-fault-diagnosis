#!/usr/bin/env python
"""Run the full experiment grid: all methods x all label fractions x all
seeds, then aggregate results, run significance tests, and produce the
central accuracy-vs-label-fraction figure plus summary tables.

Usage:
    python run_all_experiments.py --config configs/default.yaml
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.plots import plot_accuracy_vs_label_fraction
from src.evaluation.significance import paired_ttest, summarize_across_seeds
from src.utils.config import load_config
from train import main as train_main
import sys


def run_grid(cfg, config_path):
    methods = cfg["experiment"]["methods"]
    fractions = cfg["experiment"]["label_fractions"]
    seeds = cfg["experiment"]["seeds"]

    records = []
    for method in methods:
        for fraction in fractions:
            for seed in seeds:
                print(f"=== method={method} fraction={fraction} seed={seed} ===")
                argv_backup = sys.argv
                sys.argv = [
                    "train.py", "--config", config_path, "--method", method,
                    "--label-fraction", str(fraction), "--seed", str(seed),
                ]
                try:
                    result = train_main()
                    records.append(result)
                except Exception as e:
                    print(f"FAILED: {method} fraction={fraction} seed={seed}: {e}")
                finally:
                    sys.argv = argv_backup
    return records


def aggregate(records, metric="accuracy"):
    grouped = defaultdict(lambda: defaultdict(list))
    for r in records:
        grouped[r["method"]][r["label_fraction"]].append(r[metric])

    summary = {}
    for method, per_fraction in grouped.items():
        summary[method] = {}
        for fraction, scores in per_fraction.items():
            summary[method][fraction] = summarize_across_seeds(np.array(scores))
    return summary, grouped


def run_significance_tests(grouped, metric="accuracy"):
    methods = list(grouped.keys())
    rows = []
    for i, m1 in enumerate(methods):
        for m2 in methods[i + 1 :]:
            common_fractions = set(grouped[m1].keys()) & set(grouped[m2].keys())
            for fraction in sorted(common_fractions):
                s1, s2 = grouped[m1][fraction], grouped[m2][fraction]
                if len(s1) == len(s2) and len(s1) > 1:
                    stat, p = paired_ttest(np.array(s1), np.array(s2))
                    rows.append(
                        {"method_a": m1, "method_b": m2, "label_fraction": fraction, "t_stat": stat, "p_value": p}
                    )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_cfg = cfg["output"]
    Path(out_cfg["results_dir"]).mkdir(parents=True, exist_ok=True)
    Path(out_cfg["tables_dir"]).mkdir(parents=True, exist_ok=True)
    Path(out_cfg["figures_dir"]).mkdir(parents=True, exist_ok=True)

    records = run_grid(cfg, args.config)

    raw_path = Path(out_cfg["tables_dir"]) / "raw_results.json"
    with open(raw_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Saved raw results to {raw_path}")

    for metric in ["accuracy", "f1_macro"]:
        summary, grouped = aggregate(records, metric=metric)
        plot_accuracy_vs_label_fraction(
            summary, metric=metric,
            out_path=Path(out_cfg["figures_dir"]) / f"{metric}_vs_label_fraction.png",
        )

        rows = []
        for method, per_fraction in summary.items():
            for fraction, stats in per_fraction.items():
                rows.append({"method": method, "label_fraction": fraction, **stats})
        df = pd.DataFrame(rows).sort_values(["method", "label_fraction"])
        table_path = Path(out_cfg["tables_dir"]) / f"{metric}_summary.csv"
        df.to_csv(table_path, index=False)
        print(f"Saved {metric} summary table to {table_path}")

        sig_df = run_significance_tests(grouped, metric=metric)
        sig_path = Path(out_cfg["tables_dir"]) / f"{metric}_significance.csv"
        sig_df.to_csv(sig_path, index=False)
        print(f"Saved {metric} significance tests to {sig_path}")

    print("Done.")


if __name__ == "__main__":
    main()
