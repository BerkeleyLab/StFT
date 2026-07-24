"""Auto-discover finished train_graph.py runs and build the results table + figures.

Works with ANY subset of runs (masking, y25, etc.). Reads every
  <results_dir>/train_<variant>_<graph|nograph>/train_model_*/progress.csv
groups by variant, prints a summary table, and saves two PNGs next to the
results dir: results_bars.png (244-step test error at best-val checkpoint,
graph vs baseline) and results_curves.png (val-error trajectories).

Usage (Colab or local):
    python make_results_colab.py                 # scans ./ray_results
    python make_results_colab.py /path/to/ray_results  --out /path/to/save
"""

import argparse
import csv
import glob
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

GRAPH, NOGRAPH = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e7e6e2"


def discover(results_dir):
    """Return {variant: {cond: metrics_dict}} for every progress.csv found."""
    runs = {}
    for d in sorted(glob.glob(os.path.join(results_dir, "train_*"))):
        tag = os.path.basename(d)[len("train_"):]
        if tag.endswith("_nograph"):
            variant, cond = tag[:-len("_nograph")], "nograph"
        elif tag.endswith("_graph"):
            variant, cond = tag[:-len("_graph")], "graph"
        else:
            continue
        csvs = glob.glob(os.path.join(d, "train_model_*", "progress.csv"))
        if not csvs:
            continue
        rows = list(csv.DictReader(open(sorted(csvs)[0])))
        if not rows:
            continue
        f = lambda k: np.array([float(r[k]) for r in rows])
        runs.setdefault(variant, {})[cond] = dict(
            epoch=f("epoch"), val=f("val_error"), test=f("test_error"),
            best_val=f("best_val"), best_tuv=f("best_test_under_val"),
            best_test=f("best_test"),
            wall_min=(float(rows[-1].get("time_total_s") or 0)) / 60,
        )
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", nargs="?", default="ray_results")
    ap.add_argument("--out", default=None, help="dir to save PNGs (default: alongside results_dir)")
    args = ap.parse_args()
    out = args.out or os.path.dirname(os.path.abspath(args.results_dir)) or "."

    runs = discover(args.results_dir)
    if not runs:
        print(f"No runs found under {args.results_dir!r}. Did training write there?")
        return
    variants = list(runs)

    # ---------- table ----------
    print(f"{'variant':9s} {'cond':8s} {'epochs':>6s} {'wall(min)':>9s} | "
          f"{'best_test':>9s} {'best_test@val':>13s} {'test_last':>9s}")
    print("-" * 74)
    for v in variants:
        for c in ("graph", "nograph"):
            if c not in runs[v]:
                continue
            d = runs[v][c]
            print(f"{v:9s} {c:8s} {int(d['epoch'][-1]):6d} {d['wall_min']:9.1f} | "
                  f"{d['best_test'][-1]:9.4f} {d['best_tuv'][-1]:13.4f} {d['test'][-1]:9.3f}")
        if "graph" in runs[v] and "nograph" in runs[v]:
            g, n = runs[v]["graph"]["best_tuv"][-1], runs[v]["nograph"]["best_tuv"][-1]
            print(f"    -> best_test@val graph vs no-graph: {(n - g) / n * 100:+.0f}%\n")

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.linewidth": 0.8,
        "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": GRID,
        "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    })

    # ---------- bars (only variants with BOTH conditions) ----------
    paired = [v for v in variants if "graph" in runs[v] and "nograph" in runs[v]]
    if paired:
        fig, ax = plt.subplots(figsize=(9, 1.1 * len(paired) + 1.6), constrained_layout=True)
        y = np.arange(len(paired))[::-1]
        h = 0.36
        for i, v in enumerate(paired):
            g, n = runs[v]["graph"]["best_tuv"][-1], runs[v]["nograph"]["best_tuv"][-1]
            yg, yn = y[i] + h / 2 + 0.02, y[i] - h / 2 - 0.02
            ax.barh(yg, g, height=h, color=GRAPH, zorder=3, label="graph" if i == 0 else None)
            ax.barh(yn, n, height=h, color=NOGRAPH, zorder=3, label="no graph" if i == 0 else None)
            ax.text(g + 0.008, yg, f"{g:.3f}", va="center", fontsize=9, color=GRAPH, fontweight="bold")
            ax.text(n + 0.008, yn, f"{n:.3f}", va="center", fontsize=9, color=NOGRAPH, fontweight="bold")
            imp = (n - g) / n * 100
            ax.text(max(g, n) + 0.06, y[i], f"{imp:+.0f}%", va="center", fontsize=9,
                    color=(GRAPH if imp > 0 else NOGRAPH), fontweight="bold")
        ax.set_yticks(y); ax.set_yticklabels(paired, fontsize=10)
        ax.set_xlabel("244-step test error at best-val checkpoint  (lower is better)")
        ax.set_xlim(0, max(runs[v]["nograph"]["best_tuv"][-1] for v in paired) * 1.35)
        ax.grid(axis="x", color=GRID, lw=0.7, zorder=0); ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.legend(loc="lower right", frameon=False, fontsize=10)
        ax.set_title("Graph vs baseline — long-horizon test error", fontsize=12.5,
                     fontweight="bold", loc="left", pad=10, color=INK)
        fig.savefig(os.path.join(out, "results_bars.png"), dpi=150, bbox_inches="tight")
        print("saved", os.path.join(out, "results_bars.png"))

    # ---------- curves (one panel per variant) ----------
    n = len(variants)
    ncol = min(2, n)
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.5 * ncol, 3.6 * nrow),
                             constrained_layout=True, squeeze=False)
    for ax, v in zip(axes.flat, variants):
        for c, col in (("graph", GRAPH), ("nograph", NOGRAPH)):
            if c not in runs[v]:
                continue
            d = runs[v][c]
            ax.plot(d["epoch"], d["val"], color=col, lw=1.5,
                    label=("graph" if c == "graph" else "no graph"), zorder=3)
            ax.scatter([d["epoch"][-1]], [d["best_val"][-1]], s=24, color=col,
                       edgecolor="white", lw=1, zorder=4)
        ax.set_yscale("log")
        ax.set_title(v, fontsize=11, loc="left", color=INK, fontweight="bold")
        ax.grid(True, which="both", color=GRID, lw=0.6); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
        ax.set_xlabel("epoch", fontsize=9); ax.set_ylabel("val error", fontsize=9)
        ax.legend(frameon=False, fontsize=9, loc="upper right")
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    fig.suptitle("Validation-rollout error over training (dot = final best-val)",
                 fontsize=12.5, fontweight="bold", x=0.01, ha="left", color=INK)
    fig.savefig(os.path.join(out, "results_curves.png"), dpi=150, bbox_inches="tight")
    print("saved", os.path.join(out, "results_curves.png"))


if __name__ == "__main__":
    main()
