#!/usr/bin/env python3
"""Draw the E3 bias-variance figures (paper Figs. 2, 7 and 8) as PNG files.

Reads results/summary/E3_bias_variance.csv from exp/run_e3.py (default), or the authors' committed
curves (--source authors, exp/Bias_Variance_Decomp/RESULTS) for comparison. Writes to
<results>/figures/:
  fig2_bias_variance_sgbt_ailerons.png   Sgbt only on Ailerons, like paper Fig. 2
  fig7_bias_variance.png                 Ailerons, FriedmanLea, MetroTraffic, FriedmanGra, House8L, FriedmanGsg
  fig8_bias_variance.png                 Elevators, Abalone, Bike, Fried
Friedman* panels without runs of ours (their split files are not in the authors' repository) are
drawn from the authors' committed results on a shaded background labelled "authors' results, not
reproduced" (--no-fallback leaves them out); other panels without data are left out. Once the split
files are supplied (extra_data/) and run, those panels show our runs. --with-authors overlays the
authors' curves in grey on our panels.

Usage:
  python3 exp/plot_e3.py [--results results] [--source runs|authors] [--with-authors]
"""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from run_e3 import DRIFT, METHODS, authors_results  # noqa: E402

FIG7 = ["ailerons", "FriedmanLea", "MetroTraffic", "FriedmanGra", "House8L", "FriedmanGsg"]
FIG8 = ["elevators", "abalone", "bike", "fried"]
COLORS = {"B": "#2a78d6", "BB": "#eb6834", "BagB": "#1baf7a"}  # reference palette slots 1-3
COMPONENTS = {"sqrt_mean_bias": ("--", "√bias"), "sqrt_mean_var": ("-", "√variance")}
INK, INK_2, GRID, AXIS, SURFACE, REF = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#ffffff", "#898781"
BORROWED = "#f0efec"  # panel background for datasets shown from the authors' results only

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9.5, "text.color": INK, "axes.labelcolor": INK_2,
    "axes.edgecolor": AXIS, "xtick.color": INK_2, "ytick.color": INK_2, "axes.titlesize": 10.5,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def read_runs(path):
    curves = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            curves[(row["dataset"], row["method"], int(row["boosting_iterations"]))] = {
                k: float(row[k]) for k in COMPONENTS}
    return curves


def series(curves, dataset, method, key):
    points = sorted((b, v[key]) for (d, m, b), v in curves.items() if d == dataset and m == method)
    return [p[0] for p in points], [p[1] for p in points]


def draw(curves, reference, datasets, methods, title, path, columns, fallback=None):
    """fallback: curves drawn, on a shaded panel, for datasets that have no runs of ours."""
    fallback = fallback or {}
    ours = {d for d in datasets if any(k[0] == d for k in curves)}
    borrowed = {d for d in datasets if d not in ours and any(k[0] == d for k in fallback)}
    present = [d for d in datasets if d in ours or d in borrowed]
    if not ours:
        return []
    columns = min(columns, len(present))
    rows = math.ceil(len(present) / columns)
    width, height = max(5.6, 4.1 * columns), 3.0 * rows + 0.9
    fig, axes = plt.subplots(rows, columns, figsize=(width, height), squeeze=False)
    for ax, dataset in zip(axes.flat, present):
        own = dataset in ours
        for method in methods:
            for key, (style, _) in COMPONENTS.items():
                if reference and own:
                    x, y = series(reference, dataset, method, key)
                    ax.plot(x, y, color=REF, lw=3.5, alpha=0.35, linestyle=style, zorder=1)
                x, y = series(curves if own else fallback, dataset, method, key)
                ax.plot(x, y, color=COLORS[method], lw=1.8 if own else 1.4, alpha=1.0 if own else 0.6,
                        linestyle=style, zorder=2)
        scaled = " (scaled)" if dataset.startswith("Friedman") else ""
        ax.set_title(f"{dataset}{scaled}", loc="left", fontweight="bold")
        if not own:
            ax.set_facecolor(BORROWED)
            ax.text(0.98, 0.04, "authors' results, not reproduced", transform=ax.transAxes, ha="right",
                    va="bottom", fontsize=8, color=INK_2, style="italic")
        ax.set_xlabel("Boosting iterations")
        ax.set_xlim(0, 102)
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for ax in list(axes.flat)[len(present):]:
        ax.axis("off")
    handles = [Line2D([], [], color=COLORS[m], lw=1.8, label=METHODS[m][0]) for m in methods]
    handles += [Line2D([], [], color=INK_2, lw=1.4, linestyle=s, label=label) for s, label in COMPONENTS.values()]
    if reference:
        handles.append(Line2D([], [], color=REF, lw=3.5, alpha=0.35, label="authors' results"))
    if borrowed:
        handles.append(Patch(facecolor=BORROWED, edgecolor=AXIS, lw=0.6,
                             label="shaded: authors' results (data not public)"))
    # Title and legend take a fixed 0.5 inch band at the top, whatever the figure height.
    fig.suptitle(title, x=0.01, y=1 - 0.08 / height, ha="left", va="top", fontweight="bold", fontsize=11)
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1 - 0.32 / height),
               ncol=len(handles), frameon=False, handlelength=2.2, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.5 / height))
    fig.savefig(path)
    plt.close(fig)
    return present


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, help="Default: <results>/figures")
    parser.add_argument("--source", choices=["runs", "authors"], default="runs")
    parser.add_argument("--with-authors", action="store_true", help="Overlay the authors' curves in grey")
    parser.add_argument("--no-fallback", action="store_true",
                        help="Leave out Friedman* panels without runs instead of showing the authors' results")
    parser.add_argument("--toolchain", type=Path, default=Path.home() / "sgbr" / "toolchain.json")
    args = parser.parse_args(argv)
    out = args.out or args.results / "figures"
    out.mkdir(parents=True, exist_ok=True)
    upstream = Path(json.loads(args.toolchain.read_text())["upstream"])
    authors = authors_results(upstream, FIG7 + FIG8)
    fallback = {}
    if args.source == "authors":
        curves, reference, suffix, who = authors, {}, "_authors", "authors' results"
    else:
        summary = args.results / "summary" / "E3_bias_variance.csv"
        if not summary.is_file():
            parser.error(f"{summary} not found; run exp/run_e3.py first")
        curves, reference, suffix, who = read_runs(summary), authors if args.with_authors else {}, "", "reproduction"
        # Friedman* panels that cannot be rerun (split files not public) are shown from the authors' results,
        # on a shaded background, unless --no-fallback.
        if not args.no_fallback:
            fallback = {k: v for k, v in authors.items() if k[0] in DRIFT}
    made = {
        "fig2": draw(curves, reference, ["ailerons"], ["B"],
                     f"Sgbt bias and variance against boosting iterations ({who})",
                     out / f"fig2_bias_variance_sgbt_ailerons{suffix}.png", 1),
        "fig7": draw(curves, reference, FIG7, list(METHODS),
                     f"Bias and variance against boosting iterations, paper Fig. 7 datasets ({who})",
                     out / f"fig7_bias_variance{suffix}.png", 3, fallback),
        "fig8": draw(curves, reference, FIG8, list(METHODS),
                     f"Bias and variance against boosting iterations, paper Fig. 8 datasets ({who})",
                     out / f"fig8_bias_variance{suffix}.png", 2),
    }
    for name, datasets in made.items():
        print(f"{name}: {', '.join(datasets) if datasets else 'no data yet'}")
    print("Figures in", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
