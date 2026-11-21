#!/usr/bin/env python3
"""Summarise and draw E2: behaviour over time on hyperA (paper Fig. 6, p. 18).

E2 runs the authors' exp/exp_config_overTime.json: Sgb(Oza), SOKNL, ArfReg and OzaReg on hyperA
(abrupt drifts at 125k, 250k and 375k instances), seeds 1-3, windowed evaluator (-w 10000) with a
row every 10,000 instances. As in the authors' exp/StatsOverTime.py, each point is the mean over
seeds of the windowed adjusted R² and of the cumulative wall time.

Writes:
  <results>/summary/E2_over_time.csv   method, instances, seeds, mean windowed R̄², mean wall time
  <results>/summary/E2_drift.md        windowed R̄² just before each drift and its lowest value in
                                       the 50,000 instances after it (our summary, not a paper table)
  <results>/figures/fig6_over_time.png

Usage:
  python3 exp/plot_e2.py [--results results]
"""

import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from plot_e1 import AXIS, GRID, INK_2, MUTED, R2, WALL  # noqa: E402  (also applies the shared style)

SEEN = "learning evaluation instances"
DRIFTS = [125_000, 250_000, 375_000]
AFTER = 50_000
# (run_matrix method key, label as in paper Fig. 6, colour slot, dash pattern as secondary encoding)
E2_METHODS = [
    ("SGBT(Oza(FIRTDD))", "Sgb(Oza)", "#2a78d6", "solid"),
    ("SOKNL(FIRTDD)", "SOKNL", "#eb6834", (0, (4, 1.5))),
    ("ARFReg(FIRTDD)", "ArfReg", "#1baf7a", (0, (6, 1.5, 1, 1.5))),
    ("Oza(FIRTDD)", "OzaReg", "#eda100", (0, (1, 1.2))),
]


def load_runs(results, dataset="hyperA"):
    """{method: {seed: [(instances, r2, wall), ...]}} for completed E2 runs."""
    runs = {}
    for record_path in sorted((results / "raw" / "E2" / dataset).glob("*/seed=*/record.json")):
        record = json.loads(record_path.read_text())
        if record.get("status") != "completed":
            continue
        with (record_path.parent / "evaluation.csv").open(newline="", encoding="utf-8") as handle:
            rows = [(int(float(r[SEEN])), float(r[R2]), float(r[WALL])) for r in csv.DictReader(handle)]
        runs.setdefault(record["method_key"], {})[record["seed"]] = rows
    return runs


def mean_curves(runs):
    """{method: [(instances, seeds, mean r2, mean wall)]}, only at instance counts all seeds reached."""
    curves = {}
    for method, seeds in runs.items():
        points = {}
        for rows in seeds.values():
            for seen, r2, wall in rows:
                points.setdefault(seen, []).append((r2, wall))
        curves[method] = [(seen, len(v), statistics.fmean(p[0] for p in v), statistics.fmean(p[1] for p in v))
                          for seen, v in sorted(points.items()) if len(v) == len(seeds)]
    return curves


def drift_rows(curves):
    rows = []
    for method, label, _, _ in E2_METHODS:
        curve = curves.get(method)
        if not curve:
            continue
        for drift in DRIFTS:
            # Rows come every 10,000 instances, so take the last window that ends at or before the drift.
            before = [(seen, r2) for seen, _, r2, _ in curve if seen <= drift]
            after = [(r2, seen) for seen, _, r2, _ in curve if drift < seen <= drift + AFTER]
            if not before or not after:
                continue
            ends, level = before[-1]
            low, at = min(after)
            rows.append((label, drift, ends, level, low, at, level - low))
    return rows


def draw(curves, seeds, path):
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(8.6, 6.2), sharex=True)
    lowest = min(r2 for curve in curves.values() for _, _, r2, _ in curve)
    for method, label, color, dash in E2_METHODS:
        curve = curves.get(method)
        if not curve:
            continue
        seen = [p[0] for p in curve]
        width = 2.0 if method.startswith("SGBT") else 1.5
        top.plot(seen, [p[2] for p in curve], color=color, lw=width, linestyle=dash, zorder=3)
        bottom.plot(seen, [p[3] for p in curve], color=color, lw=width, linestyle=dash, zorder=3)
    floor = -1.0
    if lowest < floor:
        top.set_ylim(bottom=floor)
        top.text(1.0, 0.02, f"values below {floor:g} are cut off (lowest {lowest:.2f})", transform=top.transAxes,
                 ha="right", va="bottom", fontsize=8, color=MUTED)
    for ax in (top, bottom):
        for drift in DRIFTS:
            ax.axvline(drift, color=AXIS, lw=1.0, linestyle=(0, (3, 3)), zorder=1)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for drift in DRIFTS:
        top.annotate("drift", (drift, 1.0), xycoords=("data", "axes fraction"), xytext=(3, -2),
                     textcoords="offset points", ha="left", va="top", fontsize=8, color=MUTED)
    top.set_ylabel("Windowed R̄² (last 10,000)")
    bottom.set_ylabel("Wall time (s), cumulative")
    bottom.set_xlabel("Instances seen")
    bottom.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x / 1000:.0f}k"))
    handles = [Line2D([], [], color=color, lw=2.0 if key.startswith("SGBT") else 1.5, linestyle=dash,
                      label=label if len(set(seeds.values())) == 1 else f"{label} ({seeds[key]} seeds)")
               for key, label, color, dash in E2_METHODS if key in curves]
    counts = set(seeds.values())
    mean_of = f"mean of {counts.pop()} seeds" if len(counts) == 1 else "mean over seeds"
    fig.suptitle(f"hyperA over time, {mean_of} (like paper Fig. 6)",
                 x=0.01, ha="left", fontweight="bold", fontsize=11, y=0.995)
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 0.955), ncol=len(handles), frameon=False,
               handlelength=3.0)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, help="Default: <results>/figures")
    args = parser.parse_args(argv)
    runs = load_runs(args.results)
    curves = mean_curves(runs)
    if not curves:
        print("E2: no completed runs yet")
        return 0
    summary = args.results / "summary"
    out = args.out or args.results / "figures"
    summary.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    labels = {key: label for key, label, _, _ in E2_METHODS}
    with (summary / "E2_over_time.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["method", "instances", "seeds", "mean_windowed_adj_r2", "mean_wall_seconds"])
        for method, curve in curves.items():
            for seen, n, r2, wall in curve:
                writer.writerow([labels.get(method, method), seen, n, f"{r2:.6f}", f"{wall:.3f}"])
    lines = ["# E2 hyperA: windowed R̄² around the drifts", "",
             f"Mean over seeds. 'Before' is the last window (10,000 instances) ending at or before the drift; "
             f"'lowest after' is the lowest window in the next {AFTER:,} instances. This is a summary of our runs, "
             "not a table from the paper.", "",
             "| Method | Drift at | Before (window ends) | Lowest after (window ends) | Drop |", "|---|---|---|---|---|"]
    lines += [f"| {label} | {drift:,} | {level:.3f} ({ends:,}) | {low:.3f} ({at:,}) | {drop:.3f} |"
              for label, drift, ends, level, low, at, drop in drift_rows(curves)]
    (summary / "E2_drift.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    draw(curves, {method: len(seeds) for method, seeds in runs.items()}, out / "fig6_over_time.png")
    print("E2:", ", ".join(f"{labels.get(m, m)} ({len(c)} points)" for m, c in curves.items()))
    print("Wrote", summary / "E2_over_time.csv", summary / "E2_drift.md", out / "fig6_over_time.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
