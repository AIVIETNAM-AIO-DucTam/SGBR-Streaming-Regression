#!/usr/bin/env python3
"""Summarise and draw the Sgb(Oza) hyperparameter grids E4 and E5 on the 11 core datasets.

E4 (paper Fig. 9, p. 20): learning rate {0.5, 1.0, 1.5} x feature % {25, 75, 100}, S = 10, E = 10
    (exp/exp_config_SGBT_OZA_parameter_search_LR_FP.json).
E5 (paper Tabs. 5-6, pp. 22-23, and Fig. 10): boosting steps S x bag size E with S x E = 100,
    learning rate 1.0, 75 % features (exp/exp_config_SGBT_OZA_parameter_search.json).

Aggregation follows the authors' exp/generateSummaryXls.py: every row of a run's evaluation.csv
(MOA writes one every 100,000 instances and one at the end) is averaged per configuration and
dataset over all seeds; "avg" is the mean over datasets, and "avgRank" the mean of the per-dataset
ranks (rank 1 = highest R̄² or lowest wall time, ties averaged). Only hyperA (500,000 instances) has
more than one row, so only its value differs from the final-row value, which is also kept in the CSV.
Averages are taken only over datasets where every configuration has completed runs.

Writes, for each experiment with results:
  <results>/summary/E4_grid.csv, E4_table.md   per configuration x dataset, avg, avgRank, paper avg
  <results>/summary/E5_grid.csv, E5_table.md
  <results>/figures/fig9_lr_feature.png        avg R̄² and avg wall time per (learning rate, feature %)
  <results>/figures/fig10_steps_bagsize.png    avg R̄² and avg wall time per (S, E)

Usage:
  python3 exp/plot_e4_e5.py [--results results]
"""

import argparse
import csv
import json
import re
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from plot_e1 import BLUE, GRID, INK_2, MUTED, R2, WALL, block_ranks  # noqa: E402  (also applies the style)

CORE11 = ["fried", "House8L", "abalone", "bike", "elevators", "MetroTraffic", "ailerons",
          "FriedmanLea", "FriedmanGra", "FriedmanGsg", "hyperA"]
SHORT = dict(zip(CORE11, ["fried", "House8L", "abalone", "bike", "elev.", "Metro", "ailer.",
                          "FrLea", "FrGra", "FrGsg", "hyperA"]))
PAPER_GREY = "#898781"

# E4: paper Fig. 9 (p. 20), average mean R̄² printed on the figure, by (learning rate, feature %).
E4_PAPER_R2 = {(0.5, 25): -6.69, (0.5, 75): -6.91, (0.5, 100): -6.99,
               (1.0, 25): 0.65, (1.0, 75): 0.74, (1.0, 100): 0.74,
               (1.5, 25): -0.30, (1.5, 75): -0.17, (1.5, 100): -0.16}
# E5: paper Tab. 5 (p. 22) avg R̄² and avgRank, Tab. 6 (p. 23) avg wall time (s), by (S, E).
E5_PAPER = {(1, 100): (0.55, 7.1, 59.0), (2, 50): (0.69, 4.9, 57.1), (4, 25): (0.74, 2.6, 55.0),
            (5, 20): (0.74, 2.1, 54.1), (10, 10): (0.74, 2.4, 52.2), (20, 5): (0.70, 4.1, 51.0),
            (25, 4): (0.67, 5.2, 49.4), (50, 2): (0.51, 7.7, 44.4), (100, 1): (0.10, 9.0, 50.9)}

E4_NAME = re.compile(r"SGBT_10_([\d.]+)_(\d+)\(Oza_10\)")
E5_NAME = re.compile(r"SGBT_(\d+)\(Oza_(\d+)\)")


def parse_config(exp, method):
    if exp == "E4":
        lr, feat = E4_NAME.fullmatch(method).groups()
        return float(lr), int(feat)
    steps, bag = E5_NAME.fullmatch(method).groups()
    return int(steps), int(bag)


def config_label(exp, config):
    return f"{config[0]:g}, {config[1]}%" if exp == "E4" else f"({config[0]}, {config[1]})"


def load_cells(results, exp):
    """{(config, dataset): {"all": [(r2, wall) for every row of every seed], "final": [...], "seeds": n}}."""
    cells = {}
    for record_path in sorted((results / "raw" / exp).glob("*/*/seed=*/record.json")):
        record = json.loads(record_path.read_text())
        if record.get("status") != "completed":
            continue
        with (record_path.parent / "evaluation.csv").open(newline="", encoding="utf-8") as handle:
            rows = [(float(r[R2]), float(r[WALL])) for r in csv.DictReader(handle)]
        cell = cells.setdefault((parse_config(exp, record["method_key"]), record["dataset"]),
                                {"all": [], "final": [], "seeds": 0})
        cell["all"] += rows
        cell["final"].append(rows[-1])
        cell["seeds"] += 1
    return cells


def table(cells, configs):
    """Per configuration: values per dataset, avg and avgRank over the datasets complete for all configs."""
    complete = [d for d in CORE11 if all((c, d) in cells for c in configs)]
    value = {(c, d): (statistics.fmean(r[0] for r in cells[(c, d)]["all"]),
                      statistics.fmean(r[1] for r in cells[(c, d)]["all"])) for (c, d) in cells}
    rows = {c: {"r2": {}, "wall": {}, "r2_rank": [], "wall_rank": []} for c in configs}
    for (c, d), (r2, wall) in value.items():
        if c in rows:
            rows[c]["r2"][d], rows[c]["wall"][d] = r2, wall
    for d in complete:
        r2_ranks = block_ranks([value[(c, d)][0] for c in configs])
        wall_ranks = block_ranks([-value[(c, d)][1] for c in configs])  # rank 1 = fastest
        for c, a, b in zip(configs, r2_ranks, wall_ranks):
            rows[c]["r2_rank"].append(a)
            rows[c]["wall_rank"].append(b)
    for c in configs:
        row = rows[c]
        row["avg_r2"] = statistics.fmean(row["r2"][d] for d in complete) if complete else None
        row["avg_wall"] = statistics.fmean(row["wall"][d] for d in complete) if complete else None
        row["rank_r2"] = statistics.fmean(row["r2_rank"]) if complete else None
        row["rank_wall"] = statistics.fmean(row["wall_rank"]) if complete else None
    return rows, complete


def fmt(value, places):
    return "–" if value is None else f"{value:.{places}f}"


def write_outputs(exp, cells, configs, rows, complete, summary):
    paper = {c: (E4_PAPER_R2.get(c), None, None) for c in configs} if exp == "E4" else E5_PAPER
    with (summary / f"{exp}_grid.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["config", "dataset", "seeds", "adj_r2_authors_mean", "adj_r2_final_row",
                         "wall_seconds_authors_mean", "wall_seconds_final_row"])
        for c in configs:
            for d in CORE11:
                if (c, d) in cells:
                    cell = cells[(c, d)]
                    writer.writerow([config_label(exp, c), d, cell["seeds"], f"{rows[c]['r2'][d]:.6f}",
                                     f"{statistics.fmean(r[0] for r in cell['final']):.6f}",
                                     f"{rows[c]['wall'][d]:.3f}", f"{statistics.fmean(r[1] for r in cell['final']):.3f}"])
    head = "(lr, feat %)" if exp == "E4" else "(S, E)"
    source = "Fig. 9 (p. 20)" if exp == "E4" else "Tab. 5 (p. 22) and Tab. 6 (p. 23)"
    lines = [f"# {exp}: Sgb(Oza) grid on the core datasets", "",
             f"Mean over seeds, averaged over every evaluation row as in the authors' generateSummaryXls.py. "
             f"avg and avgRank use the {len(complete)} datasets complete for every configuration"
             f"{': ' + ', '.join(complete) if complete else ''}. Paper values: {source}.", "",
             "## Adjusted R̄²", "",
             f"| {head} | " + " | ".join(SHORT[d] for d in CORE11) + " | avg | avgRank | paper avg | paper avgRank |",
             "|---" * (len(CORE11) + 5) + "|"]
    for c in configs:
        r = rows[c]
        lines.append(f"| {config_label(exp, c)} | " + " | ".join(fmt(r["r2"].get(d), 2) for d in CORE11)
                     + f" | {fmt(r['avg_r2'], 2)} | {fmt(r['rank_r2'], 1)} | {fmt(paper[c][0], 2)} | {fmt(paper[c][1], 1)} |")
    lines += ["", "## Wall time (s)", "",
              f"| {head} | " + " | ".join(SHORT[d] for d in CORE11) + " | avg | avgRank | paper avg |",
              "|---" * (len(CORE11) + 4) + "|"]
    for c in configs:
        r = rows[c]
        lines.append(f"| {config_label(exp, c)} | " + " | ".join(fmt(r["wall"].get(d), 1) for d in CORE11)
                     + f" | {fmt(r['avg_wall'], 1)} | {fmt(r['rank_wall'], 1)} | {fmt(paper[c][2], 1)} |")
    if exp == "E5":
        lines += ["", "Paper wall times were measured on different hardware (App. A.1: Windows, Ryzen 7 3700X); "
                      "compare their pattern, not the seconds."]
    (summary / f"{exp}_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def draw(exp, configs, rows, complete, path):
    x = list(range(len(configs)))
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(8.6, 6.0), sharex=True)
    r2 = [rows[c]["avg_r2"] for c in configs]
    wall = [rows[c]["avg_wall"] for c in configs]
    top.plot(x, r2, color=BLUE, lw=2.0, marker="o", ms=6, zorder=3)
    bottom.plot(x, wall, color=BLUE, lw=2.0, marker="o", ms=6, zorder=3)
    if exp == "E4":
        paper_r2 = [E4_PAPER_R2[c] for c in configs]
        paper_wall = None
    else:
        paper_r2 = [E5_PAPER[c][0] for c in configs]
        paper_wall = [E5_PAPER[c][2] for c in configs]
    for i, (value, paper) in enumerate(zip(r2, paper_r2)):
        above = value > paper  # put our value on the side away from the paper's marker
        top.annotate(f"{value:.2f}", (i, value), xytext=(0, 9 if above else -9), textcoords="offset points",
                     ha="center", va="bottom" if above else "top", fontsize=8.5, color=INK_2)
    top.margins(y=0.15)
    top.plot(x, paper_r2, linestyle="none", marker="o", ms=10, mfc="none", mec=PAPER_GREY, mew=1.5, zorder=2)
    if paper_wall:
        bottom.plot(x, paper_wall, linestyle="none", marker="o", ms=10, mfc="none", mec=PAPER_GREY, mew=1.5, zorder=2)
    for ax in (top, bottom):
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    top.set_ylabel("Average R̄²")
    bottom.set_ylabel("Average wall time (s)")
    if exp == "E4":
        bottom.set_xticks(x, [f"{c[1]}%" for c in configs])
        bottom.set_xlabel("Feature %", labelpad=24)
        for start in range(0, len(configs), 3):
            for ax in (top, bottom):
                if start:
                    ax.axvline(start - 0.5, color=GRID, lw=1.0)
            group = [i for i, c in enumerate(configs) if c[0] == configs[start][0]]
            bottom.annotate(f"learning rate {configs[start][0]:.1f}", (statistics.fmean(group), 0),
                            xycoords=("data", "axes fraction"), xytext=(0, -22), textcoords="offset points",
                            ha="center", va="top", fontsize=9, fontweight="bold", color=INK_2)
        title = "Sgb(Oza) (10, 10): learning rate x feature % (like paper Fig. 9)"
        paper_label = "paper Fig. 9"
    else:
        bottom.set_xticks(x, [config_label(exp, c) for c in configs])
        bottom.set_xlabel("(Boosting steps S, bag size E), S x E = 100")
        title = "Sgb(Oza): boosting steps x bag size (like paper Tabs. 5-6, Fig. 10)"
        paper_label = "paper Tab. 5 / Tab. 6 (other hardware)"
    handles = [Line2D([], [], color=BLUE, lw=2.0, marker="o", ms=6, label="reproduction"),
               Line2D([], [], linestyle="none", marker="o", ms=10, mfc="none", mec=PAPER_GREY, mew=1.5,
                      label=paper_label)]
    datasets = "1 dataset" if len(complete) == 1 else f"{len(complete)} datasets"
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold", fontsize=11, y=0.995)
    fig.text(0.01, 0.955, f"Mean over {datasets} and all seeds", ha="left", va="top", fontsize=9, color=MUTED)
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.0, 0.965), ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, help="Default: <results>/figures")
    args = parser.parse_args(argv)
    summary, out = args.results / "summary", args.out or args.results / "figures"
    for exp, order, figure in (("E4", E4_PAPER_R2, "fig9_lr_feature.png"),
                               ("E5", E5_PAPER, "fig10_steps_bagsize.png")):
        cells = load_cells(args.results, exp)
        configs = [c for c in order if any(k[0] == c for k in cells)]
        if not configs:
            print(f"{exp}: no completed runs yet")
            continue
        summary.mkdir(parents=True, exist_ok=True)
        out.mkdir(parents=True, exist_ok=True)
        rows, complete = table(cells, configs)
        write_outputs(exp, cells, configs, rows, complete, summary)
        if complete:
            draw(exp, configs, rows, complete, out / figure)
        print(f"{exp}: {len(configs)} configurations, {len(cells)} configuration x dataset cells, "
              f"{len(complete)} datasets complete for all"
              + ("" if complete else " (no figure until one dataset is complete for every configuration)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
