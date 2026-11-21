#!/usr/bin/env python3
"""Draw the E1 figures as PNG files for slides and reports.

From the runs of exp/run_matrix.py (default, --source runs):
  fig4_shaffer_r2.png        like paper Fig. 4: average ranks on adjusted R² and Shaffer post-hoc test
  fig5_shaffer_walltime.png  like paper Fig. 5: the same on wall time
  r2_vs_paper.png            our mean R̄² against paper Tab. 3, one dot per method x dataset
  r2_delta_heatmap.png       our mean R̄² minus paper Tab. 3, method x dataset
  avg_rank_vs_paper.png      average rank per method, ours and paper Tab. 3
  walltime_ratio.png         our wall time / paper Tab. 4, per method across datasets

From the authors' committed per-seed results (--source post-hoc, exp/post_hoc_new):
  fig4_shaffer_r2_authors.png, fig5_shaffer_walltime_authors.png, to check this implementation
  against the paper's own figures.

The post-hoc test follows the paper's Figs. 4-5: six methods, Friedman ranks with rank 1 for the
largest value (so for wall time rank 1 is the slowest and "higher is better"), and Shaffer's static
procedure at alpha = 0.05 on the pairwise z statistics. Methods joined by a thick bar are not
significantly different. By default each (dataset, seed) pair is one block, as in the authors'
post-hoc input layout (one folder per dataset and seed).

Usage:
  python3 exp/plot_e1.py [--results results] [--out results/figures] [--source runs|post-hoc]
"""

import argparse
import csv
import io
import json
import math
import statistics
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

from summarize_e1 import DATASETS, METHODS, PIN, SHORT, TAB3, TAB4  # noqa: E402

# The six methods of paper Figs. 4-5: (label in the figure, run_matrix method key, post_hoc_new file).
FIG_METHODS = [
    ("Oza(SGBT)", "Oza(SGBT(FIRTDD))", "Oza(SGBT)"),
    ("SGB(Oza)", "SGBT(Oza(FIRTDD))", "SGBT(Oza)"),
    ("SGB(ARF)", "SGBT(ARFReg(FIRTDD))", "SGBT(ARF)"),
    ("SOKNL", "SOKNL(FIRTDD)", "SOKNL"),
    ("OzaReg", "Oza(FIRTDD)", "OzaReg"),
    ("ARFReg", "ARFReg(FIRTDD)", "ARFReg"),
]
POST_HOC_DATASETS = DATASETS  # all 16; the authors' post_hoc_new has the SUP* sets too
REPO_DATASETS = [d for d in DATASETS if not d.startswith("SUP")]  # ARFFs shipped in the authors' repo
R2 = "adjusted coefficient of determination"
WALL = "evaluation time (wall) (seconds)"

INK, INK_2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, SURFACE = "#2a78d6", "#eb6834", "#ffffff"
DIVERGING = LinearSegmentedColormap.from_list("delta", ["#184f95", "#6da7ec", "#f0efec", "#ec8a89", "#a32a29"])

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10, "text.color": INK, "axes.labelcolor": INK_2,
    "axes.edgecolor": AXIS, "xtick.color": INK_2, "ytick.color": INK_2, "axes.titlesize": 11,
    "axes.titleweight": "bold", "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.dpi": 200, "savefig.bbox": "tight", "savefig.facecolor": SURFACE,
})


# ----------------------------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------------------------

def block_ranks(values):
    """Ranks within one block, 1 = largest value, ties get the average rank."""
    order = sorted(range(len(values)), key=lambda i: -values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for t in range(i, j + 1):
            ranks[order[t]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def chi2_sf(x, df):
    """Upper tail of the chi-square distribution (regularised upper incomplete gamma)."""
    a, z = df / 2.0, x / 2.0
    if z <= 0:
        return 1.0
    log_prefix = a * math.log(z) - z - math.lgamma(a)
    if z < a + 1:  # series for the lower tail
        term = total = 1.0 / a
        n = a
        while abs(term) > abs(total) * 1e-15:
            n += 1
            term *= z / n
            total += term
        return max(0.0, 1.0 - total * math.exp(log_prefix))
    b, c, d = z + 1 - a, 1e300, 1.0 / (z + 1 - a)  # continued fraction for the upper tail
    h = d
    for i in range(1, 10000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
        c = b + an / c
        c = c if abs(c) > 1e-300 else 1e-300
        h *= d * c
        if abs(d * c - 1) < 1e-15:
            break
    return math.exp(log_prefix) * h


def shaffer_counts(k):
    """S(k): the possible numbers of true hypotheses among the k(k-1)/2 pairwise comparisons."""
    memo = {0: {0}, 1: {0}}

    def s(n):
        if n not in memo:
            memo[n] = {math.comb(j, 2) + x for j in range(1, n + 1) for x in s(n - j)}
        return memo[n]

    return s(k)


def shaffer_test(blocks, alpha=0.05):
    """Friedman test, then Shaffer's static procedure on all pairs. blocks: list of value lists."""
    n, k = len(blocks), len(blocks[0])
    sums = [0.0] * k
    for block in blocks:
        for i, r in enumerate(block_ranks(block)):
            sums[i] += r
    avg = [s / n for s in sums]
    friedman = 12 * n / (k * (k + 1)) * (sum(r * r for r in avg) - k * (k + 1) ** 2 / 4)
    se = math.sqrt(k * (k + 1) / (6 * n))
    pairs = []
    for i in range(k):
        for j in range(i + 1, k):
            z = abs(avg[i] - avg[j]) / se
            pairs.append({"i": i, "j": j, "z": z, "p": math.erfc(z / math.sqrt(2))})
    pairs.sort(key=lambda p: p["p"])
    m, counts, running = len(pairs), shaffer_counts(k), 0.0
    for index, pair in enumerate(pairs, 1):
        pair["t"] = max(c for c in counts if c <= m - index + 1)
        running = max(running, min(1.0, pair["t"] * pair["p"]))
        pair["p_adjusted"] = running
        pair["significant"] = running <= alpha
    return {"blocks": n, "avg_ranks": avg, "friedman_chi2": friedman,
            "friedman_p": chi2_sf(friedman, k - 1), "pairs": pairs}


def cliques(order, significant):
    """Maximal runs of consecutive methods (in rank order) with no significant pair among them."""
    runs = []
    for a in range(len(order)):
        b = a
        while b + 1 < len(order) and all(not significant[frozenset((order[x], order[b + 1]))]
                                         for x in range(a, b + 1)):
            b += 1
        if b > a and not any(s <= a and b <= e for s, e in runs):
            runs.append((a, b))
    return runs


# ----------------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------------

def load_runs(results):
    runs = {}
    for path in (results / "raw" / "E1").glob("*/*/seed=*/record.json"):
        record = json.loads(path.read_text())
        if record.get("status") == "completed":
            runs[(record["method_key"], record["dataset"], record["seed"])] = record
    return runs


def blocks_from_runs(runs, metric, per_dataset):
    # The six methods of Figs. 4-5 are all seeded; unseeded runs (seed None) never form a block.
    keys = sorted({(d, s) for (_, d, s) in runs if s is not None})
    rows = {}
    for dataset, seed in keys:
        found = [runs.get((key, dataset, seed)) for _, key, _ in FIG_METHODS]
        if all(found):
            rows.setdefault(dataset, []).append([r[metric] for r in found])
    if per_dataset:
        return [[statistics.fmean(c) for c in zip(*rows[d])] for d in sorted(rows)], sorted(rows)
    return [b for d in sorted(rows) for b in rows[d]], sorted(rows)


def blocks_from_post_hoc(upstream, column, datasets, per_dataset):
    rows = {}
    for dataset in datasets:
        for seed in range(1, 6):
            values = []
            for _, _, file in FIG_METHODS:
                try:
                    text = subprocess.check_output(
                        ["git", "-C", str(upstream), "show", f"{PIN}:exp/post_hoc_new/{dataset}_{seed}/{file}.csv"],
                        stderr=subprocess.DEVNULL, text=True)
                except subprocess.CalledProcessError:
                    break
                values.append(float(list(csv.DictReader(io.StringIO(text)))[-1][column]))
            if len(values) == len(FIG_METHODS):
                rows.setdefault(dataset, []).append(values)
    if per_dataset:
        return [[statistics.fmean(c) for c in zip(*rows[d])] for d in rows], list(rows)
    return [b for d in rows for b in rows[d]], list(rows)


# ----------------------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------------------

def draw_posthoc(test, labels, title, note, path):
    """Rank diagram in the style of paper Figs. 4-5."""
    k, avg = len(labels), test["avg_ranks"]
    order = sorted(range(k), key=lambda i: avg[i])
    significant = {frozenset((p["i"], p["j"])): p["significant"] for p in test["pairs"]}
    runs = cliques(order, significant)
    left, right = order[:math.ceil(k / 2)], order[math.ceil(k / 2):]
    bar_gap, row_gap = 0.16, 0.34
    first_row = 0.35 + bar_gap * len(runs)
    rows = max(len(left), len(right))
    height = first_row + row_gap * rows + 0.1
    fig, ax = plt.subplots(figsize=(7.2, 0.9 + 0.62 * height))
    margin = 0.32 * (k - 1)
    ax.set_xlim(1 - margin, k + margin)
    ax.set_ylim(height, -0.55)
    ax.axis("off")
    ax.plot([1, k], [0, 0], color=INK, lw=1.2)
    for tick in range(1, k + 1):
        ax.plot([tick, tick], [0, -0.12], color=INK, lw=1.2)
        ax.text(tick, -0.18, str(tick), ha="center", va="bottom", fontsize=10)
    for tick in range(1, k):
        ax.plot([tick + 0.5, tick + 0.5], [0, -0.06], color=INK, lw=0.8)
    for side, members, edge, align in (("L", left, 1 - margin * 0.25, "right"),
                                       ("R", right, k + margin * 0.25, "left")):
        for row, method in enumerate(members):
            y = first_row + row_gap * row
            ax.plot([avg[method], avg[method], edge], [0, y, y], color=INK, lw=1.0)
            ax.text(edge + (-0.05 if side == "L" else 0.05), y, labels[method], ha=align, va="center", fontsize=10)
            # The average rank sits on the horizontal line, next to the label, clear of the bars.
            ax.text(edge + (0.06 if side == "L" else -0.06), y - 0.04, f"{avg[method]:.2f}",
                    ha="left" if side == "L" else "right", va="bottom", fontsize=7.5, color=MUTED)
    for index, (a, b) in enumerate(runs):
        y = 0.2 + bar_gap * index
        ax.plot([avg[order[a]] - 0.03, avg[order[b]] + 0.03], [y, y], color=INK, lw=3.2, solid_capstyle="butt")
    ax.set_title(title, loc="left", pad=6)
    fig.text(0.01, 0.0, note, fontsize=8, color=INK_2, va="top", ha="left", wrap=True)
    fig.savefig(path)
    plt.close(fig)


def mean_by(runs, key):
    values = {}
    for (method, dataset, _), record in runs.items():
        values.setdefault((method, dataset), []).append(record[key])
    return {k: statistics.fmean(v) for k, v in values.items()}


def draw_r2_scatter(r2, path):
    points = []
    for paper, key, _ in METHODS:
        for i, dataset in enumerate(DATASETS):
            ours = r2.get((key, dataset))
            if ours is not None and not (paper == "Axgb" and dataset == "NZEnergy"):
                points.append((TAB3[paper][i], ours, paper, SHORT[i]))
    if not points:
        return False
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    low = min(-0.1, min(min(p[0], p[1]) for p in points) - 0.05)
    ax.fill_between([low, 1.02], [low - 0.03, 0.99], [low + 0.03, 1.05], color=GRID, alpha=0.7, lw=0,
                    label="within ±0.03 of the paper")
    ax.plot([low, 1.02], [low, 1.02], color=AXIS, lw=1)
    ax.scatter([p[0] for p in points], [p[1] for p in points], s=36, color=BLUE, edgecolor=SURFACE,
               linewidth=1.2, zorder=3, label="method × dataset")
    for x, y, method, dataset in points:
        if abs(y - x) > 0.10:
            ax.annotate(f"{method}, {dataset}", (x, y), xytext=(5, -3), textcoords="offset points",
                        fontsize=7.5, color=INK_2)
    ax.set_xlim(low, 1.02)
    ax.set_ylim(low, 1.02)
    ax.set_aspect("equal")
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("Paper Tab. 3, mean adjusted R²")
    ax.set_ylabel("This reproduction, mean adjusted R²")
    ax.set_title(f"E1 adjusted R²: reproduction vs paper ({len(points)} method × dataset cells)", loc="left")
    ax.legend(loc="upper left", frameon=False, fontsize=8.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.savefig(path)
    plt.close(fig)
    return True


def draw_delta_heatmap(r2, path):
    methods = [m for m, key, _ in METHODS if any((key, d) in r2 for d in DATASETS)]
    if not methods:
        return False
    keys = {paper: key for paper, key, _ in METHODS}
    # The shipped datasets always get a column; SUP* only once supplied and run.
    columns = [d for d in DATASETS if d in REPO_DATASETS or any((keys[m], d) in r2 for m in methods)]
    grid, notes = [], []
    for m in methods:
        row, text = [], []
        for d in columns:
            i = DATASETS.index(d)
            ours = r2.get((keys[m], d))
            if ours is None or (m == "Axgb" and d == "NZEnergy"):
                row.append(math.nan)
                text.append("–" if ours is None else "n/a")
            else:
                delta = ours - TAB3[m][i]
                row.append(delta)
                text.append(f"{delta:+.2f}" if abs(delta) >= 0.01 else "")
        grid.append(row)
        notes.append(text)
    fig, ax = plt.subplots(figsize=(10.5, 0.42 * len(methods) + 1.6))
    image = ax.imshow(grid, cmap=DIVERGING, vmin=-0.1, vmax=0.1, aspect="auto")
    image.cmap.set_bad(GRID)
    for r, text in enumerate(notes):
        for c, t in enumerate(text):
            value = grid[r][c]
            dark = not math.isnan(value) and abs(value) > 0.06
            ax.text(c, r, t, ha="center", va="center", fontsize=7.5, color=SURFACE if dark else INK_2)
    ax.set_xticks(range(len(columns)), [SHORT[DATASETS.index(d)] for d in columns], rotation=35, ha="right")
    ax.set_yticks(range(len(methods)), methods)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([x - 0.5 for x in range(1, len(columns))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(methods))], minor=True)
    ax.grid(which="minor", color=SURFACE, lw=2)
    ax.tick_params(which="minor", length=0)
    bar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.01)
    bar.set_label("R̄² reproduction − paper (clipped at ±0.10)")
    bar.outline.set_visible(False)
    ax.set_title("E1: difference from paper Tab. 3 (numbers shown where |Δ| ≥ 0.01; grey = not run yet)", loc="left")
    fig.savefig(path)
    plt.close(fig)
    return True


def draw_avg_rank(r2, path):
    keys = {paper: key for paper, key, _ in METHODS}
    methods = [m for m, key, _ in METHODS if any((key, d) in r2 for d in DATASETS)]
    done = [d for d in DATASETS if methods and all((keys[m], d) in r2 for m in methods)]
    if len(methods) < 2 or not done:
        return False
    ours = {m: [] for m in methods}
    paper = {m: [] for m in methods}
    for d in done:
        i = DATASETS.index(d)
        for m, r in zip(methods, block_ranks([r2[(keys[m], d)] for m in methods])):
            ours[m].append(r)
        for m, r in zip(methods, block_ranks([TAB3[m][i] for m in methods])):
            paper[m].append(r)
    rows = sorted(methods, key=lambda m: statistics.fmean(paper[m]))
    fig, ax = plt.subplots(figsize=(6.8, 0.36 * len(rows) + 1.4))
    for y, m in enumerate(rows):
        a, b = statistics.fmean(paper[m]), statistics.fmean(ours[m])
        ax.plot([a, b], [y, y], color=AXIS, lw=2, zorder=1)
        ax.scatter([a], [y], s=60, color=ORANGE, edgecolor=SURFACE, linewidth=1.2, zorder=3,
                   label="Paper Tab. 3" if y == 0 else None)
        ax.scatter([b], [y], s=60, color=BLUE, edgecolor=SURFACE, linewidth=1.2, zorder=3,
                   label="This reproduction" if y == 0 else None)
    ax.set_yticks(range(len(rows)), rows)
    ax.invert_yaxis()
    ax.set_xlim(0.5, len(methods) + 0.5)
    ax.set_xlabel("Average rank on adjusted R² (1 = best)")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.set_title(f"E1 average rank, {len(methods)} methods on {len(done)} datasets", loc="left", pad=22)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=8.5,
              borderaxespad=0.2, handletextpad=0.3)
    fig.savefig(path)
    plt.close(fig)
    return True


def draw_wall_ratio(wall, path):
    keys = {paper: key for paper, key, _ in METHODS}
    rows = [m for m, key, _ in METHODS if any((key, d) in wall for d in DATASETS)]
    if not rows:
        return False
    fig, ax = plt.subplots(figsize=(6.8, 0.36 * len(rows) + 1.4))
    everything = [wall[(keys[m], d)] / TAB4[m][i] for m in rows for i, d in enumerate(DATASETS)
                  if (keys[m], d) in wall]
    for y, m in enumerate(rows):
        ratios = [wall[(keys[m], d)] / TAB4[m][i] for i, d in enumerate(DATASETS) if (keys[m], d) in wall]
        ax.scatter(ratios, [y] * len(ratios), s=28, color=BLUE, alpha=0.75, edgecolor=SURFACE, linewidth=0.8,
                   zorder=3, label="one dataset" if y == 0 else None)
        median = statistics.median(ratios)
        ax.plot([median, median], [y - 0.3, y + 0.3], color=INK, lw=2, zorder=4,
                label="median" if y == 0 else None)
    ax.axvline(1, color=AXIS, lw=1, zorder=1)
    ax.set_xscale("log")
    low, high = min(0.8, min(everything) / 1.25), max(1.25, max(everything) * 1.25)
    ax.set_xlim(low, high)
    ax.set_xticks([t for t in (0.1, 0.25, 0.5, 1, 2, 3, 5, 10, 20, 50, 100) if low <= t <= high])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}×"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks(range(len(rows)), rows)
    ax.invert_yaxis()
    ax.set_xlabel("Wall time: reproduction / paper Tab. 4 (log scale, 1 = same as the paper)")
    ax.grid(axis="x", color=GRID, lw=0.6, which="both")
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.set_title("E1 wall time relative to the paper", loc="left", pad=22)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=8.5,
              borderaxespad=0.2, handletextpad=0.3)
    fig.savefig(path)
    plt.close(fig)
    return True


def posthoc_figures(blocks_r2, blocks_wall, datasets, suffix, alpha, out, unit):
    labels = [label for label, _, _ in FIG_METHODS]
    report = {}
    for name, blocks, title, direction in (
            ("fig4_shaffer_r2", blocks_r2, "Shaffer post-hoc test, p = 0.05, adjusted R²", "Lower is better"),
            ("fig5_shaffer_walltime", blocks_wall, "Shaffer post-hoc test, p = 0.05, wall time",
             "Higher is better (rank 1 = slowest)")):
        if not blocks:
            continue
        test = shaffer_test(blocks, alpha)
        note = (f"{direction}. {test['blocks']} blocks ({unit}) from {len(datasets)} datasets. Friedman χ² = "
                f"{test['friedman_chi2']:.1f}, p = {test['friedman_p']:.1e}. Thick bars join methods that are not "
                f"significantly different (Shaffer's static procedure, α = {alpha}).")
        draw_posthoc(test, labels, title, note, out / f"{name}{suffix}.png")
        report[name] = {
            "datasets": datasets, "blocks": test["blocks"], "friedman_chi2": test["friedman_chi2"],
            "friedman_p": test["friedman_p"],
            "avg_ranks": dict(zip(labels, test["avg_ranks"])),
            "pairs": [{"a": labels[p["i"]], "b": labels[p["j"]], "z": p["z"], "p": p["p"],
                       "p_shaffer": p["p_adjusted"], "significant": p["significant"]} for p in test["pairs"]],
        }
    (out / f"posthoc{suffix}.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, help="Default: <results>/figures")
    parser.add_argument("--source", choices=["runs", "post-hoc"], default="runs")
    parser.add_argument("--toolchain", type=Path, default=Path.home() / "sgbr" / "toolchain.json",
                        help="Supplies --upstream for --source post-hoc")
    parser.add_argument("--upstream", type=Path)
    parser.add_argument("--blocks", choices=["dataset-seed", "dataset"], default="dataset-seed",
                        help="Post-hoc blocks: each (dataset, seed) run, or the per-dataset mean over seeds")
    parser.add_argument("--post-hoc-datasets", choices=["all", "repo"], default="all",
                        help="With --source post-hoc: all 16 datasets, or only the 13 shipped in the repo")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args(argv)
    out = args.out or args.results / "figures"
    out.mkdir(parents=True, exist_ok=True)
    per_dataset = args.blocks == "dataset"
    unit = "per-dataset means" if per_dataset else "dataset × seed runs"

    if args.source == "post-hoc":
        upstream = args.upstream
        if upstream is None and args.toolchain.is_file():
            upstream = Path(json.loads(args.toolchain.read_text())["upstream"])
        if upstream is None:
            parser.error("Run exp/toolchain.py first, or pass --upstream")
        datasets = POST_HOC_DATASETS if args.post_hoc_datasets == "all" else REPO_DATASETS
        blocks_r2, found = blocks_from_post_hoc(upstream, R2, datasets, per_dataset)
        blocks_wall, _ = blocks_from_post_hoc(upstream, WALL, datasets, per_dataset)
        report = posthoc_figures(blocks_r2, blocks_wall, found, "_authors", args.alpha, out, unit)
    else:
        runs = load_runs(args.results)
        if not runs:
            parser.error(f"No completed runs under {args.results / 'raw' / 'E1'}")
        blocks_r2, found = blocks_from_runs(runs, "r2", per_dataset)
        blocks_wall, _ = blocks_from_runs(runs, "moa_wall_seconds", per_dataset)
        report = posthoc_figures(blocks_r2, blocks_wall, found, "", args.alpha, out, unit) if blocks_r2 else {}
        r2, wall = mean_by(runs, "r2"), mean_by(runs, "moa_wall_seconds")
        draw_r2_scatter(r2, out / "r2_vs_paper.png")
        draw_delta_heatmap(r2, out / "r2_delta_heatmap.png")
        draw_avg_rank(r2, out / "avg_rank_vs_paper.png")
        draw_wall_ratio(wall, out / "walltime_ratio.png")
    for name, item in report.items():
        ranks = ", ".join(f"{k} {v:.2f}" for k, v in sorted(item["avg_ranks"].items(), key=lambda kv: kv[1]))
        print(f"{name}: {item['blocks']} blocks; ranks {ranks}")
    print("Figures in", out)
    for path in sorted(out.glob("*.png")):
        print(" ", path.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
