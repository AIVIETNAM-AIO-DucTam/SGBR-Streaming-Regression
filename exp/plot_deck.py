#!/usr/bin/env python3
"""Draw the slide-deck figures: the paper's own figure forms, in the deck's colours and Vietnamese labels.

Each PNG is drawn at the size it takes on the 20 x 11.25 in slide (transparent background), so its
text matches the deck. The data are the same as for plot_e1.py, plot_e2.py and plot_e4_e5.py:
  deck_fig4.png, deck_fig5.png  like paper Figs. 4-5: Shaffer rank diagrams, paper (recomputed from the
                                authors' per-seed results, all 16 datasets) above ours
  deck_heatmap.png              E1 mean R̄² minus paper Tab. 3, every method x dataset
  deck_fig6.png                 like paper Fig. 6: windowed R̄² and cumulative wall time on hyperA
  deck_fig9.png, deck_fig10.png like paper Figs. 9-10, with the paper's values as rings
  deck_wall.png                 E1 wall time / paper Tab. 4 per dataset, with the median per method
The paper's twin-axis Figs. 9-10 are drawn as two panels sharing the x axis. Labels [1], [10], [16]
are the deck's reference numbers (paper, authors' code, this repository).

Usage:
  python3 exp/plot_deck.py [--results results] [--out results/figures/deck] [--upstream DIR]
"""

import argparse
import json
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

import plot_e1  # noqa: E402
import plot_e2  # noqa: E402
import plot_e4_e5  # noqa: E402
from summarize_e1 import DATASETS, METHODS, SHORT, TAB3, TAB4  # noqa: E402

# The deck uses Arial; under WSL the Windows fonts are visible. Otherwise matplotlib's default
# DejaVu Sans, which also covers Vietnamese.
FONT = "DejaVu Sans"
for folder in (Path("/mnt/c/Windows/Fonts"), Path("/usr/share/fonts/truetype/msttcorefonts")):
    if (folder / "arial.ttf").is_file():
        for name in ("arial.ttf", "arialbd.ttf", "ariali.ttf", "arialbi.ttf"):
            if (folder / name).is_file():
                font_manager.fontManager.addfont(str(folder / name))
        FONT = "Arial"
        break

GREEN, ORANGE, NAVY, RED = "#01BF63", "#E07A2F", "#1F2A44", "#C0392B"
INK, MUTED, GRID, AXIS = "#110E06", "#555555", "#E3E3E3", "#9A9A9A"
plt.rcParams.update({
    "font.family": FONT, "font.size": 13, "text.color": INK, "axes.labelcolor": MUTED,
    "axes.edgecolor": AXIS, "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 14,
    "axes.titleweight": "bold", "axes.titlecolor": NAVY, "mathtext.fontset": "custom",
    "mathtext.rm": FONT, "mathtext.it": f"{FONT}:italic", "mathtext.bf": f"{FONT}:bold",
    "savefig.dpi": 220, "savefig.transparent": True, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False,
})
R2ADJ = r"R²$_{\mathrm{adj}}$"
DECK_NAME = {"Oza(SGBT)": "Oza(Sgbt)", "SGB(Oza)": "Sgb(Oza)", "SGB(ARF)": "Sgb(Arf)",
             "SOKNL": "SOKNL", "OzaReg": "OzaReg", "ARFReg": "ArfReg"}
RESULTS = OUT = None  # set in main()


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)
    print("wrote", name)


def grid_y(ax):
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


# ------------------------------------------------------------------ Fig. 4-5: Shaffer rank diagrams
def rank_diagram(ax, test, title, note):
    avg = {DECK_NAME[k]: v for k, v in test["avg_ranks"].items()}
    sig = {frozenset((DECK_NAME[p["a"]], DECK_NAME[p["b"]])): p["significant"] for p in test["pairs"]}
    order = sorted(avg, key=avg.get)
    runs = plot_e1.cliques(order, sig)
    k = len(order)
    left, right = order[:math.ceil(k / 2)], order[math.ceil(k / 2):]
    bar_gap, row_gap = 0.17, 0.36
    first = 0.3 + bar_gap * len(runs)
    height = first + row_gap * max(len(left), len(right))
    margin = 1.45
    ax.set_xlim(1 - margin, k + margin)
    ax.set_ylim(height + 0.05, -0.62)
    ax.axis("off")
    ax.plot([1, k], [0, 0], color=INK, lw=1.3)
    for t in range(1, k + 1):
        ax.plot([t, t], [0, -0.12], color=INK, lw=1.3)
        ax.text(t, -0.17, str(t), ha="center", va="bottom", fontsize=12, color=INK)
    for t in range(1, k):
        ax.plot([t + 0.5, t + 0.5], [0, -0.06], color=INK, lw=0.8)
    for side, members, edge in (("L", left, 1 - margin * 0.3), ("R", right, k + margin * 0.3)):
        for row, m in enumerate(members):
            y = first + row_gap * row
            main = m == "Sgb(Oza)"
            ax.plot([avg[m], avg[m], edge], [0, y, y], color=GREEN if main else INK, lw=2.0 if main else 1.0)
            ax.text(edge + (-0.06 if side == "L" else 0.06), y, m, ha="right" if side == "L" else "left",
                    va="center", fontsize=12.5, fontweight="bold" if main else "normal")
            ax.text(edge + (0.05 if side == "L" else -0.05), y - 0.035, f"{avg[m]:.2f}",
                    ha="left" if side == "L" else "right", va="bottom", fontsize=10.5, color=MUTED)
    for i, (a, b) in enumerate(runs):
        y = 0.17 + bar_gap * i
        ax.plot([avg[order[a]] - 0.04, avg[order[b]] + 0.04], [y, y], color=INK, lw=4, solid_capstyle="butt")
    ax.text(1 - margin, -0.62, title, ha="left", va="top", fontsize=13, fontweight="bold", color=NAVY)
    ax.text(k + margin, -0.62, note, ha="right", va="top", fontsize=10.5, color=MUTED)


def posthoc_report(blocks):
    """avg_ranks and pairs keyed by the plot_e1 labels, as in results/figures/posthoc.json."""
    labels = [label for label, _, _ in plot_e1.FIG_METHODS]
    test = plot_e1.shaffer_test(blocks)
    return {"blocks": test["blocks"], "avg_ranks": dict(zip(labels, test["avg_ranks"])),
            "pairs": [{"a": labels[p["i"]], "b": labels[p["j"]], "significant": p["significant"]}
                      for p in test["pairs"]]}


def fig_shaffer(upstream):
    runs = plot_e1.load_runs(RESULTS)
    for key, metric, column, name, head in (
            ("r2", "r2", plot_e1.R2, "deck_fig4.png", f"Fig. 4 · {R2ADJ} · thấp hơn là tốt hơn"),
            ("wall", "moa_wall_seconds", plot_e1.WALL, "deck_fig5.png", "Fig. 5 · Wall time · cao hơn là tốt hơn")):
        ours_blocks, ours_sets = plot_e1.blocks_from_runs(runs, metric, False)
        if not ours_blocks:
            print("skip", name, "(no E1 blocks with all six methods)")
            return
        paper_blocks, paper_sets = plot_e1.blocks_from_post_hoc(upstream, column, plot_e1.POST_HOC_DATASETS, False)
        fig, (top, bottom) = plt.subplots(2, 1, figsize=(8.5, 5.2))
        fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.01, hspace=0.12)
        fig.text(0.01, 0.985, head, ha="left", va="top", fontsize=15, fontweight="bold", color=NAVY)
        p, o = posthoc_report(paper_blocks), posthoc_report(ours_blocks)
        rank_diagram(top, p, f"Bài · {len(paper_sets)} tập [10]", f"{p['blocks']} khối (tập × seed)")
        rank_diagram(bottom, o, f"Nhóm · {len(ours_sets)} tập [16]", f"{o['blocks']} khối (tập × seed)")
        save(fig, name)


# ------------------------------------------------------------------ E1 delta heatmap (vs Tab. 3)
def fig_heatmap():
    runs = plot_e1.load_runs(RESULTS)
    if not runs:
        print("skip deck_heatmap.png (no E1 runs)")
        return
    r2 = plot_e1.mean_by(runs, "r2")
    cols = [d for d in DATASETS if not d.startswith("SUP")]
    names = [m for m, _, _ in METHODS]
    keys = {m: k for m, k, _ in METHODS}
    grid, text = [], []
    for m in names:
        row, t = [], []
        for d in cols:
            ours = r2.get((keys[m], d))
            if ours is None:
                row.append(math.nan)
                t.append("không chạy")
            else:
                delta = ours - TAB3[m][DATASETS.index(d)]
                row.append(delta)
                t.append(f"{delta:+.3f}".replace("+0.000", "0.000").replace("-0.000", "0.000"))
        grid.append(row)
        text.append(t)
    done = [v for row in grid for v in row if not math.isnan(v)]
    cmap = LinearSegmentedColormap.from_list("d", ["#2C6FB7", "#9DBFE3", "#F2F1EE", "#F0B48A", ORANGE])
    cmap.set_bad("#FFFFFF")
    fig, ax = plt.subplots(figsize=(17.6, 6.85))
    fig.subplots_adjust(left=0.075, right=0.995, top=0.985, bottom=0.075)
    im = ax.imshow(grid, cmap=cmap, vmin=-0.01, vmax=0.01, aspect="auto")
    for r, row in enumerate(text):
        for c, t in enumerate(row):
            ax.text(c, r, t, ha="center", va="center", fontsize=11 if t != "không chạy" else 10,
                    color=MUTED if t == "không chạy" else INK, style="italic" if t == "không chạy" else "normal")
    ax.set_xticks(range(len(cols)), [SHORT[DATASETS.index(d)] for d in cols], fontsize=12.5)
    ax.set_yticks(range(len(names)), names, fontsize=12.5)
    ax.xaxis.tick_top()
    ax.tick_params(length=0, colors=INK)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks([x - 0.5 for x in range(1, len(cols))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(names))], minor=True)
    ax.grid(which="minor", color="#FFFFFF", lw=2.5)
    ax.tick_params(which="minor", length=0)
    cax = fig.add_axes([0.075, 0.018, 0.30, 0.022])
    bar = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[-0.01, -0.005, 0, 0.005, 0.01])
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=11, length=0, colors=MUTED)
    worst = max(done, key=abs)
    fig.text(0.385, 0.029, f"Δ {R2ADJ} = nhóm − Bảng 3. Bài in 2 chữ số nên chỉ riêng làm tròn đã lệch tới 0.005; "
                           f"ô lớn nhất {worst:+.3f}" + (", dưới ngưỡng 0.03" if abs(worst) < 0.03 else ""),
             ha="left", va="center", fontsize=11.5, color=MUTED)
    print("heatmap cells", len(done), "max |delta|", max(abs(v) for v in done),
          "cells above 0.005:", sum(abs(v) > 0.005 for v in done))
    save(fig, "deck_heatmap.png")


# ------------------------------------------------------------------ Fig. 6: E2 over time on HyperA
def fig_e2():
    curves = plot_e2.mean_curves(plot_e2.load_runs(RESULTS))
    style = [("SGBT(Oza(FIRTDD))", "Sgb(Oza)", GREEN, "solid", 2.6),
             ("Oza(FIRTDD)", "OzaReg", ORANGE, (0, (5, 2)), 2.0),
             ("ARFReg(FIRTDD)", "ArfReg", NAVY, (0, (6, 2, 1.5, 2)), 2.0)]
    if not all(curves.get(key) for key, *_ in style):
        print("skip deck_fig6.png (E2 Sgb(Oza), OzaReg or ArfReg not finished)")
        return
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(7.3, 4.85), sharex=True, gridspec_kw={"height_ratios": [1.25, 1]})
    fig.subplots_adjust(left=0.13, right=0.98, top=0.9, bottom=0.11, hspace=0.12)
    for key, label, color, dash, width in style:
        c = curves[key]
        x = [p[0] for p in c]
        top.plot(x, [p[2] for p in c], color=color, lw=width, linestyle=dash, zorder=3)
        bottom.plot(x, [p[3] for p in c], color=color, lw=width, linestyle=dash, zorder=3)
    for ax in (top, bottom):
        for d in plot_e2.DRIFTS:
            ax.axvline(d, color=AXIS, lw=1.1, linestyle=(0, (1.5, 2.5)), zorder=1)
        grid_y(ax)
        ax.tick_params(labelsize=11.5)
    top.set_ylim(0.35, 0.95)
    top.set_ylabel(f"{R2ADJ} cửa sổ 10k", fontsize=12)
    bottom.set_ylabel("Wall time (s)", fontsize=12)
    bottom.set_xlabel("Số mẫu đã thấy", fontsize=12)
    bottom.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
    bottom.set_xlim(0, 500_000)
    handles = [Line2D([], [], color=c, lw=w, linestyle=d, label=l) for _, l, c, d, w in style]
    handles.append(Line2D([], [], color=AXIS, lw=1.1, linestyle=(0, (1.5, 2.5)), label="drift"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.12, 1.0), ncol=4, fontsize=11.5,
               handlelength=2.6, columnspacing=1.2)
    save(fig, "deck_fig6.png")


# ------------------------------------------------------------------ Fig. 9 and Fig. 10: E4 and E5
def e45(exp, order):
    cells = plot_e4_e5.load_cells(RESULTS, exp)
    configs = [c for c in order if any(k[0] == c for k in cells)]
    rows, complete = plot_e4_e5.table(cells, configs)
    if len(configs) != len(order) or len(complete) != len(plot_e4_e5.CORE11):
        print(f"skip {exp} figure (needs every configuration on all 11 datasets)")
        return None, None
    return configs, rows


def paper_ring(ax, x, y, label=None):
    ax.plot(x, y, linestyle="none", marker="o", ms=13, mfc="none", mec=NAVY, mew=1.6, zorder=4, label=label)


def fig_e4():
    configs, rows = e45("E4", plot_e4_e5.E4_PAPER_R2)
    if configs is None:
        return
    x = list(range(len(configs)))
    r2 = [rows[c]["avg_r2"] for c in configs]
    wall = [rows[c]["avg_wall"] for c in configs]
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(8.5, 5.2), sharex=True, gridspec_kw={"height_ratios": [1.3, 1]})
    fig.subplots_adjust(left=0.1, right=0.99, top=0.83, bottom=0.15, hspace=0.1)
    top.plot(x, r2, color=GREEN, lw=2.6, marker="o", ms=7, zorder=3, label="Nhóm")
    paper_ring(top, x, [plot_e4_e5.E4_PAPER_R2[c] for c in configs], "Bài · số in trên Fig. 9 [1]")
    for i, v in enumerate(r2):
        top.annotate(f"{v:.2f}", (i, v), xytext=(0, -15 if v < -1 else 11), textcoords="offset points",
                     ha="center", va="top" if v < -1 else "bottom", fontsize=11, color=INK)
    top.set_ylim(-9.6, 2.4)
    bottom.plot(x, wall, color=ORANGE, lw=2.0, linestyle=(0, (1.5, 1.5)), marker="s", ms=7, zorder=3)
    top.set_ylabel(f"{R2ADJ} trung bình", fontsize=12)
    bottom.set_ylabel("Wall time (s)", fontsize=12)
    bottom.set_ylim(0, 480)
    bottom.set_xticks(x, [f"{c[0]:.1f}, {c[1]}%" for c in configs], fontsize=11, rotation=35, ha="right",
                      rotation_mode="anchor")
    for ax in (top, bottom):
        grid_y(ax)
        ax.tick_params(axis="y", labelsize=11.5)
        for b in (2.5, 5.5):
            ax.axvline(b, color=GRID, lw=1.2)
    fig.text(0.01, 0.985, "Fig. 9 · learning rate × % đặc trưng, Sgb(Oza) (10, 10)", ha="left", va="top",
             fontsize=15, fontweight="bold", color=NAVY)
    fig.text(0.01, 0.915, "Trung bình 11 tập × 5 seed", ha="left", va="top", fontsize=11.5, color=MUTED)
    handles = [Line2D([], [], color=GREEN, lw=2.6, marker="o", ms=7, label=f"{R2ADJ} nhóm"),
               Line2D([], [], linestyle="none", marker="o", ms=13, mfc="none", mec=NAVY, mew=1.6,
                      label="Bài · số in trên Fig. 9 [1]"),
               Line2D([], [], color=ORANGE, lw=2.0, linestyle=(0, (1.5, 1.5)), marker="s", ms=7, label="Wall time nhóm")]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.995, 0.93), ncol=3, fontsize=11,
               handlelength=2.2, columnspacing=1.0)
    save(fig, "deck_fig9.png")


def fig_e5():
    configs, rows = e45("E5", plot_e4_e5.E5_PAPER)
    if configs is None:
        return
    s = [c[0] for c in configs]
    r2 = [rows[c]["avg_r2"] for c in configs]
    wall = [rows[c]["avg_wall"] for c in configs]
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(8.5, 5.2), sharex=True, gridspec_kw={"height_ratios": [1.3, 1]})
    fig.subplots_adjust(left=0.1, right=0.99, top=0.83, bottom=0.12, hspace=0.1)
    top.plot(s, r2, color=GREEN, lw=2.6, marker="o", ms=7, zorder=3)
    paper_ring(top, s, [plot_e4_e5.E5_PAPER[c][0] for c in configs])
    spots = {(1, 100): (9, 0.36), (2, 50): (9, 0.45), (4, 25): (9, 0.54), (5, 20): (9, 0.63)}
    offsets = {(10, 10): (0, 14), (20, 5): (4, 14), (25, 4): (8, 10), (50, 2): (8, 10), (100, 1): (-10, 12)}
    for c, x, v in zip(configs, s, r2):
        label = f"({c[0]}, {c[1]})"
        if c in spots:
            top.annotate(label, (x, v), xytext=spots[c], textcoords="data", ha="left", va="center", fontsize=10.5,
                         color=MUTED, arrowprops={"arrowstyle": "-", "color": AXIS, "lw": 0.8, "shrinkA": 2, "shrinkB": 7})
        else:
            dx, dy = offsets[c]
            top.annotate(label, (x, v), xytext=(dx, dy), textcoords="offset points",
                         ha="center" if dx == 0 else ("left" if dx > 0 else "right"), va="bottom", fontsize=10.5,
                         color=MUTED)
    top.set_ylim(0, 0.88)
    bottom.plot(s, wall, color=ORANGE, lw=2.0, linestyle=(0, (1.5, 1.5)), marker="s", ms=7, zorder=3)
    paper_ring(bottom, s, [plot_e4_e5.E5_PAPER[c][2] for c in configs])
    bottom.set_ylim(0, 420)
    top.set_ylabel(f"{R2ADJ} trung bình", fontsize=12)
    bottom.set_ylabel("Wall time (s)", fontsize=12)
    bottom.set_xlabel("Số bước boosting S (cặp (S, E), S × E = 100)", fontsize=12)
    bottom.set_xlim(-3, 104)
    for ax in (top, bottom):
        grid_y(ax)
        ax.tick_params(labelsize=11.5)
    fig.text(0.01, 0.985, "Fig. 10 · số bước boosting × số cây mỗi túi, Sgb(Oza)", ha="left", va="top",
             fontsize=15, fontweight="bold", color=NAVY)
    fig.text(0.01, 0.915, "Trung bình 11 tập × 5 seed", ha="left", va="top", fontsize=11.5, color=MUTED)
    handles = [Line2D([], [], color=GREEN, lw=2.6, marker="o", ms=7, label=f"{R2ADJ} nhóm"),
               Line2D([], [], color=ORANGE, lw=2.0, linestyle=(0, (1.5, 1.5)), marker="s", ms=7, label="Wall time nhóm"),
               Line2D([], [], linestyle="none", marker="o", ms=13, mfc="none", mec=NAVY, mew=1.6,
                      label="Bài · Bảng 5, 6 [1]")]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.995, 0.93), ncol=3, fontsize=11,
               handlelength=2.2, columnspacing=1.0)
    save(fig, "deck_fig10.png")


# ------------------------------------------------------------------ wall time vs Tab. 4
def fig_wall_ratio():
    runs = plot_e1.load_runs(RESULTS)
    wall = plot_e1.mean_by(runs, "moa_wall_seconds")
    keys = {m: k for m, k, _ in METHODS}
    ratios = {m: [wall[(keys[m], d)] / TAB4[m][i] for i, d in enumerate(DATASETS) if (keys[m], d) in wall]
              for m in keys}
    ratios = {m: r for m, r in ratios.items() if r}
    if not ratios:
        print("skip deck_wall.png (no E1 runs)")
        return
    rows = sorted(ratios, key=lambda m: statistics.median(ratios[m]))
    fig, ax = plt.subplots(figsize=(7.3, 4.85))
    fig.subplots_adjust(left=0.17, right=0.98, top=0.9, bottom=0.13)
    for y, m in enumerate(rows):
        ax.scatter(ratios[m], [y] * len(ratios[m]), s=34, color=ORANGE, alpha=0.8, edgecolor="#FFFFFF",
                   linewidth=0.8, zorder=3)
        med = statistics.median(ratios[m])
        ax.plot([med, med], [y - 0.36, y + 0.36], color=NAVY, lw=2.6, zorder=4)
        print(f"  {m:10s} median {med:.2f} range {min(ratios[m]):.2f}-{max(ratios[m]):.2f}")
    ax.axvline(1, color=AXIS, lw=1.2, zorder=1)
    ax.set_xscale("log")
    ax.set_xlim(0.8, 200)
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100, 200])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}×"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks(range(len(rows)), rows, fontsize=11.5, color=INK)
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=11.5)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("Wall time nhóm / Bảng 4 (thang log, 1× = như máy tác giả)", fontsize=11.5)
    handles = [Line2D([], [], linestyle="none", marker="o", ms=7, color=ORANGE, label="một tập"),
               Line2D([], [], color=NAVY, lw=2.6, label="trung vị 13 tập")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.16, 1.0), ncol=2, fontsize=11.5)
    save(fig, "deck_wall.png")


def main(argv=None):
    global RESULTS, OUT
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, help="Default: <results>/figures/deck")
    parser.add_argument("--toolchain", type=Path, default=Path.home() / "sgbr" / "toolchain.json",
                        help="Supplies --upstream (the authors' repository, for the paper's post-hoc input)")
    parser.add_argument("--upstream", type=Path)
    args = parser.parse_args(argv)
    RESULTS, OUT = args.results, args.out or args.results / "figures" / "deck"
    OUT.mkdir(parents=True, exist_ok=True)
    upstream = args.upstream
    if upstream is None and args.toolchain.is_file():
        upstream = Path(json.loads(args.toolchain.read_text())["upstream"])
    if upstream is None:
        parser.error("Run exp/toolchain.py first, or pass --upstream")
    fig_shaffer(upstream)
    fig_heatmap()
    fig_e2()
    fig_e4()
    fig_e5()
    fig_wall_ratio()
    print("Figures in", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
