#!/usr/bin/env python3
"""Summarise E1 cells from exp/run_matrix.py against the paper and the authors' raw results.

Outputs results/summary/E1_results.csv (one row per method x dataset) and results/summary/E1_table.md.
Reference values: Tab. 3 (adjusted R², p. 15) and Tab. 4 (wall time, p. 17) of the paper, plus the
per-seed final rows the authors committed in exp/post_hoc_new for 6 methods.
"""

import argparse
import csv
import json
import math
import statistics
import subprocess
from pathlib import Path

PIN = "f62f7b437512e020f57c4bcd03884c4724eda5b1"
DATASETS = ["fried", "House8L", "abalone", "bike", "elevators", "MetroTraffic", "ailerons",
            "FriedmanLea", "FriedmanGra", "FriedmanGsg", "hyperA", "DemandF", "NZEnergy"]
SHORT = ["fried", "House8L", "abalone", "bike", "elev.", "Metro", "ailer.", "FrLea", "FrGra", "FrGsg",
         "hyperA", "DemandF", "NZEnergy"]
# (paper name, config label, post_hoc_new file or None)
METHODS = [
    ("Oza(Sgbt)", "Oza(SGBT(FIRTDD))", "Oza(SGBT)"),
    ("Srp(Sgbt)", "SRP(SGBT(FIRTDD))", None),
    ("Sgb(Oza)", "SGBT(Oza(FIRTDD))", "SGBT(Oza)"),
    ("Sgb(Arf)", "SGBT(ARFReg(FIRTDD))", "SGBT(ARF)"),
    ("Sgb(Srp)", "SGBT(SRP(FIRTDD))", None),
    ("Sgbt", "SGBT(FIRTDD)", None),
    ("Axgb", "AXGBr", None),
    ("OzaReg", "Oza(FIRTDD)", "OzaReg"),
    ("ArfReg", "ARFReg(FIRTDD)", "ARFReg"),
    ("SOKNL", "SOKNL(FIRTDD)", "SOKNL"),
    ("SrpReg", "SRP(FIRTDD)", None),
    ("FIRT-DD", "FIRTDD", None),
    ("HTR", "HT", None),
]
# Paper Tab.3 p.15 (mean R2 over 5 seeds), columns in DATASETS order.
TAB3 = {
    "Oza(Sgbt)": [0.89, 0.60, 0.46, 0.93, 0.70, 0.24, 0.76, 0.88, 0.88, 0.88, 0.88, 0.63, 0.97],
    "Srp(Sgbt)": [0.67, 0.56, 0.49, 0.85, 0.73, 0.36, 0.76, 0.63, 0.64, 0.66, 0.69, 0.55, 0.98],
    "Sgb(Oza)": [0.89, 0.58, 0.48, 0.93, 0.71, 0.28, 0.75, 0.88, 0.88, 0.87, 0.87, 0.66, 0.98],
    "Sgb(Arf)": [0.72, 0.55, 0.53, 0.81, 0.52, 0.51, 0.68, 0.73, 0.71, 0.69, 0.63, 0.57, 0.98],
    "Sgb(Srp)": [0.67, 0.55, 0.48, 0.79, 0.51, 0.31, 0.66, 0.63, 0.60, 0.60, 0.49, 0.60, 0.97],
    "Sgbt": [0.46, 0.18, 0.08, 0.80, 0.19, -0.05, 0.38, 0.49, 0.42, 0.42, 0.27, 0.29, 0.95],
    "Axgb": [0.80, -0.53, -0.12, 0.84, 0.38, -0.84, 0.61, 0.73, 0.70, 0.68, 0.70, -0.73, -7.7e28],
    "OzaReg": [0.82, 0.54, 0.37, 0.82, 0.53, 0.15, 0.62, 0.81, 0.79, 0.79, 0.82, 0.49, 0.81],
    "ArfReg": [0.65, 0.54, 0.48, 0.74, 0.39, 0.26, 0.55, 0.67, 0.65, 0.64, 0.61, 0.51, 0.96],
    "SOKNL": [0.82, 0.58, 0.58, 0.80, 0.46, 0.31, 0.56, 0.81, 0.81, 0.80, 0.86, 0.53, 0.96],
    "SrpReg": [0.53, 0.52, 0.45, 0.72, 0.37, 0.21, 0.57, 0.52, 0.49, 0.49, 0.43, 0.52, 0.96],
    "FIRT-DD": [0.60, 0.46, 0.16, 0.75, 0.24, 0.00, 0.46, 0.63, 0.57, 0.58, 0.47, 0.36, 0.74],
    "HTR": [0.65, 0.36, 0.14, 0.56, 0.25, 0.04, 0.44, 0.64, 0.47, 0.42, 0.35, 0.29, 0.62],
}
# Paper Tab.4 p.17 (mean wall time, seconds), columns in DATASETS order.
TAB4 = {
    "Oza(Sgbt)": [58.3, 27.5, 4.4, 14.1, 20.9, 33.6, 22.6, 172.7, 173.3, 173.2, 1.4e3, 410.7, 9.0e3],
    "Srp(Sgbt)": [83.6, 39.7, 7.0, 29.4, 46.2, 79.3, 55.6, 310.2, 353.9, 365.3, 2.8e3, 1.0e3, 2.2e4],
    "Sgb(Oza)": [30.8, 12.9, 1.9, 5.1, 9.0, 9.9, 11.0, 81.3, 79.7, 79.6, 527.4, 151.9, 3.2e3],
    "Sgb(Arf)": [38.3, 24.6, 3.3, 11.4, 11.5, 20.6, 15.5, 108.4, 111.4, 109.7, 805.4, 365.3, 6.9e3],
    "Sgb(Srp)": [41.4, 17.4, 3.8, 11.7, 23.2, 25.5, 26.3, 121.1, 115.5, 112.0, 732.4, 355.8, 2.3e4],
    "Sgbt": [54.1, 19.6, 3.3, 10.3, 17.7, 19.4, 26.1, 137.9, 146.8, 143.2, 1.1e3, 366.4, 8.2e3],
    "Axgb": [180.7, 109.4, 13.0, 97.6, 73.4, 197.5, 54.7, 3.7e3, 3.0e3, 3.0e3, 1.5e4, 5.9e3, 1.3e5],
    "OzaReg": [26.6, 9.9, 1.6, 4.7, 7.2, 6.7, 10.5, 73.4, 62.9, 63.4, 381.1, 26.3, 1.2e3],
    "ArfReg": [31.0, 22.2, 4.5, 11.5, 11.5, 20.8, 25.0, 114.3, 97.6, 99.6, 594.3, 70.5, 1.7e3],
    "SOKNL": [56.5, 34.1, 6.8, 20.3, 20.0, 44.1, 32.4, 180.9, 161.7, 162.8, 1.2e3, 151.2, 3.9e3],
    "SrpReg": [31.8, 13.9, 3.0, 11.2, 21.1, 19.2, 30.3, 87.5, 92.1, 86.7, 484.1, 78.9, 4.1e3],
    "FIRT-DD": [0.6, 0.3, 0.1, 0.2, 0.2, 0.2, 0.3, 1.8, 1.8, 1.7, 6.2, 0.7, 14.5],
    "HTR": [1.0, 0.5, 0.2, 0.3, 0.3, 0.4, 0.4, 2.7, 2.7, 2.7, 13.5, 1.1, 27.5],
}


def post_hoc(upstream, dataset, seed, file):
    try:
        text = subprocess.check_output(
            ["git", "-C", str(upstream), "show", f"{PIN}:exp/post_hoc_new/{dataset}_{seed}/{file}.csv"],
            stderr=subprocess.DEVNULL, text=True)
    except subprocess.CalledProcessError:
        return None
    rows = list(csv.DictReader(text.splitlines()))
    return float(rows[-1]["adjusted coefficient of determination"]) if rows else None


def ranks(values, higher_better):
    order = sorted(values, key=lambda kv: kv[1], reverse=higher_better)
    result, i = {}, 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and order[j + 1][1] == order[i][1]:
            j += 1
        for k in range(i, j + 1):
            result[order[k][0]] = (i + j) / 2 + 1
        i = j + 1
    return result


def fmt(value, digits=2):
    if value is None:
        return "–"
    if abs(value) >= 1e4:
        return f"{value:.1e}"
    return f"{value:.{digits}f}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toolchain", type=Path, default=Path.home() / "sgbr" / "toolchain.json",
                        help="Written by exp/toolchain.py; supplies --upstream")
    parser.add_argument("--upstream", type=Path, help="Checkout of the authors' repo at the pinned commit")
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args(argv)
    if args.upstream is None and args.toolchain.is_file():
        args.upstream = Path(json.loads(args.toolchain.read_text())["upstream"])
    if args.upstream is None:
        parser.error("Run exp/toolchain.py first, or pass --upstream")

    cells = {}
    for path in (args.results / "raw" / "E1").glob("*/*/seed=*/record.json"):
        record = json.loads(path.read_text())
        if record.get("status") == "completed":
            cells.setdefault((record["method_key"], record["dataset"]), []).append(record)

    rows, mean_r2, mean_wall = [], {}, {}
    for paper_name, label, file in METHODS:
        for index, dataset in enumerate(DATASETS):
            records = sorted(cells.get((label, dataset), []), key=lambda r: r["seed"] or 0)
            if not records:
                continue
            r2 = [r["r2"] for r in records]
            wall = [r["moa_wall_seconds"] for r in records]
            unseeded = records[0]["seed"] is None
            deltas = []
            if file:
                for record in records:
                    reference = post_hoc(args.upstream, dataset, record["seed"], file)
                    if reference is not None:
                        deltas.append(abs(record["r2"] - reference))
            mean = statistics.fmean(r2)
            paper = TAB3[paper_name][index]
            delta = mean - paper
            verdict = "match" if abs(delta) <= 0.03 else "partial" if abs(delta) <= 0.10 else "mismatch"
            if paper_name == "Axgb" and dataset == "NZEnergy":
                verdict = "n/a"
            mean_r2[(paper_name, dataset)] = mean
            mean_wall[(paper_name, dataset)] = statistics.fmean(wall)
            rows.append({
                "method": paper_name, "method_key": label, "dataset": dataset,
                "runs": len(records), "one_run_rule": unseeded,
                "r2_mean": mean, "r2_std_sample": statistics.stdev(r2) if len(r2) > 1 else 0.0,
                "r2_paper_tab3": paper, "delta_vs_tab3": delta, "tab3_verdict": verdict,
                "max_abs_delta_vs_post_hoc_per_seed": max(deltas) if deltas else None,
                "post_hoc_seeds_compared": len(deltas),
                "wall_mean_s": statistics.fmean(wall), "wall_paper_tab4_s": TAB4[paper_name][index],
                "wall_ratio_vs_paper": statistics.fmean(wall) / TAB4[paper_name][index],
                "xmx": records[0]["xmx"], "cpu_model": records[0]["cpu_model"],
                "max_concurrency": max(r["concurrency"] for r in records),
                "peak_rss_gib_max": max((r["peak_rss_bytes"] or 0) for r in records) / 2**30,
            })

    out = args.results / "summary"
    out.mkdir(parents=True, exist_ok=True)
    if rows:
        with (out / "E1_results.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    # Datasets on which every method that has any result also has this dataset: used for avg/avgRank.
    methods = [m for m, _, _ in METHODS if any((m, d) in mean_r2 for d in DATASETS)]
    done = [d for d in DATASETS if methods and all((m, d) in mean_r2 for m in methods)]
    lines = ["# E1 results (adjusted R², mean over seeds; paper Tab.3 in parentheses)", "",
             f"Datasets complete for all {len(methods)} methods: {len(done)} of 13. "
             "avg and avgRank use only those datasets, for ours and the paper alike.", ""]
    header = ["Method"] + [SHORT[DATASETS.index(d)] for d in DATASETS] + ["avg", "avgRank"]
    lines += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    rank_ours = {m: [] for m in methods}
    rank_paper = {m: [] for m in methods}
    for d in done:
        i = DATASETS.index(d)
        for m, r in ranks([(m, mean_r2[(m, d)]) for m in methods], True).items():
            rank_ours[m].append(r)
        for m, r in ranks([(m, TAB3[m][i]) for m in methods], True).items():
            rank_paper[m].append(r)
    for m in methods:
        cells_text = []
        for i, d in enumerate(DATASETS):
            ours = mean_r2.get((m, d))
            cells_text.append("–" if ours is None else f"{fmt(ours)} ({fmt(TAB3[m][i])})")
        if done:
            ours_avg = statistics.fmean(mean_r2[(m, d)] for d in done)
            paper_avg = statistics.fmean(TAB3[m][DATASETS.index(d)] for d in done)
            avg = f"{fmt(ours_avg)} ({fmt(paper_avg)})"
            rank = f"{statistics.fmean(rank_ours[m]):.1f} ({statistics.fmean(rank_paper[m]):.1f})"
        else:
            avg = rank = "–"
        lines.append("| " + " | ".join([m, *cells_text, avg, rank]) + " |")

    lines += ["", "# E1 wall time (s, mean over seeds; paper Tab.4 in parentheses)", "",
              "Absolute seconds depend on hardware and concurrency; compare ratios only.", "",
              "| " + " | ".join(["Method"] + [SHORT[DATASETS.index(d)] for d in DATASETS]) + " |",
              "|" + "---|" * (len(DATASETS) + 1)]
    for m in methods:
        cells_text = []
        for i, d in enumerate(DATASETS):
            ours = mean_wall.get((m, d))
            cells_text.append("–" if ours is None else f"{fmt(ours, 1)} ({fmt(TAB4[m][i], 1)})")
        lines.append("| " + " | ".join([m, *cells_text]) + " |")

    parity = [r for r in rows if r["max_abs_delta_vs_post_hoc_per_seed"] is not None]
    if parity:
        worst = max(r["max_abs_delta_vs_post_hoc_per_seed"] for r in parity)
        seeds = sum(r["post_hoc_seeds_compared"] for r in parity)
        lines += ["", f"Per-seed parity vs authors' raw results (exp/post_hoc_new, 6 methods): "
                      f"{seeds} runs compared, max |ΔR̄²| = {worst:.2e}."]
    verdicts = {}
    for r in rows:
        verdicts[r["tab3_verdict"]] = verdicts.get(r["tab3_verdict"], 0) + 1
    lines += [f"Tab.3 cell verdicts (|Δmean| ≤ 0.03 match, ≤ 0.10 partial): {verdicts}."]
    (out / "E1_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
