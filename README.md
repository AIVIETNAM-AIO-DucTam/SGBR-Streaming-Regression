# SGBR-Streaming-Regression

From-scratch Python reimplementation and reproduction study of "Gradient boosted bagging for evolving data stream regression" (SGBR, DMKD 2025).

This repository currently holds the reproduction harness. It runs the paper's experiments with the
authors' own MOA code, pinned to commit
[`f62f7b4`](https://github.com/nuwangunasekara/SGBR/tree/f62f7b437512e020f57c4bcd03884c4724eda5b1),
using the exact learner, evaluator and JVM settings from their `exp/*.json` configs. It then compares
every run with the paper's Tab. 3 (R̄²) and Tab. 4 (wall time) and with the authors' per-seed results.

## Layout

```
exp/
  toolchain.py     JDK 11 + Maven (checksum-verified), authors' MOA source at f62f7b4, fat-jar build, smoke test
  run_matrix.py    runs one JVM per (experiment, dataset, method, seed) cell, resumable
  summarize_e1.py  E1 results table vs paper Tab. 3 / Tab. 4 and vs the authors' exp/post_hoc_new
results/
  raw/             one folder per cell: record.json, evaluation.csv, java.log (not committed)
  summary/         summary tables
```

## Requirements

- Linux x86_64 (a rented VM or a workstation), Python 3.9+, `git`, internet access.
- A few GB of disk under `~/sgbr` for the JDK, Maven repository, MOA source, jar and datasets.
- RAM decides how many cells run at once. Most cells need 1–6 GB of heap. On NZEnergy at `-Xmx6g`,
  SGBT(SRP(FIRTDD)) ran out of memory and SRP(SGBT(FIRTDD)) did not finish within 11 hours, so both
  are given 26 GB below.

## Running E1 on one machine

```bash
git clone https://github.com/AIVIETNAM-AIO-DucTam/SGBR-Streaming-Regression.git
cd SGBR-Streaming-Regression

# 1. Toolchain: JDK 11, Maven, MOA source and jar under ~/sgbr (about 10 minutes the first time)
python3 exp/toolchain.py

# 2. All E1 cells. Run inside tmux or screen so a dropped SSH session does not stop it.
python3 exp/run_matrix.py --exp E1 --plan repro --jobs 16 --ram-budget 150g --exit-on-oom \
  --xmx "default=6g,NZEnergy:SGBT(SRP(FIRTDD))=26g,NZEnergy:SRP(SGBT(FIRTDD))=26g" 2>&1 | tee e1.log

# 3. Results table (results/summary/E1_table.md and E1_results.csv)
python3 exp/summarize_e1.py

# 4. Copy everything off the machine before you destroy it
tar czf e1_results.tgz results/ e1.log
```

Choosing the options:

- `--jobs`: at most the number of vCPUs. Each MOA run is single-threaded.
- `--ram-budget`: the machine's RAM minus about 10 GB. A cell starts only while the running JVMs fit
  (1.15 × `-Xmx` + 256 MiB reserved per JVM), so the 26 GB cells never overcommit memory. Long cells
  start first, and small cells fill the remaining space.
- `--xmx`: `default=<size>`, then optional `<dataset>=<size>` or `<dataset>:<method>=<size>` overrides.
- `--list` prints the cells and the paper-speed time estimate without running anything.

Re-running the same command resumes: completed cells are skipped, including those copied in from
another machine into `results/raw/`, provided the MOA task, heap, dataset and config are identical.
A cell that was running when the process stopped starts again from the beginning, because MOA
cannot resume a run.

`--plan repro` runs 686 cells: 13 methods × the 13 datasets shipped in the authors' repo, 5 seeds
each (methods without a seed option run once), with AXGBr only on the 10 smaller datasets. At the
paper's speed this is about 145 CPU-hours, and the longest single run (NZEnergy SGBT(SRP)) is about
6.4 hours.

## Differences from the authors' setup

| Item | Authors (exp/exp_config.json) | Here |
|---|---|---|
| Maximum heap | `-Xmx48g` for every run | Set per cell with `--xmx` (recorded in each `record.json`) |
| Datasets | 16, including SUP2I, SUP3A, SUP3G | 13. The SUP\* ARFF files are not in the authors' repository |
| AXGBr | All datasets | 10 smaller datasets (`--plan repro`); NZEnergy alone takes about 1.3e5 s in Tab. 4 |
| JVM flags | as configured | optional `-XX:+ExitOnOutOfMemoryError` (`--exit-on-oom`) so an OOM stops at once |
| Java | not specified in the config | Temurin JDK 11.0.29 (MEKA 1.9.7 in the MOA fork needs Java 11) |

The learner strings, evaluator, seeds and the remaining JVM options are unchanged.

## Results so far

- `results/summary/B1_parity.csv`: NZEnergy, SGBT(ARFReg(FIRTDD)), seed 1 (3,100,666 instances)
  reproduces the authors' final R̄² exactly (0.9741708731300197, Δ = 0) on an AMD EPYC 7B12.
- Spot checks on abalone (Oza(SGBT), SGBT(Oza), Oza(FIRTDD), seeds 1–2) also match the authors'
  per-seed results exactly.
