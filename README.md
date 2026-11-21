# SGBR-Streaming-Regression

From-scratch Python reimplementation and reproduction study of "Gradient boosted bagging for evolving data stream regression" (SGBR, DMKD 2025).

This repository currently holds the reproduction harness. It runs the paper's experiments with the
authors' own MOA code, pinned to commit
[`f62f7b4`](https://github.com/nuwangunasekara/SGBR/tree/f62f7b437512e020f57c4bcd03884c4724eda5b1),
using the exact learner, evaluator and JVM settings from their `exp/*.json` configs and their
bias-variance script. Every result is compared with the paper and, where the authors committed them,
with their own per-run results.

## Experiments

| ID | What | Paper | Authors' setup | Runs |
|---|---|---|---|---|
| E1 | 13 methods on 13 datasets, final adjusted R̄² and wall time | Tabs. 3-4, Figs. 4-5 | `exp/exp_config.json` | 686 |
| E2 | Sgb(Oza), SOKNL, ArfReg, OzaReg over time on hyperA (3 abrupt drifts) | Fig. 6 | `exp/exp_config_overTime.json` | 12 |
| E3 | Bias-variance decomposition of Sgbt, Sgb(Oza), Oza(Sgbt) against boosting iterations | Figs. 2, 7, 8 | `exp/Bias_Variance_Decomp/` | 4,200 tasks |
| E4 | Sgb(Oza) (10, 10): learning rate × feature % on 11 datasets | Fig. 9 | `exp/exp_config_SGBT_OZA_parameter_search_LR_FP.json` | 495 |
| E5 | Sgb(Oza): boosting steps × bag size (product 100) on 11 datasets | Tabs. 5-6, Fig. 10 | `exp/exp_config_SGBT_OZA_parameter_search.json` | 495 |

## Layout

```
exp/
  run_all.sh       the whole procedure: setup, all experiments, tables, figures, archive
  toolchain.py     JDK 11 + Maven (checksum-verified), authors' MOA source at f62f7b4, fat-jar build, smoke test
  run_matrix.py    MOA experiments E1, E2, E4, E5: one JVM per (experiment, dataset, method, seed), resumable
  run_e3.py        E3 bias-variance decomposition, resumable
  summarize_e1.py  E1 table vs paper Tabs. 3-4 and the authors' per-seed results
  plot_e1.py       E1 figures (paper Figs. 4-5 and comparisons with the paper)
  plot_e2.py       E2 figure (paper Fig. 6) and drift summary
  plot_e3.py       E3 figures (paper Figs. 2, 7, 8)
  plot_e4_e5.py    E4/E5 tables and figures (paper Fig. 9, Tabs. 5-6)
requirements.txt   Python packages for E3 and the figures
results/
  raw/             one folder per run or task (not committed)
  summary/         tables (CSV and Markdown)
  figures/         PNG figures
```

## Running everything on one machine

Requirements: Linux x86_64 with root or sudo (a rented VM such as vast.ai is fine), `git`,
`python3` with `venv`, internet access, and about 10 GB of free disk under `~/sgbr`. More cores make
it faster. RAM needs to cover the parallel runs: most take 1-6 GB, and two NZEnergy runs of E1 need
26 GB each. A 16 vCPU / 64 GB machine or larger is a good fit.

```bash
apt-get update && apt-get install -y git python3-venv tmux     # once, on a fresh machine
git clone https://github.com/AIVIETNAM-AIO-DucTam/SGBR-Streaming-Regression.git
cd SGBR-Streaming-Regression
tmux new -s sgbr                    # keeps running if SSH drops; reattach with: tmux attach -t sgbr
bash exp/run_all.sh                 # setup, E1 E2 E4 E5, E3, figures, archive
```

Follow progress from another shell with `tail -f logs/moa.log` or `tail -f logs/e3.log`. When it
finishes, copy the archive to your own computer before destroying the machine:

```bash
scp -P <port> root@<host>:SGBR-Streaming-Regression/sgbr_results_*.tgz .
```

`run_all.sh` runs these steps; name one or more to run only those, for example
`bash exp/run_all.sh figures` to redraw everything that has finished so far:

| Step | Does |
|---|---|
| `setup` | Builds the toolchain under `~/sgbr` (JDK 11, Maven, authors' MOA source and jar, about 10 minutes the first time) and the Python environment `~/sgbr/venv` |
| `moa` | E1, E2, E4 and E5 in one queue, longest runs first, as many in parallel as cores and memory allow |
| `e3` | E3 on the 7 datasets that can be rerun |
| `figures` | All tables in `results/summary/` and all figures in `results/figures/` |
| `pack` | `results/` and `logs/` into `sgbr_results_<date>.tgz` |

Every step resumes. After an interruption or a reboot, run the same command again: finished runs
are skipped, including runs copied in from another machine into `results/raw/`. A run that was in
progress starts again from the beginning, because MOA cannot resume a run.

Settings are environment variables, for example `JOBS=12 RAM_GB=100 bash exp/run_all.sh`:

| Variable | Default | Meaning |
|---|---|---|
| `JOBS` | number of vCPUs | parallel runs (each MOA run uses one core) |
| `RAM_GB` | total RAM − 10 GB | memory the parallel runs may reserve; a run starts only if 1.15 × its heap + 256 MiB still fits |
| `EXPERIMENTS` | `E1 E2 E4 E5` | MOA experiments for the `moa` step |
| `MOA_XMX` | `default=6g,NZEnergy:SGBT(SRP(FIRTDD))=26g,NZEnergy:SRP(SGBT(FIRTDD))=26g` | heap per run: a default, then `<dataset>=` or `<dataset>:<method>=` overrides |
| `RETRY_XMX` | `12g` | heap for a second try of any run that ran out of memory |
| `E3_XMX` | `4g` | heap per E3 worker |

The two NZEnergy runs get 26 GB because at 6 GB SGBT(SRP(FIRTDD)) ran out of memory and
SRP(SGBT(FIRTDD)) had not finished after 11 hours.

Rough time on 16 vCPUs, extrapolated from the paper's run times and our test runs (not measured end
to end yet):

| Part | CPU-hours at the paper's speed | Wall time on 16 vCPUs |
|---|---|---|
| E1 + E2 + E4 + E5 (`moa`) | about 168 | about 15-20 h; the longest single run (NZEnergy SGBT(SRP)) alone takes 6-10 h |
| E3 (`e3`) | about 100-250 (from abalone timings) | about 6-16 h |

Each script can also be run on its own; `--help` lists its options. For example, `python3
exp/run_matrix.py --exp E5 --datasets abalone --seeds 1 --jobs 4` runs one small slice, and
`--list` prints the runs without starting them.

## Outputs

| File | Content |
|---|---|
| `summary/E1_table.md`, `E1_results.csv` | E1 mean R̄² and wall time per method × dataset, against paper Tabs. 3-4 and the authors' per-seed results |
| `figures/fig4_shaffer_r2.png`, `fig5_shaffer_walltime.png` | Like paper Figs. 4-5: average ranks and Shaffer post-hoc test (α = 0.05) on R̄² and on wall time |
| `figures/r2_vs_paper.png`, `r2_delta_heatmap.png`, `avg_rank_vs_paper.png`, `walltime_ratio.png` | E1 compared with paper Tabs. 3-4 |
| `summary/E2_over_time.csv`, `E2_drift.md`, `figures/fig6_over_time.png` | Like paper Fig. 6: windowed R̄² and wall time over hyperA, and R̄² before and after each drift |
| `summary/E3_bias_variance.csv`, `E3_parity.csv` | E3 curves, and their difference from the authors' committed values at every point |
| `figures/fig2_*.png`, `fig7_*.png`, `fig8_*.png` | Like paper Figs. 2, 7, 8; grey lines are the authors' curves. `*_authors.png` are drawn from the authors' committed results only |
| `summary/E4_table.md`, `E4_grid.csv`, `figures/fig9_lr_feature.png` | Like paper Fig. 9: average R̄² and wall time per (learning rate, feature %) |
| `summary/E5_table.md`, `E5_grid.csv`, `figures/fig10_steps_bagsize.png` | Like paper Tabs. 5-6 and Fig. 10: R̄² and wall time per (boosting steps, bag size), with avg and avgRank |

The figures use the paper's layout, with two changes for readability: the paper's twin-axis plots
(Figs. 9-10) are drawn as two panels sharing the x axis, and colours replace the paper's black line
styles.

## How each experiment is run and compared

**E1, E2, E4, E5.** `run_matrix.py` takes the learner, evaluator and JVM options verbatim from the
authors' config, replaces only `-Xmx`, and appends the seed (`-Z <seed>`) to the outer learner only,
as the authors' `exp/benchmark_moa.py` does. Learners without a seed option run once per dataset.

- E1 post-hoc test (`plot_e1.py`): Friedman ranks per block with rank 1 for the largest value (for
  wall time rank 1 is the slowest, "higher is better", as in the paper), pairwise z statistics and
  Shaffer's static procedure. Each (dataset, seed) run is one block, matching the authors' post-hoc
  input layout (`exp/post_hoc_new/<dataset>_<seed>/`). Check: `python3 exp/plot_e1.py --source
  post-hoc` draws Figs. 4-5 from the authors' own per-seed results and gives the paper's average
  ranks (1.84, 2.00, 3.29, 3.96, 4.53, 5.39 for R̄²; 1.61, 1.70, 3.34, 3.56, 4.90, 5.89 for wall
  time) and the same groups of non-significant pairs. Our runs cover 13 of the 16 datasets, so our
  ranks can differ slightly even when every run matches.
- E2 (`plot_e2.py`): as in the authors' `exp/StatsOverTime.py`, each point is the mean over seeds
  1-3 of the windowed R̄² (last 10,000 instances) and of the cumulative wall time, written every
  10,000 instances. Drifts are at 125k, 250k and 375k.
- E4 and E5 (`plot_e4_e5.py`): averaged as in the authors' `exp/generateSummaryXls.py`: every row
  of a run's output (MOA writes one every 100,000 instances and one at the end) is averaged per
  configuration and dataset over all seeds; "avg" is the mean over datasets and "avgRank" the mean
  of per-dataset ranks. Only hyperA has more than one row; the final-row values are also kept in the
  CSV. Paper reference values: E4 from the numbers printed on Fig. 9; E5 from Tabs. 5-6. The paper's
  E5 wall times come from other hardware (a Ryzen 7 3700X), so compare their pattern, not seconds.

**E3.** The authors' script `exp/Bias_Variance_Decomp/BVDecompSyntheticLatest.py` wraps MOA
learners for scikit-learn with CapyMOA and runs `mlxtend.evaluate.bias_variance_decomp`. `run_e3.py`
follows the same protocol with the same MOA jar, driven through JPype:

- Methods (labels of the authors' result files): `B` = Sgbt, `BB` = Sgb(Oza) with 5 bagged trees per
  step, `BagB` = Oza(Sgbt) with 5 bagged SGBT models; boosting iterations 5, 10, ..., 100.
- Each dataset is split by `KFold(n_splits=10)` without shuffling. In each fold, 10 models are
  trained on bootstrap samples of the training fold (`num_rounds=10, random_seed=42`) and predict
  the test fold. Bias and variance are averaged over the 10 folds, and their square roots reported.
- Only the 7 datasets without drift can be rerun: House8L, MetroTraffic, abalone, ailerons, bike,
  elevators, fried. The 3 drifting Friedman datasets need split files (`<name>_1` ... `<name>_10`
  for training, `<name>_12` for testing) that are not in the authors' repository; with them, pass
  `--split-dir` to `run_e3.py`. Their panels in Fig. 7 are available from the authors' curves
  (`plot_e3.py --source authors`).
- Check: on abalone, all three methods at 5 and 10 iterations match the authors' committed values
  to within 2e-15 (floating-point summation order).

## Differences from the authors' setup

| Item | Authors | Here |
|---|---|---|
| Maximum heap | `-Xmx48g` (E1) or `-Xmx12g` (E2, E4, E5) for every run | Per run (`MOA_XMX`, recorded in each `record.json`); runs that run out of memory are retried with `RETRY_XMX`. The heap does not change results |
| Datasets (E1) | 16, including SUP2I, SUP3A, SUP3G | 13. The SUP\* ARFF files are not in the authors' repository |
| AXGBr (E1) | All datasets | 10 smaller datasets; NZEnergy alone takes about 1.3e5 s in Tab. 4 |
| JVM flags | as configured | plus `-XX:+ExitOnOutOfMemoryError` so an out-of-memory run stops at once |
| Java | not specified in the config | Temurin JDK 11.0.29 (MEKA 1.9.7 in the MOA fork needs Java 11) |
| E3 wrapper | CapyMOA with its `moa.jar` replaced by theirs | JPype calling the same jar directly, same steps; results match theirs |
| E3 datasets | 10, including 3 drifting Friedman sets from split files | 7; the split files are not in their repository |

The learner strings, evaluators, seeds and the remaining JVM options are unchanged.

## Results so far

- `results/summary/B1_parity.csv`: NZEnergy, SGBT(ARFReg(FIRTDD)), seed 1 (3,100,666 instances)
  reproduces the authors' final R̄² exactly (0.9741708731300197, Δ = 0) on an AMD EPYC 7B12.
- Spot checks on abalone (Oza(SGBT), SGBT(Oza), Oza(FIRTDD), seeds 1-2) also match the authors'
  per-seed results exactly.
- E3 on abalone (3 methods, 5 and 10 iterations) matches the authors' curves to within 2e-15.
