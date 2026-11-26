#!/usr/bin/env python3
"""Run SGBR experiment cells (E1, E2, E4, E5) with the authors' exact MOA commands.

One JVM per (experiment, dataset, learner, seed) cell. The learner and evaluator strings and the
JVM options come verbatim from the authors' JSON config at the pinned commit; only the maximum heap
(-Xmx) is replaced, and it is recorded per cell. Cells of all requested experiments share one queue:
longest first, at most --jobs at a time and, with --ram-budget, only while the heaps of the running
cells fit in that budget. With --retry-oom, cells that run out of memory are run again with that
heap. Finished cells are skipped on rerun, so an interrupted run can simply be restarted.

Usage (Linux, after exp/toolchain.py):
  python3 exp/run_matrix.py --exp E1 E2 E4 E5 --plan repro --jobs 16 --ram-budget 150g \\
      --xmx "default=6g,NZEnergy:SGBT(SRP(FIRTDD))=26g,NZEnergy:SRP(SGBT(FIRTDD))=26g" \\
      --exit-on-oom --retry-oom 12g
"""

import argparse
import concurrent.futures as futures
import csv
import datetime
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

sys.dont_write_bytecode = True
from toolchain import UPSTREAM_COMMIT, git_bytes  # noqa: E402

CONFIGS = {
    "E1": "exp/exp_config.json",
    "E2": "exp/exp_config_overTime.json",
    "E4": "exp/exp_config_SGBT_OZA_parameter_search_LR_FP.json",
    "E5": "exp/exp_config_SGBT_OZA_parameter_search.json",
}
# The 10 smaller datasets. With --plan repro, AXGBr runs only on these (its NZEnergy run alone
# takes about 1.3e5 s in the paper's Tab. 4).
CORE10 = ["fried", "House8L", "abalone", "bike", "elevators", "MetroTraffic", "ailerons",
          "FriedmanLea", "FriedmanGra", "FriedmanGsg"]
R2 = "adjusted coefficient of determination"
WALL = "evaluation time (wall) (seconds)"
CLASSIFIED = "classified instances"
# Instances processed so far. "classified instances" is the window size under the windowed
# evaluator of E2, so completion is checked on this column when it is present.
SEEN = "learning evaluation instances"
JAVA_OPTION_VARIABLES = {"JAVA_TOOL_OPTIONS", "JDK_JAVA_OPTIONS", "_JAVA_OPTIONS"}
OOM_PATTERN = re.compile(rb"OutOfMemoryError|Native memory allocation.*failed|There is insufficient memory",
                         re.IGNORECASE)
GC_THREADS = ["-XX:ParallelGCThreads=2", "-XX:ConcGCThreads=1"]
PROGRESS_PERCENT = re.compile(rb"(?<![\d.])(\d{1,3}(?:\.\d+)?)%")
# Resident memory of a JVM exceeds -Xmx (metaspace, thread stacks, GC structures). Observed: 6.83 GB
# RSS at -Xmx6g. The scheduler reserves this much per cell.
RSS_FACTOR, RSS_EXTRA = 1.15, 256 * 2**20
PRINT_LOCK = threading.Lock()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def log(message):
    with PRINT_LOCK:
        print(f"[{datetime.datetime.now():%H:%M:%S}] {message}", flush=True)


def load_config(upstream, exp):
    raw = git_bytes(upstream, CONFIGS[exp])
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def parse_xmx(spec):
    """'default=6g,NZEnergy=8g,NZEnergy:SGBT(SRP(FIRTDD))=26g' -> {key: size}."""
    mapping = {}
    for part in spec.split(","):
        key, value = part.rsplit("=", 1)
        mapping[key.strip()] = value.strip()
    if "default" not in mapping:
        raise argparse.ArgumentTypeError("--xmx needs a default=<size> entry")
    return mapping


def heap_bytes(size):
    units = {"k": 2**10, "m": 2**20, "g": 2**30}
    return int(size[:-1]) * units[size[-1].lower()] if size[-1].lower() in units else int(size)


def cell_xmx(mapping, dataset, method):
    return mapping.get(f"{dataset}:{method}", mapping.get(dataset, mapping["default"]))


def heap_for(args, cell):
    """Heap of one cell: the --retry-oom size after an OOM, otherwise from the --xmx map."""
    return args.heap_override.get(cell) or cell_xmx(args.xmx, cell[1], cell[2])


def cell_folder(results, cell):
    exp, dataset, method, _, _, seed = cell
    return results / "raw" / exp / dataset / method / f"seed={'none' if seed is None else seed}"


def reserve_bytes(xmx):
    return int(heap_bytes(xmx) * RSS_FACTOR) + RSS_EXTRA


def build_command(config, learner, seed_flag, seed, java, jar, xmx, exit_on_oom=False):
    if seed_flag and seed is not None:
        learner = f"{learner} -{seed_flag} {seed}"  # the seed goes on the outer learner only
    jvm = config["java_options"].split()
    heap = [i for i, option in enumerate(jvm) if option.startswith("-Xmx")]
    if len(heap) != 1:
        raise ValueError("Expected exactly one -Xmx option in the authors' java_options")
    paper_xmx = jvm[heap[0]]
    jvm[heap[0]] = f"-Xmx{xmx}"
    if exit_on_oom:
        # Stop the JVM at the first OutOfMemoryError instead of letting MOA catch it and keep
        # thrashing the garbage collector for hours.
        jvm.insert(heap[0] + 1, "-XX:+ExitOnOutOfMemoryError")
    # By default each JVM starts about 20 garbage-collector threads on a 30-CPU machine; with 16 JVMs
    # those threads took most of the CPU and starved the single-threaded learners. Two are enough.
    jvm[heap[0] + 1:heap[0] + 1] = GC_THREADS
    task =f'{config["evaluator"]} -s (ArffFileStream -f stream.arff) -l ({learner}) -d evaluation.csv'
    return [str(java), "-classpath", str(jar), *jvm, task], paper_xmx


def run_identity(command):
    """The part of a command that defines the run: JVM options and the MOA task.

    The java and jar paths are left out, and so is the jar digest (recorded per cell): Maven builds of
    the same pinned source are not byte-reproducible. The heap size, the OOM flag and the number of
    garbage-collector threads only decide whether and how fast a run fits in memory, not its result,
    so they are left out too (the full command is recorded).
    """
    return [part for part in command[3:]
            if part != "-XX:+ExitOnOutOfMemoryError" and not part.startswith("-Xmx")
            and part not in GC_THREADS]


def reusable(record, identity, dataset_sha, config_hash):
    """A completed run of the same MOA task, data and config, even if it ran with another heap or from
    other java/jar paths (for example on another machine, from an unpacked sgbr_results_*.tgz)."""
    return (record.get("status") == "completed" and run_identity(record.get("command", [])) == identity
            and record.get("dataset_sha256") == dataset_sha and record.get("config_sha256") == config_hash)


def report_status(cells, args, configs, data):
    """Print which cells are finished and which a rerun would still do, without running anything."""
    states, totals = {}, {}
    for cell in cells:
        exp, dataset, method, seed_flag, learner, seed = cell
        config, config_hash = configs[exp]
        path = cell_folder(args.results, cell) / "record.json"
        record = json.loads(path.read_text()) if path.is_file() else {}
        command, _ = build_command(config, learner, seed_flag, seed, args.java, args.jar, heap_for(args, cell))
        if reusable(record, run_identity(command), data.get(dataset)[1]["sha256"], config_hash):
            state = "done"
        elif record.get("status") == "completed":
            state = "outdated"  # finished, but with another command, dataset file or config
        else:
            state = record.get("status") or "not run"
        totals.setdefault(exp, {}).setdefault(state, 0)
        totals[exp][state] += 1
        if state != "done":
            states.setdefault((exp, dataset, method, state), []).append("-" if seed is None else str(seed))
    for exp, counts in totals.items():
        print(f"{exp}: {sum(counts.values())} cells, " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    if states:
        print("Not done (a rerun of the same command does these):")
        for (exp, dataset, method, state), seeds in sorted(states.items()):
            print(f"  {exp} {dataset:14s} {method:24s} {state:8s} seeds {','.join(seeds)}")


def final_row(csv_path):
    if not csv_path.is_file():
        return None
    last = None
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            last = row
    return last


def java_oom(folder):
    for path in [folder / "java.log", *folder.glob("hs_err_pid*.log")]:
        with path.open("rb") as handle:
            if any(OOM_PATTERN.search(line) for line in handle):
                return True
    return False


def extract_dataset(upstream, dataset, folder):
    """Write exp/RDatasets/<dataset>.arff (or the .arff.zip) of the pinned commit to folder/stream.arff."""
    target = folder / "stream.arff"
    name = f"exp/RDatasets/{dataset}.arff"
    with target.open("wb") as stream:
        direct = subprocess.run(["git", "-C", str(upstream), "show", f"{UPSTREAM_COMMIT}:{name}"],
                                stdout=stream, stderr=subprocess.PIPE, check=False)
    if direct.returncode:
        archive = folder / "original.zip"
        try:
            with archive.open("wb") as file:
                zipped = subprocess.run(["git", "-C", str(upstream), "show", f"{UPSTREAM_COMMIT}:{name}.zip"],
                                        stdout=file, stderr=subprocess.PIPE, check=False)
            if zipped.returncode:
                raise ValueError(f"The pinned upstream commit has no ARFF for {dataset}.")
            with zipfile.ZipFile(archive) as bundle, bundle.open(f"{dataset}.arff") as source, \
                    target.open("wb") as stream:
                shutil.copyfileobj(source, stream)
        finally:
            archive.unlink(missing_ok=True)
    return target


def dataset_evidence(path):
    digest, in_data, count = hashlib.sha256(), False, 0
    with path.open("rb") as stream:
        for line in stream:
            digest.update(line)
            value = line.strip()
            if in_data:
                count += bool(value and not value.startswith(b"%"))
            elif value.lower() == b"@data":
                in_data = True
    if not count:
        raise ValueError(f"{path} has no data rows.")
    return {"sha256": digest.hexdigest(), "expected_instances": count}


def extra_arff(extra, dataset):
    """<extra>/<dataset>.arff, a dataset supplied outside the authors' repository (e.g. SUP2I), or None."""
    path = extra / f"{dataset}.arff" if extra else None
    return path.resolve() if path and path.is_file() else None


class DataCache:
    """Materialise each pinned ARFF once and remember its digest and row count.

    A file <extra>/<dataset>.arff, when present, is used instead (it is how datasets missing from the
    authors' repository, such as SUP2I, SUP3A, SUP3G, are added later). Its digest is recorded per run.
    """

    def __init__(self, upstream, root, extra=None):
        self.upstream, self.root, self.extra = upstream, root, extra
        self.lock, self.items = threading.Lock(), {}

    def get(self, dataset):
        with self.lock:
            supplied = extra_arff(self.extra, dataset)
            if supplied and dataset not in self.items:
                self.items[dataset] = (supplied, dict(dataset_evidence(supplied), source=str(supplied)))
            if dataset not in self.items:
                folder = self.root / dataset
                folder.mkdir(parents=True, exist_ok=True)
                meta, path = folder / "evidence.json", folder / "stream.arff"
                if meta.is_file() and path.is_file():
                    evidence = json.loads(meta.read_text())
                else:
                    extract_dataset(self.upstream, dataset, folder)
                    evidence = dataset_evidence(path)
                    meta.write_text(json.dumps(evidence))
                self.items[dataset] = (path, evidence)
            return self.items[dataset]


def run_cell(cell, args, config, config_hash, data, jar_sha):
    exp, dataset, method, seed_flag, learner, seed = cell
    seed_label = "none" if seed is None else str(seed)
    folder = cell_folder(args.results, cell)
    record_path = folder / "record.json"
    xmx = heap_for(args, cell)
    command, paper_xmx = build_command(config, learner, seed_flag, seed, args.java, args.jar, xmx,
                                       args.exit_on_oom)
    stream, evidence = data.get(dataset)
    identity = run_identity(command)
    command_hash = hashlib.sha256(json.dumps([identity, evidence["sha256"], config_hash]).encode()).hexdigest()
    if record_path.is_file():
        old = json.loads(record_path.read_text())
        if reusable(old, identity, evidence["sha256"], config_hash):
            return old, True
    folder.mkdir(parents=True, exist_ok=True)
    for name in ("evaluation.csv", "java.log", "stream.arff"):
        (folder / name).unlink(missing_ok=True)
    os.symlink(stream, folder / "stream.arff")
    env = {k: v for k, v in os.environ.items() if k not in JAVA_OPTION_VARIABLES}
    record = {
        "exp": exp, "dataset": dataset, "method_key": method, "seed": seed, "command": command,
        "command_line": shlex.join(command), "command_hash": command_hash, "config": CONFIGS[exp],
        "config_sha256": config_hash, "upstream_commit": UPSTREAM_COMMIT, "jar_sha256": jar_sha,
        "xmx": f"-Xmx{xmx}", "paper_xmx": paper_xmx, "dataset_sha256": evidence["sha256"],
        "expected_instances": evidence["expected_instances"], "concurrency": args.jobs,
        "host": os.uname().nodename, "cpu_model": args.cpu_model, "started_at": now(),
    }
    log(f"start     {exp} {dataset} {method} seed={seed_label} (-Xmx{xmx})")
    started = time.monotonic()
    with (folder / "java.log").open("wb") as out:
        process = subprocess.Popen(command, cwd=folder, stdout=out, stderr=subprocess.STDOUT, env=env)
        try:
            _, status, usage = os.wait4(process.pid, 0)
            process.returncode = os.waitstatus_to_exitcode(status)
            peak = usage.ru_maxrss * 1024
        except ChildProcessError:
            process.wait()
            peak = None
    elapsed = time.monotonic() - started
    (folder / "stream.arff").unlink(missing_ok=True)
    row = final_row(folder / "evaluation.csv")
    processed = None
    if row is not None:
        try:
            processed = int(float(row[SEEN] if SEEN in row else row[CLASSIFIED]))
        except (KeyError, ValueError):
            processed = None
    oom = java_oom(folder)
    if (not oom and process.returncode == 0 and processed is None
            and peak is not None and peak >= 0.9 * heap_bytes(xmx)):
        # MOA can catch OutOfMemoryError, print "null" and exit 0 without writing its CSV
        # (observed for NZEnergy SGBT(SRP(FIRTDD)) at -Xmx6g). Heap-sized RSS and no output is an OOM.
        oom = True
    if process.returncode == 0 and processed == evidence["expected_instances"] and not oom:
        status = "completed"
    elif oom:
        status = "oom"
    else:
        status = "failed"
    with (folder / "java.log").open("rb") as handle:
        handle.seek(max(0, (folder / "java.log").stat().st_size - 65536))
        progress = PROGRESS_PERCENT.findall(handle.read())
    record.update({
        "last_progress_percent": float(progress[-1]) if progress else None,
        "finished_at": now(), "exit_code": process.returncode, "external_wall_seconds": elapsed,
        "peak_rss_bytes": peak, "processed_instances": processed, "status": status,
        "r2": float(row[R2]) if row and status == "completed" else None,
        "moa_wall_seconds": float(row[WALL]) if row and status == "completed" else None,
    })
    record_path.write_text(json.dumps(record, indent=1))
    log(f"{status:9s} {exp} {dataset} {method} seed={seed_label} R2={record['r2']} "
        f"wall={record['moa_wall_seconds']} rss={peak and round(peak / 2**30, 2)}GiB")
    return record, False


def estimated_seconds(method, dataset):
    """Paper Tab. 4 mean wall time for one run on the authors' machine; 60 s if not tabulated.

    The E4/E5 grid variants (SGBT_<...>(Oza_<...>)) are estimated by Sgb(Oza) on the same dataset.
    """
    from summarize_e1 import DATASETS, METHODS, TAB4
    names = {label: paper for paper, label, _ in METHODS}
    if method.startswith("SGBT_"):
        method = "SGBT(Oza(FIRTDD))"
    if method in names and dataset in DATASETS:
        return TAB4[names[method]][DATASETS.index(dataset)]
    return 60.0


def schedule(cells, args, run):
    """Run cells in order, at most args.jobs at once and within args.ram_budget bytes of reserved heap.

    When memory frees up, the first waiting cell that fits starts, so long cells keep priority and
    small cells fill the gaps. Returns the status counts and the cells that ran out of memory.
    """
    need = {cell: reserve_bytes(heap_for(args, cell)) for cell in cells}
    pending, running, counts, oom = list(cells), {}, {}, []
    with futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        while pending or running:
            used = sum(size for _, size in running.values())
            while len(running) < args.jobs:
                pick = next((c for c in pending if used + need[c] <= args.ram_budget), None)
                if pick is None:
                    break
                pending.remove(pick)
                running[pool.submit(run, pick)] = (pick, need[pick])
                used += need[pick]
            done, _ = futures.wait(running, return_when=futures.FIRST_COMPLETED)
            for job in done:
                cell, _ = running.pop(job)
                record, skipped = job.result()
                key = "skipped" if skipped else record["status"]
                counts[key] = counts.get(key, 0) + 1
                if key == "oom":
                    oom.append(cell)
    return counts, oom


def size_arg(text):
    return heap_bytes(text) if text[-1].lower() in "kmg" else int(float(text) * 2**30)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--toolchain", type=Path, default=Path.home() / "sgbr" / "toolchain.json",
                        help="Written by exp/toolchain.py; supplies --upstream, --java and --jar")
    parser.add_argument("--upstream", type=Path)
    parser.add_argument("--java")
    parser.add_argument("--jar", type=Path)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--data-cache", type=Path, help="Default: data/ next to the toolchain file")
    parser.add_argument("--extra-data", type=Path, default=Path("extra_data"),
                        help="Folder for datasets not in the authors' repository (<dataset>.arff, e.g. SUP2I.arff); "
                             "with --plan repro, those present are run")
    parser.add_argument("--exp", nargs="+", choices=sorted(CONFIGS), default=["E1"],
                        help="One or more experiments; their cells share one queue")
    parser.add_argument("--plan", choices=["repro"],
                        help="repro: datasets shipped in the authors' repo plus any in --extra-data, "
                             "AXGBr only on the 10 smaller ones")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--exclude-methods", nargs="+", default=[])
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--jobs", type=int, default=1, help="Maximum parallel JVMs")
    parser.add_argument("--ram-budget", type=size_arg, default=math.inf,
                        help="e.g. 150g: start a cell only while the reserved memory "
                             f"({RSS_FACTOR} x -Xmx + 256 MiB per JVM) stays within this budget")
    parser.add_argument("--xmx", type=parse_xmx, default=parse_xmx("default=6g"),
                        help="Heap per cell: default=<size>[,<dataset>=<size>][,<dataset>:<method>=<size>]")
    parser.add_argument("--exit-on-oom", action="store_true",
                        help="Add -XX:+ExitOnOutOfMemoryError so an OOM ends the run immediately")
    parser.add_argument("--retry-oom", metavar="SIZE",
                        help="Run cells that ran out of memory (now or in an earlier run) again with this heap")
    parser.add_argument("--list", action="store_true", help="Print the cells and exit without running")
    parser.add_argument("--status", action="store_true",
                        help="Print which cells are finished in --results and which are not, then exit")
    args = parser.parse_args(argv)

    if args.toolchain.is_file():
        tools = json.loads(args.toolchain.read_text())
        args.upstream = args.upstream or Path(tools["upstream"])
        args.java = args.java or tools["java"]
        args.jar = args.jar or Path(tools["jar"])
    if not (args.upstream and args.java and args.jar):
        parser.error("Run exp/toolchain.py first, or pass --upstream, --java and --jar")
    args.data_cache = args.data_cache or args.toolchain.parent / "data"

    configs, cells = {}, []
    for exp in args.exp:
        config, config_hash = configs[exp] = load_config(args.upstream, exp)
        # SUP* ARFFs are not in the authors' repo; they run once supplied in --extra-data.
        shipped = [d for d in config["datasets"] if not d.startswith("SUP") or extra_arff(args.extra_data, d)]
        if args.datasets:
            datasets = [d for d in args.datasets if d in config["datasets"]]
        else:
            datasets = shipped if args.plan else config["datasets"]
        learners = [entry for entry in config["learners"] if entry[0] not in args.exclude_methods]
        if args.methods:
            learners = [entry for entry in learners if entry[0] in args.methods]
        seeds = args.seeds or config["random_seeds"]
        for dataset in datasets:
            for name, seed_flag, learner in learners:
                if args.plan == "repro" and name == "AXGBr" and dataset not in CORE10:
                    continue
                # Learners without a seed option are deterministic and run once per dataset.
                for seed in (seeds if seed_flag else [None]):
                    cells.append((exp, dataset, name, seed_flag, learner, seed))
    unsupplied = sorted({d for c, _ in configs.values() for d in c["datasets"]
                         if d.startswith("SUP") and not extra_arff(args.extra_data, d)})
    if args.plan and unsupplied:
        log(f"Not in the authors' repo, skipped until supplied as {args.extra_data}/<name>.arff: "
            f"{', '.join(unsupplied)}")
    every = {d for c, _ in configs.values() for d in c["datasets"]}
    every_method = {entry[0] for c, _ in configs.values() for entry in c["learners"]}
    if set(args.datasets or []) - every:
        parser.error(f"Datasets not in any selected config: {sorted(set(args.datasets) - every)}")
    if set(args.methods or []) - every_method:
        parser.error(f"Methods not in any selected config: {sorted(set(args.methods) - every_method)}")

    # Cells that already ran out of memory start directly with the --retry-oom heap.
    args.heap_override = {}
    if args.retry_oom:
        for cell in cells:
            record = cell_folder(args.results, cell) / "record.json"
            if record.is_file() and json.loads(record.read_text()).get("status") == "oom":
                args.heap_override[cell] = args.retry_oom
    cells.sort(key=lambda c: (-estimated_seconds(c[2], c[1]), c[0], c[1], c[2], c[5] or 0))  # longest first
    if args.retry_oom and reserve_bytes(args.retry_oom) > args.ram_budget:
        parser.error(f"--retry-oom {args.retry_oom} does not fit in --ram-budget")
    too_big = [c for c in cells if reserve_bytes(heap_for(args, c)) > args.ram_budget]
    if too_big:
        parser.error(f"{len(too_big)} cells need more than --ram-budget, e.g. {too_big[0][1]} {too_big[0][2]}")
    total = sum(estimated_seconds(c[2], c[1]) for c in cells) / 3600
    log(f"{len(cells)} cells ({', '.join(args.exp)}), ≈{total:.1f} CPU-h at the paper's speed, {args.jobs} "
        f"parallel JVMs, RAM budget {args.ram_budget / 2**30:.0f} GiB, heap map {args.xmx}"
        + (f", OOM retry at -Xmx{args.retry_oom}" if args.retry_oom else ""))
    if args.list:
        for c in cells:
            print(f"{c[0]} {c[1]:14s} {c[2]:24s} seed={c[5]}  -Xmx{heap_for(args, c)}  "
                  f"≈{estimated_seconds(c[2], c[1]):.0f} s")
        return 0
    data = DataCache(args.upstream, args.data_cache, args.extra_data)
    if args.status:
        report_status(cells, args, configs, data)
        return 0

    with open("/proc/cpuinfo") as info:
        args.cpu_model = next((line.split(":", 1)[1].strip() for line in info if line.startswith("model name")), None)
    jar_sha = hashlib.sha256(args.jar.read_bytes()).hexdigest()

    def run(cell):
        return run_cell(cell, args, *configs[cell[0]], data, jar_sha)

    counts, oom = schedule(cells, args, run)
    retry = [c for c in oom if args.retry_oom and heap_bytes(args.retry_oom) > heap_bytes(heap_for(args, c))]
    if retry:
        log(f"{len(retry)} cells ran out of memory; running them again with -Xmx{args.retry_oom}")
        for cell in retry:
            args.heap_override[cell] = args.retry_oom
        counts["oom"] -= len(retry)
        more, _ = schedule(retry, args, run)
        for key, value in more.items():
            counts[key] = counts.get(key, 0) + value
    counts = {k: v for k, v in counts.items() if v}
    log(f"done: {counts}")
    return 0 if set(counts) <= {"completed", "skipped"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
