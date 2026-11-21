#!/usr/bin/env python3
"""Run E3: bias-variance decomposition against the number of boosting iterations (paper Sec. 4.2,
Figs. 2, 7 and 8). Resumable: each finished (dataset, method, iterations, fold) task is stored and
skipped on rerun.

The protocol is the one in the authors' exp/Bias_Variance_Decomp/BVDecompSyntheticLatest.py at the
pinned commit:

- Methods, with the labels of the authors' result files:
    B    = Sgbt       SGBT with FIRT-DD base learners
    BB   = Sgb(Oza)   SGBT whose base learner is OzaBag of 5 FIRT-DD trees
    BagB = Oza(Sgbt)  OzaBag of 5 SGBT models
  Boosting iterations 5, 10, ..., 100; all other options as in E1.
- Datasets without drift: KFold(n_splits=10) without shuffling. In each fold,
  mlxtend.evaluate.bias_variance_decomp(loss="mse", num_rounds=10, random_seed=42) trains a new
  model on each of 10 bootstrap samples of the training fold and predicts the test fold.
- FriedmanGra/Gsg/FriedmanLea: train on <name>_1 ... <name>_10, test on <name>_12. These split files
  are not in the authors' repository; pass --split-dir if you have them.
- Expected loss, bias and variance are averaged over the 10 folds (or splits), and the square roots
  of the mean bias and mean variance are reported.

The authors wrapped MOA learners for scikit-learn with CapyMOA. This script drives the same MOA jar
through JPype and performs the same steps: a new learner per fit, configured from the CLI string,
prepareForUse, resetLearning and the stream header as model context, one trainOnInstance per
training row (float32 data, as loaded by the authors), and getVotesForInstance for each test row.

Usage (Linux, after exp/toolchain.py; needs numpy scipy scikit-learn mlxtend JPype1):
  python3 exp/run_e3.py --jobs 16 [--datasets abalone bike] [--methods B BB BagB] [--iterations 5 10]
"""

import argparse
import concurrent.futures as futures
import csv
import datetime
import hashlib
import io
import json
import math
import multiprocessing
import os
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

from toolchain import UPSTREAM_COMMIT  # noqa: E402

TREE = "trees.FIMTDD -s VarianceReductionSplitCriterion -g 50 -c 0.01 -e"
SGBT = "moa.classifiers.meta.StreamingGradientBoostedTrees"
OZA = "moa.classifiers.meta.OzaBag"
# label -> (paper name, MOA class, CLI template, relative cost per boosting iteration)
METHODS = {
    "B": ("Sgbt", SGBT, "-t -L 1.0 -m 75 -s {b} -l (" + TREE + ")", 1.0),
    "BB": ("Sgb(Oza)", SGBT, "-t -L 1.0 -m 75 -s {b} -l (meta.OzaBag -s 5 -l (" + TREE + "))", 2.0),
    "BagB": ("Oza(Sgbt)", OZA, "-s 5 -l (meta.StreamingGradientBoostedTrees -t -L 1.0 -m 75 -s {b} -l (" + TREE + "))",
             5.0),
}
STATIC = ["House8L", "MetroTraffic", "abalone", "ailerons", "bike", "elevators", "fried"]
DRIFT = ["FriedmanGra", "FriedmanGsg", "FriedmanLea"]
ITERATIONS = list(range(5, 105, 5))
FOLDS = 10
ROUNDS, SEED = 10, 42
FIELDS = ["dataset", "method", "boosting_iterations", "mean_error", "sqrt_mean_bias", "sqrt_mean_var"]


def log(message):
    print(f"[{datetime.datetime.now():%H:%M:%S}] {message}", flush=True)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


# ----------------------------------------------------------------------------------------------
# Worker side (one JVM per worker process)
# ----------------------------------------------------------------------------------------------

_DATA = {}


def start_jvm(java, jar, xmx):
    """Start the JDK 11 that exp/toolchain.py selected (java = <home>/bin/java)."""
    import jpype
    libjvm = Path(java).resolve().parent.parent / "lib" / "server" / "libjvm.so"
    jpype.startJVM(str(libjvm), f"-Xmx{xmx}", "-Xss10m", classpath=[str(jar)], convertStrings=False)
    import jpype.imports  # noqa: F401  (enables "from com.yahoo... import ...")


def load_arff(path):
    """As the authors: scipy's ARFF reader, then a float32 array; the last column is the target."""
    import numpy as np
    from scipy.io import arff
    if path not in _DATA:
        data, _ = arff.loadarff(str(path))
        array = np.array(data.tolist(), dtype=np.float32)
        _DATA[path] = (array[:, :-1], array[:, -1])
    return _DATA[path]


class MoaRegressor:
    """The minimal scikit-learn interface that mlxtend needs, backed by a MOA learner."""

    def __init__(self, learner_class, cli):
        self.learner_class, self.cli = learner_class, cli

    def _header(self, features):
        from com.yahoo.labs.samoa.instances import Attribute, Instances, InstancesHeader
        from java.util import ArrayList
        attributes = ArrayList()
        for i in range(features):
            attributes.add(Attribute(f"attrib_{i}"))
        attributes.add(Attribute("target"))
        instances = Instances("stream", attributes, 0)
        instances.setClassIndex(features)
        return InstancesHeader(instances)

    def _instances(self, X, y):
        import numpy as np
        from com.yahoo.labs.samoa.instances import DenseInstance
        from jpype import JArray, JDouble
        rows = np.column_stack([X.astype(np.float64), y.astype(np.float64)])
        for row in rows:
            instance = DenseInstance(1.0, JArray(JDouble)(row))
            instance.setDataset(self.header)
            yield instance

    def fit(self, X, y):
        import jpype
        self.header = self._header(X.shape[1])
        learner = jpype.JClass(self.learner_class)()
        learner.getOptions().setViaCLIString(self.cli)
        learner.prepareForUse()
        learner.resetLearning()
        learner.setModelContext(self.header)
        for instance in self._instances(X, y):
            learner.trainOnInstance(instance)
        self.learner = learner
        return self

    def predict(self, X):
        import numpy as np
        out = np.zeros(X.shape[0], dtype=np.float64)
        for i, instance in enumerate(self._instances(X, np.zeros(X.shape[0], dtype=np.float32))):
            votes = self.learner.getVotesForInstance(instance)
            out[i] = votes[0] if len(votes) else 0.0
        return out


def run_task(task):
    from mlxtend.evaluate import bias_variance_decomp
    started = time.monotonic()
    if task["kind"] == "kfold":
        from sklearn.model_selection import KFold
        X, y = load_arff(task["data"])
        train, test = list(KFold(n_splits=FOLDS).split(X))[task["fold"] - 1]
        X_train, y_train, X_test, y_test = X[train], y[train], X[test], y[test]
    else:
        X_train, y_train = load_arff(task["train"])
        X_test, y_test = load_arff(task["test"])
    _, learner_class, template, _ = METHODS[task["method"]]
    model = MoaRegressor(learner_class, template.format(b=task["b"]))
    loss, bias, var = bias_variance_decomp(model, X_train, y_train, X_test, y_test,
                                           loss="mse", random_seed=SEED, num_rounds=ROUNDS)
    return {"expected_loss": float(loss), "bias": float(bias), "var": float(var),
            "train_rows": int(len(y_train)), "test_rows": int(len(y_test)),
            "seconds": time.monotonic() - started}


# ----------------------------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------------------------

def file_sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def task_path(results, dataset, method, b, fold):
    return results / "raw" / "E3" / dataset / method / f"b={b}" / f"fold={fold}.json"


def build_tasks(args, sources):
    tasks = []
    for dataset in args.datasets:
        for method in args.methods:
            for b in args.iterations:
                for fold in args.folds:
                    task = {"dataset": dataset, "method": method, "b": b, "fold": fold,
                            "cli": METHODS[method][2].format(b=b), **sources[dataset](fold)}
                    task["cost"] = task.pop("rows") * b * METHODS[method][3]
                    tasks.append(task)
    tasks.sort(key=lambda t: -t["cost"])  # longest first
    return tasks


def summarize(results, upstream, datasets, methods, iterations):
    rows = []
    for dataset in datasets:
        for method in methods:
            for b in iterations:
                found = [task_path(results, dataset, method, b, k) for k in range(1, FOLDS + 1)]
                records = [json.loads(p.read_text()) for p in found if p.is_file()]
                records = [r for r in records if r.get("status") == "completed"]
                if len(records) < FOLDS:
                    continue
                rows.append({
                    "dataset": dataset, "method": method, "boosting_iterations": b,
                    "mean_error": sum(r["expected_loss"] for r in records) / FOLDS,
                    "sqrt_mean_bias": math.sqrt(sum(r["bias"] for r in records) / FOLDS),
                    "sqrt_mean_var": math.sqrt(sum(r["var"] for r in records) / FOLDS),
                })
    out = results / "summary"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "E3_bias_variance.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    reference = authors_results(upstream, datasets)
    parity = []
    for row in rows:
        ref = reference.get((row["dataset"], row["method"], row["boosting_iterations"]))
        if ref:
            parity.append({**{k: row[k] for k in FIELDS[:3]},
                           **{f"delta_{k}": row[k] - ref[k] for k in FIELDS[3:]}})
    with (out / "E3_parity.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS[:3] + [f"delta_{k}" for k in FIELDS[3:]],
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(parity)
    if parity:
        worst = max(abs(p[f"delta_{k}"]) for p in parity for k in FIELDS[3:])
        log(f"{len(rows)} points complete; {len(parity)} compared with the authors' RESULTS, "
            f"max |difference| = {worst:.3e}")
    else:
        log(f"{len(rows)} points complete; none comparable with the authors' RESULTS yet")
    return rows


def authors_results(upstream, datasets):
    """The authors' committed curves: BVDecompose_Results_<d>.csv (B, BB) and ..._BagB_<d>.csv."""
    found = {}
    for dataset in datasets:
        for name in (f"BVDecompose_Results_{dataset}.csv", f"BVDecompose_Results_BagB_{dataset}.csv"):
            try:
                text = subprocess.check_output(
                    ["git", "-C", str(upstream), "show",
                     f"{UPSTREAM_COMMIT}:exp/Bias_Variance_Decomp/RESULTS/{name}"],
                    stderr=subprocess.DEVNULL, text=True)
            except subprocess.CalledProcessError:
                continue
            for row in csv.DictReader(io.StringIO(text)):
                found[(row["dataset"], row["method"], int(row["boosting_iterations"]))] = {
                    k: float(row[k]) for k in FIELDS[3:]}
    return found


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--toolchain", type=Path, default=Path.home() / "sgbr" / "toolchain.json")
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--data-cache", type=Path, help="Default: data/ next to the toolchain file")
    parser.add_argument("--datasets", nargs="+", default=STATIC, choices=STATIC + DRIFT,
                        help="Default: the 7 datasets without drift (the Friedman* split files are not shipped)")
    parser.add_argument("--split-dir", type=Path,
                        help="Folder with FriedmanGra_1 ... FriedmanGra_10 and FriedmanGra_12 (likewise Gsg, Lea)")
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    parser.add_argument("--iterations", nargs="+", type=int, default=ITERATIONS)
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(1, FOLDS + 1)),
                        help="Run only these folds (for a quick test); the summary needs all 10")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel worker processes, one JVM each")
    parser.add_argument("--xmx", default="4g", help="Heap per worker JVM")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args(argv)

    tools = json.loads(args.toolchain.read_text())
    upstream, jar = Path(tools["upstream"]), Path(tools["jar"])
    data_cache = args.data_cache or args.toolchain.parent / "data"
    if args.summary_only:
        summarize(args.results, upstream, args.datasets, args.methods, args.iterations)
        return 0

    from run_matrix import DataCache
    cache = DataCache(upstream, data_cache)
    sources = {}
    for dataset in args.datasets:
        if dataset in DRIFT:
            if args.split_dir is None:
                parser.error(f"{dataset} needs --split-dir with {dataset}_1 ... {dataset}_10 and {dataset}_12")
            train = [args.split_dir / f"{dataset}_{i}" for i in range(1, FOLDS + 1)]
            test = args.split_dir / f"{dataset}_12"
            missing = [str(p) for p in train + [test] if not p.is_file()]
            if missing:
                parser.error(f"Missing split files: {missing[:3]}")
            sources[dataset] = (lambda fold, train=train, test=test: {
                "kind": "splits", "train": str(train[fold - 1]), "test": str(test),
                "data_sha256": file_sha(train[fold - 1]) + file_sha(test), "rows": 10_000})
        else:
            path, evidence = cache.get(dataset)
            sources[dataset] = (lambda fold, path=path, evidence=evidence: {
                "kind": "kfold", "data": str(path), "data_sha256": evidence["sha256"],
                "rows": evidence["expected_instances"]})

    tasks, skipped = [], 0
    for task in build_tasks(args, sources):
        path = task_path(args.results, task["dataset"], task["method"], task["b"], task["fold"])
        if path.is_file():
            old = json.loads(path.read_text())
            if old.get("status") == "completed" and old.get("cli") == task["cli"] \
                    and old.get("data_sha256") == task["data_sha256"]:
                skipped += 1
                continue
        tasks.append(task)
    log(f"{len(tasks)} tasks to run, {skipped} already done, {args.jobs} workers, -Xmx{args.xmx} each")

    import mlxtend
    import numpy
    import sklearn
    versions = {"numpy": numpy.__version__, "scikit-learn": sklearn.__version__, "mlxtend": mlxtend.__version__,
                "python": sys.version.split()[0], "jar_sha256": tools.get("jar_sha256"),
                "upstream_commit": UPSTREAM_COMMIT}
    counts = {"completed": 0, "failed": 0}
    context = multiprocessing.get_context("spawn")
    with futures.ProcessPoolExecutor(max_workers=args.jobs, mp_context=context, initializer=start_jvm,
                                     initargs=(tools["java"], str(jar), args.xmx)) as pool:
        jobs = {pool.submit(run_task, task): task for task in tasks}
        for job in futures.as_completed(jobs):
            task = jobs[job]
            path = task_path(args.results, task["dataset"], task["method"], task["b"], task["fold"])
            path.parent.mkdir(parents=True, exist_ok=True)
            record = {k: task[k] for k in ("dataset", "method", "b", "fold", "cli", "data_sha256")}
            record.update({"seed": SEED, "rounds": ROUNDS, "xmx": args.xmx, "host": os.uname().nodename,
                           "finished_at": now(), **versions})
            try:
                record.update(job.result(), status="completed")
                counts["completed"] += 1
            except Exception as error:  # noqa: BLE001 - keep going, record the failure
                record.update(status="failed", error=repr(error))
                counts["failed"] += 1
            path.write_text(json.dumps(record, indent=1))
            log(f"{record['status']:9s} {task['dataset']} {task['method']} b={task['b']} fold={task['fold']} "
                f"loss={record.get('expected_loss')} ({record.get('seconds', 0):.0f} s)"
                + (f" {record['error']}" if "error" in record else ""))
    log(f"done: {counts}, skipped {skipped}")
    summarize(args.results, upstream, args.datasets, args.methods, args.iterations)
    return 0 if counts["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
