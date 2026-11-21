#!/usr/bin/env python3
"""Provision JDK 11 and Maven, fetch the authors' pinned MOA source and build the MOA fat jar.

Everything is cached under --work (default ~/sgbr), so rerunning is cheap. The script writes
<work>/toolchain.json with the paths that exp/run_matrix.py needs.

Usage (Linux x86_64):
  python3 exp/toolchain.py [--work ~/sgbr]
"""

import argparse
import csv
import datetime
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile
from pathlib import Path

# Authors' MOA fork and the commit the paper's experiments were run from.
UPSTREAM = "https://github.com/nuwangunasekara/SGBR"
UPSTREAM_COMMIT = "f62f7b437512e020f57c4bcd03884c4724eda5b1"

FAT_JAR_NAME = "moa-2023.04.1-SNAPSHOT-jar-with-dependencies.jar"
# SHA-256 of one earlier build of the pinned source. Maven builds are not byte-reproducible,
# so a different digest is expected and only recorded.
REFERENCE_MOA_SHA256 = "9be332b07354c13237ead9a88d431a855ada14003395bb70386efc5f56ac4216"

# MEKA 1.9.7 in the pinned MOA fork is Java 11 bytecode, so Java 8 cannot build it.
JDK_DIRECTORY = "jdk-11.0.29+7"
JDK_NAME = "OpenJDK11U-jdk_x64_linux_hotspot_11.0.29_7.tar.gz"
JDK_URL = f"https://github.com/adoptium/temurin11-binaries/releases/download/jdk-11.0.29%2B7/{JDK_NAME}"
JDK_SHA256 = "3c8f2b53dd137cd86e54f40df96fd0fc56df72c749c06469e7eab216503bc7cf"
MAVEN_DIRECTORY = "apache-maven-3.9.11"
MAVEN_NAME = "apache-maven-3.9.11-bin.tar.gz"
MAVEN_URL = f"https://repo.maven.apache.org/maven2/org/apache/maven/apache-maven/3.9.11/{MAVEN_NAME}"
MAVEN_SHA512 = (
    "bcfe4fe305c962ace56ac7b5fc7a08b87d5abd8b7e89027ab251069faebee516b0"
    "ded8961445d6d91ec1985dfe30f8153268843c89aa392733d1a3ec956c9978"
)


def stage(message):
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    print(f"[{stamp}] {message}", flush=True)


def probe(args):
    """Run a short command and return its exit code and output without raising."""
    try:
        done = subprocess.run(args, check=False, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"exit_code": None, "output": str(error)}
    return {"exit_code": done.returncode, "output": (done.stdout + done.stderr).strip()}


def command(args, cwd, timeout=120):
    result = subprocess.run(args, cwd=cwd, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=timeout, check=False)
    if result.returncode:
        raise RuntimeError(f"{args[0]} exited {result.returncode}\n{result.stdout[-5000:]}")
    return result.stdout


def digest(path, algorithm="sha256"):
    hasher = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def git_bytes(repo, path):
    """Read one file from the pinned upstream commit."""
    return subprocess.check_output(["git", "-C", str(repo), "show", f"{UPSTREAM_COMMIT}:{path}"])


def unpack_tool(work, directory, url, expected, algorithm):
    with tempfile.TemporaryDirectory(prefix=".download-", dir=work) as temporary:
        staging = Path(temporary)
        archive = staging / url.rsplit("/", 1)[1]
        with urllib.request.urlopen(url, timeout=60) as response, archive.open("wb") as stream:
            shutil.copyfileobj(response, stream)
        if digest(archive, algorithm) != expected:
            raise RuntimeError(f"Checksum mismatch for {archive.name}")
        command(["tar", "-xzf", str(archive), "-C", str(staging)], staging)
        (staging / directory).rename(work / directory)


def select_jdk(cache):
    cached = cache / JDK_DIRECTORY
    candidates = [cached]
    javac = shutil.which("javac")
    if javac:
        candidates.append(Path(javac).resolve().parent.parent)
    candidates.extend(sorted(Path("/usr/lib/jvm").glob("*11*")))
    for home in candidates:
        compiler = probe([str(home / "bin/javac"), "-version"])
        java = probe([str(home / "bin/java"), "-version"])
        if (compiler["exit_code"] == java["exit_code"] == 0
                and re.search(r"\bjavac 11\.", compiler["output"])
                and 'version "11.' in java["output"]):
            return home.resolve()
    if platform.machine() != "x86_64":
        raise RuntimeError("No usable JDK 11 found. The verified Temurin download requires x86_64.")
    if cached.exists():
        raise RuntimeError(f"Cached JDK at {cached} failed verification. Remove it and rerun.")
    stage(f"Downloading {JDK_NAME}")
    unpack_tool(cache, JDK_DIRECTORY, JDK_URL, JDK_SHA256, "sha256")
    return cached.resolve()


def select_maven(cache):
    bundled = cache / MAVEN_DIRECTORY / "bin/mvn"
    if bundled.is_file():
        return bundled
    maven = shutil.which("mvn")
    if maven and probe([maven, "-version"])["exit_code"] == 0:
        return Path(maven)
    stage(f"Downloading {MAVEN_NAME}")
    unpack_tool(cache, MAVEN_DIRECTORY, MAVEN_URL, MAVEN_SHA512, "sha512")
    return bundled


def provision(cache):
    jdk = select_jdk(cache)
    java = jdk / "bin/java"
    os.environ["JAVA_HOME"] = str(jdk)
    os.environ["PATH"] = f"{java.parent}{os.pathsep}{os.environ['PATH']}"
    java_version = command([str(java), "-version"], cache).strip()
    stage(f"Java: {java_version.splitlines()[0]}")
    mvn = select_maven(cache)
    os.environ["PATH"] = f"{mvn.parent}{os.pathsep}{os.environ['PATH']}"
    maven_version = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", command([str(mvn), "-version"], cache)).strip()
    runtime = re.search(r"runtime:\s*(.+)", maven_version)
    if runtime is None or Path(runtime[1].strip()).resolve() != jdk:
        raise RuntimeError("Maven did not pick up the selected JAVA_HOME.")
    stage(f"Maven: {maven_version.splitlines()[0]}")
    return java, mvn, {"java_home": str(jdk), "java_version": java_version, "maven_version": maven_version}


def upstream_checkout(cache):
    source = cache / "upstream"
    if not source.exists():
        source.mkdir()
        command(["git", "init", "-q"], source)
    if probe(["git", "-C", str(source), "cat-file", "-e", f"{UPSTREAM_COMMIT}^{{commit}}"])["exit_code"] != 0:
        stage(f"Fetching {UPSTREAM} @ {UPSTREAM_COMMIT[:7]}")
        command(["git", "fetch", "--depth=1", UPSTREAM, UPSTREAM_COMMIT], source, timeout=900)
    actual = command(["git", "rev-parse", "--verify", f"{UPSTREAM_COMMIT}^{{commit}}"], source).strip()
    if actual != UPSTREAM_COMMIT:
        raise RuntimeError(f"Expected MOA revision {UPSTREAM_COMMIT}, found {actual}.")
    command(["git", "checkout", "-q", "--detach", actual], source)
    command(["git", "diff", "--quiet", "HEAD", "--"], source)
    return source


def verified_jar(path):
    if not path.is_file():
        return False
    try:
        with zipfile.ZipFile(path) as jar:
            names = set(jar.namelist())
    except zipfile.BadZipFile:
        return False
    return {"moa/DoTask.class", "moa/classifiers/meta/StreamingGradientBoostedTrees.class"} <= names


def build_or_reuse(source, artifacts, mvn, tools):
    jar = artifacts / FAT_JAR_NAME
    provenance = artifacts / "moa_build.json"
    identity = {"upstream_commit": UPSTREAM_COMMIT, "build_java_version": tools["java_version"],
                "build_maven_version": tools["maven_version"]}
    try:
        cached = json.loads(provenance.read_text()) if provenance.is_file() else {}
    except (OSError, ValueError):
        cached = {}
    reuse = (isinstance(cached, dict) and all(cached.get(k) == v for k, v in identity.items())
             and verified_jar(jar) and cached.get("sha256") == digest(jar))
    if not reuse:
        stage("Building the MOA jar with Maven (several minutes)")
        log = artifacts / "moa_build.log"
        with log.open("wb") as output:
            result = subprocess.run(
                [str(mvn), "-B", "-ntp", "-C", "clean", "install", "-DskipTests=true",
                 "-Dmaven.javadoc.skip=true", "-Dlatex.skipBuild=true"],
                cwd=source / "moa", stdout=output, stderr=subprocess.STDOUT, timeout=3600, check=False)
        if result.returncode:
            raise RuntimeError(f"MOA build failed with exit {result.returncode}. See {log}.")
        built = source / "moa/target" / FAT_JAR_NAME
        if not verified_jar(built):
            raise RuntimeError(f"The MOA build did not produce a usable fat jar at {built}.")
        staged = jar.with_suffix(".jar.tmp")
        shutil.copyfile(built, staged)
        staged.replace(jar)
    checksum = cached["sha256"] if reuse else digest(jar)
    details = {**identity, "path": str(jar), "sha256": checksum, "size_bytes": jar.stat().st_size,
               "reference_sha256": REFERENCE_MOA_SHA256,
               "matches_reference_sha256": checksum == REFERENCE_MOA_SHA256}
    provenance.write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8")
    return jar, {**details, "reused": reuse}


def smoke(work, jar, java):
    """Run SGBT(Oza) on an eight-row ARFF and check that MOA writes one result row."""
    (work / "smoke.csv").unlink(missing_ok=True)
    fixture = "@relation probe\n@attribute x numeric\n@attribute z numeric\n@attribute y numeric\n@data\n"
    fixture += "".join(f"{i},{i % 2},{i * 2}\n" for i in range(1, 9))
    (work / "probe.arff").write_text(fixture, encoding="utf-8")
    task = ("EvaluatePrequentialRegression -s (ArffFileStream -f probe.arff) "
            "-l (meta.StreamingGradientBoostedTrees -t -L 1.0 -m 75 -s 10 "
            "-l (meta.OzaBag -s 10 -l (trees.FIMTDD -s VarianceReductionSplitCriterion "
            "-g 50 -c 0.01 -e)) -Z 1) -i 8 -f 8 -d smoke.csv")
    command([str(java), "-Xmx256m", "-cp", str(jar), "moa.DoTask", task], work, timeout=120)
    with (work / "smoke.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1 or float(rows[0]["learning evaluation instances"]) != 8:
        raise RuntimeError(f"Smoke test expected one MOA row with eight instances, got {rows}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work", type=Path, default=Path.home() / "sgbr")
    args = parser.parse_args(argv)
    work = args.work.expanduser().resolve()
    cache, artifacts = work / ".sgbr-cache", work / "artifacts"
    for directory in (cache, artifacts):
        directory.mkdir(parents=True, exist_ok=True)
    java, mvn, tools = provision(cache)
    source = upstream_checkout(cache)
    jar, build = build_or_reuse(source, artifacts, mvn, tools)
    stage(f"MOA jar {'reused' if build['reused'] else 'built'}: {jar} (SHA-256 {build['sha256'][:12]}…)")
    smoke(artifacts, jar, java)
    stage("Smoke test passed")
    info = {"java": str(java), "jar": str(jar), "upstream": str(source), "upstream_commit": UPSTREAM_COMMIT,
            "jar_sha256": build["sha256"], "java_version": tools["java_version"].splitlines()[0],
            "maven_version": tools["maven_version"].splitlines()[0]}
    (work / "toolchain.json").write_text(json.dumps(info, indent=1) + "\n", encoding="utf-8")
    stage(f"Wrote {work / 'toolchain.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
