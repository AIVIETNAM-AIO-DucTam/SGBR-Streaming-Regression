#!/usr/bin/env bash
# Full reproduction on one Linux machine: toolchain, every experiment, tables and figures.
#
#   bash exp/run_all.sh                 all steps: setup moa e3 figures pack
#   bash exp/run_all.sh moa figures     only the named steps, in the order given
#
# Steps
#   setup    JDK 11, Maven, the authors' MOA source and jar under ~/sgbr; Python venv ~/sgbr/venv
#   moa      E1, E2, E4, E5 with the authors' MOA commands, in one shared queue (exp/run_matrix.py)
#   e3       E3 bias-variance decomposition (exp/run_e3.py)
#   figures  tables in results/summary/ and PNG figures in results/figures/ for whatever has finished
#   pack     results/ and logs/ into sgbr_results_<date>.tgz, to copy off the machine
#
# Every step resumes: after an interruption, run the same command again.
#
# Settings (environment variables, all optional)
#   JOBS         parallel runs                       default: number of vCPUs
#   RAM_GB       memory the runs may reserve (GB)    default: total RAM minus 10 GB
#   EXPERIMENTS  MOA experiments                     default: "E1 E2 E4 E5"
#   MOA_XMX      heap map for the MOA runs           default: 6g, 26g for the two heavy NZEnergy cells
#   RETRY_XMX    heap for MOA runs that ran out of memory at the first try   default: 12g
#   E3_XMX       heap per E3 worker JVM              default: 4g
set -uo pipefail
cd "$(dirname "$0")/.."

VENV=$HOME/sgbr/venv
PY=$VENV/bin/python
JOBS=${JOBS:-$(nproc)}
RAM_GB=${RAM_GB:-$(awk '/MemTotal/ {print int($2 / 1048576) - 10}' /proc/meminfo)}
EXPERIMENTS=${EXPERIMENTS:-"E1 E2 E4 E5"}
MOA_XMX=${MOA_XMX:-"default=6g,NZEnergy:SGBT(SRP(FIRTDD))=26g,NZEnergy:SRP(SGBT(FIRTDD))=26g"}
RETRY_XMX=${RETRY_XMX:-12g}
E3_XMX=${E3_XMX:-4g}
# An E3 worker needs its heap plus about 1 GB for Python and the JVM itself.
E3_JOBS=$(( JOBS < RAM_GB / (${E3_XMX%g} + 1) ? JOBS : RAM_GB / (${E3_XMX%g} + 1) ))
[ "$E3_JOBS" -lt 1 ] && E3_JOBS=1
STEPS=("$@")
[ ${#STEPS[@]} -eq 0 ] && STEPS=(setup moa e3 figures pack)
mkdir -p logs
STATUS=0

say() { printf '\n=== %s  %s\n' "$(date '+%F %T')" "$*"; }

need_venv() {
  [ -x "$PY" ] || { echo "No Python environment at $VENV; run: bash exp/run_all.sh setup"; exit 1; }
}

step_setup() {
  python3 exp/toolchain.py || exit 1
  if [ ! -x "$PY" ]; then
    python3 -m venv "$VENV" || { echo "python3 -m venv failed; install it with: apt-get install -y python3-venv"; exit 1; }
  fi
  "$PY" -m pip install --quiet --upgrade pip && "$PY" -m pip install --quiet -r requirements.txt || exit 1
}

step_moa() {
  need_venv
  # shellcheck disable=SC2086  # EXPERIMENTS is a list
  "$PY" exp/run_matrix.py --exp $EXPERIMENTS --plan repro --jobs "$JOBS" --ram-budget "${RAM_GB}g" \
    --xmx "$MOA_XMX" --exit-on-oom --retry-oom "$RETRY_XMX" 2>&1 | tee -a logs/moa.log || STATUS=1
}

step_e3() {
  need_venv
  "$PY" exp/run_e3.py --jobs "$E3_JOBS" --xmx "$E3_XMX" 2>&1 | tee -a logs/e3.log || STATUS=1
}

step_figures() {
  need_venv
  if compgen -G "results/raw/E1/*" > /dev/null; then
    "$PY" exp/summarize_e1.py && "$PY" exp/plot_e1.py || STATUS=1
  fi
  "$PY" exp/plot_e2.py || STATUS=1
  "$PY" exp/plot_e4_e5.py || STATUS=1
  if [ -f results/summary/E3_bias_variance.csv ]; then
    "$PY" exp/plot_e3.py --with-authors || STATUS=1
  fi
  "$PY" exp/plot_e3.py --source authors || STATUS=1   # the authors' own E3 curves, for comparison
}

step_pack() {
  local archive
  archive="sgbr_results_$(date +%Y%m%d_%H%M).tgz"
  tar czf "$archive" results logs && echo "Wrote $archive ($(du -h "$archive" | cut -f1))"
}

say "settings: JOBS=$JOBS RAM_GB=$RAM_GB EXPERIMENTS=\"$EXPERIMENTS\" E3_JOBS=$E3_JOBS steps: ${STEPS[*]}"
for step in "${STEPS[@]}"; do
  case $step in
    setup|moa|e3|figures|pack) say "step $step"; "step_$step" ;;
    *) echo "Unknown step '$step' (use: setup moa e3 figures pack)"; exit 2 ;;
  esac
done
if [ $STATUS -ne 0 ]; then
  say "finished, but some runs did not complete; see logs/ and the status field in results/raw/*/.../record.json"
else
  say "finished"
fi
exit $STATUS
