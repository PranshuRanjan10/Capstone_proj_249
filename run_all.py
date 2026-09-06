"""
End-to-end pipeline for CARE-Paddy.

Usage:
  python run_all.py                # full pipeline
  python run_all.py --skip-sens    # skip sensitivity analysis (saves ~10 min)
  python run_all.py --only build   # run a single step by name

Steps:
  build     build_dataset.py    assemble multi-season dataset
  train     train.py            LOSO quantile XGBoost + RF + AdaBoost vs persistence
  evaluate  evaluate.py         false-actuation rate, gating curve, all figures
  explain   explain.py          TreeSHAP beeswarm + waterfall
  tests     pytest tests -q     unit tests (hard constraint, gate, thresholds)
  sens      sensitivity.py      ±30% perturbation of percolation + Kc [optional]
"""
from __future__ import annotations
import argparse, subprocess, sys, time, textwrap
from pathlib import Path

ROOT = Path(__file__).parent
SRC  = ROOT / "src"

STEPS = [
    ("build",    SRC / "build_dataset.py"),
    ("train",    SRC / "train.py"),
    ("evaluate", SRC / "evaluate.py"),
    ("explain",  SRC / "explain.py"),
]
TESTS = ("tests", None)      # handled separately via pytest
SENS  = ("sens", SRC / "sensitivity.py")

_COL = {
    "green":  "\033[92m",
    "yellow": "\033[93m",
    "red":    "\033[91m",
    "bold":   "\033[1m",
    "reset":  "\033[0m",
}


def _fmt(msg: str, col: str) -> str:
    return f"{_COL[col]}{msg}{_COL['reset']}"


def _banner(title: str):
    line = "=" * 70
    print(f"\n{_fmt(line, 'bold')}\n  {_fmt(title, 'bold')}\n{_fmt(line, 'bold')}")


def run_step(label: str, script: Path | None, extra_args: list[str] | None = None) -> bool:
    _banner(f"[{label.upper()}] {script.name if script else 'pytest'}")
    t0 = time.time()

    if script is None:
        # pytest
        cmd = [sys.executable, "-m", "pytest", "tests", "-q", "--tb=short"]
    else:
        cmd = [sys.executable, str(script)] + (extra_args or [])

    result = subprocess.run(cmd, cwd=ROOT)
    elapsed = time.time() - t0

    if result.returncode == 0:
        print(_fmt(f"\n  ✓ {label} completed in {elapsed:.1f}s", "green"))
        return True
    else:
        print(_fmt(f"\n  ✗ {label} FAILED (exit {result.returncode}) after {elapsed:.1f}s", "red"))
        return False


def preflight_check():
    """Verify key prerequisites before starting the pipeline."""
    issues = []
    warnings = []

    cfg_path = ROOT / "config" / "config.yaml"
    if not cfg_path.exists():
        issues.append("config/config.yaml not found")

    # Check weather source
    try:
        import yaml
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        src = cfg.get("weather", {}).get("source", "synthetic")
        if src == "power":
            power_csv = ROOT / cfg["weather"]["power_csv"]
            if not power_csv.exists():
                issues.append(
                    f"weather.source is 'power' but {power_csv} does not exist. "
                    f"Run: python src/fetch_power.py"
                )
            else:
                import pandas as pd
                df = pd.read_csv(power_csv, nrows=5)
                print(_fmt(f"  ✓ NASA POWER CSV found: {power_csv.name}", "green"))
        elif src == "synthetic":
            warnings.append("Using synthetic weather. Run fetch_power.py for real data.")

        # Check Open-Meteo archive
        om_path = ROOT / cfg["weather"].get("openmeteo_csv", "data/raw/openmeteo_forecast.csv")
        if om_path.exists():
            print(_fmt(f"  ✓ Open-Meteo forecast archive found: {om_path.name}", "green"))
        else:
            warnings.append(
                f"Open-Meteo forecast archive not found. "
                f"Run: python src/fetch_openmeteo.py  (upgrades forecast training feature)"
            )

        # Check SoilGrids
        sg_path = ROOT / cfg.get("soilgrids", {}).get("data_file", "data/raw/soilgrids.json")
        if sg_path.exists():
            print(_fmt(f"  ✓ SoilGrids data found: {sg_path.name}", "green"))
        else:
            warnings.append(
                f"SoilGrids data not found. "
                f"Run: python src/fetch_soilgrids.py  (sources the percolation parameter)"
            )

    except Exception as e:
        warnings.append(f"Could not fully check config: {e}")

    # Required packages
    for pkg in ["xgboost", "shap", "streamlit", "sklearn"]:
        try:
            __import__(pkg if pkg != "sklearn" else "sklearn")
        except ImportError:
            issues.append(f"Missing package: {pkg}  (pip install -r requirements.txt)")

    for w in warnings:
        print(_fmt(f"  ⚠  {w}", "yellow"))
    for e in issues:
        print(_fmt(f"  ✗ {e}", "red"))

    return len(issues) == 0


def main():
    ap = argparse.ArgumentParser(description="CARE-Paddy end-to-end pipeline")
    ap.add_argument("--skip-sens", action="store_true",
                    help="Skip sensitivity analysis (saves ~10 min)")
    ap.add_argument("--only", choices=[s[0] for s in STEPS] + ["tests", "sens"],
                    help="Run only one step")
    args = ap.parse_args()

    _banner("CARE-Paddy pipeline — pre-flight check")
    if not preflight_check():
        sys.exit("Pre-flight check failed. Fix the issues above before running.")

    t_start = time.time()
    failed = []

    if args.only:
        steps_to_run = []
        if args.only == "tests":
            steps_to_run = [TESTS]
        elif args.only == "sens":
            steps_to_run = [SENS]
        else:
            steps_to_run = [(s, p) for s, p in STEPS if s == args.only]
    else:
        steps_to_run = STEPS + [TESTS]
        if not args.skip_sens:
            steps_to_run.append(SENS)
        else:
            print(_fmt("\n  (Skipping sensitivity analysis — use --only sens to run later)", "yellow"))

    for label, script in steps_to_run:
        ok = run_step(label, script)
        if not ok:
            failed.append(label)
            # Hard failure: build and train must succeed for downstream steps
            if label in ("build", "train"):
                print(_fmt(
                    f"\n  Pipeline halted at '{label}' — downstream steps depend on this output.",
                    "red"
                ))
                break

    elapsed_total = time.time() - t_start
    _banner("Pipeline summary")
    print(f"  Total time: {elapsed_total:.1f}s")
    if failed:
        print(_fmt(f"  Failed steps: {', '.join(failed)}", "red"))
        sys.exit(1)
    else:
        print(_fmt("  All steps passed.", "green"))
        print(textwrap.dedent(f"""
  Outputs:
    Figures  →  outputs/figures/
    Metrics  →  outputs/metrics/
    Models   →  outputs/models/

  Dashboard:
    streamlit run dashboard/app.py

  Optional upgrades not yet run:
    python src/fetch_openmeteo.py   (genuine forecast rain from Open-Meteo archive)
    python src/fetch_soilgrids.py   (sourced soil percolation parameter)
    python src/sensitivity.py       (run separately if you skipped above)
        """))


if __name__ == "__main__":
    main()
