import time
import json
import sys
from pathlib import Path

# Ensure project root is on sys.path so `src` imports resolve when executed as a script
# scripts/ is one level under project root, so parents[1] is the repo root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.validation.robustness_experiment import load_experiment_config, make_grid, run_single_runspec, RunSpec


def select_pilot(grid):
    # select window 0_60s only, seeds 72,73, severities mild/moderate/severe
    selected = []
    for r in grid:
        if r.window_label != "0_60s":
            continue
        if r.seed not in (72, 73):
            continue
        if r.severity not in ("mild", "moderate", "severe"):
            continue
        selected.append(r)
    return selected


def main():
    cfg = load_experiment_config("configs/robustness_experiment.yaml")
    grid = make_grid(cfg)
    pilot = select_pilot(grid)
    assert len(pilot) == 6, f"Expected 6 runspecs, found {len(pilot)}"

    root = Path(__file__).resolve().parents[2]
    timings = []
    results = []
    start_all = time.perf_counter()
    for runspec in pilot:
        start = time.perf_counter()
        out = run_single_runspec(runspec, cfg, root=root)
        end = time.perf_counter()
        dur = end - start
        timings.append({"window": runspec.window_label, "seed": runspec.seed, "severity": runspec.severity, "duration_s": dur})
        # Extract compact metrics
        st = out["summary_table"]
        # Ensure three conditions
        assert set(st['condition'].tolist()) >= {"CLEAN", "SYNTHETIC-DEGRADED", "NOISE-MITIGATED"}
        # Map metrics
        metrics = {}
        for _, row in st.iterrows():
            cond = row['condition']
            metrics[cond] = {
                'possession_f1': row.get('possession_f1', None),
                'tactical_mean_absolute_score_error': row.get('tactical_mean_absolute_score_error', None),
            }
        results.append({"runspec": {"window": runspec.window_label, "seed": runspec.seed, "severity": runspec.severity}, "metrics": metrics, "paths": out.get('paths', {})})

    total_dur = time.perf_counter() - start_all
    report = {"timings": timings, "results": results, "total_duration_s": total_dur}
    outpath = Path('results') / 'runs' / 'robustness_expansion' / 'pilot_report.json'
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(json.dumps(report, indent=2))
    print(f"Pilot completed: {len(pilot)} runspecs, total_duration_s={total_dur:.2f}")
    print("Report written to:", outpath)


if __name__ == '__main__':
    main()
