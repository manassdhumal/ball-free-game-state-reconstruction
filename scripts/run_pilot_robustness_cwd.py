import time
import json
import sys
from pathlib import Path

# Use current working directory as project root when invoked from repo root
ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))

from src.validation.robustness_experiment import load_experiment_config, make_grid, run_single_runspec


def select_pilot(grid):
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

    timings = []
    results = []
    start_all = time.perf_counter()
    for runspec in pilot:
        print(f"Starting runspec: window={runspec.window_label} seed={runspec.seed} severity={runspec.severity}")
        start = time.perf_counter()
        out = run_single_runspec(runspec, cfg, root=ROOT)
        end = time.perf_counter()
        dur = end - start
        print(f"Completed runspec in {dur:.2f}s")
        timings.append({"window": runspec.window_label, "seed": runspec.seed, "severity": runspec.severity, "duration_s": dur})
        st = out["summary_table"]
        assert set(st['condition'].tolist()) >= {"CLEAN", "SYNTHETIC-DEGRADED", "NOISE-MITIGATED"}
        metrics = {}
        for _, row in st.iterrows():
            cond = row['condition']
            metrics[cond] = {
                'possession_f1': row.get('possession_f1', None),
                'tactical_mean_absolute_score_error': row.get('tactical_mean_absolute_score_error', None),
            }
        # Ensure any Path objects in paths are converted to strings for JSON
        paths = out.get('paths', {}) or {}
        paths_serializable = {k: (str(v) if v is not None else None) for k, v in paths.items()}
        results.append({"runspec": {"window": runspec.window_label, "seed": runspec.seed, "severity": runspec.severity}, "metrics": metrics, "paths": paths_serializable})

    total_dur = time.perf_counter() - start_all
    report = {"timings": timings, "results": results, "total_duration_s": total_dur}
    outpath = Path('results') / 'runs' / 'robustness_expansion' / 'pilot_report.json'
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(json.dumps(report, indent=2, default=str))
    print(f"Pilot completed: {len(pilot)} runspecs, total_duration_s={total_dur:.2f}")
    print("Report written to:", outpath)


if __name__ == '__main__':
    main()
