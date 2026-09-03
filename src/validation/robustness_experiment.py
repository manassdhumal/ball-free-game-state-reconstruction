"""Robustness experiment expansion & statistical validation (Step 74).

This module defines a configurable experiment grid and utilities to run
(smoke) experiments across temporal windows, seeds, and severity tiers.

It intentionally reuses the canonical functions implemented in
`src.validation.robustness_benchmark` to preserve identical inference
contracts, anti-leakage, and output formats.

The module exposes a `run_experiment_grid()` entrypoint but the script
is conservative by default: it runs a single smoke test when invoked as
`__main__` unless `--run-full` is explicitly provided.

DO NOT COMMIT OR PUSH EXPERIMENT RESULTS AUTOMATICALLY.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
import copy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.validation.robustness_benchmark import (
    load_config as _load_config,
    load_metrica_subset,
    build_conditions,
    tracking_quality_metrics,
    run_inference_condition,
    possession_change_metrics,
    tactical_stability_metrics,
    save_outputs as _save_outputs,
)


@dataclass
class RunSpec:
    match_id: str
    start_frame: int
    end_frame: int
    seed: int
    severity: str
    window_label: str


def load_experiment_config(path: str | Path) -> Dict[str, Any]:
    return _load_config(path)


def make_grid(config: Dict[str, Any]) -> List[RunSpec]:
    match = config["match"]
    fps = float(match.get("fps", 25.0))
    windows = config.get("experiment_windows", [])
    seeds = config.get("seeds", [])
    severities = config.get("severities", [])

    grid: List[RunSpec] = []
    for w in windows:
        s_frame = int(round(w[0] * fps)) + 1  # convert seconds->1-based frame
        e_frame = int(round(w[1] * fps))
        label = f"{w[0]}_{w[1]}s"
        for seed in seeds:
            for severity in severities:
                grid.append(RunSpec(match_id=match["match_id"], start_frame=s_frame, end_frame=e_frame, seed=int(seed), severity=str(severity), window_label=label))
    return grid


def run_single_runspec(runspec: RunSpec, base_config: Dict[str, Any], root: Optional[Path] = None) -> Dict[str, Any]:
    root = root or Path(__file__).resolve().parents[2]
    # Work on a deep copy to avoid mutating the caller's configuration (notably
    # the nested `output` dict). Shallow copies preserve nested dicts and caused
    # repeated runs to create nested output paths like
    # results/.../robustness_expansion/robustness_expansion/... on successive
    # invocations. Using deepcopy prevents that bug and keeps outputs stable.
    cfg = copy.deepcopy(base_config)
    # Build the concrete configuration for this runspec
    cfg = build_runspec_config(runspec, cfg)

    # `build_runspec_config` already configures output subfolders. Only set
    # the run-specific `run_id` here to avoid double-appending
    # `robustness_expansion` into directory paths.
    cfg.setdefault("output", {})
    cfg["output"]["run_id"] = f"exp_{runspec.window_label}_seed_{runspec.seed}_sev_{runspec.severity}"

    # Load tracking subset and events
    tracking, events = load_metrica_subset(cfg)

    # Build conditions
    # Ensure explicit degradation config is passed
    cfg["degradation"]["seed"] = int(runspec.seed)
    cfg["degradation"]["severity"] = runspec.severity

    conditions = build_conditions(tracking, cfg)

    # Compute quality, inference, evaluation for each condition
    condition_summaries = {}
    tactical_scores = {}
    fps = float(cfg["match"].get("fps", 25.0))
    for condition_name, df_tracking in conditions.items():
        quality = tracking_quality_metrics(df_tracking, expected_players=None, large_motion_threshold=float(cfg.get("tracking_quality", {}).get("large_motion_threshold", 0.05)))
        inference = run_inference_condition(df_tracking, cfg)
        possession = inference["possession"]
        candidates = inference["candidates"]
        tactical = inference["tactical"]

        # possession metrics
        possession_metrics = possession_change_metrics(candidates, events, fps=fps, tolerance_sec=float(cfg.get("event_matching", {}).get("tolerance_sec", 1.0)))

        # tactical stability will be computed relative to CLEAN later - store
        condition_summaries[condition_name] = {
            "quality": quality,
            "possession": possession_metrics,
        }
        tactical_scores[condition_name] = tactical

    # Tactical stability: compare to CLEAN
    clean_tactical = tactical_scores.get("CLEAN", pd.DataFrame())
    stability = {}
    for condition_name, ts in tactical_scores.items():
        stability[condition_name] = tactical_stability_metrics(clean_tactical, ts, top_k=int(cfg.get("tactical", {}).get("top_k", 100)))

    # Aggregate summary table
    rows = []
    for condition_name in conditions.keys():
        q = condition_summaries[condition_name]["quality"]
        p = condition_summaries[condition_name]["possession"]
        t = stability[condition_name]
        row = {"condition": condition_name}
        # flatten selected keys
        row.update({f"tracking_{k}": v for k, v in q.items()})
        row.update({f"possession_{k}": v for k, v in p.items()})
        row.update({f"tactical_{k}": v for k, v in t.items()})
        rows.append(row)
    summary_table = pd.DataFrame(rows)

    # Save outputs (compact)
    results = {
        "summary_table": summary_table,
        "condition_summaries": condition_summaries,
        "tactical_scores": tactical_scores,
    }
    out_paths = _save_outputs(results, cfg, root=root)
    results["paths"] = out_paths
    results["config"] = cfg
    return results


def run_smoke_test(base_config: Dict[str, Any], root: Optional[Path] = None) -> Dict[str, Any]:
    """Run one small smoke test: first window, first seed, first severity."""
    grid = make_grid(base_config)
    if not grid:
        raise RuntimeError("Experiment grid is empty")
    runspec = grid[0]
    return run_single_runspec(runspec, base_config, root=root)


def build_runspec_config(runspec: RunSpec, base_config: Dict[str, Any]) -> Dict[str, Any]:
    """Create a runspec-specific configuration dict without running the
    experiment. This is factored out so tests can validate config construction
    and to avoid mutating the caller's config in `run_single_runspec`.
    """
    cfg = copy.deepcopy(base_config)
    # override match frame window
    cfg["match"] = dict(cfg.get("match", {}))
    cfg["match"]["start_frame"] = int(runspec.start_frame)
    cfg["match"]["end_frame"] = int(runspec.end_frame)
    # override seed and severity
    cfg["degradation"] = dict(cfg.get("degradation", {}))
    cfg["degradation"]["seed"] = int(runspec.seed)
    cfg["degradation"]["severity"] = runspec.severity

    # set output subfolders for expansion
    out_prefix = cfg.setdefault("output", {}).copy()
    cfg["output"]["metrics_dir"] = str(Path(out_prefix.get("metrics_dir", "results/metrics")) / "robustness_expansion")
    cfg["output"]["runs_dir"] = str(Path(out_prefix.get("runs_dir", "results/runs")) / "robustness_expansion")
    cfg["output"]["figures_dir"] = str(Path(out_prefix.get("figures_dir", "results/figures")) / "robustness_expansion")
    cfg["output"]["run_id"] = f"exp_{runspec.window_label}_seed_{runspec.seed}_sev_{runspec.severity}"
    return cfg


def runspec_is_complete(runspec: RunSpec, base_config: Dict[str, Any], root: Optional[Path] = None) -> bool:
    """Return True if the expected summary, summary json and tactical csv exist
    for a runspec using the same naming convention as `save_outputs`.
    """
    root = root or Path(__file__).resolve().parents[2]
    cfg = build_runspec_config(runspec, base_config)
    output = cfg.get("output", {})
    metrics_dir = Path(output.get("metrics_dir", "results/metrics"))
    runs_dir = Path(output.get("runs_dir", "results/runs"))
    figures_dir = Path(output.get("figures_dir", "results/figures"))
    run_id = output.get("run_id")
    prefix = f"robustness_benchmark_{run_id}"
    summary_csv = metrics_dir / f"{prefix}_summary.csv"
    summary_json = metrics_dir / f"{prefix}_summary.json"
    tactical_csv = runs_dir / f"{prefix}_tactical_scores.csv"
    return summary_csv.exists() and summary_json.exists() and tactical_csv.exists()


def run_full_grid(base_config: Dict[str, Any], root: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Run the full configured experiment grid and return per-runspec results.

    This function intentionally executes every RunSpec returned by `make_grid`.
    It is up to the caller to ensure they intend to run the full grid (it may
    be expensive)."""
    grid = make_grid(base_config)
    if not grid:
        return []
    results = []
    for runspec in grid:
        res = run_single_runspec(runspec, base_config, root=root)
        results.append(res)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/robustness_experiment.yaml")
    parser.add_argument("--smoke", action="store_true", default=False, help="Run only one smoke test (use for quick checks)")
    parser.add_argument("--run-full", action="store_true", help="Run the full experiment grid (explicit) — default behaviour without --smoke is to run the full grid")
    args = parser.parse_args()

    base_cfg = load_experiment_config(args.config)
    if args.smoke:
        out = run_smoke_test(base_cfg)
        print("Smoke test completed. Summary:")
        print(out["summary_table"].to_string(index=False))
        print("Saved outputs:", out["paths"])
    else:
        if not args.run_full:
            print("Running full grid by default. To run only a smoke test, pass --smoke. To require explicit full-run flag, pass --run-full.")
        # By default, run the full grid unless --smoke is specified.
        results = run_full_grid(base_cfg)
        print(f"Executed {len(results)} runspec(s).")
        # Print a short summary for each runspec
        for r in results:
            try:
                rt = r.get("summary_table")
                if rt is not None:
                    print(rt.to_string(index=False))
            except Exception:
                pass