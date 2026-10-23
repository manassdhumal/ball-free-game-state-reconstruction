import { apiFetch } from "./client";

export interface AggregateMetric {
  count: number;
  mean: number;
  median: number;
  std: number;
  min: number;
  max: number;
  ci95_mean_lo: number;
  ci95_mean_hi: number;
}

export interface EvaluationSummary {
  n_runs: number;
  n_conditions: number;
  summary_by_condition: Record<
    string,
    {
      possession: AggregateMetric;
      tactical: AggregateMetric;
      tracking: AggregateMetric;
    }
  >;
  possession_recovery: {
    n: number;
    abs_improvement: number;
    recovery_fraction: number;
  };
  tactical_recovery: {
    n: number;
    abs_improvement: number;
    recovery_fraction: number;
  };
}

export interface Step80Run {
  condition?: string;
  severity?: string;
  seed?: number;
  window?: string;
  mitigation_method?: string;
  missingness?: number;
  possession_precision?: number;
  possession_recall?: number;
  possession_f1?: number;
  tas_mae?: number;
  unresolved_gap_count?: number;
}

export interface Step83Stability {
  method?: string;
  perturbation?: string;
  severity?: string;
  n_states?: number;
  top_1_retention?: number;
  top_3_retention?: number;
  spearman?: number;
  kendall?: number;
  average_rank_displacement?: number;
}

export function getEvaluationSummary() {
  return apiFetch<{
    availability: { status: string; message?: string };
    data: EvaluationSummary | null;
  }>("/evaluation/summary");
}

export function getRobustnessRows(experiment: string) {
  return apiFetch<{
    availability: { status: string; message?: string };
    data: Array<Record<string, unknown>>;
  }>(`/robustness/${experiment}`);
}

export function getStep80Runs() {
  return apiFetch<{
    availability: { status: string; message?: string };
    data: Step80Run[];
  }>("/robustness/step80");
}

export function getStep83Stability() {
  return apiFetch<{
    availability: { status: string; message?: string };
    data: Step83Stability[];
  }>("/robustness/step83");
}
