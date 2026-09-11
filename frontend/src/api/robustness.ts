import { apiFetch } from "./client";
export const getRobustnessSummary = () => apiFetch<Record<string, unknown>>("/robustness/summary");
export const getRobustness = (experiment: string) =>
  apiFetch<Record<string, unknown>>(`/robustness/${experiment}`);
