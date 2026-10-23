import { apiFetch } from "./client";

export interface MatchEvent extends Record<string, unknown> {
  event_id?: number | string;
  frame?: number;
  start_frame?: number;
  timestamp?: number;
  start_timestamp?: number;
  type?: string;
  event_type?: string;
  team?: string;
  player_id?: string;
}

export interface EventMetric extends Record<string, unknown> {
  event_type?: string;
  true_positive?: number;
  false_positive?: number;
  false_negative?: number;
  precision?: number;
  recall?: number;
  f1?: number;
}

export const getEvents = (matchId: string) =>
  apiFetch<{ availability: { status: string }; data: MatchEvent[] }>(`/events/${matchId}`);
export const getEventMetrics = (matchId: string, tolerance = 1) =>
  apiFetch<{ availability: { status: string }; data: EventMetric[] }>(
    `/events/${matchId}/metrics?tolerance=${tolerance}`,
  );
