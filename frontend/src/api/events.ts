import { apiFetch } from "./client";
export const getEvents = (matchId: string) =>
  apiFetch<Record<string, unknown>>(`/events/${matchId}`);
export const getEventMetrics = (matchId: string, tolerance = 1) =>
  apiFetch<Record<string, unknown>>(`/events/${matchId}/metrics?tolerance=${tolerance}`);
