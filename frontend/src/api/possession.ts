import { apiFetch } from "./client";
export const getPossession = (matchId: string, frame: number) =>
  apiFetch<Record<string, unknown>>(`/possession/${matchId}/${frame}`);
export const getPossessionTimeline = (matchId: string) =>
  apiFetch<Record<string, unknown>>(`/possession/${matchId}/timeline`);
