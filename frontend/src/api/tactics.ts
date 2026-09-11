import { apiFetch } from "./client";
export const getTacticalState = (matchId: string, frame: number) =>
  apiFetch<Record<string, unknown>>(`/tactical/${matchId}/${frame}`);
