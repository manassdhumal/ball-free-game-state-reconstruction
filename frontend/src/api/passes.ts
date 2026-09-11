import { apiFetch } from "./client";
export const getPassCandidates = (matchId: string, frame: number, playerId: string) =>
  apiFetch<Record<string, unknown>>(`/pass-candidates/${matchId}/${frame}/${playerId}`);
export const getPassRanking = (matchId: string, frame: number, playerId: string) =>
  apiFetch<Record<string, unknown>>(`/pass-ranking/${matchId}/${frame}/${playerId}`);
