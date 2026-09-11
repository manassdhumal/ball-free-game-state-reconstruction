import { apiFetch } from "./client";
import type { ApiEnvelope } from "./types";
export const getPlayers = (matchId: string) =>
  apiFetch<ApiEnvelope<Array<{ player_id: string; team: "home" | "away" }>>>(`/players/${matchId}`);
export const getTrajectory = (matchId: string, playerId: string, limit = 250) =>
  apiFetch<ApiEnvelope<unknown[]>>(`/players/${matchId}/${playerId}/trajectory?limit=${limit}`);
