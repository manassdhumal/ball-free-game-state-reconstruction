import { apiFetch } from "./client";
import type { ApiEnvelope, Match } from "./types";
export const getMatches = () => apiFetch<ApiEnvelope<Match[]>>("/matches");
export const getMatch = (matchId: string) => apiFetch<ApiEnvelope<Match>>(`/matches/${matchId}`);
