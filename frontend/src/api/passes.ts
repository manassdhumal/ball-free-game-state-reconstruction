import { apiFetch } from "./client";
export const getPassCandidates = (matchId: string, frame: number, playerId: string) =>
  apiFetch<Record<string, unknown>>(`/pass-candidates/${matchId}/${frame}/${playerId}`);

export interface PassRankingRow {
  rank: number;
  target_player_id: string;
  pass_distance: number;
  target_pressure: number;
  target_space: number;
  line_obstruction: number;
  decision_score: number;
  forward_progress?: number | null;
  support_count?: number;
  pass_completion_probability?: number;
  tactical_value?: number;
}

export const getPassRanking = (matchId: string, frame: number, playerId: string) =>
  apiFetch<{ availability: { status: string; message?: string }; data: PassRankingRow[] }>(
    `/pass-ranking/${matchId}/${frame}/${playerId}`,
  );
