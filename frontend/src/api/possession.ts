import { apiFetch } from "./client";

export interface PossessionState {
  status: "available" | "unavailable";
  possessor_id?: string;
  team?: string;
  score?: number;
  state?: string;
  message?: string;
  provenance?: Record<string, unknown>;
}

export const getPossession = (matchId: string, frame: number) =>
  apiFetch<PossessionState>(`/possession/${matchId}/${frame}`);

export interface PossessionSample {
  frame: number;
  timestamp: number;
  team: "home" | "away";
  possessor_id: string;
  score: number;
}

export const getPossessionTimeline = (
  matchId: string,
  startFrame: number,
  endFrame: number,
  stride = 25,
) =>
  apiFetch<{ availability: { status: string; message?: string }; data: PossessionSample[] }>(
    `/possession/${matchId}/timeline?start_frame=${startFrame}&end_frame=${endFrame}&stride=${stride}`,
  );
