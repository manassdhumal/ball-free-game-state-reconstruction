import { apiFetch } from "./client";

export interface TrackQuality {
  team: string;
  player_id: string;
  observed_frames: number;
  active_frame_coverage: number;
  gap_count: number;
  longest_gap_frames: number;
}

export interface TrackingQuality {
  status: "available" | "unavailable";
  message?: string;
  frame_count?: number;
  frame_step?: number;
  track_count?: number;
  observed_rows?: number;
  coordinate_validity?: number;
  mean_active_frame_coverage?: number;
  gap_count?: number;
  mean_gap_frames?: number;
  tracks?: TrackQuality[];
  definitions?: Record<string, string>;
}

export const getTrackingQuality = (matchId: string) =>
  apiFetch<TrackingQuality>(`/tracking-quality/${matchId}`);
