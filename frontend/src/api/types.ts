export type Team = "home" | "away";
export type AvailabilityStatus = "available" | "partial" | "unavailable" | "pending" | "blocked";

export interface Availability {
  status: AvailabilityStatus;
  message?: string;
  provenance?: Record<string, unknown>;
}
export interface ApiEnvelope<T> {
  availability: Availability;
  data: T;
}
export interface ApiPlayer {
  player_id: string;
  team: Team;
  x: number | null;
  y: number | null;
  confidence: number;
  visible: boolean;
}
export interface GameState {
  match_id: string;
  frame: number;
  timestamp: number;
  fps: number;
  players: ApiPlayer[];
  provenance: Record<string, unknown>;
}
export interface Match {
  match_id: string;
  label: string;
  fps: number;
  frame_count: number;
  events_available: boolean;
  tracking_available: boolean;
}
export interface GsrStatus {
  status: "ready" | "partial" | "pending" | "blocked";
  message: string;
  results: unknown;
}
