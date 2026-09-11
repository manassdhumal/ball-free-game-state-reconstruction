import { apiFetch } from "./client";
import type { GameState } from "./types";
export const getGameState = (matchId: string, frame: number) =>
  apiFetch<GameState>(`/game-state/${matchId}/${frame}`);
