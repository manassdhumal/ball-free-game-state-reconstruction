import { useQuery } from "@tanstack/react-query";
import { getGameState } from "@/api/gameState";
import { getHealth } from "@/api/client";
import { getPassRanking } from "@/api/passes";
import { getPossession } from "@/api/possession";
import { getTacticalState } from "@/api/tactics";
import type { Player } from "@/lib/data";

export function useLivePlayers(matchId: string, frame: number) {
  return useQuery({
    queryKey: ["game-state", matchId, frame],
    queryFn: () => getGameState(matchId, frame),
    select: (state): Player[] =>
      state.players
        .filter((p) => p.visible && p.x != null && p.y != null)
        .map((p) => {
          const parts = p.player_id.split("_");
          const id = `${parts[0] ?? "PLAYER"}_${parts[1] ?? p.player_id}`;
          const number = Number(parts[1]) || 0;
          return {
            id,
            team: p.team,
            number,
            name: p.player_id,
            role: "unknown",
            x: p.x! * 105,
            y: p.y! * 68,
            vx: 0,
            vy: 0,
            confidence: p.confidence,
            visible: p.visible,
          };
        }),
  });
}

export function useLivePossession(matchId: string, frame: number) {
  return useQuery({
    queryKey: ["possession", matchId, frame],
    queryFn: () => getPossession(matchId, frame),
  });
}

export function useLiveTactical(matchId: string, frame: number) {
  return useQuery({
    queryKey: ["tactical", matchId, frame],
    queryFn: () => getTacticalState(matchId, frame),
  });
}

export function useBackendHealth() {
  return useQuery({ queryKey: ["health"], queryFn: getHealth, staleTime: 30_000, retry: false });
}

export function useLivePassRanking(matchId: string, frame: number, playerId: string | null) {
  return useQuery({
    queryKey: ["pass-ranking", matchId, frame, playerId],
    queryFn: () => getPassRanking(matchId, frame, playerId!),
    enabled: Boolean(playerId),
  });
}
