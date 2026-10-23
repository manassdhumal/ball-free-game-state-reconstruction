import { createContext, useContext, useMemo, useState, type ReactNode } from "react";
import type { Team } from "@/lib/data";

interface SessionValue {
  matchId: string;
  setMatchId: (matchId: string) => void;
  frame: number;
  setFrame: (f: number) => void;
  dataset: string;
  setDataset: (d: string) => void;
  fps: number;
  selectedPlayer: string | null;
  setSelectedPlayer: (id: string | null) => void;
  teamNames: Record<Team, string>;
  setTeamName: (team: Team, name: string) => void;
  isVideo: boolean;
}

const Ctx = createContext<SessionValue | null>(null);

const getInitial = (key: string, fallback: string) => {
  if (typeof window === "undefined") return fallback;
  try {
    return localStorage.getItem(key) || fallback;
  } catch {
    return fallback;
  }
};

const getInitialNum = (key: string, fallback: number) => {
  if (typeof window === "undefined") return fallback;
  try {
    const val = localStorage.getItem(key);
    return val !== null && !isNaN(Number(val)) ? Number(val) : fallback;
  } catch {
    return fallback;
  }
};

export function SessionProvider({ children }: { children: ReactNode }) {
  const [frame, setFrameState] = useState(() => getInitialNum("bf_session_frame", 1250));
  const [matchId, setMatchIdState] = useState(() =>
    getInitial("bf_session_matchId", "metrica_game1"),
  );
  const [dataset, setDatasetState] = useState(() =>
    getInitial("bf_session_dataset", "Metrica Sample_Game_1"),
  );
  const [selectedPlayer, setSelectedPlayerState] = useState<string | null>(() =>
    getInitial("bf_session_player", "HOME_10"),
  );
  const [teamNames, setTeamNames] = useState<Record<Team, string>>(() => {
    if (typeof window === "undefined") return { home: "", away: "" };
    try {
      const stored = localStorage.getItem("bf_session_team_names");
      if (!stored) return { home: "", away: "" };
      const names = JSON.parse(stored) as Partial<Record<Team, string>>;
      return {
        home: typeof names.home === "string" ? names.home : "",
        away: typeof names.away === "string" ? names.away : "",
      };
    } catch {
      return { home: "", away: "" };
    }
  });

  const setFrame = (f: number) => {
    setFrameState(f);
    try {
      localStorage.setItem("bf_session_frame", String(f));
    } catch {
      return;
    }
  };

  const setMatchId = (m: string) => {
    setMatchIdState(m);
    try {
      localStorage.setItem("bf_session_matchId", m);
    } catch {
      return;
    }
  };

  const setDataset = (d: string) => {
    setDatasetState(d);
    try {
      localStorage.setItem("bf_session_dataset", d);
    } catch {
      return;
    }
  };

  const setSelectedPlayer = (id: string | null) => {
    setSelectedPlayerState(id);
    try {
      if (id) localStorage.setItem("bf_session_player", id);
      else localStorage.removeItem("bf_session_player");
    } catch {
      return;
    }
  };

  const setTeamName = (team: Team, name: string) => {
    setTeamNames((current) => {
      const next = { ...current, [team]: name };
      try {
        localStorage.setItem("bf_session_team_names", JSON.stringify(next));
      } catch {
        return next;
      }
      return next;
    });
  };

  const isVideo = matchId.startsWith("uploaded_") || matchId.startsWith("video_");

  const value = useMemo(
    () => ({
      matchId,
      setMatchId,
      frame,
      setFrame,
      dataset,
      setDataset,
      fps: 25,
      selectedPlayer,
      setSelectedPlayer,
      teamNames,
      setTeamName,
      isVideo,
    }),
    [matchId, frame, dataset, selectedPlayer, teamNames, isVideo],
  );
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useSession() {
  const v = useContext(Ctx);
  if (!v) throw new Error("useSession must be used inside SessionProvider");
  return v;
}
