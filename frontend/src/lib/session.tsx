import { createContext, useContext, useMemo, useState, useEffect, type ReactNode } from "react";

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
  const [matchId, setMatchIdState] = useState(() => getInitial("bf_session_matchId", "metrica_game1"));
  const [dataset, setDatasetState] = useState(() => getInitial("bf_session_dataset", "Metrica Sample_Game_1"));
  const [selectedPlayer, setSelectedPlayerState] = useState<string | null>(() => getInitial("bf_session_player", "HOME_10"));

  const setFrame = (f: number) => {
    setFrameState(f);
    try {
      localStorage.setItem("bf_session_frame", String(f));
    } catch {}
  };

  const setMatchId = (m: string) => {
    setMatchIdState(m);
    try {
      localStorage.setItem("bf_session_matchId", m);
    } catch {}
  };

  const setDataset = (d: string) => {
    setDatasetState(d);
    try {
      localStorage.setItem("bf_session_dataset", d);
    } catch {}
  };

  const setSelectedPlayer = (id: string | null) => {
    setSelectedPlayerState(id);
    try {
      if (id) localStorage.setItem("bf_session_player", id);
      else localStorage.removeItem("bf_session_player");
    } catch {}
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
      isVideo,
    }),
    [matchId, frame, dataset, selectedPlayer, isVideo],
  );
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useSession() {
  const v = useContext(Ctx);
  if (!v) throw new Error("useSession must be used inside SessionProvider");
  return v;
}

