import { createFileRoute } from "@tanstack/react-router";
import { useEffect } from "react";
import { useState } from "react";
import { Pause, Play, SkipBack, SkipForward } from "lucide-react";
import { Panel, PageHeader } from "@/components/Panel";
import { Pitch, PitchLegend } from "@/components/Pitch";
import {
  buildFrame,
  DATASETS,
  dist,
  EVENTS,
  localDensity,
  nearest,
  POSSESSION_SEGMENTS,
  pressureOn,
  supportFor,
} from "@/lib/data";
import { useSession } from "@/lib/session";
import { useLivePlayers } from "@/hooks/useGameState";

export const Route = createFileRoute("/match-explorer")({
  head: () => ({
    meta: [
      { title: "Match Explorer — Ball-Free Game State Reconstruction" },
      {
        name: "description",
        content:
          "Frame-level playback of player tracking data with event markers, possession transitions and a per-player inspector.",
      },
      { property: "og:title", content: "Match Explorer — Ball-Free Game State Reconstruction" },
      {
        property: "og:description",
        content:
          "Frame playback, event markers and player inspection on player-only tracking data.",
      },
    ],
  }),
  component: MatchExplorer,
});

const SPEEDS = [0.25, 0.5, 1, 2];

function MatchExplorer() {
  const {
    matchId,
    setMatchId,
    frame,
    setFrame,
    dataset,
    setDataset,
    selectedPlayer,
    setSelectedPlayer,
  } = useSession();
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const livePlayers = useLivePlayers(matchId, frame);
  const players = livePlayers.data ?? [];
  const sel = players.find((p) => p.id === selectedPlayer) ?? players[0];

  useEffect(() => {
    if (!playing) return;
    const id = setInterval(
      () => setFrame(Math.min(1500, frame + Math.max(1, Math.round(speed * 2)))),
      80 / speed,
    );
    return () => clearInterval(id);
  }, [playing, speed, frame, setFrame]);

  const windowStart = 1000;
  const windowEnd = 1500;
  const pct = (f: number) => ((f - windowStart) / (windowEnd - windowStart)) * 100;

  return (
    <>
      <PageHeader
        title="Match Explorer"
        description="Inspect the raw player-only tracking stream frame by frame. Possession bands are inferred, not annotated — see Possession & Events for evaluation against ground truth."
      />

      <Panel className="mb-4" bodyClassName="p-2.5">
        <div className="flex flex-wrap items-center gap-3">
          <select
            value={dataset}
            onChange={(e) => {
              setDataset(e.target.value);
              setMatchId(e.target.value.includes("Game_2") ? "metrica_game2" : "metrica_game1");
            }}
            className="border border-input bg-panel-alt px-2 py-1 text-[12px]"
          >
            {DATASETS.map((d) => (
              <option key={d.id} value={d.label}>
                {d.label}
              </option>
            ))}
          </select>

          <div className="flex items-center gap-1">
            <button
              onClick={() => setFrame(Math.max(0, frame - 1))}
              className="border border-border p-1.5 hover:bg-muted"
              aria-label="Previous frame"
            >
              <SkipBack className="h-3.5 w-3.5" />
            </button>
            <button
              onClick={() => setPlaying(!playing)}
              className="border border-border p-1.5 hover:bg-muted"
              aria-label={playing ? "Pause" : "Play"}
            >
              {playing ? <Pause className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" />}
            </button>
            <button
              onClick={() => setFrame(frame + 1)}
              className="border border-border p-1.5 hover:bg-muted"
              aria-label="Next frame"
            >
              <SkipForward className="h-3.5 w-3.5" />
            </button>
          </div>

          <div className="flex items-center gap-1">
            {SPEEDS.map((s) => (
              <button
                key={s}
                onClick={() => setSpeed(s)}
                className={`num border px-2 py-1 text-[11.5px] ${
                  speed === s
                    ? "border-primary text-primary"
                    : "border-border text-muted-foreground"
                }`}
              >
                {s}x
              </button>
            ))}
          </div>

          <input
            type="range"
            min={windowStart}
            max={windowEnd}
            value={Math.min(windowEnd, Math.max(windowStart, frame))}
            onChange={(e) => setFrame(Number(e.target.value))}
            className="h-1 min-w-[240px] flex-1 accent-[var(--primary)]"
          />
          <span className="num text-[12px]">
            Frame {frame} · {(frame / 25).toFixed(1)}s
          </span>
        </div>
      </Panel>

      <div className="grid gap-4 xl:grid-cols-[1.55fr_1fr]">
        <div className="space-y-4">
          <Panel title="Pitch View" subtitle="Click a player token to inspect">
            {livePlayers.isLoading ? (
              <div className="flex h-[460px] items-center justify-center text-sm text-muted-foreground">
                Loading live frame…
              </div>
            ) : livePlayers.isError || !sel ? (
              <div className="flex h-[460px] items-center justify-center text-sm text-destructive">
                Data not available
              </div>
            ) : (
              <Pitch players={players} selectedId={sel.id} onSelect={setSelectedPlayer} />
            )}
            <div className="mt-2">
              <PitchLegend />
            </div>
          </Panel>

          <Panel
            title="Timeline"
            subtitle="Frames 1000–1500 · inferred possession bands and event markers"
          >
            <div className="relative h-7 w-full border border-border bg-panel-alt">
              {POSSESSION_SEGMENTS.filter((s) => s.end > windowStart && s.start < windowEnd).map(
                (s) => (
                  <div
                    key={s.start}
                    className="absolute top-0 h-full"
                    style={{
                      left: `${Math.max(0, pct(s.start))}%`,
                      width: `${Math.min(100, pct(s.end)) - Math.max(0, pct(s.start))}%`,
                      background:
                        s.team === "home"
                          ? "var(--home)"
                          : s.team === "away"
                            ? "var(--away)"
                            : "var(--muted)",
                      opacity: 0.55,
                    }}
                    title={`${s.team} · confidence ${s.confidence}`}
                  />
                ),
              )}
              <div
                className="absolute top-0 h-full w-px bg-primary"
                style={{ left: `${Math.min(100, Math.max(0, pct(frame)))}%` }}
              />
            </div>
            <div className="relative mt-1 h-8">
              {EVENTS.map((e) => (
                <button
                  key={e.frame}
                  onClick={() => setFrame(e.frame)}
                  className="absolute -translate-x-1/2 text-[10px] text-muted-foreground hover:text-foreground"
                  style={{ left: `${pct(e.frame)}%` }}
                >
                  <span className="mx-auto block h-2 w-px bg-border" />
                  {e.predicted !== "—" ? e.predicted : e.annotated}
                </button>
              ))}
            </div>
          </Panel>
        </div>

        <Panel
          title="Player Inspector"
          subtitle={sel ? `#${sel.number} ${sel.name} · ${sel.role}` : "Data not available"}
        >
          {!sel ? (
            <div className="text-sm text-muted-foreground">Select a visible player to inspect.</div>
          ) : (
            <>
              <dl className="text-[12px]">
                {[
                  ["Team", sel.team === "home" ? "Home" : "Away"],
                  ["Track ID", sel.id],
                  ["Position (x, y)", `${sel.x.toFixed(2)}, ${sel.y.toFixed(2)} m`],
                  ["Speed", `${Math.hypot(sel.vx, sel.vy).toFixed(2)} m/s`],
                  ["Heading", `${((Math.atan2(sel.vy, sel.vx) * 180) / Math.PI).toFixed(0)}°`],
                  [
                    "Nearest opponent",
                    `${dist(
                      sel,
                      nearest(
                        sel,
                        players.filter((p) => p.team !== sel.team),
                      ),
                    ).toFixed(1)} m`,
                  ],
                  [
                    "Nearest teammate",
                    `${dist(
                      sel,
                      nearest(
                        sel,
                        players.filter((p) => p.team === sel.team),
                      ),
                    ).toFixed(1)} m`,
                  ],
                  ["Local density (r≤10m)", `${localDensity(sel, players)}`],
                  ["Pressure on player", pressureOn(sel, players).toFixed(3)],
                  ["Support index", supportFor(sel, players).toFixed(3)],
                ].map(([k, v]) => (
                  <div key={k} className="flex justify-between border-b border-border/60 py-1.5">
                    <dt className="text-muted-foreground">{k}</dt>
                    <dd className="num">{v}</dd>
                  </div>
                ))}
              </dl>
              <p className="mt-3 text-[11px] leading-relaxed text-muted-foreground">
                Pressure and support are model-derived quantities computed from relative player
                geometry only. They are not measurements and carry the uncertainty reported on the
                Game State view.
              </p>
            </>
          )}
        </Panel>
      </div>
    </>
  );
}
