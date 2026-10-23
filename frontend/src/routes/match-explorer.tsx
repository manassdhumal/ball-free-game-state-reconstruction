import { createFileRoute, Link } from "@tanstack/react-router";
import { useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Pause, Play, SkipBack, SkipForward, Video, ArrowRight } from "lucide-react";
import { Panel, PageHeader } from "@/components/Panel";
import { Pitch, PitchLegend } from "@/components/Pitch";
import { DATASETS, dist, localDensity, nearest, pressureOn, supportFor } from "@/lib/data";
import { useSession } from "@/lib/session";
import { teamDisplayName } from "@/lib/data";
import { useLivePlayers } from "@/hooks/useGameState";
import { getMatches } from "@/api/matches";
import { getEvents } from "@/api/events";

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
    setDataset,
    selectedPlayer,
    setSelectedPlayer,
    teamNames,
  } = useSession();
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const livePlayers = useLivePlayers(matchId, frame);
  const players = livePlayers.data ?? [];
  const sel = players.find((p) => p.id === selectedPlayer) ?? players[0];
  const matchesQuery = useQuery({ queryKey: ["matches"], queryFn: getMatches, staleTime: 15_000 });
  const availableMatches =
    matchesQuery.data?.data && matchesQuery.data.data.length > 0
      ? matchesQuery.data.data
      : DATASETS.slice(0, 2).map((d, index) => ({
          match_id: index === 0 ? "metrica_game1" : "metrica_game2",
          label: d.label,
          fps: 25,
          frame_count: 1500,
          events_available: true,
          tracking_available: true,
        }));

  const isVideo = matchId.startsWith("uploaded_") || matchId.startsWith("video_");
  const currentMatch = availableMatches.find((m) => m.match_id === matchId);
  const maxMatchFrames = currentMatch ? Math.max(1, currentMatch.frame_count) : 1500;
  const windowStart = isVideo ? 0 : 1000;
  const windowEnd = isVideo ? maxMatchFrames : 1500;

  // Real events query for match
  const eventsQuery = useQuery({
    queryKey: ["events", matchId],
    queryFn: () => getEvents(matchId),
    staleTime: 30_000,
  });

  const rawEvents = eventsQuery.data?.data;

  const matchEvents = useMemo(() => {
    return (rawEvents ?? [])
      .map((event) => ({
        frame: Number(event.frame ?? event.start_frame),
        type: String(event.type ?? event.event_type ?? "EVENT"),
        team: String(event.team ?? "unknown").toLowerCase(),
        timestamp: Number(event.timestamp ?? event.start_timestamp ?? 0),
      }))
      .filter((event) => Number.isFinite(event.frame) && event.frame >= 0)
      .sort((a, b) => a.frame - b.frame);
  }, [rawEvents]);

  const activeEvents = useMemo(() => {
    return matchEvents
      .filter((event) => event.frame >= windowStart && event.frame <= windowEnd)
      .map((event) => ({
        frame: event.frame,
        annotated: event.type,
        predicted: event.type,
        team: event.team,
      }));
  }, [matchEvents, windowStart, windowEnd]);

  useEffect(() => {
    if (!playing) return;
    const id = setInterval(
      () => setFrame(Math.min(windowEnd, frame + Math.max(1, Math.round(speed * 2)))),
      80 / speed,
    );
    return () => clearInterval(id);
  }, [playing, speed, frame, setFrame, windowEnd]);

  const pct = (f: number) =>
    Math.min(100, Math.max(0, ((f - windowStart) / Math.max(1, windowEnd - windowStart)) * 100));

  return (
    <>
      <PageHeader
        title="Match Explorer"
        description="Inspect the full-pitch player tracking frame by frame. Markers use sequence event annotations; possession timelines are not fabricated from sparse events."
      />

      {/* Reconstructed Video Match Indicator Banner */}
      {isVideo && (
        <div className="mb-3.5 flex flex-wrap items-center justify-between gap-2 border border-emerald-500/40 bg-emerald-500/10 px-3.5 py-2 rounded text-xs">
          <div className="flex items-center gap-2 text-emerald-400">
            <Video className="h-4 w-4 shrink-0" />
            <span>
              <strong>Perception Video Active:</strong> {currentMatch?.label || matchId} (
              {maxMatchFrames} frames reconstructed)
            </span>
          </div>
          <Link
            to="/broadcast"
            className="flex items-center gap-1 text-[11px] font-semibold text-emerald-400 hover:text-emerald-300 underline"
          >
            <span>Open in Video Studio</span>
            <ArrowRight className="h-3 w-3" />
          </Link>
        </div>
      )}

      <Panel className="mb-4" bodyClassName="p-2.5">
        <div className="flex flex-wrap items-center gap-3">
          <select
            value={matchId}
            onChange={(e) => {
              const selected = availableMatches.find((m) => m.match_id === e.target.value);
              if (selected) {
                const isNewVideo =
                  selected.match_id.startsWith("uploaded_") ||
                  selected.match_id.startsWith("video_");
                setMatchId(selected.match_id);
                setDataset(selected.label);
                setFrame(isNewVideo ? 0 : 1250);
              }
            }}
            className="border border-input bg-panel-alt px-2 py-1 text-[12px] font-medium"
          >
            {availableMatches.map((m) => (
              <option key={m.match_id} value={m.match_id}>
                {m.label} ({m.frame_count} frames)
              </option>
            ))}
          </select>

          <div className="flex items-center gap-1">
            <button
              onClick={() => setFrame(Math.max(windowStart, frame - 1))}
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
              onClick={() => setFrame(Math.min(windowEnd, frame + 1))}
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
              <PitchLegend players={players} />
            </div>
          </Panel>

          <Panel
            title="Timeline"
            subtitle={
              isVideo
                ? `Frames ${windowStart}–${windowEnd} · Inferred possession & tactical event markers for ${currentMatch?.label || matchId}`
                : `Frames ${windowStart}–${windowEnd} · markers use annotated event frames`
            }
          >
            <div className="relative h-7 w-full border border-border bg-panel-alt">
              <div
                className="absolute top-0 h-full w-px bg-primary"
                style={{ left: `${Math.min(100, Math.max(0, pct(frame)))}%` }}
              />
            </div>
            <div className="relative mt-1 h-8">
              {activeEvents.map((e, idx) => (
                <button
                  key={`${e.frame}-${idx}`}
                  onClick={() => setFrame(e.frame)}
                  className="absolute -translate-x-1/2 text-[10px] text-muted-foreground hover:text-foreground cursor-pointer"
                  style={{ left: `${pct(e.frame)}%` }}
                  title={`Frame ${e.frame}: ${e.predicted || e.annotated}`}
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
                  ["Team", teamDisplayName(sel.team, players, teamNames)],
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
