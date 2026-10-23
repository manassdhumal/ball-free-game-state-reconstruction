import { createFileRoute, Link } from "@tanstack/react-router";
import { useMemo, useState } from "react";
import { Panel, PageHeader } from "@/components/Panel";
import { Pitch, PitchLegend } from "@/components/Pitch";
import { useLivePassRanking, useLivePlayers } from "@/hooks/useGameState";
import { useSession } from "@/lib/session";
import { dist, teamDisplayName } from "@/lib/data";
import { Video } from "lucide-react";

export const Route = createFileRoute("/counterfactual")({
  head: () => ({
    meta: [
      { title: "Counterfactual Decisions — Ball-Free Game State Reconstruction" },
      {
        name: "description",
        content:
          "Model-ranked hypothetical passing options generated from the observed player-only state, with a full scoring breakdown.",
      },
      {
        property: "og:title",
        content: "Counterfactual Decisions — Ball-Free Game State Reconstruction",
      },
      {
        property: "og:description",
        content:
          "Hypothetical option ranking with distance, pressure, space and obstruction terms.",
      },
    ],
  }),
  component: Counterfactual,
});

export function Disclaimer() {
  return (
    <div className="border border-warn/40 bg-warn/5 px-3 py-2.5 text-[11.5px] leading-relaxed text-warn">
      <span className="mr-1.5 font-semibold uppercase tracking-[0.06em]">Disclaimer</span>
      This is a model-ranked hypothetical option generated from the observed player-only state. It
      does not indicate that the player selected this option.
    </div>
  );
}

function Counterfactual() {
  const { matchId, frame, selectedPlayer, setSelectedPlayer, teamNames } = useSession();
  const isVideo = matchId.startsWith("uploaded_") || matchId.startsWith("video_");
  const playerState = useLivePlayers(matchId, frame);
  const players = useMemo(() => playerState.data ?? [], [playerState.data]);
  const [selectedTarget, setSelectedTarget] = useState<string | null>(null);
  const source = players.find((p) => p.id === selectedPlayer) ?? players[0];
  const ranking = useLivePassRanking(matchId, frame, source?.id ?? null);
  const candidates = useMemo(() => {
    const rows = ranking.data?.data ?? [];
    return rows.flatMap((row) => {
      const targetId = `${source?.team.toUpperCase() ?? "HOME"}_${row.target_player_id}`;
      const targetPlayer = players.find((player) => player.id === targetId);
      if (!source || !targetPlayer) return [];
      return {
        rank: row.rank,
        targetId,
        target: targetPlayer.name,
        distance: dist(source, targetPlayer),
        pressure: row.target_pressure,
        space: row.target_space,
        obstruction: row.line_obstruction,
        score: row.decision_score,
        forwardProgress: row.forward_progress,
        supportCount: row.support_count,
        completion: row.pass_completion_probability,
        tacticalValue: row.tactical_value,
      };
    });
  }, [ranking.data, players, source]);
  const top = candidates[0];

  if (playerState.isLoading || ranking.isLoading)
    return (
      <PageHeader title="Counterfactual Decisions" description="Loading current-state ranking…" />
    );
  if (playerState.isError || ranking.isError || !source)
    return (
      <PageHeader
        title="Counterfactual Decisions"
        description="Data not available. Start the backend and select a valid frame."
      />
    );
  if (!top)
    return (
      <PageHeader
        title="Counterfactual Decisions"
        description="No ranked passing options are available for the selected frame and player."
      />
    );

  return (
    <>
      <PageHeader
        title="Counterfactual Decisions"
        description="For a chosen source player, the model enumerates every teammate as a hypothetical receiving option and ranks them by a weighted decision score. No ball or intent information enters this computation."
      />

      {ranking.data?.availability.message && (
        <div className="mb-4 border border-border bg-panel-alt px-3 py-2 text-[11px] text-muted-foreground">
          {ranking.data.availability.message}
        </div>
      )}

      {isVideo && (
        <div className="mb-3.5 flex flex-wrap items-center justify-between gap-2 border border-emerald-500/40 bg-emerald-500/10 px-3.5 py-2 rounded text-xs">
          <div className="flex items-center gap-2 text-emerald-400">
            <Video className="h-4 w-4 shrink-0" />
            <span>
              <strong>Video Perception Options:</strong> Counterfactual passing lines for {matchId}{" "}
              (Frame {frame})
            </span>
          </div>
          <div className="flex items-center gap-3">
            <Link
              to="/match-explorer"
              className="text-[11px] font-semibold text-emerald-400 hover:text-emerald-300 underline"
            >
              Match Explorer
            </Link>
            <Link
              to="/broadcast"
              className="text-[11px] font-semibold text-emerald-400 hover:text-emerald-300 underline"
            >
              Video Studio
            </Link>
          </div>
        </div>
      )}

      <div className="mb-4">
        <Disclaimer />
      </div>

      <Panel className="mb-4" bodyClassName="p-2.5">
        <div className="flex flex-wrap items-center gap-3">
          <span className="label-xs">Source player</span>
          <select
            value={source.id}
            onChange={(e) => setSelectedPlayer(e.target.value)}
            className="border border-input bg-panel-alt px-2 py-1 text-[12px]"
          >
            {players
              .filter((p) => p.role !== "GK")
              .map((p) => (
                <option key={p.id} value={p.id}>
                  {teamDisplayName(p.team, players, teamNames)} · #{p.number} {p.name} ({p.role})
                </option>
              ))}
          </select>
          <span className="num text-[12px] text-muted-foreground">
            Frame {frame} · {candidates.length} current-state candidates
          </span>
        </div>
      </Panel>

      <div className="grid gap-4 xl:grid-cols-[1.4fr_1fr]">
        <Panel
          title="Candidate Pass Lines"
          subtitle="Top five hypothetical options drawn from the source player"
        >
          <Pitch
            players={players}
            selectedId={source.id}
            onSelect={setSelectedPlayer}
            passLines={candidates.slice(0, 5).map((c) => ({
              fromId: source.id,
              toId: c.targetId,
              rank: c.targetId === selectedTarget ? 1 : c.rank + 1,
              score: c.score,
            }))}
          />
          <div className="mt-2">
            <PitchLegend players={players} />
          </div>
        </Panel>

        <Panel title="Rank-1 evidence" subtitle={`Rank 1 · ${top.target}`}>
          <div className="space-y-2 text-[11.5px]">
            {[
              {
                k: "Heuristic completion score",
                v: top.completion,
                note: "Fixed player-only baseline; this value is not calibrated as a real pass-completion probability.",
              },
              {
                k: "Composite tactical value",
                v: top.tacticalValue,
                note: "Weighted spatial score using progress when known, target space, support, pressure and distance.",
              },
              {
                k: "Forward progress (normalized)",
                v: top.forwardProgress,
                note:
                  top.forwardProgress == null
                    ? "Not included because attack direction is unknown for this clip."
                    : "Direction-aware pitch-axis displacement; not expected threat.",
              },
              {
                k: "Target-space index (normalized)",
                v: top.space,
                note: "Nearest-opponent separation in normalized pitch coordinates.",
              },
              {
                k: "Target-pressure index (normalized)",
                v: top.pressure,
                note: "Nearest-opponent distance proxy in normalized pitch coordinates.",
              },
              {
                k: "Supporting teammates",
                v: top.supportCount,
                note: "Count from the player-only support radius in the source model.",
              },
              {
                k: "Opponents intersecting pass line",
                v: top.obstruction,
                note: "Observed obstruction count from the candidate segment geometry.",
              },
            ].map((r) => (
              <div key={r.k} className="border border-border bg-panel-alt p-2.5">
                <div className="flex items-center justify-between">
                  <span>{r.k}</span>
                  <span className="num text-primary">
                    {typeof r.v !== "number"
                      ? "Data not available"
                      : r.k === "Supporting teammates" || r.k === "Opponents intersecting pass line"
                        ? String(r.v)
                        : r.v.toFixed(3)}
                  </span>
                </div>
                <p className="mt-1.5 text-[11px] leading-relaxed text-muted-foreground">{r.note}</p>
              </div>
            ))}
            <div className="border border-border p-2.5">
              <div className="flex items-center justify-between">
                <span className="label-xs">Composite decision score</span>
                <span className="num text-[15px] text-primary">{top.score.toFixed(3)}</span>
              </div>
              <div className="mt-1 flex items-center justify-between text-[11px] text-muted-foreground">
                <span>Margin over rank 2</span>
                <span className="num">
                  {candidates[1]
                    ? (top.score - candidates[1].score).toFixed(3)
                    : "Only one candidate"}
                </span>
              </div>
              <div className="mt-0.5 flex items-center justify-between text-[11px] text-muted-foreground">
                <span>Rank stability under perturbation</span>
                <span className="num">Data not available</span>
              </div>
            </div>
          </div>
        </Panel>
      </div>

      <Panel
        title="Candidate Ranking"
        subtitle="All teammates scored as hypothetical options"
        className="mt-4"
      >
        <table className="w-full text-[11.5px]">
          <thead>
            <tr className="border-b border-border text-left">
              <th className="label-xs py-1.5">Rank</th>
              <th className="label-xs py-1.5">Target</th>
              <th className="label-xs py-1.5 text-right">Distance (m)</th>
              <th className="label-xs py-1.5 text-right">Pressure (normalized)</th>
              <th className="label-xs py-1.5 text-right">Space (normalized)</th>
              <th className="label-xs py-1.5 text-right">Obstructing players</th>
              <th className="label-xs py-1.5 text-right">Decision score</th>
              <th className="label-xs py-1.5 text-right">Stability</th>
            </tr>
          </thead>
          <tbody>
            {candidates.map((c) => (
              <tr
                key={c.targetId}
                onClick={() => setSelectedTarget(c.targetId)}
                className={`border-b border-border/60 ${c.rank === 1 ? "bg-accent/40" : ""}`}
                style={{ cursor: "pointer" }}
              >
                <td className="num py-1.5">{c.rank}</td>
                <td className="py-1.5">{c.target}</td>
                <td className="num py-1.5 text-right">{c.distance.toFixed(1)}</td>
                <td className="num py-1.5 text-right">{c.pressure.toFixed(3)}</td>
                <td className="num py-1.5 text-right">{c.space.toFixed(3)}</td>
                <td className="num py-1.5 text-right">{c.obstruction}</td>
                <td className={`num py-1.5 text-right ${c.rank === 1 ? "text-primary" : ""}`}>
                  {c.score.toFixed(3)}
                </td>
                <td className="num py-1.5 text-right">Not measured</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Panel>
    </>
  );
}
