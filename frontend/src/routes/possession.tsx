import { createFileRoute } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { Metric, Panel, PageHeader } from "@/components/Panel";
import { getEvents, getEventMetrics } from "@/api/events";
import type { EventMetric, MatchEvent } from "@/api/events";
import { getPossessionTimeline } from "@/api/possession";
import { useLivePlayers, useLivePossession } from "@/hooks/useGameState";
import { useSession } from "@/lib/session";
import { displayTeamId, teamColor, teamDisplayName } from "@/lib/data";

export const Route = createFileRoute("/possession")({
  head: () => ({ meta: [{ title: "Possession & Events — Ball-Free Game State Reconstruction" }] }),
  component: PossessionPage,
});

const numberValue = (row: Record<string, unknown>, key: string) => {
  const value = row[key];
  return typeof value === "number" ? value : Number(value ?? 0);
};

function PossessionPage() {
  const { matchId, frame, setFrame, teamNames } = useSession();
  const players = useLivePlayers(matchId, frame).data ?? [];
  const isVideo = matchId.startsWith("uploaded_") || matchId.startsWith("video_");
  const timelineStart = Math.floor(frame / 600) * 600;
  const timelineEnd = timelineStart + 600;
  const possession = useLivePossession(matchId, frame);
  const timeline = useQuery({
    queryKey: ["possession-timeline", matchId, timelineStart],
    queryFn: () => getPossessionTimeline(matchId, timelineStart, timelineEnd),
    staleTime: 30_000,
  });
  const events = useQuery({ queryKey: ["events", matchId], queryFn: () => getEvents(matchId) });
  const metrics = useQuery({
    queryKey: ["event-metrics", matchId],
    queryFn: () => getEventMetrics(matchId, 1),
  });
  const eventRows: MatchEvent[] = events.data?.data ?? [];
  const metricRows: EventMetric[] = metrics.data?.data ?? [];
  const displayedEvents = [...eventRows]
    .filter((event) => {
      const eventFrame = Number(event.start_frame ?? event.frame);
      return Number.isFinite(eventFrame) && Math.abs(eventFrame - frame) <= 1500;
    })
    .sort((a, b) => Number(a.start_frame ?? a.frame) - Number(b.start_frame ?? b.frame))
    .slice(0, 80);

  return (
    <>
      <PageHeader
        title="Possession & Events"
        description="Possession is inferred from player configuration alone. Ground-truth event annotations and evaluation metrics are shown separately; no evaluation value is presented as live confidence."
      />
      <div className="mb-4 grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Metric
          label="Current possessor"
          value={String(possession.data?.possessor_id ?? "Data not available")}
          tone="primary"
          sub={`Frame ${frame}`}
        />
        <Metric
          label="Current team"
          value={String(possession.data?.team ?? "Data not available")}
          sub="Inferred"
        />
        <Metric
          label="Inference score"
          value={
            typeof possession.data?.score === "number"
              ? possession.data.score.toFixed(3)
              : "Data not available"
          }
          sub="Not an evaluation F1"
        />
        <Metric
          label="Evaluation tolerance"
          value={metrics.isSuccess ? "±1.00 s" : "Data not available"}
          sub="Step 81"
        />
      </div>

      <Panel
        title="Possession Timeline"
        subtitle={`Sampled player-only inference · frames ${timelineStart}–${timelineEnd} · score is not a probability`}
      >
        {timeline.isLoading ? (
          <div className="text-sm text-muted-foreground">Loading sampled possession estimates…</div>
        ) : timeline.isError ? (
          <div className="text-sm text-destructive">Could not load possession timeline.</div>
        ) : !timeline.data?.data.length ? (
          <div className="text-sm text-muted-foreground">
            No sampled frames contain visible players from both teams.
          </div>
        ) : (
          <div className="space-y-2">
            <div className="flex h-8 overflow-hidden border border-border bg-panel-alt">
              {timeline.data.data.map((sample, index, samples) => {
                const nextFrame =
                  samples[index + 1]?.frame ?? Math.min(timelineEnd, sample.frame + 25);
                return (
                  <button
                    key={sample.frame}
                    type="button"
                    onClick={() => setFrame(sample.frame)}
                    className="h-full min-w-px flex-1 border-r border-background/40"
                    style={{
                      background: teamColor(players, sample.team),
                      opacity: 0.45 + Math.min(0.5, sample.score),
                      flexGrow: Math.max(1, nextFrame - sample.frame),
                    }}
                    title={`Frame ${sample.frame} · ${teamDisplayName(sample.team, players, teamNames)} · ${sample.possessor_id} · heuristic score ${sample.score.toFixed(3)}`}
                    aria-label={`Inspect frame ${sample.frame}, inferred ${teamDisplayName(sample.team, players, teamNames)} possession`}
                  />
                );
              })}
            </div>
            <div className="flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-muted-foreground">
              <span>Jersey color shows inferred team</span>
              <span>Opacity reflects a heuristic score, not calibrated confidence</span>
            </div>
          </div>
        )}
      </Panel>

      <div className="mt-4 grid gap-4 xl:grid-cols-[1fr_1.4fr]">
        <Panel title="Current Inferred Possessor" subtitle={`Live API response · frame ${frame}`}>
          {possession.isLoading ? (
            <div className="text-sm text-muted-foreground">Loading…</div>
          ) : possession.isError ? (
            <div className="text-sm text-destructive">Backend unavailable</div>
          ) : (
            <dl className="text-[12px]">
              <div className="flex justify-between border-b border-border/60 py-1.5">
                <dt className="text-muted-foreground">Player</dt>
                <dd className="num">
                  {typeof possession.data?.possessor_id === "string"
                    ? displayTeamId(possession.data.possessor_id, players, teamNames)
                    : "Data not available"}
                </dd>
              </div>
              <div className="flex justify-between border-b border-border/60 py-1.5">
                <dt className="text-muted-foreground">Team</dt>
                <dd className="num">
                  {possession.data?.team === "home" || possession.data?.team === "away"
                    ? teamDisplayName(possession.data.team, players, teamNames)
                    : "Data not available"}
                </dd>
              </div>
              <div className="flex justify-between border-b border-border/60 py-1.5">
                <dt className="text-muted-foreground">State</dt>
                <dd className="num">{String(possession.data?.state ?? "Data not available")}</dd>
              </div>
              <div className="flex justify-between border-b border-border/60 py-1.5">
                <dt className="text-muted-foreground">Score</dt>
                <dd className="num">
                  {typeof possession.data?.score === "number"
                    ? possession.data.score.toFixed(3)
                    : "Data not available"}
                </dd>
              </div>
            </dl>
          )}
        </Panel>

        <Panel
          title={isVideo ? "Inferred Tactical Events" : "Ground Truth / Annotated Events"}
          subtitle={
            isVideo
              ? `AI-detected passes, ball lost, and recovery transitions for ${matchId}`
              : "Raw Metrica annotations. Predicted events are not currently exposed by the API."
          }
        >
          {events.isLoading ? (
            <div className="text-sm text-muted-foreground">Loading…</div>
          ) : events.isError ? (
            <div className="text-sm text-destructive">Backend unavailable</div>
          ) : displayedEvents.length === 0 ? (
            <div className="text-sm text-muted-foreground p-3 text-center">
              No events within 1500 frames of the selected frame.
            </div>
          ) : (
            <div className="max-h-72 overflow-auto">
              <table className="w-full text-[11.5px]">
                <thead>
                  <tr className="border-b border-border text-left">
                    <th className="label-xs py-1.5">Frame</th>
                    <th className="label-xs py-1.5">Time</th>
                    <th className="label-xs py-1.5">Type</th>
                    <th className="label-xs py-1.5">Team</th>
                  </tr>
                </thead>
                <tbody>
                  {displayedEvents.map((event, index) => (
                    <tr
                      key={`${String(event.event_id ?? index)}-${index}`}
                      className="border-b border-border/60"
                    >
                      <td className="num py-1.5">
                        <button
                          type="button"
                          onClick={() => {
                            const eventFrame = Number(event.start_frame ?? event.frame);
                            if (Number.isFinite(eventFrame)) setFrame(eventFrame);
                          }}
                          className="underline decoration-dotted underline-offset-2 hover:text-primary"
                        >
                          {String(event.start_frame ?? event.frame ?? "—")}
                        </button>
                      </td>
                      <td className="num py-1.5">
                        {(
                          numberValue(event, "start_timestamp") || numberValue(event, "timestamp")
                        ).toFixed(2)}
                        s
                      </td>
                      <td className="py-1.5 font-semibold text-emerald-400">
                        {String(event.event_type ?? event.type ?? "—")}
                      </td>
                      <td className="py-1.5 uppercase text-[10.5px]">
                        {event.team === "home" || event.team === "away"
                          ? teamDisplayName(event.team, players, teamNames)
                          : String(event.team ?? "—")}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Panel>
      </div>

      <Panel
        title="Step 81 Event Evaluation"
        subtitle="Evaluation metrics, not live confidence; predicted event rows remain unavailable."
        className="mt-4"
      >
        {metricRows.length === 0 ? (
          <div className="text-sm text-muted-foreground">Data not available</div>
        ) : (
          <table className="w-full text-[11.5px]">
            <thead>
              <tr className="border-b border-border text-left">
                <th className="label-xs py-1.5">Class</th>
                <th className="label-xs py-1.5 text-right">TP</th>
                <th className="label-xs py-1.5 text-right">FP</th>
                <th className="label-xs py-1.5 text-right">FN</th>
                <th className="label-xs py-1.5 text-right">Precision</th>
                <th className="label-xs py-1.5 text-right">Recall</th>
                <th className="label-xs py-1.5 text-right">F1</th>
              </tr>
            </thead>
            <tbody>
              {metricRows.map((row, index) => (
                <tr
                  key={`${String(row.event_type)}-${index}`}
                  className="border-b border-border/60"
                >
                  <td className="py-1.5">{String(row.event_type)}</td>
                  <td className="num py-1.5 text-right">{String(row.true_positive)}</td>
                  <td className="num py-1.5 text-right">{String(row.false_positive)}</td>
                  <td className="num py-1.5 text-right">{String(row.false_negative)}</td>
                  <td className="num py-1.5 text-right">
                    {numberValue(row, "precision").toFixed(3)}
                  </td>
                  <td className="num py-1.5 text-right">{numberValue(row, "recall").toFixed(3)}</td>
                  <td className="num py-1.5 text-right">{numberValue(row, "f1").toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Panel>
    </>
  );
}
