import { createFileRoute } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { Metric, Panel, PageHeader } from "@/components/Panel";
import { getEvents, getEventMetrics } from "@/api/events";
import { useLivePossession } from "@/hooks/useGameState";
import { useSession } from "@/lib/session";

export const Route = createFileRoute("/possession")({
  head: () => ({ meta: [{ title: "Possession & Events — Ball-Free Game State Reconstruction" }] }),
  component: PossessionPage,
});

type Row = Record<string, unknown>;
const numberValue = (row: Row, key: string) => {
  const value = row[key];
  return typeof value === "number" ? value : Number(value ?? 0);
};

function PossessionPage() {
  const { matchId, frame } = useSession();
  const possession = useLivePossession(matchId, frame);
  const events = useQuery({ queryKey: ["events", matchId], queryFn: () => getEvents(matchId) });
  const metrics = useQuery({
    queryKey: ["event-metrics", matchId],
    queryFn: () => getEventMetrics(matchId, 1),
  });
  const eventRows = (events.data?.data as Row[] | undefined) ?? [];
  const metricRows = (metrics.data?.data as Row[] | undefined) ?? [];

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
        subtitle="Full timeline materialization is pending; current-frame inference is live."
      >
        <div className="border border-warn/40 bg-warn/5 px-3 py-3 text-sm text-warn">
          Timeline integration pending. Use Match Explorer to scrub bounded current-frame requests.
        </div>
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
                  {String(possession.data?.possessor_id ?? "Data not available")}
                </dd>
              </div>
              <div className="flex justify-between border-b border-border/60 py-1.5">
                <dt className="text-muted-foreground">Team</dt>
                <dd className="num">{String(possession.data?.team ?? "Data not available")}</dd>
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
          title="Ground Truth / Annotated Events"
          subtitle="Raw Metrica annotations. Predicted events are not currently exposed by the API."
        >
          {events.isLoading ? (
            <div className="text-sm text-muted-foreground">Loading…</div>
          ) : events.isError ? (
            <div className="text-sm text-destructive">Backend unavailable</div>
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
                  {eventRows.slice(0, 80).map((event, index) => (
                    <tr
                      key={`${String(event.event_id)}-${index}`}
                      className="border-b border-border/60"
                    >
                      <td className="num py-1.5">{String(event.start_frame)}</td>
                      <td className="num py-1.5">
                        {numberValue(event, "start_timestamp").toFixed(2)}
                      </td>
                      <td className="py-1.5">{String(event.event_type)}</td>
                      <td className="py-1.5">{String(event.team)}</td>
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
