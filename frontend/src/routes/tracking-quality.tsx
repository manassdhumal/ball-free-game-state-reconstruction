import { createFileRoute } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { Metric, Panel, PageHeader } from "@/components/Panel";
import { BarSeries } from "@/components/Charts";
import { getTrackingQuality } from "@/api/tracking";
import { useSession } from "@/lib/session";
import { useLivePlayers } from "@/hooks/useGameState";
import { teamDisplayName } from "@/lib/data";

export const Route = createFileRoute("/tracking-quality")({
  head: () => ({
    meta: [
      { title: "Tracking Quality — Ball-Free Game State Reconstruction" },
      {
        name: "description",
        content: "Measured coverage, coordinate validity and track gaps for the active dataset.",
      },
    ],
  }),
  component: TrackingQuality,
});

function TrackingQuality() {
  const { matchId, frame, teamNames } = useSession();
  const quality = useQuery({
    queryKey: ["tracking-quality", matchId],
    queryFn: () => getTrackingQuality(matchId),
  });
  const live = useLivePlayers(matchId, frame);
  const players = live.data ?? [];
  const result = quality.data;
  const tracks = result?.tracks ?? [];
  const rows = tracks.map((track) => ({
    ...track,
    label: `${teamDisplayName(track.team === "away" ? "away" : "home", players, teamNames)} ${track.player_id}`,
    coveragePercent: +(track.active_frame_coverage * 100).toFixed(1),
  }));
  const pct = (value: number | undefined) =>
    typeof value === "number" ? `${(value * 100).toFixed(1)}%` : "Data not available";

  if (quality.isLoading)
    return (
      <PageHeader title="Tracking Quality" description="Measuring the active tracking dataset…" />
    );
  if (quality.isError || result?.status !== "available")
    return (
      <PageHeader
        title="Tracking Quality"
        description={result?.message ?? "Tracking quality is unavailable for the selected dataset."}
      />
    );

  return (
    <>
      <PageHeader
        title="Tracking Quality"
        description={`Measured from ${result.observed_rows} tracking rows across ${result.frame_count} sampled frames. Coverage refers to each track's observed active interval; it is not full-match availability.`}
      />

      <div className="mb-4 grid grid-cols-2 gap-3 lg:grid-cols-5">
        <Metric label="Observed rows" value={String(result.observed_rows)} />
        <Metric label="Tracked players" value={String(result.track_count)} />
        <Metric
          label="Coordinate validity"
          value={pct(result.coordinate_validity)}
          tone="primary"
        />
        <Metric label="Mean active coverage" value={pct(result.mean_active_frame_coverage)} />
        <Metric
          label="Observed track gaps"
          value={String(result.gap_count)}
          sub={`Mean ${result.mean_gap_frames} sampled frames`}
        />
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Panel
          title="Active-interval coverage by track"
          subtitle="Observed samples / expected samples within each track's active span"
        >
          <BarSeries
            data={rows}
            xKey="label"
            series={[{ key: "coveragePercent", label: "Coverage (%)", color: "var(--primary)" }]}
          />
        </Panel>
        <Panel
          title="Observed gaps by track"
          subtitle={`Missing sampled frames · source sampling step ${result.frame_step}`}
        >
          <BarSeries
            data={rows}
            xKey="label"
            series={[{ key: "gap_count", label: "Gap count", color: "var(--chart-5)" }]}
          />
        </Panel>
      </div>

      <Panel title="Track-level measurements" className="mt-4">
        <div className="mb-2 space-y-1 text-[10.5px] text-muted-foreground">
          {Object.values(result.definitions ?? {}).map((definition) => (
            <p key={definition}>{definition}</p>
          ))}
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-[11.5px]">
            <thead>
              <tr className="border-b border-border text-left">
                <th className="label-xs py-1.5">Track</th>
                <th className="label-xs py-1.5 text-right">Observed frames</th>
                <th className="label-xs py-1.5 text-right">Active coverage</th>
                <th className="label-xs py-1.5 text-right">Gap count</th>
                <th className="label-xs py-1.5 text-right">Longest gap (frames)</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((track) => (
                <tr key={`${track.team}-${track.player_id}`} className="border-b border-border/60">
                  <td className="py-1.5">{track.label}</td>
                  <td className="num py-1.5 text-right">{track.observed_frames}</td>
                  <td className="num py-1.5 text-right">{track.coveragePercent.toFixed(1)}%</td>
                  <td className="num py-1.5 text-right">{track.gap_count}</td>
                  <td className="num py-1.5 text-right">{track.longest_gap_frames}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>
    </>
  );
}
