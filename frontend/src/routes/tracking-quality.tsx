import { createFileRoute } from "@tanstack/react-router";
import { Metric, Panel, PageHeader } from "@/components/Panel";
import { BarSeries, LineSeries } from "@/components/Charts";
import { QUALITY_DISTRIBUTION, TRACKING_QUALITY } from "@/lib/data";

export const Route = createFileRoute("/tracking-quality")({
  head: () => ({
    meta: [
      { title: "Tracking Quality — Ball-Free Game State Reconstruction" },
      {
        name: "description",
        content:
          "Input data quality audit: missingness, visible player completeness, coordinate validity and jitter distributions.",
      },
      { property: "og:title", content: "Tracking Quality — Ball-Free Game State Reconstruction" },
      {
        property: "og:description",
        content:
          "Missingness, completeness, coordinate validity and jitter for the active dataset.",
      },
    ],
  }),
  component: TrackingQuality,
});

function TrackingQuality() {
  const perPlayer = Array.from({ length: 22 }, (_, i) => ({
    track: `${i < 11 ? "H" : "A"}${(i % 11) + 1}`,
    completeness: +(0.99 - ((i * 13) % 17) / 180).toFixed(3),
    jitter: +(0.12 + ((i * 7) % 19) / 90).toFixed(3),
  }));

  const timeline = Array.from({ length: 30 }, (_, i) => ({
    minute: i * 3,
    missingness: +(0.04 + 0.05 * Math.abs(Math.sin(i / 4))).toFixed(3),
    validity: +(0.998 - 0.012 * Math.abs(Math.cos(i / 5))).toFixed(3),
  }));

  return (
    <>
      <PageHeader
        title="Tracking Quality"
        description="Quality audit of the input tracking stream. Downstream reconstruction quality is bounded by these figures; every metric elsewhere in the workstation inherits this uncertainty."
      />

      <div className="mb-4 grid grid-cols-2 gap-3 lg:grid-cols-6">
        <Metric
          label="Missingness Rate"
          value={`${(TRACKING_QUALITY.missingness * 100).toFixed(1)}%`}
          tone="warn"
        />
        <Metric
          label="Player Completeness"
          value={`${(TRACKING_QUALITY.completeness * 100).toFixed(1)}%`}
        />
        <Metric
          label="Coordinate Validity"
          value={`${(TRACKING_QUALITY.validity * 100).toFixed(1)}%`}
          tone="primary"
        />
        <Metric label="Mean Jitter" value={`${TRACKING_QUALITY.jitter.toFixed(2)} m`} />
        <Metric label="Track Gaps" value={String(TRACKING_QUALITY.gaps)} sub="Full match" />
        <Metric
          label="Mean Gap Length"
          value={`${TRACKING_QUALITY.meanGapLen} f`}
          sub="≈0.30 s at 25 FPS"
        />
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Panel title="Gap Length Distribution" subtitle="Frames per gap-length bin">
          <BarSeries
            data={QUALITY_DISTRIBUTION}
            xKey="bin"
            series={[{ key: "frames", label: "Frames", color: "var(--chart-5)" }]}
          />
        </Panel>
        <Panel title="Missingness & Validity Over Time" subtitle="Three-minute buckets">
          <LineSeries
            data={timeline}
            xKey="minute"
            series={[
              { key: "missingness", label: "Missingness", color: "var(--away)" },
              { key: "validity", label: "Validity", color: "var(--primary)" },
            ]}
          />
        </Panel>
      </div>

      <Panel title="Per-Track Quality" className="mt-4">
        <table className="w-full text-[11.5px]">
          <thead>
            <tr className="border-b border-border text-left">
              <th className="label-xs py-1.5">Track</th>
              <th className="label-xs py-1.5 text-right">Completeness</th>
              <th className="label-xs py-1.5 text-right">Jitter (m)</th>
              <th className="label-xs py-1.5">Assessment</th>
            </tr>
          </thead>
          <tbody>
            {perPlayer.map((t) => (
              <tr key={t.track} className="border-b border-border/60">
                <td className="num py-1.5">{t.track}</td>
                <td className="num py-1.5 text-right">{t.completeness.toFixed(3)}</td>
                <td className="num py-1.5 text-right">{t.jitter.toFixed(3)}</td>
                <td
                  className={`py-1.5 text-[10.5px] uppercase tracking-[0.05em] ${
                    t.completeness > 0.95 && t.jitter < 0.28
                      ? "text-primary"
                      : t.completeness > 0.92
                        ? "text-warn"
                        : "text-destructive"
                  }`}
                >
                  {t.completeness > 0.95 && t.jitter < 0.28
                    ? "usable"
                    : t.completeness > 0.92
                      ? "marginal"
                      : "degraded"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </Panel>
    </>
  );
}
