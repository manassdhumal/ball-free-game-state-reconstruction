import { createFileRoute } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { Panel, PageHeader } from "@/components/Panel";
import { BarSeries } from "@/components/Charts";
import { getEvaluationSummary } from "@/api/evaluation";

export const Route = createFileRoute("/recovery")({
  head: () => ({
    meta: [
      { title: "Recovery & Mitigation — Ball-Free Game State Reconstruction" },
      {
        name: "description",
        content: "Observed recovery effects across verified clean, degraded and mitigated runs.",
      },
    ],
  }),
  component: Recovery,
});

function Recovery() {
  const report = useQuery({
    queryKey: ["evaluation-summary"],
    queryFn: getEvaluationSummary,
  });
  const summary = report.data?.data;
  const conditions = summary?.summary_by_condition;
  const chartData = conditions
    ? Object.entries(conditions).map(([condition, metrics]) => ({
        condition: condition.replaceAll("-", " "),
        possessionF1: metrics.possession.mean,
      }))
    : [];

  return (
    <>
      <PageHeader
        title="Recovery & Mitigation"
        description="Comparison of measured conditions in the paired evaluation report. This view does not treat a fitted curve or an unmeasured filter stage as observed performance."
      />
      {report.isLoading ? (
        <Panel title="Recovery analysis">Loading evaluation measurements…</Panel>
      ) : report.isError || !summary || !conditions ? (
        <Panel title="Recovery analysis unavailable">
          {report.data?.availability.message ?? "No verified evaluation report is available."}
        </Panel>
      ) : (
        <>
          <Panel title="Measured condition comparison" className="mb-4">
            <div className="overflow-x-auto">
              <table className="w-full text-[11.5px]">
                <thead>
                  <tr className="border-b border-border text-left">
                    <th className="label-xs py-1.5">Condition</th>
                    <th className="label-xs py-1.5 text-right">Runs</th>
                    <th className="label-xs py-1.5 text-right">Possession F1</th>
                    <th className="label-xs py-1.5 text-right">Tactical MAE</th>
                    <th className="label-xs py-1.5 text-right">Tracking missingness</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(conditions).map(([condition, metrics]) => (
                    <tr key={condition} className="border-b border-border/60">
                      <td className="py-1.5">{condition.replaceAll("-", " ")}</td>
                      <td className="num py-1.5 text-right">{metrics.possession.count}</td>
                      <td className="num py-1.5 text-right">
                        {metrics.possession.mean.toFixed(3)}
                      </td>
                      <td className="num py-1.5 text-right">{metrics.tactical.mean.toFixed(3)}</td>
                      <td className="num py-1.5 text-right">
                        {(metrics.tracking.mean * 100).toFixed(1)}%
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Panel>

          <div className="grid gap-4 lg:grid-cols-2">
            <Panel
              title="Possession F1 by measured condition"
              subtitle="Run means from the evaluation report"
            >
              <BarSeries
                data={chartData}
                xKey="condition"
                series={[{ key: "possessionF1", label: "Possession F1", color: "var(--primary)" }]}
              />
            </Panel>
            <Panel
              title="Reported paired recovery"
              subtitle={`${summary.possession_recovery.n} paired runs`}
            >
              <p className="num text-2xl">
                +{summary.possession_recovery.abs_improvement.toFixed(3)} F1
              </p>
              <p className="mt-2 text-[11.5px] leading-relaxed text-muted-foreground">
                The report estimates{" "}
                {(summary.possession_recovery.recovery_fraction * 100).toFixed(1)}% recovery of the
                degradation-associated F1 loss. This aggregate does not isolate a specific
                interpolation or filtering algorithm.
              </p>
            </Panel>
          </div>
        </>
      )}
    </>
  );
}
