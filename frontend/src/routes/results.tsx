import { createFileRoute } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { Panel, PageHeader } from "@/components/Panel";
import { getEvaluationSummary } from "@/api/evaluation";

export const Route = createFileRoute("/results")({
  head: () => ({
    meta: [
      { title: "Research Results — Ball-Free Game State Reconstruction" },
      {
        name: "description",
        content: "Aggregate metrics and confidence intervals from verified evaluation artifacts.",
      },
    ],
  }),
  component: Results,
});

function Results() {
  const report = useQuery({
    queryKey: ["evaluation-summary"],
    queryFn: getEvaluationSummary,
  });
  const summary = report.data?.data;
  const conditions = summary?.summary_by_condition;

  return (
    <>
      <PageHeader
        title="Research Results"
        description="Aggregate evaluation results loaded from the committed statistical report. These are offline benchmark measurements, not live-match predictions."
      />
      {report.isLoading ? (
        <Panel title="Evaluation report">Loading verified evaluation artifacts…</Panel>
      ) : report.isError || !summary || !conditions ? (
        <Panel title="Evaluation unavailable">
          {report.data?.availability.message ??
            "No verified aggregate report is available. No substitute or projected scores are shown."}
        </Panel>
      ) : (
        <>
          <Panel
            title="Condition-level outcomes"
            subtitle={`${summary.n_runs} paired runs · ${summary.n_conditions} condition measurements · 95% confidence intervals`}
          >
            <div className="overflow-x-auto">
              <table className="w-full text-[11.5px]">
                <thead>
                  <tr className="border-b border-border text-left">
                    <th className="label-xs py-1.5">Condition</th>
                    <th className="label-xs py-1.5 text-right">Runs</th>
                    <th className="label-xs py-1.5 text-right">Possession F1 (95% CI)</th>
                    <th className="label-xs py-1.5 text-right">Tactical MAE (95% CI)</th>
                    <th className="label-xs py-1.5 text-right">Tracking missingness</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(conditions).map(([condition, metrics]) => (
                    <tr key={condition} className="border-b border-border/60">
                      <td className="py-1.5">{condition.replaceAll("-", " ")}</td>
                      <td className="num py-1.5 text-right">{metrics.possession.count}</td>
                      <td className="num py-1.5 text-right">
                        {metrics.possession.mean.toFixed(3)} [
                        {metrics.possession.ci95_mean_lo.toFixed(3)}–
                        {metrics.possession.ci95_mean_hi.toFixed(3)}]
                      </td>
                      <td className="num py-1.5 text-right">
                        {metrics.tactical.mean.toFixed(3)} [
                        {metrics.tactical.ci95_mean_lo.toFixed(3)}–
                        {metrics.tactical.ci95_mean_hi.toFixed(3)}]
                      </td>
                      <td className="num py-1.5 text-right">
                        {(metrics.tracking.mean * 100).toFixed(1)}%
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Panel>

          <div className="mt-4 grid gap-4 md:grid-cols-2">
            <Panel title="Possession recovery">
              <p className="num text-xl">
                +{summary.possession_recovery.abs_improvement.toFixed(3)} F1
              </p>
              <p className="mt-1 text-[11px] text-muted-foreground">
                Paired mean change over {summary.possession_recovery.n} runs; recovered{" "}
                {(summary.possession_recovery.recovery_fraction * 100).toFixed(1)}% of the
                synthetic-degradation loss.
              </p>
            </Panel>
            <Panel title="Tactical score recovery">
              <p className="num text-xl">
                {summary.tactical_recovery.abs_improvement.toFixed(3)} lower MAE
              </p>
              <p className="mt-1 text-[11px] text-muted-foreground">
                Paired mean change over {summary.tactical_recovery.n} runs. Lower error indicates
                closer agreement with the clean-reference scores.
              </p>
            </Panel>
          </div>

          <Panel title="How to interpret" className="mt-4">
            <p className="text-[11.5px] leading-relaxed text-muted-foreground">
              These metrics use annotated reference data and controlled synthetic degradation. They
              do not establish accuracy on a newly uploaded clip. Possession F1 and tactical score
              error are separate outcomes; no ball-inferred or hypothetical decision result is
              presented as a completed action.
            </p>
          </Panel>
        </>
      )}
    </>
  );
}
