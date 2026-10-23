import { createFileRoute } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { Panel, PageHeader } from "@/components/Panel";
import { getStep83Stability } from "@/api/evaluation";

export const Route = createFileRoute("/benchmark")({
  head: () => ({
    meta: [
      { title: "Robustness Benchmark — Ball-Free Game State Reconstruction" },
      {
        name: "description",
        content: "Measured ranking stability by perturbation from the Step 83 evaluation artifact.",
      },
    ],
  }),
  component: Benchmark,
});

function Benchmark() {
  const results = useQuery({
    queryKey: ["robustness", "step83"],
    queryFn: getStep83Stability,
  });
  const rows = results.data?.data ?? [];
  const text = (value: string | number | undefined) => (value == null ? "—" : String(value));
  const metric = (value: number | undefined) => (value == null ? "—" : value.toFixed(3));

  return (
    <>
      <PageHeader
        title="Robustness Benchmark"
        description="Ranking stability measurements loaded from the Step 83 results artifact. Each row reports a tested method and perturbation; no interpolated degradation curve is substituted."
      />
      {results.isLoading ? (
        <Panel title="Ranking stability">Loading benchmark rows…</Panel>
      ) : results.isError || rows.length === 0 ? (
        <Panel title="Benchmark unavailable">
          {results.data?.availability.message ??
            "No Step 83 ranking-stability measurements are available."}
        </Panel>
      ) : (
        <>
          <Panel title="Measured perturbation outcomes">
            <div className="overflow-x-auto">
              <table className="w-full text-[11.5px]">
                <thead>
                  <tr className="border-b border-border text-left">
                    <th className="label-xs py-1.5">Method</th>
                    <th className="label-xs py-1.5">Perturbation</th>
                    <th className="label-xs py-1.5">Severity</th>
                    <th className="label-xs py-1.5 text-right">States</th>
                    <th className="label-xs py-1.5 text-right">Top-1 retention</th>
                    <th className="label-xs py-1.5 text-right">Top-3 retention</th>
                    <th className="label-xs py-1.5 text-right">Spearman ρ</th>
                    <th className="label-xs py-1.5 text-right">Kendall τ</th>
                    <th className="label-xs py-1.5 text-right">Mean rank change</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((row, index) => (
                    <tr
                      key={`${text(row.method)}-${text(row.perturbation)}-${text(row.severity)}-${index}`}
                      className="border-b border-border/60"
                    >
                      <td className="py-1.5">{text(row.method)}</td>
                      <td className="py-1.5">{text(row.perturbation)}</td>
                      <td className="py-1.5">{text(row.severity)}</td>
                      <td className="num py-1.5 text-right">{text(row.n_states)}</td>
                      <td className="num py-1.5 text-right">{metric(row.top_1_retention)}</td>
                      <td className="num py-1.5 text-right">{metric(row.top_3_retention)}</td>
                      <td className="num py-1.5 text-right">{metric(row.spearman)}</td>
                      <td className="num py-1.5 text-right">{metric(row.kendall)}</td>
                      <td className="num py-1.5 text-right">
                        {metric(row.average_rank_displacement)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Panel>
          <Panel title="Interpretation" className="mt-4">
            <p className="text-[11.5px] leading-relaxed text-muted-foreground">
              Retention measures whether the same candidate remains ranked first or within the top
              three after perturbation. Rank correlations measure ordering consistency across
              candidates. These are offline comparisons against the benchmark reference and do not
              show the actual choice made by a player.
            </p>
          </Panel>
        </>
      )}
    </>
  );
}
