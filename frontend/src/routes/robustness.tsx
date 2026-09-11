import { createFileRoute } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { Metric, Panel, PageHeader } from "@/components/Panel";
import { getRobustness } from "@/api/robustness";

export const Route = createFileRoute("/robustness")({
  head: () => ({ meta: [{ title: "Decision Robustness — Ball-Free Game State Reconstruction" }] }),
  component: Robustness,
});

type Row = Record<string, unknown>;
const value = (row: Row | undefined, key: string) => {
  const item = row?.[key];
  return typeof item === "number" ? item : Number(item ?? NaN);
};
const display = (row: Row | undefined, key: string) =>
  Number.isFinite(value(row, key)) ? value(row, key).toFixed(3) : "Data not available";

function Robustness() {
  const query = useQuery({
    queryKey: ["robustness", "step83"],
    queryFn: () => getRobustness("step83"),
  });
  const rows = (query.data?.data as Row[] | undefined) ?? [];
  const selected = rows.find((row) => row.severity === "moderate") ?? rows[0];

  return (
    <>
      <PageHeader
        title="Decision Robustness"
        description="Cached Step 83 ranking-stability results served by FastAPI. No robustness experiment is rerun in the browser."
      />
      {query.isLoading ? (
        <div className="text-sm text-muted-foreground">Loading cached robustness results…</div>
      ) : query.isError ? (
        <div className="border border-destructive/40 bg-destructive/5 px-3 py-3 text-sm text-destructive">
          Backend unavailable
        </div>
      ) : rows.length === 0 ? (
        <div className="border border-warn/40 bg-warn/5 px-3 py-3 text-sm text-warn">
          Data not available
        </div>
      ) : (
        <>
          <div className="mb-4 grid grid-cols-2 gap-3 lg:grid-cols-5">
            <Metric
              label="Top-1 retention"
              value={display(selected, "top_1_retention")}
              tone="primary"
              sub="Cached Step 83"
            />
            <Metric
              label="Top-3 retention"
              value={display(selected, "top_3_retention")}
              sub="Cached Step 83"
            />
            <Metric label="Spearman" value={display(selected, "spearman")} />
            <Metric label="Kendall" value={display(selected, "kendall")} />
            <Metric
              label="Decision flip rate"
              value="See Step 83 summary"
              sub="Not present in this stability row"
            />
          </div>
          <Panel
            title="Cached Ranking Stability"
            subtitle="Step 83 experiment output; severity and perturbation are source columns."
          >
            <div className="overflow-auto">
              <table className="w-full text-[11.5px]">
                <thead>
                  <tr className="border-b border-border text-left">
                    <th className="label-xs py-1.5">Method</th>
                    <th className="label-xs py-1.5">Perturbation</th>
                    <th className="label-xs py-1.5">Severity</th>
                    <th className="label-xs py-1.5 text-right">Top-1</th>
                    <th className="label-xs py-1.5 text-right">Top-3</th>
                    <th className="label-xs py-1.5 text-right">Spearman</th>
                    <th className="label-xs py-1.5 text-right">Kendall</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((row, index) => (
                    <tr
                      key={`${String(row.method)}-${String(row.perturbation)}-${String(row.severity)}-${index}`}
                      className="border-b border-border/60"
                    >
                      <td className="py-1.5">{String(row.method)}</td>
                      <td className="py-1.5">{String(row.perturbation)}</td>
                      <td className="py-1.5">{String(row.severity)}</td>
                      <td className="num py-1.5 text-right">{display(row, "top_1_retention")}</td>
                      <td className="num py-1.5 text-right">{display(row, "top_3_retention")}</td>
                      <td className="num py-1.5 text-right">{display(row, "spearman")}</td>
                      <td className="num py-1.5 text-right">{display(row, "kendall")}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Panel>
        </>
      )}
    </>
  );
}
