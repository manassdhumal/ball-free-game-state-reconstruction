import { createFileRoute } from "@tanstack/react-router";
import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Panel, PageHeader } from "@/components/Panel";
import { getStep80Runs } from "@/api/evaluation";

export const Route = createFileRoute("/experiments")({
  head: () => ({
    meta: [
      { title: "Experiments — Ball-Free Game State Reconstruction" },
      {
        name: "description",
        content: "Reproducible tracking and mitigation rows from the Step 80 evaluation artifact.",
      },
    ],
  }),
  component: Experiments,
});

function Experiments() {
  const [severity, setSeverity] = useState("all");
  const [method, setMethod] = useState("all");
  const results = useQuery({
    queryKey: ["robustness", "step80"],
    queryFn: getStep80Runs,
  });
  const data = results.data?.data;
  const availableRuns = data ?? [];
  const severities = [
    ...new Set(availableRuns.flatMap((row) => (row.severity ? [row.severity] : []))),
  ];
  const methods = [
    ...new Set(
      availableRuns.flatMap((row) => (row.mitigation_method ? [row.mitigation_method] : [])),
    ),
  ];
  const rows = useMemo(
    () =>
      (data ?? []).filter(
        (row) =>
          (severity === "all" || row.severity === severity) &&
          (method === "all" || row.mitigation_method === method),
      ),
    [data, severity, method],
  );

  return (
    <>
      <PageHeader
        title="Experiments"
        description="Measured run-level results from the Step 80 robustness artifact. Each row records its synthetic degradation severity, seed, time window and mitigation method."
      />
      {results.isLoading ? (
        <Panel title="Experiment results">Loading verified experiment rows…</Panel>
      ) : results.isError || (data?.length ?? 0) === 0 ? (
        <Panel title="Experiment results unavailable">
          {results.data?.availability.message ?? "No Step 80 experiment rows are available."}
        </Panel>
      ) : (
        <Panel title="Step 80 run register">
          <div className="mb-3 flex flex-wrap items-center gap-3 text-[12px]">
            <label className="flex items-center gap-2">
              <span className="label-xs">Severity</span>
              <select
                value={severity}
                onChange={(event) => setSeverity(event.target.value)}
                className="border border-input bg-panel-alt px-2 py-1"
              >
                <option value="all">All</option>
                {severities.map((item) => (
                  <option key={item}>{item}</option>
                ))}
              </select>
            </label>
            <label className="flex items-center gap-2">
              <span className="label-xs">Mitigation</span>
              <select
                value={method}
                onChange={(event) => setMethod(event.target.value)}
                className="border border-input bg-panel-alt px-2 py-1"
              >
                <option value="all">All</option>
                {methods.map((item) => (
                  <option key={item}>{item}</option>
                ))}
              </select>
            </label>
            <span className="num text-muted-foreground">{rows.length} measured runs</span>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-[11.5px]">
              <thead>
                <tr className="border-b border-border text-left">
                  <th className="label-xs py-1.5">Severity</th>
                  <th className="label-xs py-1.5 text-right">Seed</th>
                  <th className="label-xs py-1.5">Window</th>
                  <th className="label-xs py-1.5">Mitigation</th>
                  <th className="label-xs py-1.5 text-right">Missingness</th>
                  <th className="label-xs py-1.5 text-right">Possession precision</th>
                  <th className="label-xs py-1.5 text-right">Possession recall</th>
                  <th className="label-xs py-1.5 text-right">Possession F1</th>
                  <th className="label-xs py-1.5 text-right">TAS MAE</th>
                  <th className="label-xs py-1.5 text-right">Unresolved gaps</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row, index) => (
                  <tr
                    key={`${row.severity}-${row.seed}-${row.window}-${row.mitigation_method}-${index}`}
                    className="border-b border-border/60"
                  >
                    <td className="py-1.5">{String(row.severity ?? "—")}</td>
                    <td className="num py-1.5 text-right">{String(row.seed ?? "—")}</td>
                    <td className="py-1.5">{String(row.window ?? "—")}</td>
                    <td className="py-1.5">{String(row.mitigation_method ?? "—")}</td>
                    <td className="num py-1.5 text-right">
                      {row.missingness == null ? "—" : `${(row.missingness * 100).toFixed(1)}%`}
                    </td>
                    <td className="num py-1.5 text-right">
                      {row.possession_precision?.toFixed(3) ?? "—"}
                    </td>
                    <td className="num py-1.5 text-right">
                      {row.possession_recall?.toFixed(3) ?? "—"}
                    </td>
                    <td className="num py-1.5 text-right">
                      {row.possession_f1?.toFixed(3) ?? "—"}
                    </td>
                    <td className="num py-1.5 text-right">{row.tas_mae?.toFixed(3) ?? "—"}</td>
                    <td className="num py-1.5 text-right">
                      {String(row.unresolved_gap_count ?? "—")}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>
      )}
    </>
  );
}
