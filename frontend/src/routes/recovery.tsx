import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import { ChevronRight } from "lucide-react";
import { Panel, PageHeader } from "@/components/Panel";
import { LineSeries } from "@/components/Charts";
import { DEGRADATION_CURVE, RECOVERY_TABLE } from "@/lib/data";

export const Route = createFileRoute("/recovery")({
  head: () => ({
    meta: [
      { title: "Recovery & Mitigation — Ball-Free Game State Reconstruction" },
      {
        name: "description",
        content:
          "Reference to degraded to interpolated to Kalman-filtered recovery workflow with comparative F1, MAE and runtime.",
      },
      {
        property: "og:title",
        content: "Recovery & Mitigation — Ball-Free Game State Reconstruction",
      },
      {
        property: "og:description",
        content: "Interpolation and Kalman filtering as mitigation for degraded tracking input.",
      },
    ],
  }),
  component: Recovery,
});

const STAGES = [
  {
    key: "reference",
    name: "Reference",
    detail:
      "Clean Metrica tracking, 22 tracks, no injected degradation. Baseline for all comparisons.",
  },
  {
    key: "degraded",
    name: "Degraded",
    detail:
      "15% player dropout and 0.5 m coordinate jitter applied. Support graph is the first stage to fail.",
  },
  {
    key: "interpolated",
    name: "Linear Interpolation",
    detail:
      "Gaps shorter than 12 frames filled linearly. Cheap, but produces implausible straight-line motion.",
  },
  {
    key: "kalman",
    name: "Kalman Filter",
    detail:
      "Constant-velocity model with per-track process noise. Best recovery, materially higher runtime.",
  },
];

function Recovery() {
  const [stage, setStage] = useState(0);

  return (
    <>
      <PageHeader
        title="Recovery & Mitigation"
        description="Sequential mitigation workflow applied to degraded tracking input. Each stage is evaluated independently against the reference configuration."
      />

      <Panel title="Workflow" className="mb-4">
        <div className="flex flex-wrap items-center gap-2">
          {STAGES.map((s, i) => (
            <div key={s.key} className="flex items-center gap-2">
              <button
                onClick={() => setStage(i)}
                className={`border px-3 py-2 text-left text-[12px] ${
                  stage === i
                    ? "border-primary bg-accent text-foreground"
                    : "border-border text-muted-foreground"
                }`}
              >
                <div className="label-xs">Stage {i + 1}</div>
                {s.name}
              </button>
              {i < STAGES.length - 1 && <ChevronRight className="h-4 w-4 text-muted-foreground" />}
            </div>
          ))}
        </div>
        <p className="mt-3 max-w-3xl text-[11.5px] leading-relaxed text-muted-foreground">
          {STAGES[stage]!.detail}
        </p>
      </Panel>

      <div className="grid gap-4 xl:grid-cols-[1fr_1fr]">
        <Panel
          title="Comparative Metrics"
          subtitle="Each stage against the reference configuration"
        >
          <table className="w-full text-[11.5px]">
            <thead>
              <tr className="border-b border-border text-left">
                <th className="label-xs py-1.5">Stage</th>
                <th className="label-xs py-1.5 text-right">Possession F1</th>
                <th className="label-xs py-1.5 text-right">Tactical MAE</th>
                <th className="label-xs py-1.5 text-right">Ranking MRR</th>
                <th className="label-xs py-1.5 text-right">Runtime</th>
              </tr>
            </thead>
            <tbody>
              {RECOVERY_TABLE.map((r, i) => (
                <tr
                  key={r.stage}
                  className={`border-b border-border/60 ${stage === i ? "bg-accent/40" : ""}`}
                >
                  <td className="py-1.5">{r.stage}</td>
                  <td className="num py-1.5 text-right">{r.f1.toFixed(3)}</td>
                  <td className="num py-1.5 text-right">{r.mae.toFixed(3)}</td>
                  <td className="num py-1.5 text-right">{r.mrr.toFixed(3)}</td>
                  <td className="num py-1.5 text-right">{r.runtime}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="mt-3 border border-border bg-panel-alt p-2.5 text-[11.5px] leading-relaxed text-muted-foreground">
            Kalman filtering recovers 0.054 F1 of the 0.089 lost to degradation — approximately 61%
            — at a 74% runtime increase over the degraded pipeline. Interpolation recovers 40% at
            negligible cost.
          </div>
        </Panel>

        <Panel title="Recovery Across Severity" subtitle="Possession F1 by mitigation strategy">
          <LineSeries
            data={DEGRADATION_CURVE}
            xKey="severity"
            series={[
              { key: "degraded", label: "No mitigation", color: "var(--away)" },
              { key: "interpolated", label: "Interpolation", color: "var(--chart-5)" },
              { key: "kalman", label: "Kalman", color: "var(--primary)" },
            ]}
            height={300}
          />
        </Panel>
      </div>
    </>
  );
}
