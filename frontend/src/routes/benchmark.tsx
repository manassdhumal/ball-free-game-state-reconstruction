import { createFileRoute } from "@tanstack/react-router";
import { Panel, PageHeader } from "@/components/Panel";
import { LineSeries } from "@/components/Charts";
import { DEGRADATION_CURVE, ROBUSTNESS_CONDITIONS } from "@/lib/data";

export const Route = createFileRoute("/benchmark")({
  head: () => ({
    meta: [
      { title: "Robustness Benchmark — Ball-Free Game State Reconstruction" },
      {
        name: "description",
        content:
          "Standardised comparison of perturbation conditions with degradation and recovery curves across the evaluation suite.",
      },
      {
        property: "og:title",
        content: "Robustness Benchmark — Ball-Free Game State Reconstruction",
      },
      {
        property: "og:description",
        content: "Condition comparison table plus degradation and recovery curves.",
      },
    ],
  }),
  component: Benchmark,
});

function Benchmark() {
  return (
    <>
      <PageHeader
        title="Robustness Benchmark"
        description="Fixed benchmark suite run over 5 000 sampled frames per condition, seed-matched to the reference experiment EXP_080A_042."
      />

      <Panel title="Condition Comparison" className="mb-4">
        <table className="w-full text-[11.5px]">
          <thead>
            <tr className="border-b border-border text-left">
              <th className="label-xs py-1.5">Condition</th>
              <th className="label-xs py-1.5 text-right">Top-1</th>
              <th className="label-xs py-1.5 text-right">Top-3</th>
              <th className="label-xs py-1.5 text-right">Spearman ρ</th>
              <th className="label-xs py-1.5 text-right">Kendall τ</th>
              <th className="label-xs py-1.5 text-right">Flip rate</th>
              <th className="label-xs py-1.5 text-right">Possession F1</th>
            </tr>
          </thead>
          <tbody>
            {ROBUSTNESS_CONDITIONS.map((c) => (
              <tr key={c.condition} className="border-b border-border/60">
                <td className="py-1.5">{c.condition}</td>
                <td className="num py-1.5 text-right">{c.top1.toFixed(2)}</td>
                <td className="num py-1.5 text-right">{c.top3.toFixed(2)}</td>
                <td className="num py-1.5 text-right">{c.spearman.toFixed(2)}</td>
                <td className="num py-1.5 text-right">{c.kendall.toFixed(2)}</td>
                <td className={`num py-1.5 text-right ${c.flip > 0.25 ? "text-warn" : ""}`}>
                  {c.flip.toFixed(2)}
                </td>
                <td className={`num py-1.5 text-right ${c.f1 < 0.6 ? "text-destructive" : ""}`}>
                  {c.f1.toFixed(3)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </Panel>

      <div className="grid gap-4 lg:grid-cols-2">
        <Panel
          title="Degradation Curve"
          subtitle="Possession F1 against normalised perturbation severity"
        >
          <LineSeries
            data={DEGRADATION_CURVE}
            xKey="severity"
            series={[{ key: "degraded", label: "Degraded", color: "var(--away)" }]}
            height={260}
          />
        </Panel>

        <Panel title="Recovery Curves" subtitle="Effect of interpolation and Kalman filtering">
          <LineSeries
            data={DEGRADATION_CURVE}
            xKey="severity"
            series={[
              { key: "degraded", label: "No mitigation", color: "var(--away)" },
              { key: "interpolated", label: "Linear interpolation", color: "var(--chart-5)" },
              { key: "kalman", label: "Kalman filter", color: "var(--primary)" },
            ]}
            height={260}
          />
        </Panel>
      </div>

      <Panel title="Reading the Benchmark" className="mt-4">
        <ul className="list-disc space-y-1.5 pl-4 text-[11.5px] leading-relaxed text-muted-foreground">
          <li>
            Top-1 retention degrades roughly linearly with dropout but non-linearly with coordinate
            jitter beyond σ = 0.5 m.
          </li>
          <li>
            Rank correlation stays above 0.7 for all mild conditions, meaning the ordering of
            mid-ranked options is more stable than the identity of the single best option.
          </li>
          <li>
            Kalman filtering recovers around 62% of the dropout-induced F1 loss but adds roughly 55%
            runtime; it does not restore ranking identity at high severity.
          </li>
        </ul>
      </Panel>
    </>
  );
}
