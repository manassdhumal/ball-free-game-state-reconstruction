import { createFileRoute } from "@tanstack/react-router";
import { Panel, PageHeader } from "@/components/Panel";
import { BarSeries } from "@/components/Charts";

export const Route = createFileRoute("/results")({
  head: () => ({
    meta: [
      { title: "Research Results — Ball-Free Game State Reconstruction" },
      {
        name: "description",
        content:
          "Benchmarks, method comparisons and honest negative findings for player-only game state reconstruction.",
      },
      { property: "og:title", content: "Research Results — Ball-Free Game State Reconstruction" },
      { property: "og:description", content: "Method comparisons and negative findings." },
    ],
  }),
  component: Results,
});

const METHODS = [
  {
    method: "Nearest-player heuristic (baseline)",
    poss: 0.612,
    event: 0.481,
    mrr: 0.402,
    note: "No learning; surprisingly strong on possession.",
  },
  {
    method: "Voronoi control baseline",
    poss: 0.658,
    event: 0.502,
    mrr: 0.437,
    note: "Better in open play, worse in crowded areas.",
  },
  {
    method: "Geometric posterior (ours)",
    poss: 0.741,
    event: 0.612,
    mrr: 0.583,
    note: "Reference configuration.",
  },
  {
    method: "Geometric posterior + Kalman",
    poss: 0.706,
    event: 0.571,
    mrr: 0.512,
    note: "Only beneficial on degraded input.",
  },
  {
    method: "Ball-informed upper bound",
    poss: 0.918,
    event: 0.847,
    mrr: 0.771,
    note: "Uses ball position; not comparable, shown as ceiling.",
  },
];

function Results() {
  return (
    <>
      <PageHeader
        title="Research Results"
        description="Aggregated results across the evaluation suite, including comparisons against simple baselines and a ball-informed ceiling that this work does not attempt to reach."
      />

      <Panel title="Method Comparison" className="mb-4">
        <table className="w-full text-[11.5px]">
          <thead>
            <tr className="border-b border-border text-left">
              <th className="label-xs py-1.5">Method</th>
              <th className="label-xs py-1.5 text-right">Possession F1</th>
              <th className="label-xs py-1.5 text-right">Event F1</th>
              <th className="label-xs py-1.5 text-right">Ranking MRR</th>
              <th className="label-xs py-1.5">Note</th>
            </tr>
          </thead>
          <tbody>
            {METHODS.map((m) => (
              <tr key={m.method} className="border-b border-border/60">
                <td className="py-1.5">{m.method}</td>
                <td className="num py-1.5 text-right">{m.poss.toFixed(3)}</td>
                <td className="num py-1.5 text-right">{m.event.toFixed(3)}</td>
                <td className="num py-1.5 text-right">{m.mrr.toFixed(3)}</td>
                <td className="py-1.5 text-muted-foreground">{m.note}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Panel>

      <Panel title="Benchmark Summary" className="mb-4">
        <BarSeries
          data={METHODS.map((m) => ({
            method: m.method.split(" ")[0]!,
            poss: m.poss,
            event: m.event,
            mrr: m.mrr,
          }))}
          xKey="method"
          series={[
            { key: "poss", label: "Possession F1", color: "var(--primary)" },
            { key: "event", label: "Event F1", color: "var(--home)" },
            { key: "mrr", label: "Ranking MRR", color: "var(--chart-5)" },
          ]}
          height={250}
        />
      </Panel>

      <Panel
        title="Negative and Inconclusive Findings"
        subtitle="Reported in full; not filtered for favourable outcomes"
      >
        <ul className="list-disc space-y-2 pl-4 text-[11.5px] leading-relaxed text-muted-foreground">
          <li>
            The nearest-player heuristic reaches 0.612 possession F1 — 83% of our method's score —
            so most of the gain comes from a small number of contested frames, not from general
            modelling.
          </li>
          <li>
            Duel detection (F1 0.366) is no better than chance-adjusted marking detection. We do not
            claim duel recognition is feasible without ball information.
          </li>
          <li>
            Kalman filtering <em>reduces</em> accuracy on clean input (0.741 → 0.706). It is only a
            mitigation for degraded tracking, not a general improvement.
          </li>
          <li>
            Counterfactual ranking has no ground truth. MRR is computed against annotated actual
            passes, which measures agreement with observed choices, not decision quality.
          </li>
          <li>
            Results on Synthetic_Controlled_A do not transfer cleanly to Metrica data; the
            fragmentation sweep shows a 0.04–0.07 F1 gap that we cannot currently explain.
          </li>
        </ul>
      </Panel>
    </>
  );
}
