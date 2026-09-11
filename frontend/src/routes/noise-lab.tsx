import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import { Metric, Panel, PageHeader } from "@/components/Panel";
import { Pitch } from "@/components/Pitch";
import { LineSeries } from "@/components/Charts";
import { buildFrame } from "@/lib/data";
import { useSession } from "@/lib/session";

export const Route = createFileRoute("/noise-lab")({
  head: () => ({
    meta: [
      { title: "Noise Lab — Ball-Free Game State Reconstruction" },
      {
        name: "description",
        content:
          "Interactive noise injection for missing detections, track gaps, coordinate jitter and fragmentation with live impact preview.",
      },
      { property: "og:title", content: "Noise Lab — Ball-Free Game State Reconstruction" },
      {
        property: "og:description",
        content: "Inject tracking noise and preview its effect on the reconstructed state.",
      },
    ],
  }),
  component: NoiseLab,
});

function NoiseLab() {
  const { frame } = useSession();
  const [missing, setMissing] = useState(8);
  const [gaps, setGaps] = useState(12);
  const [sigma, setSigma] = useState(0.4);
  const [frag, setFrag] = useState(0.2);

  const base = buildFrame(frame);
  let s = 991;
  const rnd = () => {
    s = (s * 1664525 + 1013904223) % 4294967296;
    return s / 4294967296;
  };
  const noisy = base
    .filter(() => rnd() > missing / 100)
    .map((p) => ({
      ...p,
      x: p.x + (rnd() - 0.5) * 2 * sigma,
      y: p.y + (rnd() - 0.5) * 2 * sigma,
    }));

  const severity = missing / 100 + sigma / 2 + frag / 2 + gaps / 200;
  const f1 = 0.741 * Math.exp(-1.05 * severity);
  const mae = 0.118 * (1 + 1.9 * severity);
  const mrr = 0.583 * Math.exp(-0.95 * severity);

  const curve = Array.from({ length: 11 }, (_, i) => {
    const sev = (i / 10) * Math.max(0.2, severity * 2);
    return {
      severity: +sev.toFixed(2),
      f1: +(0.741 * Math.exp(-1.05 * sev)).toFixed(3),
      mrr: +(0.583 * Math.exp(-0.95 * sev)).toFixed(3),
    };
  });

  return (
    <>
      <PageHeader
        title="Noise Lab"
        description="Apply synthetic tracking degradation to the current frame and observe the immediate effect on the reconstructed state and downstream metrics. Impact estimates use the fitted degradation model from the robustness benchmark."
      />

      <Panel title="Noise Parameters" className="mb-4">
        <div className="grid gap-4 md:grid-cols-4">
          {[
            {
              label: "Missing detections (%)",
              v: missing,
              set: setMissing,
              min: 0,
              max: 40,
              step: 1,
            },
            { label: "Track gaps (per min)", v: gaps, set: setGaps, min: 0, max: 60, step: 1 },
            { label: "Jitter σ (m)", v: sigma, set: setSigma, min: 0, max: 2, step: 0.05 },
            { label: "Fragmentation", v: frag, set: setFrag, min: 0, max: 1, step: 0.05 },
          ].map((c) => (
            <div key={c.label}>
              <div className="flex justify-between">
                <span className="label-xs">{c.label}</span>
                <span className="num text-[11.5px] text-primary">{c.v}</span>
              </div>
              <input
                type="range"
                min={c.min}
                max={c.max}
                step={c.step}
                value={c.v}
                onChange={(e) => c.set(Number(e.target.value))}
                className="mt-1.5 h-1 w-full accent-[var(--primary)]"
              />
            </div>
          ))}
        </div>
      </Panel>

      <div className="mb-4 grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Metric label="Tracks Retained" value={`${noisy.length} / ${base.length}`} />
        <Metric
          label="Projected Possession F1"
          value={f1.toFixed(3)}
          tone={f1 < 0.6 ? "warn" : "primary"}
        />
        <Metric label="Projected Tactical MAE" value={mae.toFixed(3)} />
        <Metric label="Projected Ranking MRR" value={mrr.toFixed(3)} />
      </div>

      <div className="grid gap-4 xl:grid-cols-2">
        <Panel title="Reference State" subtitle={`Frame ${frame} · clean tracking`}>
          <Pitch players={base} showVectors={false} height={340} />
        </Panel>
        <Panel title="Degraded State" subtitle="After noise injection">
          <Pitch players={noisy} showVectors={false} height={340} />
        </Panel>
      </div>

      <Panel
        title="Impact Preview"
        subtitle="Fitted metric response across the swept severity range"
        className="mt-4"
      >
        <LineSeries
          data={curve}
          xKey="severity"
          series={[
            { key: "f1", label: "Possession F1", color: "var(--primary)" },
            { key: "mrr", label: "Ranking MRR", color: "var(--home)" },
          ]}
          height={240}
        />
        <p className="mt-2 text-[11px] text-muted-foreground">
          Current composite severity: <span className="num">{severity.toFixed(3)}</span>.
          Projections are extrapolations of the benchmark fit, not fresh evaluations.
        </p>
      </Panel>
    </>
  );
}
