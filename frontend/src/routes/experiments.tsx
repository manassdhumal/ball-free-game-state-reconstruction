import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import { X } from "lucide-react";
import { Panel, PageHeader } from "@/components/Panel";
import { EXPERIMENTS } from "@/lib/data";

export const Route = createFileRoute("/experiments")({
  head: () => ({
    meta: [
      { title: "Experiments — Ball-Free Game State Reconstruction" },
      {
        name: "description",
        content:
          "Experiment register with configuration metadata, evaluation metrics, run status and per-run detail panels.",
      },
      { property: "og:title", content: "Experiments — Ball-Free Game State Reconstruction" },
      { property: "og:description", content: "Experiment register and run metadata." },
    ],
  }),
  component: Experiments,
});

function Experiments() {
  const [status, setStatus] = useState("all");
  const [dataset, setDataset] = useState("all");
  const [open, setOpen] = useState<string | null>(null);

  const rows = EXPERIMENTS.filter(
    (e) =>
      (status === "all" || e.status === status) && (dataset === "all" || e.dataset === dataset),
  );
  const detail = EXPERIMENTS.find((e) => e.id === open);

  return (
    <>
      <PageHeader
        title="Experiments"
        description="Every evaluation run is recorded with its dataset, seed and resulting metrics. Runs are immutable once complete."
      />

      <Panel className="mb-4" bodyClassName="p-2.5">
        <div className="flex flex-wrap items-center gap-3 text-[12px]">
          <span className="label-xs">Filter</span>
          <select
            value={status}
            onChange={(e) => setStatus(e.target.value)}
            className="border border-input bg-panel-alt px-2 py-1"
          >
            {["all", "complete", "running", "queued"].map((s) => (
              <option key={s} value={s}>
                Status: {s}
              </option>
            ))}
          </select>
          <select
            value={dataset}
            onChange={(e) => setDataset(e.target.value)}
            className="border border-input bg-panel-alt px-2 py-1"
          >
            {["all", ...new Set(EXPERIMENTS.map((e) => e.dataset))].map((d) => (
              <option key={d} value={d}>
                Dataset: {d}
              </option>
            ))}
          </select>
          <span className="num text-muted-foreground">{rows.length} runs</span>
        </div>
      </Panel>

      <Panel title="Experiment Register">
        <table className="w-full text-[11.5px]">
          <thead>
            <tr className="border-b border-border text-left">
              <th className="label-xs py-1.5">Run ID</th>
              <th className="label-xs py-1.5">Configuration</th>
              <th className="label-xs py-1.5">Dataset</th>
              <th className="label-xs py-1.5 text-right">Seed</th>
              <th className="label-xs py-1.5 text-right">Poss F1</th>
              <th className="label-xs py-1.5 text-right">MAE</th>
              <th className="label-xs py-1.5 text-right">Event F1</th>
              <th className="label-xs py-1.5 text-right">MRR</th>
              <th className="label-xs py-1.5 text-right">Runtime</th>
              <th className="label-xs py-1.5">Status</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((e) => (
              <tr
                key={e.id}
                onClick={() => setOpen(e.id)}
                className="cursor-pointer border-b border-border/60 hover:bg-muted/50"
              >
                <td className="num py-1.5">{e.id}</td>
                <td className="py-1.5">{e.name}</td>
                <td className="py-1.5 text-muted-foreground">{e.dataset}</td>
                <td className="num py-1.5 text-right">{e.seed}</td>
                <td className="num py-1.5 text-right">
                  {e.possessionF1 ? e.possessionF1.toFixed(3) : "—"}
                </td>
                <td className="num py-1.5 text-right">
                  {e.tacticalMae ? e.tacticalMae.toFixed(3) : "—"}
                </td>
                <td className="num py-1.5 text-right">{e.eventF1 ? e.eventF1.toFixed(3) : "—"}</td>
                <td className="num py-1.5 text-right">{e.mrr ? e.mrr.toFixed(3) : "—"}</td>
                <td className="num py-1.5 text-right">{e.runtime}</td>
                <td
                  className={`py-1.5 text-[10.5px] uppercase tracking-[0.05em] ${
                    e.status === "complete"
                      ? "text-primary"
                      : e.status === "running"
                        ? "text-warn"
                        : "text-muted-foreground"
                  }`}
                >
                  {e.status}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </Panel>

      {detail && (
        <div
          className="fixed inset-0 z-50 flex justify-end bg-black/50"
          onClick={() => setOpen(null)}
        >
          <aside
            className="h-full w-[420px] overflow-y-auto border-l border-border bg-panel p-4"
            onClick={(ev) => ev.stopPropagation()}
          >
            <div className="mb-3 flex items-start justify-between">
              <div>
                <div className="num text-[14px] font-medium">{detail.id}</div>
                <div className="text-[11.5px] text-muted-foreground">{detail.name}</div>
              </div>
              <button
                onClick={() => setOpen(null)}
                aria-label="Close"
                className="p-1 hover:bg-muted"
              >
                <X className="h-4 w-4" />
              </button>
            </div>

            <dl className="text-[11.5px]">
              {[
                ["Dataset", detail.dataset],
                ["Seed", String(detail.seed)],
                ["Date", detail.date],
                ["Runtime", detail.runtime],
                ["Status", detail.status],
                ["Possession F1", detail.possessionF1 ? detail.possessionF1.toFixed(3) : "—"],
                ["Tactical MAE", detail.tacticalMae ? detail.tacticalMae.toFixed(3) : "—"],
                ["Event F1", detail.eventF1 ? detail.eventF1.toFixed(3) : "—"],
                ["Ranking MRR", detail.mrr ? detail.mrr.toFixed(3) : "—"],
              ].map(([k, v]) => (
                <div key={k} className="flex justify-between border-b border-border/60 py-1.5">
                  <dt className="text-muted-foreground">{k}</dt>
                  <dd className="num">{v}</dd>
                </div>
              ))}
            </dl>

            <div className="mt-4">
              <div className="label-xs mb-1">Notes</div>
              <p className="text-[11.5px] leading-relaxed text-muted-foreground">{detail.notes}</p>
            </div>

            <div className="mt-4">
              <div className="label-xs mb-1">Configuration</div>
              <pre className="num overflow-x-auto border border-border bg-panel-alt p-2.5 text-[11px] leading-relaxed text-muted-foreground">
                {`pipeline: player_only_v0.8
fps: 25
features: 12
possession_model: geometric_posterior
ranking_weights:
  space: 0.34
  pressure: 0.26
  clearance: 0.22
  progression: 0.18
ball_input: none`}
              </pre>
            </div>
          </aside>
        </div>
      )}
    </>
  );
}
