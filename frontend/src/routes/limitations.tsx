import { createFileRoute } from "@tanstack/react-router";
import { Panel, PageHeader } from "@/components/Panel";

export const Route = createFileRoute("/limitations")({
  head: () => ({
    meta: [
      { title: "Limitations — Ball-Free Game State Reconstruction" },
      {
        name: "description",
        content:
          "Explicit research limitations covering data, models, evaluation and broadcast tracking for player-only reconstruction.",
      },
      { property: "og:title", content: "Limitations — Ball-Free Game State Reconstruction" },
      { property: "og:description", content: "Data, model, evaluation and broadcast limitations." },
    ],
  }),
  component: Limitations,
});

const SECTIONS = [
  {
    title: "Data",
    items: [
      "The aggregate evaluation artifacts report 60 paired runs over controlled conditions; this does not establish generalisation across leagues, competition levels or playing styles.",
      "The clean reference tracking artifact reports 21.4% missingness; current-match quality is measured separately from the selected clip.",
      "Pitch coordinates are assumed to be a correct 105 × 68 m registration; registration error is not modelled.",
      "Synthetic perturbations are controlled inputs, not representative samples of every real broadcast failure mode.",
    ],
  },
  {
    title: "Models",
    items: [
      "Possession is inferred from geometry and motion only; a player may be labelled possessor while the ball is elsewhere.",
      "Pressure, support and space are unitless constructed indices with no external validation against expert judgement.",
      "Counterfactual rankings are heuristic options; they do not show which pass a player actually chose or whether a pass would succeed.",
      "Possession and tactical features use player-only geometry; without ball tracking they remain inferred proxies.",
    ],
  },
  {
    title: "Evaluation",
    items: [
      "The event-validation report evaluates 28 supported reference events; at 1 s tolerance its aggregate F1 is 0.262, with 45 false positives and 17 missed events.",
      "Ranking evaluation has 15 labeled states and uses annotated pass targets only; it does not evaluate pass completion.",
      "Aggregate confidence intervals describe the paired offline benchmark runs, not uncertainty on a newly uploaded clip.",
    ],
  },
  {
    title: "Broadcast Tracking",
    items: [
      "Video reconstruction outputs are clip-specific estimates; no ground-truth validation is included for newly uploaded clips.",
      "Camera calibration error would bias every geometric feature simultaneously and is currently unmeasured.",
      "Re-identification failures would appear as player dropout, which the robustness study shows is the most damaging degradation mode.",
    ],
  },
];

function Limitations() {
  return (
    <>
      <PageHeader
        title="Limitations"
        description="Known constraints on what this work can support. These apply to every figure shown elsewhere in the workstation."
      />

      <div className="grid gap-4 lg:grid-cols-2">
        {SECTIONS.map((s) => (
          <Panel key={s.title} title={s.title}>
            <ul className="list-disc space-y-2 pl-4 text-[11.5px] leading-relaxed text-muted-foreground">
              {s.items.map((i) => (
                <li key={i}>{i}</li>
              ))}
            </ul>
          </Panel>
        ))}
      </div>

      <div className="mt-4 border border-warn/40 bg-warn/5 px-3 py-2.5 text-[11.5px] leading-relaxed text-warn">
        <span className="mr-1.5 font-semibold uppercase tracking-[0.06em]">Scope</span>
        Offline benchmark metrics and live match estimates have different evidence bases. Live
        estimates describe model inferences from tracking and must not be interpreted as verified
        ball possession, completed events, or player decision quality.
      </div>
    </>
  );
}
