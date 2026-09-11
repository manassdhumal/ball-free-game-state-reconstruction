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
      "Results derive from two Metrica sample matches and one synthetic set; no claim of generalisation across leagues, competition levels or playing styles.",
      "Tracking has a 6.2% missingness rate and 1 483 gaps; all downstream metrics inherit this.",
      "Pitch coordinates are assumed to be a correct 105 × 68 m registration; registration error is not modelled.",
      "No possession or event annotation exists for the synthetic dataset, so it is used only for perturbation sweeps.",
    ],
  },
  {
    title: "Models",
    items: [
      "Possession is inferred from geometry and motion only; a player may be labelled possessor while the ball is elsewhere.",
      "Pressure, support and space are unitless constructed indices with no external validation against expert judgement.",
      "The counterfactual ranking weights were selected on the reference match and are not cross-validated.",
      "Constant-velocity assumptions in the Kalman stage are violated during accelerations and direction changes.",
    ],
  },
  {
    title: "Evaluation",
    items: [
      "Counterfactual option ranking has no ground truth; MRR measures agreement with observed choices, not decision quality.",
      "Event F1 is macro-averaged over imbalanced classes; DUEL (F1 0.366) drags the average and is not usable.",
      "Robustness statistics come from 24 synthetic trials per configuration, not from real degraded capture.",
      "Confidence intervals are not reported for any headline metric in this build.",
    ],
  },
  {
    title: "Broadcast Tracking",
    items: [
      "The broadcast/GSR pipeline is disabled and no output from it has been validated.",
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
        This is a v0.8 research build backed by mock evaluation data. It is not a match-analysis
        product and must not be used to assess player decision-making.
      </div>
    </>
  );
}
