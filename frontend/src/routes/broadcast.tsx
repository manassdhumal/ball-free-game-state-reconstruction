import { createFileRoute } from "@tanstack/react-router";
import { Lock } from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { Panel, PageHeader } from "@/components/Panel";
import { getGsrStatus } from "@/api/gsr";

export const Route = createFileRoute("/broadcast")({
  head: () => ({
    meta: [
      { title: "Broadcast / GSR — Ball-Free Game State Reconstruction" },
      {
        name: "description",
        content:
          "Broadcast game state reconstruction integration status: pipeline modules present but unverified and disabled.",
      },
      { property: "og:title", content: "Broadcast / GSR — Ball-Free Game State Reconstruction" },
      {
        property: "og:description",
        content: "Integration pending; modules disabled and unverified.",
      },
    ],
  }),
  component: Broadcast,
});

const MODULES = [
  {
    name: "Frame Ingest",
    detail: "Broadcast video decode at 25 FPS",
    reason: "No verified video source in this build.",
  },
  {
    name: "Player Detection",
    detail: "Per-frame bounding boxes",
    reason: "Detector weights not evaluated on this dataset.",
  },
  {
    name: "Re-Identification",
    detail: "Identity persistence across cuts",
    reason: "No labelled re-ID benchmark available.",
  },
  {
    name: "Camera Calibration",
    detail: "Homography to pitch coordinates",
    reason: "Calibration accuracy unmeasured; would silently bias all geometry.",
  },
  {
    name: "Pitch Registration",
    detail: "Line-based keypoint fitting",
    reason: "Depends on calibration module.",
  },
  {
    name: "Coordinate Export",
    detail: "Tracking stream to reconstruction pipeline",
    reason: "Blocked by upstream modules.",
  },
];

function Broadcast() {
  const status = useQuery({ queryKey: ["gsr-status"], queryFn: getGsrStatus, staleTime: 30_000 });
  const state =
    status.data?.status?.toUpperCase() ?? (status.isLoading ? "PENDING" : "UNAVAILABLE");
  return (
    <>
      <PageHeader
        title="Broadcast / GSR"
        description="Broadcast-derived game state reconstruction is not part of the verified v0.8 results. The architecture below is scaffolded but disabled; no metric shown elsewhere in this workstation depends on it."
      />

      <div className="mb-4 border border-destructive/40 bg-destructive/5 px-3 py-2.5 text-[11.5px] leading-relaxed text-destructive">
        <span className="mr-1.5 font-semibold uppercase tracking-[0.06em]">
          {state} — integration pending
        </span>
        {status.data?.message ?? "No broadcast pipeline output has been validated."} Any figure
        produced by these modules must be treated as unverified and must not be reported alongside
        tracking-based results.
      </div>

      <Panel title="Architecture Modules" subtitle="All modules disabled in this build">
        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
          {MODULES.map((m, i) => (
            <div key={m.name} className="border border-border bg-panel-alt p-3 opacity-60">
              <div className="flex items-center justify-between">
                <span className="label-xs">Module {i + 1}</span>
                <Lock className="h-3.5 w-3.5 text-destructive" />
              </div>
              <div className="mt-1 text-[12.5px] font-medium">{m.name}</div>
              <div className="mt-0.5 text-[11px] text-muted-foreground">{m.detail}</div>
              <div className="mt-2 border-t border-border pt-2 text-[11px] leading-relaxed text-muted-foreground">
                {m.reason}
              </div>
              <span className="mt-2 inline-block border border-destructive/50 px-1.5 py-px text-[10px] uppercase tracking-[0.06em] text-destructive">
                disabled
              </span>
            </div>
          ))}
        </div>
      </Panel>

      <Panel title="Conditions for Enabling" className="mt-4">
        <ol className="list-decimal space-y-1.5 pl-4 text-[11.5px] leading-relaxed text-muted-foreground">
          <li>
            Calibration error measured against surveyed pitch points, reported with a confidence
            interval.
          </li>
          <li>
            Re-identification evaluated on a labelled broadcast segment with identity-switch counts.
          </li>
          <li>End-to-end positional error benchmarked against synchronised optical tracking.</li>
          <li>
            Only then may broadcast-derived state feed the reconstruction pipeline, clearly labelled
            as such.
          </li>
        </ol>
      </Panel>
    </>
  );
}
