import { createFileRoute } from "@tanstack/react-router";
import { Metric, Panel, PageHeader } from "@/components/Panel";
import { Pitch, PitchLegend } from "@/components/Pitch";
import { EXPERIMENTS, PIPELINE } from "@/lib/data";
import { useLivePlayers, useLivePossession, useLiveTactical } from "@/hooks/useGameState";
import { useSession } from "@/lib/session";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Overview — Ball-Free Game State Reconstruction" },
      {
        name: "description",
        content:
          "Research operations dashboard for player-only football game state reconstruction: pipeline status, evaluation metrics and recent experiments.",
      },
      { property: "og:title", content: "Overview — Ball-Free Game State Reconstruction" },
      {
        property: "og:description",
        content: "Pipeline status, evaluation metrics and recent experiments.",
      },
    ],
  }),
  component: Overview,
});

const stateColor: Record<string, string> = {
  ok: "border-primary/50 text-primary",
  partial: "border-warn/50 text-warn",
  blocked: "border-destructive/50 text-destructive",
};

function Overview() {
  const { matchId, frame, selectedPlayer, setSelectedPlayer } = useSession();
  const liveState = useLivePlayers(matchId, frame);
  const possession = useLivePossession(matchId, frame);
  const tactical = useLiveTactical(matchId, frame);
  const players = liveState.data ?? [];

  return (
    <>
      <PageHeader
        title="Research Operations Overview"
        description="Reconstruction of football game state from canonical Metrica player tracking. Live frame data is served by the local FastAPI backend; experiment metrics remain explicitly labelled below."
      />

      <div className="mb-4 grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Metric
          label="Current possessor"
          value={String(possession.data?.possessor_id ?? "Data not available")}
          sub="Live ball-free inference"
          tone="primary"
        />
        <Metric
          label="Possession score"
          value={
            typeof possession.data?.score === "number"
              ? possession.data.score.toFixed(3)
              : "Data not available"
          }
          sub="Not an evaluation F1"
        />
        <Metric
          label="Tactical state"
          value={
            tactical.isSuccess ? "Available" : tactical.isLoading ? "Loading" : "Data not available"
          }
          sub="Current frame"
        />
        <Metric label="Experiment metrics" value="See results" sub="Cached Step 80–83 outputs" />
      </div>

      <Panel
        title="Processing Pipeline"
        subtitle="Stage status for the active experiment"
        className="mb-4"
      >
        <div className="grid grid-cols-1 gap-2 md:grid-cols-4 xl:grid-cols-7">
          {PIPELINE.map((s) => (
            <div key={s.id} className="border border-border bg-panel-alt p-2.5">
              <div className="label-xs">Stage {s.id}</div>
              <div className="mt-1 text-[12.5px] font-medium">{s.name}</div>
              <div className="mt-1.5 text-[11px] leading-snug text-muted-foreground">
                {s.detail}
              </div>
              <span
                className={`mt-2 inline-block border px-1.5 py-px text-[10px] uppercase tracking-[0.06em] ${stateColor[s.state]}`}
              >
                {s.state}
              </span>
            </div>
          ))}
        </div>
      </Panel>

      <div className="grid gap-4 xl:grid-cols-[1.6fr_1fr]">
        <Panel
          title="Reconstructed Player State"
          subtitle={`Frame ${frame} · ${players.length || "Data not available"} visible players · ball position not used`}
        >
          {liveState.isLoading ? (
            <div className="flex h-[460px] items-center justify-center text-sm text-muted-foreground">
              Loading live frame…
            </div>
          ) : liveState.isError ? (
            <div className="flex h-[460px] items-center justify-center text-sm text-destructive">
              Backend unavailable
            </div>
          ) : players.length === 0 ? (
            <div className="flex h-[460px] items-center justify-center text-sm text-muted-foreground">
              Data not available
            </div>
          ) : (
            <Pitch players={players} selectedId={selectedPlayer} onSelect={setSelectedPlayer} />
          )}
          <div className="mt-2">
            <PitchLegend />
          </div>
        </Panel>

        <Panel title="Recent Experiments" subtitle="Last six runs">
          <table className="w-full text-[11.5px]">
            <thead>
              <tr className="border-b border-border text-left">
                <th className="label-xs py-1.5">Run</th>
                <th className="label-xs py-1.5 text-right">Poss F1</th>
                <th className="label-xs py-1.5 text-right">MAE</th>
                <th className="label-xs py-1.5 text-right">MRR</th>
                <th className="label-xs py-1.5 text-right">Status</th>
              </tr>
            </thead>
            <tbody>
              {EXPERIMENTS.map((e) => (
                <tr key={e.id} className="border-b border-border/60">
                  <td className="py-1.5">
                    <div className="num">{e.id}</div>
                    <div className="text-[10.5px] text-muted-foreground">{e.name}</div>
                  </td>
                  <td className="num py-1.5 text-right">
                    {e.possessionF1 ? e.possessionF1.toFixed(3) : "—"}
                  </td>
                  <td className="num py-1.5 text-right">
                    {e.tacticalMae ? e.tacticalMae.toFixed(3) : "—"}
                  </td>
                  <td className="num py-1.5 text-right">{e.mrr ? e.mrr.toFixed(3) : "—"}</td>
                  <td className="py-1.5 text-right text-[10.5px] uppercase tracking-[0.05em] text-muted-foreground">
                    {e.status}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Panel>
      </div>
    </>
  );
}
