import { createFileRoute } from "@tanstack/react-router";
import { useMemo, useState } from "react";
import { Metric, Panel, PageHeader } from "@/components/Panel";
import { Pitch } from "@/components/Pitch";
import { useLivePlayers } from "@/hooks/useGameState";
import { useSession } from "@/lib/session";
import type { Player } from "@/lib/data";

const EMPTY_PLAYERS: Player[] = [];

export const Route = createFileRoute("/noise-lab")({
  head: () => ({
    meta: [
      { title: "Noise Lab — Ball-Free Game State Reconstruction" },
      {
        name: "description",
        content:
          "Apply controlled synthetic detection dropout and coordinate noise to the selected live frame.",
      },
    ],
  }),
  component: NoiseLab,
});

function stableUnit(seed: string) {
  let value = 2166136261;
  for (let index = 0; index < seed.length; index += 1) {
    value ^= seed.charCodeAt(index);
    value = Math.imul(value, 16777619);
  }
  return (value >>> 0) / 4294967295;
}

function NoiseLab() {
  const { matchId, frame } = useSession();
  const state = useLivePlayers(matchId, frame);
  const players = state.data ?? EMPTY_PLAYERS;
  const [missing, setMissing] = useState(8);
  const [sigma, setSigma] = useState(0.4);

  const noisy = useMemo(() => {
    const ranked = [...players].sort(
      (a, b) => stableUnit(`${frame}:${a.id}:drop`) - stableUnit(`${frame}:${b.id}:drop`),
    );
    const dropCount = Math.round((ranked.length * missing) / 100);
    const dropped = new Set(ranked.slice(0, dropCount).map((player) => player.id));
    return players
      .filter((player) => !dropped.has(player.id))
      .map((player) => {
        const dx = (stableUnit(`${frame}:${player.id}:x`) * 2 - 1) * sigma;
        const dy = (stableUnit(`${frame}:${player.id}:y`) * 2 - 1) * sigma;
        return {
          ...player,
          x: Math.min(105, Math.max(0, player.x + dx)),
          y: Math.min(68, Math.max(0, player.y + dy)),
        };
      });
  }, [players, frame, missing, sigma]);

  const totalDisplacement = noisy.reduce((sum, player) => {
    const original = players.find((item) => item.id === player.id);
    return sum + (original ? Math.hypot(player.x - original.x, player.y - original.y) : 0);
  }, 0);
  const meanDisplacement = noisy.length ? totalDisplacement / noisy.length : null;

  if (state.isLoading)
    return <PageHeader title="Noise Lab" description="Loading the selected tracked frame…" />;
  if (state.isError || players.length === 0)
    return (
      <PageHeader
        title="Noise Lab"
        description="No live tracking is available for the selected frame. Synthetic degradation is disabled until observed player positions are available."
      />
    );

  return (
    <>
      <PageHeader
        title="Noise Lab"
        description={`Controlled, deterministic perturbation of observed tracking at frame ${frame}. This is a visualization of the configured input damage, not a prediction of model accuracy.`}
      />

      <Panel title="Synthetic input perturbation" className="mb-4">
        <div className="grid gap-4 md:grid-cols-2">
          {[
            {
              label: "Detection dropout (%)",
              value: missing,
              set: setMissing,
              min: 0,
              max: 40,
              step: 1,
            },
            {
              label: "Coordinate jitter limit (m)",
              value: sigma,
              set: setSigma,
              min: 0,
              max: 2,
              step: 0.05,
            },
          ].map((control) => (
            <label key={control.label}>
              <span className="flex justify-between">
                <span className="label-xs">{control.label}</span>
                <span className="num text-[11.5px] text-primary">{control.value}</span>
              </span>
              <input
                type="range"
                min={control.min}
                max={control.max}
                step={control.step}
                value={control.value}
                onChange={(event) => control.set(Number(event.target.value))}
                className="mt-1.5 h-1 w-full accent-[var(--primary)]"
              />
            </label>
          ))}
        </div>
        <p className="mt-3 text-[11px] text-muted-foreground">
          Jitter is applied independently on each pitch axis and clipped to the pitch boundary.
          Dropout is selected deterministically so the same frame and settings reproduce the same
          result.
        </p>
      </Panel>

      <div className="mb-4 grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Metric label="Observed tracks" value={String(players.length)} />
        <Metric label="Synthetic tracks dropped" value={String(players.length - noisy.length)} />
        <Metric label="Tracks retained" value={`${noisy.length} / ${players.length}`} />
        <Metric
          label="Mean displacement"
          value={
            meanDisplacement === null ? "Data not available" : `${meanDisplacement.toFixed(2)} m`
          }
          sub="Among retained tracks"
        />
      </div>

      <div className="grid gap-4 xl:grid-cols-2">
        <Panel title="Observed state" subtitle={`Live tracking · frame ${frame}`}>
          <Pitch players={players} showVectors={false} height={340} />
        </Panel>
        <Panel
          title="Synthetic degraded state"
          subtitle="Observed state plus selected perturbations"
        >
          <Pitch players={noisy} showVectors={false} height={340} />
        </Panel>
      </div>
    </>
  );
}
