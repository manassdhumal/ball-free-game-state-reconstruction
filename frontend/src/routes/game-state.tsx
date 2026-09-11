import { createFileRoute } from "@tanstack/react-router";
import { Panel, PageHeader, TypeTag } from "@/components/Panel";
import { Pitch } from "@/components/Pitch";
import { dist, pressureOn, STATE_FEATURES, supportFor, teamGeometry } from "@/lib/data";
import { useSession } from "@/lib/session";
import { useLivePlayers } from "@/hooks/useGameState";

export const Route = createFileRoute("/game-state")({
  head: () => ({
    meta: [
      { title: "Game State — Ball-Free Game State Reconstruction" },
      {
        name: "description",
        content:
          "Per-frame state feature matrix, pressure map, support graph and team geometry derived from player positions only.",
      },
      { property: "og:title", content: "Game State — Ball-Free Game State Reconstruction" },
      {
        property: "og:description",
        content: "State feature matrix, pressure map, support graph and team geometry.",
      },
    ],
  }),
  component: GameState,
});

function GameState() {
  const { frame, selectedPlayer, setSelectedPlayer } = useSession();
  const liveState = useLivePlayers("metrica_game1", frame);
  const players = liveState.data ?? [];
  if (liveState.isLoading)
    return <PageHeader title="Game State" description="Loading live canonical player state…" />;
  if (liveState.isError || players.length === 0)
    return (
      <PageHeader
        title="Game State"
        description="Data not available. Start the FastAPI backend and select a valid frame."
      />
    );
  const sel = players.find((p) => p.id === selectedPlayer) ?? players[9]!;
  const home = teamGeometry(players, "home");
  const away = teamGeometry(players, "away");
  const mates = players.filter((p) => p.team === sel.team && p.id !== sel.id);
  const forward = mates
    .filter((m) => (sel.team === "home" ? m.x > sel.x + 3 : m.x < sel.x - 3))
    .sort((a, b) => dist(sel, a) - dist(sel, b))
    .slice(0, 6);

  return (
    <>
      <PageHeader
        title="Game State"
        description={`Reconstructed state vector at frame ${frame}. Each feature is labelled by provenance: model-derived quantities are inferred, heuristics are rule-based, measured proxies come directly from tracking geometry.`}
      />

      <div className="grid gap-4 xl:grid-cols-[1.1fr_1fr]">
        <Panel
          title="State Feature Matrix"
          subtitle="Frame-level features with provenance and confidence"
        >
          <table className="w-full text-[11.5px]">
            <thead>
              <tr className="border-b border-border text-left">
                <th className="label-xs py-1.5">Feature</th>
                <th className="label-xs py-1.5 text-right">Value</th>
                <th className="label-xs py-1.5">Unit</th>
                <th className="label-xs py-1.5">Type</th>
                <th className="label-xs py-1.5 text-right">Conf.</th>
              </tr>
            </thead>
            <tbody>
              {STATE_FEATURES.map((f) => (
                <tr key={f.key} className="border-b border-border/60">
                  <td className="py-1.5">{f.label}</td>
                  <td className="num py-1.5 text-right">{f.value}</td>
                  <td className="py-1.5 text-muted-foreground">{f.unit}</td>
                  <td className="py-1.5">
                    <TypeTag type={f.type} />
                  </td>
                  <td className="num py-1.5 text-right">{f.conf.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Panel>

        <div className="space-y-4">
          <Panel title="Pressure Map" subtitle="Opponent-proximity pressure field, per player">
            <Pitch
              players={players}
              selectedId={sel.id}
              onSelect={setSelectedPlayer}
              heat
              height={320}
            />
          </Panel>

          <Panel title="Team Geometry" subtitle="Outfield players only">
            <table className="w-full text-[11.5px]">
              <thead>
                <tr className="border-b border-border text-left">
                  <th className="label-xs py-1.5">Metric</th>
                  <th className="label-xs py-1.5 text-right">Home</th>
                  <th className="label-xs py-1.5 text-right">Away</th>
                </tr>
              </thead>
              <tbody>
                {[
                  ["Width (m)", home.width, away.width],
                  ["Length (m)", home.length, away.length],
                  ["Centroid X (m)", home.centroidX, away.centroidX],
                  ["Centroid Y (m)", home.centroidY, away.centroidY],
                  ["Compactness", home.compactness, away.compactness],
                  ["Defensive spread (m)", home.spread, away.spread],
                ].map(([k, h, a]) => (
                  <tr key={String(k)} className="border-b border-border/60">
                    <td className="py-1.5 text-muted-foreground">{k}</td>
                    <td className="num py-1.5 text-right text-home">{h}</td>
                    <td className="num py-1.5 text-right text-away">{a}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Panel>
        </div>
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-2">
        <Panel
          title="Support Graph"
          subtitle={`Teammates of #${sel.number} ${sel.name} within 25 m`}
        >
          <table className="w-full text-[11.5px]">
            <thead>
              <tr className="border-b border-border text-left">
                <th className="label-xs py-1.5">Teammate</th>
                <th className="label-xs py-1.5 text-right">Dist (m)</th>
                <th className="label-xs py-1.5 text-right">Pressure</th>
                <th className="label-xs py-1.5 text-right">Support</th>
              </tr>
            </thead>
            <tbody>
              {mates
                .filter((m) => dist(sel, m) <= 25)
                .sort((a, b) => dist(sel, a) - dist(sel, b))
                .map((m) => (
                  <tr key={m.id} className="border-b border-border/60">
                    <td className="py-1.5">
                      #{m.number} {m.name}
                    </td>
                    <td className="num py-1.5 text-right">{dist(sel, m).toFixed(1)}</td>
                    <td className="num py-1.5 text-right">{pressureOn(m, players).toFixed(3)}</td>
                    <td className="num py-1.5 text-right">{supportFor(m, players).toFixed(3)}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        </Panel>

        <Panel title="Forward Options" subtitle="Teammates positioned ahead of the selected player">
          <table className="w-full text-[11.5px]">
            <thead>
              <tr className="border-b border-border text-left">
                <th className="label-xs py-1.5">Teammate</th>
                <th className="label-xs py-1.5 text-right">Δx (m)</th>
                <th className="label-xs py-1.5 text-right">Dist (m)</th>
                <th className="label-xs py-1.5 text-right">Pressure</th>
              </tr>
            </thead>
            <tbody>
              {forward.map((m) => (
                <tr key={m.id} className="border-b border-border/60">
                  <td className="py-1.5">
                    #{m.number} {m.name}
                  </td>
                  <td className="num py-1.5 text-right">{(m.x - sel.x).toFixed(1)}</td>
                  <td className="num py-1.5 text-right">{dist(sel, m).toFixed(1)}</td>
                  <td className="num py-1.5 text-right">{pressureOn(m, players).toFixed(3)}</td>
                </tr>
              ))}
              {forward.length === 0 && (
                <tr>
                  <td colSpan={4} className="py-3 text-center text-muted-foreground">
                    No forward options at this frame.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </Panel>
      </div>
    </>
  );
}
