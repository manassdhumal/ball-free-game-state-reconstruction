import { createFileRoute } from "@tanstack/react-router";
import { Metric, Panel, PageHeader } from "@/components/Panel";
import { BarSeries, LineSeries, ScatterSeries } from "@/components/Charts";
import { localDensity, pressureOn, supportFor, teamGeometry } from "@/lib/data";
import { useLivePlayers, useLiveTactical } from "@/hooks/useGameState";
import { useSession } from "@/lib/session";

export const Route = createFileRoute("/tactical")({
  head: () => ({
    meta: [
      { title: "Tactical Analysis — Ball-Free Game State Reconstruction" },
      {
        name: "description",
        content:
          "Tactical advantage, pressure, support, space, forward progression, team shape and local density derived from player-only tracking.",
      },
      { property: "og:title", content: "Tactical Analysis — Ball-Free Game State Reconstruction" },
      {
        property: "og:description",
        content: "Pressure, support, space and team-shape analytics from player positions alone.",
      },
    ],
  }),
  component: Tactical,
});

function Tactical() {
  const { matchId, frame } = useSession();
  const livePlayers = useLivePlayers(matchId, frame);
  const tactical = useLiveTactical(matchId, frame);
  if (livePlayers.isLoading || tactical.isLoading)
    return (
      <PageHeader title="Tactical Analysis" description="Loading current-frame tactical state…" />
    );
  if (livePlayers.isError || tactical.isError || !livePlayers.data?.length)
    return (
      <PageHeader
        title="Tactical Analysis"
        description="Data not available. Start the backend and select a valid frame."
      />
    );
  const players = livePlayers.data;
  const home = teamGeometry(players, "home");
  const away = teamGeometry(players, "away");

  const series = Array.from({ length: 40 }, (_, i) => {
    const f = frame - 780 + i * 20;
    const h = home;
    const a = away;
    return {
      frame: f,
      advantage: +(h.compactness - a.compactness).toFixed(3),
      homeWidth: h.width,
      awayWidth: a.width,
      homeLength: h.length,
      awayLength: a.length,
    };
  });

  const perPlayer = players
    .filter((p) => p.role !== "GK")
    .map((p) => ({
      name: `#${p.number}`,
      team: p.team,
      pressure: +pressureOn(p, players).toFixed(3),
      support: +supportFor(p, players).toFixed(3),
      density: localDensity(p, players),
      progression: +((p.team === "home" ? p.x : 105 - p.x) / 105).toFixed(3),
    }));

  const homePlayers = perPlayer.filter((p) => p.team === "home");
  const awayPlayers = perPlayer.filter((p) => p.team === "away");
  const avg = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length;

  return (
    <>
      <PageHeader
        title="Tactical Analysis"
        description="Team- and player-level tactical indicators computed from the reconstructed player-only state. All quantities are relative, unitless indices unless a unit is given."
      />

      <div className="mb-4 grid grid-cols-2 gap-3 lg:grid-cols-5">
        <Metric
          label="Tactical Advantage"
          value={((home.compactness - away.compactness) * 2).toFixed(3)}
          sub="Home minus away, ±1 scale"
          tone="primary"
        />
        <Metric
          label="Mean Pressure (Home)"
          value={avg(homePlayers.map((p) => p.pressure)).toFixed(3)}
        />
        <Metric
          label="Mean Support (Home)"
          value={avg(homePlayers.map((p) => p.support)).toFixed(3)}
        />
        <Metric
          label="Home Compactness"
          value={home.compactness.toFixed(3)}
          sub={`Spread ${home.spread} m`}
        />
        <Metric
          label="Away Compactness"
          value={away.compactness.toFixed(3)}
          sub={`Spread ${away.spread} m`}
        />
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Panel
          title="Tactical Advantage"
          subtitle="Rolling window, ±800 frames around current position"
        >
          <LineSeries
            data={series}
            xKey="frame"
            series={[{ key: "advantage", label: "Advantage (home−away)", color: "var(--primary)" }]}
          />
        </Panel>

        <Panel title="Team Width & Length" subtitle="Metres, outfield players">
          <LineSeries
            data={series}
            xKey="frame"
            series={[
              { key: "homeWidth", label: "Home width", color: "var(--home)" },
              { key: "awayWidth", label: "Away width", color: "var(--away)" },
              { key: "homeLength", label: "Home length", color: "rgba(255,255,255,0.4)" },
            ]}
          />
        </Panel>

        <Panel title="Pressure vs Space" subtitle="Per outfield player at the current frame">
          <ScatterSeries
            data={homePlayers.map((p) => ({ ...p, space: +(1 - p.pressure).toFixed(3) }))}
            xKey="pressure"
            yKey="space"
            xLabel="Pressure index"
            yLabel="Space"
            color="var(--home)"
          />
        </Panel>

        <Panel title="Support vs Forward Progression" subtitle="Home players">
          <ScatterSeries
            data={homePlayers}
            xKey="progression"
            yKey="support"
            xLabel="Forward progression (normalised x)"
            yLabel="Support index"
            color="var(--primary)"
          />
        </Panel>

        <Panel title="Local Density" subtitle="Players within 10 m, per outfield player">
          <BarSeries
            data={perPlayer}
            xKey="name"
            series={[{ key: "density", label: "Neighbours (r≤10m)", color: "var(--chart-5)" }]}
          />
        </Panel>

        <Panel title="Defensive Spread & Compactness" subtitle="Current frame comparison">
          <BarSeries
            data={[
              { metric: "Width (m)", home: home.width, away: away.width },
              { metric: "Length (m)", home: home.length, away: away.length },
              { metric: "Spread (m)", home: home.spread, away: away.spread },
              {
                metric: "Compact ×50",
                home: +(home.compactness * 50).toFixed(1),
                away: +(away.compactness * 50).toFixed(1),
              },
            ]}
            xKey="metric"
            series={[
              { key: "home", label: "Home", color: "var(--home)" },
              { key: "away", label: "Away", color: "var(--away)" },
            ]}
          />
        </Panel>
      </div>

      <Panel
        title="Per-Player Tactical Table"
        subtitle="Away side shown for comparison"
        className="mt-4"
      >
        <table className="w-full text-[11.5px]">
          <thead>
            <tr className="border-b border-border text-left">
              <th className="label-xs py-1.5">Player</th>
              <th className="label-xs py-1.5">Team</th>
              <th className="label-xs py-1.5 text-right">Pressure</th>
              <th className="label-xs py-1.5 text-right">Support</th>
              <th className="label-xs py-1.5 text-right">Density</th>
              <th className="label-xs py-1.5 text-right">Progression</th>
            </tr>
          </thead>
          <tbody>
            {[...homePlayers, ...awayPlayers].map((p, i) => (
              <tr key={`${p.team}-${p.name}-${i}`} className="border-b border-border/60">
                <td className="num py-1.5">{p.name}</td>
                <td className={`py-1.5 ${p.team === "home" ? "text-home" : "text-away"}`}>
                  {p.team === "home" ? "Home" : "Away"}
                </td>
                <td className="num py-1.5 text-right">{p.pressure.toFixed(3)}</td>
                <td className="num py-1.5 text-right">{p.support.toFixed(3)}</td>
                <td className="num py-1.5 text-right">{p.density}</td>
                <td className="num py-1.5 text-right">{p.progression.toFixed(3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Panel>
    </>
  );
}
