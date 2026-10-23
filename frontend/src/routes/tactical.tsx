import { createFileRoute, Link } from "@tanstack/react-router";
import { Metric, Panel, PageHeader } from "@/components/Panel";
import { BarSeries, ScatterSeries } from "@/components/Charts";
import {
  localDensity,
  pressureOn,
  supportFor,
  teamColor,
  teamDisplayName,
  teamGeometry,
} from "@/lib/data";
import { useLivePlayers, useLiveTactical } from "@/hooks/useGameState";
import { useSession } from "@/lib/session";
import { Video, ArrowRight } from "lucide-react";

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
  const { matchId, frame, teamNames } = useSession();
  const livePlayers = useLivePlayers(matchId, frame);
  const tactical = useLiveTactical(matchId, frame);
  const isVideo = matchId.startsWith("uploaded_") || matchId.startsWith("video_");

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
  const tacticalData = tactical.data as {
    tas?: number | null;
    possessor_id?: string;
    possessor_team?: string;
  };

  const perPlayer = players.map((p) => ({
    name: `#${p.number}`,
    team: p.team,
    pressure: +pressureOn(p, players).toFixed(3),
    support: +supportFor(p, players).toFixed(3),
    density: localDensity(p, players),
    pitchX: +(p.x / 105).toFixed(3),
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
          label="TAS (current frame)"
          value={
            typeof tacticalData.tas === "number"
              ? tacticalData.tas.toFixed(3)
              : "Data not available"
          }
          sub={
            tacticalData.possessor_id
              ? `${tacticalData.possessor_id} · direction-dependent term excluded`
              : "No inferred possessor"
          }
          tone="primary"
        />
        <Metric
          label={`Mean Pressure (${teamDisplayName("home", players, teamNames)})`}
          value={avg(homePlayers.map((p) => p.pressure)).toFixed(3)}
        />
        <Metric
          label={`Mean Support (${teamDisplayName("home", players, teamNames)})`}
          value={avg(homePlayers.map((p) => p.support)).toFixed(3)}
        />
        <Metric
          label={`${teamDisplayName("home", players, teamNames)} Compactness`}
          value={home.compactness.toFixed(3)}
          sub={`Spread ${home.spread} m`}
        />
        <Metric
          label={`${teamDisplayName("away", players, teamNames)} Compactness`}
          value={away.compactness.toFixed(3)}
          sub={`Spread ${away.spread} m`}
        />
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Panel
          title="Current-frame team compactness"
          subtitle={`Frame ${frame} snapshot · not a historical TAS curve`}
        >
          <BarSeries
            data={[{ metric: "Compactness", home: home.compactness, away: away.compactness }]}
            xKey="metric"
            series={[
              {
                key: "home",
                label: teamDisplayName("home", players, teamNames),
                color: teamColor(players, "home"),
              },
              {
                key: "away",
                label: teamDisplayName("away", players, teamNames),
                color: teamColor(players, "away"),
              },
            ]}
          />
        </Panel>

        <Panel title="Team Width & Length" subtitle={`Frame ${frame} · metres, visible players`}>
          <BarSeries
            data={[
              { metric: "Width", home: home.width, away: away.width },
              { metric: "Length", home: home.length, away: away.length },
            ]}
            xKey="metric"
            series={[
              {
                key: "home",
                label: teamDisplayName("home", players, teamNames),
                color: teamColor(players, "home"),
              },
              {
                key: "away",
                label: teamDisplayName("away", players, teamNames),
                color: teamColor(players, "away"),
              },
            ]}
          />
        </Panel>

        <Panel title="Pressure vs Space" subtitle="Visible tracked players at the current frame">
          <ScatterSeries
            data={homePlayers.map((p) => ({ ...p, space: +(1 - p.pressure).toFixed(3) }))}
            xKey="pressure"
            yKey="space"
            xLabel="Pressure index"
            yLabel="Space"
            color={teamColor(players, "home")}
          />
        </Panel>

        <Panel
          title="Support vs Pitch Position"
          subtitle={`${teamDisplayName("home", players, teamNames)} players · attack direction is not assumed`}
        >
          <ScatterSeries
            data={homePlayers}
            xKey="pitchX"
            yKey="support"
            xLabel="Pitch X (normalised)"
            yLabel="Support index"
            color="var(--primary)"
          />
        </Panel>

        <Panel title="Local Density" subtitle="Visible tracked players within 10 m">
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
              {
                key: "home",
                label: teamDisplayName("home", players, teamNames),
                color: teamColor(players, "home"),
              },
              {
                key: "away",
                label: teamDisplayName("away", players, teamNames),
                color: teamColor(players, "away"),
              },
            ]}
          />
        </Panel>
      </div>

      <Panel
        title="Per-Player Tactical Table"
        subtitle="All visible tracked players; goalkeeper roles are not available"
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
              <th className="label-xs py-1.5 text-right">Pitch X</th>
            </tr>
          </thead>
          <tbody>
            {[...homePlayers, ...awayPlayers].map((p, i) => (
              <tr key={`${p.team}-${p.name}-${i}`} className="border-b border-border/60">
                <td className="num py-1.5">{p.name}</td>
                <td className="py-1.5" style={{ color: teamColor(players, p.team) }}>
                  {teamDisplayName(p.team, players, teamNames)}
                </td>
                <td className="num py-1.5 text-right">{p.pressure.toFixed(3)}</td>
                <td className="num py-1.5 text-right">{p.support.toFixed(3)}</td>
                <td className="num py-1.5 text-right">{p.density}</td>
                <td className="num py-1.5 text-right">{p.pitchX.toFixed(3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Panel>
    </>
  );
}
