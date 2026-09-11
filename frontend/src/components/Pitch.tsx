import { useState } from "react";
import { dist, localDensity, nearest, type Player } from "@/lib/data";

const W = 105;
const H = 68;
const PAD = 4;

export interface PassLine {
  fromId: string;
  toId: string;
  rank: number;
  score: number;
}

export function Pitch({
  players,
  selectedId,
  onSelect,
  passLines = [],
  showVectors = true,
  heat,
  height = 460,
}: {
  players: Player[];
  selectedId?: string | null;
  onSelect?: (id: string) => void;
  passLines?: PassLine[];
  showVectors?: boolean;
  heat?: boolean;
  height?: number;
}) {
  const [hover, setHover] = useState<Player | null>(null);
  const byId = (id: string) => players.find((p) => p.id === id);

  const lineStroke = "rgba(255,255,255,0.22)";

  return (
    <div className="relative">
      <svg
        viewBox={`${-PAD} ${-PAD} ${W + PAD * 2} ${H + PAD * 2}`}
        style={{ height }}
        className="w-full select-none"
      >
        <rect x={-PAD} y={-PAD} width={W + PAD * 2} height={H + PAD * 2} fill="var(--turf)" />
        {heat && (
          <g opacity={0.5}>
            {players.map((p) => (
              <circle
                key={`heat-${p.id}`}
                cx={p.x}
                cy={p.y}
                r={9}
                fill={p.team === "home" ? "var(--home)" : "var(--away)"}
                opacity={0.14}
              />
            ))}
          </g>
        )}

        <g fill="none" stroke={lineStroke} strokeWidth={0.28}>
          <rect x={0} y={0} width={W} height={H} />
          <line x1={W / 2} y1={0} x2={W / 2} y2={H} />
          <circle cx={W / 2} cy={H / 2} r={9.15} />
          <circle cx={W / 2} cy={H / 2} r={0.5} fill={lineStroke} />
          {/* penalty areas */}
          <rect x={0} y={H / 2 - 20.16} width={16.5} height={40.32} />
          <rect x={W - 16.5} y={H / 2 - 20.16} width={16.5} height={40.32} />
          {/* six yard */}
          <rect x={0} y={H / 2 - 9.16} width={5.5} height={18.32} />
          <rect x={W - 5.5} y={H / 2 - 9.16} width={5.5} height={18.32} />
          {/* goals */}
          <rect x={-1.8} y={H / 2 - 3.66} width={1.8} height={7.32} />
          <rect x={W} y={H / 2 - 3.66} width={1.8} height={7.32} />
          <circle cx={11} cy={H / 2} r={0.4} fill={lineStroke} />
          <circle cx={W - 11} cy={H / 2} r={0.4} fill={lineStroke} />
          <path d={`M 16.5 ${H / 2 - 7.3} A 9.15 9.15 0 0 1 16.5 ${H / 2 + 7.3}`} />
          <path d={`M ${W - 16.5} ${H / 2 - 7.3} A 9.15 9.15 0 0 0 ${W - 16.5} ${H / 2 + 7.3}`} />
        </g>

        {/* candidate pass lines */}
        {passLines.map((l) => {
          const a = byId(l.fromId);
          const b = byId(l.toId);
          if (!a || !b) return null;
          const top = l.rank === 1;
          return (
            <g key={`${l.fromId}-${l.toId}`}>
              <line
                x1={a.x}
                y1={a.y}
                x2={b.x}
                y2={b.y}
                stroke={top ? "var(--primary)" : "rgba(255,255,255,0.34)"}
                strokeWidth={top ? 0.5 : 0.28}
                strokeDasharray={top ? undefined : "1.4 1.2"}
              />
              <text
                x={(a.x + b.x) / 2}
                y={(a.y + b.y) / 2 - 0.8}
                fontSize={2}
                textAnchor="middle"
                fill={top ? "var(--primary)" : "rgba(255,255,255,0.55)"}
              >
                {l.rank}. {l.score.toFixed(2)}
              </text>
            </g>
          );
        })}

        {/* velocity vectors */}
        {showVectors &&
          players.map((p) => (
            <line
              key={`v-${p.id}`}
              x1={p.x}
              y1={p.y}
              x2={p.x + p.vx * 0.9}
              y2={p.y + p.vy * 0.9}
              stroke={p.team === "home" ? "var(--home)" : "var(--away)"}
              strokeWidth={0.28}
              opacity={0.85}
            />
          ))}

        {players.map((p) => {
          const selected = p.id === selectedId;
          return (
            <g
              key={p.id}
              onMouseEnter={() => setHover(p)}
              onMouseLeave={() => setHover(null)}
              onClick={() => onSelect?.(p.id)}
              style={{ cursor: onSelect ? "pointer" : "default" }}
            >
              <circle
                cx={p.x}
                cy={p.y}
                r={selected ? 2.4 : 2}
                fill={p.team === "home" ? "var(--home)" : "var(--away)"}
                stroke={selected ? "var(--primary)" : "rgba(0,0,0,0.55)"}
                strokeWidth={selected ? 0.6 : 0.22}
              />
              <text
                x={p.x}
                y={p.y + 0.75}
                fontSize={2.1}
                textAnchor="middle"
                fill="rgba(255,255,255,0.92)"
                style={{ pointerEvents: "none", fontVariantNumeric: "tabular-nums" }}
              >
                {p.number}
              </text>
            </g>
          );
        })}
      </svg>

      {hover && (
        <div className="pointer-events-none absolute right-2 top-2 w-56 border border-border bg-panel-alt/95 p-2.5 text-[11px]">
          <div className="mb-1 flex items-center justify-between">
            <span className="font-medium">
              #{hover.number} {hover.name}
            </span>
            <span className="label-xs">{hover.team === "home" ? "HOME" : "AWAY"}</span>
          </div>
          <Row k="Position" v={`${hover.x.toFixed(1)}, ${hover.y.toFixed(1)} m`} />
          <Row
            k="Confidence"
            v={hover.confidence == null ? "Data not available" : hover.confidence.toFixed(3)}
          />
          <Row k="Velocity" v={`${Math.hypot(hover.vx, hover.vy).toFixed(2)} m/s`} />
          <Row
            k="Nearest opp."
            v={`#${
              nearest(
                hover,
                players.filter((q) => q.team !== hover.team),
              ).number
            } · ${dist(
              hover,
              nearest(
                hover,
                players.filter((q) => q.team !== hover.team),
              ),
            ).toFixed(1)} m`}
          />
          <Row
            k="Nearest mate"
            v={`#${
              nearest(
                hover,
                players.filter((q) => q.team === hover.team),
              ).number
            } · ${dist(
              hover,
              nearest(
                hover,
                players.filter((q) => q.team === hover.team),
              ),
            ).toFixed(1)} m`}
          />
          <Row k="Local density" v={`${localDensity(hover, players)} (r≤10m)`} />
        </div>
      )}
    </div>
  );
}

function Row({ k, v }: { k: string; v: string }) {
  return (
    <div className="flex justify-between gap-2 py-px">
      <span className="text-muted-foreground">{k}</span>
      <span className="num">{v}</span>
    </div>
  );
}

export function PitchLegend() {
  return (
    <div className="flex flex-wrap items-center gap-4 text-[11px] text-muted-foreground">
      <span className="flex items-center gap-1.5">
        <i className="inline-block h-2.5 w-2.5 rounded-full bg-home" /> Home
      </span>
      <span className="flex items-center gap-1.5">
        <i className="inline-block h-2.5 w-2.5 rounded-full bg-away" /> Away
      </span>
      <span className="flex items-center gap-1.5">
        <i className="inline-block h-0.5 w-4 bg-primary" /> Top-ranked hypothetical option
      </span>
      <span>Vectors show instantaneous velocity (0.9 s projection)</span>
    </div>
  );
}
