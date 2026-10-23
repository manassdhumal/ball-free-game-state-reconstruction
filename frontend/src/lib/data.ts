// Legacy demo fixtures retained only for routes whose API integration is pending.

export type Team = "home" | "away";

export const DEFAULT_TEAM_COLORS: Record<Team, string> = {
  home: "#c83232",
  away: "#3232c8",
};

export function teamColor(players: Player[], team: Team) {
  const palette = players.find((player) => player.teamColors)?.teamColors;
  return (
    palette?.[team] ??
    players.find((player) => player.team === team && player.jerseyColor)?.jerseyColor ??
    DEFAULT_TEAM_COLORS[team]
  );
}

export function jerseyColorName(color: string) {
  const match = /^#?([0-9a-f]{6})$/i.exec(color);
  if (!match) return "Team";
  const value = match[1]!;
  const red = parseInt(value.slice(0, 2), 16);
  const green = parseInt(value.slice(2, 4), 16);
  const blue = parseInt(value.slice(4, 6), 16);
  const max = Math.max(red, green, blue);
  const min = Math.min(red, green, blue);
  if (max < 45) return "Black";
  if (max - min < 22) return max > 215 ? "White" : "Gray";
  if (red > 150 && green < 105 && blue < 125) return "Red";
  if (red > 175 && green > 90 && green < 190 && blue < 100) return "Orange";
  if (red > 165 && green > 150 && blue < 115) return "Yellow";
  if (red > 140 && blue > 115 && green < 135) return "Pink";
  if (blue > red * 1.12 && blue > green * 1.08) return red > 70 ? "Purple" : "Blue";
  if (green > red * 1.12 && green > blue * 0.9) return "Green";
  if (red > 100 && green > 65 && green < 155 && blue < 100) return "Brown";
  return "Gray";
}

export function teamDisplayName(team: Team, players: Player[], overrides: Record<Team, string>) {
  return overrides[team].trim() || jerseyColorName(teamColor(players, team));
}

export function displayTeamId(value: string, players: Player[], overrides: Record<Team, string>) {
  return value.replace(/^(HOME|AWAY)(?=_)/i, (side) =>
    teamDisplayName(side.toUpperCase() === "HOME" ? "home" : "away", players, overrides),
  );
}

export interface Player {
  id: string;
  team: Team;
  jerseyColor?: string;
  teamColors?: Record<Team, string>;
  number: number;
  name: string;
  role: string;
  x: number; // metres, 0..105
  y: number; // metres, 0..68
  vx: number;
  vy: number;
  confidence?: number;
  visible?: boolean;
}

function rng(seed: number) {
  let s = seed;
  return () => {
    s = (s * 1664525 + 1013904223) % 4294967296;
    return s / 4294967296;
  };
}

const HOME_NAMES = [
  "A. Mertens",
  "L. Okafor",
  "D. Vidal",
  "R. Haaland",
  "S. Kovac",
  "T. Ndiaye",
  "M. Rossi",
  "J. Bakker",
  "P. Ferreira",
  "N. Ivanov",
  "C. Duarte",
];
const AWAY_NAMES = [
  "K. Sørensen",
  "F. Marchetti",
  "O. Dembele",
  "H. Yildiz",
  "B. Larsson",
  "G. Silva",
  "E. Nowak",
  "V. Petrov",
  "W. Achebe",
  "I. Tanaka",
  "Z. Halilovic",
];
const ROLES = ["GK", "RB", "CB", "CB", "LB", "DM", "CM", "CM", "RW", "ST", "LW"];

const HOME_BASE: Array<[number, number]> = [
  [6, 34],
  [26, 12],
  [22, 27],
  [22, 41],
  [26, 56],
  [40, 34],
  [50, 22],
  [50, 46],
  [66, 12],
  [70, 34],
  [66, 56],
];

export function buildFrame(frame: number): Player[] {
  const r = rng(frame * 7919 + 13);
  const players: Player[] = [];
  HOME_BASE.forEach(([bx, by], i) => {
    const drift = Math.sin(frame / 90 + i) * 4.2;
    const x = Math.min(103, Math.max(2, bx + drift + (r() - 0.5) * 3));
    const y = Math.min(66, Math.max(2, by + Math.cos(frame / 110 + i * 1.7) * 3.4));
    players.push({
      id: `H${i + 1}`,
      team: "home",
      number: i === 0 ? 1 : i + 1,
      name: HOME_NAMES[i]!,
      role: ROLES[i]!,
      x,
      y,
      vx: Math.cos(frame / 90 + i) * 2.6,
      vy: -Math.sin(frame / 110 + i * 1.7) * 1.8,
    });
  });
  HOME_BASE.forEach(([bx, by], i) => {
    const drift = Math.sin(frame / 95 + i * 1.3) * 3.8;
    const x = Math.min(103, Math.max(2, 105 - bx - drift + (r() - 0.5) * 3));
    const y = Math.min(66, Math.max(2, 68 - by + Math.sin(frame / 105 + i) * 3.1));
    players.push({
      id: `A${i + 1}`,
      team: "away",
      number: i === 0 ? 1 : i + 1,
      name: AWAY_NAMES[i]!,
      role: ROLES[i]!,
      x,
      y,
      vx: -Math.cos(frame / 95 + i * 1.3) * 2.3,
      vy: Math.cos(frame / 105 + i) * 1.6,
    });
  });
  return players;
}

export const dist = (a: Player, b: Player) => Math.hypot(a.x - b.x, a.y - b.y);

export function nearest(p: Player, pool: Player[]) {
  return pool
    .filter((q) => q.id !== p.id)
    .reduce((best, q) => (dist(p, q) < dist(p, best) ? q : best));
}

export function localDensity(p: Player, all: Player[], radius = 10) {
  return all.filter((q) => q.id !== p.id && dist(p, q) <= radius).length;
}

export function pressureOn(p: Player, all: Player[]) {
  const opp = all.filter((q) => q.team !== p.team);
  const v = opp.reduce((acc, q) => acc + Math.exp(-dist(p, q) / 6), 0);
  return Math.min(1, v / 2.4);
}

export function supportFor(p: Player, all: Player[]) {
  const mates = all.filter((q) => q.team === p.team && q.id !== p.id);
  const v = mates.reduce((acc, q) => acc + Math.exp(-dist(p, q) / 12), 0);
  return Math.min(1, v / 4.2);
}

export function teamGeometry(all: Player[], team: Team) {
  const t = all.filter((p) => p.team === team);
  const xs = t.map((p) => p.x);
  const ys = t.map((p) => p.y);
  const width = Math.max(...ys) - Math.min(...ys);
  const length = Math.max(...xs) - Math.min(...xs);
  const cx = xs.reduce((a, b) => a + b, 0) / t.length;
  const cy = ys.reduce((a, b) => a + b, 0) / t.length;
  const spread = t.reduce((a, p) => a + Math.hypot(p.x - cx, p.y - cy), 0) / t.length;
  return {
    width: +width.toFixed(1),
    length: +length.toFixed(1),
    centroidX: +cx.toFixed(1),
    centroidY: +cy.toFixed(1),
    compactness: +(1 / (1 + spread / 12)).toFixed(3),
    spread: +spread.toFixed(1),
  };
}

export interface Candidate {
  rank: number;
  targetId: string;
  target: string;
  distance: number;
  pressure: number;
  space: number;
  obstruction: number;
  score: number;
  stability: number;
}

export function candidatesFor(source: Player, all: Player[]): Candidate[] {
  const mates = all.filter((p) => p.team === source.team && p.id !== source.id);
  const opp = all.filter((p) => p.team !== source.team);
  const rows = mates.map((m) => {
    const d = dist(source, m);
    const pr = pressureOn(m, all);
    const space = Math.min(1, 1 - Math.exp(-dist(m, nearest(m, opp)) / 9));
    const obstruction = Math.min(
      1,
      opp.filter((o) => {
        const t =
          ((o.x - source.x) * (m.x - source.x) + (o.y - source.y) * (m.y - source.y)) /
          (d * d || 1);
        if (t <= 0 || t >= 1) return false;
        const px = source.x + t * (m.x - source.x);
        const py = source.y + t * (m.y - source.y);
        return Math.hypot(o.x - px, o.y - py) < 3.5;
      }).length / 3,
    );
    const progression = (m.x - source.x) / 105;
    const score =
      0.34 * space + 0.26 * (1 - pr) + 0.22 * (1 - obstruction) + 0.18 * (progression + 0.5);
    return {
      rank: 0,
      targetId: m.id,
      target: `#${m.number} ${m.name}`,
      distance: +d.toFixed(1),
      pressure: +pr.toFixed(3),
      space: +space.toFixed(3),
      obstruction: +obstruction.toFixed(3),
      score: +score.toFixed(3),
      stability: +Math.max(0.31, Math.min(0.98, 0.55 + space * 0.4 - pr * 0.25)).toFixed(2),
    };
  });
  rows.sort((a, b) => b.score - a.score);
  return rows.map((r, i) => ({ ...r, rank: i + 1 }));
}

export const DATASETS = [
  { id: "metrica-1", label: "Metrica Sample_Game_1", frames: 141000, fps: 25 },
  { id: "metrica-2", label: "Metrica Sample_Game_2", frames: 139500, fps: 25 },
  { id: "synthetic-a", label: "Synthetic_Controlled_A", frames: 45000, fps: 25 },
];

export const EXPERIMENTS = [
  {
    id: "EXP_080A_042",
    name: "Controlled Analysis",
    dataset: "Metrica Sample_Game_1",
    status: "complete",
    seed: 42,
    possessionF1: 0.741,
    tacticalMae: 0.118,
    eventF1: 0.612,
    mrr: 0.583,
    runtime: "12m 41s",
    date: "2026-08-30",
    notes: "Reference configuration. No perturbation applied.",
  },
  {
    id: "EXP_080A_043",
    name: "Coordinate Jitter σ=0.5",
    dataset: "Metrica Sample_Game_1",
    status: "complete",
    seed: 42,
    possessionF1: 0.714,
    tacticalMae: 0.139,
    eventF1: 0.588,
    mrr: 0.541,
    runtime: "12m 55s",
    date: "2026-08-31",
    notes: "Mild positional noise; ranking largely preserved.",
  },
  {
    id: "EXP_080A_051",
    name: "Player Dropout 15%",
    dataset: "Metrica Sample_Game_1",
    status: "complete",
    seed: 7,
    possessionF1: 0.652,
    tacticalMae: 0.191,
    eventF1: 0.503,
    mrr: 0.446,
    runtime: "11m 08s",
    date: "2026-09-02",
    notes: "Support graph degrades faster than pressure estimates.",
  },
  {
    id: "EXP_080B_012",
    name: "Kalman Recovery",
    dataset: "Metrica Sample_Game_2",
    status: "complete",
    seed: 7,
    possessionF1: 0.706,
    tacticalMae: 0.147,
    eventF1: 0.571,
    mrr: 0.512,
    runtime: "19m 22s",
    date: "2026-09-04",
    notes: "Recovers ~62% of dropout-induced loss; runtime cost is material.",
  },
  {
    id: "EXP_080B_019",
    name: "Fragmentation Sweep",
    dataset: "Synthetic_Controlled_A",
    status: "running",
    seed: 13,
    possessionF1: 0.688,
    tacticalMae: 0.161,
    eventF1: 0.534,
    mrr: 0.478,
    runtime: "—",
    date: "2026-09-09",
    notes: "In progress. Partial results only.",
  },
  {
    id: "EXP_080B_021",
    name: "Availability Jitter",
    dataset: "Synthetic_Controlled_A",
    status: "queued",
    seed: 13,
    possessionF1: 0,
    tacticalMae: 0,
    eventF1: 0,
    mrr: 0,
    runtime: "—",
    date: "2026-09-10",
    notes: "Queued behind fragmentation sweep.",
  },
];

export const POSSESSION_SEGMENTS = Array.from({ length: 26 }, (_, i) => {
  const start = i * 96;
  return {
    start,
    end: start + 96,
    team: (i % 3 === 0 ? "away" : i % 4 === 0 ? "contested" : "home") as Team | "contested",
    confidence: +(0.52 + ((i * 37) % 43) / 100).toFixed(2),
  };
});

export const EVENTS = [
  { frame: 1102, t: "00:44.1", predicted: "PASS", annotated: "PASS", conf: 0.81 },
  { frame: 1168, t: "00:46.7", predicted: "CARRY", annotated: "CARRY", conf: 0.74 },
  { frame: 1231, t: "00:49.2", predicted: "PASS", annotated: "PASS", conf: 0.69 },
  { frame: 1250, t: "00:50.0", predicted: "RECEPTION", annotated: "RECEPTION", conf: 0.77 },
  { frame: 1298, t: "00:51.9", predicted: "PASS", annotated: "—", conf: 0.55 },
  { frame: 1344, t: "00:53.8", predicted: "—", annotated: "DUEL", conf: 0.0 },
  { frame: 1401, t: "00:56.0", predicted: "CARRY", annotated: "CARRY", conf: 0.72 },
  { frame: 1468, t: "00:58.7", predicted: "PASS", annotated: "SET_PIECE", conf: 0.51 },
];

export const EVENT_EVAL = [
  { cls: "PASS", tp: 412, fp: 96, fn: 88, precision: 0.811, recall: 0.824, f1: 0.817 },
  { cls: "CARRY", tp: 301, fp: 121, fn: 143, precision: 0.713, recall: 0.678, f1: 0.695 },
  { cls: "RECEPTION", tp: 388, fp: 104, fn: 112, precision: 0.789, recall: 0.776, f1: 0.782 },
  { cls: "DUEL", tp: 87, fp: 133, fn: 168, precision: 0.395, recall: 0.341, f1: 0.366 },
  { cls: "SET_PIECE", tp: 41, fp: 22, fn: 19, precision: 0.651, recall: 0.683, f1: 0.667 },
];

export function stateFeaturesFor(players: Player[], selected: Player) {
  const team = teamGeometry(players, selected.team);
  const opponents = players.filter((player) => player.team !== selected.team);
  const nearestOpponentDistance = opponents.length
    ? dist(selected, nearest(selected, opponents))
    : null;
  const supportCount = players.filter(
    (player) =>
      player.team === selected.team && player.id !== selected.id && dist(selected, player) <= 12,
  ).length;
  const confidenceValues = players
    .map((player) => player.confidence)
    .filter((value): value is number => value != null && Number.isFinite(value));
  const trackingConfidence = confidenceValues.length
    ? confidenceValues.reduce((sum, value) => sum + value, 0) / confidenceValues.length
    : 0;
  const feature = (
    key: string,
    label: string,
    value: number | string,
    unit: string,
    type: string,
    conf = trackingConfidence,
  ) => ({ key, label, value, unit, type, conf });

  return [
    feature(
      "pressure_index",
      "Pressure Index",
      pressureOn(selected, players).toFixed(3),
      "—",
      "Geometry proxy",
    ),
    feature("support_count", "Support Count (r≤12m)", supportCount, "players", "Measured proxy"),
    feature(
      "nearest_opponent_distance",
      "Nearest Opponent Distance",
      nearestOpponentDistance == null ? "Data not available" : nearestOpponentDistance.toFixed(1),
      "m",
      "Measured geometry",
      nearestOpponentDistance == null ? 0 : trackingConfidence,
    ),
    feature("team_width", "Team Width", team.width.toFixed(1), "m", "Measured geometry"),
    feature("team_length", "Team Length", team.length.toFixed(1), "m", "Measured geometry"),
    feature("compactness", "Team Compactness", team.compactness.toFixed(3), "—", "Geometry proxy"),
    feature("team_spread", "Team Spread", team.spread.toFixed(1), "m", "Measured geometry"),
    feature(
      "local_density",
      "Local Density (r≤10m)",
      localDensity(selected, players),
      "players",
      "Measured proxy",
    ),
    feature(
      "forward_options",
      "Forward Options",
      "Not determined",
      "—",
      "Attack direction unavailable",
      0,
    ),
  ];
}

export const ROBUSTNESS_CONDITIONS = [
  {
    condition: "Reference",
    top1: 1.0,
    top3: 1.0,
    spearman: 1.0,
    kendall: 1.0,
    flip: 0.0,
    f1: 0.741,
  },
  {
    condition: "Jitter σ=0.25m",
    top1: 0.94,
    top3: 0.98,
    spearman: 0.93,
    kendall: 0.85,
    flip: 0.06,
    f1: 0.732,
  },
  {
    condition: "Jitter σ=0.50m",
    top1: 0.88,
    top3: 0.96,
    spearman: 0.87,
    kendall: 0.77,
    flip: 0.12,
    f1: 0.714,
  },
  {
    condition: "Jitter σ=1.00m",
    top1: 0.74,
    top3: 0.9,
    spearman: 0.76,
    kendall: 0.64,
    flip: 0.26,
    f1: 0.681,
  },
  {
    condition: "Dropout 5%",
    top1: 0.91,
    top3: 0.97,
    spearman: 0.9,
    kendall: 0.8,
    flip: 0.09,
    f1: 0.722,
  },
  {
    condition: "Dropout 15%",
    top1: 0.72,
    top3: 0.86,
    spearman: 0.71,
    kendall: 0.58,
    flip: 0.28,
    f1: 0.652,
  },
  {
    condition: "Dropout 25%",
    top1: 0.55,
    top3: 0.74,
    spearman: 0.58,
    kendall: 0.44,
    flip: 0.45,
    f1: 0.581,
  },
  {
    condition: "Fragmentation 0.4",
    top1: 0.68,
    top3: 0.83,
    spearman: 0.68,
    kendall: 0.53,
    flip: 0.32,
    f1: 0.634,
  },
];

export const DEGRADATION_CURVE = Array.from({ length: 11 }, (_, i) => {
  const sev = i / 10;
  return {
    severity: +sev.toFixed(1),
    degraded: +(0.741 * Math.exp(-1.15 * sev)).toFixed(3),
    interpolated: +(0.741 * Math.exp(-0.78 * sev)).toFixed(3),
    kalman: +(0.741 * Math.exp(-0.52 * sev)).toFixed(3),
  };
});

export const RECOVERY_TABLE = [
  { stage: "Reference", f1: 0.741, mae: 0.118, mrr: 0.583, runtime: "12m 41s" },
  { stage: "Degraded (15% dropout)", f1: 0.652, mae: 0.191, mrr: 0.446, runtime: "11m 08s" },
  { stage: "Linear Interpolation", f1: 0.688, mae: 0.164, mrr: 0.489, runtime: "13m 02s" },
  { stage: "Kalman Filter", f1: 0.706, mae: 0.147, mrr: 0.512, runtime: "19m 22s" },
];

export const TRACKING_QUALITY = {
  missingness: 0.062,
  completeness: 0.938,
  validity: 0.991,
  jitter: 0.21,
  gaps: 1483,
  meanGapLen: 7.4,
};

export const QUALITY_DISTRIBUTION = Array.from({ length: 12 }, (_, i) => ({
  bin: `${i * 2}-${i * 2 + 2}`,
  frames: Math.round(4200 * Math.exp(-i / 2.6) + (i % 3) * 120),
}));

export const PIPELINE = [
  { id: 1, name: "Tracking Data", state: "ok", detail: "Metrica CSV · 25 FPS · 22 tracks" },
  { id: 2, name: "Player-Only State", state: "ok", detail: "12 features / frame" },
  { id: 3, name: "Possession Inference", state: "ok", detail: "F1 0.741" },
  { id: 4, name: "Tactical Analysis", state: "ok", detail: "MAE 0.118" },
  {
    id: 5,
    name: "Counterfactual Ranking",
    state: "partial",
    detail: "MRR 0.583 · hypothetical only",
  },
  { id: 6, name: "Robustness", state: "partial", detail: "8 conditions evaluated" },
  { id: 7, name: "Broadcast / GSR", state: "ok", detail: "YOLO + ByteTrack Studio Active" },
];
