import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";

const axis = {
  stroke: "rgba(255,255,255,0.25)",
  tick: { fill: "rgba(255,255,255,0.55)", fontSize: 10 },
  tickLine: false,
};

const tooltipStyle = {
  contentStyle: {
    background: "var(--panel-alt)",
    border: "1px solid rgba(255,255,255,0.12)",
    borderRadius: 2,
    fontSize: 11,
  },
  labelStyle: { color: "rgba(255,255,255,0.7)", fontSize: 11 },
};

export function LineSeries({
  data,
  xKey,
  series,
  height = 220,
}: {
  data: Array<Record<string, number | string>>;
  xKey: string;
  series: Array<{ key: string; label: string; color: string }>;
  height?: number;
}) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 6, right: 10, bottom: 0, left: -18 }}>
        <CartesianGrid stroke="rgba(255,255,255,0.06)" vertical={false} />
        <XAxis dataKey={xKey} {...axis} />
        <YAxis {...axis} />
        <Tooltip {...tooltipStyle} />
        {series.map((s) => (
          <Line
            key={s.key}
            type="monotone"
            dataKey={s.key}
            name={s.label}
            stroke={s.color}
            strokeWidth={1.4}
            dot={false}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}

export function BarSeries({
  data,
  xKey,
  series,
  height = 220,
}: {
  data: Array<Record<string, number | string>>;
  xKey: string;
  series: Array<{ key: string; label: string; color: string }>;
  height?: number;
}) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ top: 6, right: 10, bottom: 0, left: -18 }}>
        <CartesianGrid stroke="rgba(255,255,255,0.06)" vertical={false} />
        <XAxis dataKey={xKey} {...axis} />
        <YAxis {...axis} />
        <Tooltip {...tooltipStyle} cursor={{ fill: "rgba(255,255,255,0.04)" }} />
        {series.map((s) => (
          <Bar key={s.key} dataKey={s.key} name={s.label} fill={s.color} />
        ))}
      </BarChart>
    </ResponsiveContainer>
  );
}

export function ScatterSeries({
  data,
  xKey,
  yKey,
  xLabel,
  yLabel,
  color,
  height = 240,
}: {
  data: Array<Record<string, number | string>>;
  xKey: string;
  yKey: string;
  xLabel: string;
  yLabel: string;
  color: string;
  height?: number;
}) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <ScatterChart margin={{ top: 6, right: 10, bottom: 14, left: -18 }}>
        <CartesianGrid stroke="rgba(255,255,255,0.06)" />
        <XAxis
          type="number"
          dataKey={xKey}
          name={xLabel}
          {...axis}
          label={{
            value: xLabel,
            position: "insideBottom",
            offset: -8,
            fill: "rgba(255,255,255,0.5)",
            fontSize: 10,
          }}
        />
        <YAxis type="number" dataKey={yKey} name={yLabel} {...axis} />
        <ZAxis range={[28, 28]} />
        <Tooltip {...tooltipStyle} cursor={{ stroke: "rgba(255,255,255,0.2)" }} />
        <Scatter data={data} fill={color} />
      </ScatterChart>
    </ResponsiveContainer>
  );
}
