import { useSession } from "@/lib/session";
import { useBackendHealth } from "@/hooks/useGameState";

function Field({
  label,
  value,
  tone,
}: {
  label: string;
  value: string;
  tone?: "ok" | "warn" | "blocked";
}) {
  const color =
    tone === "ok"
      ? "text-primary"
      : tone === "warn"
        ? "text-warn"
        : tone === "blocked"
          ? "text-destructive"
          : "text-foreground";
  return (
    <div className="flex flex-col justify-center border-r border-border px-4 py-1.5">
      <span className="label-xs">{label}</span>
      <span className={`num text-[12.5px] ${color}`}>{value}</span>
    </div>
  );
}

export function TopBar() {
  const { frame, dataset, fps } = useSession();
  const health = useBackendHealth();
  const seconds = frame / fps;
  const stamp = `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${(seconds % 60)
    .toFixed(1)
    .padStart(4, "0")}`;

  return (
    <header className="flex h-12 shrink-0 items-stretch border-b border-border bg-panel">
      <Field label="Dataset" value={dataset} />
      <Field label="Experiment" value="Controlled Analysis · EXP_080A_042" />
      <Field label="Frame" value={`${frame} · ${stamp} · ${fps} FPS`} />
      <Field
        label="Backend"
        value={health.isLoading ? "CHECKING" : health.isSuccess ? "CONNECTED" : "UNAVAILABLE"}
        tone={health.isSuccess ? "ok" : health.isLoading ? "warn" : "blocked"}
      />
      <Field label="GSR" value="INTEGRATION PENDING" tone="blocked" />
      <div className="flex flex-1 items-center justify-end px-4">
        <span className="label-xs">Read-only research session</span>
      </div>
    </header>
  );
}
