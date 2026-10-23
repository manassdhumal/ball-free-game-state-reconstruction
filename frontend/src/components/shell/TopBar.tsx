import { Link } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { useSession } from "@/lib/session";
import { useBackendHealth } from "@/hooks/useGameState";
import { getGsrStatus } from "@/api/gsr";

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
  const { frame, dataset, fps, matchId } = useSession();
  const health = useBackendHealth();
  const gsrQuery = useQuery({ queryKey: ["gsr-status"], queryFn: getGsrStatus, staleTime: 30_000 });
  const isVideoMatch = matchId.startsWith("uploaded_") || matchId.startsWith("video_");

  const seconds = frame / fps;
  const stamp = `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${(seconds % 60)
    .toFixed(1)
    .padStart(4, "0")}`;

  const gsrValue = isVideoMatch
    ? "ACTIVE (VIDEO)"
    : gsrQuery.isLoading
      ? "CHECKING"
      : gsrQuery.isSuccess && gsrQuery.data?.status === "ready"
        ? "ONLINE"
        : "UNAVAILABLE";

  const gsrTone: "ok" | "warn" | "blocked" =
    isVideoMatch || (gsrQuery.isSuccess && gsrQuery.data?.status === "ready")
      ? "ok"
      : gsrQuery.isLoading
        ? "warn"
        : "blocked";

  return (
    <header className="flex h-12 shrink-0 items-stretch border-b border-border bg-panel">
      <Field label="Dataset" value={dataset} />
      <Field
        label="Mode"
        value={isVideoMatch ? "Video Perception Reconstruction" : "Metrica Benchmark Tracking"}
      />
      <Field label="Frame" value={`${frame} · ${stamp} · ${fps} FPS`} />
      <Field
        label="Backend"
        value={health.isLoading ? "CHECKING" : health.isSuccess ? "CONNECTED" : "UNAVAILABLE"}
        tone={health.isSuccess ? "ok" : health.isLoading ? "warn" : "blocked"}
      />
      <Link to="/broadcast" className="flex items-stretch hover:bg-muted/40 transition-colors" title="Open Video Game State Reconstruction Studio">
        <Field label="GSR Studio" value={gsrValue} tone={gsrTone} />
      </Link>
      {isVideoMatch && (
        <Link to="/match-explorer" className="flex items-stretch hover:bg-muted/40 transition-colors" title="Open Match Explorer for this video">
          <Field label="Explorer" value="INSPECT" tone="ok" />
        </Link>
      )}
      <div className="flex flex-1 items-center justify-end px-4">
        <span className="label-xs flex items-center gap-1.5 font-medium text-emerald-400">
          <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse" />
          Tactical Engine Active
        </span>
      </div>
    </header>
  );
}
