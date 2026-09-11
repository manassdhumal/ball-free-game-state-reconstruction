import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

export function Panel({
  title,
  subtitle,
  actions,
  children,
  className,
  bodyClassName,
}: {
  title?: string | undefined;
  subtitle?: string | undefined;
  actions?: ReactNode | undefined;
  children: ReactNode;
  className?: string | undefined;
  bodyClassName?: string | undefined;
}) {
  return (
    <section className={cn("border border-border bg-panel", className)}>
      {(title || actions) && (
        <header className="flex items-center justify-between gap-3 border-b border-border px-3 py-2">
          <div>
            <h2 className="text-[12px] font-semibold tracking-[0.06em] uppercase">{title}</h2>
            {subtitle && <p className="mt-0.5 text-[11px] text-muted-foreground">{subtitle}</p>}
          </div>
          {actions}
        </header>
      )}
      <div className={cn("p-3", bodyClassName)}>{children}</div>
    </section>
  );
}

export function Metric({
  label,
  value,
  sub,
  tone,
}: {
  label: string;
  value: string;
  sub?: string | undefined;
  tone?: "primary" | "warn" | "muted" | undefined;
}) {
  return (
    <div className="border border-border bg-panel px-3 py-2.5">
      <div className="label-xs">{label}</div>
      <div
        className={cn(
          "num mt-1 text-[22px] font-medium leading-none",
          tone === "primary" && "text-primary",
          tone === "warn" && "text-warn",
          tone === "muted" && "text-muted-foreground",
        )}
      >
        {value}
      </div>
      {sub && <div className="mt-1.5 text-[11px] text-muted-foreground">{sub}</div>}
    </div>
  );
}

export function PageHeader({ title, description }: { title: string; description: string }) {
  return (
    <div className="mb-4">
      <h1 className="text-[15px] font-semibold tracking-[0.04em] uppercase">{title}</h1>
      <p className="mt-1 max-w-3xl text-[12px] leading-relaxed text-muted-foreground">
        {description}
      </p>
    </div>
  );
}

export function TypeTag({ type }: { type: string }) {
  const map: Record<string, string> = {
    "Model-Derived": "border-primary/40 text-primary",
    Heuristic: "border-warn/40 text-warn",
    "Measured Proxy": "border-border text-muted-foreground",
  };
  return (
    <span
      className={cn(
        "inline-block border px-1.5 py-px text-[10px] tracking-[0.05em] uppercase",
        map[type] ?? "border-border text-muted-foreground",
      )}
    >
      {type}
    </span>
  );
}
