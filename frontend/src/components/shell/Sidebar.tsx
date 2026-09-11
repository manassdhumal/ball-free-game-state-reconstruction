import { Link } from "@tanstack/react-router";
import {
  LayoutDashboard,
  Film,
  Grid3x3,
  Timer,
  BarChart3,
  GitBranch,
  ShieldCheck,
  Gauge,
  Waves,
  Activity,
  Wrench,
  FlaskConical,
  FileText,
  Radio,
  AlertTriangle,
  Settings,
  BookOpen,
} from "lucide-react";

const NAV: Array<{
  group: string;
  items: Array<{ to: string; label: string; icon: React.ElementType }>;
}> = [
  {
    group: "Analysis",
    items: [
      { to: "/", label: "Overview", icon: LayoutDashboard },
      { to: "/match-explorer", label: "Match Explorer", icon: Film },
      { to: "/game-state", label: "Game State", icon: Grid3x3 },
      { to: "/possession", label: "Possession & Events", icon: Timer },
      { to: "/tactical", label: "Tactical Analysis", icon: BarChart3 },
      { to: "/counterfactual", label: "Counterfactual", icon: GitBranch },
    ],
  },
  {
    group: "Robustness",
    items: [
      { to: "/robustness", label: "Decision Robustness", icon: ShieldCheck },
      { to: "/benchmark", label: "Robustness Benchmark", icon: Gauge },
      { to: "/noise-lab", label: "Noise Lab", icon: Waves },
      { to: "/tracking-quality", label: "Tracking Quality", icon: Activity },
      { to: "/recovery", label: "Recovery & Mitigation", icon: Wrench },
    ],
  },
  {
    group: "Research",
    items: [
      { to: "/experiments", label: "Experiments", icon: FlaskConical },
      { to: "/results", label: "Research Results", icon: FileText },
      { to: "/broadcast", label: "Broadcast / GSR", icon: Radio },
      { to: "/limitations", label: "Limitations", icon: AlertTriangle },
    ],
  },
];

export function Sidebar() {
  return (
    <aside className="flex w-60 shrink-0 flex-col border-r border-border bg-panel-alt">
      <div className="border-b border-border px-4 py-4">
        <div className="text-[11px] font-semibold leading-tight tracking-[0.08em] text-foreground">
          BALL-FREE GAME STATE
          <br />
          RECONSTRUCTION
        </div>
        <div className="label-xs mt-1.5">Player-only football intelligence</div>
      </div>

      <nav className="flex-1 overflow-y-auto px-2 py-3">
        {NAV.map((section) => (
          <div key={section.group} className="mb-4">
            <div className="label-xs px-2 pb-1.5">{section.group}</div>
            {section.items.map((item) => (
              <Link
                key={item.to}
                to={item.to}
                activeOptions={{ exact: item.to === "/" }}
                activeProps={{
                  className: "bg-accent text-foreground border-l-2 border-l-primary pl-[10px]",
                }}
                inactiveProps={{ className: "text-muted-foreground hover:bg-muted/60" }}
                className="mb-0.5 flex items-center gap-2 rounded-sm px-3 py-1.5 text-[12.5px] transition-colors"
              >
                <item.icon className="h-3.5 w-3.5 shrink-0" strokeWidth={1.75} />
                <span className="truncate">{item.label}</span>
              </Link>
            ))}
          </div>
        ))}
      </nav>

      <div className="border-t border-border px-3 py-3">
        <div className="flex items-center gap-4 text-[11px] text-muted-foreground">
          <span className="flex items-center gap-1.5">
            <Settings className="h-3.5 w-3.5" strokeWidth={1.75} /> Settings
          </span>
          <span className="flex items-center gap-1.5">
            <BookOpen className="h-3.5 w-3.5" strokeWidth={1.75} /> Docs
          </span>
        </div>
        <div className="label-xs mt-2">v0.8 Research Build</div>
      </div>
    </aside>
  );
}
