import { Link } from "@tanstack/react-router";
import {
  Home,
  Radio,
  Film,
  Grid3x3,
  Timer,
  BarChart3,
  GitBranch,
  Sparkles,
  Zap,
} from "lucide-react";

const NAV: Array<{
  group: string;
  items: Array<{ to: string; label: string; icon: React.ElementType; badge?: string }>;
}> = [
  {
    group: "Perception Studio",
    items: [
      { to: "/broadcast", label: "Video Studio", icon: Radio, badge: "AI Studio" },
    ],
  },
  {
    group: "Tactical Intelligence",
    items: [
      { to: "/match-explorer", label: "Match Explorer", icon: Film },
      { to: "/game-state", label: "Game State Radar", icon: Grid3x3 },
      { to: "/possession", label: "Possession & Events", icon: Timer },
      { to: "/tactical", label: "Tactical Advantage (TAS)", icon: BarChart3 },
      { to: "/counterfactual", label: "Counterfactual Decisions", icon: GitBranch },
    ],
  },
];

export function Sidebar() {
  return (
    <aside className="flex w-64 shrink-0 flex-col border-r border-border bg-panel-alt">
      {/* Brand Header */}
      <div className="border-b border-border px-4 py-3.5 flex items-center justify-between">
        <Link to="/" className="group flex items-center gap-2.5 cursor-pointer" title="Go to Product Landing Page">
          <div className="h-7 w-7 rounded-md bg-emerald-500/20 border border-emerald-500/40 flex items-center justify-center text-emerald-400 group-hover:border-emerald-300 transition-colors">
            <Zap className="h-4 w-4" />
          </div>
          <div>
            <div className="text-[12.5px] font-bold tracking-[0.06em] text-foreground group-hover:text-emerald-400 transition-colors leading-tight">
              BALL-FREE AI
            </div>
            <div className="label-xs text-[9px] text-muted-foreground">Tactical Vision Suite</div>
          </div>
        </Link>
        <Link
          to="/"
          className="text-muted-foreground hover:text-foreground text-[11px] flex items-center gap-1 border border-border px-2 py-0.5 rounded hover:bg-muted/40 transition-colors"
          title="Return to Product Landing Page"
        >
          <Home className="h-3 w-3" />
          <span>Home</span>
        </Link>
      </div>

      {/* Navigation Sections */}
      <nav className="flex-1 overflow-y-auto px-2 py-3 space-y-4">
        {NAV.map((section) => (
          <div key={section.group}>
            <div className="label-xs px-2.5 pb-1 text-[9.5px] font-semibold text-muted-foreground/80">
              {section.group}
            </div>
            <div className="space-y-0.5">
              {section.items.map((item) => (
                <Link
                  key={item.to}
                  to={item.to}
                  activeProps={{
                    className: "bg-accent text-foreground border-l-2 border-l-primary pl-[10px] font-medium",
                  }}
                  inactiveProps={{ className: "text-muted-foreground hover:bg-muted/60" }}
                  className="flex items-center justify-between rounded-sm px-3 py-1.5 text-[12px] transition-colors"
                >
                  <div className="flex items-center gap-2 truncate">
                    <item.icon className="h-3.5 w-3.5 shrink-0 text-muted-foreground" strokeWidth={1.8} />
                    <span className="truncate">{item.label}</span>
                  </div>
                  {item.badge && (
                    <span className="bg-emerald-500/15 text-emerald-400 text-[9px] font-bold px-1.5 py-0.2 rounded border border-emerald-500/30">
                      {item.badge}
                    </span>
                  )}
                </Link>
              ))}
            </div>
          </div>
        ))}
      </nav>

      {/* Product Footer */}
      <div className="border-t border-border px-3 py-2.5 bg-panel/40">
        <div className="flex items-center justify-between text-[11px] text-muted-foreground">
          <span className="flex items-center gap-1.5">
            <Sparkles className="h-3 w-3 text-emerald-400" /> Tactical Suite Pro
          </span>
          <span className="font-mono text-[10px] text-emerald-400/90 font-medium">v1.0 Ready</span>
        </div>
      </div>
    </aside>
  );
}
