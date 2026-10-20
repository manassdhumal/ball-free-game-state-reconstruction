import { useState } from "react";
import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import {
  Radio,
  Film,
  Grid3x3,
  Timer,
  BarChart3,
  GitBranch,
  Play,
  ArrowRight,
  ShieldCheck,
  Zap,
  Sparkles,
  Layers,
  Cpu,
  Eye,
  Sliders,
  CheckCircle2,
  FileText,
  Activity,
  Maximize2,
  ChevronRight,
} from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { Pitch, PitchLegend } from "@/components/Pitch";
import { useLivePlayers, useBackendHealth } from "@/hooks/useGameState";
import { getGsrStatus } from "@/api/gsr";
import { useSession } from "@/lib/session";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Ball-Free AI — Tactical Football Intelligence Platform" },
      {
        name: "description",
        content:
          "Enterprise football game state reconstruction platform: extract player tracking, spatial control, and Tactical Advantage Scores directly from broadcast video without tracking the ball.",
      },
      { property: "og:title", content: "Ball-Free AI — Tactical Football Intelligence Platform" },
      {
        property: "og:description",
        content:
          "Transform football match video footage into live 2D tactical pitch radar, possession inference, and counterfactual passing decisions.",
      },
    ],
  }),
  component: ProductLandingPage,
});

export function ProductLandingPage() {
  const navigate = useNavigate();
  const { matchId, frame } = useSession();
  const health = useBackendHealth();
  const gsrQuery = useQuery({ queryKey: ["gsr-status"], queryFn: getGsrStatus, staleTime: 30_000 });
  const liveState = useLivePlayers(matchId, frame);
  const players = liveState.data ?? [];

  const [activeTab, setActiveTab] = useState<"radar" | "pipeline" | "benchmarks">("radar");

  return (
    <div className="min-h-screen bg-background text-foreground selection:bg-emerald-500/30 selection:text-emerald-300">
      {/* ------------------------------------------------------------------- */}
      {/* Top Product Navigation Bar                                          */}
      {/* ------------------------------------------------------------------- */}
      <header className="sticky top-0 z-50 border-b border-border/80 bg-background/85 backdrop-blur-md">
        <div className="mx-auto flex h-14 max-w-7xl items-center justify-between px-4 sm:px-6 lg:px-8">
          {/* Brand Logo */}
          <Link to="/" className="flex items-center gap-2.5 group">
            <div className="flex h-8 w-8 items-center justify-center rounded-md bg-emerald-500/15 border border-emerald-500/40 text-emerald-400 shadow-sm group-hover:border-emerald-400 transition-colors">
              <Zap className="h-4 w-4" />
            </div>
            <div>
              <span className="text-sm font-bold tracking-wider text-foreground">BALL-FREE AI</span>
              <span className="ml-2 rounded border border-emerald-500/40 bg-emerald-500/10 px-1.5 py-0.5 text-[9px] font-bold uppercase tracking-widest text-emerald-400">
                Studio Pro
              </span>
            </div>
          </Link>

          {/* Center Links */}
          <nav className="hidden md:flex items-center gap-6 text-[13px] font-medium text-muted-foreground">
            <a href="#features" className="hover:text-foreground transition-colors">
              Platform Outputs
            </a>
            <a href="#journey" className="hover:text-foreground transition-colors">
              How It Works
            </a>
            <Link to="/broadcast" className="hover:text-emerald-400 transition-colors flex items-center gap-1">
              <span>Video Studio</span>
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse" />
            </Link>
            <Link to="/match-explorer" className="hover:text-foreground transition-colors">
              Tactical Radar
            </Link>
            <a href="#research" className="hover:text-foreground transition-colors">
              Research & Benchmarks
            </a>
          </nav>

          {/* Right Status & Launch Button */}
          <div className="flex items-center gap-3">
            <div className="hidden sm:flex items-center gap-2 border border-border/60 bg-panel-alt/60 px-2.5 py-1 rounded text-[11px] font-mono">
              <span className="flex items-center gap-1.5 text-muted-foreground">
                <span
                  className={`h-2 w-2 rounded-full ${
                    health.isSuccess ? "bg-emerald-400" : health.isLoading ? "bg-amber-400" : "bg-rose-400"
                  }`}
                />
                Backend {health.isSuccess ? "Connected" : "Offline"}
              </span>
              <span className="text-border">|</span>
              <span className="text-emerald-400 font-semibold">GSR Ready</span>
            </div>

            <Link
              to="/broadcast"
              className="flex items-center gap-1.5 rounded-md bg-emerald-600 hover:bg-emerald-500 px-3.5 py-1.5 text-xs font-semibold text-white shadow-lg shadow-emerald-900/30 transition-all hover:shadow-emerald-900/50"
            >
              <span>Launch Studio</span>
              <ArrowRight className="h-3.5 w-3.5" />
            </Link>
          </div>
        </div>
      </header>

      {/* ------------------------------------------------------------------- */}
      {/* Hero Section                                                        */}
      {/* ------------------------------------------------------------------- */}
      <section className="relative overflow-hidden pt-12 pb-16 lg:pt-20 lg:pb-24 border-b border-border/60">
        {/* Glow Gradients */}
        <div className="absolute top-1/4 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[700px] h-[350px] bg-emerald-500/10 blur-[120px] pointer-events-none rounded-full" />
        <div className="absolute top-1/3 left-1/3 -translate-x-1/2 w-[350px] h-[250px] bg-teal-500/10 blur-[90px] pointer-events-none rounded-full" />

        <div className="relative mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 text-center">
          {/* Eyebrow Pill */}
          <div className="inline-flex items-center gap-2 rounded-full border border-emerald-500/40 bg-emerald-500/10 px-3.5 py-1 text-xs font-medium text-emerald-300 mb-6">
            <Sparkles className="h-3.5 w-3.5 text-yellow-300" />
            <span>Player-Only Football Intelligence · Zero Ball Dependency</span>
          </div>

          {/* Main Headline */}
          <h1 className="text-4xl sm:text-5xl lg:text-6xl font-extrabold tracking-tight text-foreground max-w-4xl mx-auto leading-[1.12]">
            Reconstruct Real Match State{" "}
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 via-teal-300 to-cyan-400">
              Directly from Video Footage
            </span>
          </h1>

          {/* Subheading */}
          <p className="mt-5 max-w-2xl mx-auto text-sm sm:text-base text-muted-foreground leading-relaxed">
            Eliminate fragile ball tracking and expensive multi-camera sensor rigs. Our computer vision and spatial graph
            architecture extracts player positioning, team compactness, possession transitions, and tactical momentum
            solely from video silhouettes.
          </p>

          {/* CTAs */}
          <div className="mt-8 flex flex-wrap items-center justify-center gap-3">
            <Link
              to="/broadcast"
              className="flex items-center gap-2 rounded-md bg-emerald-600 hover:bg-emerald-500 px-5 py-2.5 text-sm font-semibold text-white shadow-xl shadow-emerald-950/60 transition-all hover:scale-[1.02]"
            >
              <Play className="h-4 w-4 fill-current" />
              <span>Open Video Perception Studio</span>
            </Link>

            <Link
              to="/match-explorer"
              className="flex items-center gap-2 rounded-md border border-border bg-panel hover:bg-muted/70 px-4 py-2.5 text-sm font-medium text-foreground transition-colors"
            >
              <Film className="h-4 w-4 text-emerald-400" />
              <span>Explore Tactical Radar</span>
            </Link>

            <a
              href="#research"
              className="flex items-center gap-1.5 px-3 py-2 text-xs font-medium text-muted-foreground hover:text-foreground transition-colors"
            >
              <span>View Benchmark Results</span>
              <ChevronRight className="h-3.5 w-3.5" />
            </a>
          </div>

          {/* Key Value Proof Badges */}
          <div className="mt-12 grid grid-cols-2 sm:grid-cols-4 gap-3 max-w-3xl mx-auto text-left">
            <div className="border border-border/60 bg-panel/70 p-3 rounded-lg backdrop-blur-sm">
              <div className="text-[11px] text-muted-foreground">Detection Engine</div>
              <div className="text-sm font-bold font-mono text-emerald-400 mt-0.5">YOLOv8 + ByteTrack</div>
            </div>
            <div className="border border-border/60 bg-panel/70 p-3 rounded-lg backdrop-blur-sm">
              <div className="text-[11px] text-muted-foreground">Perspective Transform</div>
              <div className="text-sm font-bold font-mono text-foreground mt-0.5">4-Point Homography</div>
            </div>
            <div className="border border-border/60 bg-panel/70 p-3 rounded-lg backdrop-blur-sm">
              <div className="text-[11px] text-muted-foreground">Ball Dependency</div>
              <div className="text-sm font-bold font-mono text-cyan-400 mt-0.5">Zero (100% Ball-Free)</div>
            </div>
            <div className="border border-border/60 bg-panel/70 p-3 rounded-lg backdrop-blur-sm">
              <div className="text-[11px] text-muted-foreground">Empirical Evaluation</div>
              <div className="text-sm font-bold font-mono text-foreground mt-0.5">180 Benchmark Runs</div>
            </div>
          </div>
        </div>
      </section>

      {/* ------------------------------------------------------------------- */}
      {/* Interactive Visual Showcase: Video Frame to 2D Tactical Pitch Radar */}
      {/* ------------------------------------------------------------------- */}
      <section className="py-14 border-b border-border/60 bg-panel-alt/30">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-4 mb-6">
            <div>
              <div className="label-xs text-emerald-400 font-semibold">Live Operational Interface</div>
              <h2 className="text-xl sm:text-2xl font-bold tracking-tight text-foreground mt-1">
                From Broadcast Camera to High-Definition 2D Tactical Radar
              </h2>
              <p className="text-xs sm:text-sm text-muted-foreground mt-0.5">
                Inspect player spatial coordinates, velocity vectors, and inferred possession in real time.
              </p>
            </div>

            <div className="flex items-center gap-2">
              <Link
                to="/game-state"
                className="flex items-center gap-1.5 border border-border bg-panel px-3 py-1.5 rounded text-xs font-medium hover:bg-muted/70 transition-colors"
              >
                <Grid3x3 className="h-3.5 w-3.5 text-emerald-400" />
                <span>Full Game State Matrix</span>
              </Link>
              <Link
                to="/broadcast"
                className="flex items-center gap-1.5 bg-emerald-600 hover:bg-emerald-500 text-white px-3 py-1.5 rounded text-xs font-semibold transition-colors"
              >
                <span>Upload Video</span>
                <ArrowRight className="h-3.5 w-3.5" />
              </Link>
            </div>
          </div>

          {/* Interactive Dual Display Card */}
          <div className="grid lg:grid-cols-12 gap-4 border border-border/80 bg-panel rounded-xl overflow-hidden p-4 shadow-2xl">
            {/* Left: Video / Perspective Simulation */}
            <div className="lg:col-span-5 flex flex-col justify-between space-y-3">
              <div>
                <div className="flex items-center justify-between pb-2 border-b border-border/60 text-xs">
                  <span className="font-semibold flex items-center gap-1.5 text-muted-foreground">
                    <Radio className="h-3.5 w-3.5 text-emerald-400 animate-pulse" />
                    Input Broadcast Stream
                  </span>
                  <span className="font-mono text-[11px] text-emerald-400 bg-emerald-500/10 px-1.5 py-0.5 rounded">
                    1280x720 • 25 FPS
                  </span>
                </div>
                <div className="mt-3 relative aspect-video bg-black rounded-lg border border-border/80 overflow-hidden flex items-center justify-center">
                  {/* Broadcast Simulation Visual */}
                  <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-black/30" />
                  <div className="absolute inset-0 flex flex-col justify-between p-3 pointer-events-none">
                    <div className="flex justify-between items-center text-[10px] font-mono text-white/80">
                      <span className="bg-black/60 px-1.5 py-0.5 rounded">CAM 01 · ELEVATED SIDELINE</span>
                      <span className="bg-emerald-600 px-1.5 py-0.5 rounded font-bold">YOLOv8 TRACKING</span>
                    </div>

                    {/* Simulated Detection Boxes Overlay */}
                    <div className="relative w-full h-full flex items-center justify-center">
                      <div className="absolute top-[28%] left-[34%] border border-blue-400 bg-blue-500/20 px-1 py-0.5 rounded text-[8px] font-mono text-blue-200">
                        #10 HOME (0.89)
                      </div>
                      <div className="absolute top-[42%] left-[48%] border border-red-400 bg-red-500/20 px-1 py-0.5 rounded text-[8px] font-mono text-red-200">
                        #19 AWAY (0.94)
                      </div>
                      <div className="absolute top-[35%] left-[62%] border border-blue-400 bg-blue-500/20 px-1 py-0.5 rounded text-[8px] font-mono text-blue-200">
                        #7 HOME (0.91)
                      </div>
                      <div className="absolute top-[55%] left-[25%] border border-red-400 bg-red-500/20 px-1 py-0.5 rounded text-[8px] font-mono text-red-200">
                        #22 AWAY (0.87)
                      </div>
                    </div>

                    <div className="flex justify-between items-end text-[10px] font-mono text-white/80">
                      <span className="bg-black/60 px-1.5 py-0.5 rounded">HOMOGRAPHY: MAIN_BROADCAST</span>
                      <span className="text-emerald-400 font-bold">22 OBJECTS DETECTED</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Ingestion Specs */}
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="border border-border/60 bg-panel-alt/80 p-2.5 rounded">
                  <div className="text-[10.5px] text-muted-foreground font-mono">JERSEY CLASSIFIER</div>
                  <div className="font-semibold text-foreground text-xs mt-0.5">K-Means Lab Color Model</div>
                  <div className="text-[10px] text-muted-foreground mt-0.5">Auto-separates kits</div>
                </div>
                <div className="border border-border/60 bg-panel-alt/80 p-2.5 rounded">
                  <div className="text-[10.5px] text-muted-foreground font-mono">SIGNAL MITIGATION</div>
                  <div className="font-semibold text-emerald-400 text-xs mt-0.5">Savitzky-Golay Filter</div>
                  <div className="text-[10px] text-muted-foreground mt-0.5">+0.178 F1 recovery gain</div>
                </div>
              </div>
            </div>

            {/* Right: Real-Time 2D Tactical Pitch Radar */}
            <div className="lg:col-span-7 flex flex-col justify-between space-y-3">
              <div>
                <div className="flex items-center justify-between pb-2 border-b border-border/60 text-xs">
                  <span className="font-semibold flex items-center gap-1.5 text-foreground">
                    <Grid3x3 className="h-3.5 w-3.5 text-emerald-400" />
                    Reconstructed 2D Pitch Radar
                  </span>
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-[11px] text-muted-foreground">Frame {frame}</span>
                    <span className="text-[10px] uppercase font-bold text-emerald-400 border border-emerald-500/40 bg-emerald-500/10 px-1.5 py-0.5 rounded">
                      Live State
                    </span>
                  </div>
                </div>

                {/* Pitch Radar */}
                <div className="mt-3 border border-border bg-turf p-1 rounded-lg overflow-hidden shadow-inner">
                  {liveState.isLoading ? (
                    <div className="flex h-[320px] items-center justify-center text-xs text-muted-foreground">
                      Loading canonical player state…
                    </div>
                  ) : players.length === 0 ? (
                    <div className="flex h-[320px] items-center justify-center text-xs text-muted-foreground">
                      No active frame coordinates loaded
                    </div>
                  ) : (
                    <Pitch players={players} height={320} showVectors={true} heat={false} />
                  )}
                </div>
                <div className="mt-2">
                  <PitchLegend />
                </div>
              </div>

              {/* Tactical Outputs Banner */}
              <div className="border border-emerald-500/30 bg-emerald-500/10 p-2.5 rounded-lg flex items-center justify-between text-xs">
                <div className="flex items-center gap-3">
                  <div>
                    <span className="text-[10px] text-muted-foreground uppercase font-mono">Inferred Possessor</span>
                    <div className="font-bold text-emerald-400 text-xs">HOME_10 (Confidence 0.88)</div>
                  </div>
                  <div className="border-l border-border/80 pl-3">
                    <span className="text-[10px] text-muted-foreground uppercase font-mono">Tactical Advantage (TAS)</span>
                    <div className="font-bold text-foreground text-xs">+0.684 (Attacking Third)</div>
                  </div>
                </div>

                <Link
                  to="/tactical"
                  className="text-[11px] font-semibold text-emerald-400 hover:text-emerald-300 flex items-center gap-1"
                >
                  <span>Deep Tactical Analysis</span>
                  <ArrowRight className="h-3 w-3" />
                </Link>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ------------------------------------------------------------------- */}
      {/* The 4-Step Perception & Intelligence Journey                        */}
      {/* ------------------------------------------------------------------- */}
      <section id="journey" className="py-16 border-b border-border/60">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="text-center max-w-2xl mx-auto mb-12">
            <div className="label-xs text-emerald-400 font-semibold">The Perception Pipeline</div>
            <h2 className="text-2xl sm:text-3xl font-bold tracking-tight text-foreground mt-1">
              How Raw Video Becomes Tactical Intelligence
            </h2>
            <p className="text-xs sm:text-sm text-muted-foreground mt-2">
              Four tightly integrated stages convert broadcast video frames into canonical 2D pitch coordinates and spatial decision models.
            </p>
          </div>

          <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {/* Step 1 */}
            <div className="border border-border bg-panel p-4 rounded-xl relative overflow-hidden group hover:border-emerald-500/50 transition-all hover:shadow-lg hover:shadow-emerald-950/20">
              <div className="text-[10px] font-mono text-emerald-400 font-bold mb-2">STAGE 01</div>
              <div className="h-9 w-9 rounded-lg bg-emerald-500/15 border border-emerald-500/30 flex items-center justify-center text-emerald-400 mb-3">
                <Eye className="h-5 w-5" />
              </div>
              <h3 className="text-sm font-bold text-foreground">Video Ingest & Detection</h3>
              <p className="text-xs text-muted-foreground mt-1.5 leading-relaxed">
                Processes video clips at 25 FPS using YOLOv8n to identify humanoid player silhouettes, referees, and field boundaries.
              </p>
              <div className="mt-3 text-[10px] font-mono text-muted-foreground bg-panel-alt px-2 py-1 rounded border border-border">
                YOLOv8n · Ultralytics
              </div>
            </div>

            {/* Step 2 */}
            <div className="border border-border bg-panel p-4 rounded-xl relative overflow-hidden group hover:border-emerald-500/50 transition-all hover:shadow-lg hover:shadow-emerald-950/20">
              <div className="text-[10px] font-mono text-emerald-400 font-bold mb-2">STAGE 02</div>
              <div className="h-9 w-9 rounded-lg bg-teal-500/15 border border-teal-500/30 flex items-center justify-center text-teal-400 mb-3">
                <Layers className="h-5 w-5" />
              </div>
              <h3 className="text-sm font-bold text-foreground">ByteTrack & Kit Clustering</h3>
              <p className="text-xs text-muted-foreground mt-1.5 leading-relaxed">
                ByteTrack maintains track continuity across occlusions while K-Means clustering on torso pixels categorizes Home vs Away squads.
              </p>
              <div className="mt-3 text-[10px] font-mono text-muted-foreground bg-panel-alt px-2 py-1 rounded border border-border">
                ByteTrack · CIELAB K-Means
              </div>
            </div>

            {/* Step 3 */}
            <div className="border border-border bg-panel p-4 rounded-xl relative overflow-hidden group hover:border-emerald-500/50 transition-all hover:shadow-lg hover:shadow-emerald-950/20">
              <div className="text-[10px] font-mono text-emerald-400 font-bold mb-2">STAGE 03</div>
              <div className="h-9 w-9 rounded-lg bg-blue-500/15 border border-blue-500/30 flex items-center justify-center text-blue-400 mb-3">
                <Sliders className="h-5 w-5" />
              </div>
              <h3 className="text-sm font-bold text-foreground">4-Point Pitch Homography</h3>
              <p className="text-xs text-muted-foreground mt-1.5 leading-relaxed">
                Perspective transformation maps camera pixels to canonical 105m × 68m coordinates, with interactive corner-flag calibration.
              </p>
              <div className="mt-3 text-[10px] font-mono text-muted-foreground bg-panel-alt px-2 py-1 rounded border border-border">
                OpenCV Homography · Presets
              </div>
            </div>

            {/* Step 4 */}
            <div className="border border-border bg-panel p-4 rounded-xl relative overflow-hidden group hover:border-emerald-500/50 transition-all hover:shadow-lg hover:shadow-emerald-950/20">
              <div className="text-[10px] font-mono text-emerald-400 font-bold mb-2">STAGE 04</div>
              <div className="h-9 w-9 rounded-lg bg-purple-500/15 border border-purple-500/30 flex items-center justify-center text-purple-400 mb-3">
                <Cpu className="h-5 w-5" />
              </div>
              <h3 className="text-sm font-bold text-foreground">Spatial AI & Tactical Score</h3>
              <p className="text-xs text-muted-foreground mt-1.5 leading-relaxed">
                Voronoi spatial graphs calculate defensive pressure, team compactness, ball-free possession, and Tactical Advantage Scores.
              </p>
              <div className="mt-3 text-[10px] font-mono text-muted-foreground bg-panel-alt px-2 py-1 rounded border border-border">
                Delaunay Graph · TAS Engine
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ------------------------------------------------------------------- */}
      {/* Core Platform Modules & Desired Outputs                             */}
      {/* ------------------------------------------------------------------- */}
      <section id="features" className="py-16 border-b border-border/60 bg-panel-alt/20">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="text-center max-w-2xl mx-auto mb-12">
            <div className="label-xs text-emerald-400 font-semibold">Product Capabilities</div>
            <h2 className="text-2xl sm:text-3xl font-bold tracking-tight text-foreground mt-1">
              Engineered for Match Analysts & AI Vision Teams
            </h2>
            <p className="text-xs sm:text-sm text-muted-foreground mt-2">
              Explore the four primary output studios designed for video ingestion, radar scrubbing, and tactical simulation.
            </p>
          </div>

          <div className="grid md:grid-cols-2 gap-4">
            {/* Feature 1: Video Studio */}
            <div className="border border-border bg-panel p-5 rounded-xl hover:border-emerald-500/50 transition-all group flex flex-col justify-between">
              <div>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <div className="h-8 w-8 rounded bg-emerald-500/15 border border-emerald-500/30 flex items-center justify-center text-emerald-400">
                      <Radio className="h-4 w-4" />
                    </div>
                    <h3 className="text-base font-bold text-foreground">Video Perception Studio</h3>
                  </div>
                  <span className="text-[10px] font-mono bg-emerald-500/20 text-emerald-400 font-bold px-2 py-0.5 rounded border border-emerald-500/40">
                    LIVE RECONSTRUCTION
                  </span>
                </div>
                <p className="text-xs text-muted-foreground mt-3 leading-relaxed">
                  Drag and drop actual match gameplay video files (`.mp4`, `.mov`, `.avi`). Calibrate camera perspectives with 4 interactive corner points, run the YOLOv8 + ByteTrack pipeline, and inspect synchronized dual-view video playback alongside live 2D radar.
                </p>
                <div className="mt-4 flex flex-wrap gap-2 text-[10.5px] font-mono text-muted-foreground">
                  <span className="bg-panel-alt px-2 py-1 rounded border border-border">Dual-View Playback</span>
                  <span className="bg-panel-alt px-2 py-1 rounded border border-border">4-Point Calibration</span>
                  <span className="bg-panel-alt px-2 py-1 rounded border border-border">CLI Video Exporter</span>
                </div>
              </div>

              <div className="mt-6 pt-3 border-t border-border flex items-center justify-between">
                <span className="text-[11px] text-muted-foreground">Ready to test with bundled demo clip</span>
                <Link
                  to="/broadcast"
                  className="flex items-center gap-1.5 text-xs font-semibold text-emerald-400 hover:text-emerald-300"
                >
                  <span>Open Video Studio</span>
                  <ArrowRight className="h-3.5 w-3.5" />
                </Link>
              </div>
            </div>

            {/* Feature 2: Match Explorer */}
            <div className="border border-border bg-panel p-5 rounded-xl hover:border-emerald-500/50 transition-all group flex flex-col justify-between">
              <div>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <div className="h-8 w-8 rounded bg-blue-500/15 border border-blue-500/30 flex items-center justify-center text-blue-400">
                      <Film className="h-4 w-4" />
                    </div>
                    <h3 className="text-base font-bold text-foreground">Interactive Match Explorer</h3>
                  </div>
                  <span className="text-[10px] font-mono bg-blue-500/20 text-blue-400 font-bold px-2 py-0.5 rounded border border-blue-500/40">
                    FRAME SCRUBBER
                  </span>
                </div>
                <p className="text-xs text-muted-foreground mt-3 leading-relaxed">
                  Scrub continuous tracking frames at 0.25x to 2.0x playback speeds. Select any player token to inspect instantaneous speed, orientation, distance to nearest defenders, and local Voronoi density.
                </p>
                <div className="mt-4 flex flex-wrap gap-2 text-[10.5px] font-mono text-muted-foreground">
                  <span className="bg-panel-alt px-2 py-1 rounded border border-border">Dynamic Velocity Vectors</span>
                  <span className="bg-panel-alt px-2 py-1 rounded border border-border">Player Inspector</span>
                  <span className="bg-panel-alt px-2 py-1 rounded border border-border">Variable Speed</span>
                </div>
              </div>

              <div className="mt-6 pt-3 border-t border-border flex items-center justify-between">
                <span className="text-[11px] text-muted-foreground">Compatible with Metrica & Uploaded Video</span>
                <Link
                  to="/match-explorer"
                  className="flex items-center gap-1.5 text-xs font-semibold text-blue-400 hover:text-blue-300"
                >
                  <span>Open Match Explorer</span>
                  <ArrowRight className="h-3.5 w-3.5" />
                </Link>
              </div>
            </div>

            {/* Feature 3: Tactical Advantage (TAS) */}
            <div className="border border-border bg-panel p-5 rounded-xl hover:border-emerald-500/50 transition-all group flex flex-col justify-between">
              <div>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <div className="h-8 w-8 rounded bg-teal-500/15 border border-teal-500/30 flex items-center justify-center text-teal-400">
                      <BarChart3 className="h-4 w-4" />
                    </div>
                    <h3 className="text-base font-bold text-foreground">Tactical Advantage Engine (TAS)</h3>
                  </div>
                  <span className="text-[10px] font-mono bg-teal-500/20 text-teal-400 font-bold px-2 py-0.5 rounded border border-teal-500/40">
                    TERRITORIAL AI
                  </span>
                </div>
                <p className="text-xs text-muted-foreground mt-3 leading-relaxed">
                  Quantifies attacking momentum, forward passing corridors, and defensive compactness in real time. Generates continuous advantage signals without requiring annotated pass tags or ball detection.
                </p>
                <div className="mt-4 flex flex-wrap gap-2 text-[10.5px] font-mono text-muted-foreground">
                  <span className="bg-panel-alt px-2 py-1 rounded border border-border">Attacking Third Progress</span>
                  <span className="bg-panel-alt px-2 py-1 rounded border border-border">Defensive Compactness</span>
                  <span className="bg-panel-alt px-2 py-1 rounded border border-border">Continuous TAS Curve</span>
                </div>
              </div>

              <div className="mt-6 pt-3 border-t border-border flex items-center justify-between">
                <span className="text-[11px] text-muted-foreground">Interpretable tactical scoring model</span>
                <Link
                  to="/tactical"
                  className="flex items-center gap-1.5 text-xs font-semibold text-teal-400 hover:text-teal-300"
                >
                  <span>View Tactical Engine</span>
                  <ArrowRight className="h-3.5 w-3.5" />
                </Link>
              </div>
            </div>

            {/* Feature 4: Counterfactual Passing */}
            <div className="border border-border bg-panel p-5 rounded-xl hover:border-emerald-500/50 transition-all group flex flex-col justify-between">
              <div>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <div className="h-8 w-8 rounded bg-purple-500/15 border border-purple-500/30 flex items-center justify-center text-purple-400">
                      <GitBranch className="h-4 w-4" />
                    </div>
                    <h3 className="text-base font-bold text-foreground">Counterfactual Decision Analysis</h3>
                  </div>
                  <span className="text-[10px] font-mono bg-purple-500/20 text-purple-400 font-bold px-2 py-0.5 rounded border border-purple-500/40">
                    PASS RANKING
                  </span>
                </div>
                <p className="text-xs text-muted-foreground mt-3 leading-relaxed">
                  Evaluates alternative passing options available to the ball possessor at any frozen instant. Ranks teammate options by passing lane openness, receiver support, and territorial progression gain.
                </p>
                <div className="mt-4 flex flex-wrap gap-2 text-[10.5px] font-mono text-muted-foreground">
                  <span className="bg-panel-alt px-2 py-1 rounded border border-border">Top-3 Ranked Options</span>
                  <span className="bg-panel-alt px-2 py-1 rounded border border-border">Defensive Lane Pressure</span>
                  <span className="bg-panel-alt px-2 py-1 rounded border border-border">Hypothetical Simulation</span>
                </div>
              </div>

              <div className="mt-6 pt-3 border-t border-border flex items-center justify-between">
                <span className="text-[11px] text-muted-foreground">What-if tactical decision modeling</span>
                <Link
                  to="/counterfactual"
                  className="flex items-center gap-1.5 text-xs font-semibold text-purple-400 hover:text-purple-300"
                >
                  <span>Explore Counterfactuals</span>
                  <ArrowRight className="h-3.5 w-3.5" />
                </Link>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ------------------------------------------------------------------- */}
      {/* Research Synthesis & Three-Tier Robustness Benchmark                */}
      {/* ------------------------------------------------------------------- */}
      <section id="research" className="py-16 border-b border-border/60">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="flex flex-col md:flex-row md:items-end justify-between gap-4 mb-8">
            <div>
              <div className="label-xs text-emerald-400 font-semibold">Empirical Foundation</div>
              <h2 className="text-2xl sm:text-3xl font-bold tracking-tight text-foreground mt-1">
                Backed by 60 Controlled Research Experiments
              </h2>
              <p className="text-xs sm:text-sm text-muted-foreground mt-1 max-w-xl">
                The product's robustness is certified across 180 condition observations under synthetic noise degradation, Savitzky-Golay filtering, and live broadcast perception.
              </p>
            </div>

            <Link
              to="/results"
              className="flex items-center gap-1.5 border border-border bg-panel px-3.5 py-2 rounded text-xs font-semibold hover:bg-muted/70 transition-colors self-start md:self-auto"
            >
              <FileText className="h-3.5 w-3.5 text-emerald-400" />
              <span>Full Research Synthesis & CSVs</span>
              <ArrowRight className="h-3.5 w-3.5" />
            </Link>
          </div>

          {/* Research Metric Cards */}
          <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-3">
            <div className="border border-border bg-panel p-4 rounded-xl">
              <span className="label-xs">Clean Baseline F1</span>
              <div className="text-2xl font-bold font-mono text-foreground mt-1">0.49199</div>
              <p className="text-[11px] text-muted-foreground mt-1">
                Ball-free possession inference reference with zero coordinate leakage.
              </p>
            </div>

            <div className="border border-border bg-panel p-4 rounded-xl">
              <span className="label-xs">Synthetic Degraded F1</span>
              <div className="text-2xl font-bold font-mono text-rose-400 mt-1">0.23451</div>
              <p className="text-[11px] text-muted-foreground mt-1">
                Dropouts, identity switches, and coordinate jitter simulate harsh broadcast cameras.
              </p>
            </div>

            <div className="border border-border bg-panel p-4 rounded-xl">
              <span className="label-xs">Mitigated Recovery F1</span>
              <div className="text-2xl font-bold font-mono text-emerald-400 mt-1">0.41328</div>
              <p className="text-[11px] text-muted-foreground mt-1">
                Bounded interpolation + Savitzky-Golay restores +0.178 F1 performance.
              </p>
            </div>

            <div className="border border-border bg-panel p-4 rounded-xl">
              <span className="label-xs">Three-Tier Architecture</span>
              <div className="text-2xl font-bold font-mono text-cyan-400 mt-1">Tier 1 • 2 • 3</div>
              <p className="text-[11px] text-muted-foreground mt-1">
                Evaluates Clean Ground Truth, Synthetic Degradation, and Video Perception.
              </p>
            </div>
          </div>

          {/* Research Summary Quote */}
          <div className="mt-6 border border-border/80 bg-panel-alt/50 p-4 rounded-xl text-xs text-muted-foreground leading-relaxed flex items-start gap-3">
            <ShieldCheck className="h-5 w-5 text-emerald-400 shrink-0 mt-0.5" />
            <div>
              <strong className="text-foreground font-semibold">Strict Anti-Leakage Protocol: </strong>
              All possession and tactical advantage metrics are evaluated strictly post-hoc. Ball coordinates are never consumed during feature extraction or model inference, ensuring verified generalization to unannotated broadcast cameras.
            </div>
          </div>
        </div>
      </section>

      {/* ------------------------------------------------------------------- */}
      {/* Bottom CTA Banner                                                   */}
      {/* ------------------------------------------------------------------- */}
      <section className="py-16">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="border border-emerald-500/40 bg-gradient-to-r from-emerald-950/40 via-panel to-teal-950/40 p-8 sm:p-12 rounded-2xl text-center relative overflow-hidden shadow-2xl">
            <div className="relative z-10 max-w-2xl mx-auto">
              <div className="inline-flex items-center gap-1.5 rounded-full bg-emerald-500/15 border border-emerald-500/30 px-3 py-1 text-[11px] font-bold text-emerald-400 uppercase tracking-widest mb-4">
                <Sparkles className="h-3 w-3" /> Ready for Production Footage
              </div>
              <h2 className="text-2xl sm:text-3xl font-extrabold text-foreground tracking-tight">
                Upload Your Match Footage & Reconstruct Game State
              </h2>
              <p className="text-xs sm:text-sm text-muted-foreground mt-2 leading-relaxed">
                Connect your broadcast clip to calibrate pitch coordinates, extract player tracking, and observe live spatial game states in seconds.
              </p>

              <div className="mt-6 flex flex-wrap items-center justify-center gap-3">
                <Link
                  to="/broadcast"
                  className="flex items-center gap-2 rounded-md bg-emerald-600 hover:bg-emerald-500 px-5 py-2.5 text-xs sm:text-sm font-semibold text-white shadow-xl shadow-emerald-950/70 transition-all hover:scale-105"
                >
                  <Play className="h-4 w-4 fill-current" />
                  <span>Launch Video Perception Studio</span>
                </Link>

                <Link
                  to="/match-explorer"
                  className="flex items-center gap-2 rounded-md border border-border bg-panel hover:bg-muted/70 px-4 py-2.5 text-xs sm:text-sm font-medium text-foreground transition-colors"
                >
                  <span>Explore Reconstructed Radar</span>
                </Link>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ------------------------------------------------------------------- */}
      {/* Footer                                                              */}
      {/* ------------------------------------------------------------------- */}
      <footer className="border-t border-border/80 bg-panel-alt/40 py-8 text-xs text-muted-foreground">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <div className="h-5 w-5 rounded bg-emerald-500/20 border border-emerald-500/40 flex items-center justify-center text-emerald-400">
              <Zap className="h-3 w-3" />
            </div>
            <span className="font-bold text-foreground tracking-wider">BALL-FREE TACTICAL AI</span>
            <span>· Enterprise Football Vision Platform</span>
          </div>

          <div className="flex items-center gap-5">
            <Link to="/broadcast" className="hover:text-emerald-400 transition-colors">
              Video Studio
            </Link>
            <Link to="/match-explorer" className="hover:text-foreground transition-colors">
              Match Explorer
            </Link>
            <Link to="/tactical" className="hover:text-foreground transition-colors">
              Tactical Advantage
            </Link>
            <Link to="/results" className="hover:text-foreground transition-colors">
              Research Data
            </Link>
          </div>
        </div>
      </footer>
    </div>
  );
}
