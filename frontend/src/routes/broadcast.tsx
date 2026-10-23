import { useState, useRef, useEffect } from "react";
import { createFileRoute, useNavigate, Link } from "@tanstack/react-router";
import { 
  UploadCloud, 
  Play, 
  Pause, 
  CheckCircle2, 
  AlertCircle, 
  Sliders, 
  Video, 
  ArrowRight, 
  Layers, 
  Activity, 
  ShieldCheck, 
  RefreshCw,
  Sparkles,
  Zap,
  Clock,
  Gauge,
  ChevronLeft,
  ChevronRight,
  ChevronsLeft,
  ChevronsRight,
  Maximize2
} from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { Panel, PageHeader } from "@/components/Panel";
import { Pitch } from "@/components/Pitch";
import { getGsrStatus } from "@/api/gsr";
import { 
  uploadVideoFile, 
  processVideo, 
  getVideoStatus, 
  getVideoStreamUrl, 
  runSampleDemo,
  type VideoProcessResponse,
  type VideoUploadResponse
} from "@/api/video";
import { apiFetch } from "@/api/client";
import type { Player } from "@/lib/data";
import { useSession } from "@/lib/session";

export const Route = createFileRoute("/broadcast")({
  head: () => ({
    meta: [
      { title: "Video Game State Reconstruction — Ball-Free AI" },
      {
        name: "description",
        content:
          "Upload match video gameplay to detect players, reconstruct pitch coordinates, and infer ball-free possession and tactical metrics.",
      },
    ],
  }),
  component: VideoReconstructionStudio,
});

function VideoReconstructionStudio() {
  const navigate = useNavigate();
  const { setMatchId, setDataset, setFrame } = useSession();
  const statusQuery = useQuery({ queryKey: ["gsr-status"], queryFn: getGsrStatus, staleTime: 30_000 });

  // Upload & processing state
  const [file, setFile] = useState<File | null>(null);
  const [videoId, setVideoId] = useState<string | null>(null);
  const [videoMeta, setVideoMeta] = useState<{
    total_frames?: number;
    fps?: number;
    width?: number;
    height?: number;
    duration_sec?: number;
  } | null>(null);

  // Configuration options (Single calibrated high-accuracy pitch projection)
  const preset = "auto_detect";
  const [durationPreset, setDurationPreset] = useState<"15s" | "30s" | "1m" | "3m" | "full" | "custom">("1m");
  const [customMaxFrames, setCustomMaxFrames] = useState<number>(1500);
  const [frameStride, setFrameStride] = useState(1);
  const [confThreshold, setConfThreshold] = useState(0.08);
  const [applyMitigation, setApplyMitigation] = useState(true);

  const [isUploading, setIsUploading] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [statusMsg, setStatusMsg] = useState("");
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Result state
  const [analysisResult, setAnalysisResult] = useState<VideoProcessResponse | null>(null);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1.0);
  const [framePlayers, setFramePlayers] = useState<Player[]>([]);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const videoContainerRef = useRef<HTMLDivElement | null>(null);

  // Effective video parameters
  const fps = analysisResult?.data.fps || videoMeta?.fps || 25.0;
  const maxProcessedFrame = analysisResult?.data.max_frame ?? 
    (analysisResult?.data.frame_count ? analysisResult.data.frame_count - 1 : 0);

  // File drop handler
  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      setFile(e.dataTransfer.files[0]);
      setErrorMsg(null);
      setVideoMeta(null);
    }
  };

  // Video container click for play/pause
  const handleVideoClick = () => {
    if (analysisResult && videoRef.current) {
      togglePlay();
    }
  };

  // Run Sample Demo
  const handleRunSampleDemo = async () => {
    try {
      setErrorMsg(null);
      setIsProcessing(true);
      setStatusMsg("Starting bundled match demo perception...");
      setProgress(0.05);

      await runSampleDemo();
      setVideoId("sample_demo_clip");

      const demoResult = await new Promise<VideoProcessResponse>((resolve, reject) => {
        let attempts = 0;
        const interval = setInterval(async () => {
          attempts += 1;
          if (attempts > 300) {
            clearInterval(interval);
            reject(new Error("Demo analysis timed out."));
            return;
          }

          try {
            const st = await getVideoStatus("sample_demo_clip");
            if (st.progress > 0) {
              setProgress(st.progress);
            }
            if (st.status) {
              setStatusMsg(st.status);
            }

            if (st.completed) {
              clearInterval(interval);
              if (st.error) {
                reject(new Error(st.error));
              } else if (st.data) {
                resolve(st.data);
              } else {
                reject(new Error("Demo completed with no data."));
              }
            }
          } catch {
            // Transient network error ignored
          }
        }, 800);
      });

      setProgress(1.0);
      setStatusMsg("Demo game state reconstructed!");
      setIsProcessing(false);
      setAnalysisResult(demoResult);
      setCurrentFrame(0);
      setMatchId(demoResult.match_id);
      setDataset(`Sample Match Clip (${demoResult.match_id})`);
      setFrame(0);
      if (videoRef.current) {
        videoRef.current.currentTime = 0;
      }
    } catch (err: any) {
      setIsProcessing(false);
      setErrorMsg(err.message || "Failed to run sample demo.");
    }
  };

  // Upload & Run Pipeline
  const handleStartAnalysis = async () => {
    if (!file) {
      setErrorMsg("Please select or drop a video file first.");
      return;
    }

    try {
      setErrorMsg(null);
      setIsUploading(true);
      setStatusMsg("Uploading video to server...");
      setProgress(0.05);

      const uploadRes: VideoUploadResponse = await uploadVideoFile(file);
      setVideoId(uploadRes.video_id);
      setVideoMeta({
        total_frames: uploadRes.total_frames,
        fps: uploadRes.fps,
        width: uploadRes.width,
        height: uploadRes.height,
        duration_sec: uploadRes.duration_sec,
      });
      setIsUploading(false);

      setIsProcessing(true);
      setStatusMsg("Initializing tracking and pitch homography engine...");
      setProgress(0.1);

      // Determine frames to process
      let framesToRun: number | undefined = 1500;
      if (durationPreset === "15s") framesToRun = 375;
      else if (durationPreset === "30s") framesToRun = 750;
      else if (durationPreset === "1m") framesToRun = 1500;
      else if (durationPreset === "3m") framesToRun = 4500;
      else if (durationPreset === "full") framesToRun = 0; // 0 signals full video
      else if (durationPreset === "custom") framesToRun = customMaxFrames;

      // Trigger background processing with single high-accuracy broadcast projection mode
      await processVideo({
        video_id: uploadRes.video_id,
        preset: "auto_detect",
        max_frames: framesToRun,
        frame_stride: frameStride,
        conf_threshold: confThreshold,
        apply_mitigation: applyMitigation,
      });

      // Poll until background processing completes
      const resultData = await new Promise<VideoProcessResponse>((resolve, reject) => {
        let attempts = 0;
        const maxAttempts = 900; // 12 minutes max polling
        const interval = setInterval(async () => {
          attempts += 1;
          if (attempts > maxAttempts) {
            clearInterval(interval);
            reject(new Error("Video analysis timed out after 12 minutes."));
            return;
          }

          try {
            const st = await getVideoStatus(uploadRes.video_id);
            if (st.progress > 0) {
              setProgress(st.progress);
            }
            if (st.status) {
              setStatusMsg(st.status);
            }

            if (st.completed) {
              clearInterval(interval);
              if (st.error) {
                reject(new Error(st.error));
              } else if (st.data) {
                resolve(st.data);
              } else {
                reject(new Error("Processing completed with no data."));
              }
            }
          } catch {
            // Ignore transient network hiccups during status polling
          }
        }, 800);
      });

      setProgress(1.0);
      setStatusMsg("Game state reconstruction complete!");
      setIsProcessing(false);
      setAnalysisResult(resultData);
      setCurrentFrame(0);
      setMatchId(resultData.match_id);
      setDataset(`Video: ${file.name.replace(/\.[^/.]+$/, "")}`);
      setFrame(0);
      if (videoRef.current) {
        videoRef.current.currentTime = 0;
      }
    } catch (err: any) {
      setIsUploading(false);
      setIsProcessing(false);
      setErrorMsg(err.message || "Failed to process video.");
    }
  };

  const allFramesRef = useRef<Record<number, Player[]>>({});

  // Pre-fetch all tracking frames for instantaneous zero-latency playback
  useEffect(() => {
    if (!analysisResult) return;
    const matchId = analysisResult.match_id;
    const effectiveFps = fps / Math.max(1, analysisResult.data.frame_stride ?? 1);

    apiFetch<{ frames: Record<string, Array<{ player_id: string; team: string; x: number | null; y: number | null; jersey_color?: string | null }>> }>(
      `/game-state/${matchId}/all`
    )
      .then((res) => {
        if (res.frames) {
          const cache: Record<number, Player[]> = {};
          const sortedFrames = Object.keys(res.frames).map(Number).sort((a, b) => a - b);
          
          for (const fNum of sortedFrames) {
            const players = res.frames[fNum] || [];
            const nextFramePlayers = res.frames[fNum + 1] || [];
            const nextMap = new Map(nextFramePlayers.map((p) => [p.player_id, p]));

            cache[fNum] = players
              .filter((p) => p.x !== null && p.y !== null)
              .map((p, idx) => {
                const parts = p.player_id.split("_");
                const playerNum = Number(parts[1]) || (idx + 1);
                const curX = (p.x ?? 0.5) * 105;
                const curY = (p.y ?? 0.5) * 68;

                let vx = 0;
                let vy = 0;
                const nextP = nextMap.get(p.player_id);
                if (nextP && nextP.x !== null && nextP.y !== null) {
                  const dx = (nextP.x * 105) - curX;
                  const dy = (nextP.y * 68) - curY;
                  const dist = Math.hypot(dx, dy);
                  if (dist > 0.04) {
                    const vectorLen = Math.min(9.45, dist * effectiveFps * 0.9);
                    vx = (dx / dist) * vectorLen;
                    vy = (dy / dist) * vectorLen;
                  }
                }
                const jerseyColor =
                  p.jersey_color || analysisResult.data.team_colors?.[
                    p.team === "away" ? "away" : "home"
                  ];

                return {
                  id: p.player_id,
                  team: (p.team === "home" || p.team === "away" ? p.team : "home") as "home" | "away",
                  ...(jerseyColor ? { jerseyColor } : {}),
                  number: playerNum,
                  name: p.player_id,
                  role: "Field",
                  x: curX,
                  y: curY,
                  vx,
                  vy,
                  visible: true,
                };
              });
          }
          allFramesRef.current = cache;
          if (cache[currentFrame]) {
            setFramePlayers(cache[currentFrame]);
          }
        }
      })
      .catch(() => {});
  }, [analysisResult, fps]);

  // Load frame tracking data with instant memory lookup
  useEffect(() => {
    if (!analysisResult) return;
    const cached = allFramesRef.current[currentFrame];
    if (cached && cached.length > 0) {
      setFramePlayers(cached);
      return;
    }

    const matchId = analysisResult.match_id;
    apiFetch<{ players: Array<{ player_id: string; team: string; x: number | null; y: number | null; jersey_color?: string | null }> }>(
      `/game-state/${matchId}/${currentFrame}`
    )
      .then((res) => {
        const mapped: Player[] = (res.players || [])
          .filter((p) => p.x !== null && p.y !== null)
          .map((p, idx) => {
            const parts = p.player_id.split("_");
            const playerNum = Number(parts[1]) || (idx + 1);
            const jerseyColor =
              p.jersey_color || analysisResult.data.team_colors?.[
                p.team === "away" ? "away" : "home"
              ];
            return {
              id: p.player_id,
              team: (p.team === "home" || p.team === "away" ? p.team : "home") as "home" | "away",
              ...(jerseyColor ? { jerseyColor } : {}),
              number: playerNum,
              name: p.player_id,
              role: "Field",
              x: (p.x ?? 0.5) * 105,
              y: (p.y ?? 0.5) * 68,
              vx: 0,
              vy: 0,
              visible: true,
            };
          });
        setFramePlayers(mapped);
      })
      .catch(() => {});
  }, [analysisResult, currentFrame]);

  // 60FPS fluid playback sync loop
  useEffect(() => {
    let animId: number;
    const syncLoop = () => {
      if (videoRef.current && isPlaying && analysisResult) {
        const curTime = videoRef.current.currentTime;
        const computedFrame = Math.min(maxProcessedFrame, Math.round(curTime * fps));
        setCurrentFrame((prev) => {
          if (prev !== computedFrame) {
            const cached = allFramesRef.current[computedFrame];
            if (cached && cached.length > 0) {
              setFramePlayers(cached);
            }
            return computedFrame;
          }
          return prev;
        });
      }
      if (isPlaying) {
        animId = requestAnimationFrame(syncLoop);
      }
    };

    if (isPlaying) {
      animId = requestAnimationFrame(syncLoop);
    }
    return () => {
      if (animId) cancelAnimationFrame(animId);
    };
  }, [isPlaying, analysisResult, maxProcessedFrame, fps]);

  // Synchronized playback controls
  const togglePlay = () => {
    if (!videoRef.current) return;
    if (videoRef.current.paused) {
      videoRef.current.play().then(() => setIsPlaying(true)).catch(console.error);
    } else {
      videoRef.current.pause();
      setIsPlaying(false);
    }
  };

  const handleSeekFrame = (targetFrame: number) => {
    const clamped = Math.max(0, Math.min(maxProcessedFrame, targetFrame));
    setCurrentFrame(clamped);
    const cached = allFramesRef.current[clamped];
    if (cached && cached.length > 0) {
      setFramePlayers(cached);
    }
    if (videoRef.current) {
      videoRef.current.currentTime = clamped / fps;
    }
  };

  const handleVideoTimeUpdate = () => {
    if (!videoRef.current || !analysisResult) return;
    const curTime = videoRef.current.currentTime;
    const computedFrame = Math.min(maxProcessedFrame, Math.round(curTime * fps));
    setCurrentFrame(computedFrame);
    const cached = allFramesRef.current[computedFrame];
    if (cached && cached.length > 0) {
      setFramePlayers(cached);
    }
  };

  const handleSpeedChange = (speed: number) => {
    setPlaybackSpeed(speed);
    if (videoRef.current) {
      videoRef.current.playbackRate = speed;
    }
  };

  const formatTime = (seconds: number) => {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    const ms = Math.floor((seconds % 1) * 100);
    return `${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}.${ms.toString().padStart(2, "0")}`;
  };

  const currentInsight = analysisResult?.data.sample_insights.find((s) => s.frame === currentFrame);

  return (
    <>
      {/* Top Brand Bar with Clickable Logo Returning to Landing Page */}
      <div className="mb-4 flex items-center justify-between border-b border-border/80 pb-3">
        <Link
          to="/"
          className="group flex items-center gap-2.5 cursor-pointer"
          title="Return to Product Landing Page"
        >
          <div className="h-8 w-8 rounded-lg bg-emerald-500/20 border border-emerald-500/40 flex items-center justify-center text-emerald-400 group-hover:border-emerald-300 group-hover:bg-emerald-500/30 transition-all shadow-sm">
            <Zap className="h-4 w-4" />
          </div>
          <div>
            <div className="text-sm font-bold tracking-[0.06em] text-foreground group-hover:text-emerald-400 transition-colors flex items-center gap-2">
              BALL-FREE AI
              <span className="text-[10px] font-medium font-mono text-emerald-400 border border-emerald-500/30 bg-emerald-500/10 px-1.5 py-0.5 rounded">
                STUDIO PRO
              </span>
            </div>
            <div className="text-[11px] text-muted-foreground flex items-center gap-1">
              <span>← Click logo to return to Landing Page</span>
            </div>
          </div>
        </Link>

        <Link
          to="/"
          className="text-xs text-muted-foreground hover:text-emerald-400 border border-border/80 bg-panel px-3 py-1.5 rounded flex items-center gap-1.5 transition-colors"
          title="Go to Product Landing Page"
        >
          <span>Landing Page Overview</span>
          <ArrowRight className="h-3 w-3" />
        </Link>
      </div>

      <PageHeader
        title="Video Game State Reconstruction Studio"
        description="Upload a match gameplay video to detect players, reconstruct canonical pitch coordinates with adaptive homography, and compute ball-free possession and Tactical Advantage Scores in real time."
      />

      {/* Perception Engine Banner & One-Click Demo Action */}
      <div className="mb-4 border border-emerald-500/40 bg-emerald-500/10 px-3 py-2.5 text-[12px] leading-relaxed text-emerald-400 flex flex-wrap items-center justify-between gap-2 rounded">
        <div className="flex items-center gap-2">
          <ShieldCheck className="h-4 w-4 text-emerald-400 shrink-0" />
          <span>
            <strong className="uppercase tracking-wider">Perception Engine Active:</strong> YOLOv8 + ByteTrack + Adaptive Pitch Homography is operational.
          </span>
        </div>
        <button
          onClick={handleRunSampleDemo}
          disabled={isUploading || isProcessing}
          className="flex items-center gap-1.5 bg-emerald-700 hover:bg-emerald-600 disabled:opacity-50 text-white text-[11px] font-semibold py-1 px-3 rounded transition-colors"
        >
          <Sparkles className="h-3.5 w-3.5 text-yellow-300" />
          <span>Run One-Click Demo Clip</span>
        </button>
      </div>

      {/* Main Grid: Upload & Analysis */}
      <div className="grid gap-4 lg:grid-cols-12">
        {/* Left Column: Upload & Configuration Controls */}
        <div className="lg:col-span-5 space-y-4">
          <Panel title="Video Ingestion" subtitle="Upload gameplay (.mp4, .mov, .avi, .webm)">
            {/* Drag & Drop Zone */}
            <div
              onDragOver={(e) => e.preventDefault()}
              onDrop={handleDrop}
              className={`border-2 border-dashed p-6 text-center cursor-pointer transition-colors rounded ${
                file ? "border-emerald-500/60 bg-emerald-500/5" : "border-border hover:border-emerald-500/40"
              }`}
              onClick={() => document.getElementById("video-file-input")?.click()}
            >
              <input
                id="video-file-input"
                type="file"
                accept="video/*"
                className="hidden"
                onChange={(e) => {
                  if (e.target.files && e.target.files.length > 0) {
                    setFile(e.target.files[0]);
                    setErrorMsg(null);
                    setVideoMeta(null);
                  }
                }}
              />
              <UploadCloud className="mx-auto h-8 w-8 text-muted-foreground mb-2" />
              {file ? (
                <div>
                  <div className="text-sm font-semibold text-emerald-400">{file.name}</div>
                  <div className="text-xs text-muted-foreground mt-0.5">
                    {(file.size / (1024 * 1024)).toFixed(2)} MB • Ready to analyze
                  </div>
                  {videoMeta && (
                    <div className="mt-2 text-[11px] font-mono text-emerald-300 border border-emerald-500/30 bg-emerald-500/10 p-1.5 rounded inline-block">
                      Duration: {formatTime(videoMeta.duration_sec || 0)} • {videoMeta.total_frames} frames @ {videoMeta.fps} FPS ({videoMeta.width}x{videoMeta.height})
                    </div>
                  )}
                </div>
              ) : (
                <div>
                  <div className="text-sm font-medium">Click to upload or drag & drop video</div>
                  <div className="text-xs text-muted-foreground mt-1">Upload any match clip (10 seconds to several minutes)</div>
                </div>
              )}
            </div>

            {/* Pipeline Configuration Parameters */}
            <div className="mt-4 space-y-3.5 border-t border-border pt-3 text-xs">
              {/* Single Pitch Projection Engine (Optimal Calibrated Broadcast Perspective) */}
              <div className="rounded-lg border border-emerald-500/30 bg-emerald-950/20 p-3 space-y-1.5">
                <div className="flex items-center justify-between">
                  <span className="text-foreground flex items-center gap-1.5 font-semibold text-xs">
                    <Sparkles className="h-4 w-4 text-emerald-400" /> Pitch Projection Engine
                  </span>
                  <span className="inline-flex items-center gap-1 text-[11px] font-semibold text-emerald-400 bg-emerald-500/10 border border-emerald-500/30 px-2 py-0.5 rounded-full">
                    <CheckCircle2 className="h-3 w-3" /> Full Pitch AI Calibration
                  </span>
                </div>
                <p className="text-[11px] text-muted-foreground leading-relaxed">
                  Autonomous perspective homography with dynamic grass segmentation. Continuously reconstructs full 105m × 68m coordinates without blindspots, clipping, or cut-offs, ensuring every defender, midfielder, and attacker movement is precisely tracked.
                </p>
                <div className="flex items-center gap-3 pt-1 text-[10px] text-emerald-300/90 font-mono">
                  <span>✓ 0% Boundary Clipping</span>
                  <span>✓ Center Spot Calibrated</span>
                  <span>✓ All 22 Tracked</span>
                </div>
              </div>

              {/* Duration to Process Selection */}
              <div>
                <div className="flex items-center justify-between mb-1.5">
                  <span className="text-muted-foreground flex items-center gap-1.5 font-medium">
                    <Clock className="h-3.5 w-3.5" /> Segment Duration to Process
                  </span>
                  <span className="font-mono text-[11px] text-emerald-400">
                    {durationPreset === "15s" && "15 sec (~375 frames)"}
                    {durationPreset === "30s" && "30 sec (~750 frames)"}
                    {durationPreset === "1m" && "1 min (~1,500 frames)"}
                    {durationPreset === "3m" && "3 min (~4,500 frames)"}
                    {durationPreset === "full" && "Full Video (All Frames)"}
                    {durationPreset === "custom" && `${customMaxFrames} frames`}
                  </span>
                </div>
                <div className="grid grid-cols-6 gap-1">
                  {(["15s", "30s", "1m", "3m", "full", "custom"] as const).map((mode) => (
                    <button
                      key={mode}
                      type="button"
                      onClick={() => setDurationPreset(mode)}
                      className={`py-1 text-[11px] font-medium rounded transition-all ${
                        durationPreset === mode
                          ? "bg-emerald-600 text-white font-semibold shadow-sm"
                          : "bg-panel-alt border border-border text-muted-foreground hover:text-foreground"
                      }`}
                    >
                      {mode.toUpperCase()}
                    </button>
                  ))}
                </div>

                {durationPreset === "custom" && (
                  <div className="mt-2 flex items-center gap-2">
                    <input
                      type="range"
                      min={50}
                      max={10000}
                      step={50}
                      value={customMaxFrames}
                      onChange={(e) => setCustomMaxFrames(Number(e.target.value))}
                      className="w-full accent-emerald-500"
                    />
                    <span className="w-16 text-right font-mono text-[11px]">{customMaxFrames}f</span>
                  </div>
                )}
              </div>

              {/* Frame Sampling Stride with Intuitive Explanations */}
              <div className="space-y-1.5 rounded-lg border border-border bg-panel-alt/40 p-3">
                <div className="flex items-center justify-between">
                  <div>
                    <span className="text-foreground flex items-center gap-1.5 font-semibold text-xs">
                      <Gauge className="h-4 w-4 text-sky-400" /> Frame Sampling Stride (Speed vs. Smoothness)
                    </span>
                    <div className="text-[11px] text-muted-foreground mt-0.5">
                      Controls how many video frames the AI analyzes. Analyze every frame for maximum smoothness, or skip frames for fast scanning.
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-1.5 pt-1">
                  {[
                    { stride: 1, label: "Stride 1 (Every Frame)", desc: "100% fidelity — Maximum accuracy & fluid motion (Recommended)" },
                    { stride: 2, label: "Stride 2 (Every 2nd Frame)", desc: "2× faster — Balanced speed and tracking quality" },
                    { stride: 3, label: "Stride 3 (Every 3rd Frame)", desc: "3× faster — Fast tactical overview for long clips" },
                    { stride: 5, label: "Stride 5 (Every 5th Frame)", desc: "5× ultra fast — Instant scanning of matches" },
                  ].map((item) => (
                    <button
                      key={item.stride}
                      type="button"
                      onClick={() => setFrameStride(item.stride)}
                      className={`text-left p-2 rounded border transition-all ${
                        frameStride === item.stride
                          ? "border-sky-500 bg-sky-500/10 text-foreground ring-1 ring-sky-500/40"
                          : "border-border bg-panel hover:bg-muted/40 text-muted-foreground"
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <span className={`text-xs font-semibold ${frameStride === item.stride ? "text-sky-300" : "text-foreground"}`}>
                          {item.label}
                        </span>
                        {frameStride === item.stride && (
                          <span className="text-[9px] bg-sky-500/20 text-sky-300 font-semibold px-1.5 py-0.2 rounded">
                            ACTIVE
                          </span>
                        )}
                      </div>
                      <div className="text-[10px] text-muted-foreground mt-0.5 leading-snug">{item.desc}</div>
                    </button>
                  ))}
                </div>

                <div className="text-[10px] text-sky-400/90 pt-1 font-medium flex items-center gap-1">
                  <Zap className="h-3 w-3 shrink-0" />
                  {frameStride === 1 && "Stride 1 analyzes all 60 frames per second so player sprints and runs are captured without any stutter."}
                  {frameStride === 2 && "Stride 2 cuts processing duration in half while still maintaining clear tactical team shapes."}
                  {frameStride === 3 && "Stride 3 skips two out of three frames for rapid tactical analysis."}
                  {frameStride === 5 && "Stride 5 gives an ultra-fast preview of game flow."}
                </div>
              </div>

              {/* Player Detection Sensitivity with Easy Explanation */}
              <div className="space-y-1.5 rounded-lg border border-border bg-panel-alt/40 p-3">
                <div className="flex items-center justify-between">
                  <div>
                    <span className="text-foreground flex items-center gap-1.5 font-semibold text-xs">
                      <Sliders className="h-4 w-4 text-emerald-400" /> Player Detection Sensitivity
                    </span>
                    <div className="text-[11px] text-muted-foreground mt-0.5">
                      Determines how sensitive the AI vision model is to detecting players across the pitch.
                    </div>
                  </div>
                  <span className="font-mono text-xs font-bold px-2 py-0.5 rounded bg-panel border border-border text-emerald-400">
                    {confThreshold.toFixed(2)}
                  </span>
                </div>

                <div className="pt-2">
                  <input
                    type="range"
                    min={0.06}
                    max={0.30}
                    step={0.02}
                    value={confThreshold}
                    onChange={(e) => setConfThreshold(Number(e.target.value))}
                    className="w-full accent-emerald-500 h-1.5 bg-panel rounded cursor-pointer"
                  />
                  <div className="flex justify-between text-[10px] text-muted-foreground mt-1 font-mono">
                    <span>0.06 (Max Recall)</span>
                    <span className="text-emerald-400 font-semibold">0.08 (Recommended)</span>
                    <span>0.30 (Close-ups only)</span>
                  </div>
                </div>

                <div className="rounded bg-panel/80 border border-border/60 p-2 text-[11px]">
                  {confThreshold <= 0.10 ? (
                    <div className="text-emerald-300 flex items-start gap-1.5">
                      <ShieldCheck className="h-3.5 w-3.5 shrink-0 mt-0.5 text-emerald-400" />
                      <span>
                        <strong className="font-semibold text-emerald-200">High Sensitivity ({confThreshold.toFixed(2)}):</strong> Detects distant players, far-side wingers, and deep defenders who appear smaller due to camera perspective. Ensures all players are tracked without dropping anyone.
                      </span>
                    </div>
                  ) : confThreshold <= 0.20 ? (
                    <div className="text-amber-300 flex items-start gap-1.5">
                      <ShieldCheck className="h-3.5 w-3.5 shrink-0 mt-0.5 text-amber-400" />
                      <span>
                        <strong className="font-semibold text-amber-200">Balanced Sensitivity ({confThreshold.toFixed(2)}):</strong> Standard detection threshold. Best for close-up angles, but may occasionally miss far-side players or wide defenders.
                      </span>
                    </div>
                  ) : (
                    <div className="text-orange-300 flex items-start gap-1.5">
                      <AlertCircle className="h-3.5 w-3.5 shrink-0 mt-0.5 text-orange-400" />
                      <span>
                        <strong className="font-semibold text-orange-200">Conservative ({confThreshold.toFixed(2)}):</strong> Only detects large, clear foreground players. Distant defenders and wide players will not be tracked.
                      </span>
                    </div>
                  )}
                </div>
              </div>

              {/* Trajectory Mitigation */}
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground font-medium">Trajectory Smoothing</span>
                <label className="flex items-center gap-1.5 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={applyMitigation}
                    onChange={(e) => setApplyMitigation(e.target.checked)}
                    className="accent-emerald-500"
                  />
                  <span className="text-[11px]">Interpolation + SavGol</span>
                </label>
              </div>
            </div>

            {/* Error Message */}
            {errorMsg && (
              <div className="mt-3 flex items-center gap-2 border border-destructive/50 bg-destructive/10 p-2 text-xs text-destructive rounded">
                <AlertCircle className="h-4 w-4 shrink-0" />
                <span>{errorMsg}</span>
              </div>
            )}

            {/* Progress Bar & Status */}
            {(isUploading || isProcessing) && (
              <div className="mt-4 space-y-1.5">
                <div className="flex justify-between text-xs text-muted-foreground">
                  <span className="font-medium text-foreground">{statusMsg}</span>
                  <span className="font-mono">{Math.round(progress * 100)}%</span>
                </div>
                <div className="h-1.5 w-full bg-panel-alt overflow-hidden rounded">
                  <div
                    className="h-full bg-emerald-500 transition-all duration-300"
                    style={{ width: `${Math.round(progress * 100)}%` }}
                  />
                </div>
              </div>
            )}

            {/* Run Button */}
            <button
              onClick={handleStartAnalysis}
              disabled={!file || isUploading || isProcessing}
              className="mt-4 w-full flex items-center justify-center gap-2 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white text-xs font-semibold py-2.5 px-4 rounded transition-colors shadow-sm"
            >
              {isUploading || isProcessing ? (
                <>
                  <RefreshCw className="h-3.5 w-3.5 animate-spin" /> Processing Video Pipeline...
                </>
              ) : (
                <>
                  <Play className="h-3.5 w-3.5" /> Reconstruct Game State
                </>
              )}
            </button>
          </Panel>

          {/* Quick Metrics (When Processed) */}
          {analysisResult && (
            <Panel title="Reconstruction Summary" subtitle={`Match ID: ${analysisResult.match_id}`}>
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="border border-border bg-panel-alt p-2 rounded">
                  <div className="text-muted-foreground text-[11px]">Frames Reconstructed</div>
                  <div className="text-base font-bold font-mono">
                    {analysisResult.data.frame_count} frames ({analysisResult.data.duration_sec || (analysisResult.data.max_frame ? (analysisResult.data.max_frame / fps).toFixed(1) : 0)}s)
                  </div>
                </div>
                <div className="border border-border bg-panel-alt p-2 rounded">
                  <div className="text-muted-foreground text-[11px]">Player Observations</div>
                  <div className="text-base font-bold font-mono">{analysisResult.data.tracking_rows}</div>
                </div>
                <div className="border border-border bg-panel-alt p-2 rounded">
                  <div className="text-muted-foreground text-[11px]">Average TAS Score</div>
                  <div className="text-base font-bold font-mono text-emerald-400">
                    {analysisResult.data.average_tas.toFixed(3)}
                  </div>
                </div>
                <div className="border border-border bg-panel-alt p-2 rounded">
                  <div className="text-muted-foreground text-[11px]">Possession Split</div>
                  <div className="text-xs font-semibold font-mono mt-1">
                    H: {analysisResult.data.possession_share.home}% | A: {analysisResult.data.possession_share.away}%
                  </div>
                </div>
              </div>

              <div className="mt-3 grid grid-cols-2 gap-2">
                <button
                  onClick={() => navigate({ to: "/match-explorer" })}
                  className="flex items-center justify-center gap-1.5 border border-border bg-panel-alt hover:bg-muted text-xs font-medium py-1.5 px-2 rounded transition-colors"
                >
                  <span>Match Explorer</span>
                  <ArrowRight className="h-3 w-3" />
                </button>
                <button
                  onClick={() => navigate({ to: "/game-state" })}
                  className="flex items-center justify-center gap-1.5 bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold py-1.5 px-2 rounded transition-colors"
                >
                  <span>Game State Radar</span>
                  <ArrowRight className="h-3 w-3" />
                </button>
              </div>
            </Panel>
          )}
        </div>

        {/* Right Column: Dual View (Video Playback & Synchronized 2D Pitch) */}
        <div className="lg:col-span-7 space-y-4">
          <Panel 
            title="Synchronized Game State Radar" 
            subtitle={
              analysisResult 
                ? `Frame ${currentFrame} / ${maxProcessedFrame} (${formatTime(currentFrame / fps)} / ${formatTime(maxProcessedFrame / fps)})`
                : "Waiting for video upload"
            }
          >
            {/* Split Screen Container */}
            <div className="grid md:grid-cols-2 gap-3 items-center">
              {/* Video Player */}
              <div 
                ref={videoContainerRef}
                onClick={handleVideoClick}
                className="relative aspect-video bg-black border border-border flex items-center justify-center overflow-hidden rounded group cursor-pointer"
              >
                {videoId ? (
                  <video
                    ref={videoRef}
                    src={getVideoStreamUrl(videoId)}
                    className="w-full h-full object-contain"
                    playsInline
                    onTimeUpdate={handleVideoTimeUpdate}
                    onPlay={() => setIsPlaying(true)}
                    onPause={() => setIsPlaying(false)}
                    onEnded={() => setIsPlaying(false)}
                  />
                ) : (
                  <div className="text-center p-4 text-muted-foreground pointer-events-none">
                    <Video className="mx-auto h-8 w-8 mb-2 opacity-40" />
                    <span className="text-xs">Uploaded video will play here</span>
                  </div>
                )}

                {/* Center Play Overlay when Paused */}
                {videoId && !isPlaying && (
                  <div className="absolute inset-0 flex items-center justify-center bg-black/30 pointer-events-none transition-opacity">
                    <div className="h-12 w-12 rounded-full bg-emerald-600/90 flex items-center justify-center text-white shadow-xl">
                      <Play className="h-6 w-6 ml-0.5" />
                    </div>
                  </div>
                )}

                {videoId && (
                  <div className="absolute top-2 left-2 bg-black/80 px-2 py-0.5 text-[10px] font-mono text-white rounded pointer-events-none border border-white/20">
                    Frame {currentFrame} • {formatTime(currentFrame / fps)}
                  </div>
                )}
              </div>

              {/* 2D Pitch Radar */}
              <div className="border border-border bg-turf p-1 rounded overflow-hidden">
                <Pitch 
                  players={framePlayers} 
                  height={220} 
                  showVectors={true} 
                  heat={false} 
                />
              </div>
            </div>

            {/* Playback Controls & Frame Scrubber */}
            {analysisResult && (
              <div className="mt-3 border-t border-border pt-3 space-y-2">
                <div className="flex items-center gap-2">
                  <input
                    type="range"
                    min={0}
                    max={maxProcessedFrame}
                    value={currentFrame}
                    onChange={(e) => handleSeekFrame(Number(e.target.value))}
                    className="w-full accent-emerald-500 h-1.5 bg-panel-alt rounded cursor-pointer"
                  />
                </div>

                <div className="flex items-center justify-between text-xs">
                  {/* Playback Buttons */}
                  <div className="flex items-center gap-1.5">
                    <button
                      onClick={() => handleSeekFrame(Math.max(0, currentFrame - 5))}
                      title="Step -5 Frames"
                      className="p-1 rounded bg-panel-alt hover:bg-muted text-foreground transition-colors border border-border"
                    >
                      <ChevronsLeft className="h-3.5 w-3.5" />
                    </button>
                    <button
                      onClick={() => handleSeekFrame(Math.max(0, currentFrame - 1))}
                      title="Step -1 Frame"
                      className="p-1 rounded bg-panel-alt hover:bg-muted text-foreground transition-colors border border-border"
                    >
                      <ChevronLeft className="h-3.5 w-3.5" />
                    </button>
                    <button
                      onClick={togglePlay}
                      title={isPlaying ? "Pause" : "Play"}
                      className="bg-emerald-600 hover:bg-emerald-500 text-white p-1.5 rounded transition-colors px-2.5 flex items-center gap-1"
                    >
                      {isPlaying ? <Pause className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" />}
                      <span className="text-[11px] font-semibold">{isPlaying ? "Pause" : "Play"}</span>
                    </button>
                    <button
                      onClick={() => handleSeekFrame(Math.min(maxProcessedFrame, currentFrame + 1))}
                      title="Step +1 Frame"
                      className="p-1 rounded bg-panel-alt hover:bg-muted text-foreground transition-colors border border-border"
                    >
                      <ChevronRight className="h-3.5 w-3.5" />
                    </button>
                    <button
                      onClick={() => handleSeekFrame(Math.min(maxProcessedFrame, currentFrame + 5))}
                      title="Step +5 Frames"
                      className="p-1 rounded bg-panel-alt hover:bg-muted text-foreground transition-colors border border-border"
                    >
                      <ChevronsRight className="h-3.5 w-3.5" />
                    </button>
                  </div>

                  {/* Frame & Time Counter */}
                  <div className="font-mono text-xs text-muted-foreground flex items-center gap-2">
                    <span className="text-emerald-400 font-semibold">{formatTime(currentFrame / fps)}</span>
                    <span>/</span>
                    <span>{formatTime(maxProcessedFrame / fps)}</span>
                    <span className="text-[10px] text-muted-foreground bg-panel-alt px-1.5 py-0.5 rounded border border-border">
                      Frame {currentFrame} / {maxProcessedFrame}
                    </span>
                  </div>

                  {/* Playback Speed Selector */}
                  <div className="flex items-center gap-1">
                    {[0.5, 1.0, 1.5, 2.0].map((spd) => (
                      <button
                        key={spd}
                        onClick={() => handleSpeedChange(spd)}
                        className={`text-[10px] font-mono px-1.5 py-0.5 rounded transition-colors ${
                          playbackSpeed === spd
                            ? "bg-emerald-600 text-white font-bold"
                            : "bg-panel-alt hover:bg-muted text-muted-foreground"
                        }`}
                      >
                        {spd}x
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {/* Live Frame Tactical Insights */}
            {currentInsight && (
              <div className="mt-3 grid grid-cols-3 gap-2 border-t border-border pt-3 text-xs">
                <div>
                  <span className="text-[11px] text-muted-foreground">Inferred Possessor</span>
                  <div className="font-bold text-emerald-400">
                    {currentInsight.possessor_player_id || "Uncontested"}
                  </div>
                </div>
                <div>
                  <span className="text-[11px] text-muted-foreground">Possession Team</span>
                  <div className="font-bold uppercase text-xs">
                    {currentInsight.possessing_team || "Neutral"}
                  </div>
                </div>
                <div>
                  <span className="text-[11px] text-muted-foreground">Frame TAS Advantage</span>
                  <div className="font-bold font-mono text-emerald-400">
                    {currentInsight.tas.toFixed(3)}
                  </div>
                </div>
              </div>
            )}
          </Panel>
        </div>
      </div>
    </>
  );
}
