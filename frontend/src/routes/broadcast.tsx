import { useState, useRef, useEffect } from "react";
import { createFileRoute, useNavigate } from "@tanstack/react-router";
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
  Crosshair,
  RotateCcw
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
  type VideoProcessResponse 
} from "@/api/video";
import { apiFetch } from "@/api/client";
import type { Player } from "@/lib/data";

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

const PIPELINE_MODULES = [
  { name: "Frame Ingest", detail: "Video decode & frame buffering at native FPS", status: "Active" },
  { name: "YOLOv8 Detection", detail: "Per-frame player & referee bounding boxes", status: "Active" },
  { name: "ByteTrack Tracker", detail: "Multi-object identity association across frames", status: "Active" },
  { name: "Jersey Classifier", detail: "K-Means color clustering into Home vs Away teams", status: "Active" },
  { name: "Pitch Homography", detail: "Perspective transform to normalized 2D pitch coordinates", status: "Active" },
  { name: "Ball-Free Engine", detail: "Spatial graph, heuristic possession & TAS evaluation", status: "Active" },
];

function VideoReconstructionStudio() {
  const navigate = useNavigate();
  const statusQuery = useQuery({ queryKey: ["gsr-status"], queryFn: getGsrStatus, staleTime: 30_000 });

  // Upload & processing state
  const [file, setFile] = useState<File | null>(null);
  const [videoId, setVideoId] = useState<string | null>(null);
  const [preset, setPreset] = useState("main_broadcast");
  const [maxFrames, setMaxFrames] = useState(150);
  const [frameStride, setFrameStride] = useState(1);
  const [applyMitigation, setApplyMitigation] = useState(true);

  // 4-Point Interactive Calibration state
  const [isCalibrating, setIsCalibrating] = useState(false);
  const [customPoints, setCustomPoints] = useState<Array<[number, number]>>([]);

  const [isUploading, setIsUploading] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [statusMsg, setStatusMsg] = useState("");
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Result state
  const [analysisResult, setAnalysisResult] = useState<VideoProcessResponse | null>(null);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [framePlayers, setFramePlayers] = useState<Player[]>([]);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const videoContainerRef = useRef<HTMLDivElement | null>(null);

  // File drop handler
  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      setFile(e.dataTransfer.files[0]);
      setErrorMsg(null);
    }
  };

  // Video container click for 4-point calibration
  const handleVideoClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!isCalibrating || customPoints.length >= 4) return;
    const rect = videoContainerRef.current?.getBoundingClientRect();
    if (!rect) return;

    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;
    const relX = clickX / rect.width;
    const relY = clickY / rect.height;

    // Map relative fraction to video dimensions (default 1280x720)
    const px = Math.round(relX * 1280);
    const py = Math.round(relY * 720);

    setCustomPoints((prev) => [...prev, [px, py]]);
  };

  // Run Sample Demo
  const handleRunSampleDemo = async () => {
    try {
      setErrorMsg(null);
      setIsProcessing(true);
      setStatusMsg("Running AI perception on bundled match demo...");
      setProgress(0.2);

      const demoRes = await runSampleDemo();
      setVideoId("sample_demo_clip");
      setProgress(1.0);
      setStatusMsg("Demo game state reconstructed!");
      setIsProcessing(false);
      setAnalysisResult(demoRes);
      setCurrentFrame(0);
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
      setProgress(0.1);

      const uploadRes = await uploadVideoFile(file);
      setVideoId(uploadRes.video_id);
      setIsUploading(false);

      setIsProcessing(true);
      setStatusMsg("Tracking players and running homography calibration...");

      // Start polling status
      const pollInterval = setInterval(async () => {
        try {
          const st = await getVideoStatus(uploadRes.video_id);
          if (st.progress > 0) {
            setProgress(st.progress);
            setStatusMsg(st.status);
          }
        } catch {
          // Ignore transient errors
        }
      }, 800);

      const processRes = await processVideo({
        video_id: uploadRes.video_id,
        preset: preset === "custom_4point" ? "main_broadcast" : preset,
        max_frames: maxFrames,
        frame_stride: frameStride,
        apply_mitigation: applyMitigation,
        custom_corners: customPoints.length === 4 ? customPoints : undefined,
      });

      clearInterval(pollInterval);
      setProgress(1.0);
      setStatusMsg("Game state reconstruction complete!");
      setIsProcessing(false);
      setAnalysisResult(processRes);
      setCurrentFrame(0);
    } catch (err: any) {
      setIsUploading(false);
      setIsProcessing(false);
      setErrorMsg(err.message || "Failed to process video.");
    }
  };

  // Load frame tracking data when currentFrame changes
  useEffect(() => {
    if (!analysisResult) return;
    const matchId = analysisResult.match_id;

    apiFetch<{ players: Array<{ player_id: string; team: string; x: number | null; y: number | null }> }>(
      `/game-state/${matchId}/${currentFrame}`
    )
      .then((res) => {
        const mapped: Player[] = (res.players || [])
          .filter((p) => p.x !== null && p.y !== null)
          .map((p, idx) => ({
            id: p.player_id,
            team: (p.team === "home" || p.team === "away" ? p.team : "home") as "home" | "away",
            number: idx + 1,
            name: p.player_id,
            role: "Field",
            // Convert normalized pitch coordinates [0, 1] to meters (105 x 68) for the Pitch component
            x: (p.x ?? 0.5) * 105,
            y: (p.y ?? 0.5) * 68,
            vx: 0,
            vy: 0,
            visible: true,
          }));
        setFramePlayers(mapped);
      })
      .catch(() => {
        // Handle missing frame quietly
      });
  }, [analysisResult, currentFrame]);

  // Synchronized playback timer
  useEffect(() => {
    let timer: any = null;
    if (isPlaying && analysisResult) {
      timer = setInterval(() => {
        setCurrentFrame((prev) => {
          const next = prev + 1;
          if (next >= (analysisResult.data.frame_count || 1)) {
            setIsPlaying(false);
            return 0;
          }
          return next;
        });
      }, 100);
    }
    return () => clearInterval(timer);
  }, [isPlaying, analysisResult]);

  const currentInsight = analysisResult?.data.sample_insights.find((s) => s.frame === currentFrame);

  return (
    <>
      <PageHeader
        title="Video Game State Reconstruction Studio"
        description="Upload a gameplay video clip to detect players, reconstruct pitch coordinates, and compute ball-free possession and Tactical Advantage Scores in real time."
      />

      {/* Perception Engine Banner & One-Click Demo Action */}
      <div className="mb-4 border border-emerald-500/40 bg-emerald-500/10 px-3 py-2.5 text-[12px] leading-relaxed text-emerald-400 flex flex-wrap items-center justify-between gap-2 rounded">
        <div className="flex items-center gap-2">
          <ShieldCheck className="h-4 w-4 text-emerald-400 shrink-0" />
          <span>
            <strong className="uppercase tracking-wider">Perception Engine Active:</strong> YOLOv8 + ByteTrack + 2D Homography is operational.
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
          <Panel title="Video Ingestion" subtitle="Upload gameplay (.mp4, .mov, .avi)">
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
                  }
                }}
              />
              <UploadCloud className="mx-auto h-8 w-8 text-muted-foreground mb-2" />
              {file ? (
                <div>
                  <div className="text-sm font-semibold text-emerald-400">{file.name}</div>
                  <div className="text-xs text-muted-foreground mt-0.5">
                    {(file.size / (1024 * 1024)).toFixed(2)} MB • Ready to process
                  </div>
                </div>
              ) : (
                <div>
                  <div className="text-sm font-medium">Click to upload or drag & drop video</div>
                  <div className="text-xs text-muted-foreground mt-1">Recommended: 10–30s match clips at 25 FPS</div>
                </div>
              )}
            </div>

            {/* Pipeline Configuration Parameters */}
            <div className="mt-4 space-y-3 border-t border-border pt-3 text-xs">
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground flex items-center gap-1.5 font-medium">
                  <Sliders className="h-3.5 w-3.5" /> Camera Preset
                </span>
                <select
                  value={preset}
                  onChange={(e) => {
                    setPreset(e.target.value);
                    if (e.target.value === "custom_4point") {
                      setIsCalibrating(true);
                    } else {
                      setIsCalibrating(false);
                    }
                  }}
                  className="bg-panel border border-border px-2 py-1 text-xs rounded"
                >
                  <option value="main_broadcast">Main Broadcast (Elevated Sideline)</option>
                  <option value="tactical_wide">Tactical Wide (High Endline)</option>
                  <option value="normalized">Normalized 2D Radar View</option>
                  <option value="custom_4point">Custom 4-Point Calibration</option>
                </select>
              </div>

              {/* 4-Point Interactive Calibration Banner */}
              {preset === "custom_4point" && (
                <div className="border border-blue-500/40 bg-blue-500/10 p-2 text-xs rounded space-y-1.5">
                  <div className="flex items-center justify-between font-semibold text-blue-400">
                    <span className="flex items-center gap-1">
                      <Crosshair className="h-3.5 w-3.5" /> 4-Point Calibration Mode
                    </span>
                    <button
                      onClick={() => setCustomPoints([])}
                      className="text-[10px] text-muted-foreground hover:text-white flex items-center gap-0.5"
                    >
                      <RotateCcw className="h-3 w-3" /> Reset
                    </button>
                  </div>
                  <div className="text-[11px] text-muted-foreground">
                    Click 4 pitch points on the video: Top-Left, Top-Right, Bottom-Right, Bottom-Left.
                  </div>
                  <div className="font-mono text-[10px] text-blue-300">
                    Points placed: {customPoints.length} / 4
                  </div>
                </div>
              )}

              <div className="flex items-center justify-between">
                <span className="text-muted-foreground font-medium">Max Frames to Process</span>
                <div className="flex items-center gap-2">
                  <input
                    type="range"
                    min={30}
                    max={300}
                    step={10}
                    value={maxFrames}
                    onChange={(e) => setMaxFrames(Number(e.target.value))}
                    className="w-24 accent-emerald-500"
                  />
                  <span className="w-8 text-right font-mono text-[11px]">{maxFrames}</span>
                </div>
              </div>

              <div className="flex items-center justify-between">
                <span className="text-muted-foreground font-medium">Trajectory Cleaning</span>
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
                  <span>{statusMsg}</span>
                  <span>{Math.round(progress * 100)}%</span>
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
              className="mt-4 w-full flex items-center justify-center gap-2 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white text-xs font-semibold py-2 px-4 rounded transition-colors"
            >
              {isUploading || isProcessing ? (
                <>
                  <RefreshCw className="h-3.5 w-3.5 animate-spin" /> Processing Video...
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
                  <div className="text-base font-bold font-mono">{analysisResult.data.frame_count}</div>
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

              <button
                onClick={() => navigate({ to: "/game-state" })}
                className="mt-3 w-full flex items-center justify-center gap-1.5 border border-border hover:bg-panel-alt text-xs font-medium py-1.5 px-3 rounded transition-colors"
              >
                <span>Open in Full Match Explorer</span>
                <ArrowRight className="h-3.5 w-3.5" />
              </button>
            </Panel>
          )}
        </div>

        {/* Right Column: Dual View (Video Playback & Synchronized 2D Pitch) */}
        <div className="lg:col-span-7 space-y-4">
          <Panel 
            title="Synchronized Game State Radar" 
            subtitle={analysisResult ? `Frame ${currentFrame} / ${analysisResult.data.frame_count}` : "Waiting for video upload"}
          >
            {/* Split Screen Container */}
            <div className="grid md:grid-cols-2 gap-3 items-center">
              {/* Video Player & Calibration Canvas */}
              <div 
                ref={videoContainerRef}
                onClick={handleVideoClick}
                className={`relative aspect-video bg-black border border-border flex items-center justify-center overflow-hidden rounded ${
                  isCalibrating ? "cursor-crosshair ring-2 ring-blue-500" : ""
                }`}
              >
                {videoId ? (
                  <video
                    ref={videoRef}
                    src={getVideoStreamUrl(videoId)}
                    className="w-full h-full object-contain pointer-events-none"
                    controls={false}
                  />
                ) : (
                  <div className="text-center p-4 text-muted-foreground">
                    <Video className="mx-auto h-8 w-8 mb-2 opacity-40" />
                    <span className="text-xs">Uploaded video will play here</span>
                  </div>
                )}
                
                {/* 4-Point Calibration Markers */}
                {isCalibrating && customPoints.map((pt, i) => (
                  <div
                    key={i}
                    className="absolute w-5 h-5 -ml-2.5 -mt-2.5 rounded-full bg-blue-500 text-white font-bold text-[10px] flex items-center justify-center shadow-lg border border-white"
                    style={{ left: `${(pt[0] / 1280) * 100}%`, top: `${(pt[1] / 720) * 100}%` }}
                  >
                    {i + 1}
                  </div>
                ))}

                {videoId && (
                  <div className="absolute top-2 left-2 bg-black/70 px-1.5 py-0.5 text-[10px] font-mono text-white rounded">
                    Frame {currentFrame}
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
              <div className="mt-3 flex items-center gap-3 border-t border-border pt-3">
                <button
                  onClick={() => setIsPlaying(!isPlaying)}
                  className="bg-emerald-600 hover:bg-emerald-500 text-white p-1.5 rounded transition-colors"
                >
                  {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                </button>
                <input
                  type="range"
                  min={0}
                  max={(analysisResult.data.frame_count || 1) - 1}
                  value={currentFrame}
                  onChange={(e) => {
                    setIsPlaying(false);
                    setCurrentFrame(Number(e.target.value));
                  }}
                  className="w-full accent-emerald-500"
                />
                <span className="font-mono text-xs w-16 text-right">
                  {currentFrame} / {analysisResult.data.frame_count - 1}
                </span>
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

          {/* Active Architecture Modules */}
          <Panel title="Vision Perception Architecture" subtitle="All 6 perception stages operational">
            <div className="grid gap-2 sm:grid-cols-2 md:grid-cols-3 text-xs">
              {PIPELINE_MODULES.map((m, i) => (
                <div key={m.name} className="border border-border bg-panel-alt p-2 rounded">
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] text-muted-foreground font-mono">Stage 0{i + 1}</span>
                    <span className="flex items-center gap-1 text-[10px] text-emerald-400 font-medium">
                      <CheckCircle2 className="h-3 w-3" /> {m.status}
                    </span>
                  </div>
                  <div className="font-semibold text-xs mt-1">{m.name}</div>
                  <div className="text-[11px] text-muted-foreground mt-0.5">{m.detail}</div>
                </div>
              ))}
            </div>
          </Panel>
        </div>
      </div>
    </>
  );
}
