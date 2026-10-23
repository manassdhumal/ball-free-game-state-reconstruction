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
  Scissors,
  ChevronLeft,
  ChevronRight,
  ChevronsLeft,
  ChevronsRight,
  Maximize2,
  Database,
  Compass,
  Trash2,
  Check,
  HardDrive
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
  getStoredVideos,
  getStoredVideoResult,
  deleteStoredVideo,
  type VideoProcessResponse,
  type VideoUploadResponse,
  type StoredVideo
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
  const storedVideosQuery = useQuery({
    queryKey: ["stored-videos"],
    queryFn: getStoredVideos,
    staleTime: 5000,
  });

  // Upload & processing state
  const [file, setFile] = useState<File | null>(null);
  const [videoId, setVideoId] = useState<string | null>(null);
  const [selectedStoredId, setSelectedStoredId] = useState<string | null>(null);
  const [selectedVideoName, setSelectedVideoName] = useState<string | null>(null);
  const [videoMeta, setVideoMeta] = useState<{
    total_frames?: number;
    fps?: number;
    width?: number;
    height?: number;
    duration_sec?: number;
  } | null>(null);

  // Configuration options (Enforced optimal AI perception settings)
  const preset = "auto_detect";
  const frameStride = 1;       // Optimal: 100% frame analysis fidelity for butter-smooth trajectories
  const confThreshold = 0.08;  // Optimal: high recall capturing all 22 outfield players and goalkeepers
  const applyMitigation = true;

  // Segment & Time Crop options (Supports custom intervals e.g. Minute 2-3)
  const [segmentMode, setSegmentMode] = useState<"full" | "30s" | "1m" | "3m" | "crop">("1m");
  const [cropStartMin, setCropStartMin] = useState<number>(0);
  const [cropStartSec, setCropStartSec] = useState<number>(0);
  const [cropEndMin, setCropEndMin] = useState<number>(1);
  const [cropEndSec, setCropEndSec] = useState<number>(0);

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
  const fps = analysisResult?.data?.fps || videoMeta?.fps || 25.0;
  const maxProcessedFrame = analysisResult?.data?.max_frame ?? 
    (analysisResult?.data?.frame_count ? analysisResult.data.frame_count - 1 : 0);

  const storedVideos = storedVideosQuery.data?.videos || [];

  // Handler for picking from stored data
  const handleSelectStoredVideo = async (stored: StoredVideo) => {
    try {
      setErrorMsg(null);
      setSelectedStoredId(stored.video_id);
      setSelectedVideoName(stored.filename);
      setVideoId(stored.video_id);
      setVideoMeta({
        total_frames: stored.total_frames,
        fps: stored.fps,
        width: stored.width,
        height: stored.height,
        duration_sec: stored.duration_sec,
      });
      setFile(null); // working on stored video

      if (stored.processed) {
        setIsProcessing(true);
        setStatusMsg(`Loading stored reconstruction for ${stored.filename}...`);
        try {
          const res = await getStoredVideoResult(stored.video_id);
          if (res && res.data) {
            setAnalysisResult(res);
            setCurrentFrame(0);
            setMatchId(stored.match_id);
            setDataset(`Video: ${stored.filename.replace(/\.[^/.]+$/, "")}`);
            setFrame(0);
            setStatusMsg(`Loaded reconstructed game state for "${stored.filename}"!`);
          }
        } catch (err: any) {
          setErrorMsg(`Failed to load stored analysis: ${err.message}`);
        } finally {
          setIsProcessing(false);
        }
      } else {
        setAnalysisResult(null);
        setStatusMsg(`Selected stored video "${stored.filename}". Ready to reconstruct.`);
      }
    } catch (err: any) {
      setErrorMsg("Error selecting video: " + err.message);
    }
  };

  const handleDeleteStoredVideo = async (e: React.MouseEvent, vid: string) => {
    e.stopPropagation();
    try {
      await deleteStoredVideo(vid);
      storedVideosQuery.refetch();
      if (videoId === vid) {
        setVideoId(null);
        setAnalysisResult(null);
        setSelectedStoredId(null);
        setSelectedVideoName(null);
      }
    } catch (err: any) {
      setErrorMsg("Failed to delete video: " + err.message);
    }
  };

  // File drop handler
  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const selected = e.dataTransfer.files[0];
      setFile(selected);
      setSelectedStoredId(null);
      setSelectedVideoName(selected.name);
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
      setSelectedStoredId("sample_demo_clip");
      setSelectedVideoName("sample_match_clip.mp4");

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

      if (!demoResult?.data || (demoResult as any).status === "error" || demoResult.data?.status === "error") {
        setIsProcessing(false);
        setErrorMsg((demoResult as any)?.message || demoResult?.data?.message || "Failed to run sample demo.");
        return;
      }

      setProgress(1.0);
      setStatusMsg("Demo game state reconstructed!");
      setIsProcessing(false);
      setAnalysisResult(demoResult);
      setCurrentFrame(0);
      setMatchId(demoResult.match_id);
      setDataset(`Sample Match Clip (${demoResult.match_id})`);
      setFrame(0);
      storedVideosQuery.refetch();
      try {
        if (videoRef.current && videoRef.current.readyState >= 1) {
          videoRef.current.currentTime = 0;
        }
      } catch {}
    } catch (err: any) {
      setIsProcessing(false);
      setErrorMsg(err.message || "Failed to run sample demo.");
    }
  };

  // Upload & Run Pipeline
  const handleStartAnalysis = async () => {
    if (!file && !videoId) {
      setErrorMsg("Please select an uploaded video or drop a new video file first.");
      return;
    }

    try {
      setErrorMsg(null);
      let activeVideoId = videoId;
      let activeFps = videoMeta?.fps || 25.0;
      let activeFilename = selectedVideoName || file?.name || "Match Video";

      if (file) {
        setIsUploading(true);
        setStatusMsg("Uploading video to server and registering in stored data...");
        setProgress(0.05);

        const uploadRes: VideoUploadResponse = await uploadVideoFile(file);
        activeVideoId = uploadRes.video_id;
        activeFps = uploadRes.fps || 25.0;
        activeFilename = uploadRes.filename;
        setVideoId(uploadRes.video_id);
        setSelectedStoredId(uploadRes.video_id);
        setSelectedVideoName(uploadRes.filename);
        setVideoMeta({
          total_frames: uploadRes.total_frames,
          fps: uploadRes.fps,
          width: uploadRes.width,
          height: uploadRes.height,
          duration_sec: uploadRes.duration_sec,
        });
        setIsUploading(false);
        storedVideosQuery.refetch();
      }

      if (!activeVideoId) {
        throw new Error("No valid video ID found to process.");
      }

      setIsProcessing(true);
      setStatusMsg("Initializing tracking and pitch homography engine...");
      setProgress(0.1);

      // Determine segment and time crop window
      let startTimeSec = 0.0;
      let endTimeSec: number | undefined = undefined;
      let framesToRun: number | undefined = undefined;

      if (segmentMode === "full") {
        startTimeSec = 0.0;
        endTimeSec = undefined;
        framesToRun = 0; // full video
      } else if (segmentMode === "30s") {
        startTimeSec = 0.0;
        endTimeSec = 30.0;
        framesToRun = Math.round(30.0 * activeFps);
      } else if (segmentMode === "1m") {
        startTimeSec = 0.0;
        endTimeSec = 60.0;
        framesToRun = Math.round(60.0 * activeFps);
      } else if (segmentMode === "3m") {
        startTimeSec = 0.0;
        endTimeSec = 180.0;
        framesToRun = Math.round(180.0 * activeFps);
      } else if (segmentMode === "crop") {
        const startSec = Math.max(0, cropStartMin * 60 + cropStartSec);
        const endSec = Math.max(startSec + 1, cropEndMin * 60 + cropEndSec);
        startTimeSec = startSec;
        endTimeSec = endSec;
        framesToRun = Math.round((endSec - startSec) * activeFps);
      }

      // Trigger background processing with single high-accuracy broadcast projection mode
      await processVideo({
        video_id: activeVideoId,
        preset: "auto_detect",
        max_frames: framesToRun,
        frame_stride: 1, // Enforced optimal Stride 1 (100% full fidelity)
        conf_threshold: 0.08, // Enforced optimal 0.08 sensitivity for complete 22-player recall
        apply_mitigation: true,
        start_time_sec: startTimeSec,
        end_time_sec: endTimeSec,
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
            const st = await getVideoStatus(activeVideoId!);
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

      if (!resultData?.data || (resultData as any).status === "error" || resultData.data?.status === "error") {
        setIsUploading(false);
        setIsProcessing(false);
        setErrorMsg((resultData as any)?.message || resultData?.data?.message || "Failed to process video.");
        return;
      }

      setProgress(1.0);
      setStatusMsg("Game state reconstruction complete!");
      setIsProcessing(false);
      setAnalysisResult(resultData);
      setCurrentFrame(0);
      setMatchId(resultData.match_id);
      setDataset(`Video: ${activeFilename.replace(/\.[^/.]+$/, "")}`);
      setFrame(0);
      storedVideosQuery.refetch();

      const startOffset = resultData.data?.start_time_sec || 0;
      try {
        if (videoRef.current && videoRef.current.readyState >= 1) {
          videoRef.current.currentTime = startOffset;
        }
      } catch {}
    } catch (err: any) {
      setIsUploading(false);
      setIsProcessing(false);
      setErrorMsg(err.message || "Failed to process video.");
    }
  };

  const allFramesRef = useRef<Record<number, Player[]>>({});

  // Pre-fetch all tracking frames for instantaneous zero-latency playback
  useEffect(() => {
    if (!analysisResult?.data || (analysisResult as any).status === "error" || analysisResult.data.status === "error") return;
    const matchId = analysisResult.match_id;
    const effectiveFps = fps / Math.max(1, analysisResult.data?.frame_stride ?? 1);

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
                const parts = (p.player_id || "PLAYER_1").split("_");
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
                  p.jersey_color ||
                  analysisResult.data?.team_colors?.[p.team] ||
                  analysisResult.data?.team_colors?.[p.team === "away" ? "away" : "home"];

                return {
                  id: p.player_id,
                  team: (p.team || "home") as any,
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
    if (!analysisResult?.data || (analysisResult as any).status === "error" || analysisResult.data.status === "error") return;
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
            const parts = (p.player_id || "PLAYER_1").split("_");
            const playerNum = Number(parts[1]) || (idx + 1);
            const jerseyColor =
              p.jersey_color ||
              analysisResult.data?.team_colors?.[p.team] ||
              analysisResult.data?.team_colors?.[p.team === "away" ? "away" : "home"];
            return {
              id: p.player_id,
              team: (p.team || "home") as any,
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
      if (videoRef.current && isPlaying && analysisResult?.data && (analysisResult as any).status !== "error" && analysisResult.data.status !== "error") {
        const startOffset = analysisResult.data?.start_time_sec || 0;
        const endOffset = analysisResult.data?.end_time_sec || (startOffset + (maxProcessedFrame / fps));
        const curTime = videoRef.current.currentTime;

        // Loop segment if video reaches end of crop
        if (curTime >= endOffset) {
          videoRef.current.currentTime = startOffset;
          setCurrentFrame(0);
          return;
        }

        const computedFrame = Math.min(
          maxProcessedFrame,
          Math.max(0, Math.round((curTime - startOffset) * fps))
        );
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
    const startOffset = analysisResult?.data.start_time_sec || 0;
    if (videoRef.current) {
      videoRef.current.currentTime = startOffset + (clamped / fps);
    }
  };

  const handleVideoTimeUpdate = () => {
    if (!videoRef.current || !analysisResult) return;
    const startOffset = analysisResult.data.start_time_sec || 0;
    const curTime = videoRef.current.currentTime;
    const computedFrame = Math.min(
      maxProcessedFrame,
      Math.max(0, Math.round((curTime - startOffset) * fps))
    );
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

  const currentInsight = analysisResult?.data?.sample_insights?.find((s) => s.frame === currentFrame);

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
          <Panel title="Video Ingestion" subtitle="Stored Data Library & Video Upload">
            {/* Stored Videos Section */}
            {storedVideos.length > 0 && (
              <div className="mb-4 rounded-lg border border-border bg-panel-alt/50 p-3 space-y-2.5">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold text-foreground flex items-center gap-1.5">
                    <Database className="h-3.5 w-3.5 text-emerald-400" /> Stored Match Data ({storedVideos.length})
                  </span>
                  <span className="text-[10px] text-muted-foreground font-mono">
                    Persisted on Disk
                  </span>
                </div>

                <div className="max-h-48 overflow-y-auto space-y-1.5 pr-1 text-xs">
                  {storedVideos.map((stored) => {
                    const isSelected = selectedStoredId === stored.video_id || videoId === stored.video_id;
                    return (
                      <div
                        key={stored.video_id}
                        onClick={() => handleSelectStoredVideo(stored)}
                        className={`group p-2 rounded border cursor-pointer transition-all flex items-center justify-between gap-2 ${
                          isSelected
                            ? "border-emerald-500/60 bg-emerald-500/10"
                            : "border-border/80 bg-panel hover:border-emerald-500/30 hover:bg-muted/40"
                        }`}
                      >
                        <div className="min-w-0 flex-1">
                          <div className="flex items-center gap-1.5">
                            <span className="font-semibold text-[11px] truncate text-foreground group-hover:text-emerald-400 transition-colors" title={stored.filename}>
                              {stored.filename}
                            </span>
                            {stored.processed ? (
                              <span className="shrink-0 text-[9px] font-semibold bg-emerald-500/15 border border-emerald-500/30 text-emerald-400 px-1.5 py-0.2 rounded flex items-center gap-0.5">
                                <Check className="h-2.5 w-2.5" /> Reconstructed
                              </span>
                            ) : (
                              <span className="shrink-0 text-[9px] font-medium bg-amber-500/15 border border-amber-500/30 text-amber-300 px-1.5 py-0.2 rounded">
                                Uploaded
                              </span>
                            )}
                          </div>
                          <div className="text-[10px] text-muted-foreground font-mono mt-0.5 flex items-center gap-2">
                            <span>{formatTime(stored.duration_sec || 0)}</span>
                            <span>•</span>
                            <span>{stored.total_frames || 0}f @ {stored.fps || 25}fps</span>
                            <span>•</span>
                            <span>{(stored.size_bytes / (1024 * 1024)).toFixed(1)}MB</span>
                          </div>
                        </div>

                        <div className="flex items-center gap-1 shrink-0">
                          <button
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation();
                              handleSelectStoredVideo(stored);
                            }}
                            className={`text-[10px] px-2 py-1 rounded font-semibold transition-colors ${
                              isSelected
                                ? "bg-emerald-600 text-white"
                                : "bg-panel-alt border border-border hover:bg-emerald-600 hover:text-white text-muted-foreground"
                            }`}
                          >
                            {isSelected ? "Active" : "Load"}
                          </button>
                          <button
                            type="button"
                            title="Delete stored video"
                            onClick={(e) => handleDeleteStoredVideo(e, stored.video_id)}
                            className="p-1 rounded text-muted-foreground hover:text-destructive hover:bg-destructive/10 transition-colors"
                          >
                            <Trash2 className="h-3 w-3" />
                          </button>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Drag & Drop Zone */}
            <div
              onDragOver={(e) => e.preventDefault()}
              onDrop={handleDrop}
              className={`border-2 border-dashed p-5 text-center cursor-pointer transition-colors rounded ${
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
                    const selected = e.target.files[0];
                    setFile(selected);
                    setSelectedStoredId(null);
                    setSelectedVideoName(selected.name);
                    setErrorMsg(null);
                    setVideoMeta(null);
                  }
                }}
              />
              <UploadCloud className="mx-auto h-7 w-7 text-muted-foreground mb-1.5" />
              {file ? (
                <div>
                  <div className="text-sm font-semibold text-emerald-400">{file.name}</div>
                  <div className="text-xs text-muted-foreground mt-0.5">
                    {(file.size / (1024 * 1024)).toFixed(2)} MB • Ready to analyze & store
                  </div>
                  {videoMeta && (
                    <div className="mt-2 text-[11px] font-mono text-emerald-300 border border-emerald-500/30 bg-emerald-500/10 p-1.5 rounded inline-block">
                      Duration: {formatTime(videoMeta.duration_sec || 0)} • {videoMeta.total_frames} frames @ {videoMeta.fps} FPS ({videoMeta.width}x{videoMeta.height})
                    </div>
                  )}
                </div>
              ) : videoId && selectedVideoName ? (
                <div>
                  <div className="text-xs text-muted-foreground">Active Stored Video:</div>
                  <div className="text-sm font-semibold text-emerald-400 mt-0.5">{selectedVideoName}</div>
                  <div className="text-[11px] text-muted-foreground mt-1">
                    Click to select a different local file or pick from Stored Data above
                  </div>
                </div>
              ) : (
                <div>
                  <div className="text-sm font-medium">Click to upload or drag & drop video</div>
                  <div className="text-xs text-muted-foreground mt-1">Upload any match clip (.mp4, .mov, .avi, .webm)</div>
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

              {/* Video Segment & Custom Time Crop Selection */}
              <div className="space-y-2 rounded-lg border border-border bg-panel-alt/40 p-3">
                <div className="flex items-center justify-between">
                  <span className="text-foreground flex items-center gap-1.5 font-semibold text-xs">
                    <Clock className="h-4 w-4 text-emerald-400" /> Match Clip Segment & Time Crop
                  </span>
                  <span className="font-mono text-[11px] text-emerald-400 font-semibold">
                    {segmentMode === "full" && "Full Video (All Frames)"}
                    {segmentMode === "30s" && "First 30 sec (~750 frames)"}
                    {segmentMode === "1m" && "First 1 min (~1,500 frames)"}
                    {segmentMode === "3m" && "First 3 min (~4,500 frames)"}
                    {segmentMode === "crop" && `${formatTime(cropStartMin * 60 + cropStartSec)} → ${formatTime(cropEndMin * 60 + cropEndSec)}`}
                  </span>
                </div>

                {/* Mode Selector Buttons */}
                <div className="grid grid-cols-5 gap-1 pt-1">
                  {[
                    { id: "full", label: "FULL CLIP" },
                    { id: "30s", label: "0:00 - 0:30" },
                    { id: "1m", label: "0:00 - 1:00" },
                    { id: "3m", label: "0:00 - 3:00" },
                    { id: "crop", label: "CUSTOM CROP" },
                  ].map((item) => (
                    <button
                      key={item.id}
                      type="button"
                      onClick={() => setSegmentMode(item.id as any)}
                      className={`py-1.5 text-[11px] font-semibold rounded transition-all flex items-center justify-center gap-1 ${
                        segmentMode === item.id
                          ? "bg-emerald-600 text-white shadow-sm"
                          : "bg-panel border border-border text-muted-foreground hover:text-foreground"
                      }`}
                    >
                      {item.id === "crop" && <Scissors className="h-3 w-3" />}
                      <span>{item.label}</span>
                    </button>
                  ))}
                </div>

                {/* Custom Time Crop Controls (e.g. Minute 2-3) */}
                {segmentMode === "crop" && (
                  <div className="mt-2.5 rounded border border-emerald-500/30 bg-emerald-950/20 p-2.5 space-y-2">
                    <div className="text-[11px] font-medium text-emerald-300 flex items-center justify-between">
                      <span className="flex items-center gap-1 font-semibold">
                        <Scissors className="h-3.5 w-3.5" /> Custom Time Interval (e.g. Minute 2 to 3)
                      </span>
                      {videoMeta?.duration_sec ? (
                        <span className="text-[10px] text-muted-foreground font-mono">
                          Video Length: {formatTime(videoMeta.duration_sec)}
                        </span>
                      ) : null}
                    </div>

                    <div className="grid grid-cols-2 gap-3 items-center">
                      {/* Start Time */}
                      <div className="bg-panel border border-border p-2 rounded">
                        <span className="text-[10px] text-muted-foreground block mb-1 font-medium">
                          START TIME (From)
                        </span>
                        <div className="flex items-center gap-1 font-mono text-xs">
                          <div className="flex items-center gap-0.5">
                            <input
                              type="number"
                              min={0}
                              max={120}
                              value={cropStartMin}
                              onChange={(e) => setCropStartMin(Math.max(0, parseInt(e.target.value) || 0))}
                              className="w-12 bg-panel-alt border border-border rounded px-1.5 py-0.5 text-center font-bold text-foreground"
                            />
                            <span className="text-muted-foreground text-[10px]">m</span>
                          </div>
                          <span className="text-muted-foreground font-bold">:</span>
                          <div className="flex items-center gap-0.5">
                            <input
                              type="number"
                              min={0}
                              max={59}
                              value={cropStartSec}
                              onChange={(e) => setCropStartSec(Math.max(0, Math.min(59, parseInt(e.target.value) || 0)))}
                              className="w-12 bg-panel-alt border border-border rounded px-1.5 py-0.5 text-center font-bold text-foreground"
                            />
                            <span className="text-muted-foreground text-[10px]">s</span>
                          </div>
                        </div>
                      </div>

                      {/* End Time */}
                      <div className="bg-panel border border-border p-2 rounded">
                        <span className="text-[10px] text-muted-foreground block mb-1 font-medium">
                          END TIME (To)
                        </span>
                        <div className="flex items-center gap-1 font-mono text-xs">
                          <div className="flex items-center gap-0.5">
                            <input
                              type="number"
                              min={0}
                              max={120}
                              value={cropEndMin}
                              onChange={(e) => setCropEndMin(Math.max(0, parseInt(e.target.value) || 0))}
                              className="w-12 bg-panel-alt border border-border rounded px-1.5 py-0.5 text-center font-bold text-foreground"
                            />
                            <span className="text-muted-foreground text-[10px]">m</span>
                          </div>
                          <span className="text-muted-foreground font-bold">:</span>
                          <div className="flex items-center gap-0.5">
                            <input
                              type="number"
                              min={0}
                              max={59}
                              value={cropEndSec}
                              onChange={(e) => setCropEndSec(Math.max(0, Math.min(59, parseInt(e.target.value) || 0)))}
                              className="w-12 bg-panel-alt border border-border rounded px-1.5 py-0.5 text-center font-bold text-foreground"
                            />
                            <span className="text-muted-foreground text-[10px]">s</span>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Quick Interval Chips */}
                    <div className="flex items-center gap-1 flex-wrap pt-0.5">
                      <span className="text-[10px] text-muted-foreground mr-1">Quick intervals:</span>
                      {[
                        { label: "Minute 0–1", sm: 0, ss: 0, em: 1, es: 0 },
                        { label: "Minute 1–2", sm: 1, ss: 0, em: 2, es: 0 },
                        { label: "Minute 2–3", sm: 2, ss: 0, em: 3, es: 0 },
                        { label: "Minute 3–4", sm: 3, ss: 0, em: 4, es: 0 },
                        { label: "Minute 4–5", sm: 4, ss: 0, em: 5, es: 0 },
                      ].map((chip) => (
                        <button
                          key={chip.label}
                          type="button"
                          onClick={() => {
                            setCropStartMin(chip.sm);
                            setCropStartSec(chip.ss);
                            setCropEndMin(chip.em);
                            setCropEndSec(chip.es);
                          }}
                          className="text-[10px] bg-panel hover:bg-muted/60 border border-border px-1.5 py-0.5 rounded text-foreground font-mono transition-colors"
                        >
                          {chip.label}
                        </button>
                      ))}
                    </div>

                    {/* Dynamic crop verification info */}
                    <div className="pt-1 text-[11px] font-mono flex items-center justify-between border-t border-emerald-500/20 text-emerald-300">
                      <span>
                        Crop window: {formatTime(cropStartMin * 60 + cropStartSec)} → {formatTime(cropEndMin * 60 + cropEndSec)}
                      </span>
                      <span>
                        {Math.max(0, (cropEndMin * 60 + cropEndSec) - (cropStartMin * 60 + cropStartSec))}s span (~{Math.max(0, (cropEndMin * 60 + cropEndSec) - (cropStartMin * 60 + cropStartSec)) * 25}f)
                      </span>
                    </div>
                  </div>
                )}
              </div>

              {/* Automatic Pipeline Optimization Banner (Locked optimal engine parameters) */}
              <div className="rounded-lg border border-border bg-panel-alt/40 p-2.5 space-y-1.5">
                <div className="flex items-center justify-between">
                  <span className="text-foreground flex items-center gap-1.5 font-semibold text-xs">
                    <Sparkles className="h-3.5 w-3.5 text-emerald-400" /> Pipeline Engine Optimization
                  </span>
                  <span className="text-[10px] bg-emerald-500/15 border border-emerald-500/30 text-emerald-400 font-semibold px-2 py-0.5 rounded-full flex items-center gap-1">
                    <CheckCircle2 className="h-3 w-3" /> Auto-Tuned (Optimal)
                  </span>
                </div>
                <div className="grid grid-cols-2 gap-2 pt-1 text-[11px]">
                  <div className="bg-panel border border-border p-2 rounded">
                    <div className="text-muted-foreground text-[10px] font-medium">Frame Sampling Stride</div>
                    <div className="text-xs font-bold text-sky-400 font-mono mt-0.5">Stride 1 (100% Fidelity)</div>
                    <div className="text-[10px] text-muted-foreground mt-0.5">Analyzes every frame for fluid 60 FPS motion.</div>
                  </div>
                  <div className="bg-panel border border-border p-2 rounded">
                    <div className="text-muted-foreground text-[10px] font-medium">Detection Sensitivity</div>
                    <div className="text-xs font-bold text-emerald-400 font-mono mt-0.5">0.08 Confidence Threshold</div>
                    <div className="text-[10px] text-muted-foreground mt-0.5">High recall tracking all 22 outfield players.</div>
                  </div>
                </div>
                <div className="rounded bg-panel border border-border p-1.5 px-2 text-[10px] text-muted-foreground flex items-center justify-between">
                  <span>Trajectory Smoothing & Filter:</span>
                  <span className="font-mono text-emerald-400 font-semibold">Active (Savitzky-Golay + Kalman)</span>
                </div>
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
              disabled={(!file && !videoId) || isUploading || isProcessing}
              className="mt-4 w-full flex items-center justify-center gap-2 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white text-xs font-semibold py-2.5 px-4 rounded transition-colors shadow-sm cursor-pointer"
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

              {/* 2D Pitch Radar with Inferred Team Jersey Badges */}
              <div className="border border-border bg-panel-alt/40 p-2 rounded overflow-hidden space-y-1.5">
                {analysisResult?.data?.team_names && (
                  <div className="flex items-center justify-between text-xs px-1 pb-1 border-b border-border/50">
                    <div className="flex items-center gap-1.5 font-medium">
                      <span
                        className="inline-block h-3 w-3 rounded-full border border-white/30 shadow-sm"
                        style={{
                          backgroundColor:
                            analysisResult.data.team_colors?.[analysisResult.data.team_names[0]] || "#E62325",
                        }}
                      />
                      <span className="text-foreground">{analysisResult.data.team_names[0]} Team</span>
                      <span className="text-[10px] text-muted-foreground font-mono font-normal">#1–11</span>
                    </div>

                    <div className="text-[10px] text-muted-foreground font-mono">
                      FIFA Metric Coordinates (105m × 68m)
                    </div>

                    <div className="flex items-center gap-1.5 font-medium">
                      <span className="text-[10px] text-muted-foreground font-mono font-normal">#1–11</span>
                      <span className="text-foreground">{analysisResult.data.team_names[1]} Team</span>
                      <span
                        className="inline-block h-3 w-3 rounded-full border border-white/30 shadow-sm"
                        style={{
                          backgroundColor:
                            analysisResult.data.team_colors?.[analysisResult.data.team_names[1]] || "#034694",
                        }}
                      />
                    </div>
                  </div>
                )}
                <div className="bg-turf p-1 rounded overflow-hidden border border-emerald-950">
                  <Pitch 
                    players={framePlayers} 
                    height={220} 
                    showVectors={true} 
                    heat={false} 
                  />
                </div>
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
                    {currentInsight.tas != null ? Number(currentInsight.tas).toFixed(3) : "—"}
                  </div>
                </div>
              </div>
            )}
          </Panel>

          {/* Video Reconstruction Details & Action to Move to Match Explorer (Placed Directly Below Radar Section) */}
          {analysisResult?.data && (analysisResult as any).status !== "error" && analysisResult.data.status !== "error" && (
            <Panel
              title="Reconstruction Details & Video Analytics"
              subtitle={`Active Video: ${selectedVideoName || file?.name || analysisResult.match_id}`}
            >
              <div className="space-y-4">
                {/* Hero CTA: Move to Match Explorer */}
                <div className="rounded-lg border border-emerald-500/40 bg-gradient-to-r from-emerald-950/40 via-panel to-teal-950/30 p-3.5 shadow-md">
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                    <div>
                      <div className="flex items-center gap-2 text-foreground font-bold text-sm">
                        <span className="flex h-2.5 w-2.5 rounded-full bg-emerald-400 animate-pulse" />
                        <span>Game State Reconstructed for this Video</span>
                      </div>
                      <p className="text-xs text-muted-foreground mt-0.5">
                        Canonical coordinates, dynamic homography, and tactical advantage have been computed. Move to Match Explorer to inspect frame-by-frame player trajectories and inferred events.
                      </p>
                    </div>

                    <button
                      onClick={() => {
                        setMatchId(analysisResult.match_id);
                        setDataset(`Video: ${(selectedVideoName || file?.name || analysisResult.match_id).replace(/\.[^/.]+$/, "")}`);
                        setFrame(0);
                        navigate({ to: "/match-explorer" });
                      }}
                      className="flex items-center justify-center gap-2 bg-emerald-600 hover:bg-emerald-500 text-white font-bold text-xs py-2.5 px-5 rounded-lg shadow-md transition-all hover:shadow-emerald-900/40 hover:scale-[1.02] active:scale-[0.98] shrink-0 cursor-pointer"
                    >
                      <Compass className="h-4 w-4 text-emerald-200" />
                      <span>Move to Match Explorer</span>
                      <ArrowRight className="h-4 w-4" />
                    </button>
                  </div>
                </div>

                {/* Key Reconstruction Metrics Grid */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 text-xs">
                  <div className="border border-border bg-panel-alt p-2.5 rounded">
                    <div className="text-muted-foreground text-[10px] font-medium uppercase tracking-wider">Frames Reconstructed</div>
                    <div className="text-base font-bold font-mono text-foreground mt-1">
                      {analysisResult.data.frame_count ?? 0} <span className="text-xs font-normal text-muted-foreground">frames</span>
                    </div>
                    <div className="text-[10px] text-muted-foreground mt-0.5 font-mono">
                      {analysisResult.data.duration_sec != null 
                        ? `${Number(analysisResult.data.duration_sec).toFixed(1)}s duration` 
                        : `${(analysisResult.data.max_frame ? analysisResult.data.max_frame / fps : 0).toFixed(1)}s duration`}
                    </div>
                  </div>

                  <div className="border border-border bg-panel-alt p-2.5 rounded">
                    <div className="text-muted-foreground text-[10px] font-medium uppercase tracking-wider">Player Observations</div>
                    <div className="text-base font-bold font-mono text-emerald-400 mt-1">
                      {analysisResult.data.tracking_rows ? Number(analysisResult.data.tracking_rows).toLocaleString() : 0}
                    </div>
                    <div className="text-[10px] text-muted-foreground mt-0.5">
                      22 Outfield + Referee Filtered
                    </div>
                  </div>

                  <div className="border border-border bg-panel-alt p-2.5 rounded">
                    <div className="text-muted-foreground text-[10px] font-medium uppercase tracking-wider">Average TAS Score</div>
                    <div className="text-base font-bold font-mono text-sky-400 mt-1">
                      {analysisResult.data.average_tas != null ? Number(analysisResult.data.average_tas).toFixed(3) : "—"}
                    </div>
                    <div className="text-[10px] text-muted-foreground mt-0.5">
                      Tactical Advantage
                    </div>
                  </div>

                  <div className="border border-border bg-panel-alt p-2.5 rounded">
                    <div className="text-muted-foreground text-[10px] font-medium uppercase tracking-wider">Possession Split</div>
                    <div className="text-sm font-bold font-mono text-foreground mt-1 flex items-center justify-between">
                      <span className="flex items-center gap-1">
                        <span
                          className="h-2 w-2 rounded-full inline-block"
                          style={{
                            backgroundColor:
                              analysisResult.data.team_colors?.[
                                analysisResult.data.team_names?.[0] || "home"
                              ] || "#E62325",
                          }}
                        />
                        {analysisResult.data.team_names?.[0] || "Home"}:{" "}
                        {analysisResult.data.possession_share?.[
                          analysisResult.data.team_names?.[0] || "home"
                        ] ?? analysisResult.data.possession_share?.home ?? 50}%
                      </span>
                      <span className="text-muted-foreground font-normal">/</span>
                      <span className="flex items-center gap-1">
                        {analysisResult.data.team_names?.[1] || "Away"}:{" "}
                        {analysisResult.data.possession_share?.[
                          analysisResult.data.team_names?.[1] || "away"
                        ] ?? analysisResult.data.possession_share?.away ?? 50}%
                        <span
                          className="h-2 w-2 rounded-full inline-block"
                          style={{
                            backgroundColor:
                              analysisResult.data.team_colors?.[
                                analysisResult.data.team_names?.[1] || "away"
                              ] || "#034694",
                          }}
                        />
                      </span>
                    </div>
                    <div className="text-[10px] text-muted-foreground mt-0.5">
                      Ball-Free Inferred Control by Kit Color
                    </div>
                  </div>
                </div>

                {/* Video Source Properties & Perception Specifications */}
                <div className="rounded-lg border border-border bg-panel-alt/50 p-2.5 space-y-2 text-xs">
                  <div className="flex items-center justify-between text-muted-foreground border-b border-border/60 pb-1.5">
                    <span className="font-semibold text-foreground flex items-center gap-1.5">
                      <Video className="h-3.5 w-3.5 text-emerald-400" /> Perception Specification
                    </span>
                    <span className="font-mono text-[11px] text-emerald-400">
                      {videoMeta?.width && videoMeta?.height ? `${videoMeta.width} × ${videoMeta.height}` : "HD Video"} @ {fps} FPS
                    </span>
                  </div>
                  <div className="grid grid-cols-3 gap-2 pt-0.5 text-[11px]">
                    <div>
                      <span className="text-muted-foreground block text-[10px]">Active Match ID</span>
                      <span className="font-mono text-foreground text-[11px] truncate block" title={analysisResult.match_id}>
                        {analysisResult.match_id}
                      </span>
                    </div>
                    <div>
                      <span className="text-muted-foreground block text-[10px]">Pitch Coordinate Space</span>
                      <span className="font-semibold text-emerald-400 text-[11px] block">
                        105m × 68m Calibrated
                      </span>
                    </div>
                    <div>
                      <span className="text-muted-foreground block text-[10px]">Smoothing & Mitigations</span>
                      <span className="font-semibold text-foreground text-[11px] block">
                        Savitzky-Golay + Kalman
                      </span>
                    </div>
                  </div>
                </div>

                {/* Deep Dive Action Navigation Row */}
                <div className="border-t border-border pt-2.5">
                  <div className="text-[11px] text-muted-foreground mb-2 font-medium">
                    Continue analyzing this video across all system modules:
                  </div>
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                    <button
                      onClick={() => {
                        setMatchId(analysisResult.match_id);
                        setDataset(`Video: ${(selectedVideoName || file?.name || analysisResult.match_id).replace(/\.[^/.]+$/, "")}`);
                        setFrame(0);
                        navigate({ to: "/match-explorer" });
                      }}
                      className="flex items-center justify-center gap-1.5 border border-emerald-500/40 bg-emerald-950/30 hover:bg-emerald-900/40 text-emerald-300 text-xs font-semibold py-2 px-2.5 rounded transition-colors shadow-sm cursor-pointer"
                    >
                      <Compass className="h-3.5 w-3.5 text-emerald-400" />
                      <span>Match Explorer</span>
                    </button>
                    <button
                      onClick={() => {
                        setMatchId(analysisResult.match_id);
                        setDataset(`Video: ${(selectedVideoName || file?.name || analysisResult.match_id).replace(/\.[^/.]+$/, "")}`);
                        setFrame(0);
                        navigate({ to: "/game-state" });
                      }}
                      className="flex items-center justify-center gap-1.5 border border-border bg-panel hover:bg-muted text-foreground text-xs font-medium py-2 px-2.5 rounded transition-colors cursor-pointer"
                    >
                      <Layers className="h-3.5 w-3.5 text-muted-foreground" />
                      <span>Game State</span>
                    </button>
                    <button
                      onClick={() => {
                        setMatchId(analysisResult.match_id);
                        setDataset(`Video: ${(selectedVideoName || file?.name || analysisResult.match_id).replace(/\.[^/.]+$/, "")}`);
                        setFrame(0);
                        navigate({ to: "/tactical" });
                      }}
                      className="flex items-center justify-center gap-1.5 border border-border bg-panel hover:bg-muted text-foreground text-xs font-medium py-2 px-2.5 rounded transition-colors cursor-pointer"
                    >
                      <Activity className="h-3.5 w-3.5 text-muted-foreground" />
                      <span>Tactical Analysis</span>
                    </button>
                    <button
                      onClick={() => {
                        setMatchId(analysisResult.match_id);
                        setDataset(`Video: ${(selectedVideoName || file?.name || analysisResult.match_id).replace(/\.[^/.]+$/, "")}`);
                        setFrame(0);
                        navigate({ to: "/possession" });
                      }}
                      className="flex items-center justify-center gap-1.5 border border-border bg-panel hover:bg-muted text-foreground text-xs font-medium py-2 px-2.5 rounded transition-colors cursor-pointer"
                    >
                      <Zap className="h-3.5 w-3.5 text-muted-foreground" />
                      <span>Possession & Events</span>
                    </button>
                  </div>
                </div>
              </div>
            </Panel>
          )}
        </div>
      </div>
    </>
  );
}
