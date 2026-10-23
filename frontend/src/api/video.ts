import { apiFetch, apiBaseUrl } from "./client";

export interface VideoUploadResponse {
  status: string;
  video_id: string;
  filename: string;
  size_bytes: number;
  match_id: string;
  total_frames?: number;
  fps?: number;
  width?: number;
  height?: number;
  duration_sec?: number;
}

export interface VideoProcessResponse {
  status: string;
  match_id: string;
  data: {
    match_id: string;
    status: string;
    frame_count: number;
    min_frame?: number;
    max_frame?: number;
    fps?: number;
    duration_sec?: number;
    start_time_sec?: number;
    end_time_sec?: number;
    frame_stride?: number;
    total_video_frames?: number;
    team_colors?: { home: string; away: string };
    tracking_rows: number;
    tracking_file: string;
    possession_share: { home: number; away: number };
    average_tas: number;
    sample_insights: Array<{
      frame: number;
      possessing_team: string;
      possessor_player_id: string | null;
      tas: number;
    }>;
  };
}

export interface VideoStatusResponse {
  progress: number;
  status: string;
  completed?: boolean;
  error?: string | null;
  data?: VideoProcessResponse | null;
}

export async function uploadVideoFile(file: File): Promise<VideoUploadResponse> {
  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch(`${apiBaseUrl}/video/upload`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      let detail = "";
      try {
        const json = await response.json();
        detail = json.detail?.message || json.detail || JSON.stringify(json);
      } catch {
        detail = await response.text().catch(() => "");
      }
      throw new Error(`Upload failed (${response.status}): ${detail || response.statusText}`);
    }

    return response.json();
  } catch (err: any) {
    if (err.message && err.message.includes("Upload failed")) {
      throw err;
    }
    throw new Error(
      `Failed to upload video to server at ${apiBaseUrl}. Please verify backend is running on port 8000. (${err.message})`
    );
  }
}

export function processVideo(params: {
  video_id: string;
  preset?: string | undefined;
  max_frames?: number | undefined;
  frame_stride?: number | undefined;
  conf_threshold?: number | undefined;
  apply_mitigation?: boolean | undefined;
  custom_corners?: Array<[number, number]> | undefined;
  start_time_sec?: number | undefined;
  end_time_sec?: number | undefined;
}): Promise<{ status: string; video_id: string; match_id: string; message: string }> {
  return apiFetch("/video/process", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      video_id: params.video_id,
      preset: params.preset ?? "auto_detect",
      max_frames: params.max_frames,
      frame_stride: params.frame_stride ?? 1,
      conf_threshold: params.conf_threshold ?? 0.08,
      apply_mitigation: params.apply_mitigation ?? true,
      custom_corners: params.custom_corners,
      start_time_sec: params.start_time_sec ?? 0.0,
      end_time_sec: params.end_time_sec,
    }),
  });
}

export function runSampleDemo(): Promise<{ status: string; video_id: string; match_id: string; message: string }> {
  return apiFetch("/video/sample-demo", {
    method: "POST",
  });
}

export function getVideoStatus(videoId: string): Promise<VideoStatusResponse> {
  return apiFetch<VideoStatusResponse>(`/video/status/${videoId}`);
}

export function getVideoStreamUrl(videoId: string): string {
  return `${apiBaseUrl}/video/stream/${videoId}`;
}
