import { apiFetch, apiBaseUrl } from "./client";

export interface VideoUploadResponse {
  status: string;
  video_id: string;
  filename: string;
  size_bytes: number;
  match_id: string;
}

export interface VideoProcessResponse {
  status: string;
  match_id: string;
  data: {
    match_id: string;
    status: string;
    frame_count: number;
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
}

export async function uploadVideoFile(file: File): Promise<VideoUploadResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${apiBaseUrl}/video/upload`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`Upload failed with status ${response.status}`);
  }

  return response.json();
}

export function processVideo(params: {
  video_id: string;
  preset?: string;
  max_frames?: number;
  frame_stride?: number;
  apply_mitigation?: boolean;
}): Promise<VideoProcessResponse> {
  return apiFetch<VideoProcessResponse>("/video/process", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      video_id: params.video_id,
      preset: params.preset ?? "main_broadcast",
      max_frames: params.max_frames ?? 150,
      frame_stride: params.frame_stride ?? 1,
      apply_mitigation: params.apply_mitigation ?? true,
    }),
  });
}

export function getVideoStatus(videoId: string): Promise<VideoStatusResponse> {
  return apiFetch<VideoStatusResponse>(`/video/status/${videoId}`);
}

export function getVideoStreamUrl(videoId: string): string {
  return `${apiBaseUrl}/video/stream/${videoId}`;
}
