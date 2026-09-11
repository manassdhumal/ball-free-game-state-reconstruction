import { apiFetch } from "./client";
import type { GsrStatus } from "./types";
export const getGsrStatus = () => apiFetch<GsrStatus>("/gsr/status");
