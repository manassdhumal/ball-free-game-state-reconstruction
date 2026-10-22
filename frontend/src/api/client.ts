function resolveApiBaseUrl(): string {
  if (import.meta.env.VITE_API_BASE_URL) {
    return (import.meta.env.VITE_API_BASE_URL as string).replace(/\/$/, "");
  }
  if (typeof window !== "undefined" && window.location.hostname) {
    return `http://${window.location.hostname}:8000/api`;
  }
  return "http://localhost:8000/api";
}

const API_BASE_URL = resolveApiBaseUrl();

export class ApiError extends Error {
  constructor(
    public readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

export async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const baseUrl = resolveApiBaseUrl();
  const url = `${baseUrl}${path}`;
  try {
    const response = await fetch(url, {
      ...init,
      headers: { Accept: "application/json", ...init?.headers },
    });
    if (!response.ok) {
      const body = (await response.json().catch(() => null)) as {
        detail?: { message?: string } | string;
      } | null;
      const detailMsg = typeof body?.detail === "string" 
        ? body.detail 
        : body?.detail?.message;
      throw new ApiError(
        response.status,
        detailMsg ?? `API request failed (${response.status})`,
      );
    }
    return response.json() as Promise<T>;
  } catch (err: any) {
    if (err instanceof ApiError) {
      throw err;
    }
    throw new Error(
      `Cannot connect to backend server at ${baseUrl}. Ensure backend is running. (${err.message || "Failed to fetch"})`
    );
  }
}

export const apiBaseUrl = API_BASE_URL;

export const getHealth = () =>
  apiFetch<{
    application: string;
    backend_status: string;
    data_availability: string;
    version: string;
  }>("/health");
