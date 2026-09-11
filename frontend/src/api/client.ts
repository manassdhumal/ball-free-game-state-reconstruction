const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000/api").replace(
  /\/$/,
  "",
);

export class ApiError extends Error {
  constructor(
    public readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

export async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: { Accept: "application/json", ...init?.headers },
  });
  if (!response.ok) {
    const body = (await response.json().catch(() => null)) as {
      detail?: { message?: string };
    } | null;
    throw new ApiError(
      response.status,
      body?.detail?.message ?? `API request failed (${response.status})`,
    );
  }
  return response.json() as Promise<T>;
}

export const apiBaseUrl = API_BASE_URL;

export const getHealth = () =>
  apiFetch<{
    application: string;
    backend_status: string;
    data_availability: string;
    version: string;
  }>("/health");
