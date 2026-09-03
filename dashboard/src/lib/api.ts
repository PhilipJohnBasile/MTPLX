import type {
  DashboardSnapshot,
  HealthPayload,
  MutableSettings,
  PrefillHistoryPayload,
  SessionsPayload,
} from "./types";

const BASE = "";  // same-origin; Vite's dev proxy handles `/v1`, `/admin`, `/health`, `/metrics`.

async function getJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    ...init,
    headers: {
      Accept: "application/json",
      ...(init?.headers ?? {}),
    },
  });
  if (!response.ok) {
    const text = await response.text().catch(() => "");
    throw new Error(`${response.status} ${response.statusText}: ${text || path}`);
  }
  return response.json() as Promise<T>;
}

async function postJson<T>(path: string, body: unknown): Promise<T> {
  return getJson<T>(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body ?? {}),
  });
}

export const api = {
  getHealth: () => getJson<HealthPayload>("/health"),
  getMetrics: () =>
    getJson<{ latest: DashboardSnapshot["latest"]; recent: DashboardSnapshot["recent"] }>(
      "/metrics",
    ),
  getSessions: () => getJson<SessionsPayload>("/admin/sessions"),
  getPrefillHistory: () => getJson<PrefillHistoryPayload>("/v1/mtplx/prefill_history"),
  getSnapshot: () => getJson<DashboardSnapshot>("/v1/mtplx/snapshot"),
  // The response is the server's settings payload after the write: the
  // mutable keys at the top level (already clamped/normalised), plus
  // `applied` listing what this write changed.
  postSettings: (payload: Partial<MutableSettings>) =>
    postJson<MutableSettings & { ok: boolean; applied?: Partial<MutableSettings> }>(
      "/v1/mtplx/settings",
      payload,
    ),
  postCancel: (requestId: string) =>
    postJson<{ ok: boolean; cancelled: boolean; active_requests: number }>(
      `/v1/mtplx/cancel/${encodeURIComponent(requestId)}`,
      {},
    ),
  postClearSession: (sessionId: string) =>
    postJson<Record<string, unknown>>(
      `/admin/sessions/${encodeURIComponent(sessionId)}/clear`,
      {},
    ),
  postClearCache: () => postJson<Record<string, unknown>>("/admin/cache/clear", {}),
};

export type Api = typeof api;
