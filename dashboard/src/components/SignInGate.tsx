import { useState, type FormEvent } from "react";
import { KeyRound } from "lucide-react";
import { browserSignIn, UnauthorizedError } from "../lib/api";
import { useDashboardStore } from "../state/store";

// Shown when the server runs with an API key and this browser has no valid
// session cookie (opened directly, another browser, or the 12-hour cookie
// expired). The key is exchanged for the cookie on a same-origin POST and
// the page reloads with the session in place.
export function SignInGate() {
  const authRequired = useDashboardStore((s) => s.authRequired);
  const [apiKey, setApiKey] = useState("");
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (!authRequired) return null;

  async function submit(event: FormEvent) {
    event.preventDefault();
    const key = apiKey.trim();
    if (!key || pending) return;
    setPending(true);
    setError(null);
    try {
      await browserSignIn(key);
      window.location.reload();
    } catch (err) {
      setError(
        err instanceof UnauthorizedError
          ? "That key was not accepted."
          : `Sign-in failed: ${err instanceof Error ? err.message : String(err)}`,
      );
      setPending(false);
    }
  }

  return (
    <div className="fixed inset-0 z-50 bg-black/60 grid place-items-center p-4">
      <form
        onSubmit={submit}
        className="bg-[var(--bg-card)] border border-[var(--border-soft)] rounded-2xl p-6 max-w-md w-full"
      >
        <div className="flex items-center gap-3 mb-3">
          <span className="inline-flex items-center justify-center w-8 h-8 rounded-full bg-[var(--accent)]/15 text-[var(--accent)]">
            <KeyRound className="size-4" />
          </span>
          <h2 className="text-base font-semibold text-[var(--text-primary)]">
            Sign in to the dashboard
          </h2>
        </div>
        <p className="text-sm text-[var(--text-muted)] leading-relaxed">
          This server was started with an API key. Paste it to continue, or open the
          dashboard link the server printed at startup.
        </p>
        <label className="block mt-4">
          <span className="text-xs text-[var(--text-muted)]">API key</span>
          <input
            type="password"
            autoComplete="off"
            autoFocus
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            className="mt-1 w-full bg-[var(--bg-elevated)] border border-[var(--border-soft)] rounded px-3 py-2 text-sm text-[var(--text-primary)] font-mono"
          />
        </label>
        {error ? <div className="mt-2 text-xs text-[var(--accent-hot)]">{error}</div> : null}
        <button
          type="submit"
          disabled={pending || apiKey.trim().length === 0}
          className="mt-4 w-full rounded-md bg-[var(--accent)] text-black text-sm font-semibold py-2 disabled:opacity-50"
        >
          {pending ? "Signing in…" : "Sign in"}
        </button>
      </form>
    </div>
  );
}
