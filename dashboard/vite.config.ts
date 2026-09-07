import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import path from "node:path";

const MTPLX_BACKEND = process.env.MTPLX_BACKEND_URL ?? "http://127.0.0.1:8000";

// The server judges a browser request same-origin when its `Origin` matches
// the `Host` it arrived with. Keeping the dev page's Host on proxied requests
// (changeOrigin: false) makes `http://localhost:5173` pages same-origin to the
// backend, so dev-mode POSTs and /admin calls pass without an allowlist.
const proxyTarget = {
  target: MTPLX_BACKEND,
  changeOrigin: false,
  secure: false,
  ws: true,
};

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "src"),
    },
  },
  // Vite is invoked from `dashboard/`; bundle into the python package
  // so `python -m build` ships the SPA inside the wheel.
  build: {
    outDir: path.resolve(__dirname, "../mtplx/dashboard/_static"),
    emptyOutDir: true,
    sourcemap: false,
    chunkSizeWarningLimit: 1500,
  },
  // When served through `mtplx serve`, the SPA lives under `/dashboard/`.
  // `base` keeps asset URLs relative-friendly so the same bundle works at
  // either the dev origin (`/`) or the mounted prefix (`/dashboard/`).
  base: "./",
  server: {
    port: 5173,
    strictPort: false,
    proxy: {
      "/v1": proxyTarget,
      "/health": proxyTarget,
      "/metrics": proxyTarget,
      "/admin": proxyTarget,
      // Browser sign-in (POST /mtplx/browser-auth from the sign-in gate).
      "/mtplx": proxyTarget,
    },
  },
});
