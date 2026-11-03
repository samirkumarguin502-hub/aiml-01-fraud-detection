import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// In development the browser talks to Vite, which proxies /api to the local FastAPI process,
// so no CORS configuration is needed and no secrets or API keys exist in the frontend.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: { "/api": { target: process.env.VITE_API_PROXY ?? "http://127.0.0.1:8000", changeOrigin: false } } },
  test: { environment: "jsdom", globals: true, setupFiles: ["./src/setupTests.ts"], include: ["src/**/*.test.tsx"] },
});
