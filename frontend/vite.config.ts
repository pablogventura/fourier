import react from "@vitejs/plugin-react"
import { defineConfig } from "vitest/config"

export default defineConfig({
  base: process.env.GITHUB_PAGES === "true" ? "/fourier/" : "/",
  // Plotly lee `global` al cargar. En el navegador ese nombre no existe.
  define: {
    global: "globalThis",
  },
  plugins: [react()],
  server: {
    proxy: {
      "/api": "http://127.0.0.1:8000",
    },
  },
  test: {
    environment: "node",
    include: ["src/**/*.test.ts"],
  },
})
