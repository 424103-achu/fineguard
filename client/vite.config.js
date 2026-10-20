import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    // In development, forward /api to the Express server
    proxy: { "/api": process.env.API_PROXY || "http://localhost:5000" },
  },
});
