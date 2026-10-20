import express from "express";
import cors from "cors";
import analyzeRouter from "./routes/analyze.js";

export function createApp({ pool, mlUrl, http, clientOrigin }) {
  const app = express();
  app.use(cors({ origin: (clientOrigin || "http://localhost:5173").split(",") }));
  app.use(express.json({ limit: "2mb" }));
  app.use("/api", analyzeRouter({ pool, mlUrl, http }));

  app.use((err, _req, res, _next) => {
    if (err.type === "entity.too.large") {
      return res.status(413).json({ detail: "Request too large" });
    }
    if (err.type === "entity.parse.failed") {
      return res.status(400).json({ detail: "Invalid JSON" });
    }
    console.error(err);
    res.status(500).json({ detail: "Internal server error" });
  });
  return app;
}
