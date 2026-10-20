import { Router } from "express";
import axios from "axios";
import multer from "multer";
import FormData from "form-data";

const upload = multer({ limits: { fileSize: 10 * 1024 * 1024 } });

/**
 * @param {{ pool: import("pg").Pool, mlUrl: string, http?: import("axios").AxiosInstance }} deps
 */
export default function analyzeRouter({ pool, mlUrl, http = axios }) {
  const router = Router();

  async function saveAnalysis(r) {
    const { rows } = await pool.query(
      `INSERT INTO analyses (source, service_name, risk_score, grade, n_sentences, n_flagged, result)
       VALUES ($1, $2, $3, $4, $5, $6, $7)
       RETURNING id, created_at`,
      [r.source, r.service_name, r.risk_score, r.grade, r.n_sentences, r.n_flagged, r]
    );
    return rows[0];
  }

  async function respond(res, data) {
    const saved = await saveAnalysis(data);
    res.json({ ...data, id: saved.id, created_at: saved.created_at });
  }

  function forwardError(err, res, next) {
    if (err.response) {
      return res.status(err.response.status).json(err.response.data);
    }
    if (err.code === "ECONNREFUSED" || err.code === "ECONNABORTED" || err.code === "ENOTFOUND") {
      return res.status(502).json({ detail: "The ML service is not reachable" });
    }
    next(err);
  }

  router.get("/health", async (_req, res) => {
    const out = { status: "ok", db: false, ml: null };
    try {
      await pool.query("SELECT 1");
      out.db = true;
    } catch { /* reported as db: false */ }
    try {
      out.ml = (await http.get(`${mlUrl}/health`, { timeout: 5000 })).data;
    } catch { /* reported as ml: null */ }
    res.json(out);
  });

  router.post("/analyze/text", async (req, res, next) => {
    try {
      const { data } = await http.post(`${mlUrl}/analyze/text`, req.body, { timeout: 120000 });
      await respond(res, data);
    } catch (err) { forwardError(err, res, next); }
  });

  router.post("/analyze/url", async (req, res, next) => {
    try {
      const { data } = await http.post(`${mlUrl}/analyze/url`, req.body, { timeout: 120000 });
      await respond(res, data);
    } catch (err) { forwardError(err, res, next); }
  });

  router.post("/analyze/pdf", (req, res, next) => {
    upload.single("file")(req, res, (err) => {
      if (err?.code === "LIMIT_FILE_SIZE") {
        return res.status(413).json({ detail: "PDF larger than 10 MB" });
      }
      if (err) return next(err);
      next();
    });
  }, async (req, res, next) => {
    try {
      if (!req.file) return res.status(400).json({ detail: "No file uploaded" });
      const form = new FormData();
      form.append("file", req.file.buffer, {
        filename: req.file.originalname, contentType: req.file.mimetype,
      });
      if (req.body.service_name) form.append("service_name", req.body.service_name);
      const { data } = await http.post(`${mlUrl}/analyze/pdf`, form, {
        headers: form.getHeaders(), timeout: 180000,
      });
      await respond(res, data);
    } catch (err) { forwardError(err, res, next); }
  });

  router.get("/analyses", async (req, res, next) => {
    try {
      const limit = Math.min(Math.max(parseInt(req.query.limit, 10) || 50, 1), 200);
      const { rows } = await pool.query(
        `SELECT id, source, service_name, risk_score::float AS risk_score, grade,
                n_sentences, n_flagged, created_at
         FROM analyses ORDER BY created_at DESC LIMIT $1`, [limit]);
      res.json(rows);
    } catch (err) { next(err); }
  });

  router.get("/analyses/:id", async (req, res, next) => {
    try {
      const id = Number(req.params.id);
      if (!Number.isInteger(id) || id < 1) return res.status(404).json({ detail: "Not found" });
      const { rows } = await pool.query(
        `SELECT id, result, created_at FROM analyses WHERE id = $1`, [id]);
      if (!rows.length) return res.status(404).json({ detail: "Not found" });
      res.json({ ...rows[0].result, id: rows[0].id, created_at: rows[0].created_at });
    } catch (err) { next(err); }
  });

  router.delete("/analyses/:id", async (req, res, next) => {
    try {
      const id = Number(req.params.id);
      if (!Number.isInteger(id) || id < 1) return res.status(404).json({ detail: "Not found" });
      const { rowCount } = await pool.query(`DELETE FROM analyses WHERE id = $1`, [id]);
      if (!rowCount) return res.status(404).json({ detail: "Not found" });
      res.status(204).end();
    } catch (err) { next(err); }
  });

  return router;
}
