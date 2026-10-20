import { test } from "node:test";
import assert from "node:assert/strict";
import request from "supertest";
import { createApp } from "../app.js";

const RESULT = {
  source: "text", service_name: "Example", n_sentences: 2, n_flagged: 1,
  risk_score: 7.9, grade: "A", categories: [], flagged: [], sentences: [],
  processing_ms: 10, model_version: "test", weights_version: "v1-equal", disclaimer: "x",
};

function fakePool() {
  const rows = [];
  return {
    rows,
    async query(sql, params = []) {
      if (sql.startsWith("INSERT")) {
        rows.push({ id: rows.length + 1, result: params[6], created_at: "2026-01-01" });
        return { rows: [rows.at(-1)] };
      }
      if (sql.includes("WHERE id")) {
        return { rows: rows.filter((r) => r.id === params[0]) };
      }
      return { rows: [...rows].reverse() };
    },
  };
}

function fakeHttp(handler) {
  return { post: async (url, body) => handler(url, body), get: async () => ({ data: {} }) };
}

test("analyze text forwards to ML, stores and returns id", async () => {
  const pool = fakePool();
  let called;
  const app = createApp({
    pool, mlUrl: "http://ml",
    http: fakeHttp((url, body) => { called = { url, body }; return { data: RESULT }; }),
  });
  const res = await request(app).post("/api/analyze/text").send({ text: "hello world terms" });
  assert.equal(res.status, 200);
  assert.equal(called.url, "http://ml/analyze/text");
  assert.equal(res.body.id, 1);
  assert.equal(res.body.grade, "A");

  const got = await request(app).get("/api/analyses/1");
  assert.equal(got.status, 200);
  assert.equal(got.body.risk_score, 7.9);
  assert.equal((await request(app).get("/api/analyses/99")).status, 404);
  assert.equal((await request(app).get("/api/analyses/abc")).status, 404);
});

test("ML errors are forwarded with their status", async () => {
  const app = createApp({
    pool: fakePool(), mlUrl: "http://ml",
    http: fakeHttp(() => {
      const err = new Error("bad");
      err.response = { status: 422, data: { detail: "No readable sentences found" } };
      throw err;
    }),
  });
  const res = await request(app).post("/api/analyze/text").send({ text: "x" });
  assert.equal(res.status, 422);
  assert.equal(res.body.detail, "No readable sentences found");
});

test("unreachable ML service gives 502", async () => {
  const app = createApp({
    pool: fakePool(), mlUrl: "http://ml",
    http: fakeHttp(() => { const e = new Error("down"); e.code = "ECONNREFUSED"; throw e; }),
  });
  const res = await request(app).post("/api/analyze/url").send({ url: "https://example.com" });
  assert.equal(res.status, 502);
});

test("pdf upload without a file gives 400", async () => {
  const app = createApp({ pool: fakePool(), mlUrl: "http://ml", http: fakeHttp(() => ({})) });
  const res = await request(app).post("/api/analyze/pdf").field("service_name", "x");
  assert.equal(res.status, 400);
});
