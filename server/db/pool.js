import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import pg from "pg";

export const pool = new pg.Pool({
  connectionString: process.env.DATABASE_URL || "postgres://fpg:fpg@localhost:5432/fpg",
});

const here = path.dirname(fileURLToPath(import.meta.url));

/** Create the analyses table if it does not exist (retries while Postgres starts). */
export async function initDb(retries = 10, delayMs = 2000) {
  const sql = await fs.readFile(path.join(here, "schema.sql"), "utf8");
  for (let attempt = 1; ; attempt++) {
    try {
      await pool.query(sql);
      return;
    } catch (err) {
      if (attempt >= retries) throw err;
      console.log(`Database not ready (${err.code || err.message}), retrying...`);
      await new Promise((r) => setTimeout(r, delayMs));
    }
  }
}
