import { createApp } from "./app.js";
import { initDb, pool } from "./db/pool.js";

const app = createApp({
  pool,
  mlUrl: process.env.ML_URL || "http://localhost:8000",
  clientOrigin: process.env.CLIENT_ORIGIN,
});

await initDb();
const port = process.env.PORT || 5000;
app.listen(port, () => console.log(`API listening on ${port}`));
