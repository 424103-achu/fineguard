const BASE = import.meta.env.VITE_API_URL || "/api";

async function handle(res) {
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = Array.isArray(body.detail)
      ? body.detail.map((d) => d.msg).join("; ")   // FastAPI validation errors
      : body.detail;
    throw new Error(detail || `Request failed (${res.status})`);
  }
  return body;
}

export const analyzeText = (text, serviceName) =>
  fetch(`${BASE}/analyze/text`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, service_name: serviceName || null }),
  }).then(handle);

export const analyzeUrl = (url, serviceName) =>
  fetch(`${BASE}/analyze/url`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url, service_name: serviceName || null }),
  }).then(handle);

export const analyzePdf = (file, serviceName) => {
  const form = new FormData();
  form.append("file", file);
  if (serviceName) form.append("service_name", serviceName);
  return fetch(`${BASE}/analyze/pdf`, { method: "POST", body: form }).then(handle);
};

export const listAnalyses = () => fetch(`${BASE}/analyses`).then(handle);
export const getAnalysis = (id) => fetch(`${BASE}/analyses/${id}`).then(handle);
export const deleteAnalysis = (id) =>
  fetch(`${BASE}/analyses/${id}`, { method: "DELETE" }).then((res) => {
    if (!res.ok) throw new Error(`Delete failed (${res.status})`);
  });
