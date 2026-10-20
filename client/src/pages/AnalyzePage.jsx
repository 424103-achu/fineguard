import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { analyzePdf, analyzeText, analyzeUrl } from "../api/client.js";
import InputPanel from "../components/InputPanel.jsx";
import Disclaimer from "../components/Disclaimer.jsx";
import { CATEGORY_INFO } from "../constants.js";

export default function AnalyzePage() {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  async function handleSubmit({ kind, value, serviceName }) {
    setLoading(true);
    setError(null);
    try {
      const fn = { text: analyzeText, url: analyzeUrl, pdf: analyzePdf }[kind];
      const result = await fn(value, serviceName);
      navigate(`/result/${result.id}`, { state: { result } });
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-6">
      <section>
        <h1 className="text-2xl font-bold text-slate-900">Check the fine print</h1>
        <p className="mt-1 text-slate-600">
          Paste a Terms of Service, upload a PDF or enter a URL. Each sentence is checked for
          8 kinds of potentially unfair clauses, and the service gets a 0–100 risk score and an A–E grade.
        </p>
      </section>

      <InputPanel onSubmit={handleSubmit} loading={loading} />

      {loading && (
        <p className="text-sm text-slate-500 animate-pulse">
          Reading the document and classifying sentences. Long documents can take a little while…
        </p>
      )}
      {error && (
        <div role="alert" className="rounded-lg border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-800">
          {error}
        </div>
      )}

      <Disclaimer />

      <section>
        <h2 className="text-lg font-semibold text-slate-900 mb-3">What it looks for</h2>
        <div className="grid sm:grid-cols-2 gap-3">
          {Object.entries(CATEGORY_INFO).map(([name, desc]) => (
            <div key={name} className="rounded-lg border border-slate-200 bg-white p-3">
              <div className="font-medium text-sm">{name}</div>
              <div className="text-xs text-slate-600 mt-0.5">{desc}</div>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
