import { useEffect, useState } from "react";
import { Link, useLocation, useParams } from "react-router-dom";
import { getAnalysis } from "../api/client.js";
import ScoreGauge from "../components/ScoreGauge.jsx";
import CategoryBreakdown from "../components/CategoryBreakdown.jsx";
import FlaggedClauseList from "../components/FlaggedClauseList.jsx";
import DocumentView from "../components/DocumentView.jsx";
import Disclaimer from "../components/Disclaimer.jsx";

export default function ResultPage() {
  const { id } = useParams();
  const location = useLocation();
  const passed = location.state?.result;
  const [result, setResult] = useState(passed && String(passed.id) === id ? passed : null);
  const [error, setError] = useState(null);
  const [filter, setFilter] = useState(null);
  const [active, setActive] = useState(null);

  useEffect(() => {
    if (result && String(result.id) === id) return;
    setResult(null);
    getAnalysis(id).then(setResult).catch((e) => setError(e.message));
  }, [id]); // eslint-disable-line react-hooks/exhaustive-deps

  if (error) return <p className="text-red-700">{error}</p>;
  if (!result) return <p className="text-slate-500 animate-pulse">Loading…</p>;

  const labels = result.categories.map((c) => c.category);
  const jump = (i) => {
    setActive(i);
    document.getElementById(`sent-${i}`)?.scrollIntoView({ behavior: "smooth", block: "center" });
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div>
          <h1 className="text-2xl font-bold text-slate-900">{result.service_name || "Analysis result"}</h1>
          <p className="text-sm text-slate-500 break-all">
            Source: {result.source} · {new Date(result.created_at).toLocaleString()}
          </p>
        </div>
        <Link to="/" className="text-sm text-indigo-600 hover:underline">Analyze another →</Link>
      </div>

      <Disclaimer text={result.disclaimer} />

      <div className="grid lg:grid-cols-3 gap-6">
        <section className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm flex flex-col items-center justify-center">
          <ScoreGauge score={result.risk_score} grade={result.grade} />
          <dl className="mt-4 grid grid-cols-3 gap-4 text-center text-sm w-full">
            <div><dt className="text-slate-500">Sentences</dt><dd className="font-semibold">{result.n_sentences}</dd></div>
            <div><dt className="text-slate-500">Flagged</dt><dd className="font-semibold">{result.n_flagged}</dd></div>
            <div><dt className="text-slate-500">Time</dt><dd className="font-semibold">{(result.processing_ms / 1000).toFixed(1)} s</dd></div>
          </dl>
          <p className="mt-3 text-xs text-slate-400">
            Model {result.model_version} · weights {result.weights_version}
          </p>
        </section>
        <section className="lg:col-span-2 rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
          <h2 className="font-semibold mb-2">Flagged sentences per category</h2>
          <CategoryBreakdown categories={result.categories} selected={filter} onSelect={setFilter} />
        </section>
      </div>

      <div className="grid lg:grid-cols-2 gap-6">
        <section>
          <h2 className="font-semibold mb-3">Potentially unfair clauses</h2>
          <FlaggedClauseList
            flagged={result.flagged} labels={labels} filter={filter}
            onFilter={setFilter} onJump={result.sentences?.length ? jump : null}
          />
        </section>
        {result.sentences?.length > 0 && (
          <section className="lg:sticky lg:top-4 self-start">
            <h2 className="font-semibold mb-3">Full document</h2>
            <DocumentView sentences={result.sentences} flagged={result.flagged} labels={labels} active={active} />
          </section>
        )}
      </div>
    </div>
  );
}
