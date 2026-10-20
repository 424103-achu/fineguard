import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { deleteAnalysis, listAnalyses } from "../api/client.js";
import { GRADE_STYLE } from "../constants.js";

export default function HistoryPage() {
  const [rows, setRows] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    listAnalyses().then(setRows).catch((e) => setError(e.message));
  }, []);

  async function remove(id) {
    if (!window.confirm("Delete this analysis?")) return;
    try {
      await deleteAnalysis(id);
      setRows((r) => r.filter((x) => x.id !== id));
    } catch (e) {
      setError(e.message);
    }
  }

  if (error) return <p className="text-red-700">{error}</p>;
  if (!rows) return <p className="text-slate-500 animate-pulse">Loading…</p>;

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-bold text-slate-900">History</h1>
      {rows.length === 0 ? (
        <p className="text-slate-500">No analyses yet. <Link to="/" className="text-indigo-600 hover:underline">Analyze a document</Link>.</p>
      ) : (
        <div className="overflow-x-auto rounded-xl border border-slate-200 bg-white shadow-sm">
          <table className="w-full text-sm">
            <thead className="bg-slate-50 text-left text-slate-500">
              <tr>
                <th className="px-4 py-3 font-medium">Service</th>
                <th className="px-4 py-3 font-medium">Score</th>
                <th className="px-4 py-3 font-medium">Grade</th>
                <th className="px-4 py-3 font-medium">Flagged</th>
                <th className="px-4 py-3 font-medium">Date</th>
                <th className="px-4 py-3" />
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.id} className="border-t border-slate-100 hover:bg-slate-50">
                  <td className="px-4 py-3">
                    <Link to={`/result/${r.id}`} className="text-indigo-700 hover:underline">
                      {r.service_name || r.source}
                    </Link>
                  </td>
                  <td className="px-4 py-3">{Number(r.risk_score).toFixed(1)}</td>
                  <td className="px-4 py-3">
                    <span className={`rounded-full px-2.5 py-0.5 text-xs font-semibold ${GRADE_STYLE[r.grade]?.bg}`}>{r.grade}</span>
                  </td>
                  <td className="px-4 py-3">{r.n_flagged} / {r.n_sentences}</td>
                  <td className="px-4 py-3 text-slate-500">{new Date(r.created_at).toLocaleString()}</td>
                  <td className="px-4 py-3 text-right">
                    <button onClick={() => remove(r.id)} className="text-xs text-red-600 hover:underline">Delete</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
