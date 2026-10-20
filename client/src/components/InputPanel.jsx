import { useState } from "react";

const TABS = [
  { id: "text", label: "Paste text" },
  { id: "pdf", label: "Upload PDF" },
  { id: "url", label: "Enter URL" },
];

export default function InputPanel({ onSubmit, loading }) {
  const [tab, setTab] = useState("text");
  const [text, setText] = useState("");
  const [url, setUrl] = useState("");
  const [file, setFile] = useState(null);
  const [serviceName, setServiceName] = useState("");

  const ready =
    (tab === "text" && text.trim().length >= 20) ||
    (tab === "url" && /^https?:\/\/\S+$/i.test(url.trim())) ||
    (tab === "pdf" && file);

  function submit(e) {
    e.preventDefault();
    if (!ready || loading) return;
    const value = tab === "text" ? text : tab === "url" ? url.trim() : file;
    onSubmit({ kind: tab, value, serviceName: serviceName.trim() });
  }

  return (
    <form onSubmit={submit} className="bg-white rounded-xl shadow-sm border border-slate-200">
      <div className="flex border-b border-slate-200" role="tablist">
        {TABS.map((t) => (
          <button
            key={t.id} type="button" role="tab" aria-selected={tab === t.id}
            onClick={() => setTab(t.id)}
            className={`flex-1 py-3 text-sm font-medium border-b-2 -mb-px ${
              tab === t.id ? "border-indigo-600 text-indigo-700" : "border-transparent text-slate-500 hover:text-slate-700"
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      <div className="p-5 space-y-4">
        {tab === "text" && (
          <textarea
            value={text} onChange={(e) => setText(e.target.value)} rows={12}
            placeholder="Paste the full Terms of Service here…"
            className="w-full rounded-lg border border-slate-300 p-3 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
          />
        )}
        {tab === "pdf" && (
          <label className="flex flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed border-slate-300 p-10 text-sm text-slate-500 cursor-pointer hover:bg-slate-50">
            <span className="text-3xl">📄</span>
            <span>{file ? file.name : "Choose a PDF (max 10 MB)"}</span>
            <input
              type="file" accept="application/pdf,.pdf" className="hidden"
              onChange={(e) => setFile(e.target.files?.[0] || null)}
            />
          </label>
        )}
        {tab === "url" && (
          <div className="space-y-1">
            <input
              type="url" value={url} onChange={(e) => setUrl(e.target.value)}
              placeholder="https://example.com/terms"
              className="w-full rounded-lg border border-slate-300 p-3 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
            />
            <p className="text-xs text-slate-500">
              Pages that need JavaScript may not load. If that happens, paste the text or upload a PDF.
            </p>
          </div>
        )}

        <div className="flex flex-col sm:flex-row gap-3">
          <input
            value={serviceName} onChange={(e) => setServiceName(e.target.value)} maxLength={200}
            placeholder="Service name (optional)"
            className="flex-1 rounded-lg border border-slate-300 p-3 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
          />
          <button
            type="submit" disabled={!ready || loading}
            className="rounded-lg bg-indigo-600 px-6 py-3 text-sm font-semibold text-white hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? "Analyzing…" : "Analyze"}
          </button>
        </div>
      </div>
    </form>
  );
}
