import { CATEGORY_INFO, categoryColor } from "../constants.js";

export default function FlaggedClauseList({ flagged, labels, filter, onFilter, onJump }) {
  const shown = filter ? flagged.filter((c) => c.categories.includes(filter)) : flagged;
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2 text-sm">
        <span className="text-slate-500">Filter:</span>
        <button
          onClick={() => onFilter(null)}
          className={`rounded-full px-3 py-1 border ${!filter ? "bg-slate-800 text-white border-slate-800" : "border-slate-300"}`}
        >
          All ({flagged.length})
        </button>
        {labels.map((l) => {
          const n = flagged.filter((c) => c.categories.includes(l)).length;
          if (!n) return null;
          return (
            <button
              key={l} onClick={() => onFilter(filter === l ? null : l)}
              className="rounded-full px-3 py-1 border text-xs"
              style={filter === l
                ? { background: categoryColor(labels, l), color: "white", borderColor: categoryColor(labels, l) }
                : { borderColor: categoryColor(labels, l), color: categoryColor(labels, l) }}
            >
              {l} ({n})
            </button>
          );
        })}
      </div>

      {shown.length === 0 && (
        <p className="text-sm text-slate-500 py-6 text-center">No potentially unfair clauses were flagged.</p>
      )}
      {shown.map((c) => (
        <article key={c.index} className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
          <p className="text-sm leading-relaxed">{c.sentence}</p>
          <div className="mt-3 flex flex-wrap items-center gap-2">
            {c.categories.map((cat) => (
              <span
                key={cat} title={CATEGORY_INFO[cat]}
                className="rounded-full px-2.5 py-0.5 text-xs font-medium text-white"
                style={{ background: categoryColor(labels, cat) }}
              >
                {cat} · {Math.round(c.confidence[cat] * 100)}%
              </span>
            ))}
            {onJump && (
              <button onClick={() => onJump(c.index)} className="ml-auto text-xs text-indigo-600 hover:underline">
                Show in document (#{c.index + 1})
              </button>
            )}
          </div>
        </article>
      ))}
    </div>
  );
}
