import { categoryColor } from "../constants.js";

/** All sentences in order; flagged ones highlighted. Each sentence has id `sent-<index>`. */
export default function DocumentView({ sentences, flagged, labels, active }) {
  const byIndex = new Map(flagged.map((c) => [c.index, c]));
  return (
    <div className="max-h-[36rem] overflow-y-auto rounded-lg border border-slate-200 bg-white p-5 text-sm leading-7">
      {sentences.map((s, i) => {
        const c = byIndex.get(i);
        if (!c) return <span key={i} id={`sent-${i}`} className="text-slate-600">{s} </span>;
        const color = categoryColor(labels, c.categories[0]);
        return (
          <mark
            key={i} id={`sent-${i}`} title={c.categories.join(", ")}
            className={`rounded px-0.5 text-slate-900 ${active === i ? "ring-2 ring-offset-1" : ""}`}
            style={{ background: `${color}33`, borderBottom: `2px solid ${color}`, "--tw-ring-color": color }}
          >
            {s}{" "}
          </mark>
        );
      })}
    </div>
  );
}
