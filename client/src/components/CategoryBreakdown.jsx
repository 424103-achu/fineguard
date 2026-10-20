import { Bar, BarChart, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { CATEGORY_INFO, categoryColor } from "../constants.js";

export default function CategoryBreakdown({ categories, selected, onSelect }) {
  const labels = categories.map((c) => c.category);
  const data = categories.map((c) => ({ name: c.category, count: c.count }));
  return (
    <div>
      <ResponsiveContainer width="100%" height={categories.length * 34 + 20}>
        <BarChart data={data} layout="vertical" margin={{ left: 10, right: 24 }}>
          <XAxis type="number" allowDecimals={false} tick={{ fontSize: 12 }} />
          <YAxis type="category" dataKey="name" width={160} tick={{ fontSize: 12 }} />
          <Tooltip
            cursor={{ fill: "#f1f5f9" }}
            formatter={(v) => [v, "flagged sentences"]}
            labelFormatter={(name) => `${name}: ${CATEGORY_INFO[name] || ""}`}
          />
          <Bar
            dataKey="count" radius={[0, 4, 4, 0]} cursor="pointer" isAnimationActive={false}
            onClick={(d) => onSelect?.(selected === d.name ? null : d.name)}
          >
            {data.map((d) => (
              <Cell
                key={d.name} fill={categoryColor(labels, d.name)}
                fillOpacity={!selected || selected === d.name ? 1 : 0.3}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <p className="text-xs text-slate-500 mt-1">Click a bar to filter the flagged clauses.</p>
    </div>
  );
}
