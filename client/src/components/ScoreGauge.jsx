import { GRADE_STYLE } from "../constants.js";

/** 0–100 semicircle gauge with the grade letter, coloured by band. */
export default function ScoreGauge({ score, grade }) {
  const style = GRADE_STYLE[grade] || GRADE_STYLE.C;
  const r = 80;
  const circumference = Math.PI * r;
  const filled = (Math.min(Math.max(score, 0), 100) / 100) * circumference;
  return (
    <div className="flex flex-col items-center">
      <svg viewBox="0 0 200 115" className="w-56" role="img" aria-label={`Risk score ${score} of 100, grade ${grade}`}>
        <path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="#e2e8f0" strokeWidth="18" strokeLinecap="round" />
        <path
          d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke={style.color} strokeWidth="18"
          strokeLinecap="round" strokeDasharray={`${filled} ${circumference}`}
        />
        <text x="100" y="80" textAnchor="middle" fontSize="40" fontWeight="700" fill={style.color}>{grade}</text>
        <text x="100" y="104" textAnchor="middle" fontSize="14" fill="#475569">{score.toFixed(1)} / 100</text>
      </svg>
      <span className={`mt-1 rounded-full px-3 py-1 text-xs font-medium ${style.bg}`}>{style.text}</span>
    </div>
  );
}
