export const DISCLAIMER =
  "This tool highlights clauses that may be unfair to consumers based on a research dataset " +
  "annotated under EU consumer law. It is not legal advice and may miss or misclassify clauses.";

export const CATEGORY_INFO = {
  "Limitation of liability": "The provider limits or excludes its responsibility for losses or damages.",
  "Unilateral termination": "The provider can suspend or close your account at its discretion.",
  "Unilateral change": "The provider can change the terms or the service without your agreement.",
  "Content removal": "The provider can delete your content at its discretion.",
  "Contract by using": "Merely using the service counts as accepting the terms.",
  "Choice of law": "A specific country's or state's law governs disputes.",
  "Jurisdiction": "Disputes must be brought in a specific (often distant) court.",
  "Arbitration": "Disputes go to private arbitration instead of court.",
};

const PALETTE = ["#6366f1", "#f59e0b", "#10b981", "#ef4444", "#0ea5e9", "#a855f7", "#14b8a6", "#f97316"];
export const categoryColor = (labels, name) => PALETTE[Math.max(labels.indexOf(name), 0) % PALETTE.length];

export const GRADE_STYLE = {
  A: { color: "#16a34a", bg: "bg-green-100 text-green-800", text: "Few potentially unfair clauses" },
  B: { color: "#65a30d", bg: "bg-lime-100 text-lime-800", text: "Some potentially unfair clauses" },
  C: { color: "#ca8a04", bg: "bg-yellow-100 text-yellow-800", text: "Several potentially unfair clauses" },
  D: { color: "#ea580c", bg: "bg-orange-100 text-orange-800", text: "Many potentially unfair clauses" },
  E: { color: "#dc2626", bg: "bg-red-100 text-red-800", text: "Very many potentially unfair clauses" },
};
