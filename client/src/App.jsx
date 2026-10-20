import { NavLink, Route, Routes } from "react-router-dom";
import AnalyzePage from "./pages/AnalyzePage.jsx";
import ResultPage from "./pages/ResultPage.jsx";
import HistoryPage from "./pages/HistoryPage.jsx";

const link = ({ isActive }) =>
  `px-3 py-1.5 rounded-md text-sm font-medium ${
    isActive ? "bg-indigo-600 text-white" : "text-slate-600 hover:bg-slate-200"
  }`;

export default function App() {
  return (
    <div className="min-h-screen flex flex-col">
      <header className="bg-white border-b border-slate-200">
        <div className="max-w-6xl mx-auto px-4 h-14 flex items-center justify-between">
          <NavLink to="/" className="font-bold text-lg text-slate-900">
            🛡️ Fine-Print Guard
          </NavLink>
          <nav className="flex gap-2">
            <NavLink to="/" end className={link}>Analyze</NavLink>
            <NavLink to="/history" className={link}>History</NavLink>
          </nav>
        </div>
      </header>
      <main className="flex-1 max-w-6xl w-full mx-auto px-4 py-8">
        <Routes>
          <Route path="/" element={<AnalyzePage />} />
          <Route path="/result/:id" element={<ResultPage />} />
          <Route path="/history" element={<HistoryPage />} />
          <Route path="*" element={<p>Page not found.</p>} />
        </Routes>
      </main>
      <footer className="text-center text-xs text-slate-500 py-4">
        Flags potentially unfair clauses using the UNFAIR-ToS research dataset. Not legal advice.
      </footer>
    </div>
  );
}
