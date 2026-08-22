import "./App.css";
import { mockAnalysis } from "./data/analysis";
import AnalysisSummary from "./components/AnalysisSummary";
import StatementCard from "./components/StatementCard";

function App() {
  const { statements } = mockAnalysis;

  return (
    <main className="app">
      <header className="header">
        <div className="brand">FACTCHECK</div>

        <nav className="header-actions">
          <button>Export</button>
          <button>New analysis</button>
        </nav>
      </header>

      <section className="analysis-header">
        <p className="eyebrow">ANALYSIS</p>

        <h1>Fact-checking analysis</h1>

        <p className="statement-count">
          {statements.length} statements analyzed
        </p>
      </section>

      <AnalysisSummary statements={statements} />

      <section className="workspace">
        <div className="statements">
          <div className="section-heading">
            <p className="eyebrow">STATEMENTS</p>
          </div>

          {statements.map((statement, index) => (
            <StatementCard
              key={statement.id}
              statement={statement}
              index={index}
            />
          ))}
        </div>

        <aside className="inspector">
          <p className="eyebrow">INSPECTOR</p>

          <p>Select a statement to inspect its details.</p>
        </aside>
      </section>
    </main>
  );
}

export default App;
