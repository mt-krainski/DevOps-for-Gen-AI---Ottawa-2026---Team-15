import type { Statement } from "../types/analysis";
import { getSummary } from "../utils/analysis";

interface AnalysisSummaryProps {
  statements: Statement[];
}

function AnalysisSummary({ statements }: AnalysisSummaryProps) {
  const summary = getSummary(statements);

  const items = [
    { label: "Supported", value: summary.supported },
    { label: "Refuted", value: summary.refuted },
    { label: "Mixed", value: summary.mixed },
    { label: "Unverifiable", value: summary.unverifiable },
    { label: "Opinion", value: summary.opinion },
    { label: "Error", value: summary.error },
  ];

  return (
    <section className="summary">
      {items.map((item) => (
        <div key={item.label}>
          <strong>{item.value}</strong>
          <span>{item.label}</span>
        </div>
      ))}
    </section>
  );
}

export default AnalysisSummary;
