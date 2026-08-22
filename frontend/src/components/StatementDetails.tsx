import type { Statement } from "../types/analysis";

interface StatementDetailsProps {
  statement: Statement;
}

function StatementDetails({ statement }: StatementDetailsProps) {
  const { classification, ruling, error } = statement;

  return (
    <div className="statement-details">
      <section className="detail-section">
        <span className="detail-label">Classification</span>

        <div className="detail-value">
          <strong>{classification.class}</strong>

          <span>{Math.round(classification.confidence * 100)}% confidence</span>
        </div>
      </section>

      {ruling && (
        <>
          <section className="detail-section">
            <span className="detail-label">Verdict</span>

            <div className="detail-value">
              <strong>{ruling.verdict}</strong>

              <span>{Math.round(ruling.confidence * 100)}% confidence</span>
            </div>
          </section>

          <section className="detail-section">
            <span className="detail-label">Justification</span>

            <p className="detail-text">{ruling.justification}</p>
          </section>
        </>
      )}

      {error && (
        <section className="detail-section">
          <span className="detail-label">Verification error</span>

          <div className="detail-value">
            <strong>{error.kind}</strong>
          </div>

          <p className="detail-text">{error.message}</p>
        </section>
      )}
    </div>
  );
}

export default StatementDetails;
