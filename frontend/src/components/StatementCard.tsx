import { useState } from "react";
import type { Statement } from "../types/analysis";
import { getStatementStatus } from "../utils/analysis";
import StatementDetails from "./StatementDetails";

interface StatementCardProps {
  statement: Statement;
  index: number;
}

function StatementCard({ statement, index }: StatementCardProps) {
  const [expanded, setExpanded] = useState(false);

  const status = getStatementStatus(statement);

  return (
    <article className={`statement statement--${status}`}>
      <button
        className="statement-toggle"
        onClick={() => setExpanded((current) => !current)}
        aria-expanded={expanded}
      >
        <div className="statement-number">
          {String(index + 1).padStart(2, "0")}
        </div>

        <div className="statement-content">
          <span className="classification">
            {statement.classification.class}
          </span>

          <p className="statement-text">{statement.statement}</p>

          <span className="verdict">{status}</span>
        </div>
      </button>

      {expanded && <StatementDetails statement={statement} />}
    </article>
  );
}

export default StatementCard;
