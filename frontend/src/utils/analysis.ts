import type { Statement } from "../types/analysis";

export function getSummary(statements: Statement[]) {
  return {
    supported: statements.filter(
      (statement) => statement.ruling?.verdict === "supported",
    ).length,

    refuted: statements.filter(
      (statement) => statement.ruling?.verdict === "refuted",
    ).length,

    mixed: statements.filter(
      (statement) => statement.ruling?.verdict === "mixed",
    ).length,

    unverifiable: statements.filter(
      (statement) => statement.ruling?.verdict === "unverifiable",
    ).length,

    opinion: statements.filter(
      (statement) => statement.classification.class === "opinion",
    ).length,

    error: statements.filter((statement) => statement.error !== null).length,
  };
}

export type StatementStatus =
  "supported" | "refuted" | "mixed" | "unverifiable" | "opinion" | "error";

export function getStatementStatus(statement: Statement): StatementStatus {
  if (statement.error) {
    return "error";
  }

  if (statement.classification.class === "opinion") {
    return "opinion";
  }

  if (statement.ruling) {
    return statement.ruling.verdict;
  }

  return "error";
}
