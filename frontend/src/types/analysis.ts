export type ClassificationClass = "fact" | "opinion";

export type Verdict = "supported" | "refuted" | "mixed" | "unverifiable";

export interface Reference {
  id: string;
  source: string;
  excerpt: string;
}

export interface Classification {
  class: ClassificationClass;
  confidence: number;
}

export interface Ruling {
  verdict: Verdict;
  confidence: number;
  justification: string;
  references: Reference[];
}

export interface AnalysisError {
  kind: string;
  message: string;
}

export interface Statement {
  id: string;
  surroundingContext: string;
  statement: string;
  classification: Classification;
  ruling: Ruling | null;
  error: AnalysisError | null;
}

export interface AnalysisResult {
  statements: Statement[];
}
