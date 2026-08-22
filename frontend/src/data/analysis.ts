import type { AnalysisResult } from "../types/analysis";

export const mockAnalysis: AnalysisResult = {
  statements: [
    {
      id: "s1",
      surroundingContext: "When discussing standard atmospheric pressure...",
      statement: "Water boils at 100°C at sea level",
      classification: {
        class: "fact",
        confidence: 0.7,
      },
      ruling: {
        verdict: "supported",
        confidence: 0.92,
        justification:
          "At standard atmospheric pressure water boils at 100°C [1], confirmed by [2].",
        references: [
          {
            id: "1",
            source: "https://example.com/source-one",
            excerpt: "At 1 atm, water boils at 100 °C...",
          },
          {
            id: "2",
            source: "https://example.com/source-two",
            excerpt: "The normal boiling point is defined...",
          },
        ],
      },
      error: null,
    },
    {
      id: "s2",
      surroundingContext:
        "The author is describing their experience at the restaurant.",
      statement: "The soup was delicious",
      classification: {
        class: "opinion",
        confidence: 0.95,
      },
      ruling: null,
      error: null,
    },
    {
      id: "s3",
      surroundingContext:
        "The text discusses Acme Corporation financial performance.",
      statement: "Acme reported record revenue in Q3",
      classification: {
        class: "fact",
        confidence: 0.6,
      },
      ruling: null,
      error: {
        kind: "timeout",
        message: "check exceeded 180s",
      },
    },
    {
      id: "s4",
      surroundingContext: "The article describes the history of the company.",
      statement: "The company was founded in 1987",
      classification: {
        class: "fact",
        confidence: 0.88,
      },
      ruling: {
        verdict: "refuted",
        confidence: 0.91,
        justification:
          "Available historical records indicate that the company was founded in 1991.",
        references: [
          {
            id: "3",
            source: "https://example.com/company-history",
            excerpt: "The company was established in 1991...",
          },
        ],
      },
      error: null,
    },
    {
      id: "s5",
      surroundingContext:
        "The report discusses the company’s workforce expansion.",
      statement: "The company doubled its workforce last year",
      classification: {
        class: "fact",
        confidence: 0.82,
      },
      ruling: {
        verdict: "mixed",
        confidence: 0.78,
        justification:
          "The company increased its workforce substantially, but available reports disagree on whether it actually doubled.",
        references: [
          {
            id: "4",
            source: "https://example.com/report-a",
            excerpt: "The workforce increased by approximately 80%...",
          },
          {
            id: "5",
            source: "https://example.com/report-b",
            excerpt: "Headcount roughly doubled during the year...",
          },
        ],
      },
      error: null,
    },
    {
      id: "s6",
      surroundingContext:
        "The article speculates about the company’s future plans.",
      statement: "The company will launch a new product next year",
      classification: {
        class: "fact",
        confidence: 0.61,
      },
      ruling: {
        verdict: "unverifiable",
        confidence: 0.84,
        justification:
          "No sufficiently reliable evidence was found to establish whether this future claim will occur.",
        references: [],
      },
      error: null,
    },
  ],
};
