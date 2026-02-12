import { FormEvent, useMemo, useState } from "react";

type Source = {
  source: string;
  page: number;
  chunk_id: string;
  score: number;
  text_excerpt: string;
};

type QueryResponse = {
  answer: string;
  model: string;
  sources: Source[];
};

const backendUrl = import.meta.env.VITE_BACKEND_URL || "http://localhost:8000";

export default function App() {
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [answer, setAnswer] = useState<QueryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const canSubmit = useMemo(() => question.trim().length >= 3 && !loading, [question, loading]);

  const onSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!canSubmit) return;

    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${backendUrl}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: question.trim() })
      });

      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || "Query failed");
      }

      setAnswer(payload as QueryResponse);
    } catch (err) {
      setAnswer(null);
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="page">
      <section className="panel">
        <header>
          <p className="eyebrow">Local, Open-Source RAG</p>
          <h1>Technical PDF QA</h1>
          <p className="sub">
            Ask a question about the indexed corpus. Answers are grounded in retrieved chunks with explicit citations.
          </p>
        </header>

        <form onSubmit={onSubmit} className="form">
          <label htmlFor="question">Question</label>
          <textarea
            id="question"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="What is the complexity per layer in self-attention?"
            rows={5}
          />

          <div className="actions">
            <button type="submit" disabled={!canSubmit}>
              {loading ? "Generating..." : "Ask"}
            </button>
          </div>
        </form>

        {error && <p className="error">Error: {error}</p>}

        {answer && (
          <section className="result">
            <div className="result-header">
              <h2>Answer</h2>
              <span>Model: {answer.model}</span>
            </div>
            <article className="answer">{answer.answer}</article>

            <h3>Sources</h3>
            <div className="sources">
              {answer.sources.map((source) => (
                <details key={source.chunk_id}>
                  <summary>
                    <strong>{source.source}</strong> | page {source.page} | {source.chunk_id} | score {source.score.toFixed(3)}
                  </summary>
                  <p>{source.text_excerpt}</p>
                </details>
              ))}
            </div>
          </section>
        )}
      </section>
    </main>
  );
}
