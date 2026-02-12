import { FormEvent, useEffect, useMemo, useState } from "react";

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
  session_id?: string | null;
};

type IngestResponse = {
  status: string;
  message: string;
  documents: number;
  pages: number;
  chunks: number;
  skipped: boolean;
};

type HealthResponse = {
  status: string;
  indexed_chunks: number;
  ollama_ready: boolean;
};

const backendUrl = import.meta.env.VITE_BACKEND_URL || "http://localhost:8000";
const sessionStorageKey = "rag_session_id";

const createSessionId = (): string => {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return `session-${Date.now()}-${Math.random().toString(16).slice(2)}`;
};

export default function App() {
  const [question, setQuestion] = useState("");
  const [sessionId, setSessionId] = useState<string>(() => {
    if (typeof window === "undefined") {
      return createSessionId();
    }
    const existing = window.localStorage.getItem(sessionStorageKey);
    return existing || createSessionId();
  });
  const [loading, setLoading] = useState(false);
  const [ingestLoading, setIngestLoading] = useState(false);
  const [answer, setAnswer] = useState<QueryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [ingestMessage, setIngestMessage] = useState<string | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);

  const canSubmit = useMemo(
    () => question.trim().length >= 3 && !loading && !ingestLoading,
    [question, loading, ingestLoading]
  );

  const readErrorDetail = async (response: Response, fallback: string): Promise<string> => {
    try {
      const payload = await response.json();
      return payload.detail || payload.message || fallback;
    } catch {
      return fallback;
    }
  };

  const refreshHealth = async () => {
    setHealthError(null);
    try {
      const response = await fetch(`${backendUrl}/health`);
      if (!response.ok) {
        throw new Error(await readErrorDetail(response, "Health check failed"));
      }
      const payload = (await response.json()) as HealthResponse;
      setHealth(payload);
    } catch (err) {
      setHealth(null);
      setHealthError(err instanceof Error ? err.message : "Unknown error");
    }
  };

  const runIngestion = async (force: boolean) => {
    setIngestLoading(true);
    setIngestMessage(null);
    setError(null);

    try {
      const response = await fetch(`${backendUrl}/ingest`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ force })
      });
      if (!response.ok) {
        throw new Error(await readErrorDetail(response, "Ingestion failed"));
      }

      const payload = (await response.json()) as IngestResponse;
      setIngestMessage(
        `${payload.message} | docs=${payload.documents} pages=${payload.pages} chunks=${payload.chunks}`
      );
      await refreshHealth();
    } catch (err) {
      setIngestMessage(null);
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setIngestLoading(false);
    }
  };

  useEffect(() => {
    void refreshHealth();
  }, []);

  useEffect(() => {
    if (typeof window !== "undefined") {
      window.localStorage.setItem(sessionStorageKey, sessionId);
    }
  }, [sessionId]);

  const resetConversation = () => {
    setSessionId(createSessionId());
    setAnswer(null);
    setError(null);
  };

  const onSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!canSubmit) return;

    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${backendUrl}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: question.trim(), session_id: sessionId })
      });

      if (!response.ok) {
        throw new Error(await readErrorDetail(response, "Query failed"));
      }

      const payload = (await response.json()) as QueryResponse;
      if (payload.session_id) {
        setSessionId(payload.session_id);
      }
      setAnswer(payload);
    } catch (err) {
      setAnswer(null);
      const message = err instanceof Error ? err.message : "Unknown error";
      if (message.includes("Vector index is empty")) {
        setError("Vector index is empty. Click 'Ingest Corpus' first.");
      } else {
        setError(message);
      }
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

        <section className="ingest-panel">
          <div className="ingest-actions">
            <button
              type="button"
              className="secondary-btn"
              onClick={() => void runIngestion(false)}
              disabled={ingestLoading || loading}
            >
              {ingestLoading ? "Ingesting..." : "Ingest Corpus"}
            </button>
            <button
              type="button"
              className="secondary-btn"
              onClick={() => void runIngestion(true)}
              disabled={ingestLoading || loading}
            >
              Rebuild Index
            </button>
            <button
              type="button"
              className="ghost-btn"
              onClick={() => void refreshHealth()}
              disabled={ingestLoading || loading}
            >
              Refresh Status
            </button>
            <button
              type="button"
              className="ghost-btn"
              onClick={resetConversation}
              disabled={ingestLoading || loading}
            >
              New Conversation
            </button>
          </div>

          <p className="status-line">
            Indexed chunks: <strong>{health ? health.indexed_chunks : "?"}</strong> | Ollama:{" "}
            <strong>{health?.ollama_ready ? "ready" : "not ready"}</strong>
          </p>
          <p className="status-line">
            Conversation ID: <strong>{sessionId.slice(0, 8)}...</strong> (memory enabled across follow-up questions)
          </p>
          {healthError && <p className="error">Health error: {healthError}</p>}
          {ingestMessage && <p className="info">{ingestMessage}</p>}
        </section>

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
