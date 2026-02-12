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
  rewritten_question?: string | null;
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

type EvalSummary = Record<string, number | null>;

type EvalRunResponse = {
  job_id: string;
  status: string;
  message: string;
};

type EvalStatusResponse = {
  job_id: string;
  status: "queued" | "running" | "completed" | "failed";
  created_at: string;
  started_at?: string | null;
  finished_at?: string | null;
  processed: number;
  total: number;
  progress: number;
  current_sample_id?: string | null;
  message?: string | null;
  error?: string | null;
  summary: EvalSummary;
  artifacts: Record<string, string>;
};

type EvalRow = {
  sample_id: string;
  model: string;
  notes: string;
  cosine_similarity: number;
  citation_coverage: number;
  source_f1: number;
};

type EvalResultsResponse = {
  job_id: string;
  status: string;
  summary: EvalSummary;
  artifacts: Record<string, string>;
  rows: EvalRow[];
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
  const [evalStarting, setEvalStarting] = useState(false);
  const [evalStatus, setEvalStatus] = useState<EvalStatusResponse | null>(null);
  const [evalResults, setEvalResults] = useState<EvalResultsResponse | null>(null);
  const [evalError, setEvalError] = useState<string | null>(null);

  const evalBusy = useMemo(
    () => evalStarting || evalStatus?.status === "queued" || evalStatus?.status === "running",
    [evalStarting, evalStatus]
  );

  const canSubmit = useMemo(
    () => question.trim().length >= 3 && !loading && !ingestLoading && !evalBusy,
    [question, loading, ingestLoading, evalBusy]
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

  const fetchEvalResults = async (jobId: string) => {
    const response = await fetch(`${backendUrl}/eval/results/${jobId}`);
    if (!response.ok) {
      throw new Error(await readErrorDetail(response, "Failed to fetch evaluation results"));
    }
    const payload = (await response.json()) as EvalResultsResponse;
    setEvalResults(payload);
  };

  const fetchEvalStatus = async (jobId: string) => {
    const response = await fetch(`${backendUrl}/eval/status/${jobId}`);
    if (!response.ok) {
      throw new Error(await readErrorDetail(response, "Failed to fetch evaluation status"));
    }
    const payload = (await response.json()) as EvalStatusResponse;
    setEvalStatus(payload);
    if (payload.status === "completed") {
      await fetchEvalResults(jobId);
    }
  };

  const runEvaluation = async () => {
    setEvalStarting(true);
    setEvalError(null);
    setEvalResults(null);

    try {
      const response = await fetch(`${backendUrl}/eval/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ auto_ingest: true })
      });
      if (!response.ok) {
        throw new Error(await readErrorDetail(response, "Failed to start evaluation"));
      }
      const payload = (await response.json()) as EvalRunResponse;
      setEvalStatus({
        job_id: payload.job_id,
        status: "queued",
        created_at: new Date().toISOString(),
        processed: 0,
        total: 0,
        progress: 0,
        summary: {},
        artifacts: {},
        message: payload.message
      });
      await fetchEvalStatus(payload.job_id);
    } catch (err) {
      setEvalError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setEvalStarting(false);
    }
  };

  useEffect(() => {
    void refreshHealth();
  }, []);

  useEffect(() => {
    if (!evalStatus?.job_id) return;
    if (evalStatus.status !== "queued" && evalStatus.status !== "running") return;

    const interval = window.setInterval(() => {
      void fetchEvalStatus(evalStatus.job_id).catch((err) => {
        setEvalError(err instanceof Error ? err.message : "Failed to poll evaluation status");
      });
    }, 3000);
    return () => window.clearInterval(interval);
  }, [evalStatus?.job_id, evalStatus?.status]);

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

  const artifactUrl = (path: string): string => {
    if (path.startsWith("http://") || path.startsWith("https://")) {
      return path;
    }
    return `${backendUrl}${path}`;
  };

  const formatMetric = (value: number | null | undefined): string => {
    if (value === null || value === undefined) return "-";
    return value.toFixed(3);
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
              disabled={ingestLoading || loading || evalBusy}
            >
              {ingestLoading ? "Ingesting..." : "Ingest Corpus"}
            </button>
            <button
              type="button"
              className="secondary-btn"
              onClick={() => void runIngestion(true)}
              disabled={ingestLoading || loading || evalBusy}
            >
              Rebuild Index
            </button>
            <button
              type="button"
              className="ghost-btn"
              onClick={() => void refreshHealth()}
              disabled={ingestLoading || loading || evalBusy}
            >
              Refresh Status
            </button>
            <button
              type="button"
              className="ghost-btn"
              onClick={resetConversation}
              disabled={ingestLoading || loading || evalBusy}
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

        <section className="eval-panel">
          <div className="ingest-actions">
            <button
              type="button"
              className="secondary-btn"
              onClick={() => void runEvaluation()}
              disabled={loading || ingestLoading || evalBusy}
            >
              {evalBusy ? "Evaluation Running..." : "Run Evaluation"}
            </button>
            {evalStatus?.job_id && (
              <button
                type="button"
                className="ghost-btn"
                onClick={() => void fetchEvalStatus(evalStatus.job_id)}
                disabled={loading || ingestLoading || evalBusy}
              >
                Refresh Eval Status
              </button>
            )}
          </div>

          {evalStatus && (
            <>
              <p className="status-line">
                Eval job: <strong>{evalStatus.job_id.slice(0, 8)}...</strong> | status:{" "}
                <strong>{evalStatus.status}</strong>
              </p>
              <p className="status-line">
                Progress: <strong>{evalStatus.processed}</strong>/<strong>{evalStatus.total}</strong> (
                {(evalStatus.progress * 100).toFixed(0)}%)
                {evalStatus.current_sample_id ? ` | sample: ${evalStatus.current_sample_id}` : ""}
              </p>
              {evalStatus.message && <p className="status-line">Message: {evalStatus.message}</p>}
            </>
          )}

          {evalError && <p className="error">Evaluation error: {evalError}</p>}

          {evalStatus?.status === "completed" && (
            <div className="eval-summary">
              <p className="status-line">
                Mean cosine: <strong>{formatMetric(evalStatus.summary.mean_cosine_similarity)}</strong> | Mean source F1:{" "}
                <strong>{formatMetric(evalStatus.summary.mean_source_f1)}</strong> | Mean citation coverage:{" "}
                <strong>{formatMetric(evalStatus.summary.mean_citation_coverage)}</strong>
              </p>
              <div className="eval-links">
                {Object.entries(evalStatus.artifacts).map(([name, path]) => (
                  <a key={name} href={artifactUrl(path)} target="_blank" rel="noreferrer">
                    {name}
                  </a>
                ))}
              </div>
            </div>
          )}

          {evalResults && evalResults.rows.length > 0 && (
            <details>
              <summary>Evaluation rows ({evalResults.rows.length})</summary>
              <div className="eval-table-wrap">
                <table className="eval-table">
                  <thead>
                    <tr>
                      <th>ID</th>
                      <th>Cosine</th>
                      <th>Coverage</th>
                      <th>Source F1</th>
                      <th>Notes</th>
                    </tr>
                  </thead>
                  <tbody>
                    {evalResults.rows.map((row) => (
                      <tr key={row.sample_id}>
                        <td>{row.sample_id}</td>
                        <td>{formatMetric(row.cosine_similarity)}</td>
                        <td>{formatMetric(row.citation_coverage)}</td>
                        <td>{formatMetric(row.source_f1)}</td>
                        <td>{row.notes}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </details>
          )}
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
            {answer.rewritten_question && answer.rewritten_question !== question.trim() && (
              <p className="status-line">
                Retrieval query: <strong>{answer.rewritten_question}</strong>
              </p>
            )}
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
