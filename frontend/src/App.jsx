import { useEffect, useMemo, useState } from "react";
import { Gauge, RefreshCw, ShieldCheck, Trophy } from "lucide-react";
import BallPredictor from "./components/BallPredictor.jsx";
import ReasonCard from "./components/ReasonCard.jsx";
import WinProbChart from "./components/WinProbChart.jsx";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";

const fallbackPayload = {
  pressure_index: 55,
  prediction: {
    probability: 0.61,
    confidence: 0.22,
    label: "win",
    model_votes: { elo_prior: 0.55, monte_carlo: 0.63, run_rate_edge: 0.56 },
  },
  match_state: {
    batting_team: "MI",
    bowling_team: "CSK",
    venue: "Wankhede Stadium",
    score: 128,
    wickets: 4,
    balls_left: 28,
    current_run_rate: 8.35,
    required_run_rate: 10.71,
  },
  ball_prediction: {
    most_likely: "Single",
    distribution: { "Dot ball": 0.28, Single: 0.29, "Two runs": 0.13, Four: 0.15, Six: 0.08, Wicket: 0.07 },
    reasons: ["Base T20 outcome rates dominate this ball."],
  },
  explanation: {
    summary: "Baseline 50% moved +11.0% to 61.0%.",
    reasons: [
      { feature: "elo_prior", contribution: 0.08, text: "Team strength prior is above opponent baseline." },
      { feature: "pressure", contribution: -0.04, text: "Pressure index is rising with the chase rate." },
    ],
  },
};

function pct(value) {
  return `${Math.round((value ?? 0) * 100)}%`;
}

export default function App() {
  const [payload, setPayload] = useState(fallbackPayload);
  const [history, setHistory] = useState([
    { over: "10", probability: 48 },
    { over: "12", probability: 52 },
    { over: "14", probability: 58 },
    { over: "15.2", probability: 61 },
  ]);
  const [loading, setLoading] = useState(false);

  async function refresh() {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE}/live/demo`);
      if (!response.ok) throw new Error(`API returned ${response.status}`);
      const data = await response.json();
      setPayload(data);
      setHistory((items) => [
        ...items.slice(-7),
        {
          over: String(data.match_state?.balls_bowled ? Math.floor(data.match_state.balls_bowled / 6) : "live"),
          probability: Math.round(data.prediction.probability * 100),
        },
      ]);
    } catch {
      setPayload(fallbackPayload);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    refresh();
  }, []);

  const match = payload.match_state ?? fallbackPayload.match_state;
  const probability = payload.prediction?.probability ?? 0;
  const confidence = payload.prediction?.confidence ?? 0;
  const votes = payload.prediction?.model_votes ?? {};
  const statusText = useMemo(() => {
    const batting = match.batting_team ?? "Batting";
    const bowling = match.bowling_team ?? "Bowling";
    return `${batting} ${match.score}/${match.wickets} vs ${bowling}`;
  }, [match]);

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <h1>Cricket Intelligence Platform</h1>
          <p>{statusText} · {match.venue}</p>
        </div>
        <button className="icon-button" onClick={refresh} title="Refresh live prediction" type="button">
          <RefreshCw size={19} className={loading ? "spin" : ""} />
        </button>
      </header>

      <section className="score-band">
        <div className="metric hero-metric">
          <Trophy size={22} />
          <span>Win Probability</span>
          <strong>{pct(probability)}</strong>
        </div>
        <div className="metric">
          <Gauge size={21} />
          <span>Pressure</span>
          <strong>{Math.round(payload.pressure_index ?? 0)}/100</strong>
        </div>
        <div className="metric">
          <ShieldCheck size={21} />
          <span>Confidence</span>
          <strong>{pct(confidence)}</strong>
        </div>
        <div className="metric">
          <span>CRR</span>
          <strong>{match.current_run_rate}</strong>
        </div>
        <div className="metric">
          <span>RRR</span>
          <strong>{match.required_run_rate}</strong>
        </div>
      </section>

      <div className="layout">
        <WinProbChart data={history} />
        <BallPredictor ball={payload.ball_prediction} />
        <ReasonCard explanation={payload.explanation} />
        <section className="panel">
          <div className="panel-heading">
            <span>Model Votes</span>
            <strong>Stack</strong>
          </div>
          <div className="vote-grid">
            {Object.entries(votes).map(([key, value]) => (
              <div key={key}>
                <span>{key.replaceAll("_", " ")}</span>
                <strong>{pct(value)}</strong>
              </div>
            ))}
          </div>
        </section>
      </div>
    </main>
  );
}
