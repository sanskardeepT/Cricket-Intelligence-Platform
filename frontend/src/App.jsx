import { useEffect, useMemo, useState } from "react";
import { Gauge, RefreshCw, ShieldCheck, Trophy, UserRound } from "lucide-react";
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

const fallbackToss = {
  decision: "FIELD",
  field_score: 71.58,
  confidence: 0.432,
  probabilities: { BAT: 0.2842, FIELD: 0.7158 },
  model_name: "hist_gradient_boosting",
  source: "trained_artifact",
};

const fallbackPlayer = {
  batter: "V Kohli",
  expected_runs: 20.56,
  range: { p10: 2.56, p90: 48.56 },
  model_name: "hist_gradient_boosting",
  source: "trained_artifact",
};

function pct(value) {
  return `${Math.round((value ?? 0) * 100)}%`;
}

export default function App() {
  const [payload, setPayload] = useState(fallbackPayload);
  const [toss, setToss] = useState(fallbackToss);
  const [player, setPlayer] = useState(fallbackPlayer);
  const [accuracy, setAccuracy] = useState(null);
  const [history, setHistory] = useState([
    { over: "10", probability: 48 },
    { over: "12", probability: 52 },
    { over: "14", probability: 58 },
    { over: "15.2", probability: 61 },
  ]);

  // Optional websocket live streaming.
  const [useWebsocket, setUseWebsocket] = useState(true);

  const [loading, setLoading] = useState(false);
  const [modelLoading, setModelLoading] = useState(false);

  async function refresh() {
    setLoading(true);
    try {
      const [response, accuracyResponse] = await Promise.all([
        fetch(`${API_BASE}/live/demo`),
        fetch(`${API_BASE}/accuracy/summary`).catch(() => null),
      ]);
      if (!response.ok) throw new Error(`API returned ${response.status}`);
      const data = await response.json();
      setPayload(data);
      if (accuracyResponse?.ok) {
        setAccuracy(await accuracyResponse.json());
      }
      const probPercent = Math.round((data.prediction?.probability ?? 0) * 100);
      const overLabel = (() => {
        const ballsBowled = data.match_state?.balls_bowled;
        if (typeof ballsBowled !== "number" || !Number.isFinite(ballsBowled)) return "live";
        const wholeOvers = Math.floor(ballsBowled / 6);
        const ballsIntoOver = ballsBowled % 6;
        const fractional = ballsIntoOver === 0 ? `${wholeOvers}` : `${wholeOvers}.${ballsIntoOver}`;
        return fractional;
      })();
      setHistory((items) => [
        ...items.slice(-7),
        {
          over: String(overLabel),
          probability: probPercent,
        },
      ]);
    } catch {
      setPayload(fallbackPayload);
    } finally {
      setLoading(false);
    }
  }

  async function refreshModelPanels() {
    setModelLoading(true);
    try {
      const [tossResponse, playerResponse] = await Promise.all([
        fetch(`${API_BASE}/prematch/toss`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            venue: "Wankhede Stadium",
            toss_winner: "Mumbai Indians",
            venue_dew_factor: 78,
            pitch_deterioration: 38,
            captain_field_tendency: 67,
            chase_success_rate: 64,
          }),
        }),
        fetch(`${API_BASE}/players/runs`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            batter: "V Kohli",
            batting_team: "Royal Challengers Bangalore",
            bowling_team: "Mumbai Indians",
            venue: "Wankhede Stadium",
            innings: 1,
          }),
        }),
      ]);
      if (tossResponse.ok) setToss(await tossResponse.json());
      if (playerResponse.ok) setPlayer(await playerResponse.json());
    } catch {
      setToss(fallbackToss);
      setPlayer(fallbackPlayer);
    } finally {
      setModelLoading(false);
    }
  }


  useEffect(() => {
    if (!useWebsocket) {
      refresh();
      refreshModelPanels();
      return;
    }

    const scheme = API_BASE.startsWith("https") ? "wss" : "ws";
    const host = API_BASE.replace(/^https?:\/\//, "");
    const wsUrl = `${scheme}://${host}/ws/live`;

    let ws;
    let cancelled = false;

    try {
      ws = new WebSocket(wsUrl);
    } catch {
      refresh();
      return;
    }

    ws.onopen = () => {
      if (cancelled) return;
    };

    ws.onmessage = (event) => {
      if (cancelled) return;
      try {
        const data = JSON.parse(event.data);
        setPayload(data);
        const probPercent = Math.round((data.prediction?.probability ?? 0) * 100);

        const ballsBowled = data.match_state?.balls_bowled;
        let overLabel = "live";
        if (typeof ballsBowled === "number" && Number.isFinite(ballsBowled)) {
          const wholeOvers = Math.floor(ballsBowled / 6);
          const ballsIntoOver = ballsBowled % 6;
          overLabel = ballsIntoOver === 0 ? `${wholeOvers}` : `${wholeOvers}.${ballsIntoOver}`;
        }

        setHistory((items) => [
          ...items.slice(-7),
          { over: String(overLabel), probability: probPercent },
        ]);
      } catch {
        // ignore malformed messages
      }
    };

    ws.onerror = () => {
      if (cancelled) return;
      refresh();
    };

    ws.onclose = () => {
      if (cancelled) return;
    };

    return () => {
      cancelled = true;
      try {
        ws?.close();
      } catch {
        // ignore
      }
    };
  }, [useWebsocket]);

  useEffect(() => {
    refreshModelPanels();
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
        <section className="panel">
          <div className="panel-heading">
            <span>Accuracy Tracking</span>
            <strong>{accuracy?.configured ? "DB" : "Local"}</strong>
          </div>
          <div className="vote-grid">
            {(accuracy?.summary?.length ? accuracy.summary : [{ pred_type: "live_win_probability", total_predictions: 0, resolved_predictions: 0, accuracy: 0 }]).map((row) => (
              <div key={row.pred_type}>
                <span>{row.pred_type.replaceAll("_", " ")}</span>
                <strong>{Math.round((row.accuracy ?? 0) * 100)}%</strong>
                <small>{row.resolved_predictions ?? 0}/{row.total_predictions ?? 0} resolved</small>
              </div>
            ))}
          </div>
        </section>
        <section className="panel model-panel">
          <div className="panel-heading">
            <span>Toss Decision</span>
            <strong>{toss?.decision ?? "FIELD"}</strong>
          </div>
          <div className="decision-grid">
            <div>
              <span>Field Score</span>
              <strong>{Math.round(toss?.field_score ?? 0)}%</strong>
            </div>
            <div>
              <span>Confidence</span>
              <strong>{pct(toss?.confidence ?? 0)}</strong>
            </div>
          </div>
          <div className="source-row">
            <ShieldCheck size={16} />
            <span>{toss?.model_name ?? "toss model"} · {toss?.source ?? "fallback"}</span>
            <button className="mini-button" onClick={refreshModelPanels} type="button" title="Refresh model panels">
              <RefreshCw size={15} className={modelLoading ? "spin" : ""} />
            </button>
          </div>
        </section>
        <section className="panel model-panel">
          <div className="panel-heading">
            <span>Player Runs</span>
            <strong>{Math.round(player?.expected_runs ?? 0)}</strong>
          </div>
          <div className="player-line">
            <UserRound size={18} />
            <span>{player?.batter ?? "Batter"}</span>
          </div>
          <div className="decision-grid">
            <div>
              <span>P10</span>
              <strong>{Math.round(player?.range?.p10 ?? 0)}</strong>
            </div>
            <div>
              <span>P90</span>
              <strong>{Math.round(player?.range?.p90 ?? 0)}</strong>
            </div>
          </div>
          <div className="source-row">
            <ShieldCheck size={16} />
            <span>{player?.model_name ?? "player model"} · {player?.source ?? "fallback"}</span>
          </div>
        </section>
      </div>
    </main>
  );
}
