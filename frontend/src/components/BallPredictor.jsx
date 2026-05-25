import { Activity, CircleDot, Target } from "lucide-react";

export default function BallPredictor({ ball }) {
  const entries = Object.entries(ball?.distribution ?? {});
  return (
    <section className="panel" aria-label="Ball predictor">
      <div className="panel-heading">
        <span>Next Ball</span>
        <strong>{ball?.most_likely ?? "Single"}</strong>
      </div>
      <div className="ball-grid">
        {entries.map(([label, probability]) => (
          <div className="ball-row" key={label}>
            <span>{label}</span>
            <div className="bar">
              <i style={{ width: `${Math.round(probability * 100)}%` }} />
            </div>
            <b>{Math.round(probability * 100)}%</b>
          </div>
        ))}
      </div>
      <div className="reason-strip">
        <Activity size={16} />
        <span>{ball?.reasons?.[0] ?? "Base T20 outcome rates dominate this ball."}</span>
      </div>
      <div className="mini-metrics">
        <div>
          <Target size={17} />
          <span>{ball?.model_name ?? "Outcome Model"}</span>
        </div>
        <div>
          <CircleDot size={17} />
          <span>{ball?.source ?? "Ball-by-ball"}</span>
        </div>
      </div>
    </section>
  );
}
