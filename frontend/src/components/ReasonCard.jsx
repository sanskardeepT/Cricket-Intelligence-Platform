import { BadgeInfo, TrendingDown, TrendingUp } from "lucide-react";

export default function ReasonCard({ explanation }) {
  const reasons = explanation?.reasons ?? [];
  return (
    <section className="panel" aria-label="Scientific reasons">
      <div className="panel-heading">
        <span>Scientific Explanation</span>
        <BadgeInfo size={19} />
      </div>
      <p className="summary">{explanation?.summary ?? "Baseline 50% adjusted by live match evidence."}</p>
      <div className="reason-list">
        {reasons.map((reason) => {
          const positive = reason.contribution >= 0;
          const Icon = positive ? TrendingUp : TrendingDown;
          return (
            <article className="reason-item" key={`${reason.feature}-${reason.text}`}>
              <Icon size={18} className={positive ? "up" : "down"} />
              <div>
                <strong>{positive ? "+" : ""}{Math.round(reason.contribution * 100)}%</strong>
                <span>{reason.text}</span>
              </div>
            </article>
          );
        })}
      </div>
    </section>
  );
}
