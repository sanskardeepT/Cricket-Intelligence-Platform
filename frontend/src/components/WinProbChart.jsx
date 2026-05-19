import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

export default function WinProbChart({ data }) {
  return (
    <section className="panel chart-panel" aria-label="Win probability chart">
      <div className="panel-heading">
        <span>Live Win Probability</span>
        <strong>{data.at(-1)?.probability ?? 0}%</strong>
      </div>
      <div className="chart-box">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ left: 4, right: 14, top: 12, bottom: 0 }}>
            <defs>
              <linearGradient id="winFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#22c55e" stopOpacity={0.42} />
                <stop offset="95%" stopColor="#14b8a6" stopOpacity={0.05} />
              </linearGradient>
            </defs>
            <CartesianGrid stroke="#d7dde5" strokeDasharray="3 3" />
            <XAxis dataKey="over" tick={{ fill: "#506070", fontSize: 12 }} />
            <YAxis domain={[0, 100]} tick={{ fill: "#506070", fontSize: 12 }} />
            <Tooltip formatter={(value) => [`${value}%`, "Win"]} />
            <Area type="monotone" dataKey="probability" stroke="#0f766e" strokeWidth={3} fill="url(#winFill)" />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}
