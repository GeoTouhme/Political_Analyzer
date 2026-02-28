import { useState, useEffect } from "react";
import {
  ComposedChart, Line, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ReferenceLine, ResponsiveContainer, Legend, Area
} from "recharts";

// Default data as fallback
const DEFAULT_DATA = [
  { date: "Jan 01", avg: 36.3, articles: 1, level: "MEDIUM" }
];

const LEVEL_COLOR = {
  CRITICAL: "#ef4444",
  HIGH:     "#f97316",
  MEDIUM:   "#eab308",
  LOW:      "#22c55e",
};

const CustomDot = (props) => {
  const { cx, cy, payload } = props;
  const color = LEVEL_COLOR[payload.level] || "#94a3b8";
  if (payload.level === "CRITICAL") {
    return <circle cx={cx} cy={cy} r={7} fill={color} stroke="#fff" strokeWidth={2} />;
  }
  return <circle cx={cx} cy={cy} r={4} fill={color} stroke="#fff" strokeWidth={1.5} />;
};

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  const d = payload[0]?.payload;
  return (
    <div style={{
      background: "#0f172a", border: `1px solid ${LEVEL_COLOR[d.level] || "#1e293b"}`,
      borderRadius: 8, padding: "10px 14px", fontSize: 13, color: "#e2e8f0"
    }}>
      <div style={{ fontWeight: 700, marginBottom: 4, color: "#f8fafc" }}>{label}</div>
      <div>Risk Score: <span style={{ color: LEVEL_COLOR[d.level], fontWeight: 700 }}>{d.avg}</span></div>
      <div>Level: <span style={{ color: LEVEL_COLOR[d.level], fontWeight: 700 }}>{d.level}</span></div>
      <div>Articles: <span style={{ color: "#94a3b8" }}>{d.articles}</span></div>
    </div>
  );
};

export default function App() {
  const [data, setData] = useState(DEFAULT_DATA);
  const [meta, setMeta] = useState({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("/analysis_report_v2.json")
      .then(res => res.json())
      .then(report => {
        if (report.daily_breakdown) {
          const formatted = report.daily_breakdown.map(d => ({
            date: d.date.substring(5), 
            avg: d.anchored_risk || d.avg_risk || 0, // Fallback for different JSON versions
            articles: d.article_count || 0,
            level: d.risk_level || "UNKNOWN"
          }));
          setData(formatted);
          setMeta(report.meta || {});
        }
        setLoading(false);
      })
      .catch(err => {
        console.error("Failed to load report data", err);
        setLoading(false);
      });
  }, []);

  if (loading) return <div style={{ background: "#020817", minHeight: "100vh", color: "#e2e8f0", display: "flex", justifyContent: "center", alignItems: "center" }}>Loading Analyzer Data...</div>;

  const validData = data.filter(d => d.avg !== undefined);
  const latestDay = validData.length > 0 ? validData[validData.length - 1] : {date: "N/A", avg: 0, articles: 0, level: "UNKNOWN"};
  const peakDay = validData.length > 0 ? validData.reduce((a, b) => b.avg > a.avg ? b : a, validData[0]) : {date: "N/A", avg: 0};
  
  return (
    <div style={{
      background: "#020817", minHeight: "100vh", padding: "28px 24px",
      fontFamily: "'Inter', 'Segoe UI', sans-serif", color: "#e2e8f0",
      maxWidth: "1200px", margin: "0 auto"
    }}>
      {/* Header */}
      <div style={{ marginBottom: 24, display: "flex", justifyContent: "space-between", alignItems: "flex-end", flexWrap: "wrap", gap: 16 }}>
        <div>
          <div style={{ fontSize: 11, letterSpacing: 3, color: "#64748b", textTransform: "uppercase", marginBottom: 4 }}>
            Political Pattern Analyzer v2.2
          </div>
          <h1 style={{ fontSize: 26, fontWeight: 700, color: "#f8fafc", margin: 0 }}>
            Strategic Intelligence
          </h1>
        </div>
        <div style={{ textAlign: "right", color: "#64748b", fontSize: 13 }}>
          Status as of: <span style={{ color: "#f8fafc", fontWeight: 600 }}>{meta.generated_at ? new Date(meta.generated_at).toLocaleDateString() : 'Live'}</span>
        </div>
      </div>

      {/* Hero Element: Today Risk Gauge */}
      <div style={{
        background: "#0f172a", border: "1px solid #1e293b",
        borderRadius: 20, padding: "24px 20px", marginBottom: 24,
        display: "flex", flexDirection: "column", alignItems: "center",
        boxShadow: "0 8px 32px rgba(0,0,0,0.4)", position: "relative"
      }}>
        <div style={{ fontSize: 11, color: "#64748b", fontWeight: 700, textTransform: "uppercase", letterSpacing: 2, marginBottom: 12 }}>
          Current Threat Level
        </div>

        <div style={{ width: "100%", maxWidth: "300px", height: "180px", position: "relative", display: "flex", justifyContent: "center" }}>
          <svg viewBox="0 0 100 60" style={{ width: "100%", height: "100%" }}>
            <path d="M 10 50 A 40 40 0 0 1 90 50" fill="none" stroke="#1e293b" strokeWidth="6" strokeLinecap="round" />
            <path d="M 10 50 A 40 40 0 0 1 90 50" fill="none" stroke={LEVEL_COLOR[latestDay.level]} strokeWidth="6" strokeLinecap="round"
                  strokeDasharray={`${(latestDay.avg / 100) * 126} 126`} 
                  style={{ transition: "stroke-dasharray 1s ease-out" }} />
            <text x="50" y="42" textAnchor="middle" fill="#f8fafc" style={{ fontSize: "14px", fontWeight: 900 }}>{latestDay.avg}</text>
            <text x="50" y="55" textAnchor="middle" fill={LEVEL_COLOR[latestDay.level]} style={{ fontSize: "8px", fontWeight: 800, textTransform: "uppercase" }}>{latestDay.level}</text>
          </svg>
        </div>
      </div>

      {/* Strategic Outlook */}
      {meta.strategic_outlook && (
        <div style={{
          background: "rgba(59, 130, 246, 0.08)", border: "1px solid rgba(59, 130, 246, 0.2)",
          padding: "20px", borderRadius: "12px", marginBottom: 24,
          fontSize: 15, fontStyle: "italic", lineHeight: 1.6, color: "#cbd5e1"
        }}>
          {meta.strategic_outlook}
        </div>
      )}

      {/* Secondary Cards Grid */}
      <div style={{ 
        display: "grid", 
        gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))", 
        gap: 12, 
        marginBottom: 32 
      }}>
        {[
          { label: "Peak Day", value: `${peakDay.date}: ${peakDay.avg}`, color: "#ef4444" },
          { label: "Articles Today", value: latestDay.articles, color: "#58a6ff" },
          { label: "Today Situation", value: latestDay.level, color: LEVEL_COLOR[latestDay.level] || "#94a3b8" },
        ].map(c => (
          <div key={c.label} style={{
            background: "#0f172a", border: "1px solid #1e293b",
            borderRadius: 12, padding: "16px",
            boxShadow: "0 2px 8px rgba(0,0,0,0.2)"
          }}>
            <div style={{ fontSize: 10, color: "#64748b", marginBottom: 6, fontWeight: 600, textTransform: "uppercase", letterSpacing: 0.5 }}>{c.label}</div>
            <div style={{ fontSize: typeof c.value === 'string' && c.value.length > 8 ? 14 : 20, fontWeight: 800, color: c.color }}>{c.value}</div>
          </div>
        ))}
      </div>

      {/* Main chart */}
      <div style={{
        background: "#0f172a", border: "1px solid #1e293b",
        borderRadius: 12, padding: "20px 8px 12px"
      }}>
        <ResponsiveContainer width="100%" height={380}>
          <ComposedChart data={data} margin={{ top: 10, right: 20, left: 0, bottom: 40 }}>
            <defs>
              <linearGradient id="areaGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#3b82f6" stopOpacity={0.25} />
                <stop offset="100%" stopColor="#3b82f6" stopOpacity={0} />
              </linearGradient>
            </defs>

            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />

            <ReferenceLine y={75} stroke="#ef4444" strokeDasharray="4 3" strokeOpacity={0.5}
              label={{ value: "CRITICAL", position: "right", fill: "#ef4444", fontSize: 10 }} />
            <ReferenceLine y={50} stroke="#f97316" strokeDasharray="4 3" strokeOpacity={0.5}
              label={{ value: "HIGH", position: "right", fill: "#f97316", fontSize: 10 }} />
            <ReferenceLine y={25} stroke="#eab308" strokeDasharray="4 3" strokeOpacity={0.4}
              label={{ value: "MEDIUM", position: "right", fill: "#eab308", fontSize: 10 }} />

            <XAxis dataKey="date" tick={{ fontSize: 10, fill: "#64748b" }}
              angle={-45} textAnchor="end" interval={Math.floor(data.length / 10)} />
            <YAxis domain={[0, 100]} tick={{ fontSize: 11, fill: "#64748b" }}
              tickFormatter={v => `${v}`} width={32} />

            <Tooltip content={<CustomTooltip />} />

            <Bar dataKey="articles" yAxisId={0} barSize={6}
              fill="#1e3a5f" opacity={0.5} radius={[2,2,0,0]} />

            <Area type="monotone" dataKey="avg" fill="url(#areaGrad)"
              stroke="none" dot={false} activeDot={false} />

            <Line
              type="monotone" dataKey="avg"
              stroke="#3b82f6" strokeWidth={2}
              dot={<CustomDot />}
              activeDot={{ r: 6, fill: "#3b82f6", stroke: "#fff", strokeWidth: 2 }}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* Notable events */}
      <div style={{ marginTop: 20 }}>
        <div style={{ fontSize: 12, color: "#64748b", marginBottom: 10, letterSpacing: 1, textTransform: "uppercase" }}>
          Notable Spikes
        </div>
        <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
          {data.filter(d => d.avg >= 75).map(d => (
            <div key={d.date} style={{
              background: "#0f172a", border: `1px solid ${LEVEL_COLOR[d.level]}22`,
              borderLeft: `3px solid ${LEVEL_COLOR[d.level]}`,
              borderRadius: 8, padding: "8px 14px", fontSize: 12
            }}>
              <span style={{ color: "#f8fafc", fontWeight: 600 }}>{d.date}</span>
              <span style={{ color: LEVEL_COLOR[d.level], marginLeft: 8, fontWeight: 700 }}>{d.avg}</span>
              <span style={{ color: "#64748b", marginLeft: 6 }}>({d.articles} art.)</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
