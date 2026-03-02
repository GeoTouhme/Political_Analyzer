import { useState, useEffect } from "react";
import {
  ComposedChart, Line, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ReferenceLine, ResponsiveContainer, Area
} from "recharts";

const LEVEL_COLOR = {
  CRITICAL: "#ef4444",
  HIGH:     "#f97316",
  MEDIUM:   "#eab308",
  LOW:      "#22c55e",
};

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  const d = payload[0]?.payload;
  return (
    <div style={{
      background: "#0f172a", border: `1px solid ${LEVEL_COLOR[d.level] || "#1e293b"}`,
      borderRadius: 8, padding: "10px 14px", fontSize: 13, color: "#e2e8f0"
    }}>
      <div style={{ fontWeight: 700, marginBottom: 4 }}>{label}</div>
      <div>Risk Score: <span style={{ color: LEVEL_COLOR[d.level], fontWeight: 700 }}>{d.avg}</span></div>
      <div>Level: <span style={{ color: LEVEL_COLOR[d.level], fontWeight: 700 }}>{d.level}</span></div>
      <div>Articles: <span style={{ color: "#94a3b8" }}>{d.articles}</span></div>
    </div>
  );
};

export default function App() {
  const [data, setData] = useState([]);
  const [articles, setArticles] = useState([]);
  const [meta, setMeta] = useState({});
  const [aiAnalysis, setAiAnalysis] = useState([]);
  const [loading, setLoading] = useState(true);
  
  // Filtering states
  const [searchTerm, setSearchTerm] = useState("");
  const [filterLevel, setFilterLevel] = useState("ALL");

  useEffect(() => {
    fetch("/analysis_report_v2.json")
      .then(res => res.json())
      .then(report => {
        if (report.daily_breakdown) {
          setData(report.daily_breakdown.map(d => ({
            date: d.date.substring(5), 
            avg: d.anchored_risk || d.avg_risk || 0,
            articles: d.article_count || 0,
            level: d.risk_level || "UNKNOWN"
          })));
        }
        if (report.articles) {
          setArticles(report.articles);
        }
        if (report.ai_deep_analysis) {
          setAiAnalysis(report.ai_deep_analysis);
        }
        setMeta(report.meta || {});
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

  // Filter Logic
  const filteredArticles = articles.filter(a => {
    const matchesSearch = a.title.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesLevel = filterLevel === "ALL" || a.risk_level === filterLevel;
    return matchesSearch && matchesLevel;
  }).slice(0, 100); 

  return (
    <div style={{
      background: "#020817", minHeight: "100vh", padding: "28px 20px",
      fontFamily: "'Inter', 'Segoe UI', sans-serif", color: "#e2e8f0",
      maxWidth: "800px", margin: "0 auto"
    }}>
      {/* Header */}
      <div style={{ marginBottom: 24, textAlign: 'center' }}>
        <div style={{ fontSize: 10, letterSpacing: 3, color: "#64748b", textTransform: "uppercase", marginBottom: 4 }}>
          Intelligence Platform v4.0
        </div>
        <h1 style={{ fontSize: 24, fontWeight: 800, color: "#f8fafc", margin: 0 }}>
          Persian Gulf Conflict Monitor
        </h1>
        <div style={{ color: "#64748b", fontSize: 12, marginTop: 4 }}>
           PGCM-2026 | Updated: {meta.generated_at ? new Date(meta.generated_at).toLocaleString() : 'Live'}
        </div>
      </div>

      {/* Hero: Gauge */}
      <div style={{
        background: "#0f172a", border: "1px solid #1e293b",
        borderRadius: 20, padding: "20px", marginBottom: 20,
        display: "flex", flexDirection: "column", alignItems: "center",
        boxShadow: "0 8px 32px rgba(0,0,0,0.4)"
      }}>
        <div style={{ width: "100%", maxWidth: "260px", height: "150px", position: "relative" }}>
          <svg viewBox="0 0 100 60" style={{ width: "100%", height: "100%" }}>
            <path d="M 10 50 A 40 40 0 0 1 90 50" fill="none" stroke="#1e293b" strokeWidth="7" strokeLinecap="round" />
            <path d="M 10 50 A 40 40 0 0 1 90 50" fill="none" stroke={LEVEL_COLOR[latestDay.level]} strokeWidth="7" strokeLinecap="round"
                  strokeDasharray={`${(latestDay.avg / 100) * 126} 126`} />
            <text x="50" y="42" textAnchor="middle" fill="#f8fafc" style={{ fontSize: "14px", fontWeight: 900 }}>{latestDay.avg}</text>
            <text x="50" y="55" textAnchor="middle" fill={LEVEL_COLOR[latestDay.level]} style={{ fontSize: "8px", fontWeight: 800 }}>{latestDay.level}</text>
          </svg>
        </div>
      </div>

      {/* Main Chart */}
      <div style={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 16, padding: "16px 10px 4px", marginBottom: 32 }}>
        <ResponsiveContainer width="100%" height={260}>
          <ComposedChart data={data}>
            <defs>
              <linearGradient id="areaGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#3b82f6" stopOpacity={0.2} />
                <stop offset="100%" stopColor="#3b82f6" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
            <ReferenceLine y={75} stroke="#ef4444" strokeDasharray="4 3" strokeOpacity={0.6} />
            <ReferenceLine y={50} stroke="#f97316" strokeDasharray="4 3" strokeOpacity={0.6} />
            <ReferenceLine y={25} stroke="#eab308" strokeDasharray="4 3" strokeOpacity={0.5} />
            <XAxis dataKey="date" tick={{ fontSize: 9, fill: "#64748b" }} interval={4} angle={-45} textAnchor="end" height={50} />
            <YAxis domain={[0, 100]} tick={{ fontSize: 9, fill: "#64748b" }} width={25} />
            <Tooltip content={<CustomTooltip />} />
            <Area type="monotone" dataKey="avg" fill="url(#areaGrad)" stroke="none" />
            <Line type="monotone" dataKey="avg" stroke="#3b82f6" strokeWidth={2} dot={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* AI DEEP ANALYSIS SECTION - NEW V4 */}
      {aiAnalysis.length > 0 && (
        <div style={{ marginBottom: 40, background: "rgba(239, 68, 68, 0.05)", border: "1px solid rgba(239, 68, 68, 0.2)", borderRadius: 16, padding: "20px" }}>
          <h3 style={{ fontSize: 18, fontWeight: 700, marginBottom: 12, color: "#ef4444", display: "flex", alignItems: "center", gap: 8 }}>
            <span>🤖</span> AI Transformer Analysis (Live)
          </h3>
          <p style={{ fontSize: 12, color: "#94a3b8", marginBottom: 16 }}>
            Top risk signals detected by the neural network trained on US-Iran war patterns.
          </p>
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            {aiAnalysis.map((ai, idx) => (
              <div key={idx} style={{ background: "#0f172a", borderRadius: 8, padding: "12px", display: "flex", alignItems: "center", gap: 12, border: "1px solid #1e293b" }}>
                <div style={{ 
                  minWidth: "45px", height: "45px", borderRadius: "50%", 
                  background: `rgba(239, 68, 68, ${ai.ai_risk_score/100})`,
                  display: "flex", alignItems: "center", justifyContent: "center",
                  fontSize: 10, fontWeight: 800, color: "#fff", border: "2px solid #ef4444"
                }}>
                  {ai.ai_risk_score}%
                </div>
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: 13, fontWeight: 600, color: "#f1f5f9", lineHeight: 1.3 }}>{ai.translated_title || ai.title}</div>
                  <div style={{ fontSize: 10, color: "#64748b", marginTop: 4 }}>{ai.date} • <a href={ai.url} target="_blank" style={{ color: "#3b82f6" }}>View Source</a></div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* INTELLIGENCE FEED */}
      <div style={{ marginBottom: 20 }}>
        <h3 style={{ fontSize: 18, fontWeight: 700, marginBottom: 16 }}>Intelligence Feed</h3>
        <div style={{ display: "flex", flexDirection: "column", gap: 10, marginBottom: 20 }}>
          <input 
            type="text" placeholder="Search articles..." value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            style={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 8, padding: "10px 14px", color: "#f8fafc", fontSize: 14, outline: "none" }}
          />
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: 12, maxHeight: "500px", overflowY: "auto" }}>
          {filteredArticles.map((art, idx) => (
            <div key={idx} style={{ background: "#0f172a", border: "1px solid #1e293b", borderLeft: `4px solid ${LEVEL_COLOR[art.risk_level]}`, borderRadius: 8, padding: "14px" }}>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                <span style={{ fontSize: 10, fontWeight: 700, color: LEVEL_COLOR[art.risk_level] }}>{art.risk_level} — {art.risk_score}</span>
                <span style={{ fontSize: 10, color: "#64748b" }}>{art.date}</span>
              </div>
              <div style={{ fontSize: 14, fontWeight: 600, color: "#f1f5f9", marginBottom: 4 }}>{art.title}</div>
              <div style={{ fontSize: 11, color: "#64748b" }}>Source: {art.source_name}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
