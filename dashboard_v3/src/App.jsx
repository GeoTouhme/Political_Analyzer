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
  }).slice(0, 100); // Limit to top 100 for performance

  return (
    <div style={{
      background: "#020817", minHeight: "100vh", padding: "28px 20px",
      fontFamily: "'Inter', 'Segoe UI', sans-serif", color: "#e2e8f0",
      maxWidth: "800px", margin: "0 auto"
    }}>
      {/* Header */}
      <div style={{ marginBottom: 24, textAlign: 'center' }}>
        <div style={{ fontSize: 10, letterSpacing: 3, color: "#64748b", textTransform: "uppercase", marginBottom: 4 }}>
          Political Pattern Analyzer v2.2
        </div>
        <h1 style={{ fontSize: 24, fontWeight: 800, color: "#f8fafc", margin: 0 }}>
          Strategic Intelligence
        </h1>
        <div style={{ color: "#64748b", fontSize: 12, marginTop: 4 }}>
           Update: {meta.generated_at ? new Date(meta.generated_at).toLocaleDateString() : 'Live'}
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

      {/* 2 Main Stats */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginBottom: 10 }}>
        <div style={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 12, padding: "12px", textAlign: 'center' }}>
          <div style={{ fontSize: 9, color: "#64748b", textTransform: "uppercase", marginBottom: 4 }}>Peak Day</div>
          <div style={{ fontSize: 16, fontWeight: 800, color: "#ef4444" }}>{peakDay.date}: {peakDay.avg}</div>
        </div>
        <div style={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 12, padding: "12px", textAlign: 'center' }}>
          <div style={{ fontSize: 9, color: "#64748b", textTransform: "uppercase", marginBottom: 4 }}>Articles Today</div>
          <div style={{ fontSize: 16, fontWeight: 800, color: "#58a6ff" }}>{latestDay.articles}</div>
        </div>
      </div>

      {/* Centered Situation */}
      <div style={{ 
        background: "#0f172a", border: "1px solid #1e293b", borderRadius: 12, padding: "12px",
        textAlign: 'center', marginBottom: 24 
      }}>
        <div style={{ fontSize: 9, color: "#64748b", textTransform: "uppercase", marginBottom: 4 }}>Today Situation</div>
        <div style={{ fontSize: 18, fontWeight: 800, color: LEVEL_COLOR[latestDay.level] }}>{latestDay.level}</div>
      </div>

      {/* Main Chart */}
      <div style={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 16, padding: "16px 10px 4px", marginBottom: 32 }}>
        <ResponsiveContainer width="100%" height={220}>
          <ComposedChart data={data}>
            <defs>
              <linearGradient id="areaGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#3b82f6" stopOpacity={0.2} />
                <stop offset="100%" stopColor="#3b82f6" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
            <XAxis dataKey="date" tick={{ fontSize: 9, fill: "#64748b" }} interval={Math.floor(data.length / 6)} />
            <YAxis hide domain={[0, 100]} />
            <Tooltip content={<CustomTooltip />} />
            <Area type="monotone" dataKey="avg" fill="url(#areaGrad)" stroke="none" />
            <Line type="monotone" dataKey="avg" stroke="#3b82f6" strokeWidth={2} dot={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* ARTICLES SECTION */}
      <div style={{ marginBottom: 20 }}>
        <h3 style={{ fontSize: 18, fontWeight: 700, marginBottom: 16 }}>Intelligence Feed</h3>
        
        {/* Search & Filters */}
        <div style={{ display: "flex", flexDirection: "column", gap: 10, marginBottom: 20 }}>
          <input 
            type="text" 
            placeholder="Search articles..." 
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            style={{
              background: "#0f172a", border: "1px solid #1e293b", borderRadius: 8,
              padding: "10px 14px", color: "#f8fafc", fontSize: 14, outline: "none"
            }}
          />
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
            {["ALL", "CRITICAL", "HIGH", "MEDIUM", "LOW"].map(lvl => (
              <button
                key={lvl}
                onClick={() => setFilterLevel(lvl)}
                style={{
                  padding: "6px 12px", borderRadius: "20px", fontSize: 11, fontWeight: 600,
                  cursor: "pointer", border: "1px solid #1e293b",
                  background: filterLevel === lvl ? (LEVEL_COLOR[lvl] || "#3b82f6") : "#0f172a",
                  color: filterLevel === lvl ? (lvl === "MEDIUM" ? "#000" : "#fff") : "#94a3b8",
                  transition: "all 0.2s"
                }}
              >
                {lvl}
              </button>
            ))}
          </div>
        </div>

        {/* Article List */}
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          {filteredArticles.length > 0 ? filteredArticles.map((art, idx) => (
            <div key={idx} style={{
              background: "#0f172a", border: "1px solid #1e293b", 
              borderLeft: `4px solid ${LEVEL_COLOR[art.risk_level]}`,
              borderRadius: 8, padding: "14px"
            }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 6 }}>
                <span style={{ fontSize: 10, fontWeight: 700, color: LEVEL_COLOR[art.risk_level], letterSpacing: 0.5 }}>
                  {art.risk_level} — {art.risk_score}
                </span>
                <span style={{ fontSize: 10, color: "#64748b" }}>{art.date}</span>
              </div>
              <div style={{ fontSize: 14, fontWeight: 600, color: "#f1f5f9", marginBottom: 8, lineHeight: 1.4 }}>
                {art.title}
              </div>
              <div style={{ fontSize: 11, color: "#64748b", display: "flex", justifyContent: "space-between" }}>
                <span>Source: <strong style={{color: "#94a3b8"}}>{art.source_name}</strong></span>
              </div>
            </div>
          )) : (
            <div style={{ textAlign: "center", color: "#64748b", padding: "40px 0" }}>No articles found matching filters.</div>
          )}
        </div>
      </div>
    </div>
  );
}
