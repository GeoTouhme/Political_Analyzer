import { useState, useEffect } from "react";
import {
  ComposedChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ReferenceLine, ResponsiveContainer, Area, Brush
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
      <div style={{ fontWeight: 700, marginBottom: 4 }}>{d.fullDate}</div>
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
  const [searchTerm, setSearchTerm] = useState("");
  const [filterLevel, setFilterLevel] = useState("ALL");

  useEffect(() => {
    fetch("/analysis_report_v2.json")
      .then(res => res.json())
      .then(report => {
        if (report.daily_breakdown) {
          setData(report.daily_breakdown.map(d => ({
            date: d.date.substring(5), 
            fullDate: d.date,
            avg: d.anchored_risk || d.avg_risk || 0,
            articles: d.article_count || 0,
            level: d.risk_level || "UNKNOWN"
          })));
        }
        if (report.articles) setArticles(report.articles);
        if (report.ai_deep_analysis) setAiAnalysis(report.ai_deep_analysis);
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

  const filteredArticles = articles.filter(a => {
    const matchesSearch = a.title.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesLevel = filterLevel === "ALL" || a.risk_level === filterLevel;
    return matchesSearch && matchesLevel;
  }).slice(0, 100); 

  return (
    <div style={{
      background: "#020817", minHeight: "100vh", padding: "20px",
      fontFamily: "'Inter', sans-serif", color: "#e2e8f0",
      maxWidth: "100%", overflowX: "hidden"
    }}>
      <div style={{ maxWidth: "800px", margin: "0 auto" }}>
        {/* Header */}
        <div style={{ marginBottom: 20, textAlign: 'center' }}>
          <h1 style={{ fontSize: 22, fontWeight: 800, color: "#f8fafc" }}>
            PGCM - Persian Gulf Conflict Monitor
          </h1>
        </div>

        {/* Radar Card */}
        <div style={{
          background: "rgba(15, 23, 42, 0.8)", border: "1px solid #1e293b", borderRadius: 16,
          padding: "16px 20px", marginBottom: 24, display: "flex", alignItems: "center", justifyContent: "space-between"
        }}>
          <div style={{ display: "flex", alignItems: "center", gap: 15 }}>
            <div style={{ width: "60px", height: "40px" }}>
              <svg viewBox="0 0 100 60">
                <path d="M 10 50 A 40 40 0 0 1 90 50" fill="none" stroke="#1e293b" strokeWidth="10" />
                <path d="M 10 50 A 40 40 0 0 1 90 50" fill="none" stroke={LEVEL_COLOR[latestDay.level]} strokeWidth="10" 
                      strokeDasharray={`${(latestDay.avg / 100) * 126} 126`} />
              </svg>
            </div>
            <div>
              <div style={{ fontSize: 24, fontWeight: 900 }}>{latestDay.avg}%</div>
              <div style={{ fontSize: 10, fontWeight: 700, color: LEVEL_COLOR[latestDay.level] }}>{latestDay.level} RISK</div>
            </div>
          </div>
          <div style={{ textAlign: "right" }}>
            <div style={{ fontSize: 9, color: "#64748b" }}>STREAMS ANALYZED</div>
            <div style={{ fontSize: 16, fontWeight: 800, color: "#58a6ff" }}>{meta.article_count || articles.length}</div>
          </div>
        </div>

        {/* Fixed Chart with Brush */}
        <div style={{ 
          background: "#0f172a", border: "1px solid #1e293b", borderRadius: 16, 
          padding: "20px 10px 10px", marginBottom: 30
        }}>
          <div style={{ height: "300px", width: "100%" }}>
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={data} margin={{ top: 5, right: 5, left: -20, bottom: 0 }}>
                <defs>
                  <linearGradient id="areaGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#3b82f6" stopOpacity={0.2} />
                    <stop offset="100%" stopColor="#3b82f6" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                <ReferenceLine y={75} stroke="#ef4444" strokeDasharray="4 3" />
                <ReferenceLine y={50} stroke="#f97316" strokeDasharray="4 3" />
                <XAxis dataKey="date" tick={{ fontSize: 10, fill: "#64748b" }} hide={false} />
                <YAxis domain={[0, 100]} tick={{ fontSize: 10, fill: "#64748b" }} />
                <Tooltip content={<CustomTooltip />} />
                <Area type="monotone" dataKey="avg" fill="url(#areaGrad)" stroke="none" />
                <Line type="monotone" dataKey="avg" stroke="#3b82f6" strokeWidth={3} dot={false} activeDot={{ r: 6 }} />
                <Brush 
                  dataKey="fullDate" 
                  height={30} 
                  stroke="#3b82f6" 
                  fill="#020817"
                  startIndex={data.length > 20 ? data.length - 20 : 0}
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* AI Deep Analysis */}
        {aiAnalysis.length > 0 && (
          <div style={{ marginBottom: 40, background: "rgba(239, 68, 68, 0.05)", border: "1px solid rgba(239, 68, 68, 0.2)", borderRadius: 16, padding: "20px" }}>
            <h3 style={{ fontSize: 18, fontWeight: 700, marginBottom: 12, color: "#ef4444" }}>🤖 AI Transformer Analysis (Live)</h3>
            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              {aiAnalysis.map((ai, idx) => (
                <div key={idx} style={{ background: "#0f172a", borderRadius: 8, padding: "12px", display: "flex", alignItems: "center", gap: 12, border: "1px solid #1e293b" }}>
                  <div style={{ minWidth: "45px", height: "45px", borderRadius: "50%", background: `rgba(239, 68, 68, ${ai.ai_risk_score/100})`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 10, fontWeight: 800, color: "#fff", border: "2px solid #ef4444" }}>
                    {ai.ai_risk_score}%
                  </div>
                  <div style={{ flex: 1 }}>
                    <div style={{ fontSize: 13, fontWeight: 600, color: "#f1f5f9" }}>{ai.translated_title || ai.title}</div>
                    <div style={{ fontSize: 10, color: "#64748b" }}>{ai.date} • <a href={ai.url} target="_blank" style={{ color: "#3b82f6", textDecoration: "none" }}>Source</a></div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Feed */}
        <div style={{ marginBottom: 20 }}>
          <h3 style={{ fontSize: 18, fontWeight: 700, marginBottom: 16 }}>Intelligence Feed</h3>
          <div style={{ display: "flex", flexDirection: "column", gap: 10, marginBottom: 20 }}>
            <input 
              type="text" placeholder="Search..." value={searchTerm} onChange={(e) => setSearchTerm(e.target.value)}
              style={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 8, padding: "10px", color: "#fff", outline: "none" }}
            />
            <div style={{ display: "flex", gap: 5, flexWrap: "wrap" }}>
              {["ALL", "CRITICAL", "HIGH", "MEDIUM", "LOW"].map(lvl => (
                <button key={lvl} onClick={() => setFilterLevel(lvl)} style={{ padding: "5px 10px", borderRadius: "15px", fontSize: 10, background: filterLevel === lvl ? LEVEL_COLOR[lvl] : "#0f172a", color: "#fff", border: "1px solid #1e293b", cursor: "pointer" }}>{lvl}</button>
              ))}
            </div>
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            {filteredArticles.map((art, idx) => (
              <div key={idx} style={{ background: "#0f172a", borderLeft: `4px solid ${LEVEL_COLOR[art.risk_level]}`, borderRadius: 8, padding: "14px" }}>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: "#64748b", marginBottom: 4 }}>
                  <span>{art.risk_level} — {art.risk_score}</span>
                  <span>{art.date}</span>
                </div>
                <div style={{ fontSize: 14, fontWeight: 600 }}>{art.title}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
