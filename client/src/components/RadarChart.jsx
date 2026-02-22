import {
    Radar,
    RadarChart as ReRadarChart,
    PolarGrid,
    PolarAngleAxis,
    PolarRadiusAxis,
    ResponsiveContainer,
} from "recharts";

// Short labels for radar chart axes
const SHORT_LABELS = {
    "Product Compatibility": "Product",
    "Geography Fit": "Geo Fit",
    "High-Intent Signals": "Intent",
    "Trade Activity": "Trade",
};

export default function RadarChart({ data }) {
    if (!data) return null;

    const chartData = Object.entries(data).map(([key, pillar]) => ({
        subject: SHORT_LABELS[pillar.label] || pillar.label || key,
        value: pillar.pct || 0,
        fullMark: 100,
    }));

    return (
        <ResponsiveContainer width="100%" height={210}>
            <ReRadarChart cx="50%" cy="50%" outerRadius="55%" data={chartData}>
                <defs>
                    <linearGradient id="radarFill" x1="0" y1="0" x2="1" y2="1">
                        <stop offset="0%" stopColor="#d4af37" stopOpacity={0.3} />
                        <stop offset="50%" stopColor="#22d3ee" stopOpacity={0.15} />
                        <stop offset="100%" stopColor="#d4af37" stopOpacity={0.05} />
                    </linearGradient>
                    <linearGradient id="radarStroke" x1="0" y1="0" x2="1" y2="1">
                        <stop offset="0%" stopColor="#d4af37" />
                        <stop offset="100%" stopColor="#22d3ee" />
                    </linearGradient>
                    <filter id="radarGlow">
                        <feGaussianBlur stdDeviation="2" result="coloredBlur" />
                        <feMerge>
                            <feMergeNode in="coloredBlur" />
                            <feMergeNode in="SourceGraphic" />
                        </feMerge>
                    </filter>
                </defs>
                <PolarGrid
                    stroke="rgba(255,255,255,0.04)"
                    gridType="polygon"
                />
                <PolarAngleAxis
                    dataKey="subject"
                    tick={{ fill: "#94a3b8", fontSize: 11, fontWeight: 500 }}
                    tickLine={false}
                />
                <PolarRadiusAxis
                    angle={30}
                    domain={[0, 100]}
                    tick={false}
                    axisLine={false}
                />
                <Radar
                    name="Compatibility"
                    dataKey="value"
                    stroke="url(#radarStroke)"
                    fill="url(#radarFill)"
                    strokeWidth={2.5}
                    dot={{ r: 5, fill: "#d4af37", stroke: "#111827", strokeWidth: 2 }}
                    filter="url(#radarGlow)"
                    animationDuration={1200}
                    animationEasing="ease-out"
                />
            </ReRadarChart>
        </ResponsiveContainer>
    );
}
