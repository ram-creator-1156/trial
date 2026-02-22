import "./ExplainPanel.css";

const PILLAR_ICONS = {
    product_compatibility: "🎯",
    geography_fit: "🌍",
    high_intent_signals: "📊",
    trade_activity: "📦",
};

const PILLAR_COLORS = {
    product_compatibility: "#d4af37",
    geography_fit: "#60a5fa",
    high_intent_signals: "#34d399",
    trade_activity: "#fb923c",
};

export default function ExplainPanel({ data }) {
    if (!data) return null;

    return (
        <div className="explain-panel">
            {Object.entries(data).map(([key, pillar]) => (
                <div
                    key={key}
                    className="explain-pillar"
                    style={{ borderLeftColor: PILLAR_COLORS[key] || "var(--gold-glow)" }}
                >
                    <h4 className="explain-pillar-title">
                        <span>{PILLAR_ICONS[key] || "✦"}</span>
                        {pillar.label}
                    </h4>
                    <ul className="explain-reasons">
                        {(pillar.reasons || []).map((r, i) => (
                            <li key={i} className="explain-reason">
                                {r.replace(/<->/g, "↔").replace(/->/g, "→")}
                            </li>
                        ))}
                    </ul>
                </div>
            ))}
        </div>
    );
}
