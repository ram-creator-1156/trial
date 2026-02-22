import { useState } from "react";
import { FiZap, FiLoader, FiChevronRight } from "react-icons/fi";
import "./Sidebar.css";

const SCENARIOS = {
    bulk_buy: "📦  Bulk Purchase",
    funding_round: "💰  Series-B Funding",
    tariff_cut: "✂️  Tariff Cut",
    linkedin_spike: "📈  Hiring Spike",
    geopolitical_risk: "⚠️  Geopolitical Risk",
    new_trade_deal: "🤝  Trade Deal",
    supply_shock: "🔴  Supply Shock",
};

export default function Sidebar({
    apiLive,
    exporterIds,
    selectedExp,
    onSelectExp,
    onLoadMatches,
    loadingMatch,
    passedCount,
    connected,
    onFireSignal,
}) {
    const [scenario, setScenario] = useState("bulk_buy");
    const [nBuyers, setNBuyers] = useState(5);
    const [firing, setFiring] = useState(false);

    const handleFire = async () => {
        setFiring(true);
        await onFireSignal(scenario, nBuyers);
        setFiring(false);
    };

    return (
        <aside className="sidebar glass">
            {/* Brand */}
            <div className="brand">
                <div className="brand-logo">
                    <div className="brand-logo-inner">TM</div>
                </div>
                <h2 className="brand-name">TradeMatch</h2>
                <p className="brand-sub">AI-Powered Export Matchmaking</p>
            </div>

            {/* Status */}
            <div className={`status-pill ${apiLive ? "online" : "offline"}`}>
                <span className="status-dot" />
                {apiLive ? "Engine Online" : "Engine Offline"}
            </div>

            {!apiLive && (
                <p className="hint">
                    Run: <code>uvicorn backend.main:app --reload</code>
                </p>
            )}

            <hr className="divider" />

            {/* Exporter Picker */}
            <section>
                <h3 className="sidebar-heading">
                    <span className="heading-icon">🏭</span>
                    Select Exporter
                </h3>
                <select
                    className="select-field"
                    value={selectedExp}
                    onChange={(e) => onSelectExp(e.target.value)}
                >
                    <option value="">Choose an exporter…</option>
                    {exporterIds.map((id) => (
                        <option key={id} value={id}>
                            {id}
                        </option>
                    ))}
                </select>
                <button
                    className="primary-btn"
                    onClick={onLoadMatches}
                    disabled={!selectedExp || loadingMatch}
                >
                    {loadingMatch ? (
                        <>
                            <FiLoader className="spin" size={16} /> Scoring…
                        </>
                    ) : (
                        <>
                            🔍 Load Matches
                            <FiChevronRight size={16} />
                        </>
                    )}
                </button>
            </section>

            <hr className="divider" />

            {/* Live Signal */}
            <section>
                <h3 className="sidebar-heading">
                    <span className="heading-icon">⚡</span>
                    Live Market Shift
                </h3>
                <select
                    className="select-field"
                    value={scenario}
                    onChange={(e) => setScenario(e.target.value)}
                >
                    {Object.entries(SCENARIOS).map(([k, label]) => (
                        <option key={k} value={k}>
                            {label}
                        </option>
                    ))}
                </select>
                <div className="slider-group">
                    <label className="slider-label">
                        Buyers affected: <span className="slider-value">{nBuyers}</span>
                    </label>
                    <input
                        type="range"
                        min={1}
                        max={15}
                        value={nBuyers}
                        onChange={(e) => setNBuyers(Number(e.target.value))}
                        className="range-slider"
                    />
                </div>
                <button
                    className="fire-btn"
                    onClick={handleFire}
                    disabled={firing || !apiLive}
                >
                    {firing ? (
                        <>
                            <FiLoader className="spin" size={16} /> Injecting…
                        </>
                    ) : (
                        <>
                            <FiZap size={16} /> Fire Signal
                        </>
                    )}
                </button>
            </section>

            <hr className="divider" />

            {/* Stats */}
            <div className="stats-row">
                <div className="stat-box">
                    <span className="stat-val">{connected.length}</span>
                    <span className="stat-label">Connected</span>
                </div>
                <div className="stat-box">
                    <span className="stat-val">{passedCount}</span>
                    <span className="stat-label">Passed</span>
                </div>
            </div>

            {/* Connected list */}
            {connected.length > 0 && (
                <section className="connected-list">
                    <h3 className="sidebar-heading">
                        <span className="heading-icon">⭐</span>
                        Connected
                    </h3>
                    {connected.map((c) => (
                        <div key={c.id} className="connected-chip">
                            <span className="chip-buyer">🛒 {c.id}</span>
                            <div className="chip-right">
                                <span className="chip-grade">{c.grade}</span>
                                <span className="chip-score">{Math.round(c.score)} pts</span>
                            </div>
                        </div>
                    ))}
                </section>
            )}

            {/* Footer */}
            <div className="sidebar-footer">
                <span>Powered by TradeMatch AI</span>
            </div>
        </aside>
    );
}
