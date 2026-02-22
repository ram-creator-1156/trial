import { useState, useCallback } from "react";
import { motion, useMotionValue, useTransform, AnimatePresence } from "framer-motion";
import { FiX, FiHeart, FiChevronDown, FiChevronUp } from "react-icons/fi";
import { HiOutlineSparkles } from "react-icons/hi2";
import RadarChart from "./RadarChart";
import ExplainPanel from "./ExplainPanel";
import "./SwipeCard.css";

const SWIPE_THRESHOLD = 120;

/* ── Animated SVG Score Ring ──────────────── */
function ScoreRing({ score, grade, gradeColor }) {
    const pct = Math.min(score / 100, 1);
    const r = 42;
    const circumference = 2 * Math.PI * r;
    const offset = circumference * (1 - pct);

    return (
        <div className="score-ring-wrap">
            <svg viewBox="0 0 100 100" className="score-svg">
                {/* Track */}
                <circle
                    cx="50" cy="50" r={r}
                    fill="none"
                    stroke="rgba(255,255,255,0.04)"
                    strokeWidth="5"
                />
                {/* Glow layer */}
                <motion.circle
                    cx="50" cy="50" r={r}
                    fill="none"
                    stroke={gradeColor}
                    strokeWidth="6"
                    strokeLinecap="round"
                    strokeDasharray={circumference}
                    initial={{ strokeDashoffset: circumference }}
                    animate={{ strokeDashoffset: offset }}
                    transition={{ duration: 1.2, ease: "easeOut", delay: 0.3 }}
                    transform="rotate(-90 50 50)"
                    filter="url(#glow)"
                    opacity="0.3"
                />
                {/* Main arc */}
                <motion.circle
                    cx="50" cy="50" r={r}
                    fill="none"
                    stroke={gradeColor}
                    strokeWidth="4"
                    strokeLinecap="round"
                    strokeDasharray={circumference}
                    initial={{ strokeDashoffset: circumference }}
                    animate={{ strokeDashoffset: offset }}
                    transition={{ duration: 1.2, ease: "easeOut", delay: 0.3 }}
                    transform="rotate(-90 50 50)"
                />
                <defs>
                    <filter id="glow">
                        <feGaussianBlur stdDeviation="3" result="coloredBlur" />
                        <feMerge>
                            <feMergeNode in="coloredBlur" />
                            <feMergeNode in="SourceGraphic" />
                        </feMerge>
                    </filter>
                </defs>
            </svg>
            <div className="score-center">
                <motion.span
                    className="score-value"
                    style={{ color: gradeColor }}
                    initial={{ opacity: 0, scale: 0.5 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ duration: 0.5, delay: 0.6 }}
                >
                    {Math.round(score)}
                </motion.span>
                <span className="score-max">/ 100</span>
            </div>
        </div>
    );
}

export default function SwipeCard({ match, cardIndex, onPass, onConnect }) {
    const [showExplain, setShowExplain] = useState(false);
    const x = useMotionValue(0);

    const passOpacity = useTransform(x, [-SWIPE_THRESHOLD, 0], [1, 0]);
    const connectOpacity = useTransform(x, [0, SWIPE_THRESHOLD], [0, 1]);
    const rotate = useTransform(x, [-300, 0, 300], [-15, 0, 15]);
    const scale = useTransform(x, [-300, -80, 0, 80, 300], [0.93, 0.98, 1, 0.98, 0.93]);

    const handleDragEnd = useCallback(
        (_, info) => {
            if (info.offset.x < -SWIPE_THRESHOLD) onPass();
            else if (info.offset.x > SWIPE_THRESHOLD) onConnect();
        },
        [onPass, onConnect]
    );

    if (!match) return null;

    const { importer_buyer_id, total_score, grade, explainability_dict } = match;
    const gradeColor = getGradeColor(grade);

    // Handle 'nan' or empty buyer IDs gracefully
    const displayName = (!importer_buyer_id || importer_buyer_id === "nan" || importer_buyer_id === "NaN")
        ? `Importer #${cardIndex + 1}`
        : importer_buyer_id;

    return (
        <div className="swipe-stage">
            <AnimatePresence mode="wait">
                <motion.div
                    key={`card-${cardIndex}-${importer_buyer_id}`}
                    className="swipe-card glass"
                    style={{ x, rotate, scale }}
                    drag="x"
                    dragConstraints={{ left: 0, right: 0 }}
                    dragElastic={0.9}
                    onDragEnd={handleDragEnd}
                    initial={{ scale: 0.9, opacity: 0, y: 30 }}
                    animate={{ scale: 1, opacity: 1, y: 0 }}
                    exit={{ opacity: 0, scale: 0.85, transition: { duration: 0.2 } }}
                    transition={{ type: "spring", stiffness: 300, damping: 30 }}
                >
                    {/* Gradient border glow */}
                    <div className="card-glow" />

                    {/* Pass / Connect overlays */}
                    <motion.div className="swipe-overlay pass-overlay" style={{ opacity: passOpacity }}>
                        <FiX size={48} />
                        <span>PASS</span>
                    </motion.div>
                    <motion.div className="swipe-overlay connect-overlay" style={{ opacity: connectOpacity }}>
                        <FiHeart size={48} />
                        <span>CONNECT</span>
                    </motion.div>

                    {/* Card header */}
                    <div className="card-header">
                        <div className="card-header-left">
                            <span className="card-label">Best Match · Card {cardIndex + 1}</span>
                            <h2 className="buyer-id">🏢 {displayName}</h2>
                            <span
                                className="grade-pill"
                                style={{
                                    color: gradeColor,
                                    borderColor: `${gradeColor}40`,
                                    background: `${gradeColor}12`,
                                }}
                            >
                                ✦ Grade {grade}
                            </span>
                        </div>
                        <ScoreRing score={total_score} grade={grade} gradeColor={gradeColor} />
                    </div>

                    {/* Radar + Metrics */}
                    <div className="card-body">
                        <div className="radar-section">
                            <span className="section-label">Compatibility Radar</span>
                            <RadarChart data={explainability_dict} />
                        </div>
                        <div className="metrics-section">
                            <span className="section-label">Score Breakdown</span>
                            {explainability_dict &&
                                Object.entries(explainability_dict).map(([key, pillar], index) => (
                                    <PillarBar key={key} pillarKey={key} pillar={pillar} delay={index * 0.1} />
                                ))}
                        </div>
                    </div>

                    {/* Explain toggle */}
                    <button
                        className="explain-toggle"
                        onClick={(e) => {
                            e.stopPropagation();
                            setShowExplain(!showExplain);
                        }}
                    >
                        <HiOutlineSparkles size={16} />
                        Why this match? (Explainable AI)
                        {showExplain ? <FiChevronUp size={14} /> : <FiChevronDown size={14} />}
                    </button>

                    <AnimatePresence>
                        {showExplain && (
                            <motion.div
                                initial={{ height: 0, opacity: 0 }}
                                animate={{ height: "auto", opacity: 1 }}
                                exit={{ height: 0, opacity: 0 }}
                                transition={{ duration: 0.3 }}
                                style={{ overflow: "hidden" }}
                            >
                                <ExplainPanel data={explainability_dict} />
                            </motion.div>
                        )}
                    </AnimatePresence>
                </motion.div>
            </AnimatePresence>

            {/* Action buttons */}
            <div className="action-buttons">
                <button className="action-btn pass-btn" onClick={onPass}>
                    <FiX size={24} />
                    <span>Pass</span>
                </button>
                <button className="action-btn connect-btn" onClick={onConnect}>
                    <FiHeart size={24} />
                    <span>Connect</span>
                </button>
            </div>

            <p className="drag-hint">← Drag card to swipe →</p>
        </div>
    );
}

function PillarBar({ pillarKey, pillar, delay = 0 }) {
    const colors = {
        product_compatibility: { color: "#d4af37", glow: "rgba(212, 175, 55, 0.3)" },
        geography_fit: { color: "#60a5fa", glow: "rgba(96, 165, 250, 0.3)" },
        high_intent_signals: { color: "#34d399", glow: "rgba(52, 211, 153, 0.3)" },
        trade_activity: { color: "#fb923c", glow: "rgba(251, 146, 60, 0.3)" },
    };
    const { color, glow } = colors[pillarKey] || { color: "#888", glow: "rgba(136,136,136,0.3)" };
    const pct = pillar.pct || 0;

    return (
        <div className="pillar-bar">
            <div className="pillar-bar-header">
                <span className="pillar-name">{pillar.label}</span>
                <span className="pillar-value">
                    {pillar.score?.toFixed(1)}/{pillar.max_pts}
                </span>
            </div>
            <div className="pillar-track">
                <motion.div
                    className="pillar-fill"
                    style={{
                        background: `linear-gradient(90deg, ${color}, ${color}cc)`,
                        boxShadow: `0 0 12px ${glow}`,
                    }}
                    initial={{ width: 0 }}
                    animate={{ width: `${Math.min(pct, 100)}%` }}
                    transition={{ duration: 0.8, ease: "easeOut", delay: 0.4 + delay }}
                />
            </div>
        </div>
    );
}

function getGradeColor(g) {
    const map = {
        "A+": "#d4af37",
        A: "#34d399",
        "B+": "#60a5fa",
        B: "#a78bfa",
        C: "#fb923c",
        D: "#fb7185",
    };
    return map[g] || "#94a3b8";
}
