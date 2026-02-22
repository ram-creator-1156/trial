import { useState, useEffect, useCallback, useMemo } from "react";
import Sidebar from "./components/Sidebar";
import SwipeCard from "./components/SwipeCard";
import OutreachEmail from "./components/OutreachEmail";
import {
  checkHealth,
  fetchExporters,
  fetchNextMatch,
  generateOutreach,
  simulateSignal,
} from "./api";
import "./App.css";

/* ── Floating Particles Background ───────── */
function ParticlesBg() {
  const particles = useMemo(() => {
    const colors = [
      "rgba(212, 175, 55, 0.35)",
      "rgba(34, 211, 238, 0.25)",
      "rgba(167, 139, 250, 0.2)",
      "rgba(52, 211, 153, 0.2)",
      "rgba(212, 175, 55, 0.2)",
    ];
    return Array.from({ length: 24 }, (_, i) => ({
      id: i,
      style: {
        "--size": `${Math.random() * 5 + 2}px`,
        "--color": colors[i % colors.length],
        "--duration": `${Math.random() * 14 + 10}s`,
        "--delay": `${Math.random() * 10}s`,
        left: `${Math.random() * 100}%`,
        top: `${Math.random() * 100}%`,
      },
    }));
  }, []);

  return (
    <div className="particles-bg">
      {particles.map((p) => (
        <div key={p.id} className="particle" style={p.style} />
      ))}
    </div>
  );
}

/* ── Confetti Burst on Connect ─────────── */
function ConfettiBurst({ active }) {
  const confetti = useMemo(() => {
    const colors = ["#d4af37", "#22d3ee", "#34d399", "#a78bfa", "#fb923c", "#fb7185"];
    return Array.from({ length: 40 }, (_, i) => ({
      id: i,
      color: colors[i % colors.length],
      left: 50 + (Math.random() - 0.5) * 60,
      delay: Math.random() * 0.3,
      angle: Math.random() * 360,
      distance: 80 + Math.random() * 200,
      size: 4 + Math.random() * 6,
      duration: 0.8 + Math.random() * 0.6,
    }));
  }, []);

  if (!active) return null;

  return (
    <div className="confetti-container">
      {confetti.map((c) => (
        <div
          key={c.id}
          className="confetti-piece"
          style={{
            "--left": `${c.left}%`,
            "--delay": `${c.delay}s`,
            "--angle": `${c.angle}deg`,
            "--distance": `${c.distance}px`,
            "--size": `${c.size}px`,
            "--color": c.color,
            "--duration": `${c.duration}s`,
          }}
        />
      ))}
    </div>
  );
}

export default function App() {
  /* ── State ───────────────────────────────── */
  const [apiLive, setApiLive] = useState(false);
  const [exporterIds, setExporterIds] = useState([]);
  const [selectedExp, setSelectedExp] = useState("");
  const [loadingMatch, setLoadingMatch] = useState(false);

  const [match, setMatch] = useState(null);
  const [cardSkip, setCardSkip] = useState(0);
  const [passed, setPassed] = useState([]);
  const [connected, setConnected] = useState([]);

  const [outreach, setOutreach] = useState(null);
  const [showOutreach, setShowOutreach] = useState(false);
  const [outreachLoading, setOutreachLoading] = useState(false);

  const [toast, setToast] = useState(null);
  const [showConfetti, setShowConfetti] = useState(false);
  const [initialLoadAttempted, setInitialLoadAttempted] = useState(false);

  /* ── Toast helper ───────────────────────── */
  const showToast = useCallback((msg, icon = "🎯") => {
    setToast({ msg, icon });
    setTimeout(() => setToast(null), 3000);
  }, []);

  /* ── Startup: check API + load exporters ── */
  useEffect(() => {
    (async () => {
      try {
        const h = await checkHealth();
        setApiLive(h.engine_ready);
        if (h.engine_ready) {
          const ids = await fetchExporters();
          setExporterIds(ids);
          if (ids.length > 0) {
            setSelectedExp(ids[0]);
          }
        }
      } catch {
        setApiLive(false);
      }
    })();
  }, []);

  /* ── Auto-load first match ── */
  useEffect(() => {
    if (selectedExp && !initialLoadAttempted) {
      setInitialLoadAttempted(true);
      handleLoadMatches();
    }
  }, [selectedExp, initialLoadAttempted, handleLoadMatches]);

  /* ── Load first match for exporter ──────── */
  const handleLoadMatches = useCallback(async () => {
    if (!selectedExp) return;
    setLoadingMatch(true);
    setCardSkip(0);
    setPassed([]);
    setShowOutreach(false);
    setOutreach(null);
    try {
      const m = await fetchNextMatch(selectedExp, 0);
      setMatch(m);
      if (m) showToast(`Match loaded for ${selectedExp}!`);
    } catch {
      showToast("Failed to load match", "❌");
    }
    setLoadingMatch(false);
  }, [selectedExp, showToast]);

  /* ── Swipe: Pass ─────────────────────────── */
  const handlePass = useCallback(async () => {
    if (!match) return;
    const current = match.importer_buyer_id;
    setPassed((p) => [...p, current]);
    setShowOutreach(false);
    setOutreach(null);
    const nextSkip = cardSkip + 1;
    setCardSkip(nextSkip);
    try {
      const m = await fetchNextMatch(selectedExp, nextSkip);
      setMatch(m);
      showToast(`Passed on ${current}`, "⏭️");
    } catch {
      setMatch(null);
      showToast("No more matches", "🏁");
    }
  }, [match, cardSkip, selectedExp, showToast]);

  /* ── Swipe: Connect ──────────────────────── */
  const handleConnect = useCallback(async () => {
    if (!match) return;
    const buyer = match.importer_buyer_id;
    setConnected((c) => {
      if (c.find((x) => x.id === buyer)) return c;
      return [...c, { id: buyer, score: match.total_score, grade: match.grade }];
    });

    // Trigger confetti
    setShowConfetti(true);
    setTimeout(() => setShowConfetti(false), 1500);

    setOutreachLoading(true);
    setShowOutreach(true);
    try {
      const out = await generateOutreach(selectedExp, match, "professional");
      setOutreach(out);
      showToast(`Connected with ${buyer}! Email ready.`, "🎉");
    } catch {
      setOutreach({ error: "Could not generate email" });
    }
    setOutreachLoading(false);
  }, [match, selectedExp, showToast]);

  /* ── Regenerate outreach ─────────────────── */
  const handleRegenerate = useCallback(
    async (tone) => {
      if (!match) return;
      setOutreachLoading(true);
      try {
        const out = await generateOutreach(selectedExp, match, tone);
        setOutreach(out);
      } catch {
        setOutreach({ error: "Regeneration failed" });
      }
      setOutreachLoading(false);
    },
    [match, selectedExp]
  );

  /* ── Fire live signal ────────────────────── */
  const handleFireSignal = useCallback(
    async (scenario, nBuyers) => {
      try {
        const result = await simulateSignal(scenario, nBuyers);
        const up = result.pairs_moved_up || 0;
        const down = result.pairs_moved_down || 0;
        showToast(`Signal fired! ↑${up} pairs  ↓${down} pairs`, "⚡");
        if (selectedExp) {
          const m = await fetchNextMatch(selectedExp, cardSkip);
          setMatch(m);
        }
      } catch {
        showToast("Signal injection failed", "❌");
      }
    },
    [selectedExp, cardSkip, showToast]
  );

  /* ── Render ──────────────────────────────── */
  return (
    <div className="app-layout">
      <ParticlesBg />
      <ConfettiBurst active={showConfetti} />

      <Sidebar
        apiLive={apiLive}
        passedCount={passed.length}
        connected={connected}
        onFireSignal={handleFireSignal}
      />

      <main className="main-content">
        {/* Page header */}
        <div className="page-header">
          <h1 className="page-title">Swipe to Export</h1>
          <p className="page-subtitle">
            AI-powered trade matchmaking · Explainable scoring · Live signal adaptation
          </p>
        </div>

        {/* Empty state (engine offline or empty) */}
        {!match && !loadingMatch && (
          <div className="empty-state glass">
            <div className="empty-icon">🔌</div>
            <h3>Waiting for match engine...</h3>
            <p>
              Please make sure the backend is running to start matching buyers and sellers automatically.
            </p>
          </div>
        )}

        {/* Loading state */}
        {loadingMatch && (
          <div className="empty-state glass">
            <div className="empty-icon loading-pulse">⚡</div>
            <h3>Running AI matchmaking engine…</h3>
            <p>Analyzing 4-pillar compatibility across thousands of importers</p>
          </div>
        )}

        {/* Match card */}
        {match && !loadingMatch && (
          <SwipeCard
            match={match}
            cardIndex={cardSkip}
            onPass={handlePass}
            onConnect={handleConnect}
          />
        )}

        {/* Outreach email */}
        {showOutreach && (
          <div className="outreach-section">
            <OutreachEmail
              data={outreach}
              buyerId={match?.importer_buyer_id || "—"}
              onRegenerate={handleRegenerate}
              loading={outreachLoading}
            />
          </div>
        )}
      </main>

      {/* Toast */}
      {toast && (
        <div className="toast" key={toast.msg}>
          <span className="toast-icon">{toast.icon}</span>
          {toast.msg}
        </div>
      )}
    </div>
  );
}
