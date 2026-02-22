import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { FiDownload, FiCopy, FiCheck } from "react-icons/fi";
import "./OutreachEmail.css";

export default function OutreachEmail({ data, buyerId, onRegenerate, loading }) {
    const [tone, setTone] = useState("professional");
    const [copied, setCopied] = useState(false);

    if (!data) return null;

    const tones = ["professional", "friendly", "urgent"];
    const hasError = !!data.error;

    const handleToneChange = (t) => {
        setTone(t);
        onRegenerate(t);
    };

    const handleCopy = async () => {
        const content = `Subject: ${data.subject || ""}\n\n${data.email_body || ""}`;
        try {
            await navigator.clipboard.writeText(content);
            setCopied(true);
            setTimeout(() => setCopied(false), 2000);
        } catch {
            // fallback
            const ta = document.createElement("textarea");
            ta.value = content;
            document.body.appendChild(ta);
            ta.select();
            document.execCommand("copy");
            document.body.removeChild(ta);
            setCopied(true);
            setTimeout(() => setCopied(false), 2000);
        }
    };

    const handleDownload = () => {
        const content = `Subject: ${data.subject || ""}\n\n${data.email_body || ""}`;
        const blob = new Blob([content], { type: "text/plain" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `outreach_${buyerId}.txt`;
        a.click();
        URL.revokeObjectURL(url);
    };

    return (
        <AnimatePresence>
            <motion.div
                className="outreach-wrap"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.35, ease: "easeOut" }}
            >
                <div className="outreach-header glass">
                    <div>
                        <h3 className="outreach-title">✉️ Outreach-Ready Email</h3>
                        <p className="outreach-meta">
                            AI-generated for <strong>{buyerId}</strong>
                            &nbsp;|&nbsp; via {data.generated_by || "template"}
                        </p>
                    </div>
                    <div className="tone-controls">
                        {tones.map((t) => (
                            <button
                                key={t}
                                className={`tone-btn ${tone === t ? "active" : ""}`}
                                onClick={() => handleToneChange(t)}
                                disabled={loading}
                            >
                                {t}
                            </button>
                        ))}
                    </div>
                </div>

                {loading ? (
                    <div className="email-body glass" style={{ textAlign: "center", padding: "40px" }}>
                        <motion.div
                            animate={{ rotate: 360 }}
                            transition={{ repeat: Infinity, duration: 1, ease: "linear" }}
                            style={{ display: "inline-block", fontSize: "1.5rem" }}
                        >
                            ⏳
                        </motion.div>
                        <p style={{ color: "var(--text-muted)", marginTop: "12px", fontSize: "0.85rem" }}>
                            Generating personalized email…
                        </p>
                    </div>
                ) : hasError ? (
                    <div className="outreach-error">Email generation failed: {data.error}</div>
                ) : (
                    <>
                        <div className="outreach-subject">
                            <span className="outreach-subject-label">Subject:</span>
                            <code>{data.subject}</code>
                        </div>
                        <div className="email-body glass">{data.email_body}</div>
                        <div className="email-actions">
                            <button
                                className={`copy-btn ${copied ? "copied" : ""}`}
                                onClick={handleCopy}
                            >
                                {copied ? (
                                    <>
                                        <FiCheck size={16} /> Copied!
                                    </>
                                ) : (
                                    <>
                                        <FiCopy size={16} /> Copy Email
                                    </>
                                )}
                            </button>
                            <button className="download-btn" onClick={handleDownload}>
                                <FiDownload size={16} />
                                Download .txt
                            </button>
                        </div>
                    </>
                )}
            </motion.div>
        </AnimatePresence>
    );
}
