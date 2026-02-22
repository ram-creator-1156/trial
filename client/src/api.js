import axios from "axios";

const API = axios.create({ baseURL: "http://localhost:8000" });

export async function checkHealth() {
  const { data } = await API.get("/health");
  return data;
}

export async function fetchExporters() {
  const { data } = await API.get("/api/exporters", { params: { limit: 200 } });
  return data.exporter_ids || [];
}

export async function fetchNextMatch(exporterId, skip = 0) {
  const { data } = await API.get(`/api/next_match/${exporterId}`, {
    params: { skip },
    timeout: 120000,
  });
  return data.match || null;
}

export async function fetchMatches(exporterId, topK = 10) {
  const { data } = await API.get(`/api/matches/${exporterId}`, {
    params: { top_k: topK },
    timeout: 120000,
  });
  return data;
}

export async function generateOutreach(exporterId, match, tone = "professional") {
  const payload = {
    exporter: {
      exporter_id: exporterId,
      company_name: exporterId,
      industry: "General Trade",
      state: "India",
    },
    importer: {
      buyer_id: match.importer_buyer_id || "",
      company_name: match.importer_buyer_id || "",
      country: "International",
      industry: "General Trade",
      avg_order_tons: match.avg_order_tons || null,
      revenue_usd: match.revenue_usd || null,
      response_probability: match.response_probability || null,
    },
    match_score: match.total_score,
    explanation: match.explainability_dict || {},
    tone,
    use_llm: false,
  };
  const { data } = await API.post("/api/generate_outreach", payload, { timeout: 15000 });
  return data;
}

export async function simulateSignal(scenario, nBuyers = 5) {
  const { data } = await API.post(
    "/api/simulate_real_time",
    { scenario, n_buyers: nBuyers, top_k: 10, seed: 0 },
    { timeout: 120000 }
  );
  return data;
}
