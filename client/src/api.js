// STANDALONE MOCK MODE - No Backend Required
// Since Python 3.14 requires C++ compilers to build pandas, 
// we are mocking the backend to let you show off the premium UI instantly!

const MOCK_EXPORTERS = ["EXP_5094", "EXP_2188", "EXP_9941"];

// Fake match generation so swipe card works endlessly
const generateMockMatch = (cardIndex) => {
  const scoreBase = 85 - (cardIndex * 3);
  return {
    importer_buyer_id: `GlobalTrade Partners #${cardIndex + 1}`,
    total_score: scoreBase > 50 ? scoreBase : 50,
    grade: scoreBase > 80 ? "A" : scoreBase > 65 ? "B" : "C",
    p1_product: (scoreBase * 0.3).toFixed(1),
    p2_geography: (scoreBase * 0.2).toFixed(1),
    p3_signals: (scoreBase * 0.3).toFixed(1),
    p4_activity: (scoreBase * 0.2).toFixed(1),
    explainability_dict: {
      product_compatibility: {
        label: "Product Compatibility",
        max_pts: 30,
        score: (scoreBase * 0.3).toFixed(1),
        pct: 88.5,
        reasons: ["Strong match in HS Codes.", "Similar catalog profile."]
      },
      geography_fit: {
        label: "Geography Fit",
        max_pts: 20,
        score: (scoreBase * 0.2).toFixed(1),
        pct: 75.0,
        reasons: ["Trade route has favorable tariffs.", "Low shipping friction."]
      },
      high_intent_signals: {
        label: "High-Intent Signals",
        max_pts: 30,
        score: (scoreBase * 0.3).toFixed(1),
        pct: 92.4,
        reasons: ["Buyer recently searched your category.", "Active procurement cycle detected."]
      },
      trade_activity_score: {
        label: "Trade Activity",
        max_pts: 20,
        score: (scoreBase * 0.2).toFixed(1),
        pct: 81.0,
        reasons: ["High volume buyer.", "Excellent credit and payment history."]
      }
    }
  };
};

// Simulate network delay
const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));

export async function checkHealth() {
  await delay(300);
  return { engine_ready: true, status: "mocked" };
}

export async function fetchExporters() {
  await delay(400);
  return MOCK_EXPORTERS;
}

export async function fetchNextMatch(exporterId, skip = 0) {
  await delay(800);
  return generateMockMatch(skip);
}

export async function fetchMatches(exporterId, topK = 10) {
  await delay(1000);
  return Array.from({ length: topK }).map((_, i) => generateMockMatch(i));
}

export async function generateOutreach(exporterId, match, tone = "professional") {
  await delay(1500); // Simulate AI generation time
  return {
    subject: `Partnership Opportunity: Premium Supplies for ${match.importer_buyer_id}`,
    body: `Hello Procurement Team,\n\nBased on your recent supply chain activity, our AI matchmaking engine identified a critical synergy between our catalog and your sourcing needs.\n\nOur system gave us a Compatibility Score of ${match.total_score}/100, driven largely by our strong Product Fit and excellent Geographical trade routing.\n\nWe would love to discuss a bulk supply agreement. Let us know when you have 10 minutes to chat.\n\nBest,\n${exporterId} Trade Team`,
    metadata: { tone: tone, llm_used: "mock_model" }
  };
}

export async function simulateSignal(scenario, nBuyers = 5) {
  await delay(1000);
  return {
    pairs_moved_up: Math.floor(Math.random() * nBuyers) + 1,
    pairs_moved_down: Math.floor(Math.random() * 3),
  };
}
