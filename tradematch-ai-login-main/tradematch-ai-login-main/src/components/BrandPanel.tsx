import { Globe, Zap, Target } from "lucide-react";
import tradePatternBg from "@/assets/trade-pattern-bg.png";

const features = [
  {
    step: 1,
    icon: Globe,
    title: "Discover High-Intent Buyers",
    description: "AI-curated global opportunities tailored to your product category.",
  },
  {
    step: 2,
    icon: Zap,
    title: "Smart Outreach Automation",
    description: "Multi-channel engagement via Email, LinkedIn, and WhatsApp.",
  },
  {
    step: 3,
    icon: Target,
    title: "Close More Deals Faster",
    description: "Focus on negotiations while AI manages pipeline scoring.",
  },
];

const BrandPanel = () => {
  return (
    <div className="relative hidden lg:flex flex-col justify-between gradient-brand p-12 xl:p-16 overflow-hidden min-h-screen">
      {/* Background pattern */}
      <div
        className="absolute inset-0 opacity-30"
        style={{
          backgroundImage: `url(${tradePatternBg})`,
          backgroundSize: "cover",
          backgroundPosition: "center",
        }}
      />

      {/* Content */}
      <div className="relative z-10 flex flex-col justify-center flex-1 max-w-lg">
        <div className="mb-6">
          <span className="inline-flex items-center gap-2 text-sm font-medium text-primary-foreground/70 tracking-widest uppercase">
            <Globe className="w-4 h-4" />
            TradeMatch AI
          </span>
        </div>

        <h1 className="text-4xl xl:text-5xl font-bold text-primary-foreground leading-tight mb-4 tracking-tight">
          AI-Powered Global Trade Engine
        </h1>
        <p className="text-lg text-primary-foreground/70 mb-12 leading-relaxed">
          Scale your exports with intelligent buyer discovery and automated outreach.
        </p>

        <div className="space-y-6">
          {features.map((feature) => (
            <div
              key={feature.step}
              className="flex items-start gap-4 group"
            >
              <div className="flex-shrink-0 w-10 h-10 rounded-xl bg-white/10 backdrop-blur-sm flex items-center justify-center border border-white/10 group-hover:bg-white/15 transition-colors">
                <feature.icon className="w-5 h-5 text-primary-foreground" />
              </div>
              <div>
                <div className="flex items-center gap-2 mb-1">
                  <span className="text-xs font-semibold text-primary-foreground/40 uppercase tracking-wider">
                    Step {feature.step}
                  </span>
                </div>
                <h3 className="text-base font-semibold text-primary-foreground mb-0.5">
                  {feature.title}
                </h3>
                <p className="text-sm text-primary-foreground/60 leading-relaxed">
                  {feature.description}
                </p>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Footer */}
      <p className="relative z-10 text-sm text-primary-foreground/40 mt-8">
        © 2026 TradeSync AI. All rights reserved.
      </p>
    </div>
  );
};

export default BrandPanel;
