"""Generate MEWS architecture diagram as a PNG image."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(20, 14))
ax.set_xlim(0, 20)
ax.set_ylim(0, 14)
ax.axis("off")
fig.patch.set_facecolor("#fafafa")

# ── Colors ──
C_SRC = "#e3f2fd"
C_SRC_B = "#1565c0"
C_ING = "#fff3e0"
C_ING_B = "#e65100"
C_FEAT = "#e8f5e9"
C_FEAT_B = "#2e7d32"
C_RISK = "#fce4ec"
C_RISK_B = "#c62828"
C_OUT = "#f3e5f5"
C_OUT_B = "#6a1b9a"
C_PIPE = "#fffde7"
C_PIPE_B = "#f57f17"
C_BOX = "#ffffff"
TXT = "#212121"

def draw_section(x, y, w, h, color, border, label):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                          facecolor=color, edgecolor=border, linewidth=2, zorder=1)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h - 0.22, label, ha="center", va="top",
            fontsize=11, fontweight="bold", color=border, zorder=5)

def draw_box(x, y, w, h, text, border="#90a4ae", fill=C_BOX, fontsize=8):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                          facecolor=fill, edgecolor=border, linewidth=1.2, zorder=3)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=TXT, zorder=5, linespacing=1.3)

def arrow(x1, y1, x2, y2, color="#78909c"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.4}, zorder=4)

# ════════════════════════════════════════════
#  ROW 1 — Data Sources  (y 11.2 → 13.4)
# ════════════════════════════════════════════
draw_section(0.3, 11.2, 19.4, 2.3, C_SRC, C_SRC_B, "Data Sources")

src_labels = [
    "Market Prices\n(Yahoo Finance, Stooq)",
    "Volatility Indices\n(VIX)",
    "Macro Rates\n(FRED)",
    "Financial News\n(Common Crawl, RSS)",
]
src_xs = [1.0, 5.5, 10.0, 14.5]
for i, lbl in enumerate(src_labels):
    draw_box(src_xs[i], 11.5, 4.0, 1.4, lbl, border=C_SRC_B)

# ════════════════════════════════════════════
#  ROW 2 — Data Ingestion  (y 7.7 → 10.7)
# ════════════════════════════════════════════
draw_section(0.3, 7.7, 19.4, 3.2, C_ING, C_ING_B, "Data Ingestion Layer")

adapt_labels = [
    "market_prices\nadapter",
    "volatility_indices\nadapter",
    "macro_rates\nadapter",
    "financial_news\nadapter",
]
for i, lbl in enumerate(adapt_labels):
    draw_box(src_xs[i], 9.6, 4.0, 1.0, lbl, border=C_ING_B, fill="#fff8e1")

# Arrows from sources to adapters
for x in src_xs:
    arrow(x + 2.0, 11.5, x + 2.0, 10.6)

ing_process = [
    ("Schema Validation\n(datasets.yaml)", 1.5),
    ("Time Alignment\n(UTC / NYSE close)", 7.5),
    ("Versioned Datasets\n(Parquet)", 13.5),
]
for lbl, x in ing_process:
    draw_box(x, 8.05, 4.5, 1.1, lbl, border=C_ING_B, fill="#fff8e1")

# Arrows adapters → validation
for x in src_xs:
    arrow(x + 2.0, 9.6, 3.75, 9.15)

# validation → alignment → datasets
arrow(6.0, 8.6, 7.5, 8.6)
arrow(12.0, 8.6, 13.5, 8.6)

# ════════════════════════════════════════════
#  ROW 3 — Feature Services  (y 4.8 → 7.2)
# ════════════════════════════════════════════
draw_section(0.3, 4.8, 19.4, 2.6, C_FEAT, C_FEAT_B, "Feature Services")

feat_labels = [
    "Numeric Features\n· Realized Volatility\n· Max Drawdown\n· Liquidity",
    "Graph Features\n· Pairwise Correlation\n· Network Metrics\n· Returns Co-movement",
    "Sentiment Features\n· FinBERT Inference\n· Aggregation\n· Sentiment Mapping",
]
feat_xs = [1.0, 7.0, 13.0]
for i, lbl in enumerate(feat_labels):
    draw_box(feat_xs[i], 5.05, 5.5, 2.0, lbl, border=C_FEAT_B, fill="#e8f5e9")

# arrows datasets → features
for fx in feat_xs:
    arrow(15.75, 8.05, fx + 2.75, 7.05)

# ════════════════════════════════════════════
#  ROW 4 — Risk Engine  (y 1.8 → 4.3)
# ════════════════════════════════════════════
draw_section(0.3, 1.8, 19.4, 2.7, C_RISK, C_RISK_B, "Risk Engine")

draw_box(1.0, 2.1, 5.0, 1.9, "Heuristic Engine\n(rule-based subscores,\nnormalization, weights)\nWeight: 35%",
         border=C_RISK_B, fill="#ffebee")
draw_box(7.0, 2.1, 5.0, 1.9, "ML Engine\n(Random Forest / XGBoost,\ntrained models)\nWeight: 65%",
         border=C_RISK_B, fill="#ffebee")
draw_box(13.0, 2.1, 6.0, 1.9, "Calibrated Ensemble\n· Isotonic calibration\n· Weighted combination\n· Exponential smoothing",
         border=C_RISK_B, fill="#ffcdd2")

# arrows features → risk engines
for fx in feat_xs:
    arrow(fx + 2.75, 5.05, 3.5, 4.0)
    arrow(fx + 2.75, 5.05, 9.5, 4.0)

# heuristic/ml → ensemble
arrow(6.0, 3.05, 13.0, 3.05)
arrow(12.0, 3.05, 13.0, 3.05)

# ════════════════════════════════════════════
#  ROW 5 — Outputs  (y 0.1 → 1.5)
# ════════════════════════════════════════════
draw_section(0.3, 0.1, 19.4, 1.4, C_OUT, C_OUT_B, "Outputs")

out_labels = [
    "Risk Score ∈ [0, 1]",
    "Regime Classification\nLOW → MOD → HIGH → EXT",
    "SHAP Explanations",
    "Daily Report",
]
out_xs = [1.0, 5.5, 10.5, 15.0]
out_ws = [4.0, 4.5, 4.0, 4.0]
for i, lbl in enumerate(out_labels):
    draw_box(out_xs[i], 0.25, out_ws[i], 0.9, lbl, border=C_OUT_B, fill="#f3e5f5", fontsize=8)

# ensemble → outputs
for i in range(4):
    arrow(16.0, 2.1, out_xs[i] + out_ws[i] / 2, 1.15)

# ── Title ──
ax.text(10, 13.8, "MEWS — Market Early Warning System", ha="center", va="center",
        fontsize=18, fontweight="bold", color="#1a237e",
    bbox={"boxstyle": "round,pad=0.4", "facecolor": "#e8eaf6", "edgecolor": "#3949ab", "linewidth": 2})

plt.tight_layout()
plt.savefig("figures/architecture/mews_architecture_generated.png", dpi=180,
            bbox_inches="tight", facecolor=fig.get_facecolor())
print("Saved to figures/architecture/mews_architecture_generated.png")

