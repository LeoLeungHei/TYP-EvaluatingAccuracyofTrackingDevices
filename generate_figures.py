"""Generate all report figures:
  Figure 3.1  — System Architecture Diagram
  Table  3.1  — API Endpoints
  Figure 5.1  — Quality Score Summary Across Subjects
  Figure 5.2  — Per-Sensor Quality Breakdown (Subject S2)
  Figure 5.3  — Signal Validity (Wrist E4 vs Chest RespiBAN)
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ── shared style ─────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
})

LAYER_COLOURS = {
    "data":     "#dce6f1",   # light blue
    "backend":  "#e2efda",   # light green
    "frontend": "#fce4d6",   # light orange
}
BOX_COLOUR  = "#ffffff"
BORDER      = "#4472c4"
ARROW_CLR   = "#333333"


# ══════════════════════════════════════════════════════════════════
# Figure 3.1 — System Architecture Diagram
# ══════════════════════════════════════════════════════════════════
def draw_architecture():
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.5)
    ax.axis("off")

    # ── Helper: rounded box with label ───────────────────────────
    def box(x, y, w, h, label, colour=BOX_COLOUR, border=BORDER,
            fontsize=9, bold=False, sublabel=None):
        rect = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.1",
            facecolor=colour, edgecolor=border, linewidth=1.2,
        )
        ax.add_patch(rect)
        weight = "bold" if bold else "normal"
        ax.text(x + w / 2, y + h / 2 + (0.1 if sublabel else 0),
                label, ha="center", va="center",
                fontsize=fontsize, fontweight=weight, color="#1a1a1a")
        if sublabel:
            ax.text(x + w / 2, y + h / 2 - 0.15, sublabel,
                    ha="center", va="center",
                    fontsize=7.5, color="#555555", style="italic")

    def arrow(x1, y1, x2, y2, label=None, colour=ARROW_CLR, label_offset=0.15):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", color=colour, lw=1.5),
        )
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx, my + label_offset, label, ha="center", va="bottom",
                    fontsize=7.5, color="#555555", style="italic")

    # ── Layer backgrounds ────────────────────────────────────────
    # Data layer (left column)
    ax.add_patch(FancyBboxPatch(
        (0.1, 1.1), 2.8, 3.7,
        boxstyle="round,pad=0.15",
        facecolor=LAYER_COLOURS["data"], edgecolor="#8eaadb",
        linewidth=1, linestyle="--", alpha=0.5,
    ))
    ax.text(1.5, 4.65, "Data Processing Layer",
            ha="center", va="bottom", fontsize=9,
            fontweight="bold", color="#2e5090")

    # Backend layer (middle column)
    ax.add_patch(FancyBboxPatch(
        (3.4, 1.1), 2.9, 3.7,
        boxstyle="round,pad=0.15",
        facecolor=LAYER_COLOURS["backend"], edgecolor="#a9d18e",
        linewidth=1, linestyle="--", alpha=0.5,
    ))
    ax.text(4.85, 4.65, "API Layer (Flask)",
            ha="center", va="bottom", fontsize=9,
            fontweight="bold", color="#548235")

    # Frontend layer (right column)
    ax.add_patch(FancyBboxPatch(
        (6.8, 1.1), 3.1, 3.7,
        boxstyle="round,pad=0.15",
        facecolor=LAYER_COLOURS["frontend"], edgecolor="#f4b183",
        linewidth=1, linestyle="--", alpha=0.5,
    ))
    ax.text(8.35, 4.65, "Frontend Layer (React)",
            ha="center", va="bottom", fontsize=9,
            fontweight="bold", color="#c55a11")

    # ── Data layer boxes (stacked vertically) ────────────────────
    box(0.3, 3.7, 2.4, 0.7, "E4 CSV Files",
        sublabel="ACC · BVP · EDA · TEMP", bold=True)
    box(0.3, 2.55, 2.4, 0.7, "Data Loader",
        sublabel="Timestamps · Sample rates", fontsize=8.5)
    box(0.3, 1.4, 2.4, 0.7, "Quality Monitor",
        sublabel="NumPy · SciPy · Pandas", bold=True, fontsize=8.5)

    arrow(1.5, 3.7, 1.5, 3.27)    # CSV → Loader
    arrow(1.5, 2.55, 1.5, 2.12)   # Loader → Monitor

    # ── Backend boxes (stacked vertically) ───────────────────────
    box(3.6, 3.7, 2.5, 0.7, "REST Endpoints",
        sublabel="/subjects · /overview · /quality", fontsize=8.5)
    box(3.6, 2.55, 2.5, 0.7, "SSE Stream",
        sublabel="/api/stream/<subject>", fontsize=9)
    box(3.6, 1.4, 2.5, 0.7, "Data Cache",
        sublabel="In-memory per subject", fontsize=9)

    # cache feeds up to both endpoints
    arrow(4.5, 2.12, 4.5, 2.53, colour="#70ad47")
    arrow(5.2, 2.12, 5.2, 3.68, colour="#70ad47")

    # ── Frontend boxes (stacked vertically) ──────────────────────
    box(7.0, 3.7, 2.7, 0.7, "AggregateScore + MetricCards",
        sublabel="Per-sensor quality breakdown", fontsize=8)
    box(7.0, 2.55, 2.7, 0.7, "RealtimeChart",
        sublabel="Plotly.js interactive timeline", fontsize=8.5)
    box(7.0, 1.4, 2.7, 0.7, "App (Controller)",
        sublabel="EventSource · React state", fontsize=8.5)

    # App distributes state up to components
    arrow(8.0, 2.12, 8.0, 2.53, colour="#c55a11")
    arrow(8.7, 2.12, 8.7, 3.68, colour="#c55a11")

    # ── Cross-layer arrows ───────────────────────────────────────
    # Quality Monitor → REST/SSE
    arrow(2.72, 2.0, 3.58, 3.0, label="quality\nmetrics", label_offset=0.05)

    # REST → Frontend top
    arrow(6.12, 4.05, 6.98, 4.05, label="JSON")

    # SSE → Frontend App
    arrow(6.12, 2.9, 6.98, 1.75, label="SSE events", label_offset=0.12)

    # Caption
    ax.text(5.0, 0.5,
            "Figure 3.1: System Architecture — three-layer client-server design",
            ha="center", va="center", fontsize=10, style="italic",
            color="#333333")

    fig.tight_layout()
    fig.savefig("figure_3_1_architecture.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    print("Saved figure_3_1_architecture.png")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════
# Table 3.1 — API Endpoints
# ══════════════════════════════════════════════════════════════════
def draw_table():
    fig, ax = plt.subplots(figsize=(8, 2.8))
    ax.axis("off")

    columns = ["Endpoint", "Method", "Description"]
    rows = [
        ["/api/subjects",                     "GET", "Lists available subjects with E4 data"],
        ["/api/overview/<subject_id>",        "GET", "Recording metadata: duration, sample rates, sample counts"],
        ["/api/quality/<subject_id>",         "GET", "Full quality timeline via sliding window (params: window, step)"],
        ["/api/signals/<subject_id>/<sensor>","GET", "Downsampled raw signal values for plotting (param: max_points)"],
        ["/api/stream/<subject_id>",          "GET", "SSE stream replaying quality scores in real time (params: window, step, speed)"],
    ]

    table = ax.table(
        cellText=rows,
        colLabels=columns,
        loc="center",
        cellLoc="left",
        colWidths=[0.34, 0.08, 0.58],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)

    # Style header row
    for j in range(len(columns)):
        cell = table[0, j]
        cell.set_facecolor("#4472c4")
        cell.set_text_props(color="white", fontweight="bold")
        cell.set_edgecolor("#2e5090")

    # Alternate row colours
    for i in range(1, len(rows) + 1):
        for j in range(len(columns)):
            cell = table[i, j]
            cell.set_edgecolor("#b4c6e7")
            if i % 2 == 0:
                cell.set_facecolor("#dce6f1")
            else:
                cell.set_facecolor("#ffffff")
            # Make endpoint column monospaced-looking
            if j == 0:
                cell.set_text_props(family="monospace", fontsize=8)

    ax.text(0.5, -0.02,
            "Table 3.1: REST and SSE API endpoints exposed by the Flask backend",
            ha="center", va="top", fontsize=10, style="italic",
            color="#333333", transform=ax.transAxes)

    fig.tight_layout()
    fig.savefig("table_3_1_endpoints.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    print("Saved table_3_1_endpoints.png")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════
# Figure 5.1 — Quality Score Summary Across Subjects
# ══════════════════════════════════════════════════════════════════
def draw_quality_summary():
    subjects = ['S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10']
    mean_overall    = [82.5, 81.8, 82.6, 80.2, 81.7, 80.6, 83.2, 81.6, 82.1]
    min_score       = [76.8, 74.9, 78.0, 65.6, 77.4, 66.7, 77.6, 74.6, 76.2]
    max_score       = [89.5, 88.7, 91.1, 88.7, 90.4, 88.6, 90.6, 89.9, 90.0]
    windows_above_80 = [78, 69, 81, 48, 71, 56, 78, 68, 73]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 7), gridspec_kw={"height_ratios": [3, 2]}
    )
    fig.suptitle(
        "Figure 5.2: Quality Score Summary Across Subjects",
        fontsize=13, fontweight="bold", y=0.98,
    )

    x = np.arange(len(subjects))
    err_low  = [m - lo for m, lo in zip(mean_overall, min_score)]
    err_high = [hi - m  for m, hi in zip(mean_overall, max_score)]

    bars = ax1.bar(x, mean_overall, width=0.5, color="#4caf50",
                   edgecolor="white", linewidth=0.5, zorder=3)
    ax1.errorbar(x, mean_overall, yerr=[err_low, err_high], fmt="none",
                 ecolor="#333333", elinewidth=1.5, capsize=5, capthick=1.5, zorder=4)

    for bar, val in zip(bars, mean_overall):
        bar.set_facecolor("#4caf50" if val >= 80 else
                          "#ffc107" if val >= 60 else "#f44336")

    ax1.axhline(y=80, color="#ffc107", linestyle="--", linewidth=1,
                alpha=0.8, label="80% threshold")
    ax1.axhline(y=60, color="#f44336", linestyle="--", linewidth=1,
                alpha=0.8, label="60% threshold")
    ax1.set_ylabel("Quality Score (%)", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(subjects, fontsize=10)
    ax1.set_ylim(55, 95)
    ax1.legend(loc="lower left", fontsize=9)
    ax1.set_title("Mean Overall Quality (with Min–Max Range)", fontsize=11, pad=8)
    ax1.grid(axis="y", alpha=0.3, zorder=0)
    for i, v in enumerate(mean_overall):
        ax1.text(i, v + 0.8, f"{v}%", ha="center", va="bottom",
                 fontsize=9, fontweight="bold")

    bars2 = ax2.bar(x, windows_above_80, width=0.5, color="#2196F3",
                    edgecolor="white", linewidth=0.5, zorder=3)
    ax2.set_ylabel('Windows ≥ 80% (%)', fontsize=11)
    ax2.set_xlabel("Subject", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(subjects, fontsize=10)
    ax2.set_ylim(0, 100)
    ax2.set_title('Proportion of Windows in "Excellent" Zone (≥ 80%)',
                  fontsize=11, pad=8)
    ax2.grid(axis="y", alpha=0.3, zorder=0)
    for i, v in enumerate(windows_above_80):
        ax2.text(i, v + 1.5, f"{v}%", ha="center", va="bottom",
                 fontsize=9, fontweight="bold")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig("figure_5_1_quality_summary.png", dpi=200, bbox_inches="tight")
    print("Saved figure_5_1_quality_summary.png")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════
# Figure 5.2 — Signal Validity: Wrist E4 vs Chest RespiBAN
# ══════════════════════════════════════════════════════════════════
def draw_validation():
    fig = plt.figure(figsize=(13, 5))
    fig.suptitle(
        "Figure 6.1: Signal Validity — Wrist E4 vs Chest RespiBAN (Subject S2)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    gs = gridspec.GridSpec(1, 3, wspace=0.38)

    # ── Panel A: EDA correlation summary ─────────────────────────
    ax1 = fig.add_subplot(gs[0])
    metrics = ["Overall\ncorrelation (r)", "Mean window\ncorrelation (r)"]
    values  = [0.779, 0.347]
    colors  = ["#4caf50", "#2196F3"]
    bars = ax1.bar(metrics, values, color=colors, width=0.45,
                   edgecolor="white", linewidth=0.5, zorder=3)
    ax1.axhline(0.5, color="#ffc107", linestyle="--", linewidth=1.2,
                label="Pass threshold (0.5)", zorder=4)
    ax1.set_ylim(0, 1.0)
    ax1.set_ylabel("Pearson r", fontsize=11)
    ax1.set_title("A  EDA Reliability\n(Wrist vs Chest)", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=8, loc="lower right")
    ax1.grid(axis="y", alpha=0.3, zorder=0)
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                 f"r = {val:.3f}", ha="center", va="bottom",
                 fontsize=10, fontweight="bold")

    # ── Panel B: Per-condition EDA correlations ───────────────────
    ax2 = fig.add_subplot(gs[1])
    conditions  = ["Baseline", "Stress", "Amusement", "Meditation"]
    cond_r      = [0.390, 0.520, 0.330, 0.361]
    cond_colors = ["#2196F3", "#f44336", "#ff9800", "#9c27b0"]
    bars2 = ax2.bar(conditions, cond_r, color=cond_colors, width=0.55,
                    edgecolor="white", linewidth=0.5, zorder=3)
    ax2.axhline(0, color="grey", linestyle="-", linewidth=0.8, alpha=0.5, zorder=2)
    ax2.set_ylim(-0.1, 0.8)
    ax2.set_ylabel("Mean Pearson r (10-s windows)", fontsize=11)
    ax2.set_title("B  EDA by Experimental Condition\n(Wrist vs Chest)",
                  fontsize=11, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3, zorder=0)
    ax2.tick_params(axis="x", labelsize=9)
    for bar, val in zip(bars2, cond_r):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.015,
                 f"{val:.3f}", ha="center", va="bottom",
                 fontsize=9, fontweight="bold")

    # ── Panel C: Heart rate agreement ────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    sizes      = [70.5, 29.5]
    pie_labels = ["Within\n10 BPM\n(70.5%)", "Outside\n10 BPM\n(29.5%)"]
    pie_colors = ["#4caf50", "#e0e0e0"]
    wedges, texts, autotexts = ax3.pie(
        sizes, labels=pie_labels, colors=pie_colors, autopct="%1.1f%%",
        startangle=90, wedgeprops=dict(edgecolor="white", linewidth=1.5),
        textprops={"fontsize": 9},
    )
    autotexts[0].set_fontweight("bold")
    autotexts[0].set_color("white")
    ax3.set_title("C  Heart Rate Agreement\n(BVP vs ECG, 444 windows)",
                  fontsize=11, fontweight="bold")
    ax3.text(
        0, -1.45,
        "r = 0.201  |  MAE = 11.4 BPM\nPass threshold: ≥ 40% within 10 BPM",
        ha="center", va="center", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5f5f5",
                  edgecolor="#cccccc", alpha=0.9),
    )

    fig.savefig("figure_5_2_validation.png", dpi=200, bbox_inches="tight")
    print("Saved figure_5_2_validation.png")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════
# Figure 5.2 — Per-Sensor Quality Breakdown (Subject S2, Section 5.4)
# ══════════════════════════════════════════════════════════════════
def draw_sensor_breakdown():
    """Two-panel chart for §5.4:
    Left  — mean quality per sensor with SD error bars.
    Right — SD (variability) per sensor as a horizontal bar.
    """
    sensors      = ["EDA", "TEMP", "ACC", "BVP"]
    means        = [99.7,  98.8,   72.7,  59.1]
    sds          = [1.6,   1.1,    14.6,  8.1]
    colors       = ["#66bb6a", "#ffa726", "#ab47bc", "#ef5350"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.suptitle(
        "Figure 5.3: Per-Sensor Quality Breakdown — Subject S2",
        fontsize=13, fontweight="bold", y=1.01,
    )

    # ── Left panel: mean quality bar chart ───────────────────────
    x = np.arange(len(sensors))
    bars = ax1.bar(x, means, width=0.5, color=colors,
                   edgecolor="white", linewidth=0.5, zorder=3)
    ax1.errorbar(x, means, yerr=sds, fmt="none",
                 ecolor="#333333", elinewidth=1.5,
                 capsize=6, capthick=1.5, zorder=4)

    # threshold lines
    ax1.axhline(80, color="#ffc107", linestyle="--",
                linewidth=1, alpha=0.85, label="80% (Excellent)", zorder=2)
    ax1.axhline(60, color="#f44336", linestyle="--",
                linewidth=1, alpha=0.85, label="60% (Acceptable)", zorder=2)

    ax1.set_xticks(x)
    ax1.set_xticklabels(sensors, fontsize=11)
    ax1.set_ylabel("Mean Quality Score (%)", fontsize=11)
    ax1.set_ylim(40, 110)
    ax1.set_title("A  Mean Quality per Sensor", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=8, loc="lower left")
    ax1.grid(axis="y", alpha=0.3, zorder=0)

    for bar, mean, sd in zip(bars, means, sds):
        ax1.text(bar.get_x() + bar.get_width() / 2, mean + sd + 1.5,
                 f"{mean}%", ha="center", va="bottom",
                 fontsize=10, fontweight="bold")

    # colour-coded zone labels on bars
    zone_labels = ["Excellent", "Excellent", "Acceptable", "Acceptable"]
    zone_colors = ["#ffffff", "#ffffff", "#ffffff", "#ffffff"]
    for bar, lbl in zip(bars, zone_labels):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() / 2,
                 lbl, ha="center", va="center",
                 fontsize=8, color="white", fontweight="bold",
                 rotation=0)

    # ── Right panel: variability (SD) horizontal bars ─────────────
    y = np.arange(len(sensors))
    hbars = ax2.barh(y, sds, height=0.45, color=colors,
                     edgecolor="white", linewidth=0.5, zorder=3)
    ax2.set_yticks(y)
    ax2.set_yticklabels(sensors, fontsize=11)
    ax2.set_xlabel("Standard Deviation (percentage points)", fontsize=11)
    ax2.set_title("B  Quality Variability (SD)", fontsize=11, fontweight="bold")
    ax2.grid(axis="x", alpha=0.3, zorder=0)
    ax2.invert_yaxis()   # match top-to-bottom order with panel A

    for bar, sd in zip(hbars, sds):
        ax2.text(sd + 0.3, bar.get_y() + bar.get_height() / 2,
                 f"±{sd}pp", va="center", fontsize=10, fontweight="bold")

    ax2.text(
        0.97, 0.03,
        "EDA & TEMP: near-zero SD → stable signals\n"
        "ACC: high SD → activity-dependent\n"
        "BVP: moderate SD → motion-sensitive",
        transform=ax2.transAxes,
        ha="right", va="bottom", fontsize=8, style="italic",
        color="#555555",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#f5f5f5",
                  edgecolor="#cccccc", alpha=0.9),
    )

    fig.tight_layout()
    fig.savefig("figure_5_2_sensor_breakdown.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    print("Saved figure_5_2_sensor_breakdown.png")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    draw_architecture()
    draw_table()
    draw_quality_summary()
    draw_sensor_breakdown()
    draw_validation()
    print("Done — all figures generated.")
