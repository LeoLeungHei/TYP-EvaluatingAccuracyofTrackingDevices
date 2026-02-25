import React, { useMemo } from "react";
import Plot from "react-plotly.js";

/**
 * Real-time quality timeline chart.
 * Shows overall + per-sensor aggregate lines that scroll as new
 * data arrives, mimicking the terminal-based realtime_quality_monitor.
 *
 * Props:
 *   history   – array of the most recent window objects
 *   streaming – whether the monitor is active
 */

const COLORS = {
  overall: "#4fc3f7",
  acc: "#ab47bc",
  bvp: "#ef5350",
  eda: "#66bb6a",
  temp: "#ffa726",
};

export default function RealtimeChart({ history, streaming }) {
  const traces = useMemo(() => {
    if (!history || history.length === 0) return [];

    const times = history.map((w) => w.time / 60);

    return [
      {
        x: times,
        y: history.map((w) => w.overall),
        type: "scatter",
        mode: "lines",
        line: { width: 3, color: COLORS.overall },
        fill: "tozeroy",
        fillcolor: "rgba(79,195,247,0.08)",
        name: "Overall",
        hovertemplate: "<b>Overall: %{y:.1f}%</b><extra></extra>",
      },
      ...["acc", "bvp", "eda", "temp"].map((s) => ({
        x: times,
        y: history.map((w) => w[s]?.aggregate ?? null),
        type: "scatter",
        mode: "lines",
        line: { width: 1.5, color: COLORS[s], dash: "dot" },
        name: s.toUpperCase(),
        connectgaps: true,
        hovertemplate: `<b>${s.toUpperCase()}: %{y:.1f}%</b><extra></extra>`,
      })),
    ];
  }, [history]);

  const layout = useMemo(
    () => ({
      paper_bgcolor: "transparent",
      plot_bgcolor: "transparent",
      margin: { t: 10, r: 20, b: 45, l: 50 },
      height: 260,
      xaxis: {
        title: { text: "Time (minutes)", font: { color: "#90a4ae", size: 12 } },
        color: "#90a4ae",
        gridcolor: "rgba(255,255,255,0.06)",
        zeroline: false,
      },
      yaxis: {
        title: { text: "Quality (%)", font: { color: "#90a4ae", size: 12 } },
        range: [0, 105],
        color: "#90a4ae",
        gridcolor: "rgba(255,255,255,0.06)",
        zeroline: false,
      },
      legend: {
        orientation: "h",
        y: -0.32,
        x: 0.5,
        xanchor: "center",
        font: { color: "#90a4ae", size: 11 },
      },
      shapes: [
        {
          type: "rect",
          xref: "paper",
          yref: "y",
          x0: 0, x1: 1,
          y0: 80, y1: 105,
          fillcolor: "rgba(102,187,106,0.04)",
          line: { width: 0 },
          layer: "below",
        },
        {
          type: "rect",
          xref: "paper",
          yref: "y",
          x0: 0, x1: 1,
          y0: 60, y1: 80,
          fillcolor: "rgba(255,167,38,0.04)",
          line: { width: 0 },
          layer: "below",
        },
        {
          type: "rect",
          xref: "paper",
          yref: "y",
          x0: 0, x1: 1,
          y0: 0, y1: 60,
          fillcolor: "rgba(239,83,80,0.04)",
          line: { width: 0 },
          layer: "below",
        },
        {
          type: "line",
          xref: "paper",
          yref: "y",
          x0: 0, x1: 1,
          y0: 80, y1: 80,
          line: { color: "rgba(102,187,106,0.25)", width: 1, dash: "dash" },
        },
        {
          type: "line",
          xref: "paper",
          yref: "y",
          x0: 0, x1: 1,
          y0: 60, y1: 60,
          line: { color: "rgba(255,167,38,0.25)", width: 1, dash: "dash" },
        },
      ],
    }),
    []
  );

  if (!history || history.length === 0) {
    return (
      <div className="chart-empty">
        {streaming
          ? "Waiting for data..."
          : "Press Start Monitor to begin"}
      </div>
    );
  }

  return (
    <Plot
      data={traces}
      layout={layout}
      config={{ displayModeBar: false, responsive: true }}
      style={{ width: "100%", height: "260px" }}
      useResizeHandler
    />
  );
}
