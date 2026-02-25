import React from "react";

/**
 * Individual sensor metric card showing quality breakdown.
 *
 * Props:
 *   sensor    – sensor key ("ACC", "BVP", etc.)
 *   label     – human-readable label
 *   data      – window data for this sensor (density, on_body, signal_quality, aggregate, values)
 *   color     – accent colour for this sensor
 *   streaming – whether monitor is active
 */

function qualityColor(v) {
  if (v >= 80) return "var(--quality-good)";
  if (v >= 60) return "var(--quality-ok)";
  return "var(--quality-bad)";
}

function MiniBar({ value, color }) {
  return (
    <div className="mini-bar-track">
      <div
        className="mini-bar-fill"
        style={{
          width: `${Math.min(value, 100)}%`,
          background: color,
          transition: "width 0.3s ease",
        }}
      />
    </div>
  );
}

export default function MetricCard({ sensor, label, data, color, streaming }) {
  const agg = data?.aggregate ?? 0;
  const density = data?.density ?? 0;
  const sigQ = data?.signal_quality ?? 0;
  const onBody = data?.on_body ?? false;
  const hasData = data != null;

  return (
    <div className="metric-card" style={{ borderTopColor: color }}>
      <div className="metric-header">
        <span className="metric-sensor" style={{ color }}>{sensor}</span>
        <span className={`metric-onbody ${onBody ? "on" : "off"}`}>
          {hasData ? (onBody ? "On-body" : "Off-body") : "—"}
        </span>
      </div>

      <div className="metric-score" style={{ color: hasData ? qualityColor(agg) : "var(--text-secondary)" }}>
        {hasData ? `${agg.toFixed(1)}%` : "—"}
      </div>
      <div className="metric-label">{label}</div>

      {/* Sub-metrics */}
      <div className="metric-breakdown">
        <div className="metric-row">
          <span className="metric-row-lbl">Density</span>
          <MiniBar value={density} color="var(--accent-blue)" />
          <span className="metric-row-val">{hasData ? `${density.toFixed(0)}%` : "—"}</span>
        </div>
        <div className="metric-row">
          <span className="metric-row-lbl">Signal Q</span>
          <MiniBar value={sigQ} color={qualityColor(sigQ)} />
          <span className="metric-row-val">{hasData ? `${sigQ.toFixed(0)}%` : "—"}</span>
        </div>
      </div>
    </div>
  );
}
