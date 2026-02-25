import React from "react";

/**
 * Large aggregate quality score display with running stats.
 *
 * Props:
 *   current      – current window overall score
 *   average      – running average
 *   min / max    – session min / max
 *   windowCount  – number of windows processed
 *   streaming    – whether the monitor is active
 */

function qualityColor(v) {
  if (v >= 80) return "var(--quality-good)";
  if (v >= 60) return "var(--quality-ok)";
  return "var(--quality-bad)";
}

function qualityLabel(v) {
  if (v >= 80) return "Excellent";
  if (v >= 60) return "Acceptable";
  return "Poor";
}

export default function AggregateScore({
  current,
  average,
  min,
  max,
  windowCount,
  streaming,
}) {
  const color = qualityColor(current);
  const ringPct = Math.min(current, 100);

  return (
    <div className="agg-card">
      <div className="agg-ring-wrap">
        <svg viewBox="0 0 120 120" className="agg-ring">
          {/* background track */}
          <circle
            cx="60"
            cy="60"
            r="52"
            fill="none"
            stroke="rgba(255,255,255,0.06)"
            strokeWidth="8"
          />
          {/* coloured arc */}
          <circle
            cx="60"
            cy="60"
            r="52"
            fill="none"
            stroke={color}
            strokeWidth="8"
            strokeLinecap="round"
            strokeDasharray={`${(ringPct / 100) * 326.7} 326.7`}
            transform="rotate(-90 60 60)"
            style={{ transition: "stroke-dasharray 0.4s ease, stroke 0.4s ease" }}
          />
        </svg>
        <div className="agg-center">
          <span className="agg-value" style={{ color }}>
            {streaming || windowCount > 0 ? `${current.toFixed(1)}%` : "—"}
          </span>
          <span className="agg-label">{qualityLabel(current)}</span>
        </div>
      </div>

      <div className="agg-title">Overall Quality</div>

      {windowCount > 0 && (
        <div className="agg-stats">
          <div className="agg-stat">
            <span className="agg-stat-val">{average.toFixed(1)}%</span>
            <span className="agg-stat-lbl">Average</span>
          </div>
          <div className="agg-stat">
            <span className="agg-stat-val">{min.toFixed(1)}%</span>
            <span className="agg-stat-lbl">Min</span>
          </div>
          <div className="agg-stat">
            <span className="agg-stat-val">{max.toFixed(1)}%</span>
            <span className="agg-stat-lbl">Max</span>
          </div>
        </div>
      )}
    </div>
  );
}
