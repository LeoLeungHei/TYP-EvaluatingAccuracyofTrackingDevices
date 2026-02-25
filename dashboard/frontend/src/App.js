import React, { useState, useEffect, useRef, useCallback } from "react";
import "./App.css";
import RealtimeChart from "./components/RealtimeChart";
import MetricCard from "./components/MetricCard";
import AggregateScore from "./components/AggregateScore";

/* ── helpers ──────────────────────────────────────────────────── */

function fmtTime(sec) {
  const m = Math.floor(sec / 60);
  const s = Math.round(sec % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

const MAX_HISTORY = 150; // ~5 min of data at 2s step

/* ── App ──────────────────────────────────────────────────────── */

export default function App() {
  const [subjects, setSubjects] = useState([]);
  const [subject, setSubject] = useState(null);
  const [streaming, setStreaming] = useState(false);
  const [done, setDone] = useState(false);
  const [speed, setSpeed] = useState(10);

  // Real-time state
  const [currentWindow, setCurrentWindow] = useState(null);
  const [history, setHistory] = useState([]);
  const [stats, setStats] = useState({ count: 0, sum: 0, min: 100, max: 0 });

  const eventSourceRef = useRef(null);

  /* ── load subject list ────────────────────────────────────── */
  useEffect(() => {
    fetch("/api/subjects")
      .then((r) => r.json())
      .then((list) => {
        setSubjects(list);
        if (list.length > 0) setSubject(list[0]);
      })
      .catch(() => {});
  }, []);

  /* ── start / stop streaming ───────────────────────────────── */
  const startStream = useCallback(() => {
    if (!subject) return;

    setHistory([]);
    setCurrentWindow(null);
    setStats({ count: 0, sum: 0, min: 100, max: 0 });
    setDone(false);
    setStreaming(true);

    const es = new EventSource(
      `/api/stream/${subject}?window=10&step=2&speed=${speed}`
    );
    eventSourceRef.current = es;

    es.onmessage = (event) => {
      const data = JSON.parse(event.data);

      if (data.done) {
        es.close();
        setStreaming(false);
        setDone(true);
        return;
      }

      setCurrentWindow(data);

      setHistory((prev) => {
        const next = [...prev, data];
        return next.length > MAX_HISTORY ? next.slice(-MAX_HISTORY) : next;
      });

      setStats((prev) => {
        const overall = data.overall || 0;
        return {
          count: prev.count + 1,
          sum: prev.sum + overall,
          min: Math.min(prev.min, overall),
          max: Math.max(prev.max, overall),
        };
      });
    };

    es.onerror = () => {
      es.close();
      setStreaming(false);
    };
  }, [subject, speed]);

  const stopStream = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
    setStreaming(false);
  }, []);

  useEffect(() => {
    return () => {
      if (eventSourceRef.current) eventSourceRef.current.close();
    };
  }, []);

  /* ── derived values ───────────────────────────────────────── */
  const overall = currentWindow?.overall ?? 0;
  const onBody = currentWindow?.on_body ?? false;
  const elapsed = currentWindow?.time ?? 0;
  const progress = currentWindow?.progress ?? 0;
  const avgOverall = stats.count > 0 ? stats.sum / stats.count : 0;

  /* ── render ────────────────────────────────────────────────── */
  return (
    <div className="dashboard">
      {/* ── Header ──────────────────────────────────────────── */}
      <header className="header">
        <div className="header-left">
          <h1>Data Quality Monitor</h1>
          <span className="header-sub">10-Second Sliding Window</span>
        </div>
        <div className="header-controls">
          <select
            className="subject-select"
            value={subject || ""}
            onChange={(e) => {
              stopStream();
              setSubject(e.target.value);
              setCurrentWindow(null);
              setHistory([]);
              setStats({ count: 0, sum: 0, min: 100, max: 0 });
              setDone(false);
            }}
            disabled={streaming}
          >
            {subjects.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>

          <label className="speed-label">
            Speed
            <select
              className="speed-select"
              value={speed}
              onChange={(e) => setSpeed(Number(e.target.value))}
              disabled={streaming}
            >
              <option value={1}>1×</option>
              <option value={5}>5×</option>
              <option value={10}>10×</option>
              <option value={25}>25×</option>
              <option value={50}>50×</option>
            </select>
          </label>

          {!streaming ? (
            <button className="btn-start" onClick={startStream}>
              {done ? "Restart" : "Start Monitor"}
            </button>
          ) : (
            <button className="btn-stop" onClick={stopStream}>
              Stop
            </button>
          )}
        </div>
      </header>

      {/* ── Status bar ──────────────────────────────────────── */}
      <div className={`status-bar ${streaming ? "live" : done ? "done" : "idle"}`}>
        <div className="status-dot" />
        <span className="status-text">
          {streaming
            ? `LIVE — ${fmtTime(elapsed)}`
            : done
            ? `COMPLETE — ${stats.count} windows analyzed`
            : "Ready to start"}
        </span>
        {streaming && (
          <span className={`status-onbody ${onBody ? "on" : "off"}`}>
            {onBody ? "ON-BODY" : "OFF-BODY"}
          </span>
        )}
        {(streaming || done) && (
          <div className="progress-track">
            <div className="progress-fill" style={{ width: `${progress}%` }} />
          </div>
        )}
      </div>

      {/* ── Main grid: 6 components ─────────────────────────── */}
      <div className="monitor-grid">
        {/* 1. Aggregate score */}
        <div className="grid-aggregate">
          <AggregateScore
            current={overall}
            average={avgOverall}
            min={stats.min <= 100 ? stats.min : 0}
            max={stats.max}
            windowCount={stats.count}
            streaming={streaming}
          />
        </div>

        {/* 2-5. Four sensor metric cards */}
        <div className="grid-acc">
          <MetricCard
            sensor="ACC"
            label="Accelerometer"
            data={currentWindow?.acc}
            color="#ab47bc"
            streaming={streaming}
          />
        </div>
        <div className="grid-bvp">
          <MetricCard
            sensor="BVP"
            label="Blood Volume Pulse"
            data={currentWindow?.bvp}
            color="#ef5350"
            streaming={streaming}
          />
        </div>
        <div className="grid-eda">
          <MetricCard
            sensor="EDA"
            label="Electrodermal Activity"
            data={currentWindow?.eda}
            color="#66bb6a"
            streaming={streaming}
          />
        </div>
        <div className="grid-temp">
          <MetricCard
            sensor="TEMP"
            label="Skin Temperature"
            data={currentWindow?.temp}
            color="#ffa726"
            streaming={streaming}
          />
        </div>

        {/* 6. Timeline chart */}
        <div className="grid-chart">
          <RealtimeChart history={history} streaming={streaming} />
        </div>
      </div>
    </div>
  );
}
