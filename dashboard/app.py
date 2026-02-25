"""
Flask backend for the Data Quality Dashboard.
Exposes the SlidingWindowQualityMonitor results as JSON API endpoints.
"""

import sys
import os

# Add parent directory so we can import the quality monitor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, jsonify, request, send_from_directory, Response
from flask_cors import CORS
import numpy as np
import json
import time

from realtime_quality_monitor import (
    SlidingWindowQualityMonitor,
    load_e4_data,
    get_window_data,
)

app = Flask(__name__, static_folder="frontend/build", static_url_path="")
CORS(app)

# ── Data cache ───────────────────────────────────────────────────────────────
# Pre-load subject data so API calls are fast
DATA_CACHE: dict = {}
WESAD_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_subject_data(subject_id: str) -> dict:
    """Load and cache E4 data for a subject."""
    if subject_id not in DATA_CACHE:
        e4_folder = os.path.join(WESAD_ROOT, subject_id, f"{subject_id}_E4_Data")
        if not os.path.isdir(e4_folder):
            return None
        data = load_e4_data(e4_folder)
        if not data:
            return None

        # Compute time bounds
        all_ts = []
        for sd in data.values():
            all_ts.extend([sd["timestamps"].min(), sd["timestamps"].max()])
        start_time = min(all_ts)
        end_time = max(all_ts)

        DATA_CACHE[subject_id] = {
            "data": data,
            "start_time": float(start_time),
            "end_time": float(end_time),
            "duration": float(end_time - start_time),
        }
    return DATA_CACHE[subject_id]


# ── API routes ───────────────────────────────────────────────────────────────

@app.route("/api/subjects")
def list_subjects():
    """Return list of subjects that have extracted E4 data."""
    subjects = []
    for name in sorted(os.listdir(WESAD_ROOT)):
        e4_path = os.path.join(WESAD_ROOT, name, f"{name}_E4_Data")
        if os.path.isdir(e4_path):
            subjects.append(name)
    return jsonify(subjects)


@app.route("/api/overview/<subject_id>")
def subject_overview(subject_id: str):
    """Return metadata about a subject's recording."""
    cache = get_subject_data(subject_id)
    if cache is None:
        return jsonify({"error": "Subject not found or no E4 data"}), 404

    sensors = {}
    for name, sd in cache["data"].items():
        sensors[name] = {
            "sample_rate": float(sd["sample_rate"]),
            "total_samples": len(sd["data"]),
            "duration_seconds": float(
                sd["timestamps"][-1] - sd["timestamps"][0]
            ),
        }

    return jsonify({
        "subject": subject_id,
        "start_time": cache["start_time"],
        "end_time": cache["end_time"],
        "duration_seconds": cache["duration"],
        "duration_minutes": cache["duration"] / 60,
        "sensors": sensors,
    })


@app.route("/api/quality/<subject_id>")
def full_quality_timeline(subject_id: str):
    """
    Compute quality scores across the entire recording using a sliding window.

    Query params:
        window  – window size in seconds (default 10)
        step    – step size in seconds (default 2)
    """
    cache = get_subject_data(subject_id)
    if cache is None:
        return jsonify({"error": "Subject not found"}), 404

    window_size = float(request.args.get("window", 10))
    step = float(request.args.get("step", 2))

    data = cache["data"]
    start = cache["start_time"]
    end = cache["end_time"] - window_size
    monitor = SlidingWindowQualityMonitor(window_size)

    timeline = []
    t = start
    while t < end:
        elapsed = t - start
        window_result = {"time": round(elapsed, 1)}

        for sensor_name, sensor_data in data.items():
            wd = get_window_data(sensor_data, t, window_size)
            if wd is None or len(wd) == 0:
                continue

            sr = sensor_data["sample_rate"]
            if sensor_name == "acc":
                density, on_body, sig_q, vals = monitor.calculate_acc_quality(wd)
            elif sensor_name == "bvp":
                density, on_body, sig_q, vals = monitor.calculate_bvp_quality(wd, sr)
            elif sensor_name == "eda":
                density, on_body, sig_q, vals = monitor.calculate_eda_quality(wd, sr)
            elif sensor_name == "temp":
                density, on_body, sig_q, vals = monitor.calculate_temp_quality(wd, sr)
            else:
                continue

            agg = monitor.calculate_aggregate_score(density, on_body, sig_q)
            window_result[sensor_name] = {
                "density": round(float(density), 1),
                "on_body": bool(on_body),
                "signal_quality": round(float(sig_q), 1),
                "aggregate": round(float(agg), 1),
                "values": {k: round(float(v), 4) for k, v in vals.items()},
            }

        # Overall aggregate
        sensor_aggs = [
            window_result[s]["aggregate"]
            for s in ["acc", "bvp", "eda", "temp"]
            if s in window_result
        ]
        window_result["overall"] = round(float(np.mean(sensor_aggs)), 1) if sensor_aggs else 0

        # Combined on-body (EDA + TEMP)
        eda_on = window_result.get("eda", {}).get("on_body", False)
        temp_on = window_result.get("temp", {}).get("on_body", False)
        window_result["on_body"] = eda_on and temp_on

        timeline.append(window_result)
        t += step

    return jsonify({
        "subject": subject_id,
        "window_size": window_size,
        "step": step,
        "total_windows": len(timeline),
        "timeline": timeline,
    })


@app.route("/api/signals/<subject_id>/<sensor_name>")
def raw_signal(subject_id: str, sensor_name: str):
    """
    Return downsampled raw signal values for plotting.

    Query params:
        max_points – maximum number of points to return (default 2000)
    """
    cache = get_subject_data(subject_id)
    if cache is None:
        return jsonify({"error": "Subject not found"}), 404

    if sensor_name not in cache["data"]:
        return jsonify({"error": f"Sensor '{sensor_name}' not found"}), 404

    sd = cache["data"][sensor_name]
    max_points = int(request.args.get("max_points", 2000))

    timestamps = sd["timestamps"] - cache["start_time"]  # relative
    values = sd["data"]

    # Downsample if too many points
    total = len(timestamps)
    if total > max_points:
        step = total // max_points
        indices = np.arange(0, total, step)[:max_points]
        timestamps = timestamps[indices]
        values = values[indices] if values.ndim == 1 else values[indices, :]

    result = {
        "sensor": sensor_name,
        "sample_rate": float(sd["sample_rate"]),
        "total_samples": total,
        "returned_samples": len(timestamps),
        "times": np.round(timestamps, 2).tolist(),
    }

    if values.ndim == 1:
        result["values"] = np.round(values, 4).tolist()
    else:
        # ACC has x, y, z columns
        result["values"] = {
            "x": np.round(values[:, 0], 4).tolist(),
            "y": np.round(values[:, 1], 4).tolist(),
            "z": np.round(values[:, 2], 4).tolist(),
        }

    return jsonify(result)


# ── Real-time streaming endpoint (SSE) ────────────────────────────────────────

@app.route("/api/stream/<subject_id>")
def stream_quality(subject_id: str):
    """
    Server-Sent Events stream that replays E4 data window-by-window,
    computing quality scores identical to realtime_quality_monitor.py.

    Query params:
        window  – window size in seconds (default 10)
        step    – step size in seconds (default 2)
        speed   – playback speed multiplier (default 10)
    """
    cache = get_subject_data(subject_id)
    if cache is None:
        return jsonify({"error": "Subject not found"}), 404

    window_size = float(request.args.get("window", 10))
    step = float(request.args.get("step", 2))
    speed = float(request.args.get("speed", 10))

    data = cache["data"]
    start = cache["start_time"]
    end = cache["end_time"] - window_size
    total_duration = end - start
    monitor = SlidingWindowQualityMonitor(window_size)

    def generate():
        t = start
        while t < end:
            elapsed = t - start
            progress = elapsed / total_duration * 100

            window_result = {
                "time": round(elapsed, 1),
                "progress": round(progress, 1),
            }

            for sensor_name, sensor_data in data.items():
                wd = get_window_data(sensor_data, t, window_size)
                if wd is None or len(wd) == 0:
                    continue

                sr = sensor_data["sample_rate"]
                if sensor_name == "acc":
                    density, on_body, sig_q, vals = monitor.calculate_acc_quality(wd)
                elif sensor_name == "bvp":
                    density, on_body, sig_q, vals = monitor.calculate_bvp_quality(wd, sr)
                elif sensor_name == "eda":
                    density, on_body, sig_q, vals = monitor.calculate_eda_quality(wd, sr)
                elif sensor_name == "temp":
                    density, on_body, sig_q, vals = monitor.calculate_temp_quality(wd, sr)
                else:
                    continue

                agg = monitor.calculate_aggregate_score(density, on_body, sig_q)
                window_result[sensor_name] = {
                    "density": round(float(density), 1),
                    "on_body": bool(on_body),
                    "signal_quality": round(float(sig_q), 1),
                    "aggregate": round(float(agg), 1),
                    "values": {k: round(float(v), 4) for k, v in vals.items()},
                }

            # Overall aggregate
            sensor_aggs = [
                window_result[s]["aggregate"]
                for s in ["acc", "bvp", "eda", "temp"]
                if s in window_result
            ]
            window_result["overall"] = (
                round(float(np.mean(sensor_aggs)), 1) if sensor_aggs else 0
            )

            # Combined on-body
            eda_on = window_result.get("eda", {}).get("on_body", False)
            temp_on = window_result.get("temp", {}).get("on_body", False)
            window_result["on_body"] = eda_on and temp_on

            yield f"data: {json.dumps(window_result)}\n\n"

            t += step
            time.sleep(step / speed)

        # Signal end of stream
        yield f"data: {json.dumps({'done': True})}\n\n"

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})


# ── Serve React frontend ─────────────────────────────────────────────────────

@app.route("/")
def serve_react():
    return send_from_directory(app.static_folder, "index.html")


@app.errorhandler(404)
def not_found(e):
    # For React client-side routing, serve index.html for non-API routes
    if not request.path.startswith("/api/"):
        return send_from_directory(app.static_folder, "index.html")
    return jsonify({"error": "Not found"}), 404


if __name__ == "__main__":
    print(f"WESAD root: {WESAD_ROOT}")
    print("Starting dashboard server...")
    app.run(debug=True, port=5000, threaded=True)
