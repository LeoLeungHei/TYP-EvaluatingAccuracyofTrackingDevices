# WESAD Data Quality Dashboard

Real-time data quality monitoring dashboard for E4 wristband data from the WESAD dataset.

## Prerequisites

- Python 3.x
- Node.js & npm

## Setup

### 1. Install Python dependencies

From the `WESAD/` root directory:

```bash
pip install flask flask-cors numpy pandas scipy
```

### 2. Install frontend dependencies

```bash
cd dashboard/frontend
npm install
```

## Running the project

You need two terminals running simultaneously.

### Terminal 1 — Flask backend

From the `WESAD/` root directory:

```bash
cd dashboard
py app.py
```

The API will be available at `http://localhost:5000`.

### Terminal 2 — React frontend (development mode)

```bash
cd dashboard/frontend
npm start
```

The dashboard will open at `http://localhost:3000`. API requests are proxied to the Flask backend automatically.

## Running as a single server (production mode)

Build the React app first, then serve everything from Flask alone:

```bash
cd dashboard/frontend
npm run build

cd ..
py app.py
```

Open `http://localhost:5000` in your browser.

## Project structure

```
WESAD/
├── realtime_quality_monitor.py   # Core sliding-window quality monitor
├── test_quality_monitor.py       # Tests for the quality monitor
├── S2/ … S10/                    # Subject data folders (E4 wristband CSV files)
└── dashboard/
    ├── app.py                    # Flask backend (REST API + SSE streaming)
    └── frontend/                 # React frontend
        ├── package.json
        └── src/
```

## API endpoints

| Endpoint | Description |
|---|---|
| `GET /api/subjects` | List all available subjects |
| `GET /api/overview/<subject_id>` | Recording metadata for a subject |
| `GET /api/quality/<subject_id>` | Full quality timeline (sliding window) |
| `GET /api/signals/<subject_id>/<sensor>` | Downsampled raw signal values |
| `GET /api/stream/<subject_id>` | Real-time SSE stream of quality scores |

Query parameters for `/api/quality` and `/api/stream`: `window` (seconds, default 10), `step` (seconds, default 2), `speed` (playback multiplier, default 10, stream only).
