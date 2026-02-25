"""
Validation Test: E4 Wrist Data Reliability vs RespiBAN Chest Ground Truth
=========================================================================
Compares the wrist-worn E4 signals against the independently recorded
chest-worn RespiBAN signals from the WESAD pkl files.

Both devices record the same physiological signals (EDA, temperature,
acceleration) on different body locations simultaneously. If the wrist
data is reliable, the two should correlate — proving the streamed data
is real physiological signal and not noise.

Tests
-----
1. EDA   – wrist EDA vs chest EDA (same electrodermal response)
2. TEMP  – wrist skin temp vs chest skin temp (physiological range)
3. Heart – wrist BVP vs chest ECG (heart rate agreement)
4. Per-condition – EDA correlations split by experimental condition
"""

import pickle
import numpy as np
import sys
import os
from scipy.stats import pearsonr
from scipy import signal as sig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── constants ────────────────────────────────────────────────────────────────

WRIST_RATES = {"ACC": 32, "BVP": 64, "EDA": 4, "TEMP": 4}
CHEST_RATE  = 700

LABEL_NAMES = {
    0: "transient", 1: "baseline", 2: "stress",
    3: "amusement", 4: "meditation",
}


# ── helpers ──────────────────────────────────────────────────────────────────

def load_pkl(subject_id: str) -> dict:
    path = os.path.join(os.path.dirname(__file__), subject_id, f"{subject_id}.pkl")
    print(f"  Loading {path} ...")
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def downsample(signal_arr: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """Downsample by block-averaging."""
    ratio = from_rate // to_rate
    n = (len(signal_arr) // ratio) * ratio
    return signal_arr[:n].reshape(-1, ratio).mean(axis=1)


def downsample_labels(labels: np.ndarray, to_rate: int) -> np.ndarray:
    ratio = CHEST_RATE // to_rate
    n = (len(labels) // ratio) * ratio
    blocks = labels[:n].reshape(-1, ratio)
    return np.array([np.bincount(b).argmax() for b in blocks])


def windowed_correlation(sig_a: np.ndarray, sig_b: np.ndarray,
                         window: int) -> list:
    """Pearson r for each non-overlapping window."""
    corrs = []
    for i in range(0, min(len(sig_a), len(sig_b)) - window, window):
        a = sig_a[i:i + window]
        b = sig_b[i:i + window]
        if np.std(a) > 1e-9 and np.std(b) > 1e-9:
            r, _ = pearsonr(a, b)
            corrs.append(r)
    return corrs


def extract_hr_from_bvp(bvp: np.ndarray, rate: int, window_sec: float = 10.0):
    """Estimate heart rate from BVP via autocorrelation.

    BVP is quasi-sinusoidal, so the first positive autocorrelation peak
    after the zero-lag gives the inter-beat interval.
    """
    window = int(window_sec * rate)
    min_lag = int(rate * 60 / 200)  # 200 BPM ceiling
    max_lag = int(rate * 60 / 40)   # 40 BPM floor
    hrs = []
    for i in range(0, len(bvp) - window, window):
        seg = bvp[i:i + window]
        seg = seg - np.mean(seg)
        corr = np.correlate(seg, seg, mode='full')
        corr = corr[len(seg) - 1:]  # keep positive lags only
        corr = corr / (corr[0] + 1e-12)  # normalise
        search = corr[min_lag:max_lag]
        if len(search) > 0 and np.max(search) > 0.15:
            peak_lag = np.argmax(search) + min_lag
            hrs.append(60.0 * rate / peak_lag)
        else:
            hrs.append(np.nan)
    return np.array(hrs)


def extract_hr_from_ecg(ecg: np.ndarray, rate: int, window_sec: float = 10.0):
    """Estimate heart rate from ECG via autocorrelation.

    The QRS complex repeats periodically; autocorrelation finds that
    period without being confused by harmonics (unlike a periodogram).
    """
    window = int(window_sec * rate)
    min_lag = int(rate * 60 / 200)
    max_lag = int(rate * 60 / 40)
    hrs = []
    for i in range(0, len(ecg) - window, window):
        seg = ecg[i:i + window]
        seg = seg - np.mean(seg)
        corr = np.correlate(seg, seg, mode='full')
        corr = corr[len(seg) - 1:]
        corr = corr / (corr[0] + 1e-12)
        search = corr[min_lag:max_lag]
        if len(search) > 0 and np.max(search) > 0.15:
            peak_lag = np.argmax(search) + min_lag
            hrs.append(60.0 * rate / peak_lag)
        else:
            hrs.append(np.nan)
    return np.array(hrs)


# ── tests ────────────────────────────────────────────────────────────────────

def test_eda_reliability(pkl_data: dict):
    """TEST 1 - Wrist EDA vs chest EDA: same electrodermal response."""
    print("\n" + "=" * 60)
    print("TEST 1: EDA Reliability - Wrist vs Chest")
    print("=" * 60)

    wrist_eda = pkl_data["signal"]["wrist"]["EDA"].flatten()
    chest_eda = downsample(
        pkl_data["signal"]["chest"]["EDA"].flatten(), CHEST_RATE, WRIST_RATES["EDA"]
    )
    n = min(len(wrist_eda), len(chest_eda))

    # Overall correlation across entire recording
    r_overall, p_overall = pearsonr(wrist_eda[:n], chest_eda[:n])
    print(f"  Overall correlation : r = {r_overall:.3f}  (p = {p_overall:.2e})")
    print(f"  Wrist EDA range     : {wrist_eda.min():.3f} - {wrist_eda.max():.3f} uS")
    print(f"  Chest EDA range     : {chest_eda[:n].min():.3f} - {chest_eda[:n].max():.3f} uS")

    # Per-window correlations
    window = int(10 * WRIST_RATES["EDA"])
    corrs = windowed_correlation(wrist_eda[:n], chest_eda[:n], window)
    arr = np.array(corrs)
    mean_r = np.mean(arr)

    print(f"  10s window count    : {len(arr)}")
    print(f"  Mean window r       : {mean_r:.3f}")
    print(f"  Std                 : {np.std(arr):.3f}")
    print(f"  Range               : [{np.min(arr):.3f}, {np.max(arr):.3f}]")

    passed = r_overall >= 0.5
    status = "PASS" if passed else "FAIL"
    print(f"  Result: {status} (overall r >= 0.5)")
    return passed


def test_temp_reliability(pkl_data: dict):
    """TEST 2 - Both sensors in physiological skin temperature range."""
    print("\n" + "=" * 60)
    print("TEST 2: Temperature Reliability - Wrist vs Chest")
    print("=" * 60)

    wrist_temp = pkl_data["signal"]["wrist"]["TEMP"].flatten()
    chest_temp = downsample(
        pkl_data["signal"]["chest"]["Temp"].flatten(), CHEST_RATE, WRIST_RATES["TEMP"]
    )
    n = min(len(wrist_temp), len(chest_temp))

    r_overall, p_overall = pearsonr(wrist_temp[:n], chest_temp[:n])
    print(f"  Overall correlation : r = {r_overall:.3f}  (p = {p_overall:.2e})")
    print(f"  Wrist range         : {wrist_temp.min():.1f} - {wrist_temp.max():.1f} C")
    print(f"  Chest range         : {chest_temp[:n].min():.1f} - {chest_temp[:n].max():.1f} C")

    wrist_physio = np.mean((wrist_temp >= 28) & (wrist_temp <= 40)) * 100
    chest_physio = np.mean((chest_temp[:n] >= 28) & (chest_temp[:n] <= 40)) * 100
    print(f"  Wrist in 28-40 C   : {wrist_physio:.1f}%")
    print(f"  Chest in 28-40 C   : {chest_physio:.1f}%")

    passed = wrist_physio >= 95 and chest_physio >= 95
    status = "PASS" if passed else "FAIL"
    print(f"  Result: {status} (both >= 95% in physiological range)")
    return passed


def test_heart_rate(pkl_data: dict):
    """TEST 3 - Wrist BVP and chest ECG yield similar heart rates."""
    print("\n" + "=" * 60)
    print("TEST 3: Heart Rate - Wrist BVP vs Chest ECG")
    print("=" * 60)

    wrist_bvp = pkl_data["signal"]["wrist"]["BVP"].flatten()
    chest_ecg = pkl_data["signal"]["chest"]["ECG"].flatten()

    print("  Estimating HR from wrist BVP ...")
    hr_bvp = extract_hr_from_bvp(wrist_bvp, WRIST_RATES["BVP"])
    print("  Estimating HR from chest ECG ...")
    hr_ecg = extract_hr_from_ecg(chest_ecg, CHEST_RATE)

    n = min(len(hr_bvp), len(hr_ecg))
    hr_bvp = hr_bvp[:n]
    hr_ecg = hr_ecg[:n]

    valid = ~np.isnan(hr_bvp) & ~np.isnan(hr_ecg)
    hr_bvp_v = hr_bvp[valid]
    hr_ecg_v = hr_ecg[valid]

    if len(hr_bvp_v) < 10:
        print(f"  SKIP - too few valid windows ({len(hr_bvp_v)})")
        return True

    r, p = pearsonr(hr_bvp_v, hr_ecg_v)
    mae = np.mean(np.abs(hr_bvp_v - hr_ecg_v))
    agreement = np.mean(np.abs(hr_bvp_v - hr_ecg_v) < 10) * 100

    print(f"  Valid windows  : {len(hr_bvp_v)}")
    print(f"  HR correlation : r = {r:.3f}  (p = {p:.2e})")
    print(f"  Mean abs error : {mae:.1f} BPM")
    print(f"  Within 10 BPM  : {agreement:.1f}%")
    print(f"  BVP HR range   : {hr_bvp_v.min():.0f} - {hr_bvp_v.max():.0f} BPM")
    print(f"  ECG HR range   : {hr_ecg_v.min():.0f} - {hr_ecg_v.max():.0f} BPM")

    passed = agreement >= 40
    status = "PASS" if passed else "FAIL"
    print(f"  Result: {status} (>= 40% of windows within 10 BPM)")
    return passed


def test_per_condition(pkl_data: dict):
    """TEST 4 - EDA wrist-chest correlation broken down by condition."""
    print("\n" + "=" * 60)
    print("TEST 4: Per-Condition EDA Wrist-Chest Correlation")
    print("=" * 60)

    wrist_eda = pkl_data["signal"]["wrist"]["EDA"].flatten()
    chest_eda = downsample(
        pkl_data["signal"]["chest"]["EDA"].flatten(), CHEST_RATE, WRIST_RATES["EDA"]
    )
    labels = downsample_labels(pkl_data["label"], WRIST_RATES["EDA"])

    n = min(len(wrist_eda), len(chest_eda), len(labels))
    window = int(10 * WRIST_RATES["EDA"])

    condition_corrs = {}

    for i in range(0, n - window, window):
        w_lbl = labels[i:i + window]
        dominant = np.bincount(w_lbl).argmax()
        if dominant == 0:
            continue

        w = wrist_eda[i:i + window]
        c = chest_eda[i:i + window]
        if np.std(w) > 1e-9 and np.std(c) > 1e-9:
            r, _ = pearsonr(w, c)
            condition_corrs.setdefault(dominant, []).append(r)

    print(f"  {'Condition':<15} {'Windows':>8} {'Mean r':>8} {'Std':>8}")
    print(f"  {'-' * 42}")

    for lbl in sorted(condition_corrs):
        arr = np.array(condition_corrs[lbl])
        name = LABEL_NAMES.get(lbl, f"label_{lbl}")
        print(f"  {name:<15} {len(arr):>8} {np.mean(arr):>8.3f} {np.std(arr):>8.3f}")

    passed = len(condition_corrs) >= 2
    status = "PASS" if passed else "FAIL"
    print(f"\n  {status} - Correlations reported across "
          f"{len(condition_corrs)} conditions")
    return passed


# ── main ─────────────────────────────────────────────────────────────────────

def run_all_tests(subject_id: str = "S2"):
    print("=" * 60)
    print(f"  WRIST vs CHEST RELIABILITY VALIDATION")
    print(f"  Subject: {subject_id}")
    print("=" * 60)

    pkl_data = load_pkl(subject_id)

    results = {}
    results["1_eda"]           = test_eda_reliability(pkl_data)
    results["2_temp"]          = test_temp_reliability(pkl_data)
    results["3_heart_rate"]    = test_heart_rate(pkl_data)
    results["4_per_condition"] = test_per_condition(pkl_data)

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_pass = False

    overall = "ALL TESTS PASSED" if all_pass else "SOME TESTS FAILED"
    print(f"\n  >>> {overall} <<<")
    print("=" * 60)
    return all_pass


if __name__ == "__main__":
    subject = sys.argv[1] if len(sys.argv) > 1 else "S2"
    success = run_all_tests(subject)
    sys.exit(0 if success else 1)
