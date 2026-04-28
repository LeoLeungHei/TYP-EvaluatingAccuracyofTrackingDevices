"""
Cascade Test: Off-Body Detection → Signal Quality → Aggregate Score
===================================================================
Verifies that when the device is off-body, signal quality scores also
degrade — so the effective penalty is much larger than the 20% on-body
weight alone.

Scenarios
---------
1. On-body synthetic data  → all three dimensions should score high
2. Off-body synthetic data → on-body fails AND signal quality degrades
3. Aggregate correctly reflects the cascade (off-body overall << on-body overall)
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from realtime_quality_monitor import SlidingWindowQualityMonitor


# ── synthetic data generators ────────────────────────────────────────────────

def make_on_body_data(monitor: SlidingWindowQualityMonitor):
    """Generate 10 s of plausible on-body sensor data."""
    window = monitor.window_size  # 10 s

    # ACC: gentle motion around 1 g with micro-movements
    # Variance of magnitude must exceed 0.01 for on-body detection
    n_acc = int(window * monitor.sample_rates['acc'])
    acc = np.column_stack([
        np.random.normal(0.0, 0.15, n_acc),   # x
        np.random.normal(0.0, 0.15, n_acc),   # y
        np.random.normal(1.0, 0.15, n_acc),   # z  (gravity)
    ])

    # BVP: clean sine at ~1.2 Hz (72 BPM) + small noise
    n_bvp = int(window * monitor.sample_rates['bvp'])
    t_bvp = np.linspace(0, window, n_bvp, endpoint=False)
    bvp = np.sin(2 * np.pi * 1.2 * t_bvp) + np.random.normal(0, 0.05, n_bvp)

    # EDA: slowly drifting around 2 µS (above 0.05 threshold)
    n_eda = int(window * monitor.sample_rates['eda'])
    eda = 2.0 + np.cumsum(np.random.normal(0, 0.005, n_eda))

    # TEMP: stable around 34 °C (well within 30–40 range)
    n_temp = int(window * monitor.sample_rates['temp'])
    temp = 34.0 + np.random.normal(0, 0.05, n_temp)

    return acc, bvp, eda, temp


def make_off_body_data(monitor: SlidingWindowQualityMonitor):
    """Generate 10 s of off-body (desk / ambient) sensor data."""
    window = monitor.window_size

    # ACC: near-zero variance (stationary on a desk)
    n_acc = int(window * monitor.sample_rates['acc'])
    acc = np.column_stack([
        np.full(n_acc, 0.0),
        np.full(n_acc, 0.0),
        np.full(n_acc, 1.0),   # gravity only, no movement
    ])

    # BVP: random noise, no cardiac periodicity
    n_bvp = int(window * monitor.sample_rates['bvp'])
    bvp = np.random.normal(0, 0.01, n_bvp)

    # EDA: near-zero (no skin contact)
    n_eda = int(window * monitor.sample_rates['eda'])
    eda = np.random.uniform(0.0, 0.02, n_eda)

    # TEMP: ambient ~22 °C (outside 30–40 range)
    n_temp = int(window * monitor.sample_rates['temp'])
    temp = 22.0 + np.random.normal(0, 0.3, n_temp)

    return acc, bvp, eda, temp


# ── score helper ─────────────────────────────────────────────────────────────

def score_all_sensors(monitor, acc, bvp, eda, temp):
    """Run quality calculations and return per-sensor + overall results."""
    results = {}

    comp, on_body, sig_q, _ = monitor.calculate_acc_quality(acc)
    agg = monitor.calculate_aggregate_score(comp, on_body, sig_q)
    results['acc'] = dict(completeness=comp, on_body=on_body,
                          signal_quality=sig_q, aggregate=agg)

    comp, on_body, sig_q, _ = monitor.calculate_bvp_quality(bvp, monitor.sample_rates['bvp'])
    agg = monitor.calculate_aggregate_score(comp, on_body, sig_q)
    results['bvp'] = dict(completeness=comp, on_body=on_body,
                          signal_quality=sig_q, aggregate=agg)

    comp, on_body, sig_q, _ = monitor.calculate_eda_quality(eda, monitor.sample_rates['eda'])
    agg = monitor.calculate_aggregate_score(comp, on_body, sig_q)
    results['eda'] = dict(completeness=comp, on_body=on_body,
                          signal_quality=sig_q, aggregate=agg)

    comp, on_body, sig_q, _ = monitor.calculate_temp_quality(temp, monitor.sample_rates['temp'])
    agg = monitor.calculate_aggregate_score(comp, on_body, sig_q)
    results['temp'] = dict(completeness=comp, on_body=on_body,
                           signal_quality=sig_q, aggregate=agg)

    overall = np.mean([s['aggregate'] for s in results.values()])
    return results, overall


# ── tests ────────────────────────────────────────────────────────────────────

def test_on_body_high_scores(monitor, results_on, overall_on):
    """All on-body sensors should score well across all dimensions."""
    print("\n" + "=" * 60)
    print("TEST 1: On-body data produces high quality scores")
    print("=" * 60)

    passed = True
    for sensor, s in results_on.items():
        ok = s['on_body'] and s['signal_quality'] >= 60 and s['aggregate'] >= 60
        tag = "OK" if ok else "FAIL"
        print(f"  {sensor.upper():4}  on_body={str(s['on_body']):5}  "
              f"sig_q={s['signal_quality']:5.1f}  agg={s['aggregate']:5.1f}  [{tag}]")
        if not ok:
            passed = False

    ok = overall_on >= 60
    print(f"\n  Overall = {overall_on:.1f}%  (>= 60)  [{'OK' if ok else 'FAIL'}]")
    if not ok:
        passed = False

    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def test_off_body_low_scores(monitor, results_off, overall_off):
    """Off-body sensors should fail on-body AND produce lower signal quality."""
    print("\n" + "=" * 60)
    print("TEST 2: Off-body data produces low quality scores")
    print("=" * 60)

    passed = True
    # At least 3 of 4 sensors should report off-body
    off_count = sum(1 for s in results_off.values() if not s['on_body'])
    ok = off_count >= 3
    print(f"  Sensors reporting off-body: {off_count}/4  (>= 3)  [{'OK' if ok else 'FAIL'}]")
    if not ok:
        passed = False

    for sensor, s in results_off.items():
        print(f"  {sensor.upper():4}  on_body={str(s['on_body']):5}  "
              f"sig_q={s['signal_quality']:5.1f}  agg={s['aggregate']:5.1f}")

    # Overall should be significantly below the on-body score
    ok = overall_off < 70
    print(f"\n  Overall = {overall_off:.1f}%  (< 70)  [{'OK' if ok else 'FAIL'}]")
    if not ok:
        passed = False

    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def test_cascade_effect(monitor, results_on, results_off, overall_on, overall_off):
    """Off-body penalty should be > 20 points due to cascade into signal quality."""
    print("\n" + "=" * 60)
    print("TEST 3: Cascade — off-body degrades signal quality too")
    print("=" * 60)

    passed = True

    # Check signal quality is lower off-body for BVP and TEMP (clear cascade sensors).
    # NOTE: EDA signal quality uses rate-of-change, so flat near-zero off-body
    # readings still pass — the cascade does NOT apply to EDA's signal quality.
    for sensor in ['bvp', 'temp']:
        on_sq = results_on[sensor]['signal_quality']
        off_sq = results_off[sensor]['signal_quality']
        drop = on_sq - off_sq
        ok = drop > 0
        print(f"  {sensor.upper():4}  sig_q on-body={on_sq:5.1f}  "
              f"off-body={off_sq:5.1f}  drop={drop:+.1f}  [{'OK' if ok else 'FAIL'}]")

        if not ok:
            passed = False

    # Report EDA separately as a known non-cascading sensor
    eda_on  = results_on['eda']['signal_quality']
    eda_off = results_off['eda']['signal_quality']
    print(f"  EDA   sig_q on-body={eda_on:5.1f}  "
          f"off-body={eda_off:5.1f}  (no cascade — flat signal passes rate-of-change check)")

    # The overall drop should exceed the 20-point on-body weight alone
    overall_drop = overall_on - overall_off
    # If there were no cascade, the maximum drop from on-body alone would be
    # 20 points (0.2 * 100). The cascade should push the actual drop higher.
    ok = overall_drop > 20
    print(f"\n  Overall on-body  = {overall_on:.1f}%")
    print(f"  Overall off-body = {overall_off:.1f}%")
    print(f"  Drop             = {overall_drop:.1f} pts  (> 20 proves cascade)  "
          f"[{'OK' if ok else 'FAIL'}]")
    if not ok:
        passed = False

    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def test_aggregate_formula(monitor):
    """Verify the aggregate formula matches documented weights."""
    print("\n" + "=" * 60)
    print("TEST 4: Aggregate formula uses correct weights (20/20/60)")
    print("=" * 60)

    passed = True
    cases = [
        # (completeness, on_body, signal_quality, expected)
        (100.0, True,  100.0, 100.0),
        (100.0, False, 100.0,  80.0),   # only on-body penalty
        (  0.0, True,  100.0,  80.0),   # only completeness penalty
        (100.0, True,    0.0,  40.0),   # only signal quality penalty
        (  0.0, False,   0.0,   0.0),
        ( 50.0, False,  50.0,  40.0),   # 50*0.2 + 0*0.2 + 50*0.6 = 10+0+30
    ]

    for comp, on_body, sig_q, expected in cases:
        actual = monitor.calculate_aggregate_score(comp, on_body, sig_q)
        ok = abs(actual - expected) < 0.01
        tag = "OK" if ok else "FAIL"
        print(f"  agg({comp:5.1f}, {str(on_body):5}, {sig_q:5.1f}) "
              f"= {actual:5.1f}  expected {expected:5.1f}  [{tag}]")
        if not ok:
            passed = False

    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    return passed


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  CASCADE TEST: Off-Body → Signal Quality → Aggregate")
    print("=" * 60)

    monitor = SlidingWindowQualityMonitor(window_size_seconds=10.0)

    # Generate synthetic data
    acc_on, bvp_on, eda_on, temp_on = make_on_body_data(monitor)
    acc_off, bvp_off, eda_off, temp_off = make_off_body_data(monitor)

    results_on,  overall_on  = score_all_sensors(monitor, acc_on, bvp_on, eda_on, temp_on)
    results_off, overall_off = score_all_sensors(monitor, acc_off, bvp_off, eda_off, temp_off)

    results = {}
    results["1_on_body_high"]    = test_on_body_high_scores(monitor, results_on, overall_on)
    results["2_off_body_low"]    = test_off_body_low_scores(monitor, results_off, overall_off)
    results["3_cascade"]         = test_cascade_effect(monitor, results_on, results_off,
                                                       overall_on, overall_off)
    results["4_aggregate_formula"] = test_aggregate_formula(monitor)

    # Summary
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
    main()
