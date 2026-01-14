"""
Real-Time Data Quality Monitor with 10-Second Sliding Window
Simulates real-time processing of E4 wristband data using a sliding window approach
"""

import pandas as pd
import numpy as np
import os
import time
import sys
from datetime import datetime
from scipy import signal
from scipy.stats import entropy
from typing import Dict, Tuple, Optional


class SlidingWindowQualityMonitor:
    """Real-time quality monitoring using sliding windows"""
    
    def __init__(self, window_size_seconds: float = 10.0):
        self.window_size = window_size_seconds
        self.sample_rates = {
            'acc': 32.0,    # Hz
            'bvp': 64.0,    # Hz
            'eda': 4.0,     # Hz
            'temp': 4.0     # Hz
        }
    
    def calculate_acc_quality(self, acc_data: np.ndarray) -> Tuple[float, float, float]:
        """Calculate ACC quality metrics for a window"""
        if len(acc_data) < 10:
            return 0.0, 0.0, 0.0
        
        magnitude = np.sqrt(np.sum(acc_data**2, axis=1))
        
        # On-body detection (variance threshold)
        variance = np.var(magnitude)
        on_body = 100.0 if variance > 0.01 else 0.0
        
        # Signal quality checks
        not_stuck = 100.0 if variance > 0.001 else 0.0
        reasonable = np.mean((magnitude > 0.1) & (magnitude < 3.0)) * 100
        not_saturated = np.mean(np.abs(acc_data) < 1.9) * 100
        signal_quality = not_stuck * 0.3 + reasonable * 0.4 + not_saturated * 0.3
        
        # Completeness (assume complete if we have data)
        completeness = 100.0
        
        return completeness, on_body, signal_quality
    
    def calculate_bvp_quality(self, bvp_data: np.ndarray, sample_rate: float) -> Tuple[float, float, float]:
        """Calculate BVP quality using spectral entropy"""
        if len(bvp_data) < 64:
            return 0.0, 0.0, 0.0
        
        # Compute power spectral density
        freqs, psd = signal.periodogram(bvp_data, fs=sample_rate)
        
        # Focus on heart rate range (0.5-4 Hz)
        hr_range = (freqs >= 0.5) & (freqs <= 4.0)
        psd_hr = psd[hr_range]
        
        if len(psd_hr) > 0 and np.sum(psd_hr) > 0:
            psd_norm = psd_hr / np.sum(psd_hr)
            spectral_entropy = entropy(psd_norm)
            max_entropy = np.log(len(psd_norm)) if len(psd_norm) > 0 else 1
            signal_quality = (1 - spectral_entropy / max_entropy) * 100 if max_entropy > 0 else 0
            signal_quality = max(0, min(100, signal_quality))
        else:
            signal_quality = 0.0
        
        completeness = 100.0
        on_body = 100.0 if np.std(bvp_data) > 0.1 else 0.0
        
        return completeness, on_body, signal_quality
    
    def calculate_eda_quality(self, eda_data: np.ndarray, sample_rate: float) -> Tuple[float, float, float]:
        """Calculate EDA quality metrics"""
        if len(eda_data) < 4:
            return 0.0, 0.0, 0.0
        
        # On-body detection
        on_body = 100.0 if np.mean(eda_data > 0.05) > 0.5 else 0.0
        
        # Signal quality - check rate of change
        max_reasonable_change = 0.5  # μS/second
        changes = np.abs(np.diff(eda_data)) * sample_rate
        signal_quality = np.mean(changes <= max_reasonable_change) * 100
        
        completeness = 100.0
        
        return completeness, on_body, signal_quality
    
    def calculate_temp_quality(self, temp_data: np.ndarray, sample_rate: float) -> Tuple[float, float, float]:
        """Calculate temperature quality metrics"""
        if len(temp_data) < 4:
            return 0.0, 0.0, 0.0
        
        # On-body detection (physiological range)
        on_body = np.mean((temp_data >= 30.0) & (temp_data <= 40.0)) * 100
        
        # Signal quality
        in_range = np.mean((temp_data >= 30.0) & (temp_data <= 40.0))
        max_change_rate = 0.1  # °C/second
        changes = np.abs(np.diff(temp_data)) * sample_rate
        reasonable_changes = np.mean(changes <= max_change_rate)
        signal_quality = (in_range * 0.5 + reasonable_changes * 0.5) * 100
        
        completeness = 100.0
        
        return completeness, on_body, signal_quality
    
    def calculate_aggregate_score(self, completeness: float, on_body: float, signal_quality: float) -> float:
        """Calculate weighted aggregate score"""
        return completeness * 0.3 + on_body * 0.3 + signal_quality * 0.4


def load_e4_data(e4_folder: str) -> Dict:
    """Load all E4 sensor data"""
    data = {}
    
    # Load ACC
    acc_file = os.path.join(e4_folder, 'ACC.csv')
    if os.path.exists(acc_file):
        with open(acc_file, 'r') as f:
            initial_ts = float(f.readline().strip().split(',')[0])
            sample_rate = float(f.readline().strip().split(',')[0])
        acc_df = pd.read_csv(acc_file, skiprows=2, header=None)
        acc_df.columns = ['x', 'y', 'z']
        acc_df = acc_df / 64.0  # Convert to g
        timestamps = initial_ts + np.arange(len(acc_df)) / sample_rate
        data['acc'] = {'data': acc_df.values, 'timestamps': timestamps, 'sample_rate': sample_rate}
    
    # Load BVP
    bvp_file = os.path.join(e4_folder, 'BVP.csv')
    if os.path.exists(bvp_file):
        with open(bvp_file, 'r') as f:
            initial_ts = float(f.readline().strip().split(',')[0])
            sample_rate = float(f.readline().strip().split(',')[0])
        bvp_df = pd.read_csv(bvp_file, skiprows=2, header=None)
        timestamps = initial_ts + np.arange(len(bvp_df)) / sample_rate
        data['bvp'] = {'data': bvp_df[0].values, 'timestamps': timestamps, 'sample_rate': sample_rate}
    
    # Load EDA
    eda_file = os.path.join(e4_folder, 'EDA.csv')
    if os.path.exists(eda_file):
        with open(eda_file, 'r') as f:
            initial_ts = float(f.readline().strip().split(',')[0])
            sample_rate = float(f.readline().strip().split(',')[0])
        eda_df = pd.read_csv(eda_file, skiprows=2, header=None)
        timestamps = initial_ts + np.arange(len(eda_df)) / sample_rate
        data['eda'] = {'data': eda_df[0].values, 'timestamps': timestamps, 'sample_rate': sample_rate}
    
    # Load TEMP
    temp_file = os.path.join(e4_folder, 'TEMP.csv')
    if os.path.exists(temp_file):
        with open(temp_file, 'r') as f:
            initial_ts = float(f.readline().strip().split(',')[0])
            sample_rate = float(f.readline().strip().split(',')[0])
        temp_df = pd.read_csv(temp_file, skiprows=2, header=None)
        timestamps = initial_ts + np.arange(len(temp_df)) / sample_rate
        data['temp'] = {'data': temp_df[0].values, 'timestamps': timestamps, 'sample_rate': sample_rate}
    
    return data


def get_window_data(sensor_data: Dict, start_time: float, window_size: float) -> Optional[np.ndarray]:
    """Extract data for a specific time window"""
    timestamps = sensor_data['timestamps']
    data = sensor_data['data']
    
    mask = (timestamps >= start_time) & (timestamps < start_time + window_size)
    indices = np.where(mask)[0]
    
    if len(indices) == 0:
        return None
    
    return data[indices] if data.ndim == 1 else data[indices, :]


def clear_screen():
    """Clear terminal screen (cross-platform)"""
    os.system('cls' if os.name == 'nt' else 'clear')


def enable_windows_ansi():
    """Enable ANSI escape codes on Windows"""
    if os.name == 'nt':
        import ctypes
        kernel32 = ctypes.windll.kernel32
        # Enable ENABLE_VIRTUAL_TERMINAL_PROCESSING
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)


def get_quality_indicator(value: float) -> str:
    """Get a compact quality indicator"""
    if value >= 80:
        return f"\033[92m{value:5.1f}%\033[0m"  # Green
    elif value >= 60:
        return f"\033[93m{value:5.1f}%\033[0m"  # Yellow
    else:
        return f"\033[91m{value:5.1f}%\033[0m"  # Red


def print_quality_bar(label: str, value: float, width: int = 20) -> str:
    """Create a visual progress bar for quality score"""
    filled = int(value / 100 * width)
    bar = '█' * filled + '░' * (width - filled)
    
    # Color based on quality
    if value >= 80:
        color = '\033[92m'  # Green
    elif value >= 60:
        color = '\033[93m'  # Yellow
    else:
        color = '\033[91m'  # Red
    reset = '\033[0m'
    
    return f"{label}: {color}{bar}{reset} {value:5.1f}%"


def run_realtime_monitor(e4_folder: str, window_size: float = 10.0, update_interval: float = 0.5, speed: float = 1.0):
    """
    Run real-time quality monitoring simulation
    
    Args:
        e4_folder: Path to E4 data folder
        window_size: Size of sliding window in seconds
        update_interval: How often to update display (in simulated seconds)
        speed: Playback speed multiplier (1.0 = real-time, 10.0 = 10x faster)
    """
    # Clear screen first to remove any previous output
    clear_screen()
    
    print(f"{'='*70}")
    print("  REAL-TIME DATA QUALITY MONITOR - 10 Second Sliding Window")
    print(f"{'='*70}")
    print(f"\nLoading E4 data from: {e4_folder}")
    
    # Load all data
    data = load_e4_data(e4_folder)
    
    if not data:
        print("Error: No data found!")
        return
    
    print(f"Loaded sensors: {list(data.keys())}")
    
    # Initialize monitor
    monitor = SlidingWindowQualityMonitor(window_size)
    
    # Find time range
    all_timestamps = []
    for sensor_data in data.values():
        all_timestamps.extend([sensor_data['timestamps'].min(), sensor_data['timestamps'].max()])
    
    start_time = min(all_timestamps)
    end_time = max(all_timestamps) - window_size
    total_duration = end_time - start_time
    
    print(f"\nTotal recording duration: {total_duration/60:.1f} minutes")
    print(f"Window size: {window_size} seconds")
    print(f"Playback speed: {speed}x")
    print(f"\nStarting in 2 seconds...")
    
    time.sleep(2)
    
    # Enable ANSI codes and clear screen once
    enable_windows_ansi()
    clear_screen()
    
    # Hide cursor for cleaner display
    sys.stdout.write('\033[?25l')
    sys.stdout.flush()
    
    current_time = start_time
    
    # Store history for overall statistics
    quality_history = []
    
    try:
        while current_time < end_time:
            elapsed = current_time - start_time
            progress = elapsed / total_duration * 100
            
            # Calculate quality for each sensor in current window
            sensor_scores = {}
            
            for sensor_name, sensor_data in data.items():
                window_data = get_window_data(sensor_data, current_time, window_size)
                
                if window_data is not None and len(window_data) > 0:
                    sample_rate = sensor_data['sample_rate']
                    
                    if sensor_name == 'acc':
                        comp, on_body, sig_qual = monitor.calculate_acc_quality(window_data)
                    elif sensor_name == 'bvp':
                        comp, on_body, sig_qual = monitor.calculate_bvp_quality(window_data, sample_rate)
                    elif sensor_name == 'eda':
                        comp, on_body, sig_qual = monitor.calculate_eda_quality(window_data, sample_rate)
                    elif sensor_name == 'temp':
                        comp, on_body, sig_qual = monitor.calculate_temp_quality(window_data, sample_rate)
                    else:
                        continue
                    
                    aggregate = monitor.calculate_aggregate_score(comp, on_body, sig_qual)
                    sensor_scores[sensor_name] = {
                        'completeness': comp,
                        'on_body': on_body,
                        'signal_quality': sig_qual,
                        'aggregate': aggregate
                    }
            
            # Calculate overall aggregate across all sensors
            if sensor_scores:
                overall_aggregate = np.mean([s['aggregate'] for s in sensor_scores.values()])
                quality_history.append(overall_aggregate)
            else:
                overall_aggregate = 0.0
            
            # Build and print display (without clearing first - reduces flicker)
            output_lines = []
            output_lines.append(f"{'='*70}")
            output_lines.append("  REAL-TIME DATA QUALITY MONITOR | 10-Second Sliding Window")
            output_lines.append(f"{'='*70}")
            
            # Progress bar
            prog_width = 50
            prog_filled = int(progress / 100 * prog_width)
            prog_bar = '█' * prog_filled + '░' * (prog_width - prog_filled)
            output_lines.append(f"  [{prog_bar}] {progress:5.1f}%")
            output_lines.append(f"  Elapsed: {elapsed:7.1f}s / {total_duration:.0f}s | Window: {elapsed:.0f}s-{elapsed+window_size:.0f}s")
            output_lines.append(f"{'─'*70}")
            
            # Compact sensor display
            output_lines.append("  SENSOR   | Complete | On-Body  | Signal Q | AGGREGATE")
            output_lines.append(f"  {'─'*60}")
            
            for sensor_name in ['acc', 'bvp', 'eda', 'temp']:
                if sensor_name in sensor_scores:
                    s = sensor_scores[sensor_name]
                    output_lines.append(f"  {sensor_name.upper():8} |  {get_quality_indicator(s['completeness'])}  |  {get_quality_indicator(s['on_body'])}  |  {get_quality_indicator(s['signal_quality'])}  |  {get_quality_indicator(s['aggregate'])}")
                else:
                    output_lines.append(f"  {sensor_name.upper():8} |    ---   |    ---   |    ---   |    ---")
            
            output_lines.append(f"{'─'*70}")
            
            # Overall score
            if overall_aggregate >= 80:
                color, status = '\033[92m', "EXCELLENT"
            elif overall_aggregate >= 60:
                color, status = '\033[93m', "ACCEPTABLE"
            else:
                color, status = '\033[91m', "POOR"
            reset = '\033[0m'
            
            output_lines.append(f"  {color}>>> OVERALL QUALITY: {overall_aggregate:5.1f}%  [{status}] <<<{reset}")
            
            # Running stats
            if len(quality_history) > 1:
                output_lines.append(f"  Stats: Avg={np.mean(quality_history):5.1f}% | Min={np.min(quality_history):5.1f}% | Max={np.max(quality_history):5.1f}%")
            else:
                output_lines.append(f"  Stats: Collecting data...")
            
            output_lines.append(f"{'='*70}")
            output_lines.append("  Press Ctrl+C to stop")
            
            # Move cursor to home position and overwrite (no flash)
            sys.stdout.write('\033[H')  # Move cursor to top-left
            print('\n'.join(output_lines))
            sys.stdout.flush()
            
            # Advance time
            current_time += update_interval
            time.sleep(update_interval / speed)
    
    except KeyboardInterrupt:
        pass
    finally:
        # Show cursor again
        sys.stdout.write('\033[?25h')
        sys.stdout.flush()
    
    # Final summary
    clear_screen()
    print(f"\n{'='*70}")
    print("  MONITORING SESSION SUMMARY")
    print(f"{'='*70}")
    print(f"\n  Total windows analyzed: {len(quality_history)}")
    print(f"  Average quality score:  {np.mean(quality_history):.1f}%")
    print(f"  Minimum quality score:  {np.min(quality_history):.1f}%")
    print(f"  Maximum quality score:  {np.max(quality_history):.1f}%")
    print(f"  Standard deviation:     {np.std(quality_history):.1f}%")
    
    # Quality distribution
    excellent = np.sum(np.array(quality_history) >= 80) / len(quality_history) * 100
    acceptable = np.sum((np.array(quality_history) >= 60) & (np.array(quality_history) < 80)) / len(quality_history) * 100
    poor = np.sum(np.array(quality_history) < 60) / len(quality_history) * 100
    
    print(f"\n  Quality Distribution:")
    print(f"    Excellent (≥80%): {excellent:.1f}% of time")
    print(f"    Acceptable (60-80%): {acceptable:.1f}% of time")
    print(f"    Poor (<60%): {poor:.1f}% of time")
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    # Default path for S2 subject
    e4_folder = r"c:\Users\Leo\Desktop\Coding stuff\TYP\WESAD\S2\S2_E4_Data"
    
    # Run with 10x speed for demonstration (change to 1.0 for real-time)
    run_realtime_monitor(
        e4_folder=e4_folder,
        window_size=10.0,      # 10 second sliding window
        update_interval=2.0,   # Update every 2 seconds (simulated time)
        speed=10.0             # 10x playback speed (set to 1.0 for real-time)
    )
