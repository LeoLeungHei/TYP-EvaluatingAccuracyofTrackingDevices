"""
Data Quality Assessment Framework for Wearable Device Data
Based on methodologies from Böttcher et al. (2022) and project requirements
"""

import pandas as pd
import numpy as np
from scipy import signal, stats
from scipy.stats import entropy
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


@dataclass
class QualityMetrics:
    """Container for data quality assessment results"""
    completeness: float  # Percentage of expected data present (0-100)
    on_body_detection: float  # Percentage of time device was on body (0-100)
    signal_quality: float  # Signal quality score (0-100)
    aggregate_score: float  # Overall reliability score (0-100)
    quality_timeline: pd.DataFrame  # Time-series of quality indicators
    metadata: Dict  # Additional contextual information


class DataCompletenessAssessor:
    """Assess data completeness by detecting gaps and missing samples"""
    
    @staticmethod
    def assess(timestamps: np.ndarray, expected_sample_rate: float, 
               tolerance: float = 0.1) -> Tuple[float, pd.DataFrame]:
        """
        Calculate data completeness percentage
        
        Args:
            timestamps: Array of unix timestamps
            expected_sample_rate: Expected sampling rate in Hz
            tolerance: Tolerance for timing variations (default 10%)
            
        Returns:
            completeness_percentage: Overall completeness (0-100)
            gaps_df: DataFrame with detected gaps
        """
        if len(timestamps) < 2:
            return 0.0, pd.DataFrame()
        
        # Calculate expected interval between samples
        expected_interval = 1.0 / expected_sample_rate
        tolerance_interval = expected_interval * (1 + tolerance)
        
        # Calculate actual intervals
        intervals = np.diff(timestamps)
        
        # Detect gaps (intervals larger than expected + tolerance)
        gap_mask = intervals > tolerance_interval
        gaps = intervals[gap_mask]
        gap_indices = np.where(gap_mask)[0]
        
        # Calculate expected total samples based on time span
        total_duration = timestamps[-1] - timestamps[0]
        expected_samples = int(total_duration * expected_sample_rate)
        actual_samples = len(timestamps)
        
        # Completeness calculation
        completeness = (actual_samples / expected_samples * 100) if expected_samples > 0 else 0
        completeness = min(completeness, 100.0)  # Cap at 100%
        
        # Create gaps dataframe
        gaps_df = pd.DataFrame({
            'gap_start_idx': gap_indices,
            'gap_start_time': timestamps[gap_indices],
            'gap_duration': gaps,
            'missing_samples': (gaps / expected_interval).astype(int)
        }) if len(gaps) > 0 else pd.DataFrame()
        
        return completeness, gaps_df


class OnBodyDetector:
    """Detect when device is being worn using multi-modal approach"""
    
    @staticmethod
    def detect_from_acc(acc_data: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        """
        Detect on-body periods from accelerometer data
        Device off-body typically shows very low variance
        
        Args:
            acc_data: Nx3 array of accelerometer data (x, y, z)
            threshold: Minimum standard deviation for on-body detection
            
        Returns:
            on_body_mask: Boolean array indicating on-body periods
        """
        # Calculate magnitude of acceleration
        magnitude = np.sqrt(np.sum(acc_data**2, axis=1))
        
        # Use rolling window to calculate local variance
        window_size = 32  # 1 second at 32Hz
        rolling_std = pd.Series(magnitude).rolling(window=window_size, center=True).std()
        
        # On-body if variance exceeds threshold
        on_body = rolling_std > threshold
        return on_body.fillna(False).values
    
    @staticmethod
    def detect_from_temp(temp_data: np.ndarray, min_temp: float = 30.0, 
                         max_temp: float = 40.0) -> np.ndarray:
        """
        Detect on-body periods from temperature data
        Body temperature typically in range 32-38°C
        
        Args:
            temp_data: Array of temperature readings in Celsius
            min_temp: Minimum temperature for on-body (default 30°C)
            max_temp: Maximum temperature for on-body (default 40°C)
            
        Returns:
            on_body_mask: Boolean array indicating on-body periods
        """
        return (temp_data >= min_temp) & (temp_data <= max_temp)
    
    @staticmethod
    def detect_from_eda(eda_data: np.ndarray, min_eda: float = 0.05) -> np.ndarray:
        """
        Detect on-body periods from EDA data
        Off-body typically shows very low or zero EDA
        
        Args:
            eda_data: Array of EDA readings in microsiemens
            min_eda: Minimum EDA for on-body detection
            
        Returns:
            on_body_mask: Boolean array indicating on-body periods
        """
        return eda_data > min_eda
    
    @staticmethod
    def combine_detections(acc_on_body: Optional[np.ndarray] = None,
                          temp_on_body: Optional[np.ndarray] = None,
                          eda_on_body: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
        """
        Combine multiple on-body detection signals using majority voting
        
        Returns:
            combined_mask: Boolean array of final on-body detection
            on_body_percentage: Percentage of time device was on body
        """
        signals = [s for s in [acc_on_body, temp_on_body, eda_on_body] if s is not None]
        
        if not signals:
            return np.array([]), 0.0
        
        # Ensure all signals are same length (use shortest)
        min_length = min(len(s) for s in signals)
        signals = [s[:min_length] for s in signals]
        
        # Majority voting: on-body if majority of signals agree
        stacked = np.stack(signals)
        combined = np.sum(stacked, axis=0) > (len(signals) / 2)
        
        percentage = np.mean(combined) * 100
        return combined, percentage


class SignalQualityEvaluator:
    """Evaluate signal quality for different physiological modalities"""
    
    @staticmethod
    def evaluate_bvp_quality(bvp_data: np.ndarray, sample_rate: float,
                            window_size: int = 256) -> Tuple[float, np.ndarray]:
        """
        Evaluate BVP quality using spectral entropy
        Lower entropy = better quality (more regular heartbeat signal)
        
        Args:
            bvp_data: Array of BVP samples
            sample_rate: Sampling rate in Hz
            window_size: Window size for sliding quality assessment
            
        Returns:
            mean_quality: Overall quality percentage (0-100)
            quality_timeline: Per-window quality scores
        """
        if len(bvp_data) < window_size:
            return 0.0, np.array([])
        
        # Calculate spectral entropy for sliding windows
        quality_scores = []
        hop_size = window_size // 2
        
        for i in range(0, len(bvp_data) - window_size, hop_size):
            window = bvp_data[i:i+window_size]
            
            # Compute power spectral density
            freqs, psd = signal.periodogram(window, fs=sample_rate)
            
            # Focus on physiological heart rate range (0.5-4 Hz = 30-240 bpm)
            hr_range = (freqs >= 0.5) & (freqs <= 4.0)
            psd_hr = psd[hr_range]
            
            if len(psd_hr) > 0 and np.sum(psd_hr) > 0:
                # Normalize PSD to create probability distribution
                psd_norm = psd_hr / np.sum(psd_hr)
                
                # Calculate Shannon entropy
                spectral_entropy = entropy(psd_norm)
                
                # Convert to quality score (lower entropy = higher quality)
                # Normalize: typical entropy range is 0-5, invert to quality
                max_entropy = np.log(len(psd_norm))  # Maximum possible entropy
                quality = (1 - spectral_entropy / max_entropy) * 100 if max_entropy > 0 else 0
                quality = max(0, min(100, quality))
            else:
                quality = 0.0
            
            quality_scores.append(quality)
        
        quality_array = np.array(quality_scores)
        mean_quality = np.mean(quality_array) if len(quality_array) > 0 else 0.0
        
        return mean_quality, quality_array
    
    @staticmethod
    def evaluate_eda_quality(eda_data: np.ndarray, sample_rate: float,
                            window_size_seconds: int = 5) -> Tuple[float, np.ndarray]:
        """
        Evaluate EDA quality using rate of amplitude change
        Realistic EDA shows gradual changes, not abrupt spikes
        
        Args:
            eda_data: Array of EDA samples in microsiemens
            sample_rate: Sampling rate in Hz
            window_size_seconds: Window size in seconds
            
        Returns:
            mean_quality: Overall quality percentage (0-100)
            quality_timeline: Per-window quality scores
        """
        if len(eda_data) < 2:
            return 0.0, np.array([])
        
        window_size = int(window_size_seconds * sample_rate)
        hop_size = window_size // 2
        
        quality_scores = []
        
        # Define thresholds for rate of change (μS/second)
        # EDA changes should be gradual (typical range: 0-0.5 μS/s)
        max_reasonable_change = 0.5  # μS/second
        
        for i in range(0, len(eda_data) - window_size, hop_size):
            window = eda_data[i:i+window_size]
            
            # Calculate rate of change
            changes = np.abs(np.diff(window)) * sample_rate  # Convert to per-second
            
            # Quality based on proportion of reasonable changes
            reasonable = changes <= max_reasonable_change
            quality = np.mean(reasonable) * 100
            
            quality_scores.append(quality)
        
        quality_array = np.array(quality_scores)
        mean_quality = np.mean(quality_array) if len(quality_array) > 0 else 0.0
        
        return mean_quality, quality_array
    
    @staticmethod
    def evaluate_temp_quality(temp_data: np.ndarray, sample_rate: float,
                             window_size_seconds: int = 10) -> Tuple[float, np.ndarray]:
        """
        Evaluate temperature quality using rate of change and physiological range
        Body temperature changes very slowly
        
        Args:
            temp_data: Array of temperature samples in Celsius
            sample_rate: Sampling rate in Hz
            window_size_seconds: Window size in seconds
            
        Returns:
            mean_quality: Overall quality percentage (0-100)
            quality_timeline: Per-window quality scores
        """
        if len(temp_data) < 2:
            return 0.0, np.array([])
        
        window_size = int(window_size_seconds * sample_rate)
        hop_size = window_size // 2
        
        quality_scores = []
        
        # Temperature should be in physiological range and change slowly
        valid_range = (30.0, 40.0)  # °C
        max_change_rate = 0.1  # °C/second (very conservative)
        
        for i in range(0, len(temp_data) - window_size, hop_size):
            window = temp_data[i:i+window_size]
            
            # Check physiological range
            in_range = np.mean((window >= valid_range[0]) & (window <= valid_range[1]))
            
            # Check rate of change
            changes = np.abs(np.diff(window)) * sample_rate
            reasonable_changes = np.mean(changes <= max_change_rate)
            
            # Combined quality score
            quality = (in_range * 0.5 + reasonable_changes * 0.5) * 100
            quality_scores.append(quality)
        
        quality_array = np.array(quality_scores)
        mean_quality = np.mean(quality_array) if len(quality_array) > 0 else 0.0
        
        return mean_quality, quality_array
    
    @staticmethod
    def evaluate_acc_quality(acc_data: np.ndarray, sample_rate: float) -> Tuple[float, np.ndarray]:
        """
        Evaluate accelerometer quality by detecting movement artifacts
        Extreme values or stuck sensors indicate poor quality
        
        Args:
            acc_data: Nx3 array of accelerometer data
            sample_rate: Sampling rate in Hz
            
        Returns:
            mean_quality: Overall quality percentage (0-100)
            quality_timeline: Per-window quality scores
        """
        if len(acc_data) < 32:
            return 0.0, np.array([])
        
        window_size = int(sample_rate * 2)  # 2-second windows
        hop_size = window_size // 2
        
        quality_scores = []
        
        # Calculate magnitude
        magnitude = np.sqrt(np.sum(acc_data**2, axis=1))
        
        for i in range(0, len(magnitude) - window_size, hop_size):
            window = magnitude[i:i+window_size]
            
            # Check for stuck sensor (no variance)
            variance = np.var(window)
            not_stuck = variance > 0.001
            
            # Check for extreme values (should be around 1g typically)
            reasonable = np.mean((window > 0.1) & (window < 3.0))
            
            # Check for saturation (max values)
            not_saturated = np.mean(np.abs(acc_data[i:i+window_size]) < 1.9)
            
            quality = (not_stuck * 0.3 + reasonable * 0.4 + not_saturated * 0.3) * 100
            quality_scores.append(quality)
        
        quality_array = np.array(quality_scores)
        mean_quality = np.mean(quality_array) if len(quality_array) > 0 else 0.0
        
        return mean_quality, quality_array


class DataQualityFramework:
    """Main framework for comprehensive data quality assessment"""
    
    def __init__(self):
        self.completeness_assessor = DataCompletenessAssessor()
        self.on_body_detector = OnBodyDetector()
        self.signal_evaluator = SignalQualityEvaluator()
    
    def assess_quality(self, data_dict: Dict, sensor_type: str) -> QualityMetrics:
        """
        Perform comprehensive quality assessment for a sensor modality
        
        Args:
            data_dict: Dictionary containing sensor data and metadata
            sensor_type: Type of sensor ('acc', 'bvp', 'eda', 'temp', 'hr')
            
        Returns:
            QualityMetrics object with all assessment results
        """
        # Extract common elements
        timestamps = data_dict['timestamps']
        sample_rate = data_dict['sample_rate']
        
        # 1. Assess data completeness
        completeness, gaps_df = self.completeness_assessor.assess(timestamps, sample_rate)
        
        # 2. Detect on-body periods (sensor-specific)
        on_body_percentage = 0.0
        
        if sensor_type == 'acc':
            acc_xyz = data_dict['data']
            on_body_mask = self.on_body_detector.detect_from_acc(acc_xyz)
            on_body_percentage = np.mean(on_body_mask) * 100
            
        elif sensor_type == 'temp':
            temp_values = data_dict['data']
            on_body_mask = self.on_body_detector.detect_from_temp(temp_values)
            on_body_percentage = np.mean(on_body_mask) * 100
            
        elif sensor_type == 'eda':
            eda_values = data_dict['data']
            on_body_mask = self.on_body_detector.detect_from_eda(eda_values)
            on_body_percentage = np.mean(on_body_mask) * 100
        
        # 3. Evaluate signal quality (sensor-specific)
        signal_quality = 0.0
        quality_timeline = np.array([])
        
        if sensor_type == 'bvp':
            bvp_values = data_dict['data']
            signal_quality, quality_timeline = self.signal_evaluator.evaluate_bvp_quality(
                bvp_values, sample_rate
            )
            
        elif sensor_type == 'eda':
            eda_values = data_dict['data']
            signal_quality, quality_timeline = self.signal_evaluator.evaluate_eda_quality(
                eda_values, sample_rate
            )
            
        elif sensor_type == 'temp':
            temp_values = data_dict['data']
            signal_quality, quality_timeline = self.signal_evaluator.evaluate_temp_quality(
                temp_values, sample_rate
            )
            
        elif sensor_type == 'acc':
            acc_xyz = data_dict['data']
            signal_quality, quality_timeline = self.signal_evaluator.evaluate_acc_quality(
                acc_xyz, sample_rate
            )
        
        # 4. Calculate aggregate reliability score
        # Weighted average: completeness (30%), on-body (30%), signal quality (40%)
        aggregate = (completeness * 0.3 + on_body_percentage * 0.3 + signal_quality * 0.4)
        
        # 5. Create quality timeline dataframe
        quality_df = pd.DataFrame({
            'quality_score': quality_timeline
        }) if len(quality_timeline) > 0 else pd.DataFrame()
        
        # 6. Compile metadata
        metadata = {
            'sensor_type': sensor_type,
            'sample_rate': sample_rate,
            'total_samples': len(timestamps),
            'duration_seconds': timestamps[-1] - timestamps[0] if len(timestamps) > 0 else 0,
            'gaps_detected': len(gaps_df),
            'gaps_info': gaps_df.to_dict('records') if len(gaps_df) > 0 else []
        }
        
        return QualityMetrics(
            completeness=completeness,
            on_body_detection=on_body_percentage,
            signal_quality=signal_quality,
            aggregate_score=aggregate,
            quality_timeline=quality_df,
            metadata=metadata
        )
    
    def assess_multi_modal(self, acc_dict: Optional[Dict] = None,
                          bvp_dict: Optional[Dict] = None,
                          eda_dict: Optional[Dict] = None,
                          temp_dict: Optional[Dict] = None) -> Dict[str, QualityMetrics]:
        """
        Assess quality across multiple sensor modalities
        
        Returns:
            Dictionary mapping sensor type to QualityMetrics
        """
        results = {}
        
        if acc_dict:
            results['acc'] = self.assess_quality(acc_dict, 'acc')
        if bvp_dict:
            results['bvp'] = self.assess_quality(bvp_dict, 'bvp')
        if eda_dict:
            results['eda'] = self.assess_quality(eda_dict, 'eda')
        if temp_dict:
            results['temp'] = self.assess_quality(temp_dict, 'temp')
        
        return results
    
    def generate_summary_report(self, quality_results: Dict[str, QualityMetrics]) -> pd.DataFrame:
        """
        Generate summary report of quality metrics across all sensors
        
        Returns:
            DataFrame with summary statistics
        """
        summary_data = []
        
        for sensor_type, metrics in quality_results.items():
            summary_data.append({
                'Sensor': sensor_type.upper(),
                'Completeness (%)': f"{metrics.completeness:.1f}",
                'On-Body (%)': f"{metrics.on_body_detection:.1f}",
                'Signal Quality (%)': f"{metrics.signal_quality:.1f}",
                'Aggregate Score (%)': f"{metrics.aggregate_score:.1f}",
                'Duration (min)': f"{metrics.metadata['duration_seconds']/60:.1f}",
                'Sample Rate (Hz)': metrics.metadata['sample_rate'],
                'Total Samples': metrics.metadata['total_samples'],
                'Gaps Detected': metrics.metadata['gaps_detected']
            })
        
        return pd.DataFrame(summary_data)
