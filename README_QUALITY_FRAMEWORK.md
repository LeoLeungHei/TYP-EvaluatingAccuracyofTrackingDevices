# E4 Data Processing with Quality Assessment Framework

## Overview
The revamped E4 data processing system now returns comprehensive quality metrics alongside processed data, aligned with your third-year project requirements.

## Key Components

### 1. **data_quality_framework.py**
Core framework implementing quality assessment algorithms:

- **DataCompletenessAssessor**: Detects gaps and calculates % of expected samples
- **OnBodyDetector**: Multi-modal on-body detection using ACC, EDA, and TEMP
- **SignalQualityEvaluator**: Modality-specific signal quality assessment
  - BVP: Spectral entropy (60.2% typical reliability)
  - EDA: Rate of amplitude change (70.4% typical reliability)
  - TEMP: Rate of change + physiological range (96.1% typical reliability)
  - ACC: Movement artifact detection
- **DataQualityFramework**: Main class orchestrating all assessments

### 2. **process_e4_data.py**
Enhanced data loading that integrates quality assessment:

- Loads all E4 sensor modalities (ACC, BVP, EDA, TEMP, HR, IBI)
- Prepares data dictionaries for quality framework
- Performs comprehensive quality assessment
- Segments data by experimental conditions
- Generates quality reports

## Return Values

### From `process_subject()`
Returns tuple: `(data, quality_results)`

#### `data` dictionary contains:
- `'acc'`: DataFrame with accelerometer data (timestamp, datetime, acc_x, acc_y, acc_z)
- `'bvp'`: DataFrame with blood volume pulse data
- `'eda'`: DataFrame with electrodermal activity data
- `'temp'`: DataFrame with temperature data
- `'hr'`: DataFrame with heart rate data
- `'ibi'`: DataFrame with inter-beat intervals
- `'tags'`: DataFrame with event markers
- `'conditions'`: Dictionary with experimental condition timings

#### `quality_results` dictionary contains:
Maps sensor type → `QualityMetrics` object

### QualityMetrics Object
```python
@dataclass
class QualityMetrics:
    completeness: float              # 0-100%
    on_body_detection: float         # 0-100%
    signal_quality: float            # 0-100%
    aggregate_score: float           # 0-100% (weighted: 30/30/40)
    quality_timeline: pd.DataFrame   # Time-series of quality scores
    metadata: Dict                   # Sample rate, duration, gaps, etc.
```

## Quality Assessment Results (S2 Example)

### Overall Quality Summary:
- **ACC**: 74.5% (100% complete, 44% on-body, 78% signal quality)
- **BVP**: 44.5% (100% complete, 0% on-body*, 36% signal quality)
- **EDA**: 99.8% (100% complete, 100% on-body, 99% signal quality)
- **TEMP**: 99.3% (100% complete, 100% on-body, 98% signal quality)

*BVP on-body detection not implemented (requires different approach than ACC/EDA/TEMP)

### Per-Condition Quality (ACC):
- **Base** (baseline): 76.5%
- **TSST** (stress): 68.3%
- **Fun** (amusement): 83.5%
- **Medi 1**: 71.9%
- **Medi 2**: 60.8%


## References to Project Proposal
This implementation addresses:
- **Objective 1**: Data Quality Assessment Framework with completeness, on-body detection, and signal quality
- **Framework metrics**: All specified algorithms implemented (spectral entropy for BVP, rate of change for EDA/TEMP, threshold-based for ACC)
- **Aggregate scoring**: Weighted combination (30% completeness, 30% on-body, 40% signal quality)
- **Contextual metadata**: Duration, sample rates, gaps detected, condition segmentation

## Metric Calculations

# Signal Quality (0-100%)
Calculated differently per sensor type:

BVP: Uses spectral entropy of heart rate frequency range (0.5-4 Hz)

Lower entropy = better quality (more regular heartbeat)
quality = (1 - spectral_entropy / max_entropy) * 100
EDA: Based on rate of amplitude change

Measures percentage of changes ≤ 0.5 μS/second (gradual changes are normal)
Temperature: Combines two factors (50/50 weight):

Data within physiological range (30-40°C)
Slow rate of change (≤ 0.1°C/second)
Accelerometer: Combines three factors:

Not stuck (variance > 0.001): 30% weight
Reasonable values (0.1-3.0g): 40% weight
Not saturated (< 1.9g): 30% weight
# On-Body % (0-100%)
Percentage of time the device was detected on the body:

ACC: Movement variance > 0.01 threshold in 1-second rolling windows
Temperature: Within 30-40°C range
EDA: Above 0.05 μS
Multiple sensors use majority voting for combined detection.

# Aggregate Score (0-100%)
Weighted average of three components:

Where:

Completeness (30%): Actual samples / expected samples based on duration and sample rate
On-Body % (30%): Percentage from on-body detection
Signal Quality (40%): Sensor-specific quality evaluation
