import pandas as pd
import numpy as np
import os
from datetime import datetime
from data_quality_framework import DataQualityFramework, QualityMetrics
from typing import Dict, Tuple

def load_e4_csv(filepath):
    """
    Load E4 wristband CSV file.
    Returns: (initial_timestamp, sample_rate, data_df, timestamps_array)
    """
    with open(filepath, 'r') as f:
        # Read first two lines for metadata
        initial_timestamp = float(f.readline().strip().split(',')[0])
        sample_rate = float(f.readline().strip().split(',')[0])
    
    # Read the actual data (skip first 2 rows)
    data = pd.read_csv(filepath, skiprows=2, header=None)
    
    # Create timestamps for each sample
    num_samples = len(data)
    timestamps = initial_timestamp + np.arange(num_samples) / sample_rate
    data.insert(0, 'timestamp', timestamps)
    data.insert(1, 'datetime', pd.to_datetime(timestamps, unit='s'))
    
    return initial_timestamp, sample_rate, data, timestamps

def load_acc_data(filepath):
    """
    Load accelerometer data (3-axis: x, y, z)
    Returns: (dataframe, quality_dict)
    """
    initial_ts, sample_rate, data, timestamps = load_e4_csv(filepath)
    data.columns = ['timestamp', 'datetime', 'acc_x', 'acc_y', 'acc_z']
    
    # Convert from 1/64g to g
    data['acc_x'] = data['acc_x'] / 64.0
    data['acc_y'] = data['acc_y'] / 64.0
    data['acc_z'] = data['acc_z'] / 64.0
    
    # Prepare data for quality assessment
    acc_xyz = data[['acc_x', 'acc_y', 'acc_z']].values
    quality_dict = {
        'timestamps': timestamps,
        'sample_rate': sample_rate,
        'data': acc_xyz
    }
    
    print(f"ACC - Sample rate: {sample_rate} Hz, Samples: {len(data)}")
    return data, quality_dict

def load_bvp_data(filepath):
    """Load blood volume pulse data"""
    initial_ts, sample_rate, data, timestamps = load_e4_csv(filepath)
    data.columns = ['timestamp', 'datetime', 'bvp']
    
    # Prepare data for quality assessment
    bvp_values = data['bvp'].values
    quality_dict = {
        'timestamps': timestamps,
        'sample_rate': sample_rate,
        'data': bvp_values
    }
    
    print(f"BVP - Sample rate: {sample_rate} Hz, Samples: {len(data)}")
    return data, quality_dict

def load_eda_data(filepath):
    """Load electrodermal activity data (μS)"""
    initial_ts, sample_rate, data, timestamps = load_e4_csv(filepath)
    data.columns = ['timestamp', 'datetime', 'eda']
    
    # Prepare data for quality assessment
    eda_values = data['eda'].values
    quality_dict = {
        'timestamps': timestamps,
        'sample_rate': sample_rate,
        'data': eda_values
    }
    
    print(f"EDA - Sample rate: {sample_rate} Hz, Samples: {len(data)}")
    return data, quality_dict

def load_temp_data(filepath):
    """Load temperature data (°C)"""
    initial_ts, sample_rate, data, timestamps = load_e4_csv(filepath)
    data.columns = ['timestamp', 'datetime', 'temp']
    
    # Prepare data for quality assessment
    temp_values = data['temp'].values
    quality_dict = {
        'timestamps': timestamps,
        'sample_rate': sample_rate,
        'data': temp_values
    }
    
    print(f"TEMP - Sample rate: {sample_rate} Hz, Samples: {len(data)}")
    return data, quality_dict

def load_hr_data(filepath):
    """Load heart rate data"""
    initial_ts, sample_rate, data, timestamps = load_e4_csv(filepath)
    data.columns = ['timestamp', 'datetime', 'hr']
    print(f"HR - Sample rate: {sample_rate} Hz, Samples: {len(data)}")
    return data  # HR derived from BVP, quality assessed via BVP

def load_ibi_data(filepath):
    """Load inter-beat interval data"""
    data = pd.read_csv(filepath, skiprows=1, header=None)
    data.columns = ['time_from_start', 'ibi']
    print(f"IBI - Samples: {len(data)}")
    return data

def load_tags_data(filepath):
    """Load event tags (button presses)"""
    try:
        data = pd.read_csv(filepath, header=None)
        if data.empty or len(data.columns) == 0:
            print(f"TAGS - No events found (empty file)")
            return pd.DataFrame(columns=['timestamp', 'datetime'])
        data.columns = ['timestamp']
        data['datetime'] = pd.to_datetime(data['timestamp'], unit='s')
        print(f"TAGS - Events: {len(data)}")
        return data
    except pd.errors.EmptyDataError:
        print(f"TAGS - No events found (empty file)")
        return pd.DataFrame(columns=['timestamp', 'datetime'])

def load_quest_data(filepath):
    """Load questionnaire data and extract timing information"""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Extract timing info (START and END rows)
    conditions = {}
    for line in lines:
        if line.startswith('# ORDER'):
            parts = line.strip().split(';')
            condition_names = [p for p in parts[1:] if p]  # Remove empty strings
        elif line.startswith('# START'):
            parts = line.strip().split(';')
            start_times = [float(x) for x in parts[1:] if x]  # Remove empty strings
        elif line.startswith('# END'):
            parts = line.strip().split(';')
            end_times = [float(x) for x in parts[1:] if x]  # Remove empty strings
    
    # Create timing dictionary
    for i, name in enumerate(condition_names):
        conditions[name] = {
            'start': start_times[i] * 60,  # Convert to seconds
            'end': end_times[i] * 60
        }
    
    return conditions

def process_subject(subject_folder):
    """
    Process all E4 data for a subject with quality assessment
    Returns: (data_dict, quality_results)
    """
    e4_folder = os.path.join(subject_folder, f"{os.path.basename(subject_folder)}_E4_Data")
    
    if not os.path.exists(e4_folder):
        print(f"E4 data folder not found: {e4_folder}")
        return None, None
    
    print(f"\nProcessing: {os.path.basename(subject_folder)}")
    print("=" * 50)
    
    # Load all sensor data
    data = {}
    quality_inputs = {}
    
    acc_file = os.path.join(e4_folder, 'ACC.csv')
    if os.path.exists(acc_file):
        data['acc'], quality_inputs['acc'] = load_acc_data(acc_file)
    
    bvp_file = os.path.join(e4_folder, 'BVP.csv')
    if os.path.exists(bvp_file):
        data['bvp'], quality_inputs['bvp'] = load_bvp_data(bvp_file)
    
    eda_file = os.path.join(e4_folder, 'EDA.csv')
    if os.path.exists(eda_file):
        data['eda'], quality_inputs['eda'] = load_eda_data(eda_file)
    
    temp_file = os.path.join(e4_folder, 'TEMP.csv')
    if os.path.exists(temp_file):
        data['temp'], quality_inputs['temp'] = load_temp_data(temp_file)
    
    hr_file = os.path.join(e4_folder, 'HR.csv')
    if os.path.exists(hr_file):
        data['hr'] = load_hr_data(hr_file)
    
    ibi_file = os.path.join(e4_folder, 'IBI.csv')
    if os.path.exists(ibi_file):
        data['ibi'] = load_ibi_data(ibi_file)
    
    tags_file = os.path.join(e4_folder, 'tags.csv')
    if os.path.exists(tags_file):
        data['tags'] = load_tags_data(tags_file)
    
    # Load questionnaire data for condition timing
    quest_file = os.path.join(subject_folder, f"{os.path.basename(subject_folder)}_quest.csv")
    if os.path.exists(quest_file):
        data['conditions'] = load_quest_data(quest_file)
        print(f"\nConditions and timing:")
        for cond, times in data['conditions'].items():
            print(f"  {cond}: {times['start']:.1f}s - {times['end']:.1f}s")
    
    # Perform quality assessment
    print(f"\n{'='*50}")
    print("Performing Data Quality Assessment...")
    print("=" * 50)
    
    quality_framework = DataQualityFramework()
    quality_results = quality_framework.assess_multi_modal(
        acc_dict=quality_inputs.get('acc'),
        bvp_dict=quality_inputs.get('bvp'),
        eda_dict=quality_inputs.get('eda'),
        temp_dict=quality_inputs.get('temp')
    )
    
    # Generate and display summary report
    if quality_results:
        summary_df = quality_framework.generate_summary_report(quality_results)
        print("\nQuality Assessment Summary:")
        print(summary_df.to_string(index=False))
    
    return data, quality_results

def segment_by_condition(data_df, conditions, initial_timestamp):
    """Segment data by experimental conditions"""
    segments = {}
    
    for cond_name, times in conditions.items():
        # Calculate absolute timestamps for condition
        start_ts = initial_timestamp + times['start']
        end_ts = initial_timestamp + times['end']
        
        # Filter data within this time range
        mask = (data_df['timestamp'] >= start_ts) & (data_df['timestamp'] <= end_ts)
        segments[cond_name] = data_df[mask].copy()
        
    return segments

# Example usage
if __name__ == "__main__":
    # Process S2 subject
    subject_folder = r"c:\Users\Leo\Desktop\Coding stuff\TYP\WESAD\S2"
    data, quality_results = process_subject(subject_folder)
    
    if data and quality_results:
        print(f"\n{'='*50}")
        print("Data loaded successfully!")
        print(f"\nAvailable data streams: {list(data.keys())}")
        
        # Display detailed quality metrics
        print(f"\n{'='*50}")
        print("DETAILED QUALITY METRICS")
        print("=" * 50)
        
        for sensor_type, metrics in quality_results.items():
            print(f"\n{sensor_type.upper()} Quality Breakdown:")
            print(f"  Completeness:      {metrics.completeness:.2f}%")
            print(f"  On-Body Detection: {metrics.on_body_detection:.2f}%")
            print(f"  Signal Quality:    {metrics.signal_quality:.2f}%")
            print(f"  ─────────────────────────────")
            print(f"  AGGREGATE SCORE:   {metrics.aggregate_score:.2f}%")
            print(f"  Duration:          {metrics.metadata['duration_seconds']/60:.1f} minutes")
            print(f"  Gaps Detected:     {metrics.metadata['gaps_detected']}")
        
        # Example: Show first few rows of ACC data
        if 'acc' in data:
            print(f"\n{'='*50}")
            print("ACC Data Sample (first 5 rows):")
            print(data['acc'].head())
        
        # Example: Segment ACC data by conditions and assess quality per condition
        if 'acc' in data and 'conditions' in data:
            print(f"\n{'='*50}")
            print("Quality Assessment by Experimental Condition:")
            print("=" * 50)
            
            initial_ts = data['acc']['timestamp'].iloc[0]
            acc_segments = segment_by_condition(data['acc'], data['conditions'], initial_ts)
            
            # Create quality framework for per-condition assessment
            qf = DataQualityFramework()
            
            for cond, segment_df in acc_segments.items():
                if len(segment_df) > 0:
                    # Prepare data for quality assessment
                    acc_xyz = segment_df[['acc_x', 'acc_y', 'acc_z']].values
                    timestamps = segment_df['timestamp'].values
                    
                    cond_quality_dict = {
                        'timestamps': timestamps,
                        'sample_rate': 32.0,  # ACC sample rate
                        'data': acc_xyz
                    }
                    
                    cond_metrics = qf.assess_quality(cond_quality_dict, 'acc')
                    
                    print(f"\n{cond}:")
                    print(f"  Samples: {len(segment_df)} ({len(segment_df)/32:.1f}s)")
                    print(f"  Aggregate Quality: {cond_metrics.aggregate_score:.1f}%")
                    print(f"    - Completeness:  {cond_metrics.completeness:.1f}%")
                    print(f"    - On-Body:       {cond_metrics.on_body_detection:.1f}%")
                    print(f"    - Signal Quality: {cond_metrics.signal_quality:.1f}%")
