"""
Utility functions for EMG pipeline
"""

import numpy as np
import json
import pickle
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt

def timestamp_to_str(timestamp=None):
    """Convert timestamp to readable string"""
    if timestamp is None:
        timestamp = time.time()
    return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

def save_json(data: Dict, filepath: str):
    """Save data as JSON file"""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)

def load_json(filepath: str) -> Dict:
    """Load data from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def save_pickle(data: Any, filepath: str):
    """Save data as pickle file"""
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(filepath: str) -> Any:
    """Load data from pickle file"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def calculate_snr(signal: np.ndarray, noise_floor: Optional[np.ndarray] = None) -> float:
    """Calculate Signal-to-Noise Ratio"""
    signal_power = np.mean(signal ** 2)
    
    if noise_floor is not None:
        noise_power = np.mean(noise_floor ** 2)
    else:
        # Estimate noise from high-frequency components
        from scipy import fftpack
        freq_signal = fftpack.fft(signal)
        power_spectrum = np.abs(freq_signal) ** 2
        # Assume high frequencies (above 200Hz) are noise
        noise_power = np.mean(power_spectrum[len(power_spectrum)//2:])
    
    if noise_power == 0:
        return float('inf')
    
    return 10 * np.log10(signal_power / noise_power)

def moving_average(data: np.ndarray, window_size: int = 3) -> np.ndarray:
    """Apply moving average filter"""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def normalize_array(data: np.ndarray, method: str = 'zscore') -> np.ndarray:
    """Normalize array using specified method"""
    if method == 'zscore':
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return data - mean
        return (data - mean) / std
    elif method == 'minmax':
        min_val = np.min(data)
        max_val = np.max(data)
        if max_val - min_val == 0:
            return np.zeros_like(data)
        return (data - min_val) / (max_val - min_val)
    elif method == 'unit_norm':
        norm = np.linalg.norm(data)
        if norm == 0:
            return data
        return data / norm
    else:
        return data

def plot_emg_comparison(raw_signal: np.ndarray, processed_signal: np.ndarray, 
                        sampling_rate: float = 200, title: str = "EMG Signal Comparison"):
    """Plot raw vs processed EMG signals"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    
    time_axis = np.arange(len(raw_signal)) / sampling_rate
    
    # Raw signal
    axes[0].plot(time_axis, raw_signal, color='blue', alpha=0.7, linewidth=0.5)
    axes[0].set_title(f'Raw EMG Signal (SNR: {calculate_snr(raw_signal):.2f} dB)')
    axes[0].set_ylabel('Amplitude (μV)')
    axes[0].grid(True, alpha=0.3)
    
    # Processed signal
    axes[1].plot(time_axis, processed_signal, color='green', alpha=0.7, linewidth=0.5)
    axes[1].set_title(f'Processed EMG Signal (SNR: {calculate_snr(processed_signal):.2f} dB)')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Amplitude')
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    return fig

def create_gesture_report(predictions: List[str], confidences: List[float], 
                          true_labels: Optional[List[str]] = None) -> Dict:
    """Create a report of gesture classification performance"""
    unique_gestures = list(set(predictions))
    
    report = {
        'total_predictions': len(predictions),
        'gesture_distribution': {},
        'average_confidence': np.mean(confidences) if confidences else 0,
        'timestamp': timestamp_to_str()
    }
    
    # Count predictions per gesture
    for gesture in unique_gestures:
        count = predictions.count(gesture)
        report['gesture_distribution'][gesture] = {
            'count': count,
            'percentage': 100 * count / len(predictions),
            'avg_confidence': np.mean([c for p, c in zip(predictions, confidences) if p == gesture])
        }
    
    # Calculate accuracy if true labels provided
    if true_labels and len(true_labels) == len(predictions):
        correct = sum([1 for p, t in zip(predictions, true_labels) if p == t])
        report['accuracy'] = 100 * correct / len(predictions)
        
        # Confusion matrix
        confusion = {}
        for pred, true in zip(predictions, true_labels):
            if true not in confusion:
                confusion[true] = {}
            if pred not in confusion[true]:
                confusion[true][pred] = 0
            confusion[true][pred] += 1
        
        report['confusion_matrix'] = confusion
    
    return report

def print_pipeline_status(pipeline_info: Dict):
    """Print pipeline status in a formatted way"""
    print("\n" + "="*50)
    print("EMG PIPELINE STATUS")
    print("="*50)
    print(f"Mode: {pipeline_info['mode']}")
    print(f"Board: {pipeline_info['board_type']}")
    print(f"Running: {pipeline_info['is_running']}")
    print(f"Components: {', '.join(pipeline_info['components'])}")
    print(f"Data buffer: {pipeline_info['buffer_size']} frames")
    
    if 'classifier_performance' in pipeline_info:
        perf = pipeline_info['classifier_performance']
        print(f"\nClassifier Performance:")
        print(f"  Avg latency: {perf.get('avg_latency_ms', 0):.2f} ms")
        print(f"  Predictions: {perf.get('num_predictions', 0)}")
    print("="*50)

def setup_data_directories():
    """Create necessary directories for the pipeline"""
    import os
    
    directories = [
        'data/raw',
        'data/processed', 
        'data/features',
        'models',
        'calibration',
        'logs',
        'results'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    
    print("Data directories setup complete")