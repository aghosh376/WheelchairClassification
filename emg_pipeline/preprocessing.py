"""
Signal preprocessing module for EMG signals
Modular design with individual processing steps
"""

import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt
import warnings

from config import PREPROCESSING_CONFIG

class Preprocessor:
    """Modular EMG signal preprocessor"""
    
    def __init__(self, sampling_rate=200, config=None):
        self.sampling_rate = sampling_rate
        self.config = config or PREPROCESSING_CONFIG
        
        # Initialize filter coefficients
        self.init_filters()
        
        # State for adaptive filtering
        self.baseline_mean = None
        self.baseline_std = None
    
    def init_filters(self):
        """Initialize filter coefficients"""
        nyquist = self.sampling_rate / 2
        
        # Bandpass filter for EMG
        low = self.config['filter_bands']['emg'][0] / nyquist
        high = self.config['filter_bands']['emg'][1] / nyquist
        self.b_bp, self.a_bp = butter(4, [low, high], btype='band')
        
        # Notch filter for powerline (50/60 Hz)
        notch_freq = self.config['filter_bands']['notch']
        quality = 30
        self.b_notch, self.a_notch = signal.iirnotch(
            notch_freq / nyquist, quality
        )
        
        # Low-pass for envelope
        low_cutoff = 5 / nyquist
        self.b_lp, self.a_lp = butter(4, low_cutoff, btype='low')
    
    def apply_step(self, data, step_name, **kwargs):
        """
        Apply a single preprocessing step
        Available steps: bandpass, notch, rectify, envelope, normalize, detrend
        """
        if step_name == 'bandpass':
            return self.bandpass_filter(data)
        elif step_name == 'notch':
            return self.notch_filter(data)
        elif step_name == 'rectify':
            return self.full_wave_rectify(data)
        elif step_name == 'envelope':
            return self.extract_envelope(data)
        elif step_name == 'normalize':
            method = kwargs.get('method', self.config['normalization'])
            return self.normalize(data, method)
        elif step_name == 'detrend':
            return self.remove_baseline(data)
        elif step_name == 'segment':
            window_size = kwargs.get('window_size', self.config['window_size'])
            overlap = kwargs.get('overlap', self.config['overlap'])
            return self.segment_signal(data, window_size, overlap)
        else:
            warnings.warn(f"Unknown preprocessing step: {step_name}")
            return data
    
    def process_pipeline(self, data, steps=None):
        """
        Process data through a customizable pipeline
        Args:
            data: Input signal (numpy array)
            steps: List of step names to apply
        Returns:
            Processed signal
        """
        if steps is None:
            # Default pipeline
            steps = ['bandpass', 'notch', 'rectify', 'envelope', 'normalize']
        
        processed = data.copy()
        
        for step in steps:
            processed = self.apply_step(processed, step)
        
        return processed
    
    def bandpass_filter(self, data):
        """Apply bandpass filter (20-250 Hz for EMG)"""
        if len(data) < 10:
            return data
        return filtfilt(self.b_bp, self.a_bp, data)
    
    def notch_filter(self, data):
        """Apply notch filter to remove powerline noise"""
        if len(data) < 10:
            return data
        return filtfilt(self.b_notch, self.a_notch, data)
    
    def full_wave_rectify(self, data):
        """Full-wave rectification"""
        return np.abs(data)
    
    def extract_envelope(self, data):
        """Extract signal envelope using low-pass filter"""
        if len(data) < 10:
            return data
        return filtfilt(self.b_lp, self.a_lp, data)
    
    def normalize(self, data, method='zscore'):
        """Normalize signal"""
        if len(data) == 0:
            return data
        
        if method == 'zscore':
            mean_val = np.mean(data)
            std_val = np.std(data) + 1e-8  # Avoid division by zero
            return (data - mean_val) / std_val
        
        elif method == 'minmax':
            min_val = np.min(data)
            max_val = np.max(data)
            if max_val - min_val > 1e-8:
                return (data - min_val) / (max_val - min_val)
            return data
        
        elif method == 'baseline':
            if self.baseline_mean is not None and self.baseline_std is not None:
                return (data - self.baseline_mean) / self.baseline_std
            else:
                return self.normalize(data, 'zscore')
        
        else:  # 'none' or unknown
            return data
    
    def remove_baseline(self, data):
        """Remove baseline drift using high-pass filter"""
        if len(data) < 10:
            return data
        
        # High-pass filter at 0.5 Hz
        nyquist = self.sampling_rate / 2
        b_hp, a_hp = butter(2, 0.5 / nyquist, btype='high')
        return filtfilt(b_hp, a_hp, data)
    
    def segment_signal(self, data, window_size, overlap):
        """Segment signal into overlapping windows"""
        step_size = window_size - overlap
        segments = []
        
        for start in range(0, len(data) - window_size + 1, step_size):
            segment = data[start:start + window_size]
            segments.append(segment)
        
        return np.array(segments)
    
    def calibrate_baseline(self, baseline_data):
        """Calibrate baseline from relaxed EMG data"""
        self.baseline_mean = np.mean(baseline_data)
        self.baseline_std = np.std(baseline_data) + 1e-8
    
    def reset_baseline(self):
        """Reset baseline calibration"""
        self.baseline_mean = None
        self.baseline_std = None


class MultiChannelPreprocessor:
    """Preprocessor for multi-channel EMG data"""
    
    def __init__(self, num_channels, sampling_rate=200):
        self.num_channels = num_channels
        self.preprocessors = [Preprocessor(sampling_rate) for _ in range(num_channels)]
    
    def process_channels(self, data, steps=None):
        """
        Process multi-channel data
        Args:
            data: Shape (channels, samples) or (samples, channels)
        Returns:
            Processed data in same shape
        """
        # Ensure shape (channels, samples)
        if data.shape[0] != self.num_channels and data.shape[1] == self.num_channels:
            data = data.T
        
        processed = []
        for i in range(self.num_channels):
            channel_data = self.preprocessors[i].process_pipeline(data[i], steps)
            processed.append(channel_data)
        
        return np.array(processed)