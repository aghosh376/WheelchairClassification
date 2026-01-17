"""
Feature extraction from EMG signals
Time-domain, frequency-domain, and time-frequency features
"""

import numpy as np
from scipy import stats, signal
from scipy.fft import fft, fftfreq

from config import FEATURE_CONFIG

class FeatureExtractor:
    """Extract features from EMG signals"""
    
    def __init__(self, sampling_rate=200, config=None):
        self.sampling_rate = sampling_rate
        self.config = config or FEATURE_CONFIG
    
    def extract_all(self, signal_data, feature_set=None):
        """
        Extract all configured features
        Args:
            signal_data: Raw or preprocessed EMG signal
            feature_set: List of feature names, or 'all' for all features
        Returns:
            Dictionary of feature names and values
        """
        if feature_set is None:
            feature_set = self.config['time_domain'] + self.config['frequency_domain']
        elif feature_set == 'all':
            feature_set = self.get_available_features()
        
        features = {}
        
        for feature_name in feature_set:
            try:
                if feature_name in self.get_time_domain_features():
                    features[feature_name] = self.extract_time_domain(signal_data, feature_name)
                elif feature_name in self.get_frequency_domain_features():
                    features[feature_name] = self.extract_frequency_domain(signal_data, feature_name)
                else:
                    print(f"Warning: Unknown feature '{feature_name}'")
            except Exception as e:
                print(f"Error extracting {feature_name}: {e}")
                features[feature_name] = 0.0
        
        return features
    
    def extract_time_domain(self, signal_data, feature_name):
        """Extract time-domain features"""
        if feature_name == 'rms':
            return np.sqrt(np.mean(signal_data ** 2))
        elif feature_name == 'mav':
            return np.mean(np.abs(signal_data))
        elif feature_name == 'wl':
            return np.sum(np.abs(np.diff(signal_data)))
        elif feature_name == 'zc':
            zero_crossings = np.where(np.diff(np.signbit(signal_data)))[0]
            return len(zero_crossings)
        elif feature_name == 'ssc':
            diff_signal = np.diff(signal_data)
            ssc = np.sum((diff_signal[:-1] * diff_signal[1:]) < 0)
            return ssc
        elif feature_name == 'var':
            return np.var(signal_data)
        elif feature_name == 'skew':
            return stats.skew(signal_data)
        elif feature_name == 'kurt':
            return stats.kurtosis(signal_data)
        else:
            raise ValueError(f"Unknown time-domain feature: {feature_name}")
    
    def extract_frequency_domain(self, signal_data, feature_name):
        """Extract frequency-domain features"""
        # Compute PSD
        freqs, psd = signal.welch(
            signal_data, 
            fs=self.sampling_rate,
            nperseg=min(len(signal_data), 256)
        )
        
        if feature_name == 'mdf':
            # Median frequency
            cumulative_power = np.cumsum(psd)
            total_power = cumulative_power[-1]
            return freqs[np.where(cumulative_power >= total_power / 2)[0][0]]
        
        elif feature_name == 'mnf':
            # Mean frequency
            return np.sum(freqs * psd) / np.sum(psd)
        
        elif feature_name == 'psd_ratio':
            # Ratio of low to high frequency power
            low_band = (freqs >= 20) & (freqs <= 60)
            high_band = (freqs >= 60) & (freqs <= 150)
            low_power = np.sum(psd[low_band])
            high_power = np.sum(psd[high_band])
            return low_power / (high_power + 1e-8)
        
        elif feature_name == 'spectral_entropy':
            # Spectral entropy
            psd_norm = psd / np.sum(psd)
            return -np.sum(psd_norm * np.log(psd_norm + 1e-8))
        
        else:
            raise ValueError(f"Unknown frequency-domain feature: {feature_name}")
    
    def extract_windowed_features(self, signal_data, window_size, overlap):
        """Extract features from sliding windows"""
        step_size = window_size - overlap
        features_list = []
        
        for start in range(0, len(signal_data) - window_size + 1, step_size):
            window = signal_data[start:start + window_size]
            features = self.extract_all(window)
            features_list.append(features)
        
        return np.array([list(f.values()) for f in features_list])
    
    def get_available_features(self):
        """Get list of all available feature names"""
        time_features = self.get_time_domain_features()
        freq_features = self.get_frequency_domain_features()
        return time_features + freq_features
    
    def get_time_domain_features(self):
        """Get available time-domain features"""
        return ['rms', 'mav', 'wl', 'zc', 'ssc', 'var', 'skew', 'kurt']
    
    def get_frequency_domain_features(self):
        """Get available frequency-domain features"""
        return ['mdf', 'mnf', 'psd_ratio', 'spectral_entropy']


class MultiChannelFeatureExtractor:
    """Extract features from multiple EMG channels"""
    
    def __init__(self, num_channels, sampling_rate=200):
        self.num_channels = num_channels
        self.extractors = [FeatureExtractor(sampling_rate) for _ in range(num_channels)]
    
    def extract_channel_features(self, data, feature_set=None):
        """Extract features from each channel separately"""
        # Ensure shape (channels, samples)
        if data.shape[0] != self.num_channels and data.shape[1] == self.num_channels:
            data = data.T
        
        all_features = []
        
        for i in range(self.num_channels):
            channel_features = self.extractors[i].extract_all(data[i], feature_set)
            all_features.append(channel_features)
        
        return all_features
    
    def extract_combined_features(self, data, feature_set=None):
        """Extract features and combine across channels"""
        channel_features = self.extract_channel_features(data, feature_set)
        
        # Combine into single feature vector
        combined = {}
        for i, features in enumerate(channel_features):
            for feat_name, feat_value in features.items():
                combined[f'ch{i}_{feat_name}'] = feat_value
        
        # Add cross-channel features
        if self.num_channels > 1:
            # Mean across channels
            for feat_name in channel_features[0].keys():
                values = [cf[feat_name] for cf in channel_features]
                combined[f'mean_{feat_name}'] = np.mean(values)
                combined[f'std_{feat_name}'] = np.std(values)
        
        return combined