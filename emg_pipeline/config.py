"""
Configuration and constants for EMG pipeline
"""

import numpy as np

# ============ DATA ACQUISITION ============
BOARD_CONFIG = {
    'GANGLION': {
        'board_id': 1,
        'sampling_rate': 200,
        'num_channels': 4,
        'emg_channels': [0, 1, 2, 3],
        'accel_channels': [4, 5, 6],
        'default_port': 'COM3'
    },
    'CYTON': {
        'board_id': 0,
        'sampling_rate': 250,
        'num_channels': 8,
        'emg_channels': list(range(8)),
        'default_port': 'COM4'
    }
}

# ============ SIGNAL PROCESSING ============
PREPROCESSING_CONFIG = {
    'filter_bands': {
        'emg': [20, 250],  # Hz
        'notch': 50,       # Hz (for powerline noise)
    },
    'window_size': 150,    # samples (0.75s at 200Hz)
    'overlap': 75,         # 50% overlap
    'normalization': 'zscore'  # 'zscore', 'minmax', 'none'
}

# ============ FEATURE EXTRACTION ============
FEATURE_CONFIG = {
    'time_domain': ['rms', 'mav', 'wl', 'zc', 'ssc'],
    'frequency_domain': ['mdf', 'mnf', 'psd'],
    'window_features': True,
    'segment_length': 100  # samples per segment for windowed features
}

# ============ CLASSIFICATION ============
CLASSIFICATION_CONFIG = {
    'gestures': ['Relax', 'Flex', 'Pinch', 'Extension', 'Grasp'],
    'model_types': ['lstm', 'cnn', 'hybrid', 'svm', 'threshold'],
    'default_model': 'hybrid',
    'sequence_length': 150,
    'num_classes': 5
}

# ============ TRAINING ============
TRAINING_CONFIG = {
    'batch_size': 32,
    'learning_rate': 0.001,
    'epochs': 100,
    'validation_split': 0.2,
    'early_stopping_patience': 10,
    'model_save_path': 'models/'
}

# ============ VISUALIZATION ============
VISUALIZATION_CONFIG = {
    'update_interval': 0.1,  # seconds
    'plot_duration': 5,      # seconds of data to display
    'show_features': True,
    'show_predictions': True
}

# ============ PIPELINE MODES ============
PIPELINE_MODES = {
    'RAW': 'raw',           # Just acquire data
    'PREPROCESSED': 'preprocessed',  # Acquire + preprocess
    'FEATURES': 'features', # Acquire + preprocess + extract features
    'CLASSIFY': 'classify', # Full pipeline to classification
    'TRAIN': 'train'        # Training mode
}

# ============ DATA PATHS ============
DATA_PATHS = {
    'raw_data': 'data/raw/',
    'processed_data': 'data/processed/',
    'features': 'data/features/',
    'models': 'models/',
    'calibration': 'calibration/',
    'logs': 'logs/'
}