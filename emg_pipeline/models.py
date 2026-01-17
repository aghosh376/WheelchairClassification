"""
PyTorch model architectures for EMG classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CLASSIFICATION_CONFIG

class EMGLSTM(nn.Module):
    """LSTM model for EMG sequence classification"""
    
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, num_classes=5, dropout=0.3):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x, use_attention=True):
        # x shape: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*2)
        
        if use_attention:
            # Attention mechanism
            attention_weights = torch.softmax(self.attention(lstm_out).squeeze(-1), dim=1)
            context = torch.sum(lstm_out * attention_weights.unsqueeze(-1), dim=1)
        else:
            # Use last time step
            context = lstm_out[:, -1, :]
        
        return self.classifier(context)


class EMGCNN(nn.Module):
    """1D CNN for EMG classification"""
    
    def __init__(self, input_channels=1, num_classes=5):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            # Block 1
            nn.Conv1d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
            
            # Block 2
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
            
            # Block 3
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(10)  # Fixed output length
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128 * 10, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch, channels, seq_len)
        features = self.conv_layers(x)
        features = features.view(features.size(0), -1)  # Flatten
        return self.classifier(features)


class HybridModel(nn.Module):
    """CNN-LSTM hybrid model"""
    
    def __init__(self, input_channels=1, num_classes=5):
        super().__init__()
        
        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        
        # LSTM for temporal dependencies
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),  # 64*2 for bidirectional
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch, channels, seq_len)
        cnn_features = self.cnn(x)  # (batch, 128, seq_len/4)
        cnn_features = cnn_features.permute(0, 2, 1)  # (batch, seq_len/4, 128)
        
        lstm_out, _ = self.lstm(cnn_features)
        context = lstm_out[:, -1, :]  # Last time step
        
        return self.classifier(context)


class SimpleThresholdClassifier:
    """Simple threshold-based classifier for fallback"""
    
    def __init__(self, thresholds=None):
        self.thresholds = thresholds or {}
        self.gestures = CLASSIFICATION_CONFIG['gestures']
    
    def fit(self, calibration_data):
        """Fit thresholds from calibration data"""
        for gesture, data in calibration_data.items():
            if len(data) > 0:
                rms_values = np.sqrt(np.mean(data**2, axis=1))
                self.thresholds[gesture] = {
                    'mean': np.mean(rms_values),
                    'std': np.std(rms_values),
                    'threshold': np.mean(rms_values) + np.std(rms_values)
                }
    
    def predict(self, signal):
        """Predict using threshold rules"""
        if len(signal) == 0:
            return 'Unknown'
        
        rms = np.sqrt(np.mean(signal**2))
        
        # Check thresholds in priority order
        for gesture in ['Flex', 'Pinch', 'Extension', 'Grasp']:
            if gesture in self.thresholds and rms > self.thresholds[gesture]['threshold']:
                return gesture
        
        return 'Relax'


def create_model(model_type='hybrid', **kwargs):
    """Factory function to create models"""
    num_classes = kwargs.get('num_classes', CLASSIFICATION_CONFIG['num_classes'])
    
    if model_type == 'lstm':
        return EMGLSTM(
            input_size=kwargs.get('input_size', 1),
            num_classes=num_classes,
            hidden_size=kwargs.get('hidden_size', 128)
        )
    elif model_type == 'cnn':
        return EMGCNN(
            input_channels=kwargs.get('input_channels', 1),
            num_classes=num_classes
        )
    elif model_type == 'hybrid':
        return HybridModel(
            input_channels=kwargs.get('input_channels', 1),
            num_classes=num_classes
        )
    elif model_type == 'threshold':
        return SimpleThresholdClassifier()
    else:
        raise ValueError(f"Unknown model type: {model_type}")