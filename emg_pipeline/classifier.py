"""
Classification module - orchestrates models and prediction logic
"""

import numpy as np
import torch
from collections import deque
import time

from config import CLASSIFICATION_CONFIG
from models import create_model

class EMGClassifier:
    """Main classification orchestrator"""
    
    def __init__(self, model_type='hybrid', model_path=None, device=None):
        self.model_type = model_type
        self.gestures = CLASSIFICATION_CONFIG['gestures']
        self.num_classes = len(self.gestures)
        
        # Set device
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Create model
        if model_type != 'threshold':
            self.model = create_model(model_type, num_classes=self.num_classes)
            self.model.to(self.device)
            
            if model_path:
                self.load_model(model_path)
            
            self.model.eval()
        else:
            self.model = create_model('threshold')
        
        # Prediction smoothing
        self.prediction_history = deque(maxlen=10)
        self.confidence_history = deque(maxlen=10)
        
        # Performance tracking
        self.latencies = deque(maxlen=100)
    
    def load_model(self, path):
        """Load trained model weights"""
        try:
            state_dict = torch.load(path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"Model loaded from {path}")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def preprocess_for_model(self, signal_data):
        """Prepare signal data for model input"""
        if self.model_type == 'threshold':
            return signal_data  # Threshold model uses raw/processed signal
        
        # Convert to tensor
        if isinstance(signal_data, np.ndarray):
            signal_tensor = torch.FloatTensor(signal_data)
        else:
            signal_tensor = signal_data
        
        # Add batch dimension if needed
        if len(signal_tensor.shape) == 1:
            signal_tensor = signal_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len)
        elif len(signal_tensor.shape) == 2:
            # Assume (seq_len, features) or (features, seq_len)
            if signal_tensor.shape[0] < signal_tensor.shape[1]:
                signal_tensor = signal_tensor.T.unsqueeze(0)  # (1, features, seq_len)
            else:
                signal_tensor = signal_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len)
        
        return signal_tensor.to(self.device)
    
    def predict(self, signal_data, use_smoothing=True):
        """Make prediction on signal data"""
        start_time = time.time()
        
        if self.model_type == 'threshold':
            # Simple threshold classification
            prediction = self.model.predict(signal_data)
            confidence = 1.0 if prediction != 'Unknown' else 0.0
        else:
            # Deep learning model prediction
            with torch.no_grad():
                input_tensor = self.preprocess_for_model(signal_data)
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, pred_idx = torch.max(probabilities, dim=1)
                
                prediction = self.gestures[pred_idx.item()]
                confidence = confidence.item()
        
        # Track latency
        self.latencies.append(time.time() - start_time)
        
        # Update history for smoothing
        self.prediction_history.append(prediction)
        self.confidence_history.append(confidence)
        
        # Apply smoothing if requested
        if use_smoothing and len(self.prediction_history) >= 5:
            smoothed_prediction = self._apply_smoothing()
            smoothed_confidence = np.mean(list(self.confidence_history)[-5:])
            return smoothed_prediction, smoothed_confidence
        
        return prediction, confidence
    
    def _apply_smoothing(self):
        """Apply majority voting smoothing"""
        if len(self.prediction_history) == 0:
            return 'Unknown'
        
        # Find most common prediction in recent history
        unique, counts = np.unique(list(self.prediction_history), return_counts=True)
        return unique[np.argmax(counts)]
    
    def predict_batch(self, signal_batch):
        """Predict on batch of signals"""
        predictions = []
        confidences = []
        
        for signal in signal_batch:
            pred, conf = self.predict(signal, use_smoothing=False)
            predictions.append(pred)
            confidences.append(conf)
        
        return predictions, confidences
    
    def get_performance_stats(self):
        """Get classification performance statistics"""
        if len(self.latencies) == 0:
            return {}
        
        return {
            'avg_latency_ms': np.mean(self.latencies) * 1000,
            'max_latency_ms': np.max(self.latencies) * 1000,
            'min_latency_ms': np.min(self.latencies) * 1000,
            'num_predictions': len(self.latencies)
        }
    
    def reset_history(self):
        """Reset prediction history"""
        self.prediction_history.clear()
        self.confidence_history.clear()
        self.latencies.clear()


class MultiModelEnsemble:
    """Ensemble of multiple classifiers"""
    
    def __init__(self, models_config):
        """
        Args:
            models_config: List of dicts with 'type', 'path', 'weight'
        """
        self.classifiers = []
        self.weights = []
        
        for config in models_config:
            classifier = EMGClassifier(
                model_type=config['type'],
                model_path=config.get('path'),
                device=config.get('device')
            )
            self.classifiers.append(classifier)
            self.weights.append(config.get('weight', 1.0))
        
        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
    
    def predict(self, signal_data, voting='weighted'):
        """Make ensemble prediction"""
        predictions = []
        confidences = []
        
        for classifier in self.classifiers:
            pred, conf = classifier.predict(signal_data, use_smoothing=False)
            predictions.append(pred)
            confidences.append(conf)
        
        if voting == 'weighted':
            # Weighted voting based on confidence
            vote_dict = {}
            for pred, conf, weight in zip(predictions, confidences, self.weights):
                score = conf * weight
                if pred not in vote_dict:
                    vote_dict[pred] = 0
                vote_dict[pred] += score
            
            final_prediction = max(vote_dict.items(), key=lambda x: x[1])[0]
            final_confidence = vote_dict[final_prediction]
            
        elif voting == 'majority':
            # Simple majority voting
            from collections import Counter
            pred_counter = Counter(predictions)
            final_prediction = pred_counter.most_common(1)[0][0]
            final_confidence = pred_counter[final_prediction] / len(predictions)
        
        elif voting == 'confidence':
            # Use most confident classifier
            max_conf_idx = np.argmax(confidences)
            final_prediction = predictions[max_conf_idx]
            final_confidence = confidences[max_conf_idx]
        
        return final_prediction, final_confidence