"""
Training module for EMG models
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pickle
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from config import TRAINING_CONFIG, DATA_PATHS
from models import create_model

class EMGDataset(Dataset):
    """PyTorch Dataset for EMG data"""
    
    def __init__(self, data_path, transform=None, sequence_length=None):
        """
        Load data from:
        - .npz file with 'data' and 'labels' arrays
        - Pickle file with dict
        """
        if data_path.endswith('.npz'):
            data = np.load(data_path)
            self.signals = data['data']
            self.labels = data['labels']
            if 'gestures' in data:
                self.gesture_names = data['gestures']
            else:
                self.gesture_names = [f'Gesture_{i}' for i in range(np.max(self.labels)+1)]
        elif data_path.endswith('.pkl'):
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
            self.signals = data['data']
            self.labels = data['labels']
            self.gesture_names = data.get('gestures', [f'Gesture_{i}' for i in range(np.max(self.labels)+1)])
        else:
            raise ValueError("Unsupported file format. Use .npz or .pkl")
        
        # Reshape if needed (for LSTM/CNN)
        if len(self.signals.shape) == 2:
            # Assuming (samples, seq_len) -> add channel dimension
            self.signals = self.signals.reshape(-1, 1, self.signals.shape[1])
        elif len(self.signals.shape) == 3 and self.signals.shape[1] != 1:
            # Assuming (samples, seq_len, channels) -> permute to (samples, channels, seq_len) for CNN
            self.signals = np.transpose(self.signals, (0, 2, 1))
        
        self.transform = transform
        self.sequence_length = sequence_length
    
    def __len__(self):
        return len(self.signals)
    
    def __getitem__(self, idx):
        signal = self.signals[idx]
        label = self.labels[idx]
        
        if self.transform:
            signal = self.transform(signal)
        
        # Ensure sequence length
        if self.sequence_length and signal.shape[-1] > self.sequence_length:
            signal = signal[..., :self.sequence_length]
        
        return torch.FloatTensor(signal), torch.LongTensor([label]).squeeze()

class DataAugmentation:
    """Data augmentation for EMG signals"""
    
    @staticmethod
    def add_noise(signal, noise_level=0.05):
        """Add Gaussian noise"""
        noise = np.random.normal(0, noise_level * np.std(signal), signal.shape)
        return signal + noise
    
    @staticmethod
    def time_shift(signal, max_shift=0.1):
        """Random time shift"""
        shift = np.random.randint(0, int(max_shift * len(signal)))
        return np.roll(signal, shift)
    
    @staticmethod
    def amplitude_scale(signal, scale_range=(0.8, 1.2)):
        """Random amplitude scaling"""
        scale = np.random.uniform(*scale_range)
        return signal * scale
    
    @staticmethod
    def apply_augmentation(signal, augmentations=None):
        """Apply random augmentations"""
        if augmentations is None:
            augmentations = ['noise', 'shift', 'scale']
        
        augmented = signal.copy()
        
        if 'noise' in augmentations and np.random.random() > 0.5:
            augmented = DataAugmentation.add_noise(augmented)
        
        if 'shift' in augmentations and np.random.random() > 0.5:
            augmented = DataAugmentation.time_shift(augmented)
        
        if 'scale' in augmentations and np.random.random() > 0.5:
            augmented = DataAugmentation.amplitude_scale(augmented)
        
        return augmented

class ModelTrainer:
    """Train and evaluate EMG models"""
    
    def __init__(self, model_type='hybrid', config=None):
        self.model_type = model_type
        self.config = config or TRAINING_CONFIG
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rate': []
        }
    
    def prepare_data(self, data_path, test_size=0.2, augment=False):
        """Prepare train/validation splits"""
        dataset = EMGDataset(data_path)
        
        # Split indices
        indices = list(range(len(dataset)))
        train_idx, val_idx = train_test_split(
            indices, test_size=test_size, random_state=42, stratify=dataset.labels
        )
        
        # Create datasets
        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        val_dataset = torch.utils.data.Subset(dataset, val_idx)
        
        # Apply augmentation to training set if requested
        if augment:
            class AugmentedDataset(Dataset):
                def __init__(self, subset):
                    self.subset = subset
                
                def __len__(self):
                    return len(self.subset)
                
                def __getitem__(self, idx):
                    signal, label = self.subset[idx]
                    signal_np = signal.numpy()
                    augmented = DataAugmentation.apply_augmentation(signal_np)
                    return torch.FloatTensor(augmented), label
            
            train_dataset = AugmentedDataset(train_dataset)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=0
        )
        
        return train_loader, val_loader, dataset.gesture_names
    
    def train(self, train_loader, val_loader, num_classes, input_shape=None):
        """Train the model"""
        # Create model
        if input_shape is None:
            # Infer input shape from first batch
            sample_batch, _ = next(iter(train_loader))
            input_shape = sample_batch.shape[1:]  # (channels, seq_len)
        
        model = create_model(
            model_type=self.model_type,
            input_channels=input_shape[0],
            num_classes=num_classes
        )
        model.to(self.device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config['learning_rate']
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=self.config['early_stopping_patience'] // 2,
            factor=0.5,
            verbose=True
        )
        
        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"Training {self.model_type.upper()} model on {self.device}")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        
        for epoch in range(self.config['epochs']):
            # Training phase
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += targets.size(0)
                train_correct += predicted.eq(targets).sum().item()
            
            train_acc = 100. * train_correct / train_total
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation phase
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()
            
            val_acc = 100. * val_correct / val_total
            avg_val_loss = val_loss / len(val_loader)
            
            # Update learning rate
            current_lr = optimizer.param_groups[0]['lr']
            scheduler.step(avg_val_loss)
            
            # Store history
            self.history['train_loss'].append(avg_train_loss)
            self.history['val_loss'].append(avg_val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(current_lr)
            
            # Print progress
            print(f'Epoch {epoch+1:3d}/{self.config["epochs"]}: '
                  f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
                  f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}% | '
                  f'LR: {current_lr:.6f}')
            
            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                
                # Save best model
                model_path = f'{self.config["model_save_path"]}/best_{self.model_type}_model.pth'
                torch.save(model.state_dict(), model_path)
                print(f'  -> Saved best model to {model_path}')
            else:
                patience_counter += 1
                if patience_counter >= self.config['early_stopping_patience']:
                    print(f'Early stopping at epoch {epoch+1}')
                    break
        
        # Save final model
        final_model_path = f'{self.config["model_save_path"]}/final_{self.model_type}_model.pth'
        torch.save(model.state_dict(), final_model_path)
        
        # Save training history
        history_path = f'{self.config["model_save_path"]}/training_history_{self.model_type}.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f)
        
        return model, self.history
    
    def plot_training_history(self, save_path=None):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Loss plot
        axes[0, 0].plot(self.history['train_loss'], label='Train')
        axes[0, 0].plot(self.history['val_loss'], label='Validation')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy plot
        axes[0, 1].plot(self.history['train_acc'], label='Train')
        axes[0, 1].plot(self.history['val_acc'], label='Validation')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate plot
        axes[1, 0].plot(self.history['learning_rate'])
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Confusion matrix (placeholder - would need validation predictions)
        axes[1, 1].text(0.5, 0.5, 'Confusion Matrix\n(Run evaluation to generate)',
                       horizontalalignment='center', verticalalignment='center')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Training plot saved to {save_path}")
        
        plt.show()
    
    def evaluate(self, model, test_loader):
        """Evaluate model on test data"""
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                _, preds = outputs.max(1)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        return np.array(all_preds), np.array(all_targets)