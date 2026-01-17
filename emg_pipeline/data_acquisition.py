"""
Data acquisition interface for OpenBCI Ganglion/Cyton
Supports multiple data formats and streaming modes
"""

import time
import numpy as np
from collections import deque
from brainflow.board_shim import BoardShim, BrainFlowInputParams

from config import BOARD_CONFIG, DATA_PATHS

class DataAcquisition:
    """Flexible data acquisition from OpenBCI boards"""
    
    def __init__(self, board_type='GANGLION', port=None, streaming=True):
        self.board_type = board_type
        self.config = BOARD_CONFIG[board_type]
        self.board = None
        self.streaming = streaming
        
        # Data buffers
        self.raw_buffer = deque(maxlen=self.config['sampling_rate'] * 10)  # 10 seconds
        self.timestamp_buffer = deque(maxlen=self.config['sampling_rate'] * 10)
        
        # Initialize board params
        self.params = BrainFlowInputParams()
        self.params.serial_port = port or self.config['default_port']
        
    def connect(self):
        """Connect to the board"""
        try:
            self.board = BoardShim(self.config['board_id'], self.params)
            self.board.prepare_session()
            print(f"Connected to {self.board_type} on {self.params.serial_port}")
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False
    
    def start_stream(self):
        """Start data streaming"""
        if self.board:
            self.board.start_stream()
            print("Streaming started")
            time.sleep(2)  # Allow buffer to fill
            return True
        return False
    
    def get_raw_data(self, num_samples=None, channels='emg'):
        """
        Get raw data from board
        Args:
            num_samples: Number of samples to retrieve (None = all available)
            channels: 'emg', 'all', or list of channel indices
        Returns:
            dict with data, timestamps, and metadata
        """
        if not self.board:
            return None
        
        # Get all available data
        data = self.board.get_board_data()
        
        if len(data) == 0:
            return None
        
        # Select channels
        if channels == 'emg':
            channel_indices = self.config['emg_channels']
        elif channels == 'all':
            channel_indices = list(range(data.shape[0]))
        elif isinstance(channels, list):
            channel_indices = channels
        else:
            channel_indices = self.config['emg_channels']
        
        # Limit to requested samples
        if num_samples and num_samples < data.shape[1]:
            data = data[:, :num_samples]
        
        # Extract selected channels
        channel_data = data[channel_indices, :]
        
        # Get timestamps
        timestamp_channel = self.config.get('timestamp_channel', -1)
        if timestamp_channel < data.shape[0]:
            timestamps = data[timestamp_channel, :]
        else:
            timestamps = np.arange(channel_data.shape[1]) / self.config['sampling_rate']
        
        # Update buffers
        for i in range(channel_data.shape[1]):
            self.raw_buffer.append(channel_data[:, i])
            self.timestamp_buffer.append(timestamps[i])
        
        result = {
            'data': channel_data,
            'timestamps': timestamps,
            'channels': channel_indices,
            'sampling_rate': self.config['sampling_rate'],
            'board_type': self.board_type,
            'timestamp': time.time()
        }
        
        return result
    
    def stream_data(self, callback=None, duration=None, interval=0.1):
        """
        Stream data continuously
        Args:
            callback: Function to call with each data batch
            duration: Total streaming duration in seconds
            interval: Time between batches in seconds
        """
        start_time = time.time()
        
        try:
            while True:
                if duration and (time.time() - start_time) > duration:
                    break
                
                # Get new data
                data_packet = self.get_raw_data(
                    num_samples=int(self.config['sampling_rate'] * interval)
                )
                
                if data_packet and callback:
                    callback(data_packet)
                
                time.sleep(max(0, interval - 0.01))
        
        except KeyboardInterrupt:
            print("\nStreaming stopped")
        except Exception as e:
            print(f"Streaming error: {e}")
    
    def record_data(self, duration, save_path=None):
        """Record data for specified duration"""
        print(f"Recording {duration} seconds of data...")
        
        all_data = []
        start_time = time.time()
        
        while time.time() - start_time < duration:
            data_packet = self.get_raw_data()
            if data_packet:
                all_data.append(data_packet)
        
        if save_path:
            self.save_data(all_data, save_path)
        
        return all_data
    
    def save_data(self, data, filepath):
        """Save data to file"""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Data saved to {filepath}")
    
    def load_data(self, filepath):
        """Load data from file"""
        import pickle
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def disconnect(self):
        """Disconnect from board"""
        if self.board:
            self.board.stop_stream()
            self.board.release_session()
            print("Disconnected from board")