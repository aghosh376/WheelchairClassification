"""
Main pipeline orchestrator - connects all modules
"""

import time
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Callable
import json

from config import PIPELINE_MODES, DATA_PATHS
from data_acquisition import DataAcquisition
from preprocessing import Preprocessor, MultiChannelPreprocessor
from feature_extraction import FeatureExtractor, MultiChannelFeatureExtractor
from classifier import EMGClassifier, MultiModelEnsemble

@dataclass
class PipelineConfig:
    """Configuration for the EMG pipeline"""
    mode: str = PIPELINE_MODES['CLASSIFY']
    board_type: str = 'GANGLION'
    serial_port: Optional[str] = None
    model_type: str = 'hybrid'
    model_path: Optional[str] = None
    use_features: bool = True
    stream_interval: float = 0.1  # seconds
    buffer_size: int = 1000
    calibration_file: Optional[str] = None
    output_callback: Optional[Callable] = None
    visualize: bool = False

class EMGPipeline:
    """Main pipeline orchestrator"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.components = {}
        self.data_buffer = []
        self.is_running = False
        
        # Initialize based on mode
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize pipeline components based on mode"""
        print(f"Initializing EMG Pipeline in {self.config.mode} mode...")
        
        # Always initialize data acquisition if not in pure inference mode
        if self.config.mode in [PIPELINE_MODES['RAW'], PIPELINE_MODES['PREPROCESSED'], 
                               PIPELINE_MODES['FEATURES'], PIPELINE_MODES['CLASSIFY']]:
            self.components['acquisition'] = DataAcquisition(
                board_type=self.config.board_type,
                port=self.config.serial_port
            )
        
        # Initialize preprocessor if needed
        if self.config.mode in [PIPELINE_MODES['PREPROCESSED'], PIPELINE_MODES['FEATURES'], 
                               PIPELINE_MODES['CLASSIFY']]:
            self.components['preprocessor'] = Preprocessor(sampling_rate=200)
            
            # Load calibration if available
            if self.config.calibration_file:
                self.load_calibration(self.config.calibration_file)
        
        # Initialize feature extractor if needed
        if self.config.mode in [PIPELINE_MODES['FEATURES'], PIPELINE_MODES['CLASSIFY']] and self.config.use_features:
            self.components['feature_extractor'] = FeatureExtractor(sampling_rate=200)
        
        # Initialize classifier if needed
        if self.config.mode == PIPELINE_MODES['CLASSIFY']:
            self.components['classifier'] = EMGClassifier(
                model_type=self.config.model_type,
                model_path=self.config.model_path
            )
        
        # Initialize visualizer if requested
        if self.config.visualize:
            try:
                from stream_visualizer import EMGVisualizer
                self.components['visualizer'] = EMGVisualizer()
            except ImportError:
                print("Visualizer module not available")
    
    def connect(self) -> bool:
        """Connect to EMG hardware"""
        if 'acquisition' not in self.components:
            print("No acquisition component initialized")
            return False
        
        return self.components['acquisition'].connect()
    
    def start(self):
        """Start the pipeline"""
        if not self.connect():
            print("Failed to connect to EMG hardware")
            return False
        
        # Start acquisition stream
        if 'acquisition' in self.components:
            self.components['acquisition'].start_stream()
        
        self.is_running = True
        print("Pipeline started")
        return True
    
    def process_single_frame(self, raw_data: Optional[Dict] = None) -> Dict:
        """
        Process a single frame of data through the pipeline
        Args:
            raw_data: Optional raw data input. If None, acquires from hardware
        Returns:
            Dictionary with all processing results
        """
        result = {
            'timestamp': time.time(),
            'mode': self.config.mode
        }
        
        # Step 1: Acquire data
        if raw_data is None and 'acquisition' in self.components:
            raw_data = self.components['acquisition'].get_raw_data()
            result['raw_data'] = raw_data
        elif raw_data is not None:
            result['raw_data'] = raw_data
        
        if raw_data is None:
            return result
        
        # Step 2: Preprocess (if enabled)
        if 'preprocessor' in self.components:
            # Extract EMG channels
            emg_data = raw_data['data'][:4]  # First 4 channels for Ganglion
            
            # Process each channel
            processed_channels = []
            for i in range(emg_data.shape[0]):
                processed = self.components['preprocessor'].process_pipeline(
                    emg_data[i]
                )
                processed_channels.append(processed)
            
            result['processed_data'] = np.array(processed_channels)
        
        # Step 3: Extract features (if enabled)
        if 'feature_extractor' in self.components and 'processed_data' in result:
            features = self.components['feature_extractor'].extract_all(
                result['processed_data'][0]  # Use first channel
            )
            result['features'] = features
        
        # Step 4: Classify (if enabled)
        if 'classifier' in self.components:
            if 'processed_data' in result:
                # Use processed data for classification
                classification_input = result['processed_data'][0]  # First channel
            else:
                # Fall back to raw data
                classification_input = raw_data['data'][0]
            
            prediction, confidence = self.components['classifier'].predict(
                classification_input
            )
            
            result['prediction'] = prediction
            result['confidence'] = confidence
            result['gesture'] = prediction  # Alias for compatibility
        
        # Step 5: Visualize (if enabled)
        if 'visualizer' in self.components:
            self.components['visualizer'].update(result)
        
        # Call output callback if provided
        if self.config.output_callback:
            self.config.output_callback(result)
        
        return result
    
    def stream(self, duration: Optional[float] = None):
        """
        Stream data through pipeline continuously
        Args:
            duration: Total streaming duration in seconds. None = infinite
        """
        if not self.start():
            return
        
        start_time = time.time()
        
        try:
            while self.is_running:
                # Check duration limit
                if duration and (time.time() - start_time) > duration:
                    break
                
                # Process single frame
                result = self.process_single_frame()
                
                # Store in buffer
                self.data_buffer.append(result)
                
                # Maintain buffer size
                if len(self.data_buffer) > self.config.buffer_size:
                    self.data_buffer = self.data_buffer[-self.config.buffer_size:]
                
                # Sleep to control processing rate
                time.sleep(self.config.stream_interval)
        
        except KeyboardInterrupt:
            print("\nPipeline stopped by user")
        finally:
            self.stop()
    
    def calibrate(self, calibration_time: float = 5.0) -> Dict:
        """
        Calibrate the pipeline for a specific user/gesture set
        Returns:
            Calibration data dictionary
        """
        print("Starting calibration...")
        
        if 'acquisition' not in self.components:
            print("Cannot calibrate: No acquisition component")
            return {}
        
        # Start stream
        self.components['acquisition'].start_stream()
        
        calibration_data = {
            'gestures': ['Relax', 'Flex', 'Pinch', 'Extension'],
            'samples': {},
            'timestamp': time.time(),
            'board_type': self.config.board_type
        }
        
        for gesture in calibration_data['gestures']:
            print(f"\nCalibrating: {gesture}")
            print(f"Assume position in 3 seconds...")
            time.sleep(3)
            
            print(f"Recording {gesture}...")
            samples = []
            start_time = time.time()
            
            while time.time() - start_time < calibration_time:
                raw_data = self.components['acquisition'].get_raw_data()
                if raw_data:
                    samples.append(raw_data['data'][0])  # First channel
                time.sleep(0.01)
            
            calibration_data['samples'][gesture] = np.array(samples)
            print(f"  Collected {len(samples)} samples")
        
        # Calculate calibration parameters
        if 'preprocessor' in self.components:
            # Calibrate baseline from relax data
            relax_data = calibration_data['samples']['Relax']
            if len(relax_data) > 0:
                self.components['preprocessor'].calibrate_baseline(
                    relax_data.flatten()
                )
        
        # Save calibration
        calibration_file = f"{DATA_PATHS['calibration']}/calibration_{int(time.time())}.json"
        self.save_calibration(calibration_data, calibration_file)
        
        print(f"\nCalibration complete. Data saved to {calibration_file}")
        self.components['acquisition'].stop_stream()
        
        return calibration_data
    
    def save_calibration(self, calibration_data: Dict, filepath: str):
        """Save calibration data to file"""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(calibration_data, f)
    
    def load_calibration(self, filepath: str) -> bool:
        """Load calibration from file"""
        try:
            import pickle
            with open(filepath, 'rb') as f:
                calibration_data = pickle.load(f)
            
            # Apply calibration to preprocessor
            if 'preprocessor' in self.components and 'Relax' in calibration_data['samples']:
                relax_data = calibration_data['samples']['Relax']
                self.components['preprocessor'].calibrate_baseline(relax_data.flatten())
            
            print(f"Loaded calibration from {filepath}")
            return True
        
        except Exception as e:
            print(f"Failed to load calibration: {e}")
            return False
    
    def get_pipeline_info(self) -> Dict:
        """Get information about the current pipeline configuration"""
        info = {
            'mode': self.config.mode,
            'board_type': self.config.board_type,
            'components': list(self.components.keys()),
            'is_running': self.is_running,
            'buffer_size': len(self.data_buffer)
        }
        
        # Add component-specific info
        if 'classifier' in self.components:
            info['classifier_type'] = self.config.model_type
            info['classifier_performance'] = self.components['classifier'].get_performance_stats()
        
        return info
    
    def stop(self):
        """Stop the pipeline"""
        self.is_running = False
        
        # Stop all components
        for name, component in self.components.items():
            if hasattr(component, 'disconnect'):
                component.disconnect()
            elif hasattr(component, 'stop'):
                component.stop()
        
        print("Pipeline stopped")
    
    def save_data(self, filepath: str):
        """Save buffered data to file"""
        if not self.data_buffer:
            print("No data to save")
            return
        
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.data_buffer, f)
        
        print(f"Saved {len(self.data_buffer)} frames to {filepath}")
    
    def load_and_process(self, filepath: str):
        """Load data from file and process through pipeline"""
        import pickle
        with open(filepath, 'rb') as f:
            data_frames = pickle.load(f)
        
        results = []
        for frame in data_frames:
            result = self.process_single_frame(frame)
            results.append(result)
        
        return results


# Factory function for creating pipelines
def create_pipeline(mode: str = 'classify', **kwargs) -> EMGPipeline:
    """
    Factory function to create EMG pipelines
    Args:
        mode: One of 'raw', 'preprocessed', 'features', 'classify', 'train'
        **kwargs: Pipeline configuration parameters
    Returns:
        Configured EMGPipeline instance
    """
    # Map mode string to enum
    mode_enum = PIPELINE_MODES.get(mode.upper(), PIPELINE_MODES['CLASSIFY'])
    
    config = PipelineConfig(
        mode=mode_enum,
        board_type=kwargs.get('board_type', 'GANGLION'),
        serial_port=kwargs.get('serial_port'),
        model_type=kwargs.get('model_type', 'hybrid'),
        model_path=kwargs.get('model_path'),
        use_features=kwargs.get('use_features', True),
        stream_interval=kwargs.get('stream_interval', 0.1),
        calibration_file=kwargs.get('calibration_file'),
        output_callback=kwargs.get('output_callback'),
        visualize=kwargs.get('visualize', False)
    )
    
    return EMGPipeline(config)