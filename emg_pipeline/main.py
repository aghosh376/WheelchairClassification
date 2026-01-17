"""
Main entry point for EMG pipeline
Command-line interface for different pipeline modes
"""

import argparse
import time
import sys
from typing import Dict, Any

# Add current directory to path for module imports
sys.path.append('.')

from pipeline import create_pipeline, EMGPipeline, PipelineConfig
from utils import setup_data_directories, print_pipeline_status
from config import PIPELINE_MODES

def parse_arguments():
    parser = argparse.ArgumentParser(description='EMG Processing Pipeline')
    
    # Pipeline mode
    parser.add_argument('--mode', type=str, default='classify',
                       choices=['raw', 'preprocessed', 'features', 'classify', 'train'],
                       help='Pipeline mode')
    
    # Hardware configuration
    parser.add_argument('--board', type=str, default='GANGLION',
                       choices=['GANGLION', 'CYTON'],
                       help='OpenBCI board type')
    parser.add_argument('--port', type=str, help='Serial port (e.g., COM3)')
    
    # Model configuration
    parser.add_argument('--model-type', type=str, default='hybrid',
                       choices=['lstm', 'cnn', 'hybrid', 'threshold'],
                       help='Classification model type')
    parser.add_argument('--model-path', type=str, help='Path to trained model')
    
    # Data configuration
    parser.add_argument('--duration', type=float, default=30.0,
                       help='Streaming duration in seconds')
    parser.add_argument('--buffer-size', type=int, default=1000,
                       help='Data buffer size')
    parser.add_argument('--calibration', type=str,
                       help='Path to calibration file')
    
    # Output configuration
    parser.add_argument('--output', type=str, help='Output file path')
    parser.add_argument('--visualize', action='store_true',
                       help='Enable real-time visualization')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    # Training configuration (if mode=train)
    parser.add_argument('--train-data', type=str, help='Training data file')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Setup directories
    setup_data_directories()
    
    # Create pipeline based on mode
    if args.mode == 'train':
        # Training mode - use training module directly
        from training import ModelTrainer
        
        if not args.train_data:
            print("Error: Training mode requires --train-data argument")
            return
        
        print(f"Training {args.model_type} model on {args.train_data}")
        
        trainer = ModelTrainer(model_type=args.model_type)
        train_loader, val_loader, gesture_names = trainer.prepare_data(
            args.train_data, augment=True
        )
        
        model, history = trainer.train(
            train_loader, val_loader, 
            num_classes=len(gesture_names)
        )
        
        # Plot training history
        trainer.plot_training_history(
            save_path=f"results/training_{args.model_type}.png"
        )
        
        print(f"\nTraining complete!")
        print(f"Gesture classes: {gesture_names}")
        return
    
    else:
        # Real-time processing modes
        print(f"Starting EMG Pipeline in {args.mode} mode")
        
        # Create pipeline
        pipeline = create_pipeline(
            mode=args.mode,
            board_type=args.board,
            serial_port=args.port,
            model_type=args.model_type,
            model_path=args.model_path,
            calibration_file=args.calibration,
            visualize=args.visualize
        )
        
        # Define output callback for verbose mode
        def verbose_callback(result: Dict[str, Any]):
            if 'prediction' in result:
                print(f"[{time.strftime('%H:%M:%S')}] "
                      f"Prediction: {result['prediction']} "
                      f"(Confidence: {result['confidence']:.2f})")
        
        # Configure pipeline
        if args.verbose:
            pipeline.config.output_callback = verbose_callback
        
        # Run pipeline
        try:
            if args.mode in ['raw', 'preprocessed', 'features']:
                # Data collection modes
                print(f"Collecting data for {args.duration} seconds...")
                pipeline.stream(duration=args.duration)
                
                if args.output:
                    pipeline.save_data(args.output)
                    print(f"Data saved to {args.output}")
            
            elif args.mode == 'classify':
                # Real-time classification
                print(f"Starting real-time classification for {args.duration} seconds...")
                print("Press Ctrl+C to stop early")
                
                pipeline.stream(duration=args.duration)
                
                # Print performance stats
                if 'classifier' in pipeline.components:
                    stats = pipeline.components['classifier'].get_performance_stats()
                    print(f"\nClassification Performance:")
                    print(f"  Total predictions: {stats.get('num_predictions', 0)}")
                    print(f"  Average latency: {stats.get('avg_latency_ms', 0):.2f} ms")
            
            # Print final pipeline status
            print_pipeline_status(pipeline.get_pipeline_info())
        
        except KeyboardInterrupt:
            print("\n\nPipeline interrupted by user")
        except Exception as e:
            print(f"\nError in pipeline: {e}")
            import traceback
            traceback.print_exc()
        finally:
            pipeline.stop()

if __name__ == "__main__":
    main()