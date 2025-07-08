#!/usr/bin/env python3
"""
Raspberry Pi TensorFlow Lite Model Performance Test

This script tests the TFLite jamming detection model on Raspberry Pi hardware,
measuring accuracy and inference speed without GUI dependencies.
Designed to run on Raspberry Pi 3 with minimal dependencies.

Usage:
    python3 rpi_tflite_test.py [--dataset test_dataset_name] [--samples num_samples]
"""

import os
import sys
import time
import argparse
import numpy as np
import pickle
from pathlib import Path

# Try to import TensorFlow Lite
try:
    import tensorflow as tf
    TFLITE_AVAILABLE = True
except ImportError:
    try:
        # If full TensorFlow not available, try TFLite runtime
        import tflite_runtime.interpreter as tflite
        TFLITE_AVAILABLE = True
        USE_TFLITE_RUNTIME = True
    except ImportError:
        print("❌ Neither TensorFlow nor TFLite Runtime is available!")
        print("Please install one of the following:")
        print("  - pip3 install tensorflow")
        print("  - pip3 install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-linux_armv7l.whl")
        sys.exit(1)

# Try to import librosa for preprocessing
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    print("⚠️ Librosa not available. Using basic preprocessing fallback.")
    LIBROSA_AVAILABLE = False


class RPiTFLiteEvaluator:
    """TensorFlow Lite model evaluator optimized for Raspberry Pi."""
    
    def __init__(self, model_path, config_path=None):
        """
        Initialize the evaluator.
        
        Args:
            model_path (str): Path to the TFLite model file
            config_path (str): Path to preprocessing config (optional)
        """
        self.model_path = model_path
        self.config_path = config_path
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.class_names = ['Normal', 'Constant Jammer', 'Periodic Jammer']
        
        # Preprocessing parameters (fallback if config not available)
        self.preprocessing_config = {
            'sampling_rate': 1000,
            'n_fft': 256,
            'hop_length': 128,
            'n_mels': 64,
            'sequence_length': 1000
        }
        
        self.load_config()
        self.load_model()
    
    def load_config(self):
        """Load preprocessing configuration if available."""
        if self.config_path and os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'rb') as f:
                    config = pickle.load(f)
                    self.preprocessing_config.update(config)
                print(f"✅ Loaded preprocessing config from {self.config_path}")
            except Exception as e:
                print(f"⚠️ Could not load config: {e}. Using defaults.")
    
    def load_model(self):
        """Load the TFLite model."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"TFLite model not found: {self.model_path}")
        
        try:
            if 'USE_TFLITE_RUNTIME' in globals():
                self.interpreter = tflite.Interpreter(model_path=self.model_path)
            else:
                self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
            
            self.interpreter.allocate_tensors()
            
            # Get input and output details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            # Display model info
            print(f"✅ Loaded TFLite model: {self.model_path}")
            print(f"📱 Model size: {os.path.getsize(self.model_path) / 1024:.1f} KB")
            print(f"🔢 Input shape: {self.input_details[0]['shape']}")
            print(f"🎯 Output shape: {self.output_details[0]['shape']}")
            
        except Exception as e:
            print(f"❌ Error loading TFLite model: {e}")
            raise
    
    def signal_to_mel_spectrogram(self, signal):
        """
        Convert 1D signal to 2D Mel spectrogram.
        Uses librosa if available, otherwise basic fallback.
        """
        # Ensure signal is the right length
        seq_len = self.preprocessing_config['sequence_length']
        if len(signal) != seq_len:
            if len(signal) > seq_len:
                signal = signal[:seq_len]
            else:
                signal = np.pad(signal, (0, seq_len - len(signal)), 'constant')
        
        if LIBROSA_AVAILABLE:
            # Use librosa for proper Mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=signal.astype(np.float32),
                sr=self.preprocessing_config['sampling_rate'],
                n_fft=self.preprocessing_config['n_fft'],
                hop_length=self.preprocessing_config['hop_length'],
                n_mels=self.preprocessing_config['n_mels']
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        else:
            # Basic fallback: simple STFT-based spectrogram
            print("⚠️ Using basic fallback preprocessing (install librosa for full compatibility)")
            # Simple sliding window approach
            window_size = self.preprocessing_config['n_fft']
            hop_length = self.preprocessing_config['hop_length']
            n_mels = self.preprocessing_config['n_mels']
            
            # Create basic spectrogram
            n_frames = (len(signal) - window_size) // hop_length + 1
            mel_spec_db = np.zeros((n_mels, n_frames))
            
            for i in range(n_frames):
                start = i * hop_length
                window = signal[start:start + window_size]
                # Basic power spectrum (simplified)
                fft = np.fft.fft(window)
                power = np.abs(fft[:window_size//2])**2
                # Simple mel-scale approximation
                mel_bins = np.linspace(0, len(power)-1, n_mels, dtype=int)
                mel_spec_db[:, i] = np.log10(power[mel_bins] + 1e-10)
        
        return mel_spec_db
    
    def preprocess_signal(self, signal):
        """Preprocess a signal for inference."""
        mel_spec = self.signal_to_mel_spectrogram(signal)
        
        # Add batch and channel dimensions: (1, height, width, 1)
        processed = mel_spec[np.newaxis, :, :, np.newaxis].astype(np.float32)
        
        return processed
    
    def predict_single(self, signal):
        """
        Make a prediction on a single signal.
        
        Args:
            signal (np.ndarray): 1D signal array
            
        Returns:
            tuple: (predicted_class, confidence, inference_time)
        """
        start_time = time.time()
        
        # Preprocess
        processed_signal = self.preprocess_signal(signal)
        
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], processed_signal)
        
        # Run inference
        inference_start = time.time()
        self.interpreter.invoke()
        inference_time = time.time() - inference_start
        
        # Get output
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        predicted_class = np.argmax(output_data[0])
        confidence = np.max(output_data[0])
        
        total_time = time.time() - start_time
        
        return predicted_class, confidence, inference_time, total_time
    
    def load_test_data(self, dataset_path):
        """Load test data from text files."""
        test_files = {
            'normal': os.path.join(dataset_path, 'Test_Rssi_Normal.txt'),
            'cj': os.path.join(dataset_path, 'Test_Rssi_CJ.txt'),
            'pj': os.path.join(dataset_path, 'Test_Rssi_PJ.txt')
        }
        
        signals = []
        labels = []
        label_map = {'normal': 0, 'cj': 1, 'pj': 2}
        
        for class_name, file_path in test_files.items():
            if os.path.exists(file_path):
                print(f"📁 Loading {class_name} data from {file_path}")
                data = np.loadtxt(file_path)
                
                # Create sequences
                seq_len = self.preprocessing_config['sequence_length']
                n_sequences = len(data) // seq_len
                
                class_signals = data[:n_sequences * seq_len].reshape(n_sequences, seq_len)
                signals.append(class_signals)
                labels.extend([label_map[class_name]] * n_sequences)
                
                print(f"  ✅ Loaded {n_sequences} sequences of length {seq_len}")
            else:
                print(f"⚠️ File not found: {file_path}")
        
        if signals:
            all_signals = np.concatenate(signals, axis=0)
            all_labels = np.array(labels)
            return all_signals, all_labels
        else:
            raise FileNotFoundError("No test data files found!")
    
    def evaluate_dataset(self, dataset_path, max_samples=None):
        """Evaluate the model on a test dataset."""
        print(f"\n🧪 Evaluating on dataset: {dataset_path}")
        
        # Load test data
        signals, labels = self.load_test_data(dataset_path)
        
        if max_samples and len(signals) > max_samples:
            # Randomly sample for faster testing
            indices = np.random.choice(len(signals), max_samples, replace=False)
            signals = signals[indices]
            labels = labels[indices]
            print(f"📊 Testing on {max_samples} random samples")
        
        print(f"📊 Testing on {len(signals)} samples")
        
        # Track metrics
        correct_predictions = 0
        inference_times = []
        total_times = []
        class_correct = {i: 0 for i in range(3)}
        class_total = {i: 0 for i in range(3)}
        
        print("\n🚀 Running inference...")
        start_eval = time.time()
        
        for i, (signal, true_label) in enumerate(zip(signals, labels)):
            if (i + 1) % 50 == 0:
                print(f"  Progress: {i + 1}/{len(signals)} ({100*(i+1)/len(signals):.1f}%)")
            
            pred_class, confidence, inf_time, total_time = self.predict_single(signal)
            
            # Track metrics
            inference_times.append(inf_time * 1000)  # Convert to milliseconds
            total_times.append(total_time * 1000)
            
            if pred_class == true_label:
                correct_predictions += 1
                class_correct[true_label] += 1
            
            class_total[true_label] += 1
        
        eval_time = time.time() - start_eval
        
        # Calculate metrics
        accuracy = correct_predictions / len(signals)
        avg_inference_time = np.mean(inference_times)
        std_inference_time = np.std(inference_times)
        avg_total_time = np.mean(total_times)
        
        # Per-class accuracy
        class_accuracies = {}
        for i in range(3):
            if class_total[i] > 0:
                class_accuracies[self.class_names[i]] = class_correct[i] / class_total[i]
            else:
                class_accuracies[self.class_names[i]] = 0.0
        
        # Display results
        print(f"\n📊 EVALUATION RESULTS")
        print(f"{'='*50}")
        print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Total Samples: {len(signals)}")
        print(f"Correct Predictions: {correct_predictions}")
        print(f"Total Evaluation Time: {eval_time:.2f} seconds")
        
        print(f"\n⏱️ TIMING PERFORMANCE")
        print(f"{'='*50}")
        print(f"Average Inference Time: {avg_inference_time:.2f} ± {std_inference_time:.2f} ms")
        print(f"Average Total Time: {avg_total_time:.2f} ms")
        print(f"Throughput: {1000/avg_total_time:.1f} samples/second")
        
        print(f"\n🎯 PER-CLASS ACCURACY")
        print(f"{'='*50}")
        for class_name, acc in class_accuracies.items():
            count = class_total[list(self.class_names).index(class_name)]
            print(f"{class_name:15}: {acc:.4f} ({acc*100:.2f}%) - {count} samples")
        
        return {
            'accuracy': accuracy,
            'class_accuracies': class_accuracies,
            'avg_inference_time_ms': avg_inference_time,
            'std_inference_time_ms': std_inference_time,
            'avg_total_time_ms': avg_total_time,
            'throughput_sps': 1000/avg_total_time,
            'total_samples': len(signals),
            'correct_predictions': correct_predictions
        }


def get_system_info():
    """Get system information for the report."""
    try:
        with open('/proc/cpuinfo', 'r') as f:
            cpu_info = f.read()
        
        # Extract CPU model
        cpu_model = "Unknown"
        for line in cpu_info.split('\n'):
            if 'model name' in line:
                cpu_model = line.split(':')[1].strip()
                break
        
        # Get memory info
        with open('/proc/meminfo', 'r') as f:
            mem_info = f.read()
        
        total_memory = "Unknown"
        for line in mem_info.split('\n'):
            if 'MemTotal' in line:
                total_memory = line.split()[1]
                total_memory = f"{int(total_memory) // 1024} MB"
                break
        
        print(f"\n💻 SYSTEM INFORMATION")
        print(f"{'='*50}")
        print(f"CPU: {cpu_model}")
        print(f"Memory: {total_memory}")
        
        # Check if running on Raspberry Pi
        try:
            with open('/proc/device-tree/model', 'r') as f:
                device_model = f.read().strip('\x00')
            print(f"Device: {device_model}")
        except:
            print("Device: Unknown (not a Raspberry Pi)")
            
    except Exception as e:
        print(f"Could not get system info: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Test TFLite model performance on Raspberry Pi'
    )
    parser.add_argument(
        '--model', 
        default='model/jamming_detector_lightweight.tflite',
        help='Path to TFLite model file'
    )
    parser.add_argument(
        '--config',
        default='preprocessed_data/config.pkl',
        help='Path to preprocessing config file'
    )
    parser.add_argument(
        '--dataset',
        default='Dataset/test',
        help='Path to test dataset directory'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=None,
        help='Maximum number of samples to test (default: all)'
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=['Dataset/test'],
        help='Multiple dataset paths to test'
    )
    
    args = parser.parse_args()
    
    print("🤖 Raspberry Pi TensorFlow Lite Model Tester")
    print("=" * 60)
    
    # Display system information
    get_system_info()
    
    # Initialize evaluator
    try:
        evaluator = RPiTFLiteEvaluator(args.model, args.config)
    except Exception as e:
        print(f"❌ Failed to initialize evaluator: {e}")
        return 1
    
    # Test on datasets
    all_results = {}
    
    for dataset_path in args.datasets:
        if os.path.exists(dataset_path):
            try:
                results = evaluator.evaluate_dataset(dataset_path, args.samples)
                all_results[dataset_path] = results
            except Exception as e:
                print(f"❌ Error testing dataset {dataset_path}: {e}")
        else:
            print(f"⚠️ Dataset not found: {dataset_path}")
    
    # Summary
    if all_results:
        print(f"\n📈 SUMMARY")
        print(f"{'='*60}")
        for dataset, results in all_results.items():
            dataset_name = os.path.basename(dataset)
            print(f"{dataset_name:20}: {results['accuracy']:.4f} accuracy, "
                  f"{results['avg_inference_time_ms']:.1f}ms avg inference")
    
    return 0


if __name__ == "__main__":
    exit(main())
