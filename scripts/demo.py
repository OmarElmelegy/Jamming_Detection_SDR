#!/usr/bin/env python3
"""
Real-time Inference Demo for Lightweight 2D CNN Jamming Detection

This script demonstrates real-time jamming detection using the production-ready
lightweight 2D CNN model that achieves 99.6% accuracy on independent test data.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import librosa
import argparse
import os


class JammingDetector:
    """Production jamming detection using lightweight 2D CNN."""
    
    def __init__(self, model_path='model/jamming_detector_2d_cnn_lightweight_best.h5'):
        """Initialize the detector with the production model."""
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
            
        self.model = load_model(model_path)
        self.class_names = ['Normal', 'Constant Jammer', 'Periodic Jammer']
        self.colors = {
            'Normal': '\033[92m',           # Green
            'Constant Jammer': '\033[91m',  # Red  
            'Periodic Jammer': '\033[94m',  # Blue
            'ENDC': '\033[0m'               # End color
        }
        
        # Mel spectrogram parameters (matches training)
        self.config = {
            'sampling_rate': 1000,
            'n_fft': 256,
            'hop_length': 128,
            'n_mels': 64,
            'sequence_length': 1000
        }
        
        print(f"✅ Loaded production model: {model_path}")
        print(f"📊 Model input shape: {self.model.input_shape}")
        print(f"🎯 Classes: {', '.join(self.class_names)}")
        print(f"🏆 Expected accuracy: 99.6% (independent test)")
    
    def rssi_to_spectrogram(self, rssi_signal):
        """Convert RSSI signal to Mel spectrogram."""
        if len(rssi_signal) != self.config['sequence_length']:
            raise ValueError(f"Expected {self.config['sequence_length']} samples, got {len(rssi_signal)}")
        
        # Convert to Mel spectrogram
        spectrogram = librosa.feature.melspectrogram(
            y=rssi_signal.astype(np.float32),
            sr=self.config['sampling_rate'],
            n_fft=self.config['n_fft'],
            hop_length=self.config['hop_length'],
            n_mels=self.config['n_mels']
        )
        
        # Convert to dB scale and normalize
        spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
        spec_min = spectrogram_db.min()
        spec_max = spectrogram_db.max()
        spectrogram_norm = (spectrogram_db - spec_min) / (spec_max - spec_min)
        
        # Add channel dimension
        return np.expand_dims(spectrogram_norm, axis=-1)
    
    def predict(self, rssi_signal, return_probs=False):
        """Predict jamming type from RSSI signal."""
        # Convert to spectrogram and add batch dimension
        spectrogram = self.rssi_to_spectrogram(rssi_signal)
        spectrogram_batch = np.expand_dims(spectrogram, axis=0)
        
        # Predict
        predictions = self.model.predict(spectrogram_batch, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        result = {
            'class': self.class_names[predicted_class],
            'class_id': predicted_class,
            'confidence': confidence,
            'probabilities': dict(zip(self.class_names, predictions[0]))
        }
        
        return result if return_probs else (result['class'], result['confidence'])
    
    def print_prediction(self, result):
        """Print prediction with colored output."""
        class_name = result['class']
        confidence = result['confidence']
        color = self.colors.get(class_name, '')
        end_color = self.colors['ENDC']
        
        print(f"🔍 Prediction: {color}{class_name}{end_color}")
        print(f"📈 Confidence: {confidence:.1%}")
        print("📊 Probabilities:")
        
        for class_name, prob in result['probabilities'].items():
            color = self.colors.get(class_name, '')
            bar_length = int(prob * 20)  # Scale to 20 chars
            bar = "█" * bar_length + "░" * (20 - bar_length)
            print(f"   {color}{class_name:15}{end_color} {bar} {prob:.1%}")


def load_sample_data():
    """Load sample RSSI data for demonstration."""
    data_files = {
        'Normal': 'Dataset/training/Rssi_Normal.txt',
        'Constant Jammer': 'Dataset/training/Rssi_CJ.txt', 
        'Periodic Jammer': 'Dataset/training/Rssi_PJ.txt'
    }
    
    samples = {}
    
    for label, file_path in data_files.items():
        if os.path.exists(file_path):
            print(f"📁 Loading {label} samples from {file_path}")
            data = np.loadtxt(file_path)
            
            if len(data) >= 1000:
                max_start = len(data) - 1000
                start_idx = np.random.randint(0, max_start + 1) if max_start > 0 else 0
                sample = data[start_idx:start_idx + 1000]
                samples[label] = sample
            else:
                print(f"⚠️  Warning: {file_path} has insufficient data")
        else:
            print(f"❌ File not found: {file_path}")
    
    return samples


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description='Lightweight 2D CNN Jamming Detection Demo')
    parser.add_argument('--model', 
                       default='model/jamming_detector_2d_cnn_lightweight_best.h5',
                       help='Path to trained model')
    parser.add_argument('--samples', type=int, default=3,
                       help='Number of samples to test per class')
    
    args = parser.parse_args()
    
    print("🚀 LIGHTWEIGHT 2D CNN JAMMING DETECTION DEMO")
    print("=" * 60)
    print("Production Model: 99.6% Independent Test Accuracy")
    print("")
    
    try:
        # Initialize detector
        detector = JammingDetector(args.model)
        print()
        
        # Load and test sample data
        print("📊 Demo Mode - Testing with sample data")
        samples = load_sample_data()
        
        if not samples:
            print("❌ No sample data available")
            return
        
        print(f"\n🧪 Testing {args.samples} samples per class...")
        
        all_correct = 0
        all_total = 0
        
        for true_label, full_signal in samples.items():
            print(f"\n" + "=" * 30)
            print(f"🎯 True Label: {true_label}")
            print("=" * 30)
            
            for i in range(args.samples):
                if len(full_signal) >= 1000:
                    max_start = len(full_signal) - 1000
                    start_idx = np.random.randint(0, max_start + 1) if max_start > 0 else 0
                    sample = full_signal[start_idx:start_idx + 1000]
                else:
                    print(f"⚠️  Signal too short for sample {i+1}")
                    continue
                
                # Predict
                result = detector.predict(sample, return_probs=True)
                
                print(f"\n📋 Sample {i+1}:")
                detector.print_prediction(result)
                
                # Check accuracy
                if result['class'] == true_label:
                    print("✅ Correct!")
                    all_correct += 1
                else:
                    print("❌ Incorrect!")
                
                all_total += 1
                print("-" * 40)
        
        # Overall accuracy
        accuracy = all_correct / all_total if all_total > 0 else 0
        print(f"\n🎯 Demo Accuracy: {accuracy:.1%} ({all_correct}/{all_total})")
        print(f"🏆 Expected Production Accuracy: 99.6%")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("\nAvailable models:")
        model_dir = 'model'
        if os.path.exists(model_dir):
            for f in os.listdir(model_dir):
                if f.endswith('.h5') and 'lightweight' in f:
                    print(f"   📁 {os.path.join(model_dir, f)}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n👋 Demo completed!")


if __name__ == "__main__":
    main()
