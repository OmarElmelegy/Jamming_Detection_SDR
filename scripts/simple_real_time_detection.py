#!/usr/bin/env python3
"""
Simple Real-time Jamming Detection Script

This script implements real-time jamming detection using USRP 2901R
with your preprocessing pipeline and TensorFlow Lite model.
"""

import numpy as np
import time
import sys
import os
import pickle
import threading
import queue
from datetime import datetime

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import GNU Radio
try:
    from gnuradio import gr, blocks, fft, uhd
    from gnuradio.fft import window
    print("✅ GNU Radio imported successfully")
except ImportError as e:
    print(f"❌ GNU Radio import failed: {e}")
    sys.exit(1)

# Import TensorFlow Lite
try:
    import tensorflow as tf
    TFLITE_AVAILABLE = True
    print("✅ TensorFlow imported successfully")
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
        TFLITE_AVAILABLE = True
        USE_TFLITE_RUNTIME = True
        print("✅ TensorFlow Lite Runtime imported successfully")
    except ImportError:
        print("❌ Neither TensorFlow nor TFLite Runtime available!")
        sys.exit(1)

# Import preprocessing
try:
    from preprocess import RSSSIToMelSpectrogram
    print("✅ Preprocessing module imported successfully")
except ImportError as e:
    print(f"❌ Failed to import preprocessing: {e}")
    sys.exit(1)


class JammingDetectionSink(gr.sync_block):
    """
    Custom GNU Radio sink block for jamming detection.
    """
    
    def __init__(self, model_path, config_path, vector_length=1024):
        gr.sync_block.__init__(
            self,
            name="jamming_detection_sink",
            in_sig=[(np.float32, vector_length)],
            out_sig=None
        )
        
        self.vector_length = vector_length
        self.downsample_factor = 40  # Take mean of every 40 points
        self.signal_length = 1000
        self.class_names = ['Normal', 'Constant Jammer', 'Periodic Jammer']
        
        # Buffer for accumulating signal points
        self.signal_buffer = []
        self.prediction_count = 0
        
        # Load model and config
        self.load_model(model_path)
        self.load_config(config_path)
        
        # Initialize preprocessing
        self.preprocessor = RSSSIToMelSpectrogram(
            sampling_rate=self.config.get('sampling_rate', 1000),
            n_fft=self.config.get('n_fft', 256),
            hop_length=self.config.get('hop_length', 128),
            n_mels=self.config.get('n_mels', 64),
            sequence_length=self.config.get('sequence_length', 1000)
        )
        
        print(f"🔧 Jamming Detection Sink initialized:")
        print(f"   Vector length: {self.vector_length}")
        print(f"   Downsample factor: {self.downsample_factor}")
        print(f"   Signal length: {self.signal_length}")
    
    def load_model(self, model_path):
        """Load TensorFlow Lite model."""
        try:
            if 'USE_TFLITE_RUNTIME' in globals():
                self.interpreter = tflite.Interpreter(model_path=model_path)
            else:
                self.interpreter = tf.lite.Interpreter(model_path=model_path)
            
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            print(f"🤖 Model loaded: {model_path}")
            print(f"   Input shape: {self.input_details[0]['shape']}")
            print(f"   Output shape: {self.output_details[0]['shape']}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def load_config(self, config_path):
        """Load preprocessing configuration."""
        try:
            with open(config_path, 'rb') as f:
                self.config = pickle.load(f)
            print(f"⚙️  Config loaded: {config_path}")
        except Exception as e:
            print(f"⚠️  Could not load config: {e}. Using defaults.")
            self.config = {
                'sampling_rate': 1000,
                'n_fft': 256,
                'hop_length': 128,
                'n_mels': 64,
                'sequence_length': 1000
            }
    
    def predict_signal(self, signal):
        """Make prediction on a complete signal."""
        try:
            # Ensure signal is the right length
            if len(signal) != self.signal_length:
                if len(signal) > self.signal_length:
                    signal = signal[:self.signal_length]
                else:
                    signal = np.pad(signal, (0, self.signal_length - len(signal)), 'constant')
            
            # Convert to mel spectrogram
            mel_spec = self.preprocessor.signal_to_mel_spectrogram(signal)
            
            # Prepare input for model
            input_data = mel_spec[np.newaxis, :, :, np.newaxis].astype(np.float32)
            
            # Run inference
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            
            # Get output
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            predicted_class = np.argmax(output_data[0])
            confidence = np.max(output_data[0])
            
            return predicted_class, confidence
            
        except Exception as e:
            print(f"❌ Error in prediction: {e}")
            return 0, 0.0
    
    def work(self, input_items, output_items):
        """Process input vectors."""
        in0 = input_items[0]
        
        for vector in in0:
            # Downsample by taking mean of every 40 points
            downsampled = []
            for i in range(0, len(vector), self.downsample_factor):
                chunk = vector[i:i + self.downsample_factor]
                if len(chunk) > 0:
                    downsampled.append(np.mean(chunk))
            
            # Add to signal buffer
            self.signal_buffer.extend(downsampled)
            
            # Check if we have enough points for a complete signal
            while len(self.signal_buffer) >= self.signal_length:
                # Extract signal
                signal = np.array(self.signal_buffer[:self.signal_length])
                self.signal_buffer = self.signal_buffer[self.signal_length:]
                
                # Make prediction
                predicted_class, confidence = self.predict_signal(signal)
                
                # Display prediction
                class_name = self.class_names[predicted_class]
                timestamp = datetime.now().strftime("%H:%M:%S")
                self.prediction_count += 1
                
                # Color coding for terminal output
                if class_name == "Normal":
                    color = "\033[32m"  # Green
                elif class_name == "Constant Jammer":
                    color = "\033[31m"  # Red
                elif class_name == "Periodic Jammer":
                    color = "\033[33m"  # Yellow
                else:
                    color = "\033[0m"   # Default
                
                print(f"{color}[{timestamp}] Prediction #{self.prediction_count}: {class_name} "
                      f"(Confidence: {confidence:.2%})\033[0m")
        
        return len(input_items[0])


class JammingDetectionFlowgraph(gr.top_block):
    """
    Main flowgraph for real-time jamming detection.
    """
    
    def __init__(self):
        gr.top_block.__init__(self, "Real-time Jamming Detection")
        
        # Parameters
        self.samp_rate = 1000000  # 1 MHz
        self.center_freq = 915000000  # 915 MHz
        self.gain = 70  # 70 dB
        self.fft_size = 1024
        
        # Model paths
        self.model_path = "model/jamming_detector_lightweight.tflite"
        self.config_path = "preprocessed_data/config.pkl"
        
        # Build flowgraph
        self.build_flowgraph()
        
        print("🚀 Flowgraph built successfully!")
    
    def build_flowgraph(self):
        """Build the GNU Radio flowgraph."""
        
        # USRP Source
        print("📡 Initializing USRP 2901R...")
        self.uhd_usrp_source = uhd.usrp_source(
            ",".join(("", "")),
            uhd.stream_args(
                cpu_format="fc32",
                channels=list(range(0, 1)),
            ),
        )
        
        # Configure USRP
        self.uhd_usrp_source.set_samp_rate(self.samp_rate)
        self.uhd_usrp_source.set_center_freq(self.center_freq, 0)
        self.uhd_usrp_source.set_gain(self.gain, 0)
        self.uhd_usrp_source.set_antenna('RX2', 0)
        
        print(f"   Sample Rate: {self.samp_rate/1e6:.1f} MHz")
        print(f"   Center Frequency: {self.center_freq/1e6:.1f} MHz")
        print(f"   Gain: {self.gain} dB")
        print(f"   Antenna: RX2")
        
        # Stream to Vector (for FFT)
        self.stream_to_vector = blocks.stream_to_vector(
            gr.sizeof_gr_complex, self.fft_size
        )
        
        # FFT
        self.fft_block = fft.fft_vcc(self.fft_size, True, (), True, 1)
        
        # Complex to Magnitude
        self.complex_to_mag = blocks.complex_to_mag(self.fft_size)
        
        # Log10
        self.log10_block = blocks.nlog10_ff(10, self.fft_size, 0)
        
        # Custom jamming detection sink
        self.jamming_detector = JammingDetectionSink(
            self.model_path,
            self.config_path,
            self.fft_size
        )
        
        # Connect blocks
        self.connect((self.uhd_usrp_source, 0), (self.stream_to_vector, 0))
        self.connect((self.stream_to_vector, 0), (self.fft_block, 0))
        self.connect((self.fft_block, 0), (self.complex_to_mag, 0))
        self.connect((self.complex_to_mag, 0), (self.log10_block, 0))
        self.connect((self.log10_block, 0), (self.jamming_detector, 0))
        
        print("🔗 Flowgraph connections established!")


def main():
    """Main function."""
    print("🎯 Real-time Jamming Detection System")
    print("=" * 50)
    
    # Check if model files exist
    if not os.path.exists("model/jamming_detector_lightweight.tflite"):
        print("❌ TensorFlow Lite model not found!")
        print("Please ensure 'model/jamming_detector_lightweight.tflite' exists.")
        return 1
    
    if not os.path.exists("preprocessed_data/config.pkl"):
        print("⚠️  Config file not found, using defaults.")
    
    # Create and start the flowgraph
    try:
        print("🔧 Creating flowgraph...")
        tb = JammingDetectionFlowgraph()
        
        print("▶️  Starting real-time detection...")
        tb.start()
        
        print("🎉 System is now running! Press Ctrl+C to stop.")
        print("📊 Monitoring 915 MHz with 1 MHz bandwidth...")
        print("🔍 Jamming detection results will appear below:")
        print("-" * 50)
        
        # Keep the program running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\n🛑 Stopping system...")
            tb.stop()
            tb.wait()
            print("✅ System stopped successfully!")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
