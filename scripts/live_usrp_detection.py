#!/usr/bin/env python3
"""
Live Real-time Jamming Detection with USRP Hardware

This script implements real-time jamming detection using a connected USRP device.
It integrates GNU Radio for signal acquisition with our ML-based detection system.
"""

import numpy as np
import sys
import os
import time
import threading
import queue
import signal
from datetime import datetime
from collections import deque

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# Import GNU Radio modules
try:
    # Add system path for GNU Radio
    sys.path.append('/usr/lib/python3/dist-packages')
    from gnuradio import gr, blocks, uhd, fft, analog
    from gnuradio.fft import window
    from gnuradio.filter import firdes
    import pmt
    GNU_RADIO_AVAILABLE = True
    print("✅ GNU Radio imported successfully")
except ImportError as e:
    print(f"❌ GNU Radio import failed: {e}")
    print("Please install GNU Radio: sudo apt install gnuradio uhd-host")
    sys.exit(1)

# Import ML modules
try:
    import tensorflow as tf
    TFLITE_AVAILABLE = True
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
        TFLITE_AVAILABLE = True
        USE_TFLITE_RUNTIME = True
    except ImportError:
        print("❌ TensorFlow Lite not available!")
        sys.exit(1)

try:
    import librosa
    from preprocess import RSSSIToMelSpectrogram
    from jamming_alert_system import JammingAlertSystem
    print("✅ All required modules imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

class JammingDetectionSink(gr.sync_block):
    """
    GNU Radio sink block that performs jamming detection on incoming signals.
    """
    
    def __init__(self, 
                 model_path='../model/jamming_detector_lightweight.tflite',
                 config_path='../preprocessed_data/config.pkl',
                 vector_length=1024,
                 enable_alerts=True):
        
        gr.sync_block.__init__(
            self,
            name="jamming_detection_sink",
            in_sig=[(np.float32, vector_length)],
            out_sig=None
        )
        
        print("🔧 Initializing Jamming Detection Sink...")
        
        self.vector_length = vector_length
        self.downsample_factor = 40  # Reduce 1024 points to ~25 points
        self.signal_length = 1000
        self.class_names = ['Normal', 'Constant Jammer', 'Periodic Jammer']
        
        # Buffer for accumulating signal points
        self.signal_buffer = []
        self.detection_count = 0
        
        # Initialize ML components
        self.init_model(model_path, config_path)
        
        # Initialize alert system
        if enable_alerts:
            try:
                self.alert_system = JammingAlertSystem()
                print("✅ Alert system initialized")
            except Exception as e:
                print(f"⚠️  Alert system failed to initialize: {e}")
                self.alert_system = None
        else:
            self.alert_system = None
        
        print("✅ Jamming Detection Sink initialized successfully")
    
    def init_model(self, model_path, config_path):
        """Initialize the ML model and preprocessing."""
        try:
            # Load model
            if 'USE_TFLITE_RUNTIME' in globals():
                self.interpreter = tflite.Interpreter(model_path=model_path)
            else:
                self.interpreter = tf.lite.Interpreter(model_path=model_path)
            
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            print(f"✅ Model loaded: {model_path}")
            
            # Load config
            import pickle
            try:
                with open(config_path, 'rb') as f:
                    self.config = pickle.load(f)
                print(f"✅ Config loaded: {config_path}")
            except Exception as e:
                print(f"⚠️  Using default config: {e}")
                self.config = {
                    'sampling_rate': 1000,
                    'n_fft': 256,
                    'hop_length': 128,
                    'n_mels': 64,
                    'sequence_length': 1000
                }
            
            # Initialize preprocessing
            self.preprocessor = RSSSIToMelSpectrogram(
                sampling_rate=self.config.get('sampling_rate', 1000),
                n_fft=self.config.get('n_fft', 256),
                hop_length=self.config.get('hop_length', 128),
                n_mels=self.config.get('n_mels', 64),
                sequence_length=self.config.get('sequence_length', 1000)
            )
            
            print("✅ Preprocessing initialized")
            
        except Exception as e:
            print(f"❌ Failed to initialize model: {e}")
            self.model_available = False
        else:
            self.model_available = True
    
    def predict_signal(self, signal):
        """Make prediction on a signal."""
        if not self.model_available:
            return 0, 0.0, np.array([1.0, 0.0, 0.0])
        
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
            probabilities = output_data[0]
            
            return predicted_class, confidence, probabilities
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return 0, 0.0, np.array([1.0, 0.0, 0.0])
    
    def work(self, input_items, output_items):
        """Process input vectors from GNU Radio."""
        in0 = input_items[0]
        
        for vector in in0:
            # Downsample by taking mean of chunks
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
                predicted_class, confidence, probabilities = self.predict_signal(signal)
                
                # Process with alert system
                self.detection_count += 1
                timestamp = datetime.now()
                
                if self.alert_system:
                    self.alert_system.process_detection(
                        predicted_class, confidence, probabilities, timestamp
                    )
                else:
                    # Simple console output if no alert system
                    class_name = self.class_names[predicted_class]
                    print(f"[{timestamp.strftime('%H:%M:%S')}] {class_name} | Confidence: {confidence:.1%}")
                
                # Throttle output for performance
                if self.detection_count % 10 == 0:
                    print(f"Processed {self.detection_count} detections...")
        
        return len(input_items[0])

class LiveJammingDetection(gr.top_block):
    """
    GNU Radio flowgraph for live jamming detection using USRP hardware.
    """
    
    def __init__(self, 
                 center_freq=915e6,
                 sample_rate=1e6,
                 gain=30,
                 fft_size=1024,
                 enable_alerts=True):
        
        gr.top_block.__init__(self, "Live Jamming Detection")
        
        print("🔧 Initializing Live Jamming Detection...")
        print(f"   Center Frequency: {center_freq/1e6:.1f} MHz")
        print(f"   Sample Rate: {sample_rate/1e6:.1f} MHz")
        print(f"   Gain: {gain} dB")
        print(f"   FFT Size: {fft_size}")
        
        # Parameters
        self.center_freq = center_freq
        self.sample_rate = sample_rate
        self.gain = gain
        self.fft_size = fft_size
        
        # USRP Source
        self.usrp_source = uhd.usrp_source(
            ",".join(("", "")),
            uhd.stream_args(
                cpu_format="fc32",
                channels=list(range(0,1)),
            ),
        )
        self.usrp_source.set_samp_rate(sample_rate)
        self.usrp_source.set_center_freq(center_freq, 0)
        self.usrp_source.set_gain(gain, 0)
        self.usrp_source.set_antenna('RX2', 0)
        
        # Stream to Vector (for FFT)
        self.stream_to_vector = blocks.stream_to_vector(
            gr.sizeof_gr_complex*1, 
            fft_size
        )
        
        # FFT
        self.fft_block = fft.fft_vcc(
            fft_size, 
            True, 
            window.blackmanharris(fft_size), 
            True, 
            1
        )
        
        # Complex to Magnitude
        self.complex_to_mag = blocks.complex_to_mag(fft_size)
        
        # Convert to dB (log10)
        self.log10_block = blocks.nlog10_ff(10, fft_size, 0)
        
        # Our custom jamming detection block
        self.jamming_detector = JammingDetectionSink(
            vector_length=fft_size,
            enable_alerts=enable_alerts
        )
        
        # Connect the blocks
        self.connect(self.usrp_source, self.stream_to_vector)
        self.connect(self.stream_to_vector, self.fft_block)
        self.connect(self.fft_block, self.complex_to_mag)
        self.connect(self.complex_to_mag, self.log10_block)
        self.connect(self.log10_block, self.jamming_detector)
        
        print("✅ GNU Radio flowgraph created successfully")

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    print("\n⚠️  Received interrupt signal. Stopping detection...")
    global detection_running
    detection_running = False

def main():
    """Main function for live USRP detection."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Live Real-time Jamming Detection with USRP',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage with default settings
    python3 live_usrp_detection.py
    
    # Custom frequency and gain
    python3 live_usrp_detection.py --freq 2.4e9 --gain 40
    
    # Disable alerts for testing
    python3 live_usrp_detection.py --no-alerts
    
    # Different sample rate
    python3 live_usrp_detection.py --sample-rate 2e6
        """
    )
    
    parser.add_argument('--freq', type=float, default=915e6,
                       help='Center frequency in Hz (default: 915 MHz)')
    parser.add_argument('--sample-rate', type=float, default=1e6,
                       help='Sample rate in Hz (default: 1 MHz)')
    parser.add_argument('--gain', type=float, default=30,
                       help='RX gain in dB (default: 30 dB)')
    parser.add_argument('--fft-size', type=int, default=1024,
                       help='FFT size (default: 1024)')
    parser.add_argument('--no-alerts', action='store_true',
                       help='Disable alert system')
    parser.add_argument('--duration', type=int, default=0,
                       help='Duration in seconds (0 = indefinite)')
    
    args = parser.parse_args()
    
    print("🚀 LIVE USRP JAMMING DETECTION SYSTEM")
    print("=" * 60)
    print(f"Center Frequency: {args.freq/1e6:.1f} MHz")
    print(f"Sample Rate: {args.sample_rate/1e6:.1f} MHz")
    print(f"Gain: {args.gain} dB")
    print(f"FFT Size: {args.fft_size}")
    print(f"Alerts: {'Disabled' if args.no_alerts else 'Enabled'}")
    if args.duration > 0:
        print(f"Duration: {args.duration} seconds")
    else:
        print("Duration: Indefinite (Ctrl+C to stop)")
    print("=" * 60)
    
    # Set up signal handler for graceful shutdown
    global detection_running
    detection_running = True
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Create and start the flowgraph
        tb = LiveJammingDetection(
            center_freq=args.freq,
            sample_rate=args.sample_rate,
            gain=args.gain,
            fft_size=args.fft_size,
            enable_alerts=not args.no_alerts
        )
        
        print("\n🔄 Starting signal acquisition and detection...")
        print("📡 Monitoring RF spectrum for jamming attacks...")
        print("⚠️  Press Ctrl+C to stop\n")
        
        # Start the flowgraph
        tb.start()
        
        # Wait for specified duration or until interrupted
        start_time = time.time()
        try:
            while detection_running:
                if args.duration > 0 and (time.time() - start_time) >= args.duration:
                    print(f"\n⏰ Duration of {args.duration} seconds completed")
                    break
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        
        # Stop the flowgraph
        print("\n🛑 Stopping detection system...")
        tb.stop()
        tb.wait()
        
        print("✅ Detection system stopped successfully")
        
    except Exception as e:
        print(f"\n❌ Error during live detection: {e}")
        print("💡 Troubleshooting tips:")
        print("   1. Check USRP connection: uhd_find_devices")
        print("   2. Verify antenna is connected")
        print("   3. Try different frequency/gain settings")
        print("   4. Check USB cable and power supply")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
