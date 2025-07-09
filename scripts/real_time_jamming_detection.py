#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Real-time Jamming Detection with USRP 2901R
# Author: Omar
# Description: Real-time jamming detection using USRP 2901R and TensorFlow Lite
# Generated: Wed Jul  9 2025
#

import os
import sys
sys.path.append(os.environ.get('GRC_HIER_PATH', os.path.expanduser('~/.grc_gnuradio')))

from distutils.version import StrictVersion

if StrictVersion("4.5.0") <= StrictVersion(Qt.qVersion()) < StrictVersion("5.0.0"):
    style = gr.prefs().get_string('qtgui', 'style', 'raster')
    Qt.QApplication.setGraphicsSystem(style)

from PyQt5 import Qt
from PyQt5.QtCore import QObject, pyqtSlot
from gnuradio import qtgui
from gnuradio.filter import firdes
from gnuradio import blocks
from gnuradio import fft
from gnuradio import gr
from gnuradio import uhd
import time
import sip
from gnuradio import eng_notation
from gnuradio import zeromq
import numpy as np
import threading
import pickle
import sys
import os

# Import TensorFlow Lite
try:
    import tensorflow as tf
    TFLITE_AVAILABLE = True
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
        TFLITE_AVAILABLE = True
        USE_TFLITE_RUNTIME = True
    except ImportError:
        print("Error: TensorFlow Lite not available!")
        sys.exit(1)

# Import preprocessing
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from scripts.preprocess import RSSSIToMelSpectrogram

class JammingDetectionBlock(gr.sync_block):
    """
    Custom GNU Radio block for real-time jamming detection.
    """
    
    def __init__(self, model_path, config_path, vector_length=1024):
        gr.sync_block.__init__(
            self,
            name="jamming_detection",
            in_sig=[(np.float32, vector_length)],
            out_sig=None
        )
        
        self.vector_length = vector_length
        self.downsample_factor = 40  # Take mean of every 40 points
        self.points_per_signal = int(vector_length / self.downsample_factor)  # 25.6 -> 25
        self.signal_length = 1000
        self.class_names = ['Normal', 'Constant Jammer', 'Periodic Jammer']
        
        # Buffer for accumulating signal points
        self.signal_buffer = []
        self.prediction_callback = None
        
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
        
        print(f"Jamming Detection Block initialized:")
        print(f"  Vector length: {self.vector_length}")
        print(f"  Downsample factor: {self.downsample_factor}")
        print(f"  Points per vector: {self.points_per_signal}")
        print(f"  Signal length: {self.signal_length}")
    
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
            
            print(f"Model loaded: {model_path}")
            print(f"Input shape: {self.input_details[0]['shape']}")
            print(f"Output shape: {self.output_details[0]['shape']}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def load_config(self, config_path):
        """Load preprocessing configuration."""
        try:
            with open(config_path, 'rb') as f:
                self.config = pickle.load(f)
            print(f"Config loaded: {config_path}")
        except Exception as e:
            print(f"Warning: Could not load config: {e}")
            self.config = {
                'sampling_rate': 1000,
                'n_fft': 256,
                'hop_length': 128,
                'n_mels': 64,
                'sequence_length': 1000
            }
    
    def set_prediction_callback(self, callback):
        """Set callback function for predictions."""
        self.prediction_callback = callback
    
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
            print(f"Error in prediction: {e}")
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
                
                # Call callback if set
                if self.prediction_callback:
                    self.prediction_callback(predicted_class, confidence, self.class_names)
        
        return len(input_items[0])


class JammingDetectionGUI(gr.top_block, Qt.QWidget):
    """
    Main GUI application for real-time jamming detection.
    """
    
    def __init__(self):
        gr.top_block.__init__(self, "Real-time Jamming Detection")
        Qt.QWidget.__init__(self)
        self.setWindowTitle("Real-time Jamming Detection")
        self.qtgui_sink_x_0 = None
        self.prediction_label = None
        self.confidence_label = None
        self.status_label = None
        
        # Variables
        self.samp_rate = 1000000  # 1 MHz
        self.center_freq = 915000000  # 915 MHz
        self.gain = 70  # 70 dB
        self.vector_length = 1024
        self.fft_size = 1024
        
        # Model paths
        self.model_path = "model/jamming_detector_lightweight.tflite"
        self.config_path = "preprocessed_data/config.pkl"
        
        # Build flowgraph
        self.build_flowgraph()
        self.build_gui()
        
        # Start prediction updates
        self.prediction_timer = Qt.QTimer()
        self.prediction_timer.timeout.connect(self.update_display)
        self.prediction_timer.start(100)  # Update every 100ms
        
        # Latest prediction
        self.latest_prediction = None
        self.latest_confidence = 0.0
        self.latest_class_name = "Unknown"
        
        print("GUI initialized successfully!")
    
    def build_flowgraph(self):
        """Build the GNU Radio flowgraph."""
        
        # USRP Source
        self.uhd_usrp_source_0 = uhd.usrp_source(
            ",".join(("", "")),
            uhd.stream_args(
                cpu_format="fc32",
                channels=list(range(0, 1)),
            ),
        )
        self.uhd_usrp_source_0.set_samp_rate(self.samp_rate)
        self.uhd_usrp_source_0.set_center_freq(self.center_freq, 0)
        self.uhd_usrp_source_0.set_gain(self.gain, 0)
        self.uhd_usrp_source_0.set_antenna('RX2', 0)
        
        # Stream to Vector (for FFT)
        self.blocks_stream_to_vector_0 = blocks.stream_to_vector(
            gr.sizeof_gr_complex * 1, self.fft_size
        )
        
        # FFT
        self.fft_vxx_0 = fft.fft_vcc(self.fft_size, True, (), True, 1)
        
        # Complex to Magnitude
        self.blocks_complex_to_mag_0 = blocks.complex_to_mag(self.vector_length)
        
        # Log10
        self.blocks_nlog10_ff_0 = blocks.nlog10_ff(10, self.vector_length, 0)
        
        # Custom jamming detection block
        self.jamming_detector = JammingDetectionBlock(
            self.model_path, 
            self.config_path, 
            self.vector_length
        )
        self.jamming_detector.set_prediction_callback(self.on_prediction)
        
        # GUI Sink for visualization
        self.qtgui_vector_sink_f_0 = qtgui.vector_sink_f(
            self.vector_length,
            0,
            1.0,
            "Frequency Bin",
            "Magnitude (dB)",
            "FFT Spectrum",
            1  # Number of inputs
        )
        self.qtgui_vector_sink_f_0.set_update_time(0.1)
        self.qtgui_vector_sink_f_0.set_y_axis(-100, 0)
        self.qtgui_vector_sink_f_0.enable_autoscale(False)
        self.qtgui_vector_sink_f_0.enable_grid(False)
        self.qtgui_vector_sink_f_0.set_x_axis_units("")
        self.qtgui_vector_sink_f_0.set_y_axis_units("")
        self.qtgui_vector_sink_f_0.set_ref_level(0)
        
        labels = ['FFT', '', '', '', '', '', '', '', '', '']
        widths = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        colors = ["blue", "red", "green", "black", "cyan", "magenta", "yellow", "dark red", "dark green", "dark blue"]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        
        for i in range(1):
            if len(labels[i]) == 0:
                self.qtgui_vector_sink_f_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_vector_sink_f_0.set_line_label(i, labels[i])
            self.qtgui_vector_sink_f_0.set_line_width(i, widths[i])
            self.qtgui_vector_sink_f_0.set_line_color(i, colors[i])
            self.qtgui_vector_sink_f_0.set_line_alpha(i, alphas[i])
        
        self.qtgui_sink_x_0 = sip.wrapinstance(self.qtgui_vector_sink_f_0.pyqwidget(), Qt.QWidget)
        
        # Connections
        self.connect((self.uhd_usrp_source_0, 0), (self.blocks_stream_to_vector_0, 0))
        self.connect((self.blocks_stream_to_vector_0, 0), (self.fft_vxx_0, 0))
        self.connect((self.fft_vxx_0, 0), (self.blocks_complex_to_mag_0, 0))
        self.connect((self.blocks_complex_to_mag_0, 0), (self.blocks_nlog10_ff_0, 0))
        self.connect((self.blocks_nlog10_ff_0, 0), (self.qtgui_vector_sink_f_0, 0))
        self.connect((self.blocks_nlog10_ff_0, 0), (self.jamming_detector, 0))
        
        print("Flowgraph built successfully!")
    
    def build_gui(self):
        """Build the user interface."""
        
        # Main layout
        self.top_layout = Qt.QVBoxLayout()
        self.top_widget = Qt.QWidget()
        self.top_widget.setLayout(self.top_layout)
        self.setCentralWidget(self.top_widget)
        
        # Control panel
        control_layout = Qt.QHBoxLayout()
        
        # System status
        status_group = Qt.QGroupBox("System Status")
        status_layout = Qt.QVBoxLayout()
        
        self.status_label = Qt.QLabel("Status: Initializing...")
        self.status_label.setStyleSheet("color: orange; font-weight: bold;")
        status_layout.addWidget(self.status_label)
        
        status_group.setLayout(status_layout)
        control_layout.addWidget(status_group)
        
        # USRP Settings
        usrp_group = Qt.QGroupBox("USRP Settings")
        usrp_layout = Qt.QVBoxLayout()
        
        freq_layout = Qt.QHBoxLayout()
        freq_layout.addWidget(Qt.QLabel("Center Freq:"))
        self.freq_label = Qt.QLabel(f"{self.center_freq/1e6:.1f} MHz")
        freq_layout.addWidget(self.freq_label)
        usrp_layout.addLayout(freq_layout)
        
        gain_layout = Qt.QHBoxLayout()
        gain_layout.addWidget(Qt.QLabel("Gain:"))
        self.gain_label = Qt.QLabel(f"{self.gain} dB")
        gain_layout.addWidget(self.gain_label)
        usrp_layout.addLayout(gain_layout)
        
        bw_layout = Qt.QHBoxLayout()
        bw_layout.addWidget(Qt.QLabel("Bandwidth:"))
        self.bw_label = Qt.QLabel(f"{self.samp_rate/1e6:.1f} MHz")
        bw_layout.addWidget(self.bw_label)
        usrp_layout.addLayout(bw_layout)
        
        usrp_group.setLayout(usrp_layout)
        control_layout.addWidget(usrp_group)
        
        # Prediction results
        prediction_group = Qt.QGroupBox("Jamming Detection")
        prediction_layout = Qt.QVBoxLayout()
        
        self.prediction_label = Qt.QLabel("Prediction: Unknown")
        self.prediction_label.setStyleSheet("font-size: 16px; font-weight: bold; color: blue;")
        prediction_layout.addWidget(self.prediction_label)
        
        self.confidence_label = Qt.QLabel("Confidence: 0.0%")
        self.confidence_label.setStyleSheet("font-size: 14px; color: gray;")
        prediction_layout.addWidget(self.confidence_label)
        
        prediction_group.setLayout(prediction_layout)
        control_layout.addWidget(prediction_group)
        
        self.top_layout.addLayout(control_layout)
        
        # Spectrum display
        if self.qtgui_sink_x_0:
            self.top_layout.addWidget(self.qtgui_sink_x_0)
        
        # Set window properties
        self.setWindowTitle("Real-time Jamming Detection with USRP 2901R")
        self.resize(1200, 800)
        
        # Update initial status
        self.update_status("Ready")
        
        print("GUI built successfully!")
    
    def on_prediction(self, predicted_class, confidence, class_names):
        """Callback for new predictions."""
        self.latest_prediction = predicted_class
        self.latest_confidence = confidence
        self.latest_class_name = class_names[predicted_class]
        
        # Print to console
        print(f"Prediction: {self.latest_class_name} (Confidence: {confidence:.2%})")
    
    def update_display(self):
        """Update the GUI display with latest predictions."""
        if self.latest_prediction is not None:
            # Update prediction label
            self.prediction_label.setText(f"Prediction: {self.latest_class_name}")
            
            # Update confidence label
            self.confidence_label.setText(f"Confidence: {self.latest_confidence:.1%}")
            
            # Color coding based on prediction
            if self.latest_class_name == "Normal":
                color = "green"
            elif self.latest_class_name == "Constant Jammer":
                color = "red"
            elif self.latest_class_name == "Periodic Jammer":
                color = "orange"
            else:
                color = "blue"
            
            self.prediction_label.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {color};")
            
            # Update confidence color based on confidence level
            if self.latest_confidence > 0.8:
                conf_color = "green"
            elif self.latest_confidence > 0.5:
                conf_color = "orange"
            else:
                conf_color = "red"
            
            self.confidence_label.setStyleSheet(f"font-size: 14px; color: {conf_color};")
    
    def update_status(self, status):
        """Update system status."""
        self.status_label.setText(f"Status: {status}")
        
        if status == "Ready":
            self.status_label.setStyleSheet("color: green; font-weight: bold;")
        elif status == "Running":
            self.status_label.setStyleSheet("color: blue; font-weight: bold;")
        elif status == "Error":
            self.status_label.setStyleSheet("color: red; font-weight: bold;")
        else:
            self.status_label.setStyleSheet("color: orange; font-weight: bold;")
    
    def closeEvent(self, event):
        """Handle window close event."""
        self.stop()
        self.wait()
        event.accept()


def main(top_block_cls=JammingDetectionGUI, options=None):
    """Main function."""
    
    # Check if model files exist
    if not os.path.exists("model/jamming_detector_lightweight.tflite"):
        print("Error: TensorFlow Lite model not found!")
        print("Please ensure 'model/jamming_detector_lightweight.tflite' exists.")
        return 1
    
    if not os.path.exists("preprocessed_data/config.pkl"):
        print("Warning: Config file not found, using defaults.")
    
    # Create QApplication
    qapp = Qt.QApplication(sys.argv)
    
    # Create and start the application
    tb = top_block_cls()
    
    def signal_handler(sig, frame):
        print("Interrupted")
        tb.stop()
        tb.wait()
        qapp.quit()
    
    import signal
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start the flowgraph
    tb.start()
    tb.update_status("Running")
    
    # Show GUI
    tb.show()
    
    # Start Qt event loop
    try:
        qapp.exec_()
    finally:
        tb.stop()
        tb.wait()
    
    return 0


if __name__ == '__main__':
    main()
