# Real-time Jamming Detection System

This directory contains scripts for real-time jamming detection using USRP 2901R and GNU Radio.

## Files Overview

### 1. `simple_real_time_detection.py`
**Recommended for most users** - Simple command-line real-time detection
- Easy to run and debug
- Clear terminal output with color coding
- Minimal dependencies
- Perfect for headless operation

### 2. `real_time_jamming_detection.py`
Advanced GUI-based real-time detection with PyQt5
- Full graphical interface
- Spectrum visualization
- System monitoring
- Real-time performance metrics

### 3. `real_time_jamming_detection.grc`
GNU Radio Companion flowgraph file
- Can be opened in GNU Radio Companion
- Visual flowgraph editing
- Easy parameter modification

## System Requirements

### Hardware
- **USRP 2901R** (or compatible)
- **Computer with USB 3.0** (recommended)
- **Stable power supply** for USRP

### Software
- **GNU Radio 3.8+** with UHD drivers
- **TensorFlow Lite Runtime** (or full TensorFlow)
- **Python 3.7+** with NumPy, SciPy
- **LibROSA** for audio processing

## Quick Start Guide

### 1. Setup (if not already done)
```bash
# Install GNU Radio and UHD
sudo apt install gnuradio uhd-host

# Download UHD images
sudo uhd_images_downloader

# Install Python dependencies
pip3 install numpy scipy librosa tensorflow
# OR for lighter install:
pip3 install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-linux_armv7l.whl
```

### 2. Connect USRP
```bash
# Connect USRP 2901R via USB
# Verify connection
uhd_find_devices

# Test USRP
uhd_usrp_probe
```

### 3. Run Detection System
```bash
# Simple command-line version (recommended)
python3 simple_real_time_detection.py

# Advanced GUI version
python3 real_time_jamming_detection.py

# Or open in GNU Radio Companion
gnuradio-companion real_time_jamming_detection.grc
```

## System Configuration

### Default Parameters
- **Center Frequency**: 915 MHz (ISM band)
- **Bandwidth**: 1 MHz
- **Gain**: 70 dB
- **FFT Size**: 1024 points
- **Downsample Factor**: 40 (1024 → 25.6 → 25 points)
- **Signal Length**: 1000 points per prediction

### Signal Processing Chain
```
USRP → Stream to Vector → FFT → Magnitude → Log10 → Downsample → Mel Transform → TFLite Model
```

1. **USRP 2901R**: Receives RF signals at 915 MHz
2. **FFT**: Converts time domain to frequency domain (1024 points)
3. **Magnitude**: Extracts power spectrum
4. **Log10**: Converts to dB scale
5. **Downsample**: Takes mean of every 40 points (1024 → ~25 points)
6. **Accumulate**: Collects 1000 points to form complete signal
7. **Mel Transform**: Converts to mel spectrogram using preprocessing
8. **TFLite Model**: Predicts jamming type

## Expected Output

### Normal Operation
```
[10:15:32] Prediction #1: Normal (Confidence: 94.2%)
[10:15:33] Prediction #2: Normal (Confidence: 91.8%)
[10:15:34] Prediction #3: Normal (Confidence: 96.1%)
```

### Jamming Detected
```
[10:15:35] Prediction #4: Constant Jammer (Confidence: 87.4%)
[10:15:36] Prediction #5: Constant Jammer (Confidence: 92.1%)
[10:15:37] Prediction #6: Periodic Jammer (Confidence: 89.3%)
```

## Performance Expectations

### Timing
- **Prediction Rate**: ~1-2 predictions per second
- **Latency**: ~500ms from signal to prediction
- **Throughput**: Real-time processing of 1 MHz bandwidth

### Accuracy (based on test data)
- **Normal Signals**: ~96% accuracy
- **Constant Jammer**: ~95% accuracy  
- **Periodic Jammer**: ~94% accuracy

## Troubleshooting

### Common Issues

1. **USRP Not Found**
   ```bash
   # Check USB connection
   lsusb | grep Ettus
   
   # Check UHD installation
   uhd_find_devices
   
   # Check permissions
   sudo usermod -a -G dialout $USER
   # Then logout and login again
   ```

2. **GNU Radio Import Error**
   ```bash
   # Check GNU Radio installation
   python3 -c "import gnuradio"
   
   # If failed, reinstall
   sudo apt install gnuradio-dev
   ```

3. **TensorFlow Lite Error**
   ```bash
   # Check TFLite installation
   python3 -c "import tflite_runtime.interpreter"
   
   # Or try full TensorFlow
   python3 -c "import tensorflow as tf"
   ```

4. **Model File Not Found**
   ```bash
   # Check model exists
   ls -la model/jamming_detector_lightweight.tflite
   
   # Check config exists
   ls -la preprocessed_data/config.pkl
   ```

5. **Poor Performance**
   - Check USRP gain settings (try 60-80 dB)
   - Verify antenna connection (use RX2)
   - Check for interference in 915 MHz band
   - Monitor CPU usage during operation

### Performance Optimization

1. **Reduce CPU Usage**
   ```bash
   # Lower prediction rate by increasing buffer size
   # Edit signal_length parameter in the script
   ```

2. **Improve Accuracy**
   ```bash
   # Adjust gain based on signal environment
   # Use proper antenna for 915 MHz
   # Minimize interference sources
   ```

3. **Real-time Operation**
   ```bash
   # Set CPU governor to performance
   echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
   
   # Increase process priority
   sudo nice -n -10 python3 simple_real_time_detection.py
   ```

## Customization

### Changing Frequency
Edit the center_freq parameter in the script:
```python
self.center_freq = 433000000  # 433 MHz
```

### Adjusting Sensitivity
Modify the gain parameter:
```python
self.gain = 60  # Lower gain for strong signals
self.gain = 80  # Higher gain for weak signals
```

### Different Bandwidth
Change the sample rate:
```python
self.samp_rate = 2000000  # 2 MHz bandwidth
```

## Integration with Other Systems

### Logging to File
```python
# Add to the prediction display code
with open('detection_log.txt', 'a') as f:
    f.write(f"{timestamp},{class_name},{confidence:.4f}\n")
```

### Network Alerts
```python
# Send alerts via network
import requests
if class_name != "Normal":
    requests.post('http://your-server.com/alert', 
                  json={'type': class_name, 'confidence': confidence})
```

### Database Storage
```python
# Store results in database
import sqlite3
conn = sqlite3.connect('detection_results.db')
cursor = conn.cursor()
cursor.execute("INSERT INTO detections VALUES (?, ?, ?)", 
               (timestamp, class_name, confidence))
conn.commit()
```

## License and Support

This system is part of the SDR Jamming Detection project. For issues and questions, please refer to the main project documentation or create an issue in the repository.

## Next Steps

1. **Calibrate for your environment** - Run with known signals
2. **Optimize parameters** - Adjust gain and thresholds
3. **Add logging** - Store detection results
4. **Integrate alerts** - Set up automated responses
5. **Monitor performance** - Track accuracy over time
