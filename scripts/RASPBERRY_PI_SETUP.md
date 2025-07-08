# Raspberry Pi Setup and Testing Scripts

This directory contains scripts to set up and test the SDR jamming detection system on a Raspberry Pi 3.

## Files

### 1. `setup_rpi_environment.sh`
**Purpose**: Complete environment setup for Raspberry Pi 3  
**Usage**: Run this script first to install all dependencies

```bash
# On Raspberry Pi 3:
bash setup_rpi_environment.sh
```

**What it installs:**
- TensorFlow Lite Runtime (optimized for ARM)
- GNU Radio with USRP support
- UHD drivers for USRP 2901R
- Python ML packages (NumPy, SciPy, LibROSA, etc.)
- System optimizations for real-time performance
- USB permissions for USRP access

**After installation:** Reboot required!

### 2. `rpi_tflite_test.py`
**Purpose**: Test TFLite model accuracy and inference speed  
**Usage**: Run after setup to evaluate model performance

```bash
# Basic test with default dataset
python3 rpi_tflite_test.py

# Test with specific dataset
python3 rpi_tflite_test.py --dataset Dataset/testv1_Higher_Gain

# Test with limited samples for quick check
python3 rpi_tflite_test.py --samples 100

# Test multiple datasets
python3 rpi_tflite_test.py --datasets Dataset/test Dataset/testv1_Higher_Gain Dataset/testv2_Lower_Gain

# Custom model and config paths
python3 rpi_tflite_test.py --model model/custom_model.tflite --config preprocessed_data/config.pkl
```

**Output includes:**
- Overall accuracy and per-class accuracy
- Inference timing (average, std dev)
- Throughput (samples per second)
- System information
- Detailed per-class performance

## Complete Setup Process

### On Your Development Machine:
1. Clone this repository
2. Ensure the TFLite model is in `model/jamming_detector_lightweight.tflite`
3. Transfer the entire project to Raspberry Pi

### On Raspberry Pi 3:
1. Run the setup script:
   ```bash
   bash setup_rpi_environment.sh
   ```
2. Reboot when prompted:
   ```bash
   sudo reboot
   ```
3. Test the installation:
   ```bash
   python3 ~/jamming_detection_sdr/scripts/test_installation.py
   ```
4. Connect USRP 2901R and verify:
   ```bash
   uhd_find_devices
   ```
5. Test the model:
   ```bash
   cd /path/to/your/project
   python3 scripts/rpi_tflite_test.py
   ```

## Expected Performance

### Raspberry Pi 3 Model B+ Expected Results:
- **Inference Time**: ~10-50ms per sample (depending on model complexity)
- **Throughput**: ~20-100 samples/second
- **Memory Usage**: <500MB RAM
- **Model Size**: ~1-5MB (TFLite compressed)

### Test Datasets Expected Accuracy:
- **Standard Test**: ~96.57%
- **Higher Gain**: ~100%
- **Lower Gain**: ~33.86% (known limitation)

## Troubleshooting

### Common Issues:

1. **Import Error for TensorFlow Lite**:
   ```bash
   pip3 install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-linux_armv7l.whl
   ```

2. **USRP Not Detected**:
   ```bash
   # Check USB connection
   lsusb
   
   # Check UHD devices
   uhd_find_devices
   
   # Check user permissions
   groups $USER  # Should include 'dialout'
   ```

3. **LibROSA Installation Issues**:
   ```bash
   sudo apt install libsndfile1-dev
   pip3 install librosa
   ```

4. **Memory Issues**:
   - Enable swap file
   - Increase GPU memory split to give more RAM to system
   - Test with fewer samples using `--samples` parameter

5. **Real-time Performance Issues**:
   - Check CPU governor: `cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor`
   - Should show "performance"
   - Reduce other running processes

### Performance Optimization Tips:

1. **For faster inference**:
   - Use fewer test samples during development
   - Consider INT8 quantization if accuracy allows
   - Close unnecessary services

2. **For GNU Radio**:
   - Use lower sample rates when possible
   - Optimize buffer sizes
   - Consider external USB hub for USRP power

3. **Monitoring**:
   ```bash
   # Monitor CPU and memory
   htop
   
   # Monitor temperature
   vcgencmd measure_temp
   
   # Check USB devices
   lsusb
   ```

## Hardware Requirements

### Minimum:
- Raspberry Pi 3 Model B+ (1GB RAM)
- 16GB microSD card (Class 10)
- USRP 2901R
- Stable 5V power supply (2.5A+)

### Recommended:
- Raspberry Pi 4 (2GB+ RAM) for better performance
- 32GB microSD card
- Active cooling (heatsink + fan)
- External powered USB hub for USRP

## Next Steps

After successful setup and testing:
1. Integrate with your GNU Radio flowgraphs
2. Implement real-time data collection
3. Add automated logging and monitoring
4. Consider edge deployment optimizations
