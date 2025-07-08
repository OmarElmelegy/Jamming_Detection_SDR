#!/bin/bash

# Raspberry Pi 3 Environment Setup Script for SDR Jamming Detection
# This script sets up a Raspberry Pi 3 for running TensorFlow Lite models
# and GNU Radio with USRP 2901R support

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warning() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}"
}

info() {
    echo -e "${BLUE}[INFO] $1${NC}"
}

# Check if running on Raspberry Pi
check_raspberry_pi() {
    log "Checking if running on Raspberry Pi..."
    
    if [ -f /proc/device-tree/model ]; then
        DEVICE_MODEL=$(cat /proc/device-tree/model)
        if [[ $DEVICE_MODEL == *"Raspberry Pi"* ]]; then
            log "✅ Detected: $DEVICE_MODEL"
            
            # Check for Raspberry Pi 3
            if [[ $DEVICE_MODEL == *"Raspberry Pi 3"* ]]; then
                log "✅ Raspberry Pi 3 detected - optimal for this setup"
            else
                warning "This script is optimized for Raspberry Pi 3. You may experience different performance."
            fi
        else
            error "Not running on a Raspberry Pi!"
            exit 1
        fi
    else
        error "Cannot detect device type. This script is designed for Raspberry Pi."
        exit 1
    fi
}

# Check available memory and storage
check_system_resources() {
    log "Checking system resources..."
    
    # Check memory
    TOTAL_MEM=$(grep MemTotal /proc/meminfo | awk '{print $2}')
    TOTAL_MEM_MB=$((TOTAL_MEM / 1024))
    log "Total Memory: ${TOTAL_MEM_MB} MB"
    
    if [ $TOTAL_MEM_MB -lt 900 ]; then
        warning "Low memory detected. Consider enabling swap or using a Pi with more RAM."
    fi
    
    # Check storage
    AVAILABLE_SPACE=$(df -h / | awk 'NR==2{print $4}')
    log "Available Storage: $AVAILABLE_SPACE"
    
    # Check if we have at least 2GB free
    AVAILABLE_KB=$(df / | awk 'NR==2{print $4}')
    if [ $AVAILABLE_KB -lt 2097152 ]; then  # 2GB in KB
        warning "Low storage space. You may need to free up space."
    fi
}

# Update system packages
update_system() {
    log "Updating system packages..."
    
    sudo apt update
    sudo apt upgrade -y
    
    log "✅ System updated successfully"
}

# Install basic dependencies
install_basic_dependencies() {
    log "Installing basic dependencies..."
    
    sudo apt install -y \
        python3 \
        python3-pip \
        python3-dev \
        python3-setuptools \
        build-essential \
        cmake \
        git \
        wget \
        curl \
        nano \
        htop \
        usbutils \
        libusb-1.0-0-dev \
        libfftw3-dev \
        libboost-all-dev \
        libasound2-dev \
        portaudio19-dev \
        libportaudio2 \
        libportaudiocpp0 \
        libatlas-base-dev \
        libopenblas-dev \
        gfortran \
        libblas-dev \
        liblapack-dev
    
    log "✅ Basic dependencies installed"
}

# Install Python packages for machine learning
install_python_ml_packages() {
    log "Installing Python machine learning packages..."
    
    # Upgrade pip first
    python3 -m pip install --upgrade pip
    
    # Install TensorFlow Lite Runtime (lighter than full TensorFlow)
    log "Installing TensorFlow Lite Runtime..."
    
    # For ARM7 (Raspberry Pi 3)
    python3 -m pip install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-linux_armv7l.whl
    
    # Alternative: Install full TensorFlow (commented out as it's heavier)
    # python3 -m pip install tensorflow==2.8.0
    
    # Install other ML packages
    log "Installing other Python packages..."
    python3 -m pip install \
        numpy==1.21.6 \
        scipy==1.7.3 \
        scikit-learn==1.0.2 \
        matplotlib==3.5.3 \
        librosa==0.9.2 \
        soundfile==0.10.3.post1 \
        pandas==1.3.5 \
        seaborn==0.11.2
    
    log "✅ Python ML packages installed"
}

# Install GNU Radio
install_gnuradio() {
    log "Installing GNU Radio..."
    
    # Install GNU Radio from apt (stable version)
    sudo apt install -y gnuradio gnuradio-dev
    
    # Install additional GNU Radio packages
    sudo apt install -y \
        gr-osmosdr \
        gr-iio \
        gr-fcdproplus \
        soapysdr-tools \
        soapysdr-module-all
    
    # Verify installation
    if command -v gnuradio-companion &> /dev/null; then
        GR_VERSION=$(gnuradio-config-info --version)
        log "✅ GNU Radio $GR_VERSION installed successfully"
    else
        error "GNU Radio installation failed"
        exit 1
    fi
}

# Install UHD (USRP Hardware Driver)
install_uhd() {
    log "Installing UHD for USRP support..."
    
    # Install UHD
    sudo apt install -y libuhd-dev uhd-host
    
    # Download UHD images (this might take a while)
    log "Downloading UHD FPGA images (this may take several minutes)..."
    sudo uhd_images_downloader
    
    # Verify UHD installation
    if command -v uhd_find_devices &> /dev/null; then
        UHD_VERSION=$(uhd_config_info --version)
        log "✅ UHD $UHD_VERSION installed successfully"
    else
        error "UHD installation failed"
        exit 1
    fi
}

# Configure USB permissions for USRP
configure_usb_permissions() {
    log "Configuring USB permissions for USRP..."
    
    # Add user to dialout group for USB access
    sudo usermod -a -G dialout $USER
    
    # Create udev rules for USRP
    sudo tee /etc/udev/rules.d/10-usrp.rules > /dev/null <<EOF
# USRP1
SUBSYSTEM=="usb", ATTR{idVendor}=="fffe", ATTR{idProduct}=="0002", GROUP="dialout", MODE="0664"
# USRP B200/B210/B200mini/B205mini
SUBSYSTEM=="usb", ATTR{idVendor}=="2500", ATTR{idProduct}=="0020", GROUP="dialout", MODE="0664"
SUBSYSTEM=="usb", ATTR{idVendor}=="2500", ATTR{idProduct}=="0021", GROUP="dialout", MODE="0664"
SUBSYSTEM=="usb", ATTR{idVendor}=="2500", ATTR{idProduct}=="0022", GROUP="dialout", MODE="0664"
SUBSYSTEM=="usb", ATTR{idVendor}=="2500", ATTR{idProduct}=="0023", GROUP="dialout", MODE="0664"
# USRP2 / USRP N200/N210
SUBSYSTEM=="net", ATTR{address}=="00:50:c2:85:*", NAME="usrp2"
EOF
    
    # Reload udev rules
    sudo udevadm control --reload-rules
    sudo udevadm trigger
    
    log "✅ USB permissions configured"
}

# Optimize system for real-time performance
optimize_system() {
    log "Optimizing system for real-time performance..."
    
    # Increase USB buffer sizes
    echo 'net.core.rmem_default = 262144' | sudo tee -a /etc/sysctl.conf
    echo 'net.core.rmem_max = 16777216' | sudo tee -a /etc/sysctl.conf
    echo 'net.core.wmem_default = 262144' | sudo tee -a /etc/sysctl.conf
    echo 'net.core.wmem_max = 16777216' | sudo tee -a /etc/sysctl.conf
    
    # GPU memory split (more RAM for system)
    echo 'gpu_mem=16' | sudo tee -a /boot/config.txt
    
    # Enable performance governor
    echo 'performance' | sudo tee /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || true
    
    log "✅ System optimized"
}

# Create project directory structure
setup_project_structure() {
    log "Setting up project directory structure..."
    
    PROJECT_DIR="$HOME/jamming_detection_sdr"
    
    if [ ! -d "$PROJECT_DIR" ]; then
        mkdir -p "$PROJECT_DIR"/{data,models,scripts,logs,gnuradio_flowgraphs}
        
        log "✅ Created project directory: $PROJECT_DIR"
        
        # Create README
        cat > "$PROJECT_DIR/README.md" << EOF
# SDR Jamming Detection on Raspberry Pi

This directory contains the jamming detection project files.

## Directory Structure
- \`data/\` - Test data and samples
- \`models/\` - TensorFlow Lite models
- \`scripts/\` - Python scripts for testing and evaluation
- \`logs/\` - Log files from tests
- \`gnuradio_flowgraphs/\` - GNU Radio flowgraph files

## Usage
1. Place your TFLite model in the \`models/\` directory
2. Place test data in the \`data/\` directory
3. Run tests using scripts in the \`scripts/\` directory

## Quick Test
\`\`\`bash
# Test USRP connection
uhd_find_devices

# Test TFLite model (when available)
python3 scripts/rpi_tflite_test.py --model models/your_model.tflite
\`\`\`
EOF
        
        info "Project structure created at: $PROJECT_DIR"
    else
        warning "Project directory already exists: $PROJECT_DIR"
    fi
}

# Create test script for verifying installation
create_test_script() {
    log "Creating installation test script..."
    
    PROJECT_DIR="$HOME/jamming_detection_sdr"
    TEST_SCRIPT="$PROJECT_DIR/scripts/test_installation.py"
    
    cat > "$TEST_SCRIPT" << 'EOF'
#!/usr/bin/env python3
"""
Installation Test Script for Raspberry Pi SDR Setup

This script tests all the installed components to ensure they work correctly.
"""

import sys
import subprocess
import importlib

def test_python_packages():
    """Test if required Python packages are installed."""
    print("🐍 Testing Python packages...")
    
    packages = [
        'numpy',
        'scipy', 
        'sklearn',
        'matplotlib',
        'librosa',
        'pandas'
    ]
    
    # Test TensorFlow Lite
    try:
        import tflite_runtime.interpreter
        print("  ✅ TensorFlow Lite Runtime")
    except ImportError:
        try:
            import tensorflow as tf
            print("  ✅ TensorFlow")
        except ImportError:
            print("  ❌ Neither TensorFlow nor TFLite Runtime found")
            return False
    
    # Test other packages
    for package in packages:
        try:
            importlib.import_module(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package}")
            return False
    
    return True

def test_gnuradio():
    """Test GNU Radio installation."""
    print("\n📻 Testing GNU Radio...")
    
    try:
        result = subprocess.run(['gnuradio-config-info', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  ✅ GNU Radio {result.stdout.strip()}")
            return True
        else:
            print("  ❌ GNU Radio not working")
            return False
    except FileNotFoundError:
        print("  ❌ GNU Radio not installed")
        return False

def test_uhd():
    """Test UHD installation."""
    print("\n📡 Testing UHD...")
    
    try:
        result = subprocess.run(['uhd_config_info', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  ✅ UHD {result.stdout.strip()}")
            
            # Try to find USRP devices
            print("  🔍 Searching for USRP devices...")
            result = subprocess.run(['uhd_find_devices'], 
                                  capture_output=True, text=True, timeout=10)
            if "No UHD Devices Found" in result.stdout:
                print("  ⚠️  No USRP devices found (this is normal if no device is connected)")
            else:
                print("  ✅ USRP devices detected!")
                print(f"      {result.stdout.strip()}")
            
            return True
        else:
            print("  ❌ UHD not working")
            return False
    except FileNotFoundError:
        print("  ❌ UHD not installed")
        return False
    except subprocess.TimeoutExpired:
        print("  ⚠️  UHD device search timed out")
        return True

def test_system_info():
    """Display system information."""
    print("\n💻 System Information:")
    
    try:
        with open('/proc/device-tree/model', 'r') as f:
            model = f.read().strip('\x00')
        print(f"  Device: {model}")
    except:
        print("  Device: Unknown")
    
    with open('/proc/meminfo', 'r') as f:
        for line in f:
            if 'MemTotal' in line:
                mem_kb = int(line.split()[1])
                mem_mb = mem_kb // 1024
                print(f"  Memory: {mem_mb} MB")
                break
    
    # Check available disk space
    result = subprocess.run(['df', '-h', '/'], capture_output=True, text=True)
    if result.returncode == 0:
        lines = result.stdout.strip().split('\n')
        if len(lines) > 1:
            parts = lines[1].split()
            print(f"  Disk Space: {parts[3]} available")

def main():
    """Run all tests."""
    print("🧪 Raspberry Pi SDR Installation Test")
    print("=" * 50)
    
    all_passed = True
    
    # Run tests
    all_passed &= test_python_packages()
    all_passed &= test_gnuradio()
    all_passed &= test_uhd()
    test_system_info()
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! Your Raspberry Pi is ready for SDR jamming detection.")
        print("\nNext steps:")
        print("1. Clone your project repository")
        print("2. Copy your TFLite model to the models/ directory")
        print("3. Copy test data to the data/ directory")
        print("4. Run: python3 scripts/rpi_tflite_test.py")
    else:
        print("❌ Some tests failed. Please check the installation.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
EOF
    
    chmod +x "$TEST_SCRIPT"
    log "✅ Test script created: $TEST_SCRIPT"
}

# Main setup function
main() {
    log "🚀 Starting Raspberry Pi 3 SDR Environment Setup"
    log "This script will install and configure:"
    log "  - TensorFlow Lite Runtime"
    log "  - GNU Radio with USRP support"
    log "  - UHD drivers"
    log "  - Python ML packages"
    log "  - System optimizations"
    
    read -p "Continue with installation? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log "Installation cancelled."
        exit 0
    fi
    
    # Run setup steps
    check_raspberry_pi
    check_system_resources
    update_system
    install_basic_dependencies
    install_python_ml_packages
    install_gnuradio
    install_uhd
    configure_usb_permissions
    optimize_system
    setup_project_structure
    create_test_script
    
    log "🎉 Setup completed successfully!"
    log ""
    log "📋 IMPORTANT NOTES:"
    log "1. Please reboot your Raspberry Pi for all changes to take effect"
    log "2. After reboot, test the installation by running:"
    log "   python3 ~/jamming_detection_sdr/scripts/test_installation.py"
    log "3. Connect your USRP 2901R and test with: uhd_find_devices"
    log "4. Clone your project repository to get the actual model and test data"
    log ""
    log "🔧 Project directory created at: ~/jamming_detection_sdr"
    
    warning "REBOOT REQUIRED - Please run 'sudo reboot' to complete setup"
}

# Run main function
main "$@"
