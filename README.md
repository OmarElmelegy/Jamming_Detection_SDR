# RF Jamming Detection with 2D CNN

## 🎯 Overview

RF jamming detection system using a 2D CNN architecture that operates on Mel spectrograms. This system achieves excellent performance on standard test data but exhibits performance degradation on lower gain signals, highlighting the challenges of RF signal classification across different operating conditions.

## 🏆 Performance

### Standard Operating Conditions
- **Standard Test Accuracy**: 96.57%
- **Higher Gain Test Accuracy**: 100.0%
- **Model Size**: ~1.4 MB (64,451 parameters)
- **Classes**: Normal, Constant Jammer (CJ), Periodic Jammer (PJ)

### ⚠️ Known Issues
- **Lower Gain Performance**: 33.86% accuracy (significant degradation)
  - Model exhibits strong bias toward Periodic Jammer predictions
  - Root cause: Poor signal separation in lower gain conditions (3.52 dB vs 26.65 dB in standard)
  - This represents a critical limitation for deployment in variable gain environments

## 📁 Project Structure

```
SDRv2/
├── 📋 README.md                # Project documentation
├── 📓 notebooks/               # Interactive workflow demonstrations
│   └── rf_jamming_detection_workflow.ipynb  # Complete workflow tutorial
├── 🐍 scripts/                 # Core implementation modules
│   ├── model.py               # Lightweight 2D CNN architecture
│   ├── train.py               # Training pipeline
│   ├── demo.py                # Real-time inference demo
│   ├── evaluate.py            # Model evaluation
│   ├── preprocess.py          # Data preprocessing pipeline
│   └── requirements.txt       # Python dependencies
├── 🤖 model/                   # Trained models
│   └── jamming_detector_lightweight_best.h5  # Current model with gain limitations
├── 📊 Dataset/                 # RSSI signal data
│   ├── training/              # Training data (Normal, CJ, PJ)
│   ├── test/                  # Standard test data
│   ├── testv1_Higher_Gain/    # Higher gain test data (100% accuracy)
│   └── testv2_Lower_Gain/     # Lower gain test data (33.86% accuracy - problematic)
├── 💾 preprocessed_data/       # Processed training data
│   ├── config.pkl             # Preprocessing configuration
│   ├── X_train.npy            # Training spectrograms
│   ├── X_test.npy             # Test spectrograms
│   ├── y_train.npy            # Training labels
│   └── y_test.npy             # Test labels
└── 📈 results/                 # Generated outputs and analysis
    ├── reports/               # Performance reports
    └── visualizations/        # Training plots and confusion matrices
```

## 🚀 Quick Start

### Installation & Verification
```bash
# 1. Clone or download the repository
cd /path/to/SDRv2

# 2. Install dependencies
pip install -r scripts/requirements.txt

# 3. Verify installation (optional but recommended)
python scripts/demo.py --samples 3
```

### Usage Options

#### Option 1: Interactive Tutorial (Recommended for Learning)
```bash
# Launch Jupyter notebook for complete workflow demonstration
jupyter notebook notebooks/rf_jamming_detection_workflow.ipynb
```

#### Option 2: Command Line Usage

##### Real-time Demo
```bash
# Test with sample data - works well on standard/higher gain data
python scripts/demo.py --samples 5
```

##### Model Evaluation
```bash
# Comprehensive evaluation across all gain levels
python scripts/evaluate.py
```

##### Data Preprocessing (if needed)
```bash
# Preprocess raw RSSI data to mel spectrograms
python scripts/preprocess.py
```

##### Model Training (if needed)
```bash
# Train new model (current approach has gain limitations)
python scripts/train.py --epochs 50 --batch-size 32
```

## 🧠 Model Architecture

### 2D CNN Architecture
- **Input**: Mel spectrograms (64×8×1) from 1000-sample RSSI sequences
- **Architecture**: 2 Conv2D blocks + Dense layers
- **Parameters**: 64,451 (lightweight design)
- **Performance**: Variable across gain conditions (96.57% standard, 33.86% lower gain)

### Data Pipeline
```
RSSI Signal [1000 samples] 
    ↓
STFT (n_fft=256, hop_length=128)
    ↓
Mel Filtering (n_mels=64)
    ↓
Mel Spectrogram [64×8×1]
    ↓
2D CNN Classification
```

## 📊 Key Features

- ✅ **Effective on Standard Conditions**: 96.57% accuracy on standard test data
- ✅ **Perfect on Higher Gain**: 100% accuracy on higher gain signals
- ✅ **Efficient Architecture**: Fast inference and small model size
- ✅ **Complete Pipeline**: End-to-end preprocessing and evaluation
- ⚠️ **Gain Sensitivity**: Performance degrades significantly on lower gain signals
- ⚠️ **Research Opportunity**: Demonstrates challenges in multi-domain RF classification

## 🎯 Use Cases

1. **Research**: RF security and signal processing challenges
2. **Education**: Deep learning limitations in multi-domain scenarios
3. **Development**: Baseline for robust jamming detection systems
4. **Analysis**: Study of gain-dependent performance in RF classification

## 📈 Performance Metrics

| Dataset | Accuracy | Notes |
|---------|----------|-------|
| Standard Test | 96.57% | Good performance |
| Higher Gain Test | 100.0% | Perfect classification |
| Lower Gain Test | 33.86% | **Critical limitation** |
| **Overall Average** | **76.81%** | **Highly variable** |

### Per-Class Performance (Lower Gain - Problematic)
| Class | Precision | Recall | Issue |
|-------|-----------|--------|-------|
| Normal | ~0% | ~1.7% | Almost never predicted |
| Constant Jammer | ~0% | ~0% | Almost never predicted |
| Periodic Jammer | ~34% | ~99.6% | Heavily over-predicted |

### Signal Analysis
- **Standard Dataset**: 26.65 dB class separation (good)
- **Lower Gain Dataset**: 3.52 dB class separation (problematic overlap)
- **Root Cause**: Insufficient signal-to-noise ratio in lower gain conditions

## 🔬 Technical Details

### Signal Processing
- **Sampling Rate**: 1000 Hz
- **Sequence Length**: 1000 samples (1 second)
- **Mel Bands**: 64 frequency bins
- **Time Steps**: 8 (125ms resolution)

### Model Training
- **Optimizer**: Adam (lr=0.001)
- **Loss**: Categorical crossentropy
- **Regularization**: Dropout (0.25, 0.5)
- **Training Data**: Single gain level (standard conditions)
- **Limitation**: Model not exposed to gain variations during training

## 📖 Documentation & Learning

### Interactive Learning
- **Complete Workflow**: `notebooks/rf_jamming_detection_workflow.ipynb` - Step-by-step tutorial from data analysis to model deployment
- **Hands-on Examples**: Interactive code cells with visualizations and explanations

### Technical Documentation
- **API Documentation**: Comprehensive docstrings in all script modules
- **Performance Analysis**: Training metrics and confusion matrices in `results/visualizations/`
- **Model Architecture**: Detailed model structure in `scripts/model.py`

### Quick References
- **Setup Verification**: Run any script to see automatic dependency checking
- **Demo Examples**: `scripts/demo.py` provides real-time inference examples
- **Evaluation Metrics**: `scripts/evaluate.py` generates comprehensive performance reports

## 🛠️ Development

### Current Status: Research & Development
This system demonstrates a 2D CNN approach for RF jamming detection with notable performance characteristics:

**Strengths:**
- **Excellent Standard Performance**: 96.57% accuracy on standard test data
- **Perfect Higher Gain**: 100% accuracy shows model capability
- **Clean Architecture**: Well-structured, maintainable codebase
- **Complete Pipeline**: Full workflow from raw RSSI to predictions

**Critical Limitations:**
- **Lower Gain Performance**: 33.86% accuracy represents a major deployment blocker
- **Domain Adaptation**: Model fails to generalize across gain conditions
- **Signal Overlap**: Lower gain signals have insufficient class separation (3.52 dB)
- **Training Bias**: Single-domain training leads to poor cross-domain performance

### Research Challenges Identified
1. **Multi-Domain Robustness**: Need for training across multiple gain conditions
2. **Signal Enhancement**: Potential for preprocessing to improve SNR
3. **Architecture Limitations**: Current CNN may need domain adaptation techniques
4. **Data Imbalance**: Different gain levels create distribution shift

### Key Design Decisions
- **Architecture**: 2D CNN with Mel spectrograms for time-frequency analysis
- **Training Strategy**: Single-domain training (limitation identified)
- **Data Pipeline**: Standard preprocessing without gain normalization
- **Repository Structure**: Clean organization focused on core functionality

### Development History & Current State
- ✅ **Baseline**: 1D CNN temporal analysis
- ✅ **Evolution**: 2D CNN time-frequency analysis
- ✅ **Standard Performance**: Achieved 96.57% on standard conditions
- ⚠️ **Multi-Domain Challenge**: Lower gain performance issues discovered
- 🔄 **Current Focus**: Understanding and addressing gain-dependent limitations

## 📞 Support & Resources

### Getting Started
1. **📓 Interactive Tutorial**: Start with `notebooks/rf_jamming_detection_workflow.ipynb` for complete walkthrough
2. **🚀 Quick Demo**: Run `python scripts/demo.py --samples 5` for immediate results
3. **📊 Performance Check**: Execute `python scripts/evaluate.py` for comprehensive metrics

### Troubleshooting
- **Dependencies**: All scripts automatically check and report missing packages
- **Data Issues**: Demo and evaluation scripts verify data availability
- **Model Loading**: Trained model included and automatically validated
- **Performance Issues**: Lower gain dataset shows critical limitations (33.86% accuracy)
- **Deployment Concerns**: Current model not suitable for variable gain environments

### Resources
- **Code Examples**: Every script includes usage examples and comprehensive docstrings
- **Visualizations**: Training plots and performance matrices in `results/visualizations/`
- **Best Practices**: Production-ready code structure demonstrates ML deployment patterns

## 📓 Interactive Notebook

The project includes a comprehensive Jupyter notebook (`notebooks/rf_jamming_detection_workflow.ipynb`) that demonstrates the complete workflow:

### Notebook Contents
1. **📊 Data Analysis**: Statistical analysis and visualization of RSSI signals
2. **🔄 Preprocessing**: Step-by-step conversion from RSSI to Mel spectrograms  
3. **🏋️ Model Training**: Train the lightweight 2D CNN with real-time monitoring
4. **📈 Evaluation**: Performance analysis with confusion matrices and confidence metrics
5. **🎯 Demo**: Real-time jamming detection on sample and test data

### Benefits
- **Educational**: Learn the complete RF signal processing pipeline
- **Interactive**: Modify parameters and see immediate results
- **Comprehensive**: Covers theory, implementation, and evaluation
- **Production Integration**: Uses the same modules as the production scripts

## 🏁 Status

**🔬 Research & Development Stage**

### Current State
- ✅ **Model**: Trained 2D CNN with excellent standard performance (96.57%)
- ✅ **Code**: Clean, modular implementation with comprehensive evaluation
- ✅ **Documentation**: Complete tutorials and interactive examples
- ⚠️ **Limitations**: Critical performance degradation on lower gain signals
- 🔄 **Research Need**: Multi-domain robustness for production deployment

### Repository Health
- 🧹 **Clean Structure**: Organized codebase with clear separation of concerns
- 📦 **Minimal Dependencies**: Streamlined requirements for easy setup
- 🔧 **Comprehensive Testing**: Evaluation across multiple gain conditions
- 📊 **Detailed Analysis**: Performance metrics and signal statistics included
- ⚠️ **Known Issues**: Lower gain performance documented and analyzed

### Next Steps for Production
1. **Domain Adaptation**: Train on multiple gain conditions
2. **Signal Enhancement**: Improve preprocessing for lower gain signals  
3. **Architecture Research**: Explore robustness techniques
4. **Validation**: Test across broader range of operating conditions

---

*RF Jamming Detection System - 2D CNN with Multi-Domain Challenges*  
*Current Performance: 96.57% (Standard) | 33.86% (Lower Gain)*  
*Status: Research & Development*  
*Last Updated: June 2025*

