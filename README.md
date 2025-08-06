# RF Jamming Detection with 2D CNN

## 🎯 Overview

RF jamming detection system using a lightweight 2D CNN architecture that operates on Mel spectrograms derived from RSSI signals. This research demonstrates both the promise and critical limitations of CNN-based approaches for RF signal classification across different operating conditions.

## 🏆 Performance Summary

### Multi-Gain Evaluation Results (August 2025)
- **Standard Test Dataset**: 96.57% accuracy (excellent performance)
- **Higher Gain Test Dataset**: 100.0% accuracy (perfect classification)  
- **Lower Gain Test Dataset**: 33.86% accuracy (critical failure)
- **Average Across All Conditions**: 76.81% (highly variable)

### Model Specifications
- **Architecture**: Lightweight 2D CNN (64,451 parameters)
- **Model Size**: 68.5 KB TensorFlow Lite (optimized for edge deployment)
- **Classes**: Normal, Constant Jammer (CJ), Periodic Jammer (PJ)
- **Input**: Mel spectrograms (64×8×1) from 1000-sample RSSI sequences

### ⚠️ Critical Research Findings
- **Severe Gain Sensitivity**: 62.71% accuracy drop between higher and lower gain conditions
- **Periodic Jammer Bias**: Model heavily over-predicts Periodic Jammer class in low gain scenarios
- **Signal Separation Issue**: Class separation drops from 26.65 dB (standard) to 3.52 dB (lower gain)
- **Domain Adaptation Challenge**: Demonstrates fundamental limitation of single-domain training

## 📁 Project Structure

```
SDRv2/
├── 📋 README.md                # Project documentation and findings
├── 📓 notebooks/               # Interactive analysis and evaluation
│   ├── rf_jamming_detection_workflow.ipynb     # Complete tutorial workflow
│   ├── model_evaluation_comprehensive.ipynb    # Multi-gain evaluation results  
│   └── tflite_test.ipynb                      # TensorFlow Lite optimization
├── 🐍 scripts/                 # Core implementation modules
│   ├── demo.py                # Real-time inference demonstration
│   ├── evaluate.py            # Model evaluation and metrics
│   ├── live_usrp_detection.py # GNU Radio USRP integration
│   ├── model.py               # Lightweight 2D CNN architecture
│   ├── preprocess.py          # RSSI to Mel spectrogram pipeline
│   ├── train.py               # Model training pipeline
│   └── requirements.txt       # Python dependencies
├── 🤖 model/                   # Trained models and configurations
│   ├── jamming_detector_lightweight_best.h5    # Original Keras model
│   ├── jamming_detector_lightweight.tflite     # Optimized TFLite model (68.5 KB)
│   └── jamming_detector_lightweight_results.pkl # Training metrics
├── 📊 Dataset/                 # Multi-gain RSSI signal datasets
│   ├── training/              # Training data (single gain level)
│   ├── test/                  # Standard test data (96.57% accuracy)
│   ├── testv1_Higher_Gain/    # Higher gain test data (100% accuracy)
│   └── testv2_Lower_Gain/     # Lower gain test data (33.86% accuracy - problematic)
├── 💾 preprocessed_data/       # Processed training data and configuration
│   ├── config.pkl             # Preprocessing configuration parameters
│   ├── X_train.npy, X_test.npy # Training/test mel spectrograms
│   └── y_train.npy, y_test.npy # Training/test labels
└── 📈 results/                 # Comprehensive evaluation outputs
    ├── reports/               # Performance analysis reports
    └── visualizations/        # Training plots, confusion matrices, confidence analysis
```

## 🚀 Quick Start

### Installation & Setup
```bash
# 1. Navigate to project directory
cd /path/to/SDRv2

# 2. Install Python dependencies
pip install -r scripts/requirements.txt

# 3. Verify setup with demo
python scripts/demo.py --samples 3
```

### Usage Options

#### Option 1: Interactive Learning (Recommended)
```bash
# Complete workflow tutorial with visualizations
jupyter notebook notebooks/rf_jamming_detection_workflow.ipynb

# Comprehensive multi-gain evaluation analysis
jupyter notebook notebooks/model_evaluation_comprehensive.ipynb
```

#### Option 2: Command Line Evaluation
```bash
# Quick performance demo (works well on standard/higher gain data)
python scripts/demo.py --samples 5

# Comprehensive model evaluation across all gain levels
python scripts/evaluate.py

# Hardware-based real-time detection (requires USRP and GNU Radio)
python scripts/live_usrp_detection.py --freq 915e6 --gain 70 --sample-rate 1e6
```

#### Option 3: Development and Research
```bash
# Data preprocessing (if modifying pipeline)
python scripts/preprocess.py

# Model retraining (to address gain limitations)
python scripts/train.py --epochs 50 --batch-size 32
```

## 🧠 Model Architecture & Technical Details

### 2D CNN Architecture
- **Input**: Mel spectrograms (64×8×1) from 1000-sample RSSI sequences
- **Architecture**: 2 Conv2D blocks + Dense layers with dropout regularization
- **Parameters**: 64,451 (lightweight design for edge deployment)
- **Model Sizes**: 
  - Original: ~1.4 MB (Keras H5 format)
  - Optimized: 68.5 KB (TensorFlow Lite format)

### Data Processing Pipeline
```
Raw RSSI Signal [1000 samples, 1 kHz] 
    ↓
Short-Time Fourier Transform (n_fft=256, hop_length=128)
    ↓
Mel-scale Filtering (n_mels=64, frequency range optimized)
    ↓ 
Log-scale Transformation + Normalization
    ↓
Mel Spectrogram [64×8×1] → 2D CNN Classification
```

### Key Features & Limitations
✅ **Strengths:**
- Fast inference (~0.1-0.4 ms per sample)
- Compact model suitable for edge deployment
- Excellent performance on controlled conditions
- Complete end-to-end processing pipeline

❌ **Critical Limitations:**
- Severe gain dependency (62.71% accuracy range)
- Single-domain training approach
- No domain adaptation mechanisms
- Periodic Jammer prediction bias in low SNR conditions

## 🎯 Use Cases & Applications

### Research Applications
1. **Multi-Domain Signal Classification**: Study of CNN robustness across operating conditions
2. **Domain Adaptation Research**: Baseline for developing gain-invariant models
3. **Signal Processing Analysis**: RF signal characteristics under different gain levels
4. **Edge AI Deployment**: Lightweight model optimization for embedded systems

### Educational Value
1. **Deep Learning Limitations**: Demonstrates real-world deployment challenges
2. **RF Signal Processing**: Complete pipeline from RSSI to classification
3. **Performance Analysis**: Comprehensive evaluation methodology
4. **Research Methodology**: Proper handling of multi-condition datasets

### Current Deployment Considerations
- ✅ **Excellent for Controlled Environments**: 96.57-100% accuracy on standard/higher gain
- ⚠️ **Not Suitable for Variable Gain Scenarios**: 33.86% accuracy in lower gain conditions
- 🔬 **Research Platform**: Ideal for studying domain adaptation techniques
- 📚 **Educational Tool**: Demonstrates both promise and limitations of CNN approaches

## 📈 Comprehensive Performance Analysis

| Dataset | Accuracy | Precision | Recall | F1-Score | Key Observations |
|---------|----------|-----------|--------|----------|------------------|
| **Standard** | **96.57%** | 0.9659 | 0.9659 | 0.9654 | Excellent baseline performance |
| **Higher Gain** | **100.0%** | 1.0000 | 1.0000 | 1.0000 | Perfect classification demonstrates model capability |
| **Lower Gain** | **33.86%** | 0.1761 | 0.3386 | 0.1761 | Critical failure - heavy Periodic Jammer bias |

### Lower Gain Performance Breakdown (Critical Issue)
| Class | Precision | Recall | F1-Score | Predicted Distribution |
|-------|-----------|--------|----------|----------------------|
| Normal | ~0.00 | ~0.017 | ~0.00 | Almost never predicted |
| Constant Jammer | ~0.00 | ~0.000 | ~0.00 | Almost never predicted |
| **Periodic Jammer** | **0.34** | **0.996** | **0.51** | **99.6% of all predictions** |

### Root Cause Analysis
- **Signal Separation**: Class separation drops from 26.65 dB (standard) to 3.52 dB (lower gain)
- **Training Domain**: Model trained on single gain level, fails to generalize
- **Feature Overlap**: Lower gain signals become indistinguishable in feature space
- **Architecture Limitation**: CNN lacks domain adaptation mechanisms

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

## 📖 Documentation & Learning Resources

### Interactive Learning Materials
- **📓 Complete Workflow**: `notebooks/rf_jamming_detection_workflow.ipynb`
  - Step-by-step tutorial from raw data to model deployment
  - Interactive visualizations and parameter exploration
  - Production-ready code examples
  
- **📊 Comprehensive Evaluation**: `notebooks/model_evaluation_comprehensive.ipynb`
  - Multi-gain performance analysis with detailed metrics
  - Confidence analysis and model calibration studies
  - Visualization of critical limitations and bias patterns

### Technical Documentation
- **🔧 Implementation Details**: Comprehensive docstrings in all modules
- **📈 Performance Analysis**: Detailed metrics and confusion matrices in `results/`
- **🏗️ Architecture Specs**: Model structure and parameters in `scripts/model.py`
- **⚙️ Configuration**: Preprocessing parameters and pipeline settings

### Getting Started Guide
1. **🚀 Quick Demo**: `python scripts/demo.py --samples 5` for immediate results
2. **📊 Full Evaluation**: `python scripts/evaluate.py` for comprehensive metrics
3. **🔬 Research Analysis**: Open evaluation notebook for detailed findings
4. **🛠️ Development**: Use training script for model modifications

### Key Research Papers & References
This work contributes to understanding of:
- CNN-based RF signal classification limitations
- Domain adaptation challenges in signal processing
- Edge AI deployment considerations for RF applications

## 🛠️ Development Status & Research Insights

### Current Status: Advanced Research Prototype
This system represents a comprehensive study of 2D CNN approaches for RF jamming detection, with significant findings about domain adaptation challenges in RF signal classification.

**Research Achievements:**
- ✅ **Robust Baseline**: 96.57% accuracy on standard conditions validates approach
- ✅ **Comprehensive Evaluation**: Multi-gain analysis reveals critical limitations
- ✅ **Production-Ready Code**: Clean, modular architecture suitable for further research
- ✅ **Edge Optimization**: 68.5 KB TensorFlow Lite model for embedded deployment
- ✅ **Complete Documentation**: Extensive analysis and visualization of results

**Critical Research Findings:**
- 🔍 **Domain Adaptation Challenge**: 62.71% accuracy variation across gain conditions
- 🔍 **Feature Space Collapse**: Class separation drops from 26.65 dB to 3.52 dB  
- 🔍 **Prediction Bias**: Strong tendency toward Periodic Jammer in low SNR conditions
- 🔍 **Training Limitations**: Single-domain training insufficient for robust deployment

### Future Research Directions
1. **Multi-Domain Training**: Incorporate gain variations during training phase
2. **Domain Adversarial Networks**: Implement gain-invariant feature learning
3. **Signal Enhancement**: Preprocessing techniques to improve lower gain performance
4. **Ensemble Methods**: Combine multiple models trained on different gain levels
5. **Transfer Learning**: Adapt models across different RF environments

### Repository Health & Maintenance
- 🧹 **Clean Architecture**: Streamlined codebase focused on core functionality
- 📦 **Minimal Dependencies**: Easy setup and reproducible environment
- 🔧 **Comprehensive Testing**: Validated across multiple datasets and conditions
- 📊 **Detailed Analysis**: Performance metrics, confusion matrices, and confidence analysis
- 📚 **Educational Resources**: Interactive notebooks and comprehensive documentation

## 📞 Support & Troubleshooting

### Getting Started
1. **📓 Interactive Tutorial**: Start with `notebooks/rf_jamming_detection_workflow.ipynb` for complete walkthrough
2. **🚀 Quick Demo**: Run `python scripts/demo.py --samples 5` for immediate results  
3. **📊 Performance Analysis**: Execute `python scripts/evaluate.py` for comprehensive metrics
4. **🔬 Research Insights**: Open `notebooks/model_evaluation_comprehensive.ipynb` for detailed findings

### Common Issues & Solutions
- **Dependencies**: All scripts automatically check and report missing packages
- **Data Loading**: Demo and evaluation scripts verify dataset availability
- **Model Files**: Trained models included and automatically validated
- **Performance Expectations**: Lower gain performance limitations are documented features, not bugs
- **Hardware Requirements**: Live USRP detection requires GNU Radio and compatible SDR hardware

### Research & Development Support
- **Performance Questions**: Refer to comprehensive evaluation results in notebooks
- **Implementation Details**: Detailed docstrings and comments throughout codebase
- **Deployment Guidance**: Current model suitable for controlled environments only
- **Future Research**: See development status section for identified research directions

## 📓 Interactive Notebooks

### Available Notebooks
1. **📊 `rf_jamming_detection_workflow.ipynb`**: Complete end-to-end workflow tutorial
   - Data analysis and visualization of RSSI signals across gain levels
   - Step-by-step preprocessing pipeline demonstration
   - Model training with real-time performance monitoring
   - Inference examples and deployment considerations

2. **🔬 `model_evaluation_comprehensive.ipynb`**: Multi-gain performance analysis
   - Detailed evaluation across Standard, Higher Gain, and Lower Gain datasets
   - Confidence analysis and model calibration studies
   - Visualization of critical performance limitations
   - Statistical analysis of class separation and feature overlap

3. **⚡ `tflite_test.ipynb`**: TensorFlow Lite optimization analysis
   - Model size reduction from 1.4 MB to 68.5 KB
   - Inference speed comparison and optimization
   - Edge deployment considerations

### Educational Benefits
- **Hands-on Learning**: Interactive parameter modification with immediate feedback
- **Research Insights**: Direct exploration of gain-dependent performance issues
- **Production Integration**: Code examples directly usable in applications
- **Comprehensive Analysis**: Complete performance evaluation methodology

## 🏁 Project Status & Conclusions

### Current Status: Advanced Research Prototype (August 2025)

**🔬 Research Contributions:**
- ✅ **Comprehensive CNN Analysis**: Thorough evaluation of 2D CNN approach for RF signal classification
- ✅ **Multi-Domain Dataset**: Created and evaluated datasets across three gain levels
- ✅ **Critical Limitation Discovery**: Identified and documented severe gain-dependency issues (62.71% accuracy variation)
- ✅ **Edge Optimization**: Achieved 68.5 KB model size suitable for embedded deployment
- ✅ **Open Research Platform**: Provides baseline for domain adaptation research

**⚠️ Key Findings & Limitations:**
- **Excellent Controlled Performance**: 96.57-100% accuracy under standard/higher gain conditions
- **Critical Deployment Blocker**: 33.86% accuracy in lower gain scenarios due to feature space collapse
- **Periodic Jammer Bias**: Model heavily over-predicts this class in low SNR conditions
- **Domain Adaptation Need**: Current single-domain training approach insufficient for robust deployment

### Repository Health & Quality
- 🧹 **Production-Ready Code**: Clean, modular architecture with comprehensive documentation
- 📦 **Streamlined Dependencies**: Minimal requirements for easy setup and reproduction
- 🔧 **Thorough Testing**: Validated across multiple datasets and operating conditions
- 📊 **Comprehensive Analysis**: Detailed performance metrics, visualizations, and statistical analysis
- 📚 **Educational Value**: Interactive notebooks and extensive research documentation

### Impact & Future Research
This work provides crucial insights into the challenges of deploying CNN-based RF signal classifiers across varying operating conditions. The documented performance collapse in lower gain scenarios represents an important finding for the RF signal processing community and highlights the need for:

1. **Domain Adaptation Techniques**: Multi-domain training approaches
2. **Signal Enhancement Methods**: Preprocessing to improve SNR in challenging conditions  
3. **Robust Architecture Design**: Models that maintain performance across operating conditions
4. **Transfer Learning Applications**: Adapting models to new RF environments

### Next Steps for Production Deployment
1. **Multi-Gain Training**: Incorporate all gain levels during training phase
2. **Domain Adversarial Networks**: Implement gain-invariant feature learning
3. **Ensemble Approaches**: Combine models specialized for different operating conditions
4. **Signal Enhancement**: Develop preprocessing to improve lower gain performance

---

**RF Jamming Detection System - 2D CNN Research Platform**  
*Performance Range: 33.86% - 100% (gain-dependent)*  
*Status: Advanced Research Prototype with Critical Findings*  
*Current Version: Multi-Gain Evaluation Complete*  
*Last Updated: August 6, 2025*

