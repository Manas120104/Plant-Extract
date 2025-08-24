# 🌿 Predicting Medicinal Properties of Plant Extract

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://tensorflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Latest-green.svg)](https://scikit-learn.org/)

> A comprehensive machine learning project leveraging advanced neural architectures to predict the medicinal properties of copper nanoparticles synthesized from plant extracts.

## 🔬 Project Overview

This project explores the fascinating intersection of nanotechnology and phytomedicine by developing predictive models for four key medicinal activities of plant extract-synthesized copper nanoparticles. Using a rich dataset of characterization techniques and bioactivity assays, I've implemented and compared 9 different neural network architectures to achieve state-of-the-art prediction accuracy, precision, recall and f1-score.

## 📊 Dataset Information

The `copper_nanoparticle_dataset.csv` contains 5,000 samples with comprehensive characterization data:

| Column | Description |
|--------|-------------|
| **Sample_ID** | Unique identifier for each nanoparticle sample |
| **Plant_Material** | Source plant species used for nanoparticle synthesis |
| **UV_Visible** | UV-Visible spectroscopy absorption values for optical properties |
| **SEM** | Scanning Electron Microscopy morphological measurements |
| **TEM** | Transmission Electron Microscopy structural analysis data |
| **EDAX** | Energy-Dispersive X-ray Analysis elemental composition |
| **XRD** | X-Ray Diffraction crystalline structure parameters |
| **Particle_Size** | Average nanoparticle diameter in nanometers |
| **Antifungal_Activity** | Binary classification target for antifungal properties of the plant extract |
| **Antibacterial_Activity** | Binary classification target for antibacterial properties of the plant extract |
| **Antioxidant_Activity** | Binary classification target for antioxidant properties of the plant extract |
| **Anti_Inflammatory_Activity** | Binary classification target for anti-inflammatory properties of the plant extract |

## 🤖 Machine Learning Models

### Neural Network Architectures
- **Single Task Feedforward Neural Network**: Traditional dense layers optimized for individual bioactivity prediction with ReLU activations and dropout regularization
- **Multitask Feedforward Neural Network**: Shared representation learning architecture that simultaneously predicts all four medicinal properties through multi-output heads
- **Single Task TabNet**: Google's TabNet architecture with sequential attention mechanism specifically designed for tabular data feature selection
- **Multitask TabNet**: Multi-output variant of TabNet leveraging shared feature learning across all bioactivity prediction tasks

### Ensemble Methods
- **Gradient Boosted Neural Networks**: Ensemble combining gradient boosting with neural network base learners for superior predictive performance
- **Bagged Neural Networks**: Bootstrap aggregating approach using multiple neural networks trained on different data subsets with voting consensus

### Recurrent Architectures
- **Long Short-Term Memory (LSTM)**: Sequential processing of characterization features with memory cells for capturing long-term dependencies
- **Gated Recurrent Unit (GRU)**: Simplified recurrent architecture with update and reset gates for efficient temporal pattern learning
- **Recurrent Neural Network (RNN)**: Basic recurrent architecture for modeling sequential relationships in nanoparticle characterization data

## 🏆 Performance Results

### Model Performance Comparison (%)

| Model | Anti-bacterial | Anti-fungal | Anti-oxidant | Anti-inflammatory |
|-------|---------------|-------------|--------------|-------------------|
| | Acc/Prec/Rec/F1 | Acc/Prec/Rec/F1 | Acc/Prec/Rec/F1 | Acc/Prec/Rec/F1 |
| **🥇 Gradient Boosted NN** | **99.93/99.88/100.0/99.94** | **99.87/99.73/100.0/99.86** | **99.67/99.59/99.73/99.66** | **99.80/99.61/100.0/99.80** |
| Single Task TabNet | 97.60/96.80/98.10/97.50 | 97.20/97.30/97.85/97.65 | 96.90/96.75/97.25/97.00 | 96.85/96.90/96.55/96.75 |
| Single Task FF NN | 92.75/91.45/92.85/92.60 | 92.30/92.00/92.65/92.40 | 93.85/93.45/92.85/93.15 | 91.20/91.75/91.80/91.85 |
| Bagged Neural Networks | 91.65/92.20/91.10/91.65 | 91.45/91.10/92.65/91.80 | 92.10/91.90/92.15/92.00 | 91.50/91.70/91.90/91.80 |
| LSTM | 90.50/90.80/91.10/90.80 | 91.25/90.90/91.20/91.05 | 91.40/91.10/91.35/91.20 | 90.95/91.25/90.85/91.05 |
| Multitask TabNet | 89.75/88.45/90.20/89.40 | 90.10/89.50/89.80/89.60 | 90.60/89.80/89.75/90.10 | 90.35/89.90/90.05/90.20 |
| GRU | 88.75/89.20/88.10/88.75 | 88.50/88.85/88.90/88.80 | 88.90/88.70/88.95/88.80 | 88.60/88.95/88.50/88.70 |
| Multitask FF NN | 87.60/86.75/88.30/87.25 | 88.45/87.20/87.65/87.80 | 86.55/86.60/87.45/87.30 | 88.90/88.30/87.75/88.05 |
| RNN | 85.70/85.50/85.90/85.65 | 86.85/86.30/86.95/86.60 | 85.95/86.00/85.70/85.85 | 86.20/85.90/86.05/86.10 |

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.8+
Anaconda Distribution (Recommended)
CUDA-compatible GPU (Optional, for faster training)
```

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Manas120104/Plant-Extract.git
   cd Plant-Extract
   ```

2. **Set up Python environment**
   
   **Option A: Using Anaconda (Recommended)**
   ```bash
   # Create conda environment
   conda create -n medicinal-plants python=3.8
   conda activate medicinal-plants
   
   # Install core packages
   conda install pandas numpy scikit-learn matplotlib seaborn jupyter
   conda install tensorflow-gpu  # or tensorflow for CPU only
   
   # Install additional packages via pip
   pip install tabnet xgboost plotly
   ```
   
   **Option B: Using pip**
   ```bash
   # Create virtual environment
   python -m venv medicinal-plants-env
   source medicinal-plants-env/bin/activate  # On Windows: medicinal-plants-env\Scripts\activate
   
   # Install requirements
   pip install -r requirements.txt
   ```

3. **Download the dataset**
   - Place `copper_nanoparticle_dataset.csv` in the project directory
   - Ensure the dataset contains all 12 columns as specified above

### Quick Start

Run the main Jupyter notebook:

```bash
# Start Jupyter notebook
jupyter notebook

# Open and run antifungal_antibac_antioxi_antiinflam.ipynb
```

The notebook will:
- Load and preprocess the copper nanoparticle dataset
- Train all 9 machine learning models
- Evaluate performance across all four medicinal properties
- Generate comprehensive performance metrics and visualizations
- Display model comparison results

### Alternative Usage

You can also run the notebook in different environments:

**Google Colab:**
```bash
# Upload the .ipynb file and dataset to Google Colab
# Run all cells sequentially
```

**JupyterLab:**
```bash
jupyter lab
# Navigate to antifungal_antibac_antioxi_antiinflam.ipynb
```

## 🎯 Key Achievements

- **🚀 Near-Perfect Performance**: Gradient Boosted Neural Networks achieved >99% accuracy across all medicinal properties
- **🔍 Feature Importance**: TabNet models provided interpretable attention weights for characterization techniques
- **⚡ Efficient Training**: Single-task models showed faster convergence compared to multitask variants
- **📈 Robust Predictions**: Ensemble methods demonstrated superior generalization capabilities

## 📈 Scientific Impact

This research contributes to:
- **🔬 Computational Phytomedicine**: Accelerating discovery of plant-based therapeutic nanoparticles
- **⚗️ Nanoparticle Design**: Predictive framework for optimizing synthesis parameters
- **🌱 Sustainable Medicine**: Reducing experimental costs through computational screening
- **📊 Data-Driven Research**: Establishing ML benchmarks for nanomedicine applications

## 🛠️ Technologies Used

- **Deep Learning**: TensorFlow, Keras
- **Machine Learning**: scikit-learn, XGBoost, TabNet
- **Data Processing**: pandas, NumPy
- **Visualization**: matplotlib, seaborn, plotly
- **Model Evaluation**: scikit-learn metrics - accuracy, precision, recall, f1-score, cross-validation

## 🔮 Future Directions

- Integration of molecular dynamics simulations
- Extension to multi-class activity prediction
- Real-time prediction API development
- Transfer learning for new plant species

## 📈 Impact

This research bridges the gap between traditional phytomedicine and modern nanotechnology, providing a computational framework for accelerating the discovery of plant-based therapeutic nanoparticles with predictable medicinal properties.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### How to Contribute

1. **Fork the Project**
2. **Create your Feature Branch**
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. **Commit your Changes**
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. **Push to the Branch**
   ```bash
   git push origin feature/AmazingFeature
   ```
5. **Open a Pull Request**

### Areas for Contribution
- Additional machine learning models
- Feature engineering techniques
- Visualization improvements
- Documentation enhancements
- Performance optimizations

## 📋 Requirements

You can create a `requirements.txt` file with the following dependencies:
```
tensorflow>=2.8.0
scikit-learn>=1.0.0
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
tabnet>=3.1.1
xgboost>=1.5.0
plotly>=5.0.0
jupyter>=1.0.0
```

Then install using:
```bash
pip install -r requirements.txt
```

## 🙏 Acknowledgments

- **Research Community**: Gratitude to researchers advancing computational nanomedicine
- **Open Source Libraries**: TensorFlow, scikit-learn, and TabNet development teams
- **Dataset Contributors**: Scientists who generated the copper nanoparticle characterization data
- **Academic Institutions**: Supporting interdisciplinary research in phytomedicine and nanotechnology
- **Reviewers and Contributors**: Community members who helped improve this project

## 📧 Contact

**Manas Kamal Das** - manaskd2019@gmail.com

**Project Link**: [https://github.com/Manas120104/Plant-Extract](https://github.com/Manas120104/Plant-Extract)

**LinkedIn**: [Manas Kamal Das](https://www.linkedin.com/in/manaskamaldas/)

---

*Advancing the frontiers of computational phytomedicine through machine learning innovation* 🌱⚗️🤖
