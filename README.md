# Random Forest Algorithm Implementation for Heart Disease Prediction

A comprehensive Random Forest implementation from scratch with interactive web-based visualizations and educational insights into ensemble learning mechanisms for heart disease prediction.

## 🎯 Project Overview

This project presents a complete Random Forest implementation built from the ground up for heart disease prediction, developed as part of an Artificial Intelligence course project at Tribhuvan University, Institute of Engineering, Pulchowk Campus. The implementation demonstrates theoretical concepts through practical application and interactive visualizations.

### Key Features
- **Custom Implementation**: Built from scratch without using scikit-learn's RandomForestClassifier
- **Interactive Web Dashboard**: Real-time hyperparameter tuning with Flask framework
- **Educational Visualizations**: Decision boundaries, confusion matrices, bootstrap sampling effects
- **Medical Application**: Heart disease prediction using UCI Machine Learning Repository dataset
- **Performance Analysis**: Comprehensive comparison with standard library implementations
- **Algorithm Transparency**: Complete visibility into ensemble learning mechanisms

## 👥 Team Members

- **Sahadev Chaulagain** (PUL078BEI032)
- **Samyam Giri** (PUL078BEI035) 
- **Sandip Acharya** (PUL078BEI036)

**Course**: Artificial Intelligence  
**Institution**: Tribhuvan University, IOE, Pulchowk Campus  
**Date**: August 2025

## 🚀 Quick Start

### Prerequisites
```bash
# Python 3.7 or higher required
pip install flask numpy pandas matplotlib seaborn scikit-learn
```

### Installation and Running
```bash
# Clone the repository
git clone https://github.com/samyamcodesavvy/random-forest-heart-disease.git
cd random-forest-heart-disease

# Start the web application
python app.py

# Open your browser and navigate to:
# http://127.0.0.1:5000
```

### Alternative Training Scripts
```bash
# Train on heart disease dataset
python train.py

# Train on synthetic dataset for comparison
python train1.py
```

## 📊 Algorithm Implementation

### Core Components

**Decision Tree Construction:**
- Entropy-based information gain for optimal splitting
- Recursive feature space partitioning
- Configurable stopping criteria (max depth, min samples)
- Robust handling of edge cases and empty splits

**Random Forest Ensemble:**
- **Bootstrap Sampling (Bagging)**: Each tree trains on random samples with replacement
- **Feature Randomness**: Random subset of features considered at each node split
- **Majority Voting**: Final prediction based on ensemble consensus
- **Configurable Parameters**: Number of trees, max depth, feature selection strategies

### Mathematical Foundation

**Entropy Calculation:**
```
Entropy(S) = -Σ(p_i * log₂(p_i))
```

**Information Gain:**
```
InfoGain(S,A) = Entropy(S) - Σ(|S_v|/|S| * Entropy(S_v))
```

**Ensemble Prediction:**
```
ŷ = mode{h₁(x), h₂(x), ..., h_B(x)}
```

## 🏗️ Project Structure

```
random-forest-heart-disease/
├── app.py                    # Flask web application (main server)
├── decisiontree.py          # Custom decision tree implementation
├── randomforest.py          # Random forest ensemble implementation
├── train.py                 # Heart disease dataset training script
├── train1.py                # Synthetic dataset training script
├── templates/
│   └── index.html           # Interactive web interface
├── data/
│   └── heart.csv           # UCI heart disease dataset (303 samples)
└── README.md
```

## 📈 Dataset Information

### Heart Disease Dataset (UCI ML Repository)
- **Size**: 303 instances with 13 medical features
- **Task**: Binary classification (heart disease presence/absence)
- **Source**: UCI Machine Learning Repository
- **Target Variable**: Presence of heart disease (0 = no disease, 1 = disease present)

**Medical Features Include:**
- **Demographics**: Age, Sex
- **Clinical Measurements**: Resting Blood Pressure, Cholesterol levels, Maximum Heart Rate
- **Diagnostic Results**: Chest Pain Type, Fasting Blood Sugar, Resting ECG Results
- **Exercise Testing**: Exercise Induced Angina, ST Depression values
- **Advanced Diagnostics**: Number of Major Vessels, Thalassemia results

## 🎛️ Interactive Web Dashboard

### Real-time Parameter Tuning
- **Number of Estimators**: Adjust from 1-200 trees
- **Maximum Depth**: Control tree depth (1-50 levels)
- **Feature Selection**: Choose from 'auto', 'sqrt', 'log2', or 'None'
- **Bootstrap Sampling**: Toggle bootstrap sampling on/off
- **Sample Size**: Configure bootstrap sample size (50-500)

### Dynamic Visualizations
- **Decision Boundary Plots**: 2D visualization of how the forest partitions feature space
- **Confusion Matrix Heatmaps**: Visual representation of classification performance
- **Performance Metrics**: Real-time accuracy, precision, recall, and F1-score updates
- **Feature Selection Interface**: Choose different feature pairs for boundary visualization

### Educational Components
- **Bootstrap Sampling Visualization**: See how sample diversity affects model variance
- **Individual vs Ensemble**: Compare single tree predictions with forest consensus
- **Algorithm Step Demonstration**: Visualize tree construction and voting process

## 📊 Performance Results

### Baseline Performance
With default parameters (50 trees, max_depth=15):

| Metric    | Training Set | Test Set |
|-----------|-------------|----------|
| Accuracy  | 98.76%      | 83.61%   |
| Precision | 98.89%      | 84.21%   |
| Recall    | 98.76%      | 83.61%   |
| F1-Score  | 98.81%      | 83.72%   |

### Hyperparameter Impact Analysis
- **Number of Trees**: Performance stabilizes around 20-30 trees
- **Max Depth**: Optimal depth around 10-15 for this dataset
- **Feature Selection**: 'sqrt' provides good balance of performance and efficiency

### Comparison with Scikit-learn

| Implementation | Accuracy | Training Time | Memory Usage | Notes |
|---------------|----------|---------------|--------------|-------|
| Custom RF     | 83.61%   | 2.3s         | Moderate     | Educational transparency |
| Scikit-learn  | 85.24%   | 0.8s         | Low          | Optimized C implementation |

*The custom implementation achieves competitive accuracy with expected trade-offs for educational purposes and algorithm transparency.*

## 🔧 Technical Implementation Details

### Code Architecture
The project follows modular design principles with clear separation of concerns:

**Core Node Structure:**
```python
class Node:
    def __init__(self, feature=None, threshold=None, 
                 left=None, right=None, *, value=None):
        self.feature = feature      # Feature index for splitting
        self.threshold = threshold  # Threshold value for split
        self.left = left           # Left child node
        self.right = right         # Right child node
        self.value = value         # Prediction value (leaf nodes)
    
    def is_leaf_node(self):
        return self.value is not None
```

### Key Algorithmic Features
- **Efficient Feature Selection**: Random sampling without replacement at each split
- **Robust Splitting Criteria**: Handles edge cases with empty splits and identical features
- **Memory Optimization**: Efficient tree storage and traversal algorithms
- **Numerical Stability**: Safe entropy calculations with proper zero-division handling
- **Bootstrap Validation**: Configurable sampling strategies with replacement

### Web Interface Architecture
- **Frontend**: HTML/CSS/JavaScript for responsive user interaction
- **Backend**: Flask web framework handling API requests and responses
- **Visualization**: Matplotlib/Seaborn for dynamic plot generation
- **Real-time Updates**: AJAX for seamless parameter adjustment without page refresh
- **Data Encoding**: Efficient base64 encoding for plot transmission

## 🎓 Educational Value

### Learning Outcomes
This implementation provides hands-on experience with:
- **Ensemble Learning Concepts**: Understanding how multiple weak learners combine
- **Bootstrap Sampling Effects**: Visualizing how sample diversity reduces overfitting
- **Feature Randomness Impact**: Seeing how random feature selection affects splits
- **Hyperparameter Influence**: Real-time feedback on parameter adjustment effects
- **Algorithm Transparency**: Complete visibility into decision-making process

### Theoretical Concepts Demonstrated
- Information theory and entropy-based splitting
- Bias-variance tradeoff in ensemble methods
- Bootstrap aggregating (bagging) techniques
- Feature importance and selection strategies
- Cross-validation and performance evaluation

## 🚧 Implementation Challenges & Solutions

### Technical Challenges Addressed
- **Numerical Stability**: Implemented safe logarithm calculations to handle zero probabilities
- **Memory Management**: Optimized tree storage and efficient traversal algorithms
- **Real-time Visualization**: Developed efficient plot generation and base64 encoding
- **Parameter Validation**: Robust input validation and error handling throughout
- **Bootstrap Efficiency**: Optimized sampling algorithms for large datasets

### Performance Optimizations
- Vectorized operations using NumPy for faster computations
- Efficient data structures for tree representation
- Optimized split finding algorithms
- Memory-efficient bootstrap sampling

## 🔮 Future Enhancements

### Algorithm Improvements
- [ ] Feature importance calculation and ranking
- [ ] Out-of-bag (OOB) error estimation
- [ ] Support for regression tasks alongside classification
- [ ] Parallel tree training implementation
- [ ] Advanced pruning techniques (cost-complexity pruning)

### Interface Enhancements
- [ ] 3D decision boundary visualization
- [ ] Animation of tree construction process
- [ ] Interactive feature importance plots
- [ ] Model comparison dashboard
- [ ] Export functionality for trained models

### Dataset Extensions
- [ ] Support for categorical features
- [ ] Missing value handling strategies
- [ ] Multi-class classification support
- [ ] Cross-validation visualization

## 📚 References and Resources

1. **Breiman, L.** (2001). Random Forests. *Machine Learning*, 45(1), 5-32.
2. **Hastie, T., Tibshirani, R., & Friedman, J.** (2009). *The Elements of Statistical Learning*.
3. **Quinlan, J. R.** (1986). Induction of decision trees. *Machine Learning*, 1(1), 81-106.
4. **UCI Machine Learning Repository** - Heart Disease Dataset
5. **Pedregosa, F., et al.** (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825-2830.

- **Primary Repository**: [@samyamcodesavvy](https://github.com/samyamcodesavvy)
- **Project Link**: [Random Forest Heart Disease Prediction](https://github.com/samyamcodesavvy/random-forest-heart-disease)



This project is open source and available under the MIT License. Feel free to use, modify, and distribute for educational and research purposes.

---

**Developed with ❤️ for educational purposes at IOE, Pulchowk Campus**  
*Contributing to machine learning education through hands-on implementation and visualization*
