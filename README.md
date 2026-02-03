# AI Bias Analysis Project

**A Comprehensive Framework for Detecting and Mitigating Bias in AI Models**

Based on the research paper: *"Analyzing Bias in AI Models"* by Mustafiz Ahmed

---

## 🎯 Overview

This project implements a complete system for analyzing and mitigating bias in artificial intelligence models. It provides tools for:

- **Bias Detection**: Measure fairness using 5 industry-standard metrics
- **Data Generation**: Create synthetic biased datasets for testing
- **Bias Mitigation**: Apply pre-processing, in-processing, and post-processing techniques
- **Visualization**: Generate professional charts and reports

---

## 📊 Features

### 1. Multiple Domain Support
- **Hiring/Recruitment** decisions
- **Credit approval** systems
- **Healthcare** risk assessment
- **Criminal justice** predictions

### 2. Comprehensive Fairness Metrics
- **Demographic Parity** (Statistical Parity)
- **Equalized Odds**
- **Equal Opportunity**
- **Predictive Parity**
- **Calibration**

### 3. Three-Stage Bias Mitigation
- **Pre-processing**: Data reweighting, resampling
- **In-processing**: Fairness-constrained training
- **Post-processing**: Threshold optimization, calibration

### 4. Professional Visualizations
- Selection rate comparisons
- ROC curves by protected group
- Confusion matrices
- Before/after mitigation charts

---

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/MZ-314/Analysing-bias-in-AI-models.git
cd AI-Bias-Analysis-Project

# Install dependencies
pip install -r requirements.txt
```

### Run Demo
```bash
python examples/demo.py
```

### Run Full Analysis
```bash
python main.py
```

---

## 📁 Project Structure
```
AI-Bias-Analysis-Project/
├── src/                    # Source code
│   ├── data/              # Data generation
│   ├── metrics/           # Fairness metrics
│   ├── mitigation/        # Bias mitigation
│   └── visualization/     # Plotting tools
├── examples/              # Usage examples
├── notebooks/             # Jupyter notebooks
├── tests/                 # Unit tests
├── docs/                  # Documentation
├── results/               # Output directory
├── main.py               # Main analysis script
├── setup.py              # Package installer
└── requirements.txt      # Dependencies
```

---

## 💻 Basic Usage

### Generate Biased Dataset
```python
from src.data.data_generator import BiasedDataGenerator

generator = BiasedDataGenerator()
df = generator.generate_hiring_dataset(n_samples=1000, bias_strength=0.4)
```

### Measure Fairness
```python
from src.metrics.fairness_metrics import FairnessMetrics

fm = FairnessMetrics()
results = fm.calculate_all_metrics(y_true, y_pred, protected_attr, "gender")
print(fm.generate_fairness_report(results))
```

### Apply Bias Mitigation
```python
from src.mitigation.bias_mitigation import BiasMitigationPipeline

pipeline = BiasMitigationPipeline(
    preprocessing_method='reweighting',
    inprocessing_constraint='demographic_parity',
    postprocessing_method='threshold_optimization'
)
pipeline.fit(X_train, y_train, protected_train)
predictions = pipeline.predict(X_test, protected_test)
```

### Create Visualizations
```python
from src.visualization.visualizer import BiasVisualizer

visualizer = BiasVisualizer()
visualizer.plot_selection_rates(results, "gender", save_path="rates.png")
```

---

## 📖 Example Results

### Bias Detection Output
```
FAIRNESS ANALYSIS REPORT: gender
======================================================================
1. DEMOGRAPHIC PARITY
   Selection Rates by Group:
      Male: 0.4523 (45.23%)
      Female: 0.3012 (30.12%)
   Disparate Impact Ratio: 0.6659
   Fair (80% rule)? ✗ NO
```

### Bias Mitigation Impact
```
Disparate Impact:
  Before: 0.6659
  After:  0.8234
  Change: ↑ 0.1575 (Better)
```

---

## 📊 Fairness Metrics Explained

| Metric | Description | Use Case |
|--------|-------------|----------|
| **Demographic Parity** | Equal positive outcome rates | University admissions |
| **Equalized Odds** | Equal TPR and FPR across groups | Medical diagnosis |
| **Equal Opportunity** | Equal TPR across groups | Job hiring |
| **Predictive Parity** | Equal precision across groups | Credit scoring |

---

## 🛠️ Advanced Usage

### Custom Dataset Analysis
```python
from src.metrics.fairness_metrics import FairnessMetrics
import pandas as pd

# Load your data
df = pd.read_csv('your_data.csv')

# Calculate metrics
fm = FairnessMetrics()
results = fm.calculate_all_metrics(
    y_true=df['actual_outcome'],
    y_pred=df['predicted_outcome'],
    protected_attr=df['gender'],
    protected_attr_name="gender"
)

# Generate report
print(fm.generate_fairness_report(results))
```

### Complete Analysis Pipeline
```python
from main import AIBiasAnalysisSystem

# Create system
system = AIBiasAnalysisSystem(
    dataset_type='hiring',  # 'credit', 'healthcare', 'criminal_justice'
    bias_strength=0.4
)

# Run complete analysis
results = system.run_complete_analysis(
    n_samples=2000,
    output_dir='./my_results'
)
```

---

## 🔬 Research Background

This implementation is based on the research paper **"Analyzing Bias in AI Models"** which addresses:

1. **Sources of bias** in AI systems
2. **Methods for bias detection**
3. **Mitigation strategies**
4. **Ethical considerations**
5. **Real-world case studies**

The project implements techniques from:
- Mehrabi et al. (2021) - Survey on Bias and Fairness in ML
- Barocas, Hardt, Narayanan (2019) - Fairness and Machine Learning
- Hardt et al. (2016) - Equality of Opportunity

---

## 📈 Performance Benchmarks

- **Dataset Generation**: 1,000 samples in < 1 second
- **Model Training**: Complete pipeline in ~15 seconds (2,000 samples)
- **Visualization**: High-resolution exports in ~2 seconds per figure

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

---

## 📝 License

MIT License - See LICENSE file for details

---

## 👤 Author

**Mustafiz Ahmed**
- Institution: Apex Institute of Technology, Chandigarh University
- Email: aimjetkhalifa10@gmail.com
- GitHub: [@MZ-314](https://github.com/MZ-314)

---

## 🙏 Acknowledgments

- Research methodology from the paper "Analyzing Bias in AI Models"
- Fairness metrics based on IBM AIF360 and Microsoft Fairlearn
- Implementation inspired by academic research in ML fairness

---

## 📚 References

1. Mehrabi et al. (2021) - Survey on Bias and Fairness in ML
2. Barocas, Hardt, Narayanan (2019) - Fairness and Machine Learning
3. Dwork et al. (2012) - Fairness through Awareness
4. Hardt et al. (2016) - Equality of Opportunity
5. IBM AI Fairness 360 (Bellamy et al., 2018)
6. Microsoft Fairlearn

---

## ⚠️ Disclaimer

This project is for **educational and research purposes**. Always consult with legal and ethical experts when deploying AI systems in production.

---

## 🌟 Features Roadmap

- [ ] Deep learning model support
- [ ] Real-world dataset integration
- [ ] Interactive web dashboard
- [ ] Automated fairness auditing
- [ ] Causal fairness analysis
- [ ] Intersectional fairness metrics

---

**Star this repository if you find it useful!** ⭐