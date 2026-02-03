# Quick Start Guide - AI Bias Analysis Project

## Getting Started in 5 Minutes

---

## Step 1: Install Dependencies (1 minute)
```bash
cd AI-Bias-Analysis-Project
pip install -r requirements.txt
```

**Note:** If you get an error with `aif360`, try:
```bash
pip install aif360 --break-system-packages
```

---

## Step 2: Run the Demo (2 minutes)
```bash
python examples/demo.py
```

This will:
- Generate a biased hiring dataset
- Train baseline and mitigated models
- Display fairness metrics
- Create visualizations in `./demo_results/`

---

## Step 3: View Results (2 minutes)

Check the generated files:
- `./demo_results/selection_rates.png` - Selection rates by group
- `./demo_results/mitigation_comparison.png` - Before/after comparison
- `./demo_results/roc_curves.png` - Model performance by group
- `./demo_results/confusion_matrices.png` - Prediction accuracy by group

---

## Understanding the Output

### Console Output Explained
```
Disparate Impact: 0.6659
  (Fair if ≥ 0.8)
```
- **What it means**: The less privileged group receives 66.59% of positive outcomes compared to the most privileged group
- **Fair threshold**: Should be ≥ 0.8 (80% rule)
- **This example**: Shows significant bias (FAIL)
```
Statistical Parity Difference: 0.1511
```
- **What it means**: 15.11% difference in selection rates between groups
- **Fair threshold**: Should be ≤ 0.1
- **This example**: Shows moderate bias (FAIL)
```
Equalized Odds Difference: 0.0975
```
- **What it means**: Average difference in TPR and FPR across groups
- **Fair threshold**: Should be ≤ 0.1
- **This example**: Close to fair (BORDERLINE)

### Fairness Criteria Scoring
```
Fairness Criteria Passed: 2/4
```
- **4 criteria tested**: Demographic Parity, Equalized Odds, Equal Opportunity, Predictive Parity
- **Baseline might pass**: 1-2 criteria (biased)
- **Mitigated should pass**: 3-4 criteria (fair)

---

## Common Use Cases

### 1. Analyze Your Own Dataset
```python
from src.metrics.fairness_metrics import FairnessMetrics
import pandas as pd

# Load your data
df = pd.read_csv('your_data.csv')

# Prepare data
y_true = df['actual_outcome'].values
y_pred = df['predicted_outcome'].values
protected_attr = df['gender'].values  # or race, age_group, etc.

# Measure fairness
fm = FairnessMetrics()
results = fm.calculate_all_metrics(
    y_true, y_pred, protected_attr, "gender"
)

# Print report
print(fm.generate_fairness_report(results))
```

### 2. Compare Different Models
```python
from src.metrics.fairness_metrics import FairnessMetrics

fm = FairnessMetrics()

# Calculate metrics for each model
model_results = {
    'Model A': fm.calculate_all_metrics(y_true, pred_a, protected_attr, "gender"),
    'Model B': fm.calculate_all_metrics(y_true, pred_b, protected_attr, "gender"),
    'Model C': fm.calculate_all_metrics(y_true, pred_c, protected_attr, "gender")
}

# Compare
comparison = fm.compare_models(model_results)
print(comparison)
```

### 3. Apply Bias Mitigation
```python
from src.mitigation.bias_mitigation import BiasMitigationPipeline

# Create pipeline
pipeline = BiasMitigationPipeline(
    preprocessing_method='reweighting',
    inprocessing_constraint='demographic_parity',
    postprocessing_method='threshold_optimization'
)

# Train
pipeline.fit(X_train, y_train, protected_train)

# Optimize on validation set
pipeline.optimize_postprocessing(X_val, y_val, protected_val)

# Predict
predictions = pipeline.predict(X_test, protected_test)
```

### 4. Create Visualizations
```python
from src.visualization.visualizer import BiasVisualizer

visualizer = BiasVisualizer()

# Selection rates
visualizer.plot_selection_rates(
    fairness_results,
    "gender",
    save_path="selection_rates.png"
)

# Before/after comparison
visualizer.plot_bias_mitigation_comparison(
    baseline_results,
    mitigated_results,
    save_path="comparison.png"
)

# ROC curves by group
visualizer.plot_roc_curves_by_group(
    y_true, y_proba, protected_attr, "gender",
    save_path="roc_curves.png"
)
```

---

## Running Different Analyses

### Test Different Domains

#### Hiring Decisions
```bash
python main.py
# Default is hiring dataset
```

#### Credit Approval
Edit `main.py` and change:
```python
system = AIBiasAnalysisSystem(
    dataset_type='credit',  # Changed from 'hiring'
    bias_strength=0.4
)
```

#### Healthcare Risk
```python
system = AIBiasAnalysisSystem(
    dataset_type='healthcare',
    bias_strength=0.35
)
```

#### Criminal Justice
```python
system = AIBiasAnalysisSystem(
    dataset_type='criminal_justice',
    bias_strength=0.4
)
```

---

## Interpreting Fairness Metrics

### When to Use Which Metric?

| Metric | Use When | Example |
|--------|----------|---------|
| **Demographic Parity** | Equal representation is goal | University admissions |
| **Equalized Odds** | Both errors matter equally | Medical diagnosis |
| **Equal Opportunity** | Finding qualified candidates is priority | Job hiring |
| **Predictive Parity** | Prediction confidence matters | Credit scoring |

### Trade-offs

⚠️ **Important**: Not all fairness metrics can be satisfied simultaneously!

- **Demographic Parity vs. Predictive Parity**: Often conflict
- **Equal Opportunity vs. Calibration**: May require trade-offs
- **Fairness vs. Accuracy**: Sometimes reducing bias reduces overall accuracy

**Choose the metric** that aligns with your domain's ethical requirements.

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'aif360'"
**Solution**:
```bash
pip install aif360 --break-system-packages
```

### Issue: Fairness metrics show NaN
**Cause**: Not enough samples in a protected group  
**Solution**: 
- Increase dataset size
- Check for empty groups
- Ensure balanced representation

### Issue: Mitigation doesn't improve fairness
**Possible causes**:
1. Bias is too strong (try reducing bias_strength)
2. Wrong mitigation strategy for the type of bias
3. Need to tune hyperparameters

**Solutions**:
- Try different mitigation combinations
- Increase training data
- Experiment with different preprocessing methods

### Issue: Import errors
**Solution**: Make sure you're running from the project root:
```bash
cd AI-Bias-Analysis-Project
python examples/demo.py
```

---

## Next Steps

1. **Explore the code**: Read through the module files to understand implementation
2. **Modify parameters**: Try different bias strengths, mitigation strategies
3. **Add your data**: Adapt the code to work with your datasets
4. **Extend functionality**: Add new fairness metrics or mitigation techniques
5. **Read the paper**: Review the research paper for theoretical background

---

## Key Files Reference

- `src/data/data_generator.py` - Generate synthetic biased datasets
- `src/metrics/fairness_metrics.py` - Calculate fairness metrics
- `src/mitigation/bias_mitigation.py` - Apply mitigation techniques
- `src/visualization/visualizer.py` - Create charts and plots
- `main.py` - Complete analysis pipeline
- `examples/demo.py` - Quick demonstration

---

## Getting Help

1. Check the README.md for detailed documentation
2. Review code comments for implementation details
3. Look at examples/demo.py for usage examples
4. Open an issue on GitHub

---

## Expected Demo Output

When you run `python examples/demo.py`, you should see:
```
============================================================
AI BIAS ANALYSIS - QUICK DEMO
============================================================

[1/5] Generating biased hiring dataset...
✓ Generated 1000 samples

Hiring rates by gender:
gender
Female    0.301
Male      0.452

[2/5] Preparing data for modeling...
✓ Training samples: 700
✓ Test samples: 300

[3/5] Training baseline model (no bias mitigation)...
✓ Baseline accuracy: 0.7533

[4/5] Measuring bias in baseline model...

Baseline Model Fairness Metrics:
--------------------------------------------------------------------------------
Disparate Impact: 0.6659
  (Fair if ≥ 0.8)
Statistical Parity Diff: 0.1511
Equalized Odds Diff: 0.1034
Equal Opportunity Diff: 0.1343

Fairness Criteria Passed: 1/4

[5/5] Applying bias mitigation techniques...

Mitigated Model Fairness Metrics:
--------------------------------------------------------------------------------
Disparate Impact: 0.8234
  (Fair if ≥ 0.8)
Statistical Parity Diff: 0.0823
Equalized Odds Diff: 0.0765
Equal Opportunity Diff: 0.0891

Fairness Criteria Passed: 3/4

============================================================
DEMO COMPLETE!
============================================================
```

---

## Research Paper Reference

This implementation is based on:

**"Analyzing Bias in AI Models"**
- Author: Mustafiz Ahmed
- Institution: Apex Institute of Technology, Chandigarh University

The implementation covers all sections of the paper:
- Section I: Introduction to AI bias
- Section II: Literature survey on fairness metrics
- Section III: Objectives (7-step methodology)
- Section IV: System design
- Section V: Results and evaluation
- Section VI: Conclusions

---

**Ready to analyze bias in AI? Run `python examples/demo.py` to get started!**