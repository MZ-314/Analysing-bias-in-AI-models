"""
Quick Demo Script for AI Bias Analysis
Demonstrates the key features of the bias analysis system
"""

import sys
import os
# Add parent directory to path to import src modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

from src.data.data_generator import BiasedDataGenerator
from src.metrics.fairness_metrics import FairnessMetrics
from src.mitigation.bias_mitigation import BiasMitigationPipeline
from src.visualization.visualizer import BiasVisualizer

print("="*80)
print("AI BIAS ANALYSIS - QUICK DEMO")
print("="*80)

# Step 1: Generate biased hiring dataset
print("\n[1/5] Generating biased hiring dataset...")
generator = BiasedDataGenerator()
df = generator.generate_hiring_dataset(n_samples=1000, bias_strength=0.4)

print(f"✓ Generated {len(df)} samples")
print("\nHiring rates by gender:")
print(df.groupby('gender')['hired'].mean())
print("\nHiring rates by race:")
print(df.groupby('race')['hired'].mean())

# Step 2: Prepare data
print("\n[2/5] Preparing data for modeling...")

# Prepare features
feature_cols = ['age', 'education_score', 'experience_years', 'skills_score', 'interview_score']
X = df[feature_cols].values
y = df['hired'].values
protected_attr = df['gender'].values

# Split data
X_train, X_test, y_train, y_test, protected_train, protected_test = \
    train_test_split(X, y, protected_attr, test_size=0.3, random_state=42, stratify=y)

print(f"✓ Training samples: {len(X_train)}")
print(f"✓ Test samples: {len(X_test)}")

# Step 3: Train baseline model (without bias mitigation)
print("\n[3/5] Training baseline model (no bias mitigation)...")

baseline_model = LogisticRegression(max_iter=1000, random_state=42)
baseline_model.fit(X_train, y_train)

baseline_pred = baseline_model.predict(X_test)
baseline_proba = baseline_model.predict_proba(X_test)[:, 1]
baseline_accuracy = baseline_model.score(X_test, y_test)

print(f"✓ Baseline accuracy: {baseline_accuracy:.4f}")

# Step 4: Measure bias in baseline model
print("\n[4/5] Measuring bias in baseline model...")

fm = FairnessMetrics()
baseline_results = fm.calculate_all_metrics(
    y_test, baseline_pred, protected_test, "gender"
)

print("\nBaseline Model Fairness Metrics:")
print("-" * 80)
print(f"Disparate Impact: {baseline_results['demographic_parity']['disparate_impact']:.4f}")
print(f"  (Fair if ≥ 0.8)")
print(f"Statistical Parity Diff: {baseline_results['demographic_parity']['statistical_parity_difference']:.4f}")
print(f"Equalized Odds Diff: {baseline_results['equalized_odds']['equalized_odds_difference']:.4f}")
print(f"Equal Opportunity Diff: {baseline_results['equal_opportunity']['tpr_difference']:.4f}")

# Check how many fairness criteria passed
baseline_fair_count = sum([
    baseline_results['demographic_parity']['is_fair'],
    baseline_results['equalized_odds']['is_fair'],
    baseline_results['equal_opportunity']['is_fair'],
    baseline_results['predictive_parity']['is_fair']
])

print(f"\nFairness Criteria Passed: {baseline_fair_count}/4")

# Step 5: Apply bias mitigation
print("\n[5/5] Applying bias mitigation techniques...")

# Split test set for validation and final testing
val_size = len(X_test) // 2
X_val, X_test_final = X_test[:val_size], X_test[val_size:]
y_val, y_test_final = y_test[:val_size], y_test[val_size:]
protected_val, protected_test_final = protected_test[:val_size], protected_test[val_size:]

# Create and train mitigation pipeline
mitigated_model = BiasMitigationPipeline(
    preprocessing_method='reweighting',
    inprocessing_constraint='demographic_parity',
    postprocessing_method='threshold_optimization'
)

mitigated_model.fit(X_train, y_train, protected_train)
mitigated_model.optimize_postprocessing(X_val, y_val, protected_val)

# Predict on final test set
mitigated_pred = mitigated_model.predict(X_test_final, protected_test_final)

# Also get baseline predictions on final test set
baseline_pred_final = baseline_model.predict(X_test_final)
baseline_proba_final = baseline_model.predict_proba(X_test_final)[:, 1]

# Measure fairness of mitigated model
mitigated_results = fm.calculate_all_metrics(
    y_test_final, mitigated_pred, protected_test_final, "gender"
)

print("\nMitigated Model Fairness Metrics:")
print("-" * 80)
print(f"Disparate Impact: {mitigated_results['demographic_parity']['disparate_impact']:.4f}")
print(f"  (Fair if ≥ 0.8)")
print(f"Statistical Parity Diff: {mitigated_results['demographic_parity']['statistical_parity_difference']:.4f}")
print(f"Equalized Odds Diff: {mitigated_results['equalized_odds']['equalized_odds_difference']:.4f}")
print(f"Equal Opportunity Diff: {mitigated_results['equal_opportunity']['tpr_difference']:.4f}")

mitigated_fair_count = sum([
    mitigated_results['demographic_parity']['is_fair'],
    mitigated_results['equalized_odds']['is_fair'],
    mitigated_results['equal_opportunity']['is_fair'],
    mitigated_results['predictive_parity']['is_fair']
])

print(f"\nFairness Criteria Passed: {mitigated_fair_count}/4")

# Results comparison
print("\n" + "="*80)
print("RESULTS SUMMARY")
print("="*80)

# Recalculate baseline on final test set for fair comparison
baseline_results_final = fm.calculate_all_metrics(
    y_test_final, baseline_pred_final, protected_test_final, "gender"
)

print("\nFairness Improvement:")
print("-" * 80)

metrics_comparison = {
    'Disparate Impact': [
        baseline_results_final['demographic_parity']['disparate_impact'],
        mitigated_results['demographic_parity']['disparate_impact']
    ],
    'Stat Parity Diff': [
        baseline_results_final['demographic_parity']['statistical_parity_difference'],
        mitigated_results['demographic_parity']['statistical_parity_difference']
    ],
    'Eq Odds Diff': [
        baseline_results_final['equalized_odds']['equalized_odds_difference'],
        mitigated_results['equalized_odds']['equalized_odds_difference']
    ]
}

for metric, (before, after) in metrics_comparison.items():
    if metric == 'Disparate Impact':
        improvement = after - before
        symbol = "↑" if improvement > 0 else "↓"
        status = "Better" if improvement > 0 else "Worse"
    else:
        improvement = before - after
        symbol = "↓" if improvement > 0 else "↑"
        status = "Better" if improvement > 0 else "Worse"
    
    print(f"\n{metric}:")
    print(f"  Before: {before:.4f}")
    print(f"  After:  {after:.4f}")
    print(f"  Change: {symbol} {abs(improvement):.4f} ({status})")

baseline_fair_final = sum([
    baseline_results_final['demographic_parity']['is_fair'],
    baseline_results_final['equalized_odds']['is_fair'],
    baseline_results_final['equal_opportunity']['is_fair'],
    baseline_results_final['predictive_parity']['is_fair']
])

print(f"\n" + "-"*80)
print(f"Baseline Model: {baseline_fair_final}/4 fairness criteria passed")
print(f"Mitigated Model: {mitigated_fair_count}/4 fairness criteria passed")
print(f"Improvement: +{mitigated_fair_count - baseline_fair_final} criteria")

# Generate visualizations
print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

os.makedirs('./demo_results', exist_ok=True)

visualizer = BiasVisualizer()

print("\nCreating visualizations...")

# 1. Selection rates
fig1 = visualizer.plot_selection_rates(
    mitigated_results,
    "gender",
    save_path='./demo_results/selection_rates.png'
)
print("✓ Selection rates chart saved")

# 2. Comparison chart
fig2 = visualizer.plot_bias_mitigation_comparison(
    baseline_results_final,
    mitigated_results,
    save_path='./demo_results/mitigation_comparison.png'
)
print("✓ Mitigation comparison chart saved")

# 3. ROC curves
mitigated_proba = mitigated_model.model.predict_proba(X_test_final)[:, 1]
fig3 = visualizer.plot_roc_curves_by_group(
    y_test_final,
    mitigated_proba,
    protected_test_final,
    "gender",
    save_path='./demo_results/roc_curves.png'
)
print("✓ ROC curves saved")

# 4. Confusion matrices
fig4 = visualizer.plot_confusion_matrix_by_group(
    y_test_final,
    mitigated_pred,
    protected_test_final,
    "gender",
    save_path='./demo_results/confusion_matrices.png'
)
print("✓ Confusion matrices saved")

print("\nAll visualizations saved to: ./demo_results/")

print("\n" + "="*80)
print("DEMO COMPLETE!")
print("="*80)
print("\nKey Takeaways:")
print("1. The baseline model showed significant bias across demographic groups")
print("2. Bias mitigation techniques improved fairness metrics")
print(f"3. Fairness criteria passed increased from {baseline_fair_final}/4 to {mitigated_fair_count}/4")
print("4. Visualizations show the impact of bias mitigation")
print("\nNext Steps:")
print("- Explore different datasets: 'credit', 'healthcare', 'criminal_justice'")
print("- Experiment with different mitigation strategies")
print("- Review the generated visualizations in ./demo_results/")
print("\n" + "="*80)