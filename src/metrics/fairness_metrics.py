"""
Fairness Metrics Module
Implements various fairness metrics for bias detection in AI models
Based on the research paper's methodology
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')


class FairnessMetrics:
    """Calculate various fairness metrics for bias assessment"""
    
    def __init__(self):
        self.metrics_results = {}
    
    def demographic_parity(self, 
                          y_pred: np.ndarray, 
                          protected_attr: np.ndarray,
                          favorable_label: int = 1) -> Dict:
        """
        Calculate Demographic Parity (Statistical Parity)
        Measures if different groups receive positive outcomes at equal rates
        
        Parameters:
        -----------
        y_pred : array-like
            Predicted labels
        protected_attr : array-like
            Protected attribute values
        favorable_label : int
            The favorable outcome label
        
        Returns:
        --------
        Dictionary with demographic parity metrics
        """
        unique_groups = np.unique(protected_attr)
        selection_rates = {}
        
        for group in unique_groups:
            mask = protected_attr == group
            selection_rate = np.mean(y_pred[mask] == favorable_label)
            selection_rates[group] = selection_rate
        
        # Calculate disparate impact (ratio of lowest to highest rate)
        rates = list(selection_rates.values())
        disparate_impact = min(rates) / max(rates) if max(rates) > 0 else 0
        
        # Calculate statistical parity difference
        stat_parity_diff = max(rates) - min(rates)
        
        return {
            'selection_rates': selection_rates,
            'disparate_impact': disparate_impact,
            'statistical_parity_difference': stat_parity_diff,
            'is_fair': disparate_impact >= 0.8  # 80% rule
        }
    
    def equalized_odds(self,
                      y_true: np.ndarray,
                      y_pred: np.ndarray,
                      protected_attr: np.ndarray) -> Dict:
        """
        Calculate Equalized Odds
        Measures if true positive rate and false positive rate are equal across groups
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
        protected_attr : array-like
            Protected attribute values
        
        Returns:
        --------
        Dictionary with equalized odds metrics
        """
        unique_groups = np.unique(protected_attr)
        tpr_dict = {}
        fpr_dict = {}
        
        for group in unique_groups:
            mask = protected_attr == group
            y_true_group = y_true[mask]
            y_pred_group = y_pred[mask]
            
            tn, fp, fn, tp = confusion_matrix(y_true_group, y_pred_group, 
                                             labels=[0, 1]).ravel()
            
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            tpr_dict[group] = tpr
            fpr_dict[group] = fpr
        
        # Calculate differences
        tpr_values = list(tpr_dict.values())
        fpr_values = list(fpr_dict.values())
        
        tpr_difference = max(tpr_values) - min(tpr_values)
        fpr_difference = max(fpr_values) - min(fpr_values)
        
        # Average absolute difference
        eq_odds_diff = (tpr_difference + fpr_difference) / 2
        
        return {
            'true_positive_rates': tpr_dict,
            'false_positive_rates': fpr_dict,
            'tpr_difference': tpr_difference,
            'fpr_difference': fpr_difference,
            'equalized_odds_difference': eq_odds_diff,
            'is_fair': eq_odds_diff <= 0.1  # Threshold
        }
    
    def equal_opportunity(self,
                         y_true: np.ndarray,
                         y_pred: np.ndarray,
                         protected_attr: np.ndarray) -> Dict:
        """
        Calculate Equal Opportunity
        Measures if true positive rate is equal across groups
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
        protected_attr : array-like
            Protected attribute values
        
        Returns:
        --------
        Dictionary with equal opportunity metrics
        """
        unique_groups = np.unique(protected_attr)
        tpr_dict = {}
        
        for group in unique_groups:
            mask = protected_attr == group
            y_true_group = y_true[mask]
            y_pred_group = y_pred[mask]
            
            # Calculate TPR
            tp = np.sum((y_true_group == 1) & (y_pred_group == 1))
            fn = np.sum((y_true_group == 1) & (y_pred_group == 0))
            
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            tpr_dict[group] = tpr
        
        tpr_values = list(tpr_dict.values())
        tpr_difference = max(tpr_values) - min(tpr_values)
        
        return {
            'true_positive_rates': tpr_dict,
            'tpr_difference': tpr_difference,
            'is_fair': tpr_difference <= 0.1
        }
    
    def predictive_parity(self,
                         y_true: np.ndarray,
                         y_pred: np.ndarray,
                         protected_attr: np.ndarray) -> Dict:
        """
        Calculate Predictive Parity
        Measures if precision (PPV) is equal across groups
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
        protected_attr : array-like
            Protected attribute values
        
        Returns:
        --------
        Dictionary with predictive parity metrics
        """
        unique_groups = np.unique(protected_attr)
        ppv_dict = {}
        
        for group in unique_groups:
            mask = protected_attr == group
            y_true_group = y_true[mask]
            y_pred_group = y_pred[mask]
            
            tp = np.sum((y_true_group == 1) & (y_pred_group == 1))
            fp = np.sum((y_true_group == 0) & (y_pred_group == 1))
            
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
            ppv_dict[group] = ppv
        
        ppv_values = list(ppv_dict.values())
        ppv_difference = max(ppv_values) - min(ppv_values)
        
        return {
            'positive_predictive_values': ppv_dict,
            'ppv_difference': ppv_difference,
            'is_fair': ppv_difference <= 0.1
        }
    
    def calibration_metrics(self,
                           y_true: np.ndarray,
                           y_pred: np.ndarray,
                           protected_attr: np.ndarray) -> Dict:
        """
        Calculate calibration metrics across groups
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
        protected_attr : array-like
            Protected attribute values
        
        Returns:
        --------
        Dictionary with calibration metrics
        """
        unique_groups = np.unique(protected_attr)
        accuracy_dict = {}
        
        for group in unique_groups:
            mask = protected_attr == group
            y_true_group = y_true[mask]
            y_pred_group = y_pred[mask]
            
            accuracy = accuracy_score(y_true_group, y_pred_group)
            accuracy_dict[group] = accuracy
        
        acc_values = list(accuracy_dict.values())
        accuracy_difference = max(acc_values) - min(acc_values)
        
        return {
            'accuracy_by_group': accuracy_dict,
            'accuracy_difference': accuracy_difference,
            'is_fair': accuracy_difference <= 0.05
        }
    
    def calculate_all_metrics(self,
                             y_true: np.ndarray,
                             y_pred: np.ndarray,
                             protected_attr: np.ndarray,
                             protected_attr_name: str = "protected_attribute") -> Dict:
        """
        Calculate all fairness metrics at once
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
        protected_attr : array-like
            Protected attribute values
        protected_attr_name : str
            Name of the protected attribute
        
        Returns:
        --------
        Dictionary containing all fairness metrics
        """
        results = {
            'protected_attribute': protected_attr_name,
            'demographic_parity': self.demographic_parity(y_pred, protected_attr),
            'equalized_odds': self.equalized_odds(y_true, y_pred, protected_attr),
            'equal_opportunity': self.equal_opportunity(y_true, y_pred, protected_attr),
            'predictive_parity': self.predictive_parity(y_true, y_pred, protected_attr),
            'calibration': self.calibration_metrics(y_true, y_pred, protected_attr)
        }
        
        self.metrics_results = results
        return results
    
    def generate_fairness_report(self, results: Dict) -> str:
        """
        Generate a human-readable fairness report
        
        Parameters:
        -----------
        results : dict
            Results from calculate_all_metrics
        
        Returns:
        --------
        String containing formatted report
        """
        report = []
        report.append("=" * 70)
        report.append(f"FAIRNESS ANALYSIS REPORT: {results['protected_attribute']}")
        report.append("=" * 70)
        
        # Demographic Parity
        dp = results['demographic_parity']
        report.append("\n1. DEMOGRAPHIC PARITY (Statistical Parity)")
        report.append("-" * 70)
        report.append(f"   Selection Rates by Group:")
        for group, rate in dp['selection_rates'].items():
            report.append(f"      {group}: {rate:.4f} ({rate*100:.2f}%)")
        report.append(f"   Disparate Impact Ratio: {dp['disparate_impact']:.4f}")
        report.append(f"   Statistical Parity Difference: {dp['statistical_parity_difference']:.4f}")
        report.append(f"   Fair (80% rule)? {'✓ YES' if dp['is_fair'] else '✗ NO'}")
        
        # Equalized Odds
        eo = results['equalized_odds']
        report.append("\n2. EQUALIZED ODDS")
        report.append("-" * 70)
        report.append(f"   True Positive Rates:")
        for group, tpr in eo['true_positive_rates'].items():
            report.append(f"      {group}: {tpr:.4f}")
        report.append(f"   False Positive Rates:")
        for group, fpr in eo['false_positive_rates'].items():
            report.append(f"      {group}: {fpr:.4f}")
        report.append(f"   TPR Difference: {eo['tpr_difference']:.4f}")
        report.append(f"   FPR Difference: {eo['fpr_difference']:.4f}")
        report.append(f"   Fair? {'✓ YES' if eo['is_fair'] else '✗ NO'}")
        
        # Equal Opportunity
        eop = results['equal_opportunity']
        report.append("\n3. EQUAL OPPORTUNITY")
        report.append("-" * 70)
        report.append(f"   True Positive Rates:")
        for group, tpr in eop['true_positive_rates'].items():
            report.append(f"      {group}: {tpr:.4f}")
        report.append(f"   TPR Difference: {eop['tpr_difference']:.4f}")
        report.append(f"   Fair? {'✓ YES' if eop['is_fair'] else '✗ NO'}")
        
        # Predictive Parity
        pp = results['predictive_parity']
        report.append("\n4. PREDICTIVE PARITY (Precision)")
        report.append("-" * 70)
        report.append(f"   Positive Predictive Values:")
        for group, ppv in pp['positive_predictive_values'].items():
            report.append(f"      {group}: {ppv:.4f}")
        report.append(f"   PPV Difference: {pp['ppv_difference']:.4f}")
        report.append(f"   Fair? {'✓ YES' if pp['is_fair'] else '✗ NO'}")
        
        # Calibration
        cal = results['calibration']
        report.append("\n5. CALIBRATION (Accuracy by Group)")
        report.append("-" * 70)
        report.append(f"   Accuracy by Group:")
        for group, acc in cal['accuracy_by_group'].items():
            report.append(f"      {group}: {acc:.4f}")
        report.append(f"   Accuracy Difference: {cal['accuracy_difference']:.4f}")
        report.append(f"   Fair? {'✓ YES' if cal['is_fair'] else '✗ NO'}")
        
        # Overall Assessment
        report.append("\n" + "=" * 70)
        report.append("OVERALL FAIRNESS ASSESSMENT")
        report.append("=" * 70)
        
        fair_count = sum([
            dp['is_fair'],
            eo['is_fair'],
            eop['is_fair'],
            pp['is_fair'],
            cal['is_fair']
        ])
        
        report.append(f"Metrics Passed: {fair_count}/5")
        
        if fair_count == 5:
            report.append("Status: ✓ MODEL PASSES ALL FAIRNESS CRITERIA")
        elif fair_count >= 3:
            report.append("Status: ⚠ MODEL HAS MODERATE FAIRNESS ISSUES")
        else:
            report.append("Status: ✗ MODEL HAS SIGNIFICANT FAIRNESS CONCERNS")
        
        report.append("=" * 70)
        
        return "\n".join(report)
    
    def compare_models(self,
                      model_results: Dict[str, Dict]) -> pd.DataFrame:
        """
        Compare fairness metrics across multiple models
        
        Parameters:
        -----------
        model_results : dict
            Dictionary mapping model names to their fairness results
        
        Returns:
        --------
        DataFrame comparing models
        """
        comparison_data = []
        
        for model_name, results in model_results.items():
            row = {
                'Model': model_name,
                'Disparate_Impact': results['demographic_parity']['disparate_impact'],
                'Stat_Parity_Diff': results['demographic_parity']['statistical_parity_difference'],
                'Eq_Odds_Diff': results['equalized_odds']['equalized_odds_difference'],
                'TPR_Diff': results['equal_opportunity']['tpr_difference'],
                'PPV_Diff': results['predictive_parity']['ppv_difference'],
                'Acc_Diff': results['calibration']['accuracy_difference'],
                'Fairness_Score': sum([
                    results['demographic_parity']['is_fair'],
                    results['equalized_odds']['is_fair'],
                    results['equal_opportunity']['is_fair'],
                    results['predictive_parity']['is_fair'],
                    results['calibration']['is_fair']
                ]) / 5.0
            }
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)


if __name__ == "__main__":
    # Test the fairness metrics
    print("Testing Fairness Metrics Module\n")
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    
    y_true = np.random.choice([0, 1], n_samples)
    y_pred = np.random.choice([0, 1], n_samples)
    protected_attr = np.random.choice(['Group A', 'Group B'], n_samples)
    
    # Create biased predictions for Group B
    bias_mask = protected_attr == 'Group B'
    y_pred[bias_mask] = np.random.choice([0, 1], sum(bias_mask), p=[0.7, 0.3])
    
    # Calculate metrics
    fm = FairnessMetrics()
    results = fm.calculate_all_metrics(y_true, y_pred, protected_attr, "Test Attribute")
    
    # Generate report
    report = fm.generate_fairness_report(results)
    print(report)