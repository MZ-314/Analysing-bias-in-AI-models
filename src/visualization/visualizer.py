"""
Visualization Module for Bias Analysis
Creates comprehensive visualizations for bias detection and mitigation results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


class BiasVisualizer:
    """Create visualizations for bias analysis"""
    
    def __init__(self, style: str = 'seaborn-v0_8'):
        """
        Parameters:
        -----------
        style : str
            Matplotlib style
        """
        plt.style.use('default')
        sns.set_palette("husl")
        self.colors = sns.color_palette("husl", 8)
    
    def plot_selection_rates(self,
                            results: Dict,
                            protected_attr_name: str,
                            save_path: Optional[str] = None):
        """
        Plot selection rates by protected group
        
        Parameters:
        -----------
        results : dict
            Fairness metrics results
        protected_attr_name : str
            Name of protected attribute
        save_path : str, optional
            Path to save figure
        """
        selection_rates = results['demographic_parity']['selection_rates']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        groups = list(selection_rates.keys())
        rates = [selection_rates[g] for g in groups]
        
        bars = ax.bar(groups, rates, color=self.colors[:len(groups)], alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for bar, rate in zip(bars, rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{rate:.3f}\n({rate*100:.1f}%)',
                   ha='center', va='bottom', fontweight='bold')
        
        # Add 80% rule threshold line
        disparate_impact = results['demographic_parity']['disparate_impact']
        if disparate_impact < 1.0:
            threshold = max(rates) * 0.8
            ax.axhline(y=threshold, color='red', linestyle='--', 
                      label=f'80% Rule Threshold ({threshold:.3f})', linewidth=2)
        
        ax.set_xlabel(protected_attr_name, fontsize=12, fontweight='bold')
        ax.set_ylabel('Selection Rate', fontsize=12, fontweight='bold')
        ax.set_title(f'Selection Rates by {protected_attr_name}\n' + 
                    f'Disparate Impact: {disparate_impact:.3f}',
                    fontsize=14, fontweight='bold')
        ax.set_ylim(0, max(rates) * 1.2)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        return fig
    
    def plot_confusion_matrix_by_group(self,
                                      y_true: np.ndarray,
                                      y_pred: np.ndarray,
                                      protected_attr: np.ndarray,
                                      protected_attr_name: str,
                                      save_path: Optional[str] = None):
        """
        Plot confusion matrices for each protected group
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
        protected_attr : array-like
            Protected attribute values
        protected_attr_name : str
            Name of protected attribute
        save_path : str, optional
            Path to save figure
        """
        from sklearn.metrics import confusion_matrix
        
        unique_groups = np.unique(protected_attr)
        n_groups = len(unique_groups)
        
        fig, axes = plt.subplots(1, n_groups, figsize=(6*n_groups, 5))
        
        if n_groups == 1:
            axes = [axes]
        
        for idx, (group, ax) in enumerate(zip(unique_groups, axes)):
            mask = protected_attr == group
            cm = confusion_matrix(y_true[mask], y_pred[mask])
            
            # Normalize
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            # Plot
            sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                       ax=ax, cbar=True, square=True,
                       xticklabels=['Negative', 'Positive'],
                       yticklabels=['Negative', 'Positive'])
            
            ax.set_title(f'{group}\n(n={sum(mask)})', fontsize=12, fontweight='bold')
            ax.set_xlabel('Predicted', fontsize=11)
            ax.set_ylabel('Actual', fontsize=11)
        
        plt.suptitle(f'Confusion Matrices by {protected_attr_name}',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        return fig
    
    def plot_fairness_metrics_comparison(self,
                                        model_results: Dict[str, Dict],
                                        save_path: Optional[str] = None):
        """
        Compare fairness metrics across multiple models
        
        Parameters:
        -----------
        model_results : dict
            Dictionary mapping model names to their fairness results
        save_path : str, optional
            Path to save figure
        """
        metrics_data = []
        
        for model_name, results in model_results.items():
            metrics_data.append({
                'Model': model_name,
                'Disparate Impact': results['demographic_parity']['disparate_impact'],
                'Stat Parity Diff': results['demographic_parity']['statistical_parity_difference'],
                'Eq Odds Diff': results['equalized_odds']['equalized_odds_difference'],
                'TPR Diff': results['equal_opportunity']['tpr_difference'],
                'PPV Diff': results['predictive_parity']['ppv_difference']
            })
        
        df = pd.DataFrame(metrics_data)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        metrics = ['Disparate Impact', 'Stat Parity Diff', 'Eq Odds Diff', 
                  'TPR Diff', 'PPV Diff']
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            bars = ax.bar(df['Model'], df[metric], color=self.colors[:len(df)], 
                         alpha=0.7, edgecolor='black')
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontweight='bold')
            
            # Add fairness thresholds
            if metric == 'Disparate Impact':
                ax.axhline(y=0.8, color='green', linestyle='--', 
                          label='Fair Threshold (≥0.8)', linewidth=2)
                ax.set_ylim(0, 1.1)
            else:
                ax.axhline(y=0.1, color='green', linestyle='--',
                          label='Fair Threshold (≤0.1)', linewidth=2)
            
            ax.set_title(metric, fontsize=12, fontweight='bold')
            ax.set_ylabel('Value', fontsize=11)
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Remove extra subplot
        fig.delaxes(axes[-1])
        
        plt.suptitle('Fairness Metrics Comparison Across Models',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        return fig
    
    def plot_roc_curves_by_group(self,
                                y_true: np.ndarray,
                                y_proba: np.ndarray,
                                protected_attr: np.ndarray,
                                protected_attr_name: str,
                                save_path: Optional[str] = None):
        """
        Plot ROC curves for each protected group
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_proba : array-like
            Predicted probabilities
        protected_attr : array-like
            Protected attribute values
        protected_attr_name : str
            Name of protected attribute
        save_path : str, optional
            Path to save figure
        """
        from sklearn.metrics import roc_curve, auc
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        unique_groups = np.unique(protected_attr)
        
        for idx, group in enumerate(unique_groups):
            mask = protected_attr == group
            
            fpr, tpr, _ = roc_curve(y_true[mask], y_proba[mask])
            roc_auc = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, color=self.colors[idx], lw=2,
                   label=f'{group} (AUC = {roc_auc:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
        
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title(f'ROC Curves by {protected_attr_name}',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        return fig
    
    def plot_calibration_curves(self,
                               y_true: np.ndarray,
                               y_proba: np.ndarray,
                               protected_attr: np.ndarray,
                               protected_attr_name: str,
                               n_bins: int = 10,
                               save_path: Optional[str] = None):
        """
        Plot calibration curves for each protected group
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_proba : array-like
            Predicted probabilities
        protected_attr : array-like
            Protected attribute values
        protected_attr_name : str
            Name of protected attribute
        n_bins : int
            Number of bins for calibration
        save_path : str, optional
            Path to save figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        unique_groups = np.unique(protected_attr)
        
        for idx, group in enumerate(unique_groups):
            mask = protected_attr == group
            
            # Calculate calibration
            bins = np.linspace(0, 1, n_bins + 1)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            
            bin_true_rates = []
            for i in range(n_bins):
                bin_mask = mask & (y_proba >= bins[i]) & (y_proba < bins[i + 1])
                if np.sum(bin_mask) > 0:
                    bin_true_rate = np.mean(y_true[bin_mask])
                else:
                    bin_true_rate = np.nan
                bin_true_rates.append(bin_true_rate)
            
            # Plot
            ax.plot(bin_centers, bin_true_rates, 'o-', color=self.colors[idx],
                   label=group, markersize=8, linewidth=2)
        
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Perfect Calibration')
        
        ax.set_xlabel('Predicted Probability', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Probability', fontsize=12, fontweight='bold')
        ax.set_title(f'Calibration Curves by {protected_attr_name}',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=11)
        ax.grid(alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        return fig
    
    def plot_bias_mitigation_comparison(self,
                                       before_results: Dict,
                                       after_results: Dict,
                                       save_path: Optional[str] = None):
        """
        Compare metrics before and after bias mitigation
        
        Parameters:
        -----------
        before_results : dict
            Fairness results before mitigation
        after_results : dict
            Fairness results after mitigation
        save_path : str, optional
            Path to save figure
        """
        metrics = {
            'Disparate Impact': [
                before_results['demographic_parity']['disparate_impact'],
                after_results['demographic_parity']['disparate_impact']
            ],
            'Stat Parity Diff': [
                before_results['demographic_parity']['statistical_parity_difference'],
                after_results['demographic_parity']['statistical_parity_difference']
            ],
            'Eq Odds Diff': [
                before_results['equalized_odds']['equalized_odds_difference'],
                after_results['equalized_odds']['equalized_odds_difference']
            ],
            'TPR Diff': [
                before_results['equal_opportunity']['tpr_difference'],
                after_results['equal_opportunity']['tpr_difference']
            ]
        }
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(metrics))
        width = 0.35
        
        before_vals = [metrics[m][0] for m in metrics]
        after_vals = [metrics[m][1] for m in metrics]
        
        bars1 = ax.bar(x - width/2, before_vals, width, label='Before Mitigation',
                      color=self.colors[0], alpha=0.7, edgecolor='black')
        bars2 = ax.bar(x + width/2, after_vals, width, label='After Mitigation',
                      color=self.colors[2], alpha=0.7, edgecolor='black')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Fairness Metric', fontsize=12, fontweight='bold')
        ax.set_ylabel('Value', fontsize=12, fontweight='bold')
        ax.set_title('Bias Mitigation Impact on Fairness Metrics',
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics.keys())
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        return fig
    
    def create_comprehensive_report(self,
                                   y_true: np.ndarray,
                                   y_pred: np.ndarray,
                                   y_proba: np.ndarray,
                                   protected_attr: np.ndarray,
                                   protected_attr_name: str,
                                   fairness_results: Dict,
                                   save_dir: str = './'):
        """
        Create a comprehensive visualization report
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
        y_proba : array-like
            Predicted probabilities
        protected_attr : array-like
            Protected attribute values
        protected_attr_name : str
            Name of protected attribute
        fairness_results : dict
            Fairness metrics results
        save_dir : str
            Directory to save figures
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        print("Generating comprehensive bias analysis visualizations...")
        
        # 1. Selection rates
        print("  - Selection rates plot...")
        self.plot_selection_rates(
            fairness_results, protected_attr_name,
            save_path=f'{save_dir}/selection_rates.png'
        )
        
        # 2. Confusion matrices
        print("  - Confusion matrices...")
        self.plot_confusion_matrix_by_group(
            y_true, y_pred, protected_attr, protected_attr_name,
            save_path=f'{save_dir}/confusion_matrices.png'
        )
        
        # 3. ROC curves
        print("  - ROC curves...")
        self.plot_roc_curves_by_group(
            y_true, y_proba, protected_attr, protected_attr_name,
            save_path=f'{save_dir}/roc_curves.png'
        )
        
        # 4. Calibration curves
        print("  - Calibration curves...")
        self.plot_calibration_curves(
            y_true, y_proba, protected_attr, protected_attr_name,
            save_path=f'{save_dir}/calibration_curves.png'
        )
        
        print(f"\nAll visualizations saved to: {save_dir}")


if __name__ == "__main__":
    print("Testing Visualization Module\n")
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    
    y_true = np.random.choice([0, 1], n_samples)
    y_pred = np.random.choice([0, 1], n_samples)
    y_proba = np.random.rand(n_samples)
    protected_attr = np.random.choice(['Group A', 'Group B', 'Group C'], n_samples)
    
    # Create sample fairness results
    sample_results = {
        'demographic_parity': {
            'selection_rates': {'Group A': 0.45, 'Group B': 0.30, 'Group C': 0.38},
            'disparate_impact': 0.67,
            'statistical_parity_difference': 0.15
        },
        'equalized_odds': {
            'true_positive_rates': {'Group A': 0.75, 'Group B': 0.65, 'Group C': 0.70},
            'false_positive_rates': {'Group A': 0.20, 'Group B': 0.30, 'Group C': 0.25},
            'equalized_odds_difference': 0.10
        },
        'equal_opportunity': {
            'true_positive_rates': {'Group A': 0.75, 'Group B': 0.65, 'Group C': 0.70},
            'tpr_difference': 0.10
        },
        'predictive_parity': {
            'positive_predictive_values': {'Group A': 0.70, 'Group B': 0.60, 'Group C': 0.65},
            'ppv_difference': 0.10
        }
    }
    
    # Test visualizer
    visualizer = BiasVisualizer()
    
    print("Creating sample visualizations...")
    
    # Selection rates
    fig1 = visualizer.plot_selection_rates(sample_results, "Test Attribute")
    print("✓ Selection rates plot created")
    
    # ROC curves
    fig2 = visualizer.plot_roc_curves_by_group(y_true, y_proba, protected_attr, "Test Attribute")
    print("✓ ROC curves plot created")
    
    print("\nVisualization module test complete!")