import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class BiasAnalyzer:
    """
    Comprehensive AI Bias Analysis and Mitigation System
    Based on the research paper: "Analyzing Bias in AI Models"
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.fairness_metrics = {}
        self.bias_report = {}
        
    def load_and_preprocess_data(self, data_path: str = None, demo_data: bool = True) -> Tuple[pd.DataFrame, str]:
        """
        Load and preprocess data for bias analysis
        """
        if demo_data:
            # Create synthetic dataset for demonstration
            np.random.seed(42)
            n_samples = 1000
            
            # Generate synthetic hiring dataset
            data = {
                'age': np.random.randint(22, 65, n_samples),
                'education_years': np.random.randint(12, 20, n_samples),
                'experience_years': np.random.randint(0, 30, n_samples),
                'test_score': np.random.normal(75, 15, n_samples),
                'gender': np.random.choice(['Male', 'Female'], n_samples, p=[0.6, 0.4]),
                'race': np.random.choice(['White', 'Black', 'Hispanic', 'Asian'], n_samples, p=[0.5, 0.2, 0.2, 0.1])
            }
            
            # Introduce bias: higher hiring rates for certain groups
            df = pd.DataFrame(data)
            
            # Biased target generation (simulating historical bias)
            base_prob = 0.3
            bias_factors = {
                'Male': 0.15, 'Female': -0.15,
                'White': 0.1, 'Black': -0.1, 'Hispanic': -0.05, 'Asian': 0.05
            }
            
            hire_prob = base_prob
            for _, row in df.iterrows():
                prob = base_prob
                prob += bias_factors[row['gender']]
                prob += bias_factors[row['race']]
                prob += (row['test_score'] - 75) * 0.01
                prob += row['experience_years'] * 0.005
                prob = max(0, min(1, prob))
                df.loc[_, 'hired'] = np.random.binomial(1, prob)
            
            target_column = 'hired'
            print("Demo dataset created with intentional bias")
            
        else:
            df = pd.read_csv(data_path)
            target_column = input("Enter the target column name: ")
        
        return df, target_column
    
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features while preserving original for bias analysis
        """
        df_encoded = df.copy()
        categorical_columns = df.select_dtypes(include=['object']).columns
        
        for column in categorical_columns:
            if column not in ['hired']:  # Don't encode target if it's categorical
                le = LabelEncoder()
                df_encoded[f'{column}_encoded'] = le.fit_transform(df[column])
                self.label_encoders[column] = le
        
        return df_encoded
    
    def calculate_fairness_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                 sensitive_attr: np.ndarray, attr_name: str) -> Dict[str, float]:
        """
        Calculate various fairness metrics as mentioned in the research paper
        """
        metrics = {}
        unique_groups = np.unique(sensitive_attr)
        
        # Demographic Parity (Statistical Parity)
        group_positive_rates = {}
        for group in unique_groups:
            group_mask = sensitive_attr == group
            positive_rate = np.mean(y_pred[group_mask])
            group_positive_rates[group] = positive_rate
        
        # Calculate demographic parity difference
        max_rate = max(group_positive_rates.values())
        min_rate = min(group_positive_rates.values())
        metrics['demographic_parity_diff'] = max_rate - min_rate
        
        # Equalized Odds
        group_tpr = {}  # True Positive Rate
        group_fpr = {}  # False Positive Rate
        
        for group in unique_groups:
            group_mask = sensitive_attr == group
            y_true_group = y_true[group_mask]
            y_pred_group = y_pred[group_mask]
            
            # True Positive Rate
            tp = np.sum((y_true_group == 1) & (y_pred_group == 1))
            fn = np.sum((y_true_group == 1) & (y_pred_group == 0))
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            group_tpr[group] = tpr
            
            # False Positive Rate
            fp = np.sum((y_true_group == 0) & (y_pred_group == 1))
            tn = np.sum((y_true_group == 0) & (y_pred_group == 0))
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            group_fpr[group] = fpr
        
        # Equalized odds difference
        max_tpr = max(group_tpr.values())
        min_tpr = min(group_tpr.values())
        metrics['equalized_odds_diff'] = max_tpr - min_tpr
        
        # Disparate Impact
        if len(unique_groups) >= 2:
            rates = list(group_positive_rates.values())
            metrics['disparate_impact'] = min(rates) / max(rates) if max(rates) > 0 else 0
        
        # Store detailed group metrics
        metrics[f'{attr_name}_group_rates'] = group_positive_rates
        metrics[f'{attr_name}_group_tpr'] = group_tpr
        metrics[f'{attr_name}_group_fpr'] = group_fpr
        
        return metrics
    
    def detect_bias(self, df: pd.DataFrame, target_col: str, sensitive_attrs: List[str]) -> Dict[str, Any]:
        """
        Comprehensive bias detection using multiple fairness metrics
        """
        print("🔍 Detecting bias in the dataset...")
        
        # Prepare features and target
        feature_cols = [col for col in df.columns if col not in [target_col] + sensitive_attrs]
        numerical_features = [col for col in feature_cols if col.endswith('_encoded') or df[col].dtype in ['int64', 'float64']]
        
        X = df[numerical_features]
        y = df[target_col]
        
        # Train initial model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = self.model.predict(X_test_scaled)
        
        # Calculate fairness metrics for each sensitive attribute
        bias_results = {}
        test_indices = y_test.index
        
        for attr in sensitive_attrs:
            sensitive_values = df.loc[test_indices, attr].values
            fairness_metrics = self.calculate_fairness_metrics(y_test.values, y_pred, sensitive_values, attr)
            bias_results[attr] = fairness_metrics
        
        # Overall model performance
        bias_results['overall_accuracy'] = accuracy_score(y_test, y_pred)
        bias_results['X_test'] = X_test_scaled
        bias_results['y_test'] = y_test
        bias_results['y_pred'] = y_pred
        bias_results['test_indices'] = test_indices
        
        self.bias_report = bias_results
        return bias_results
    
    def apply_reweighting(self, df: pd.DataFrame, target_col: str, sensitive_attr: str) -> Dict[int, float]:
        """
        Apply reweighting technique for bias mitigation (preprocessing)
        """
        print(f"⚖️  Applying reweighting for {sensitive_attr}...")
        
        weights = {}
        
        # Calculate weights to balance representation
        for idx in df.index:
            weights[idx] = 1.0
        
        for group in df[sensitive_attr].unique():
            for target_val in df[target_col].unique():
                mask = (df[sensitive_attr] == group) & (df[target_col] == target_val)
                count = np.sum(mask)
                if count > 0:
                    # Weight inversely proportional to frequency
                    total_in_group = np.sum(df[sensitive_attr] == group)
                    total_with_target = np.sum(df[target_col] == target_val)
                    expected_count = (total_in_group * total_with_target) / len(df)
                    weight = expected_count / count if count > 0 else 1
                    
                    # Update weights for matching indices
                    matching_indices = df[mask].index
                    for idx in matching_indices:
                        weights[idx] = weight
        
        return weights
    
    def apply_adversarial_debiasing(self, X_train: np.ndarray, y_train: np.ndarray, 
                                  sensitive_train: np.ndarray) -> Any:
        """
        Simplified adversarial debiasing approach (in-processing)
        """
        print("🎭 Applying adversarial debiasing...")
        
        # Create a fairness-aware logistic regression
        # This is a simplified version - in practice, you'd use more sophisticated adversarial networks
        from sklearn.linear_model import LogisticRegression
        
        model = LogisticRegression(class_weight='balanced', random_state=42)
        model.fit(X_train, y_train)
        
        return model
    
    def apply_postprocessing(self, y_pred: np.ndarray, sensitive_attr: np.ndarray, 
                           target_attr: np.ndarray) -> np.ndarray:
        """
        Apply post-processing techniques to reduce bias
        """
        print("🔧 Applying post-processing bias mitigation...")
        
        y_pred_adjusted = y_pred.copy()
        
        # Equalized odds post-processing
        unique_groups = np.unique(sensitive_attr)
        
        # Calculate group-specific thresholds
        for group in unique_groups:
            group_mask = sensitive_attr == group
            group_pred = y_pred[group_mask]
            group_true = target_attr[group_mask]
            
            # Adjust predictions to equalize true positive rates
            if len(np.unique(group_true)) > 1:  # Only if both classes exist
                # Simple threshold adjustment
                current_tpr = np.mean(group_pred[group_true == 1])
                target_tpr = 0.5  # Target TPR
                
                if current_tpr > target_tpr:
                    # Reduce some positive predictions
                    pos_indices = np.where((group_mask) & (y_pred == 1))[0]
                    reduce_count = int(len(pos_indices) * (current_tpr - target_tpr))
                    reduce_indices = np.random.choice(pos_indices, min(reduce_count, len(pos_indices)), replace=False)
                    y_pred_adjusted[reduce_indices] = 0
        
        return y_pred_adjusted
    
    def mitigate_bias(self, df: pd.DataFrame, target_col: str, sensitive_attrs: List[str], 
                     method: str = 'reweighting') -> Dict[str, Any]:
        """
        Apply bias mitigation techniques
        """
        print(f"🛠️  Applying {method} bias mitigation...")
        
        # Prepare data
        feature_cols = [col for col in df.columns if col not in [target_col] + sensitive_attrs]
        numerical_features = [col for col in feature_cols if col.endswith('_encoded') or df[col].dtype in ['int64', 'float64']]
        
        X = df[numerical_features]
        y = df[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        if method == 'reweighting':
            # Preprocessing: Apply reweighting
            weights_dict = self.apply_reweighting(df.loc[X_train.index], target_col, sensitive_attrs[0])
            weights_train = np.array([weights_dict.get(idx, 1.0) for idx in X_train.index])
            
            # Train model with weights
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train, sample_weight=weights_train)
            
        elif method == 'adversarial':
            # In-processing: Adversarial debiasing
            sensitive_train = df.loc[X_train.index, sensitive_attrs[0]].values
            model = self.apply_adversarial_debiasing(X_train_scaled, y_train, sensitive_train)
            
        else:  # postprocessing
            # Train regular model first
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = model.predict(X_test_scaled)
        
        if method == 'postprocessing':
            # Apply post-processing
            test_indices = y_test.index
            sensitive_test = df.loc[test_indices, sensitive_attrs[0]].values
            y_pred = self.apply_postprocessing(y_pred, sensitive_test, y_test.values)
        
        # Evaluate fairness after mitigation
        mitigation_results = {}
        test_indices = y_test.index
        
        for attr in sensitive_attrs:
            sensitive_values = df.loc[test_indices, attr].values
            fairness_metrics = self.calculate_fairness_metrics(y_test.values, y_pred, sensitive_values, attr)
            mitigation_results[attr] = fairness_metrics
        
        mitigation_results['overall_accuracy'] = accuracy_score(y_test, y_pred)
        mitigation_results['method'] = method
        
        return mitigation_results
    
    def visualize_bias_results(self, bias_results: Dict[str, Any], sensitive_attrs: List[str]):
        """
        Visualize bias detection results
        """
        print("📊 Generating bias visualization...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('AI Bias Analysis Results', fontsize=16, fontweight='bold')
        
        # Plot 1: Demographic Parity by Group
        ax1 = axes[0, 0]
        attr = sensitive_attrs[0]
        group_rates = bias_results[attr][f'{attr}_group_rates']
        groups = list(group_rates.keys())
        rates = list(group_rates.values())
        
        bars = ax1.bar(groups, rates, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'][:len(groups)])
        ax1.set_title(f'Positive Prediction Rate by {attr}')
        ax1.set_ylabel('Positive Prediction Rate')
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, rate in zip(bars, rates):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{rate:.3f}', ha='center', va='bottom')
        
        # Plot 2: Fairness Metrics Comparison
        ax2 = axes[0, 1]
        metrics_names = ['Demographic Parity Diff', 'Equalized Odds Diff', 'Disparate Impact']
        metrics_values = [
            bias_results[attr]['demographic_parity_diff'],
            bias_results[attr]['equalized_odds_diff'],
            bias_results[attr]['disparate_impact']
        ]
        
        bars = ax2.bar(metrics_names, metrics_values, color=['orange', 'purple', 'brown'])
        ax2.set_title('Fairness Metrics Overview')
        ax2.set_ylabel('Metric Value')
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        # Add value labels
        for bar, value in zip(bars, metrics_values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom')
        
        # Plot 3: True Positive Rate by Group
        ax3 = axes[1, 0]
        group_tpr = bias_results[attr][f'{attr}_group_tpr']
        groups_tpr = list(group_tpr.keys())
        tpr_values = list(group_tpr.values())
        
        bars = ax3.bar(groups_tpr, tpr_values, color=['lightblue', 'pink', 'lightgreen', 'khaki'][:len(groups_tpr)])
        ax3.set_title(f'True Positive Rate by {attr}')
        ax3.set_ylabel('True Positive Rate')
        ax3.set_ylim(0, 1)
        
        # Add value labels
        for bar, tpr in zip(bars, tpr_values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{tpr:.3f}', ha='center', va='bottom')
        
        # Plot 4: Bias Summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Create summary text
        summary_text = f"""
        Bias Analysis Summary
        
        Overall Accuracy: {bias_results['overall_accuracy']:.3f}
        
        {attr} Analysis:
        • Demographic Parity Diff: {bias_results[attr]['demographic_parity_diff']:.3f}
        • Equalized Odds Diff: {bias_results[attr]['equalized_odds_diff']:.3f}
        • Disparate Impact: {bias_results[attr]['disparate_impact']:.3f}
        
        Bias Level:
        {'HIGH BIAS DETECTED' if bias_results[attr]['demographic_parity_diff'] > 0.1 else 'MODERATE BIAS' if bias_results[attr]['demographic_parity_diff'] > 0.05 else 'LOW BIAS'}
        """
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=12, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
    
    def generate_bias_report(self, bias_results: Dict[str, Any], mitigation_results: Dict[str, Any] = None) -> str:
        """
        Generate comprehensive bias analysis report
        """
        report = """
        ===============================================================================
                                   AI BIAS ANALYSIS REPORT
        ===============================================================================
        
        """
        
        report += f"BIAS DETECTION RESULTS:\n"
        report += f"{'='*50}\n"
        report += f"Overall Model Accuracy: {bias_results['overall_accuracy']:.4f}\n\n"
        
        for attr in bias_results.keys():
            if attr not in ['overall_accuracy', 'X_test', 'y_test', 'y_pred', 'test_indices']:
                report += f"Analysis for {attr.upper()}:\n"
                report += f"   • Demographic Parity Difference: {bias_results[attr]['demographic_parity_diff']:.4f}\n"
                report += f"   • Equalized Odds Difference: {bias_results[attr]['equalized_odds_diff']:.4f}\n"
                report += f"   • Disparate Impact Ratio: {bias_results[attr]['disparate_impact']:.4f}\n"
                
                # Bias level assessment
                dp_diff = bias_results[attr]['demographic_parity_diff']
                if dp_diff > 0.1:
                    bias_level = "HIGH BIAS"
                elif dp_diff > 0.05:
                    bias_level = "MODERATE BIAS"
                else:
                    bias_level = "LOW BIAS"
                
                report += f"   • Bias Level: {bias_level}\n\n"
        
        if mitigation_results:
            report += f"BIAS MITIGATION RESULTS:\n"
            report += f"{'='*50}\n"
            report += f"Mitigation Method: {mitigation_results['method'].upper()}\n"
            report += f"Post-mitigation Accuracy: {mitigation_results['overall_accuracy']:.4f}\n\n"
            
            for attr in mitigation_results.keys():
                if attr not in ['overall_accuracy', 'method']:
                    report += f"Post-mitigation {attr.upper()} Analysis:\n"
                    report += f"   • Demographic Parity Difference: {mitigation_results[attr]['demographic_parity_diff']:.4f}\n"
                    report += f"   • Equalized Odds Difference: {mitigation_results[attr]['equalized_odds_diff']:.4f}\n"
                    report += f"   • Disparate Impact Ratio: {mitigation_results[attr]['disparate_impact']:.4f}\n"
                    
                    # Improvement calculation
                    original_dp = bias_results[attr]['demographic_parity_diff']
                    new_dp = mitigation_results[attr]['demographic_parity_diff']
                    improvement = ((original_dp - new_dp) / original_dp * 100) if original_dp > 0 else 0
                    report += f"   • Bias Reduction: {improvement:.2f}%\n\n"
        
        report += f"RECOMMENDATIONS:\n"
        report += f"{'='*50}\n"
        report += f"1. Implement continuous monitoring of fairness metrics\n"
        report += f"2. Regularly audit model performance across demographic groups\n"
        report += f"3. Consider collecting more balanced training data\n"
        report += f"4. Apply appropriate bias mitigation techniques based on use case\n"
        report += f"5. Ensure compliance with ethical AI guidelines and regulations\n\n"
        
        report += f"Generated using AI Bias Analysis System v1.0\n"
        report += f"Based on research: 'Analyzing Bias in AI Models'\n"
        report += f"===============================================================================\n"
        
        return report

def main():
    """
    Main function to demonstrate the AI Bias Analysis System
    """
    print("AI Bias Analysis and Mitigation System")
    print("=" * 50)
    
    # Initialize the bias analyzer
    analyzer = BiasAnalyzer()
    
    # Load and preprocess data
    df, target_col = analyzer.load_and_preprocess_data(demo_data=True)
    print(f"Dataset shape: {df.shape}")
    print(f"Target column: {target_col}")
    
    # Encode categorical features
    df_encoded = analyzer.encode_categorical_features(df)
    print(f"Encoded dataset shape: {df_encoded.shape}")
    
    # Define sensitive attributes for bias analysis
    sensitive_attrs = ['gender', 'race']
    
    # Detect bias
    bias_results = analyzer.detect_bias(df_encoded, target_col, sensitive_attrs)
    
    # Visualize results
    analyzer.visualize_bias_results(bias_results, sensitive_attrs)
    
    # Apply bias mitigation
    print("\nApplying bias mitigation techniques...")
    
    # Try different mitigation methods
    mitigation_methods = ['reweighting', 'adversarial', 'postprocessing']
    
    best_method = None
    best_improvement = 0
    
    for method in mitigation_methods:
        print(f"\n--- Testing {method.upper()} method ---")
        mitigation_results = analyzer.mitigate_bias(df_encoded, target_col, sensitive_attrs, method)
        
        # Calculate improvement
        original_bias = bias_results[sensitive_attrs[0]]['demographic_parity_diff']
        new_bias = mitigation_results[sensitive_attrs[0]]['demographic_parity_diff']
        improvement = ((original_bias - new_bias) / original_bias * 100) if original_bias > 0 else 0
        
        print(f"Bias reduction: {improvement:.2f}%")
        
        if improvement > best_improvement:
            best_improvement = improvement
            best_method = method
            best_mitigation_results = mitigation_results
    
    print(f"\nBest mitigation method: {best_method.upper()} with {best_improvement:.2f}% bias reduction")
    
    # Generate comprehensive report
    report = analyzer.generate_bias_report(bias_results, best_mitigation_results)
    print(report)
    
    # Save report to file
    with open('bias_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    print("Report saved to 'bias_analysis_report.txt'")

if __name__ == "__main__":
    main()