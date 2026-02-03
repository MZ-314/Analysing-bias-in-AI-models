"""
Main AI Bias Analysis System
Complete implementation demonstrating bias detection and mitigation
Based on the research paper: "Analyzing Bias in AI Models"
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from src.data.data_generator import BiasedDataGenerator
from src.metrics.fairness_metrics import FairnessMetrics
from src.mitigation.bias_mitigation import BiasMitigationPipeline
from src.visualization.visualizer import BiasVisualizer


class AIBiasAnalysisSystem:
    """
    Complete AI Bias Analysis System
    Implements the methodology described in the research paper
    """
    
    def __init__(self, dataset_type: str = 'hiring', bias_strength: float = 0.4):
        """
        Parameters:
        -----------
        dataset_type : str
            Type of dataset ('hiring', 'credit', 'healthcare', 'criminal_justice')
        bias_strength : float
            Strength of bias in generated data (0 to 1)
        """
        self.dataset_type = dataset_type
        self.bias_strength = bias_strength
        
        self.data_generator = BiasedDataGenerator()
        self.fairness_metrics = FairnessMetrics()
        self.visualizer = BiasVisualizer()
        
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.protected_train = None
        self.protected_test = None
        
        self.baseline_model = None
        self.mitigated_model = None
        
        self.baseline_results = None
        self.mitigated_results = None
    
    def step1_data_collection(self, n_samples: int = 2000):
        """Step 1: Generate or load biased dataset"""
        print("="*80)
        print("STEP 1: DATA COLLECTION")
        print("="*80)
        
        if self.dataset_type == 'hiring':
            self.df = self.data_generator.generate_hiring_dataset(
                n_samples=n_samples, bias_strength=self.bias_strength)
            self.target_col = 'hired'
            self.protected_attrs = ['gender', 'race']
            feature_cols = ['age', 'education_score', 'experience_years', 
                          'skills_score', 'interview_score']
        
        elif self.dataset_type == 'credit':
            self.df = self.data_generator.generate_credit_dataset(
                n_samples=n_samples, bias_strength=self.bias_strength)
            self.target_col = 'approved'
            self.protected_attrs = ['gender', 'race']
            feature_cols = ['age', 'income', 'credit_score', 'debt_ratio',
                          'employment_length', 'num_credit_lines']
        
        elif self.dataset_type == 'healthcare':
            self.df = self.data_generator.generate_healthcare_dataset(
                n_samples=n_samples, bias_strength=self.bias_strength)
            self.target_col = 'high_risk'
            self.protected_attrs = ['gender', 'race']
            feature_cols = ['age', 'bmi', 'blood_pressure', 'cholesterol',
                          'glucose', 'smoking']
        
        elif self.dataset_type == 'criminal_justice':
            self.df = self.data_generator.generate_criminal_justice_dataset(
                n_samples=n_samples, bias_strength=self.bias_strength)
            self.target_col = 'predicted_recidivism'
            self.protected_attrs = ['gender', 'race']
            feature_cols = ['age', 'prior_arrests', 'prior_convictions',
                          'time_since_release', 'employment_status']
        
        print(f"\n✓ Generated {self.dataset_type} dataset")
        print(f"  - Samples: {len(self.df)}")
        print(f"  - Bias strength: {self.bias_strength}")
        print(f"  - Features: {len(feature_cols)}")
        print(f"  - Protected attributes: {self.protected_attrs}")
        
        print(f"\n{self.target_col.upper()} RATE BY PROTECTED ATTRIBUTES:")
        for attr in self.protected_attrs:
            print(f"\n  {attr}:")
            rates = self.df.groupby(attr)[self.target_col].mean()
            for group, rate in rates.items():
                print(f"    {group}: {rate:.3f} ({rate*100:.1f}%)")
        
        self.feature_cols = feature_cols
        return self.df
    
    def step2_preprocessing(self):
        """Step 2: Data preprocessing and feature engineering"""
        print("\n" + "="*80)
        print("STEP 2: DATA PREPROCESSING")
        print("="*80)
        
        df_encoded = self.df.copy()
        categorical_cols = df_encoded.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col not in self.protected_attrs + [self.target_col]:
                df_encoded[col] = pd.factorize(df_encoded[col])[0]
        
        X = df_encoded[self.feature_cols].values
        y = df_encoded[self.target_col].values
        primary_protected_attr = self.protected_attrs[0]
        protected = df_encoded[primary_protected_attr].values
        
        X_train, X_test, y_train, y_test, protected_train, protected_test = \
            train_test_split(X, y, protected, test_size=0.3, random_state=42, stratify=y)
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.protected_train = protected_train
        self.protected_test = protected_test
        self.primary_protected_attr = primary_protected_attr
        
        print(f"\n✓ Data preprocessed and split")
        print(f"  - Training samples: {len(X_train)}")
        print(f"  - Test samples: {len(X_test)}")
        print(f"  - Features: {X_train.shape[1]}")
        print(f"  - Primary protected attribute: {primary_protected_attr}")
    
    def step3_baseline_model_training(self):
        """Step 3: Train baseline model without bias mitigation"""
        print("\n" + "="*80)
        print("STEP 3: BASELINE MODEL TRAINING (Without Bias Mitigation)")
        print("="*80)
        
        self.baseline_model = LogisticRegression(max_iter=1000, random_state=42)
        self.baseline_model.fit(self.X_train, self.y_train)
        
        y_pred_train = self.baseline_model.predict(self.X_train)
        y_pred_test = self.baseline_model.predict(self.X_test)
        y_proba_test = self.baseline_model.predict_proba(self.X_test)[:, 1]
        
        train_acc = accuracy_score(self.y_train, y_pred_train)
        test_acc = accuracy_score(self.y_test, y_pred_test)
        
        print(f"\n✓ Baseline model trained")
        print(f"  - Training accuracy: {train_acc:.4f}")
        print(f"  - Test accuracy: {test_acc:.4f}")
        
        self.baseline_pred = y_pred_test
        self.baseline_proba = y_proba_test
        
        return test_acc
    
    def step4_bias_detection(self):
        """Step 4: Detect and measure bias using fairness metrics"""
        print("\n" + "="*80)
        print("STEP 4: BIAS DETECTION & MEASUREMENT")
        print("="*80)
        
        self.baseline_results = self.fairness_metrics.calculate_all_metrics(
            self.y_test,
            self.baseline_pred,
            self.protected_test,
            self.primary_protected_attr
        )
        
        report = self.fairness_metrics.generate_fairness_report(self.baseline_results)
        print(f"\n{report}")
        
        return self.baseline_results
    
    def step5_bias_mitigation(self):
        """Step 5: Apply bias mitigation techniques"""
        print("\n" + "="*80)
        print("STEP 5: BIAS MITIGATION")
        print("="*80)
        
        self.mitigated_model = BiasMitigationPipeline(
            preprocessing_method='reweighting',
            inprocessing_constraint='demographic_parity',
            postprocessing_method='threshold_optimization'
        )
        
        self.mitigated_model.fit(self.X_train, self.y_train, self.protected_train)
        
        val_size = len(self.X_test) // 2
        X_val = self.X_test[:val_size]
        y_val = self.y_test[:val_size]
        protected_val = self.protected_test[:val_size]
        
        self.mitigated_model.optimize_postprocessing(X_val, y_val, protected_val)
        
        X_test_final = self.X_test[val_size:]
        y_test_final = self.y_test[val_size:]
        protected_test_final = self.protected_test[val_size:]
        
        y_pred_mitigated = self.mitigated_model.predict(X_test_final, protected_test_final)
        test_acc_mitigated = accuracy_score(y_test_final, y_pred_mitigated)
        
        print(f"\n✓ Bias mitigation pipeline trained")
        print(f"  - Test accuracy: {test_acc_mitigated:.4f}")
        
        self.X_test_final = X_test_final
        self.y_test_final = y_test_final
        self.protected_test_final = protected_test_final
        self.mitigated_pred = y_pred_mitigated
        
        return test_acc_mitigated
    
    def step6_evaluation(self):
        """Step 6: Evaluate bias mitigation effectiveness"""
        print("\n" + "="*80)
        print("STEP 6: BIAS MITIGATION EVALUATION")
        print("="*80)
        
        baseline_pred_final = self.baseline_model.predict(self.X_test_final)
        
        self.mitigated_results = self.fairness_metrics.calculate_all_metrics(
            self.y_test_final,
            self.mitigated_pred,
            self.protected_test_final,
            self.primary_protected_attr
        )
        
        baseline_results_final = self.fairness_metrics.calculate_all_metrics(
            self.y_test_final,
            baseline_pred_final,
            self.protected_test_final,
            self.primary_protected_attr
        )
        
        print("\nMITIGATED MODEL:")
        print(self.fairness_metrics.generate_fairness_report(self.mitigated_results))
        
        return baseline_results_final, self.mitigated_results
    
    def step7_visualization(self, output_dir: str = './results'):
        """Step 7: Create comprehensive visualizations"""
        print("\n" + "="*80)
        print("STEP 7: GENERATING VISUALIZATIONS")
        print("="*80)
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        baseline_pred_final = self.baseline_model.predict(self.X_test_final)
        baseline_proba_final = self.baseline_model.predict_proba(self.X_test_final)[:, 1]
        
        print("\n1. Creating selection rates comparison...")
        self.visualizer.plot_selection_rates(
            self.mitigated_results,
            self.primary_protected_attr,
            save_path=f'{output_dir}/selection_rates.png'
        )
        
        print("2. Creating confusion matrices...")
        self.visualizer.plot_confusion_matrix_by_group(
            self.y_test_final,
            self.mitigated_pred,
            self.protected_test_final,
            self.primary_protected_attr,
            save_path=f'{output_dir}/confusion_matrices.png'
        )
        
        print(f"\n✓ All visualizations saved to: {output_dir}")
    
    def run_complete_analysis(self, n_samples: int = 2000, output_dir: str = './results'):
        """Run the complete bias analysis pipeline"""
        print("\n" + "="*80)
        print("AI BIAS ANALYSIS SYSTEM - COMPLETE PIPELINE")
        print(f"Dataset: {self.dataset_type.upper()}")
        print(f"Bias Strength: {self.bias_strength}")
        print("="*80)
        
        self.step1_data_collection(n_samples)
        self.step2_preprocessing()
        self.step3_baseline_model_training()
        self.step4_bias_detection()
        self.step5_bias_mitigation()
        baseline_results, mitigated_results = self.step6_evaluation()
        self.step7_visualization(output_dir)
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80)
        
        return {
            'baseline_results': baseline_results,
            'mitigated_results': mitigated_results,
            'output_directory': output_dir
        }


def main():
    """Main function to run bias analysis"""
    print("\n" + "="*80)
    print("AI BIAS ANALYSIS PROJECT")
    print("Implementation of Research Paper: 'Analyzing Bias in AI Models'")
    print("="*80)
    
    system = AIBiasAnalysisSystem(
        dataset_type='hiring',
        bias_strength=0.4
    )
    
    results = system.run_complete_analysis(
        n_samples=2000,
        output_dir='./results_hiring'
    )
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()