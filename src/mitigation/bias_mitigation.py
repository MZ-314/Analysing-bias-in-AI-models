"""
Bias Mitigation Module
Implements various bias mitigation techniques:
- Pre-processing: Data reweighting, resampling
- In-processing: Fairness-aware algorithms, adversarial debiasing
- Post-processing: Threshold adjustment, calibration
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from typing import Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class BiasPreprocessor:
    """Pre-processing techniques to reduce bias in training data"""
    
    def __init__(self):
        self.weights = None
        self.scaler = StandardScaler()
    
    def reweighting(self,
                   X: np.ndarray,
                   y: np.ndarray,
                   protected_attr: np.ndarray) -> np.ndarray:
        """
        Reweight samples to achieve demographic parity
        
        Parameters:
        -----------
        X : array-like
            Feature matrix
        y : array-like
            Labels
        protected_attr : array-like
            Protected attribute values
        
        Returns:
        --------
        Sample weights array
        """
        weights = np.ones(len(y))
        
        # Calculate expected probability for each group
        unique_groups = np.unique(protected_attr)
        overall_positive_rate = np.mean(y)
        
        for group in unique_groups:
            group_mask = protected_attr == group
            group_positive_rate = np.mean(y[group_mask])
            
            # Calculate reweighting factor
            if group_positive_rate > 0:
                # For positive samples in this group
                pos_mask = group_mask & (y == 1)
                weights[pos_mask] = overall_positive_rate / group_positive_rate
                
                # For negative samples in this group
                neg_mask = group_mask & (y == 0)
                weights[neg_mask] = (1 - overall_positive_rate) / (1 - group_positive_rate)
        
        # Normalize weights
        weights = weights / weights.mean()
        self.weights = weights
        
        return weights
    
    def resampling(self,
                   X: pd.DataFrame,
                   y: pd.Series,
                   protected_attr: pd.Series,
                   strategy: str = 'oversample') -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Resample data to balance representation across protected groups
        
        Parameters:
        -----------
        X : DataFrame
            Feature matrix
        y : Series
            Labels
        protected_attr : Series
            Protected attribute values
        strategy : str
            'oversample' or 'undersample'
        
        Returns:
        --------
        Resampled X, y, protected_attr
        """
        # Combine data
        data = X.copy()
        data['target'] = y.values
        data['protected'] = protected_attr.values
        
        # Calculate target size for each group
        unique_groups = protected_attr.unique()
        group_sizes = [len(data[data['protected'] == g]) for g in unique_groups]
        
        if strategy == 'oversample':
            target_size = max(group_sizes)
        else:  # undersample
            target_size = min(group_sizes)
        
        # Resample each group
        resampled_dfs = []
        for group in unique_groups:
            group_data = data[data['protected'] == group]
            
            if len(group_data) < target_size:
                # Oversample
                resampled_group = resample(group_data, 
                                          n_samples=target_size,
                                          replace=True,
                                          random_state=42)
            elif len(group_data) > target_size:
                # Undersample
                resampled_group = resample(group_data,
                                          n_samples=target_size,
                                          replace=False,
                                          random_state=42)
            else:
                resampled_group = group_data
            
            resampled_dfs.append(resampled_group)
        
        # Combine resampled data
        resampled_data = pd.concat(resampled_dfs, ignore_index=True)
        
        # Shuffle
        resampled_data = resampled_data.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Split back
        X_resampled = resampled_data.drop(['target', 'protected'], axis=1)
        y_resampled = resampled_data['target']
        protected_resampled = resampled_data['protected']
        
        return X_resampled, y_resampled, protected_resampled
    
    def feature_suppression(self,
                           X: pd.DataFrame,
                           protected_columns: list) -> pd.DataFrame:
        """
        Remove protected attributes from features
        
        Parameters:
        -----------
        X : DataFrame
            Feature matrix
        protected_columns : list
            List of column names to remove
        
        Returns:
        --------
        DataFrame with protected attributes removed
        """
        return X.drop(columns=protected_columns, errors='ignore')


class FairnessConstrainedModel:
    """In-processing: Train models with fairness constraints"""
    
    def __init__(self, fairness_constraint: str = 'demographic_parity'):
        """
        Parameters:
        -----------
        fairness_constraint : str
            Type of fairness constraint ('demographic_parity', 'equalized_odds')
        """
        self.fairness_constraint = fairness_constraint
        self.model = LogisticRegression(max_iter=1000)
        self.lambda_fairness = 1.0
    
    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            protected_attr: np.ndarray):
        """
        Fit model with fairness constraints
        
        Parameters:
        -----------
        X : array-like
            Feature matrix
        y : array-like
            Labels
        protected_attr : array-like
            Protected attribute values
        """
        # Calculate sample weights to promote fairness
        weights = self._calculate_fairness_weights(y, protected_attr)
        
        # Train model with weights
        self.model.fit(X, y, sample_weight=weights)
        
        return self
    
    def _calculate_fairness_weights(self,
                                   y: np.ndarray,
                                   protected_attr: np.ndarray) -> np.ndarray:
        """Calculate sample weights based on fairness constraint"""
        weights = np.ones(len(y))
        
        unique_groups = np.unique(protected_attr)
        
        if self.fairness_constraint == 'demographic_parity':
            # Weight to achieve equal positive prediction rates
            overall_rate = np.mean(y)
            
            for group in unique_groups:
                mask = protected_attr == group
                group_rate = np.mean(y[mask])
                
                if group_rate > 0:
                    # Adjust weights to balance positive rates
                    pos_mask = mask & (y == 1)
                    weights[pos_mask] *= overall_rate / group_rate
        
        elif self.fairness_constraint == 'equalized_odds':
            # Weight to achieve equal TPR and FPR
            for group in unique_groups:
                mask = protected_attr == group
                group_pos_rate = np.mean(y[mask])
                overall_pos_rate = np.mean(y)
                
                if group_pos_rate > 0:
                    weights[mask] *= overall_pos_rate / group_pos_rate
        
        # Normalize
        weights = weights / weights.mean()
        
        return weights
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities"""
        return self.model.predict_proba(X)


class BiasPostprocessor:
    """Post-processing techniques to adjust model outputs"""
    
    def __init__(self):
        self.group_thresholds = {}
    
    def threshold_optimization(self,
                              y_true: np.ndarray,
                              y_proba: np.ndarray,
                              protected_attr: np.ndarray,
                              constraint: str = 'demographic_parity') -> Dict[str, float]:
        """
        Optimize decision thresholds for each group to satisfy fairness constraints
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_proba : array-like
            Predicted probabilities
        protected_attr : array-like
            Protected attribute values
        constraint : str
            Fairness constraint to satisfy
        
        Returns:
        --------
        Dictionary mapping groups to optimal thresholds
        """
        unique_groups = np.unique(protected_attr)
        
        if constraint == 'demographic_parity':
            # Find thresholds that equalize selection rates
            overall_selection_rate = np.mean(y_true)
            
            for group in unique_groups:
                mask = protected_attr == group
                group_proba = y_proba[mask]
                
                # Find threshold that achieves target selection rate
                sorted_proba = np.sort(group_proba)[::-1]
                target_count = int(len(group_proba) * overall_selection_rate)
                
                if target_count < len(sorted_proba):
                    threshold = sorted_proba[target_count]
                else:
                    threshold = 0.0
                
                self.group_thresholds[group] = threshold
        
        elif constraint == 'equalized_odds':
            # Find thresholds that equalize TPR across groups
            for group in unique_groups:
                mask = protected_attr == group
                group_y_true = y_true[mask]
                group_proba = y_proba[mask]
                
                # Calculate optimal threshold using ROC analysis
                thresholds = np.linspace(0, 1, 100)
                best_threshold = 0.5
                best_tpr_diff = float('inf')
                
                target_tpr = np.mean(y_true)  # Overall TPR target
                
                for thresh in thresholds:
                    y_pred = (group_proba >= thresh).astype(int)
                    tp = np.sum((group_y_true == 1) & (y_pred == 1))
                    fn = np.sum((group_y_true == 1) & (y_pred == 0))
                    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                    
                    tpr_diff = abs(tpr - target_tpr)
                    if tpr_diff < best_tpr_diff:
                        best_tpr_diff = tpr_diff
                        best_threshold = thresh
                
                self.group_thresholds[group] = best_threshold
        
        return self.group_thresholds
    
    def apply_thresholds(self,
                        y_proba: np.ndarray,
                        protected_attr: np.ndarray) -> np.ndarray:
        """
        Apply group-specific thresholds to predictions
        
        Parameters:
        -----------
        y_proba : array-like
            Predicted probabilities
        protected_attr : array-like
            Protected attribute values
        
        Returns:
        --------
        Adjusted binary predictions
        """
        y_pred = np.zeros(len(y_proba), dtype=int)
        
        for group, threshold in self.group_thresholds.items():
            mask = protected_attr == group
            y_pred[mask] = (y_proba[mask] >= threshold).astype(int)
        
        return y_pred
    
    def calibration_adjustment(self,
                              y_proba: np.ndarray,
                              protected_attr: np.ndarray,
                              calibration_method: str = 'platt') -> np.ndarray:
        """
        Adjust probability calibration for each group
        
        Parameters:
        -----------
        y_proba : array-like
            Predicted probabilities
        protected_attr : array-like
            Protected attribute values
        calibration_method : str
            Calibration method ('platt', 'isotonic')
        
        Returns:
        --------
        Calibrated probabilities
        """
        # Simplified calibration adjustment
        calibrated_proba = y_proba.copy()
        
        unique_groups = np.unique(protected_attr)
        overall_mean = np.mean(y_proba)
        
        for group in unique_groups:
            mask = protected_attr == group
            group_mean = np.mean(y_proba[mask])
            
            # Adjust to match overall mean
            if group_mean > 0:
                adjustment = overall_mean / group_mean
                calibrated_proba[mask] = np.clip(y_proba[mask] * adjustment, 0, 1)
        
        return calibrated_proba


class BiasMitigationPipeline:
    """Complete bias mitigation pipeline"""
    
    def __init__(self,
                 preprocessing_method: str = 'reweighting',
                 inprocessing_constraint: str = 'demographic_parity',
                 postprocessing_method: str = 'threshold_optimization'):
        """
        Parameters:
        -----------
        preprocessing_method : str
            'reweighting', 'resampling', or 'none'
        inprocessing_constraint : str
            'demographic_parity', 'equalized_odds', or 'none'
        postprocessing_method : str
            'threshold_optimization', 'calibration', or 'none'
        """
        self.preprocessing_method = preprocessing_method
        self.inprocessing_constraint = inprocessing_constraint
        self.postprocessing_method = postprocessing_method
        
        self.preprocessor = BiasPreprocessor()
        self.model = FairnessConstrainedModel(fairness_constraint=inprocessing_constraint)
        self.postprocessor = BiasPostprocessor()
        
        self.sample_weights = None
    
    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            protected_attr: np.ndarray):
        """
        Fit the complete bias mitigation pipeline
        
        Parameters:
        -----------
        X : array-like
            Feature matrix
        y : array-like
            Labels
        protected_attr : array-like
            Protected attribute values
        """
        # Pre-processing
        if self.preprocessing_method == 'reweighting':
            self.sample_weights = self.preprocessor.reweighting(X, y, protected_attr)
            self.model.model.fit(X, y, sample_weight=self.sample_weights)
        elif self.preprocessing_method == 'resampling':
            # Convert to DataFrame for resampling
            X_df = pd.DataFrame(X)
            y_series = pd.Series(y)
            protected_series = pd.Series(protected_attr)
            
            X_resampled, y_resampled, protected_resampled = \
                self.preprocessor.resampling(X_df, y_series, protected_series)
            
            self.model.fit(X_resampled.values, y_resampled.values, protected_resampled.values)
        else:
            # In-processing only
            self.model.fit(X, y, protected_attr)
        
        return self
    
    def predict(self,
                X: np.ndarray,
                protected_attr: np.ndarray) -> np.ndarray:
        """
        Make predictions with bias mitigation
        
        Parameters:
        -----------
        X : array-like
            Feature matrix
        protected_attr : array-like
            Protected attribute values
        
        Returns:
        --------
        Predictions
        """
        # Get probability predictions
        y_proba = self.model.predict_proba(X)[:, 1]
        
        # Post-processing
        if self.postprocessing_method == 'threshold_optimization':
            # Use pre-computed thresholds or default
            if self.postprocessor.group_thresholds:
                y_pred = self.postprocessor.apply_thresholds(y_proba, protected_attr)
            else:
                y_pred = (y_proba >= 0.5).astype(int)
        elif self.postprocessing_method == 'calibration':
            y_proba_calibrated = self.postprocessor.calibration_adjustment(
                y_proba, protected_attr)
            y_pred = (y_proba_calibrated >= 0.5).astype(int)
        else:
            y_pred = (y_proba >= 0.5).astype(int)
        
        return y_pred
    
    def optimize_postprocessing(self,
                               X_val: np.ndarray,
                               y_val: np.ndarray,
                               protected_attr_val: np.ndarray):
        """
        Optimize post-processing on validation set
        
        Parameters:
        -----------
        X_val : array-like
            Validation feature matrix
        y_val : array-like
            Validation labels
        protected_attr_val : array-like
            Validation protected attributes
        """
        y_proba = self.model.predict_proba(X_val)[:, 1]
        
        if self.postprocessing_method == 'threshold_optimization':
            self.postprocessor.threshold_optimization(
                y_val, y_proba, protected_attr_val,
                constraint=self.inprocessing_constraint
            )


if __name__ == "__main__":
    print("Testing Bias Mitigation Techniques\n")
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 5
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.choice([0, 1], n_samples)
    protected_attr = np.random.choice(['A', 'B'], n_samples)
    
    # Introduce bias
    bias_mask = protected_attr == 'B'
    y[bias_mask] = np.random.choice([0, 1], sum(bias_mask), p=[0.7, 0.3])
    
    # Test pipeline
    pipeline = BiasMitigationPipeline(
        preprocessing_method='reweighting',
        inprocessing_constraint='demographic_parity',
        postprocessing_method='threshold_optimization'
    )
    
    # Split data
    split = int(0.8 * n_samples)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    protected_train, protected_val = protected_attr[:split], protected_attr[split:]
    
    # Fit pipeline
    print("Training bias mitigation pipeline...")
    pipeline.fit(X_train, y_train, protected_train)
    
    # Optimize post-processing
    pipeline.optimize_postprocessing(X_val, y_val, protected_val)
    
    # Make predictions
    y_pred = pipeline.predict(X_val, protected_val)
    
    print(f"Predictions made: {len(y_pred)}")
    print(f"Positive rate for group A: {np.mean(y_pred[protected_val == 'A']):.3f}")
    print(f"Positive rate for group B: {np.mean(y_pred[protected_val == 'B']):.3f}")
    print("\nBias mitigation pipeline test complete!")