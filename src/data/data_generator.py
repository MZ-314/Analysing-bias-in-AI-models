"""
Data Generation Module for AI Bias Analysis
Generates synthetic datasets with controllable bias for testing
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class BiasedDataGenerator:
    """Generate synthetic datasets with controllable bias parameters"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)
    
    def generate_hiring_dataset(self, 
                                n_samples: int = 1000,
                                bias_strength: float = 0.3) -> pd.DataFrame:
        """
        Generate a biased hiring dataset
        
        Parameters:
        -----------
        n_samples : int
            Number of samples to generate
        bias_strength : float
            Strength of bias (0 to 1, higher means more bias)
        
        Returns:
        --------
        pd.DataFrame with features and hiring decision
        """
        # Generate protected attributes
        gender = np.random.choice(['Male', 'Female'], n_samples, p=[0.6, 0.4])
        race = np.random.choice(['White', 'Black', 'Asian', 'Hispanic'], 
                                n_samples, p=[0.5, 0.2, 0.2, 0.1])
        age = np.random.normal(35, 10, n_samples).clip(22, 65)
        
        # Generate features
        education_score = np.random.uniform(0, 100, n_samples)
        experience_years = np.random.exponential(5, n_samples).clip(0, 30)
        skills_score = np.random.uniform(0, 100, n_samples)
        interview_score = np.random.uniform(0, 100, n_samples)
        
        # Calculate base hiring probability
        base_score = (education_score * 0.3 + 
                     experience_years * 2 + 
                     skills_score * 0.4 + 
                     interview_score * 0.3)
        
        # Introduce bias based on protected attributes
        bias_factor = np.ones(n_samples)
        
        # Gender bias (favor males)
        bias_factor[gender == 'Male'] *= (1 + bias_strength * 0.5)
        
        # Race bias (favor White candidates)
        bias_factor[race == 'White'] *= (1 + bias_strength * 0.4)
        bias_factor[race == 'Black'] *= (1 - bias_strength * 0.3)
        
        # Age bias (favor younger candidates)
        age_normalized = (age - 22) / (65 - 22)
        bias_factor *= (1 - bias_strength * age_normalized * 0.3)
        
        # Apply bias to scores
        biased_score = base_score * bias_factor
        
        # Generate hiring decision
        threshold = np.percentile(biased_score, 70)
        hired = (biased_score >= threshold).astype(int)
        
        # Create DataFrame
        df = pd.DataFrame({
            'gender': gender,
            'race': race,
            'age': age,
            'education_score': education_score,
            'experience_years': experience_years,
            'skills_score': skills_score,
            'interview_score': interview_score,
            'hired': hired
        })
        
        return df
    
    def generate_credit_dataset(self, 
                               n_samples: int = 1000,
                               bias_strength: float = 0.3) -> pd.DataFrame:
        """
        Generate a biased credit approval dataset
        
        Parameters:
        -----------
        n_samples : int
            Number of samples to generate
        bias_strength : float
            Strength of bias (0 to 1)
        
        Returns:
        --------
        pd.DataFrame with features and credit approval decision
        """
        # Protected attributes
        gender = np.random.choice(['Male', 'Female'], n_samples, p=[0.55, 0.45])
        race = np.random.choice(['White', 'Black', 'Asian', 'Hispanic'], 
                                n_samples, p=[0.5, 0.25, 0.15, 0.1])
        age = np.random.normal(40, 15, n_samples).clip(18, 80)
        
        # Financial features
        income = np.random.lognormal(10.5, 0.8, n_samples)
        credit_score = np.random.normal(650, 100, n_samples).clip(300, 850)
        debt_ratio = np.random.beta(2, 5, n_samples)
        employment_length = np.random.exponential(5, n_samples).clip(0, 40)
        num_credit_lines = np.random.poisson(3, n_samples).clip(0, 15)
        
        # Calculate base approval probability
        base_score = (
            (credit_score - 300) / 550 * 40 +
            np.log1p(income) * 5 +
            (1 - debt_ratio) * 20 +
            np.minimum(employment_length / 10, 1) * 20 +
            np.minimum(num_credit_lines / 5, 1) * 15
        )
        
        # Introduce bias
        bias_factor = np.ones(n_samples)
        
        # Gender bias
        bias_factor[gender == 'Male'] *= (1 + bias_strength * 0.4)
        
        # Race bias
        bias_factor[race == 'White'] *= (1 + bias_strength * 0.35)
        bias_factor[race == 'Black'] *= (1 - bias_strength * 0.25)
        
        # Apply bias
        biased_score = base_score * bias_factor
        
        # Generate approval decision
        threshold = np.percentile(biased_score, 65)
        approved = (biased_score >= threshold).astype(int)
        
        df = pd.DataFrame({
            'gender': gender,
            'race': race,
            'age': age,
            'income': income,
            'credit_score': credit_score,
            'debt_ratio': debt_ratio,
            'employment_length': employment_length,
            'num_credit_lines': num_credit_lines,
            'approved': approved
        })
        
        return df
    
    def generate_healthcare_dataset(self,
                                   n_samples: int = 1000,
                                   bias_strength: float = 0.3) -> pd.DataFrame:
        """
        Generate a biased healthcare risk assessment dataset
        
        Parameters:
        -----------
        n_samples : int
            Number of samples to generate
        bias_strength : float
            Strength of bias (0 to 1)
        
        Returns:
        --------
        pd.DataFrame with features and high-risk classification
        """
        # Protected attributes
        gender = np.random.choice(['Male', 'Female'], n_samples, p=[0.48, 0.52])
        race = np.random.choice(['White', 'Black', 'Asian', 'Hispanic'], 
                                n_samples, p=[0.55, 0.25, 0.12, 0.08])
        age = np.random.normal(55, 18, n_samples).clip(18, 95)
        
        # Health features
        bmi = np.random.normal(27, 6, n_samples).clip(15, 50)
        blood_pressure = np.random.normal(130, 20, n_samples).clip(90, 200)
        cholesterol = np.random.normal(200, 40, n_samples).clip(120, 350)
        glucose = np.random.normal(100, 30, n_samples).clip(60, 300)
        smoking = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
        
        # Historical spending (biased proxy)
        base_spending = (
            age * 50 +
            (bmi - 18.5) * 100 +
            (blood_pressure - 120) * 20 +
            (cholesterol - 150) * 10 +
            glucose * 15 +
            smoking * 5000
        )
        
        # Introduce bias through spending (mimics real healthcare bias)
        bias_factor = np.ones(n_samples)
        
        # Race bias in spending (White patients have higher spending)
        bias_factor[race == 'White'] *= (1 + bias_strength * 0.5)
        bias_factor[race == 'Black'] *= (1 - bias_strength * 0.3)
        
        healthcare_spending = base_spending * bias_factor + np.random.normal(0, 1000, n_samples)
        healthcare_spending = healthcare_spending.clip(0)
        
        # Classify as high risk based on biased spending
        threshold = np.percentile(healthcare_spending, 75)
        high_risk = (healthcare_spending >= threshold).astype(int)
        
        df = pd.DataFrame({
            'gender': gender,
            'race': race,
            'age': age,
            'bmi': bmi,
            'blood_pressure': blood_pressure,
            'cholesterol': cholesterol,
            'glucose': glucose,
            'smoking': smoking,
            'healthcare_spending': healthcare_spending,
            'high_risk': high_risk
        })
        
        return df
    
    def generate_criminal_justice_dataset(self,
                                         n_samples: int = 1000,
                                         bias_strength: float = 0.3) -> pd.DataFrame:
        """
        Generate a biased recidivism prediction dataset
        
        Parameters:
        -----------
        n_samples : int
            Number of samples to generate
        bias_strength : float
            Strength of bias (0 to 1)
        
        Returns:
        --------
        pd.DataFrame with features and recidivism prediction
        """
        # Protected attributes
        gender = np.random.choice(['Male', 'Female'], n_samples, p=[0.85, 0.15])
        race = np.random.choice(['White', 'Black', 'Hispanic', 'Other'], 
                                n_samples, p=[0.40, 0.35, 0.20, 0.05])
        age = np.random.normal(32, 12, n_samples).clip(18, 70)
        
        # Criminal history features
        prior_arrests = np.random.poisson(2, n_samples).clip(0, 20)
        prior_convictions = (prior_arrests * np.random.uniform(0.3, 0.7, n_samples)).astype(int)
        time_since_release = np.random.exponential(2, n_samples).clip(0, 15)
        charge_severity = np.random.choice(['Low', 'Medium', 'High'], 
                                          n_samples, p=[0.4, 0.4, 0.2])
        
        # Social factors
        employment_status = np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
        education_level = np.random.choice(['High School', 'Some College', 'Bachelor', 'Graduate'],
                                          n_samples, p=[0.5, 0.3, 0.15, 0.05])
        
        # Calculate base recidivism risk
        severity_score = {'Low': 1, 'Medium': 2, 'High': 3}
        base_score = (
            prior_arrests * 3 +
            prior_convictions * 5 +
            (10 - time_since_release) * 2 +
            np.array([severity_score[s] for s in charge_severity]) * 10 +
            (1 - employment_status) * 15 +
            (1 - age / 70) * 10
        )
        
        # Introduce bias
        bias_factor = np.ones(n_samples)
        
        # Race bias (Black defendants scored as higher risk)
        bias_factor[race == 'Black'] *= (1 + bias_strength * 0.6)
        bias_factor[race == 'White'] *= (1 - bias_strength * 0.2)
        
        # Age bias (younger scored as higher risk)
        age_normalized = (age - 18) / (70 - 18)
        bias_factor *= (1 - bias_strength * age_normalized * 0.3)
        
        biased_score = base_score * bias_factor
        
        # Generate recidivism prediction
        threshold = np.percentile(biased_score, 60)
        predicted_recidivism = (biased_score >= threshold).astype(int)
        
        df = pd.DataFrame({
            'gender': gender,
            'race': race,
            'age': age,
            'prior_arrests': prior_arrests,
            'prior_convictions': prior_convictions,
            'time_since_release': time_since_release,
            'charge_severity': charge_severity,
            'employment_status': employment_status,
            'education_level': education_level,
            'predicted_recidivism': predicted_recidivism
        })
        
        return df


if __name__ == "__main__":
    # Test the data generator
    generator = BiasedDataGenerator()
    
    print("Generating sample datasets...")
    
    # Generate hiring dataset
    hiring_df = generator.generate_hiring_dataset(n_samples=1000, bias_strength=0.4)
    print("\n=== Hiring Dataset ===")
    print(f"Shape: {hiring_df.shape}")
    print("\nHiring rate by gender:")
    print(hiring_df.groupby('gender')['hired'].mean())
    print("\nHiring rate by race:")
    print(hiring_df.groupby('race')['hired'].mean())
    
    # Generate credit dataset
    credit_df = generator.generate_credit_dataset(n_samples=1000, bias_strength=0.3)
    print("\n=== Credit Dataset ===")
    print(f"Shape: {credit_df.shape}")
    print("\nApproval rate by gender:")
    print(credit_df.groupby('gender')['approved'].mean())
    
    # Generate healthcare dataset
    health_df = generator.generate_healthcare_dataset(n_samples=1000, bias_strength=0.35)
    print("\n=== Healthcare Dataset ===")
    print(f"Shape: {health_df.shape}")
    print("\nHigh-risk rate by race:")
    print(health_df.groupby('race')['high_risk'].mean())
    
    print("\nData generation complete!")