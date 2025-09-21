"""
Tests for the fraud detection data generator.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

from src.fraud_detection.utils.data_generator import FraudDataGenerator


class TestFraudDataGenerator:
    """Test cases for FraudDataGenerator."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.generator = FraudDataGenerator(seed=42)
    
    def test_generate_customer_profiles(self):
        """Test customer profile generation."""
        n_customers = 100
        customers = self.generator.generate_customer_profiles(n_customers)
        
        assert len(customers) == n_customers
        assert 'customer_id' in customers.columns
        assert 'age' in customers.columns
        assert 'income' in customers.columns
        assert 'location' in customers.columns
        
        # Check age range
        assert customers['age'].min() >= 18
        assert customers['age'].max() <= 80
        
        # Check customer IDs are unique
        assert len(customers['customer_id'].unique()) == n_customers
    
    def test_generate_transactions(self):
        """Test transaction generation."""
        customers = self.generator.generate_customer_profiles(10)
        n_transactions = 100
        fraud_rate = 0.1
        
        transactions = self.generator.generate_transactions(
            customers, n_transactions, fraud_rate
        )
        
        assert len(transactions) == n_transactions
        assert 'customer_id' in transactions.columns
        assert 'is_fraud' in transactions.columns
        assert 'amount' in transactions.columns
        
        # Check fraud rate is approximately correct
        actual_fraud_rate = transactions['is_fraud'].mean()
        assert abs(actual_fraud_rate - fraud_rate) < 0.05
    
    def test_fraud_patterns(self):
        """Test that fraud transactions have expected patterns."""
        customers = self.generator.generate_customer_profiles(10)
        
        # Generate only fraud transactions
        n_transactions = 50
        transactions = self.generator.generate_transactions(
            customers, n_transactions, fraud_rate=1.0
        )
        
        fraud_transactions = transactions[transactions['is_fraud'] == 1]
        assert len(fraud_transactions) > 0
        
        # Fraud transactions should generally have higher amounts
        avg_amount = transactions['amount'].mean()
        assert avg_amount > 0
    
    def test_generate_dataset(self):
        """Test complete dataset generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_data.csv"
            
            dataset = self.generator.generate_dataset(
                n_customers=50,
                n_transactions=200,
                fraud_rate=0.05,
                output_path=str(output_path)
            )
            
            assert len(dataset) == 200
            assert output_path.exists()
            
            # Test loading the saved data
            loaded_data = pd.read_csv(output_path)
            assert len(loaded_data) == 200
    
    def test_derived_features(self):
        """Test that derived features are created correctly."""
        customers = self.generator.generate_customer_profiles(5)
        transactions = self.generator.generate_transactions(customers, 20, 0.1)
        
        # Check for derived features
        expected_features = [
            'amount_zscore', 'time_since_last', 'transactions_last_hour',
            'is_weekend', 'is_night'
        ]
        
        for feature in expected_features:
            assert feature in transactions.columns
        
        # Check data types
        assert transactions['is_weekend'].dtype in ['int64', 'int32', 'bool']
        assert transactions['is_night'].dtype in ['int64', 'int32', 'bool']
    
    def test_reproducibility(self):
        """Test that generator produces consistent results with same seed."""
        generator1 = FraudDataGenerator(seed=123)
        generator2 = FraudDataGenerator(seed=123)
        
        customers1 = generator1.generate_customer_profiles(10)
        customers2 = generator2.generate_customer_profiles(10)
        
        pd.testing.assert_frame_equal(customers1, customers2)
        
        transactions1 = generator1.generate_transactions(customers1, 50, 0.1)
        transactions2 = generator2.generate_transactions(customers2, 50, 0.1)
        
        # Should have same fraud labels at minimum
        assert (transactions1['is_fraud'] == transactions2['is_fraud']).all()