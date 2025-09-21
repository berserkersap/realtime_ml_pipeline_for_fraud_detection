"""
Tests for the XGBoost fraud detection model.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
from unittest.mock import Mock

from src.fraud_detection.models.xgboost_model import XGBoostFraudDetector
from src.fraud_detection.utils.data_generator import FraudDataGenerator


class TestXGBoostFraudDetector:
    """Test cases for XGBoostFraudDetector."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = {
            'xgboost': {
                'max_depth': 3,
                'n_estimators': 10,  # Small for testing
                'learning_rate': 0.3
            }
        }
        self.model = XGBoostFraudDetector(self.config)
        
        # Generate test data
        generator = FraudDataGenerator(seed=42)
        self.test_data = generator.generate_dataset(
            n_customers=100,
            n_transactions=500,
            fraud_rate=0.1
        )
    
    def test_model_initialization(self):
        """Test model initialization."""
        assert self.model.config == self.config
        assert not self.model.is_trained
        assert self.model.model is None
    
    def test_prepare_features(self):
        """Test feature preparation."""
        features = self.model.prepare_features(self.test_data, is_training=True)
        
        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(self.test_data)
        assert len(self.model.feature_names) > 0
        
        # Test inference mode
        inference_features = self.model.prepare_features(
            self.test_data.head(10), 
            is_training=False
        )
        assert len(inference_features) == 10
    
    def test_training(self):
        """Test model training."""
        results = self.model.train(self.test_data)
        
        assert self.model.is_trained
        assert 'auc_score' in results
        assert 'feature_importance' in results
        assert results['auc_score'] > 0.5  # Should be better than random
    
    def test_prediction(self):
        """Test model prediction."""
        # Train model first
        self.model.train(self.test_data)
        
        # Test prediction
        test_sample = self.test_data.head(10)
        probabilities = self.model.predict_proba(test_sample)
        predictions = self.model.predict(test_sample)
        
        assert len(probabilities) == 10
        assert len(predictions) == 10
        assert all(0 <= p <= 1 for p in probabilities)
        assert all(p in [0, 1] for p in predictions)
    
    def test_feature_importance(self):
        """Test feature importance extraction."""
        self.model.train(self.test_data)
        
        importance = self.model.get_feature_importance()
        
        assert isinstance(importance, dict)
        assert len(importance) > 0
        assert all(isinstance(v, (int, float)) for v in importance.values())
    
    def test_save_load_model(self):
        """Test model saving and loading."""
        # Train model
        self.model.train(self.test_data)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "test_model.joblib"
            
            # Save model
            self.model.save_model(str(model_path))
            assert model_path.exists()
            
            # Load model
            new_model = XGBoostFraudDetector()
            new_model.load_model(str(model_path))
            
            assert new_model.is_trained
            assert len(new_model.feature_names) == len(self.model.feature_names)
            
            # Test predictions are similar
            test_sample = self.test_data.head(5)
            orig_pred = self.model.predict_proba(test_sample)
            new_pred = new_model.predict_proba(test_sample)
            
            np.testing.assert_array_almost_equal(orig_pred, new_pred, decimal=5)
    
    def test_prediction_without_training(self):
        """Test that prediction fails without training."""
        with pytest.raises(ValueError, match="Model must be trained"):
            self.model.predict_proba(self.test_data.head(1))
    
    def test_explain_prediction(self):
        """Test prediction explanation."""
        self.model.train(self.test_data)
        
        test_sample = self.test_data.head(3)
        explanations = self.model.explain_prediction(test_sample, top_k=5)
        
        assert len(explanations) == 3
        for exp in explanations:
            assert 'prediction_proba' in exp
            assert 'prediction' in exp
            assert 'top_features' in exp
            assert len(exp['top_features']) <= 5