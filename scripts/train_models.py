"""
Model training script for fraud detection models.
Trains XGBoost, LSTM, and TabTransformer models on the generated dataset.
"""

import logging
import pandas as pd
from pathlib import Path
import time
import json
from typing import Dict, Any

from src.fraud_detection.models.xgboost_model import XGBoostFraudDetector
from src.fraud_detection.models.lstm_model import LSTMFraudDetectorWrapper
from src.fraud_detection.models.tab_transformer import TabTransformerWrapper
from src.fraud_detection.utils.data_generator import FraudDataGenerator
from src.fraud_detection.utils.config import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_training_data():
    """Generate training and test datasets."""
    logger.info("Generating training data...")
    
    generator = FraudDataGenerator(seed=42)
    
    # Create output directories
    data_dir = Path("data/processed")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate training dataset
    train_data = generator.generate_dataset(
        n_customers=5000,
        n_transactions=50000,
        fraud_rate=0.002,
        output_path=data_dir / "train_data.csv"
    )
    
    # Generate test dataset
    test_data = generator.generate_dataset(
        n_customers=1000,
        n_transactions=10000,
        fraud_rate=0.002,
        output_path=data_dir / "test_data.csv"
    )
    
    logger.info(f"Training data: {len(train_data)} transactions")
    logger.info(f"Test data: {len(test_data)} transactions")
    logger.info(f"Training fraud rate: {train_data['is_fraud'].mean():.4f}")
    logger.info(f"Test fraud rate: {test_data['is_fraud'].mean():.4f}")
    
    return train_data, test_data


def train_xgboost_model(train_data: pd.DataFrame, test_data: pd.DataFrame) -> Dict[str, Any]:
    """Train XGBoost model."""
    logger.info("Training XGBoost model...")
    
    config = load_config()
    model = XGBoostFraudDetector(config)
    
    start_time = time.time()
    results = model.train(train_data)
    training_time = time.time() - start_time
    
    # Test on test set
    start_time = time.time()
    test_probabilities = model.predict_proba(test_data)
    inference_time = time.time() - start_time
    
    # Calculate metrics
    from sklearn.metrics import roc_auc_score, classification_report
    test_auc = roc_auc_score(test_data['is_fraud'], test_probabilities)
    test_predictions = (test_probabilities >= 0.5).astype(int)
    
    # Save model
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(models_dir / "xgboost_model.joblib")
    
    return {
        'model_type': 'xgboost',
        'training_time': training_time,
        'inference_time': inference_time,
        'test_auc': test_auc,
        'validation_auc': results['auc_score'],
        'feature_importance': results['feature_importance']
    }


def train_lstm_model(train_data: pd.DataFrame, test_data: pd.DataFrame) -> Dict[str, Any]:
    """Train LSTM model."""
    logger.info("Training LSTM model...")
    
    config = load_config()
    model = LSTMFraudDetectorWrapper(config)
    
    start_time = time.time()
    results = model.train(train_data)
    training_time = time.time() - start_time
    
    # Test on test set
    start_time = time.time()
    test_probabilities = model.predict_proba(test_data)
    inference_time = time.time() - start_time
    
    # Calculate test AUC if predictions are available
    test_auc = 0.0
    if len(test_probabilities) > 0:
        # For LSTM, we need to create sequences, so we might have fewer predictions
        # than test samples. Let's handle this case.
        from sklearn.metrics import roc_auc_score
        if len(test_probabilities) == len(test_data):
            test_auc = roc_auc_score(test_data['is_fraud'], test_probabilities)
        else:
            logger.warning(f"LSTM predictions ({len(test_probabilities)}) != test samples ({len(test_data)})")
    
    # Save model
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(models_dir / "lstm_model.pt")
    
    return {
        'model_type': 'lstm',
        'training_time': training_time,
        'inference_time': inference_time,
        'test_auc': test_auc,
        'validation_auc': results.get('best_val_auc', 0.0),
        'final_val_auc': results.get('final_val_auc', 0.0)
    }


def train_tab_transformer_model(train_data: pd.DataFrame, test_data: pd.DataFrame) -> Dict[str, Any]:
    """Train TabTransformer model."""
    logger.info("Training TabTransformer model...")
    
    config = load_config()
    model = TabTransformerWrapper(config)
    
    start_time = time.time()
    results = model.train(train_data)
    training_time = time.time() - start_time
    
    # Test on test set
    start_time = time.time()
    test_probabilities = model.predict_proba(test_data)
    inference_time = time.time() - start_time
    
    # Calculate test AUC
    from sklearn.metrics import roc_auc_score
    test_auc = roc_auc_score(test_data['is_fraud'], test_probabilities)
    
    # Save model
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(models_dir / "tab_transformer_model.pt")
    
    return {
        'model_type': 'tab_transformer',
        'training_time': training_time,
        'inference_time': inference_time,
        'test_auc': test_auc,
        'validation_auc': results.get('best_val_auc', 0.0),
        'final_val_auc': results.get('final_val_auc', 0.0)
    }


def compare_batch_vs_streaming_performance():
    """Compare batch vs streaming inference performance."""
    logger.info("Comparing batch vs streaming performance...")
    
    # Load test data
    test_data = pd.read_csv("data/processed/test_data.csv")
    sample_data = test_data.head(1000)  # Use smaller sample for timing
    
    # Load models
    config = load_config()
    models = {}
    
    # Load XGBoost
    try:
        models['xgboost'] = XGBoostFraudDetector(config)
        models['xgboost'].load_model("models/xgboost_model.joblib")
    except Exception as e:
        logger.error(f"Failed to load XGBoost: {e}")
    
    performance_results = {}
    
    for model_name, model in models.items():
        logger.info(f"Testing {model_name} performance...")
        
        # Batch inference
        start_time = time.time()
        batch_predictions = model.predict_proba(sample_data)
        batch_time = time.time() - start_time
        
        # Streaming inference (single transaction at a time)
        start_time = time.time()
        streaming_predictions = []
        for idx in range(len(sample_data)):
            single_pred = model.predict_proba(sample_data.iloc[[idx]])
            streaming_predictions.append(single_pred[0])
        streaming_time = time.time() - start_time
        
        # Calculate metrics
        latency_reduction = (batch_time - streaming_time) / batch_time * 100
        batch_latency = batch_time / len(sample_data) * 1000  # ms per prediction
        streaming_latency = streaming_time / len(sample_data) * 1000  # ms per prediction
        
        performance_results[model_name] = {
            'batch_time': batch_time,
            'streaming_time': streaming_time,
            'batch_latency_ms': batch_latency,
            'streaming_latency_ms': streaming_latency,
            'latency_reduction_percent': latency_reduction,
            'throughput_batch_per_sec': len(sample_data) / batch_time,
            'throughput_streaming_per_sec': len(sample_data) / streaming_time
        }
        
        logger.info(f"{model_name} - Batch: {batch_latency:.2f}ms/pred, Streaming: {streaming_latency:.2f}ms/pred")
    
    return performance_results


def main():
    """Main training function."""
    logger.info("Starting model training pipeline...")
    
    # Generate training data if it doesn't exist
    train_data_path = Path("data/processed/train_data.csv")
    test_data_path = Path("data/processed/test_data.csv")
    
    if not train_data_path.exists() or not test_data_path.exists():
        train_data, test_data = generate_training_data()
    else:
        logger.info("Loading existing training data...")
        train_data = pd.read_csv(train_data_path)
        test_data = pd.read_csv(test_data_path)
    
    # Train all models
    results = []
    
    try:
        xgb_results = train_xgboost_model(train_data, test_data)
        results.append(xgb_results)
        logger.info(f"XGBoost - Val AUC: {xgb_results['validation_auc']:.4f}, Test AUC: {xgb_results['test_auc']:.4f}")
    except Exception as e:
        logger.error(f"XGBoost training failed: {e}")
    
    try:
        lstm_results = train_lstm_model(train_data, test_data)
        results.append(lstm_results)
        logger.info(f"LSTM - Val AUC: {lstm_results['validation_auc']:.4f}, Test AUC: {lstm_results['test_auc']:.4f}")
    except Exception as e:
        logger.error(f"LSTM training failed: {e}")
    
    try:
        tab_results = train_tab_transformer_model(train_data, test_data)
        results.append(tab_results)
        logger.info(f"TabTransformer - Val AUC: {tab_results['validation_auc']:.4f}, Test AUC: {tab_results['test_auc']:.4f}")
    except Exception as e:
        logger.error(f"TabTransformer training failed: {e}")
    
    # Performance comparison
    try:
        performance_results = compare_batch_vs_streaming_performance()
        results.append({'performance_comparison': performance_results})
    except Exception as e:
        logger.error(f"Performance comparison failed: {e}")
    
    # Save all results
    results_path = Path("models/training_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Training complete! Results saved to {results_path}")
    
    # Print summary
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    
    for result in results:
        if 'model_type' in result:
            print(f"{result['model_type'].upper()}:")
            print(f"  Validation AUC: {result.get('validation_auc', 'N/A'):.4f}")
            print(f"  Test AUC: {result.get('test_auc', 'N/A'):.4f}")
            print(f"  Training Time: {result.get('training_time', 'N/A'):.2f}s")
            print(f"  Inference Time: {result.get('inference_time', 'N/A'):.4f}s")
            print()


if __name__ == "__main__":
    main()