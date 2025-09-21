"""
Kafka consumer for processing streaming transactions and making predictions.
Consumes transactions, makes fraud predictions, and publishes results.
"""

import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
from kafka import KafkaConsumer, KafkaProducer
import pandas as pd
from pathlib import Path

from ..utils.config import load_config, get_kafka_config
from ..models.xgboost_model import XGBoostFraudDetector
from ..models.lstm_model import LSTMFraudDetectorWrapper
from ..models.tab_transformer import TabTransformerWrapper

logger = logging.getLogger(__name__)


class FraudDetectionConsumer:
    """Kafka consumer for real-time fraud detection."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize fraud detection consumer."""
        self.config = config or load_config()
        self.kafka_config = get_kafka_config(self.config)
        
        # Initialize Kafka consumer
        self.consumer = KafkaConsumer(
            self.kafka_config.topics.get('transactions', 'transactions'),
            bootstrap_servers=self.kafka_config.bootstrap_servers,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            key_deserializer=lambda k: k.decode('utf-8') if k else None,
            **self.kafka_config.consumer_config
        )
        
        # Initialize Kafka producer for predictions
        self.producer = KafkaProducer(
            bootstrap_servers=self.kafka_config.bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8') if k else None,
            **self.kafka_config.producer_config
        )
        
        self.predictions_topic = self.kafka_config.topics.get('predictions', 'predictions')
        
        # Initialize models
        self.models = {}
        self.load_models()
        
        # Processing statistics
        self.stats = {
            'total_processed': 0,
            'fraud_detected': 0,
            'start_time': time.time(),
            'processing_times': []
        }
        
        logger.info("FraudDetectionConsumer initialized")
    
    def load_models(self) -> None:
        """Load trained models."""
        model_path = Path("models")
        
        if not model_path.exists():
            logger.warning("Models directory not found. Creating empty models dict.")
            return
        
        # Load XGBoost model
        xgb_path = model_path / "xgboost_model.joblib"
        if xgb_path.exists():
            try:
                self.models["xgboost"] = XGBoostFraudDetector(self.config)
                self.models["xgboost"].load_model(str(xgb_path))
                logger.info("XGBoost model loaded")
            except Exception as e:
                logger.error(f"Failed to load XGBoost model: {e}")
        
        # Load LSTM model
        lstm_path = model_path / "lstm_model.pt"
        if lstm_path.exists():
            try:
                self.models["lstm"] = LSTMFraudDetectorWrapper(self.config)
                self.models["lstm"].load_model(str(lstm_path))
                logger.info("LSTM model loaded")
            except Exception as e:
                logger.error(f"Failed to load LSTM model: {e}")
        
        # Load TabTransformer model
        tab_path = model_path / "tab_transformer_model.pt"
        if tab_path.exists():
            try:
                self.models["tab_transformer"] = TabTransformerWrapper(self.config)
                self.models["tab_transformer"].load_model(str(tab_path))
                logger.info("TabTransformer model loaded")
            except Exception as e:
                logger.error(f"Failed to load TabTransformer model: {e}")
        
        if not self.models:
            logger.warning("No models loaded. Consumer will run without predictions.")
    
    def prepare_transaction_data(self, transaction: Dict[str, Any]) -> pd.DataFrame:
        """Prepare transaction data for model prediction."""
        # Convert single transaction to DataFrame
        df = pd.DataFrame([transaction])
        
        # Ensure required columns exist with defaults
        required_columns = {
            'customer_id': transaction.get('customer_id', 'unknown'),
            'amount': transaction.get('amount', 0.0),
            'merchant_category': transaction.get('merchant_category', 'unknown'),
            'location': transaction.get('location', 'unknown'),
            'day_of_week': transaction.get('day_of_week', 0),
            'hour_of_day': transaction.get('hour_of_day', 0),
            'is_weekend': transaction.get('is_weekend', 0),
            'is_night': transaction.get('is_night', 0),
            'avg_amount': transaction.get('avg_amount', transaction.get('amount', 0.0)),
            'std_amount': transaction.get('std_amount', 1.0),
            'transaction_count': transaction.get('transaction_count', 1),
            'unique_categories': transaction.get('unique_categories', 1),
            'amount_zscore': transaction.get('amount_zscore', 0.0),
            'time_since_last': transaction.get('time_since_last', 24.0),
            'transactions_last_hour': transaction.get('transactions_last_hour', 0)
        }
        
        # Ensure all required columns are present
        for col, default_value in required_columns.items():
            if col not in df.columns:
                df[col] = default_value
        
        # Add transaction_time if not present
        if 'transaction_time' not in df.columns:
            df['transaction_time'] = datetime.now()
        
        return df
    
    def make_predictions(self, transaction_data: pd.DataFrame) -> Dict[str, Any]:
        """Make fraud predictions using available models."""
        predictions = {}
        
        for model_name, model in self.models.items():
            try:
                # Make prediction
                start_time = time.time()
                probability = model.predict_proba(transaction_data)[0]
                prediction = int(probability >= 0.5)
                prediction_time = time.time() - start_time
                
                predictions[model_name] = {
                    'probability': float(probability),
                    'prediction': prediction,
                    'confidence': float(abs(probability - 0.5) * 2),
                    'prediction_time_ms': prediction_time * 1000
                }
                
            except Exception as e:
                logger.error(f"Error making prediction with {model_name}: {e}")
                predictions[model_name] = {
                    'probability': 0.0,
                    'prediction': 0,
                    'confidence': 0.0,
                    'error': str(e),
                    'prediction_time_ms': 0.0
                }
        
        return predictions
    
    def process_transaction(self, message) -> None:
        """Process a single transaction message."""
        start_time = time.time()
        
        try:
            # Extract transaction data
            transaction = message.value
            customer_id = message.key
            
            logger.debug(f"Processing transaction for customer: {customer_id}")
            
            # Prepare data for prediction
            transaction_data = self.prepare_transaction_data(transaction)
            
            # Make predictions
            predictions = {}
            if self.models:
                predictions = self.make_predictions(transaction_data)
            
            # Create prediction result
            result = {
                'transaction': transaction,
                'customer_id': customer_id,
                'predictions': predictions,
                'timestamp': datetime.now().isoformat(),
                'processing_time_ms': (time.time() - start_time) * 1000
            }
            
            # Determine if any model predicted fraud
            fraud_detected = any(
                pred_data.get('prediction', 0) == 1 
                for pred_data in predictions.values()
            )
            
            result['fraud_detected'] = fraud_detected
            
            # Send prediction to output topic
            self.producer.send(
                self.predictions_topic,
                key=customer_id,
                value=result
            )
            
            # Update statistics
            self.stats['total_processed'] += 1
            if fraud_detected:
                self.stats['fraud_detected'] += 1
            
            processing_time = time.time() - start_time
            self.stats['processing_times'].append(processing_time)
            
            # Log progress
            if self.stats['total_processed'] % 100 == 0:
                self.log_statistics()
            
            # Log fraud detection
            if fraud_detected:
                logger.warning(f"FRAUD DETECTED for customer {customer_id}: {predictions}")
            
        except Exception as e:
            logger.error(f"Error processing transaction: {e}")
    
    def log_statistics(self) -> None:
        """Log processing statistics."""
        elapsed_time = time.time() - self.stats['start_time']
        throughput = self.stats['total_processed'] / elapsed_time
        
        avg_processing_time = 0
        if self.stats['processing_times']:
            avg_processing_time = sum(self.stats['processing_times']) / len(self.stats['processing_times'])
        
        fraud_rate = (self.stats['fraud_detected'] / max(1, self.stats['total_processed'])) * 100
        
        logger.info(
            f"Processed: {self.stats['total_processed']}, "
            f"Fraud: {self.stats['fraud_detected']} ({fraud_rate:.2f}%), "
            f"Throughput: {throughput:.2f} msg/sec, "
            f"Avg processing: {avg_processing_time*1000:.2f}ms"
        )
    
    def run(self) -> None:
        """Run the fraud detection consumer."""
        logger.info("Starting fraud detection consumer...")
        logger.info(f"Listening to topic: {self.consumer.subscription()}")
        logger.info(f"Output topic: {self.predictions_topic}")
        logger.info(f"Loaded models: {list(self.models.keys())}")
        
        try:
            # Start consuming messages
            for message in self.consumer:
                self.process_transaction(message)
                
        except KeyboardInterrupt:
            logger.info("Consumer interrupted by user")
        except Exception as e:
            logger.error(f"Consumer error: {e}")
        finally:
            self.stop()
    
    def stop(self) -> None:
        """Stop the consumer and producer."""
        logger.info("Stopping fraud detection consumer...")
        
        # Flush producer
        self.producer.flush()
        
        # Close connections
        self.consumer.close()
        self.producer.close()
        
        # Log final statistics
        self.log_statistics()
        
        logger.info("Fraud detection consumer stopped")


def main():
    """Main function to run the fraud detection consumer."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Kafka Fraud Detection Consumer")
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize and run consumer
    consumer = FraudDetectionConsumer()
    
    try:
        consumer.run()
    except Exception as e:
        logger.error(f"Failed to run consumer: {e}")


if __name__ == "__main__":
    main()