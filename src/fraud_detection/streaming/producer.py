"""
Kafka producer for streaming transaction data.
Simulates real-time transaction generation and publishing to Kafka.
"""

import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from kafka import KafkaProducer
import pandas as pd
import numpy as np
from pathlib import Path

from ..utils.config import load_config, get_kafka_config
from ..utils.data_generator import FraudDataGenerator

logger = logging.getLogger(__name__)


class TransactionProducer:
    """Kafka producer for streaming transaction data."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Kafka producer."""
        self.config = config or load_config()
        self.kafka_config = get_kafka_config(self.config)
        
        # Initialize Kafka producer
        self.producer = KafkaProducer(
            bootstrap_servers=self.kafka_config.bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8') if k else None,
            **self.kafka_config.producer_config
        )
        
        self.topic = self.kafka_config.topics.get('transactions', 'transactions')
        self.is_running = False
        
        # Data generator for creating realistic transactions
        self.data_generator = FraudDataGenerator()
        
        logger.info(f"TransactionProducer initialized. Topic: {self.topic}")
    
    def send_transaction(self, transaction: Dict[str, Any]) -> None:
        """Send a single transaction to Kafka."""
        try:
            # Use customer_id as key for partitioning
            key = transaction.get('customer_id', None)
            
            # Add timestamp
            transaction['timestamp'] = datetime.now().isoformat()
            
            # Send to Kafka
            future = self.producer.send(self.topic, key=key, value=transaction)
            
            # Optional: wait for confirmation (can slow down throughput)
            # result = future.get(timeout=10)
            
            logger.debug(f"Sent transaction for customer {key}")
            
        except Exception as e:
            logger.error(f"Failed to send transaction: {e}")
    
    def stream_from_file(
        self, 
        file_path: str, 
        rate_per_second: float = 10.0,
        duration_seconds: Optional[int] = None
    ) -> None:
        """Stream transactions from a CSV file at specified rate."""
        
        logger.info(f"Starting to stream from file: {file_path}")
        logger.info(f"Rate: {rate_per_second} transactions/second")
        
        try:
            # Load data
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(df)} transactions from file")
            
            # Calculate delay between sends
            delay = 1.0 / rate_per_second
            
            self.is_running = True
            start_time = time.time()
            sent_count = 0
            
            for idx, row in df.iterrows():
                if not self.is_running:
                    break
                
                # Check duration limit
                if duration_seconds and (time.time() - start_time) > duration_seconds:
                    logger.info("Duration limit reached, stopping stream")
                    break
                
                # Convert row to dictionary
                transaction = row.to_dict()
                
                # Convert numpy types to Python types for JSON serialization
                for key, value in transaction.items():
                    if isinstance(value, (np.integer, np.floating)):
                        transaction[key] = value.item()
                    elif pd.isna(value):
                        transaction[key] = None
                
                # Send transaction
                self.send_transaction(transaction)
                sent_count += 1
                
                # Log progress
                if sent_count % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = sent_count / elapsed
                    logger.info(f"Sent {sent_count} transactions. Rate: {rate:.2f}/sec")
                
                # Wait before next send
                time.sleep(delay)
            
            total_time = time.time() - start_time
            avg_rate = sent_count / total_time
            logger.info(f"Streaming complete. Sent {sent_count} transactions in {total_time:.2f}s (avg rate: {avg_rate:.2f}/sec)")
            
        except Exception as e:
            logger.error(f"Error streaming from file: {e}")
        finally:
            self.is_running = False
    
    def stream_synthetic_data(
        self,
        rate_per_second: float = 10.0,
        duration_seconds: Optional[int] = None,
        fraud_rate: float = 0.002,
        n_customers: int = 1000
    ) -> None:
        """Stream synthetic transaction data in real-time."""
        
        logger.info("Starting synthetic transaction streaming")
        logger.info(f"Rate: {rate_per_second} transactions/second")
        logger.info(f"Fraud rate: {fraud_rate:.1%}")
        
        try:
            # Generate customer profiles once
            customers = self.data_generator.generate_customer_profiles(n_customers)
            logger.info(f"Generated {len(customers)} customer profiles")
            
            delay = 1.0 / rate_per_second
            self.is_running = True
            start_time = time.time()
            sent_count = 0
            
            while self.is_running:
                # Check duration limit
                if duration_seconds and (time.time() - start_time) > duration_seconds:
                    logger.info("Duration limit reached, stopping stream")
                    break
                
                # Select random customer
                customer = customers.sample(1).iloc[0]
                
                # Generate transaction
                if np.random.random() < fraud_rate:
                    transaction = self.data_generator._generate_fraud_transaction(customer)
                    transaction['is_fraud'] = 1
                else:
                    transaction = self.data_generator._generate_legitimate_transaction(customer)
                    transaction['is_fraud'] = 0
                
                # Convert datetime to string
                transaction['transaction_time'] = transaction['transaction_time'].isoformat()
                
                # Send transaction
                self.send_transaction(transaction)
                sent_count += 1
                
                # Log progress
                if sent_count % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = sent_count / elapsed
                    logger.info(f"Sent {sent_count} transactions. Rate: {rate:.2f}/sec")
                
                # Wait before next send
                time.sleep(delay)
                
        except Exception as e:
            logger.error(f"Error in synthetic streaming: {e}")
        finally:
            self.is_running = False
    
    def stop(self) -> None:
        """Stop the producer."""
        logger.info("Stopping transaction producer...")
        self.is_running = False
        
        # Flush remaining messages
        self.producer.flush()
        
        # Close producer
        self.producer.close()
        
        logger.info("Transaction producer stopped")
    
    def send_batch(self, transactions: list) -> None:
        """Send a batch of transactions."""
        logger.info(f"Sending batch of {len(transactions)} transactions")
        
        for transaction in transactions:
            self.send_transaction(transaction)
        
        # Ensure all messages are sent
        self.producer.flush()
        
        logger.info("Batch sent successfully")


def main():
    """Main function to run the transaction producer."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Kafka Transaction Producer")
    parser.add_argument('--mode', choices=['file', 'synthetic'], default='synthetic',
                       help='Streaming mode: file or synthetic')
    parser.add_argument('--file', type=str, help='Path to CSV file (for file mode)')
    parser.add_argument('--rate', type=float, default=10.0,
                       help='Transactions per second')
    parser.add_argument('--duration', type=int, help='Duration in seconds')
    parser.add_argument('--fraud-rate', type=float, default=0.002,
                       help='Fraud rate for synthetic data')
    parser.add_argument('--customers', type=int, default=1000,
                       help='Number of customers for synthetic data')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize producer
    producer = TransactionProducer()
    
    try:
        if args.mode == 'file':
            if not args.file:
                print("Error: --file is required for file mode")
                return
            
            if not Path(args.file).exists():
                print(f"Error: File {args.file} not found")
                return
            
            producer.stream_from_file(
                args.file,
                rate_per_second=args.rate,
                duration_seconds=args.duration
            )
        
        elif args.mode == 'synthetic':
            producer.stream_synthetic_data(
                rate_per_second=args.rate,
                duration_seconds=args.duration,
                fraud_rate=args.fraud_rate,
                n_customers=args.customers
            )
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        producer.stop()


if __name__ == "__main__":
    main()