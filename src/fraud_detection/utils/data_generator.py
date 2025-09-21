"""
Synthetic data generator for credit card fraud detection.
Creates realistic transaction data with fraud patterns.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, Dict, List
import json
import random
from pathlib import Path


class FraudDataGenerator:
    """Generate synthetic credit card transaction data with fraud patterns."""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        random.seed(seed)
        
        # Define merchant categories
        self.merchant_categories = [
            'grocery', 'gas', 'restaurant', 'retail', 'online', 
            'pharmacy', 'entertainment', 'travel', 'utility', 'other'
        ]
        
        # Define transaction patterns
        self.fraud_patterns = {
            'amount_multiplier': (5.0, 20.0),  # Fraud transactions tend to be larger
            'time_anomaly': True,  # Unusual transaction times
            'location_anomaly': True,  # Transactions in unusual locations
            'frequency_anomaly': True,  # Multiple transactions in short time
        }
    
    def generate_customer_profiles(self, n_customers: int) -> pd.DataFrame:
        """Generate customer profiles with spending patterns."""
        customers = []
        
        for i in range(n_customers):
            # Customer demographics
            age = np.random.normal(45, 15)
            age = max(18, min(80, int(age)))
            
            # Income affects spending patterns
            income = np.random.lognormal(10.5, 0.7)  # Mean ~$50k
            
            # Location (simplified to regions)
            location = np.random.choice([
                'Northeast', 'Southeast', 'Midwest', 'Southwest', 'West'
            ])
            
            # Spending patterns
            avg_transaction = income / 12 / 30 * np.random.uniform(0.02, 0.05)
            preferred_categories = np.random.choice(
                self.merchant_categories, 
                size=np.random.randint(3, 6), 
                replace=False
            )
            
            customers.append({
                'customer_id': f'C{i:06d}',
                'age': age,
                'income': income,
                'location': location,
                'avg_transaction': avg_transaction,
                'preferred_categories': preferred_categories.tolist()
            })
        
        return pd.DataFrame(customers)
    
    def generate_transactions(
        self, 
        customers: pd.DataFrame, 
        n_transactions: int, 
        fraud_rate: float = 0.002
    ) -> pd.DataFrame:
        """Generate transaction data with fraud labels."""
        transactions = []
        
        # Calculate number of fraud transactions
        n_fraud = int(n_transactions * fraud_rate)
        n_legitimate = n_transactions - n_fraud
        
        # Generate legitimate transactions
        for _ in range(n_legitimate):
            customer = customers.sample(1).iloc[0]
            transaction = self._generate_legitimate_transaction(customer)
            transaction['is_fraud'] = 0
            transactions.append(transaction)
        
        # Generate fraudulent transactions
        for _ in range(n_fraud):
            customer = customers.sample(1).iloc[0]
            transaction = self._generate_fraud_transaction(customer)
            transaction['is_fraud'] = 1
            transactions.append(transaction)
        
        # Shuffle transactions
        df = pd.DataFrame(transactions)
        df = df.sample(frac=1).reset_index(drop=True)
        
        # Add derived features
        df = self._add_derived_features(df)
        
        return df
    
    def _generate_legitimate_transaction(self, customer: Dict) -> Dict:
        """Generate a legitimate transaction for a customer."""
        # Time: Normal business hours with some variation
        base_time = datetime.now() - timedelta(days=np.random.randint(0, 30))
        hour = np.random.normal(14, 4)  # Peak at 2 PM
        hour = max(0, min(23, int(hour)))
        transaction_time = base_time.replace(
            hour=hour, 
            minute=np.random.randint(0, 60),
            second=np.random.randint(0, 60)
        )
        
        # Amount: Based on customer profile
        amount = np.random.lognormal(
            np.log(customer['avg_transaction']), 
            0.8
        )
        amount = max(1.0, amount)
        
        # Merchant category: Prefer customer's preferred categories
        if random.random() < 0.7:
            category = np.random.choice(customer['preferred_categories'])
        else:
            category = np.random.choice(self.merchant_categories)
        
        # Location: Usually same as customer location
        if random.random() < 0.9:
            location = customer['location']
        else:
            location = np.random.choice([
                'Northeast', 'Southeast', 'Midwest', 'Southwest', 'West'
            ])
        
        return {
            'customer_id': customer['customer_id'],
            'transaction_time': transaction_time,
            'amount': round(amount, 2),
            'merchant_category': category,
            'location': location,
            'day_of_week': transaction_time.weekday(),
            'hour_of_day': transaction_time.hour,
        }
    
    def _generate_fraud_transaction(self, customer: Dict) -> Dict:
        """Generate a fraudulent transaction with anomalous patterns."""
        # Start with legitimate transaction
        transaction = self._generate_legitimate_transaction(customer)
        
        # Apply fraud patterns
        # 1. Higher amounts
        if random.random() < 0.8:
            multiplier = np.random.uniform(*self.fraud_patterns['amount_multiplier'])
            transaction['amount'] *= multiplier
            transaction['amount'] = round(transaction['amount'], 2)
        
        # 2. Unusual times (very early or very late)
        if random.random() < 0.6:
            unusual_hour = np.random.choice([
                np.random.randint(0, 6),  # Very early
                np.random.randint(22, 24)  # Very late
            ])
            transaction['hour_of_day'] = unusual_hour
            transaction['transaction_time'] = transaction['transaction_time'].replace(
                hour=unusual_hour
            )
        
        # 3. Unusual location
        if random.random() < 0.5:
            # Different location than customer's usual location
            other_locations = [
                'Northeast', 'Southeast', 'Midwest', 'Southwest', 'West'
            ]
            other_locations.remove(customer['location'])
            transaction['location'] = np.random.choice(other_locations)
        
        # 4. Unusual merchant category
        if random.random() < 0.4:
            unusual_categories = [
                cat for cat in self.merchant_categories 
                if cat not in customer['preferred_categories']
            ]
            if unusual_categories:
                transaction['merchant_category'] = np.random.choice(unusual_categories)
        
        return transaction
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add engineered features to the transaction data."""
        df = df.copy()
        
        # Sort by customer and time
        df = df.sort_values(['customer_id', 'transaction_time'])
        
        # Time-based features
        df['transaction_time'] = pd.to_datetime(df['transaction_time'])
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_night'] = ((df['hour_of_day'] < 6) | (df['hour_of_day'] >= 22)).astype(int)
        
        # Customer-based aggregations
        customer_stats = df.groupby('customer_id').agg({
            'amount': ['mean', 'std', 'count'],
            'merchant_category': lambda x: len(x.unique())
        }).reset_index()
        
        customer_stats.columns = ['customer_id', 'avg_amount', 'std_amount', 
                                'transaction_count', 'unique_categories']
        
        df = df.merge(customer_stats, on='customer_id', how='left')
        
        # Amount anomaly score (z-score)
        df['amount_zscore'] = (df['amount'] - df['avg_amount']) / (df['std_amount'] + 1e-6)
        
        # Time since last transaction
        df['time_since_last'] = df.groupby('customer_id')['transaction_time'].diff()
        df['time_since_last'] = df['time_since_last'].dt.total_seconds() / 3600  # hours
        df['time_since_last'] = df['time_since_last'].fillna(24)
        
        # Transaction velocity (transactions in last hour)
        df['transactions_last_hour'] = 0
        for idx, row in df.iterrows():
            customer_transactions = df[
                (df['customer_id'] == row['customer_id']) &
                (df['transaction_time'] >= row['transaction_time'] - timedelta(hours=1)) &
                (df['transaction_time'] < row['transaction_time'])
            ]
            df.at[idx, 'transactions_last_hour'] = len(customer_transactions)
        
        # One-hot encode categorical variables
        category_dummies = pd.get_dummies(df['merchant_category'], prefix='merchant')
        location_dummies = pd.get_dummies(df['location'], prefix='location')
        
        df = pd.concat([df, category_dummies, location_dummies], axis=1)
        
        return df
    
    def generate_dataset(
        self, 
        n_customers: int = 10000,
        n_transactions: int = 100000,
        fraud_rate: float = 0.002,
        output_path: str = None
    ) -> pd.DataFrame:
        """Generate complete dataset and optionally save to file."""
        print(f"Generating {n_customers} customer profiles...")
        customers = self.generate_customer_profiles(n_customers)
        
        print(f"Generating {n_transactions} transactions ({fraud_rate:.1%} fraud rate)...")
        transactions = self.generate_transactions(customers, n_transactions, fraud_rate)
        
        if output_path:
            transactions.to_csv(output_path, index=False)
            print(f"Dataset saved to {output_path}")
            
            # Save summary statistics
            summary = {
                'total_transactions': len(transactions),
                'fraud_transactions': int(transactions['is_fraud'].sum()),
                'fraud_rate': float(transactions['is_fraud'].mean()),
                'avg_amount': float(transactions['amount'].mean()),
                'date_range': {
                    'start': transactions['transaction_time'].min().isoformat(),
                    'end': transactions['transaction_time'].max().isoformat()
                }
            }
            
            summary_path = Path(output_path).with_suffix('.json')
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
        
        return transactions


if __name__ == "__main__":
    # Generate dataset
    generator = FraudDataGenerator(seed=42)
    
    # Create output directory
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate training dataset
    train_data = generator.generate_dataset(
        n_customers=10000,
        n_transactions=100000,
        fraud_rate=0.002,
        output_path=output_dir / "train_data.csv"
    )
    
    # Generate test dataset  
    test_data = generator.generate_dataset(
        n_customers=2000,
        n_transactions=20000,
        fraud_rate=0.002,
        output_path=output_dir / "test_data.csv"
    )
    
    print("\nDataset generation complete!")
    print(f"Training data: {len(train_data)} transactions")
    print(f"Test data: {len(test_data)} transactions")
    print(f"Training fraud rate: {train_data['is_fraud'].mean():.4f}")
    print(f"Test fraud rate: {test_data['is_fraud'].mean():.4f}")