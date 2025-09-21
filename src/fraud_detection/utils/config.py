"""
Configuration utilities for the fraud detection pipeline.
Handles loading and validation of configuration from various sources.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class KafkaConfig:
    """Kafka configuration."""
    bootstrap_servers: str
    topics: Dict[str, str]
    consumer_config: Dict[str, Any]
    producer_config: Dict[str, Any]


@dataclass
class APIConfig:
    """API configuration."""
    host: str
    port: int
    workers: int
    log_level: str


@dataclass
class ModelConfig:
    """Model configuration."""
    xgboost: Dict[str, Any]
    lstm: Dict[str, Any]
    tab_transformer: Dict[str, Any]


@dataclass
class MonitoringConfig:
    """Monitoring configuration."""
    prometheus: Dict[str, Any]
    logging: Dict[str, Any]


@dataclass
class DatabaseConfig:
    """Database configuration."""
    host: str
    port: int
    name: str
    user: str
    password: str


@dataclass
class Config:
    """Main configuration class."""
    kafka: KafkaConfig
    api: APIConfig
    models: ModelConfig
    monitoring: MonitoringConfig
    database: DatabaseConfig


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from YAML file and environment variables."""
    
    # Default config path
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent.parent / "config" / "config.yaml"
    
    config = {}
    
    # Load from YAML file if it exists
    if Path(config_path).exists():
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
    else:
        logger.warning(f"Config file not found at {config_path}, using defaults")
    
    # Override with environment variables
    config = _override_with_env_vars(config)
    
    return config


def _override_with_env_vars(config: Dict[str, Any]) -> Dict[str, Any]:
    """Override configuration with environment variables."""
    
    # Kafka configuration
    if 'kafka' not in config:
        config['kafka'] = {}
    
    config['kafka']['bootstrap_servers'] = [
        os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
    ]
    
    if 'topics' not in config['kafka']:
        config['kafka']['topics'] = {}
    
    config['kafka']['topics']['transactions'] = os.getenv('KAFKA_TOPIC_TRANSACTIONS', 'transactions')
    config['kafka']['topics']['predictions'] = os.getenv('KAFKA_TOPIC_PREDICTIONS', 'predictions')
    
    if 'consumer_config' not in config['kafka']:
        config['kafka']['consumer_config'] = {}
    
    config['kafka']['consumer_config']['group_id'] = os.getenv('KAFKA_GROUP_ID', 'fraud-detection-consumer')
    
    # API configuration
    if 'api' not in config:
        config['api'] = {}
    
    config['api']['host'] = os.getenv('API_HOST', '0.0.0.0')
    config['api']['port'] = int(os.getenv('API_PORT', '8000'))
    config['api']['workers'] = int(os.getenv('API_WORKERS', '1'))
    config['api']['log_level'] = os.getenv('LOG_LEVEL', 'info')
    
    # Database configuration
    if 'database' not in config:
        config['database'] = {}
    
    config['database']['host'] = os.getenv('DB_HOST', 'localhost')
    config['database']['port'] = int(os.getenv('DB_PORT', '5432'))
    config['database']['name'] = os.getenv('DB_NAME', 'fraud_detection')
    config['database']['user'] = os.getenv('DB_USER', 'postgres')
    config['database']['password'] = os.getenv('DB_PASSWORD', 'postgres')
    
    # Monitoring configuration
    if 'monitoring' not in config:
        config['monitoring'] = {}
    
    if 'prometheus' not in config['monitoring']:
        config['monitoring']['prometheus'] = {}
    
    config['monitoring']['prometheus']['port'] = int(os.getenv('PROMETHEUS_PORT', '8080'))
    config['monitoring']['prometheus']['enabled'] = os.getenv('METRICS_ENABLED', 'true').lower() == 'true'
    
    if 'logging' not in config['monitoring']:
        config['monitoring']['logging'] = {}
    
    config['monitoring']['logging']['level'] = os.getenv('LOG_LEVEL', 'INFO')
    
    # Model configuration defaults
    if 'models' not in config:
        config['models'] = {}
    
    if 'xgboost' not in config['models']:
        config['models']['xgboost'] = {
            'path': os.getenv('MODEL_PATH', 'models/') + 'xgboost_model.joblib',
            'features': 30
        }
    
    if 'lstm' not in config['models']:
        config['models']['lstm'] = {
            'path': os.getenv('MODEL_PATH', 'models/') + 'lstm_model.pt',
            'sequence_length': 10,
            'hidden_size': 64
        }
    
    if 'tab_transformer' not in config['models']:
        config['models']['tab_transformer'] = {
            'path': os.getenv('MODEL_PATH', 'models/') + 'tab_transformer_model.pt',
            'embedding_dim': 32,
            'num_heads': 8
        }
    
    return config


def get_kafka_config(config: Dict[str, Any]) -> KafkaConfig:
    """Get Kafka configuration object."""
    kafka_conf = config.get('kafka', {})
    
    return KafkaConfig(
        bootstrap_servers=kafka_conf.get('bootstrap_servers', ['localhost:9092']),
        topics=kafka_conf.get('topics', {}),
        consumer_config=kafka_conf.get('consumer_config', {}),
        producer_config=kafka_conf.get('producer_config', {})
    )


def get_api_config(config: Dict[str, Any]) -> APIConfig:
    """Get API configuration object."""
    api_conf = config.get('api', {})
    
    return APIConfig(
        host=api_conf.get('host', '0.0.0.0'),
        port=api_conf.get('port', 8000),
        workers=api_conf.get('workers', 1),
        log_level=api_conf.get('log_level', 'info')
    )


def get_model_config(config: Dict[str, Any]) -> ModelConfig:
    """Get model configuration object."""
    models_conf = config.get('models', {})
    
    return ModelConfig(
        xgboost=models_conf.get('xgboost', {}),
        lstm=models_conf.get('lstm', {}),
        tab_transformer=models_conf.get('tab_transformer', {})
    )


def get_monitoring_config(config: Dict[str, Any]) -> MonitoringConfig:
    """Get monitoring configuration object."""
    monitoring_conf = config.get('monitoring', {})
    
    return MonitoringConfig(
        prometheus=monitoring_conf.get('prometheus', {}),
        logging=monitoring_conf.get('logging', {})
    )


def get_database_config(config: Dict[str, Any]) -> DatabaseConfig:
    """Get database configuration object."""
    db_conf = config.get('database', {})
    
    return DatabaseConfig(
        host=db_conf.get('host', 'localhost'),
        port=db_conf.get('port', 5432),
        name=db_conf.get('name', 'fraud_detection'),
        user=db_conf.get('user', 'postgres'),
        password=db_conf.get('password', 'postgres')
    )


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration completeness and correctness."""
    required_sections = ['kafka', 'api', 'models', 'monitoring', 'database']
    
    for section in required_sections:
        if section not in config:
            logger.error(f"Missing required config section: {section}")
            return False
    
    # Validate Kafka configuration
    kafka_conf = config['kafka']
    if 'bootstrap_servers' not in kafka_conf:
        logger.error("Missing Kafka bootstrap_servers")
        return False
    
    # Validate API configuration
    api_conf = config['api']
    required_api_keys = ['host', 'port']
    for key in required_api_keys:
        if key not in api_conf:
            logger.error(f"Missing API config key: {key}")
            return False
    
    # Validate model configuration
    models_conf = config['models']
    required_models = ['xgboost', 'lstm', 'tab_transformer']
    for model in required_models:
        if model not in models_conf:
            logger.warning(f"Missing model config: {model}")
    
    logger.info("Configuration validation passed")
    return True