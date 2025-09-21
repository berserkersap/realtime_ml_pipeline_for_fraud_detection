"""
Tests for the configuration utilities.
"""

import pytest
import tempfile
import yaml
from pathlib import Path
import os

from src.fraud_detection.utils.config import (
    load_config, 
    get_kafka_config, 
    get_api_config,
    validate_config
)


class TestConfig:
    """Test cases for configuration utilities."""
    
    def test_load_config_from_file(self):
        """Test loading configuration from YAML file."""
        config_data = {
            'kafka': {
                'bootstrap_servers': ['localhost:9092'],
                'topics': {'transactions': 'test_transactions'}
            },
            'api': {
                'host': '0.0.0.0',
                'port': 8000
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            config = load_config(config_path)
            assert 'kafka' in config
            assert 'api' in config
            assert config['kafka']['bootstrap_servers'] == ['localhost:9092']
        finally:
            os.unlink(config_path)
    
    def test_load_config_with_env_override(self):
        """Test that environment variables override config file."""
        # Set environment variables
        os.environ['KAFKA_BOOTSTRAP_SERVERS'] = 'test-server:9092'
        os.environ['API_PORT'] = '9000'
        
        try:
            config = load_config()
            
            assert config['kafka']['bootstrap_servers'] == ['test-server:9092']
            assert config['api']['port'] == 9000
        finally:
            # Clean up
            os.environ.pop('KAFKA_BOOTSTRAP_SERVERS', None)
            os.environ.pop('API_PORT', None)
    
    def test_load_config_missing_file(self):
        """Test loading config when file doesn't exist."""
        config = load_config('/nonexistent/path/config.yaml')
        
        # Should still return default config from env vars
        assert 'kafka' in config
        assert 'api' in config
        assert 'models' in config
    
    def test_get_kafka_config(self):
        """Test Kafka configuration extraction."""
        config = {
            'kafka': {
                'bootstrap_servers': ['localhost:9092'],
                'topics': {'transactions': 'tx', 'predictions': 'pred'},
                'consumer_config': {'group_id': 'test-group'},
                'producer_config': {'acks': 'all'}
            }
        }
        
        kafka_config = get_kafka_config(config)
        
        assert kafka_config.bootstrap_servers == ['localhost:9092']
        assert kafka_config.topics == {'transactions': 'tx', 'predictions': 'pred'}
        assert kafka_config.consumer_config == {'group_id': 'test-group'}
        assert kafka_config.producer_config == {'acks': 'all'}
    
    def test_get_api_config(self):
        """Test API configuration extraction."""
        config = {
            'api': {
                'host': '127.0.0.1',
                'port': 9000,
                'workers': 4,
                'log_level': 'debug'
            }
        }
        
        api_config = get_api_config(config)
        
        assert api_config.host == '127.0.0.1'
        assert api_config.port == 9000
        assert api_config.workers == 4
        assert api_config.log_level == 'debug'
    
    def test_validate_config_valid(self):
        """Test configuration validation with valid config."""
        config = {
            'kafka': {'bootstrap_servers': ['localhost:9092']},
            'api': {'host': '0.0.0.0', 'port': 8000},
            'models': {'xgboost': {}},
            'monitoring': {'prometheus': {}},
            'database': {'host': 'localhost'}
        }
        
        assert validate_config(config) is True
    
    def test_validate_config_missing_section(self):
        """Test configuration validation with missing section."""
        config = {
            'kafka': {'bootstrap_servers': ['localhost:9092']},
            'api': {'host': '0.0.0.0', 'port': 8000}
            # Missing other required sections
        }
        
        assert validate_config(config) is False
    
    def test_validate_config_missing_required_key(self):
        """Test configuration validation with missing required key."""
        config = {
            'kafka': {},  # Missing bootstrap_servers
            'api': {'host': '0.0.0.0'},  # Missing port
            'models': {},
            'monitoring': {},
            'database': {}
        }
        
        assert validate_config(config) is False
    
    def test_default_values(self):
        """Test that default values are set correctly."""
        config = load_config()
        
        # Check default values are present
        assert 'kafka' in config
        assert 'api' in config
        assert 'models' in config
        
        # Check specific defaults
        assert config['api']['host'] == '0.0.0.0'
        assert config['api']['port'] == 8000
        assert isinstance(config['kafka']['bootstrap_servers'], list)