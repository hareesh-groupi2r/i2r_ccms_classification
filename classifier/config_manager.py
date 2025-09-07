"""
Configuration Manager Module
Handles loading and managing configuration from YAML files and environment variables
"""

import os
import yaml
from typing import Dict, List, Any, Optional
from pathlib import Path
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Manages configuration for the classification system.
    """
    
    def __init__(self, config_path: str = 'config.yaml', env_path: str = '.env'):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to YAML configuration file
            env_path: Path to .env file
        """
        self.config_path = Path(config_path)
        self.env_path = Path(env_path)
        self.config = {}
        
        # Load environment variables
        if self.env_path.exists():
            load_dotenv(self.env_path)
            logger.info(f"Loaded environment variables from {self.env_path}")
        else:
            logger.warning(f"Environment file {self.env_path} not found")
        
        # Load configuration
        self.load_config()
    
    def load_config(self):
        """Load configuration from YAML file."""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    self.config = yaml.safe_load(f)
                logger.info(f"Loaded configuration from {self.config_path}")
            else:
                logger.error(f"Configuration file {self.config_path} not found")
                raise FileNotFoundError(f"Configuration file {self.config_path} not found")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    def save_config(self):
        """Save current configuration back to YAML file."""
        try:
            with open(self.config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
            logger.info(f"Saved configuration to {self.config_path}")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            raise
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'approaches.pure_llm.model')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """
        Set configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'approaches.pure_llm.enabled')
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
        logger.debug(f"Set configuration {key} = {value}")
    
    def get_enabled_approaches(self) -> List[str]:
        """
        Get list of enabled classification approaches.
        
        Returns:
            List of enabled approach names
        """
        enabled = []
        approaches = self.get('approaches', {})
        
        for approach, settings in approaches.items():
            if settings.get('enabled', False):
                enabled.append(approach)
        
        logger.info(f"Enabled approaches: {enabled}")
        return enabled
    
    def get_approach_config(self, approach: str) -> Dict:
        """
        Get configuration for a specific approach.
        
        Args:
            approach: Name of the approach
            
        Returns:
            Configuration dictionary for the approach
        """
        config = self.get(f'approaches.{approach}', {})
        
        # Add API keys from environment if needed
        if approach == 'pure_llm':
            if 'gpt' in config.get('model', '').lower():
                config['api_key'] = os.getenv('OPENAI_API_KEY')
            elif 'claude' in config.get('model', '').lower():
                config['api_key'] = os.getenv('ANTHROPIC_API_KEY')
        
        elif approach == 'hybrid_rag':
            if 'gpt' in config.get('llm_model', '').lower():
                config['api_key'] = os.getenv('OPENAI_API_KEY')
        
        elif approach == 'google_docai':
            config['project_id'] = os.getenv('GOOGLE_CLOUD_PROJECT_ID', config.get('project_id'))
            config['gemini_api_key'] = os.getenv('GEMINI_API_KEY')
        
        return config
    
    def update_approach_status(self, approach: str, enabled: bool):
        """
        Enable or disable a specific approach.
        
        Args:
            approach: Name of the approach
            enabled: Whether to enable or disable
        """
        self.set(f'approaches.{approach}.enabled', enabled)
        logger.info(f"Updated {approach} status to {'enabled' if enabled else 'disabled'}")
    
    def get_data_paths(self) -> Dict[str, str]:
        """
        Get all data paths from configuration.
        
        Returns:
            Dictionary of data paths
        """
        return self.get('data', {})
    
    def get_validation_config(self) -> Dict:
        """
        Get validation configuration.
        
        Returns:
            Validation configuration dictionary
        """
        return self.get('validation', {})
    
    def get_data_sufficiency_config(self) -> Dict:
        """
        Get data sufficiency configuration.
        
        Returns:
            Data sufficiency configuration dictionary
        """
        return self.get('data_sufficiency', {})
    
    def get_api_config(self) -> Dict:
        """
        Get API configuration.
        
        Returns:
            API configuration dictionary
        """
        config = self.get('api', {})
        config['secret_key'] = os.getenv('SECRET_KEY', config.get('secret_key', 'dev-secret-key'))
        return config
    
    def get_cache_config(self) -> Dict:
        """
        Get cache configuration.
        
        Returns:
            Cache configuration dictionary
        """
        config = self.get('cache', {})
        
        # Override with environment variables if available
        config['redis_host'] = os.getenv('REDIS_HOST', config.get('redis_host', 'localhost'))
        config['redis_port'] = int(os.getenv('REDIS_PORT', config.get('redis_port', 6379)))
        config['redis_password'] = os.getenv('REDIS_PASSWORD', config.get('redis_password', ''))
        
        return config
    
    def get_ensemble_config(self) -> Dict:
        """
        Get ensemble configuration.
        
        Returns:
            Ensemble configuration dictionary
        """
        return self.get('ensemble', {})
    
    def get_monitoring_config(self) -> Dict:
        """
        Get monitoring configuration.
        
        Returns:
            Monitoring configuration dictionary
        """
        config = self.get('monitoring', {})
        config['log_level'] = os.getenv('LOG_LEVEL', config.get('log_level', 'INFO'))
        return config
    
    def validate_config(self) -> bool:
        """
        Validate that required configuration is present.
        
        Returns:
            True if configuration is valid
        """
        required_keys = [
            'approaches',
            'data.training_data',
            'validation.enable_strict_validation'
        ]
        
        for key in required_keys:
            if self.get(key) is None:
                logger.error(f"Missing required configuration: {key}")
                return False
        
        # Check that at least one approach is enabled
        if not self.get_enabled_approaches():
            logger.error("No classification approaches are enabled")
            return False
        
        # Check for required API keys for enabled approaches
        for approach in self.get_enabled_approaches():
            if approach == 'pure_llm':
                model = self.get(f'approaches.{approach}.model', '')
                if 'gpt' in model.lower() and not os.getenv('OPENAI_API_KEY'):
                    logger.error("OpenAI API key required for GPT models")
                    return False
                elif 'claude' in model.lower() and not os.getenv('ANTHROPIC_API_KEY'):
                    logger.error("Anthropic API key required for Claude models")
                    return False
        
        return True
    
    def get_all_config(self) -> Dict:
        """
        Get entire configuration dictionary.
        
        Returns:
            Complete configuration dictionary
        """
        return self.config.copy()
    
    def __repr__(self):
        enabled = self.get_enabled_approaches()
        return f"ConfigManager(approaches={enabled})"