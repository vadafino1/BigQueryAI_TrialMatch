#!/usr/bin/env python3
"""
Configuration Manager for BigQuery 2025 Clinical Trial Matching System
Handles loading, validation, and access to configuration settings
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConfigManager:
    """Manages configuration for the clinical trial matching system"""

    def __init__(self, config_file: Optional[str] = None, use_default: bool = False):
        """
        Initialize configuration manager

        Args:
            config_file: Path to specific config file
            use_default: Force use of default config (for testing/demo)
        """
        self.config: Dict[str, Any] = {}
        self.config_path: Path

        # Determine which config file to use
        if use_default:
            self.config_path = Path(__file__).parent / "default.config.json"
            logger.info("Using DEFAULT configuration (competition winner settings)")
        elif config_file and Path(config_file).exists():
            self.config_path = Path(config_file)
            logger.info(f"Using specified config: {config_file}")
        elif Path("config/user.config.json").exists():
            self.config_path = Path("config/user.config.json")
            logger.info("Using USER configuration")
        else:
            self.config_path = Path(__file__).parent / "default.config.json"
            logger.info("No user config found, using DEFAULT configuration")

        self.load_config()
        self.validate_config()
        self.apply_environment_overrides()

    def load_config(self) -> None:
        """Load configuration from JSON file"""
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
            logger.info(f"Configuration loaded from: {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to load config from {self.config_path}: {e}")
            sys.exit(1)

    def validate_config(self) -> None:
        """Validate configuration for required fields"""
        errors = []

        # Check for placeholder values that need to be updated
        if "YOUR_PROJECT_ID" in str(self.config.get('gcp', {}).get('project_id', '')):
            errors.append("GCP project_id not configured - update config/user.config.json")

        if "YOUR_EMAIL" in str(self.config.get('gcp', {}).get('service_account', '')):
            errors.append("Service account email not configured - update config/user.config.json")

        # Check required top-level keys
        required_keys = ['gcp', 'bigquery', 'mimic', 'api', 'temporal', 'ml']
        for key in required_keys:
            if key not in self.config:
                errors.append(f"Missing required configuration section: {key}")

        # If using user config, show errors
        if errors and "user.config" in str(self.config_path):
            logger.error("\n⚠️  Configuration errors found:")
            for error in errors:
                logger.error(f"  ❌ {error}")
            logger.error("\nPlease update config/user.config.json with your settings")
            sys.exit(1)
        elif errors:
            logger.warning("Using default config - some features may not work without proper configuration")

    def apply_environment_overrides(self) -> None:
        """Override config values with environment variables if set"""
        overrides = {
            'GCP_PROJECT_ID': 'gcp.project_id',
            'GCP_DATASET_ID': 'gcp.dataset_id',
            'GCP_REGION': 'gcp.region',
            'API_PORT': 'api.port',
            'API_KEY': 'api.api_key',
            'VERTEX_AI_LOCATION': 'ml.vertex_ai_location'
        }

        for env_var, config_path in overrides.items():
            if env_value := os.getenv(env_var):
                self.set(config_path, env_value)
                logger.info(f"Override from environment: {env_var} -> {config_path}")

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-notation path

        Args:
            key_path: Dot-separated path to config value (e.g., 'gcp.project_id')
            default: Default value if key doesn't exist

        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self.config

        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default

        return value

    def set(self, key_path: str, value: Any) -> None:
        """
        Set configuration value by dot-notation path

        Args:
            key_path: Dot-separated path to config value
            value: Value to set
        """
        keys = key_path.split('.')
        config_ref = self.config

        for key in keys[:-1]:
            if key not in config_ref:
                config_ref[key] = {}
            config_ref = config_ref[key]

        config_ref[keys[-1]] = value

    def get_bigquery_table(self, table_name: str) -> str:
        """
        Get fully qualified BigQuery table name

        Args:
            table_name: Short table name (e.g., 'patients')

        Returns:
            Fully qualified table name
        """
        project = self.get('gcp.project_id')
        dataset = self.get('gcp.dataset_id')
        table = self.get(f'bigquery.tables.{table_name}', table_name)
        return f"`{project}.{dataset}.{table}`"

    def get_mimic_table(self, table_name: str) -> str:
        """
        Get fully qualified MIMIC-IV table name

        Args:
            table_name: MIMIC table name

        Returns:
            Fully qualified MIMIC table name
        """
        project = self.get('mimic.project')
        dataset = self.get('mimic.dataset')
        table = self.get(f'mimic.tables.{table_name}', table_name)
        return f"`{project}.{dataset}.{table}`"

    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary"""
        return self.config.copy()

    def save(self, output_path: Optional[str] = None) -> None:
        """
        Save current configuration to file

        Args:
            output_path: Path to save config (defaults to original path)
        """
        save_path = Path(output_path) if output_path else self.config_path
        with open(save_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        logger.info(f"Configuration saved to: {save_path}")

    def print_summary(self) -> None:
        """Print configuration summary"""
        print("\n" + "="*60)
        print("CONFIGURATION SUMMARY")
        print("="*60)
        print(f"Config File: {self.config_path}")
        print(f"Project ID: {self.get('gcp.project_id')}")
        print(f"Dataset: {self.get('gcp.dataset_id')}")
        print(f"Region: {self.get('gcp.region')}")
        print(f"API Port: {self.get('api.port')}")
        print(f"Embedding Model: {self.get('ml.embedding_model')}")
        print(f"Current Date (for temporal): {self.get('temporal.current_date')}")
        print("="*60 + "\n")


# Convenience function for quick access
def load_config(config_file: Optional[str] = None, use_default: bool = False) -> ConfigManager:
    """Load configuration with sensible defaults"""
    return ConfigManager(config_file, use_default)


if __name__ == "__main__":
    # Test configuration loading
    config = ConfigManager()
    config.print_summary()

    # Example usage
    print(f"\nExample - Get project ID: {config.get('gcp.project_id')}")
    print(f"Example - Get patients table: {config.get_bigquery_table('patients')}")
    print(f"Example - Get MIMIC patients: {config.get_mimic_table('patients')}")