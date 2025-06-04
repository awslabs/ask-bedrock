"""
Unit tests for the configuration initialization in ask_bedrock.main.
"""

import unittest
from unittest.mock import patch, MagicMock
import json

from ask_bedrock.main import init_config


class TestInitConfig(unittest.TestCase):
    """Test the configuration initialization function."""

    @patch("ask_bedrock.main.get_config")
    @patch("ask_bedrock.main.put_config")
    @patch("ask_bedrock.main.create_config")
    @patch("ask_bedrock.main.click.echo")
    def test_init_config_new_configuration(self, mock_echo, mock_create_config, mock_put_config, mock_get_config):
        """Test initializing a new configuration when none exists."""
        # Setup
        mock_get_config.return_value = None
        new_config = {
            "region": "us-west-2",
            "model_id": "anthropic.claude-v2",
            "aws_profile": "default",
            "parameters": {
                "temperature": 0.5,
                "max_tokens": 1000,
            }
        }
        mock_create_config.return_value = new_config
    
        # Execute
        result = init_config("test-context")
    
        # Assert
        mock_get_config.assert_called_once_with("test-context")
        mock_echo.assert_called_once()
        mock_create_config.assert_called_once_with(None)
        mock_put_config.assert_called_once_with("test-context", new_config)
        self.assertEqual(result, new_config)
    
    @patch("ask_bedrock.main.get_config")
    @patch("ask_bedrock.main.put_config")
    @patch("ask_bedrock.main.create_config")
    def test_init_config_existing_configuration(self, mock_create_config, mock_put_config, mock_get_config):
        """Test initializing with an existing configuration."""
        # Setup
        existing_config = {
            "region": "us-west-2",
            "model_id": "anthropic.claude-v2",
            "aws_profile": "default",
            "parameters": {
                "temperature": 0.5,
                "max_tokens": 1000,
            }
        }
        mock_get_config.return_value = existing_config
    
        # Execute
        result = init_config("test-context")
    
        # Assert
        mock_get_config.assert_called_once_with("test-context")
        mock_create_config.assert_not_called()
        mock_put_config.assert_not_called()
        self.assertEqual(result, existing_config)
    
    @patch("ask_bedrock.main.get_config")
    @patch("ask_bedrock.main.put_config")
    @patch("ask_bedrock.main.migrate_model_params")
    @patch("ask_bedrock.main.logger")
    @patch("ask_bedrock.main.json")
    def test_init_config_migration(self, mock_json, mock_logger, mock_migrate, mock_put_config, mock_get_config):
        """Test migrating from old config format to new format."""
        # Setup
        old_model_params = {"temperature": 0.7, "top_p": 0.9, "max_tokens_to_sample": 2000}
        old_config = {
            "region": "us-west-2",
            "model_id": "anthropic.claude-v2",
            "aws_profile": "default",
            "model_params": '{"temperature": 0.7, "top_p": 0.9, "max_tokens_to_sample": 2000}'
        }
    
        new_inference_config = {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 2000,
        }
    
        mock_get_config.return_value = old_config
        mock_migrate.return_value = new_inference_config
        mock_json.loads.return_value = old_model_params
        mock_json.dumps.return_value = '{"temperature": 0.7, "top_p": 0.9, "max_tokens": 2000}'
    
        # Execute
        result = init_config("test-context")
    
        # Assert
        mock_get_config.assert_called_once_with("test-context")
        mock_json.loads.assert_called_once_with(old_config["model_params"])
        mock_migrate.assert_called_once_with("anthropic.claude-v2", old_model_params)
        mock_json.dumps.assert_called_once_with(new_inference_config)
    
        # Check that put_config was called with updated config
        updated_config = old_config.copy()
        updated_config["inference_config"] = '{"temperature": 0.7, "top_p": 0.9, "max_tokens": 2000}'
        mock_put_config.assert_called_once_with("test-context", updated_config)
    
        # Result should be the updated config
        self.assertEqual(result, updated_config)
        self.assertIn("inference_config", result)


if __name__ == "__main__":
    unittest.main()