"""
Unit tests for the configuration functions in ask_bedrock.main.
"""

import unittest
from unittest.mock import patch, MagicMock, mock_open

from ask_bedrock.main import get_config, put_config


class TestConfigFunctions(unittest.TestCase):
    """Test the configuration handling functions."""

    def setUp(self):
        """Set up test environment."""
        self.test_config = {
            "contexts": {
                "test-context": {
                    "region": "us-west-2",
                    "model_id": "anthropic.claude-v2",
                    "parameters": {
                        "temperature": 0.5,
                        "max_tokens": 1000,
                    }
                }
            }
        }
        
    @patch("ask_bedrock.main.yaml.safe_load")
    @patch("ask_bedrock.main.open", mock_open(), create=True)
    @patch("ask_bedrock.main.os.path.exists")
    def test_get_config_existing_context(self, mock_exists, mock_yaml_load):
        """Test retrieving an existing context from config."""
        # Setup
        mock_exists.return_value = True
        mock_yaml_load.return_value = self.test_config
        
        # Execute
        config = get_config("test-context")
        
        # Assert
        self.assertIsNotNone(config)
        self.assertEqual(config["region"], "us-west-2")
        self.assertEqual(config["model_id"], "anthropic.claude-v2")
        self.assertEqual(config["parameters"]["temperature"], 0.5)
        
    @patch("ask_bedrock.main.os.path.exists")
    def test_get_config_no_config_file(self, mock_exists):
        """Test behavior when config file doesn't exist."""
        # Setup
        mock_exists.return_value = False
        
        # Execute
        config = get_config("test-context")
        
        # Assert
        self.assertIsNone(config)
        
    @patch("ask_bedrock.main.yaml.safe_load")
    @patch("ask_bedrock.main.open", mock_open(), create=True)
    @patch("ask_bedrock.main.os.path.exists")
    def test_get_config_missing_context(self, mock_exists, mock_yaml_load):
        """Test behavior when requested context doesn't exist."""
        # Setup
        mock_exists.return_value = True
        mock_yaml_load.return_value = self.test_config
        
        # Execute
        config = get_config("non-existent-context")
        
        # Assert
        self.assertIsNone(config)
        
    @patch("ask_bedrock.main.yaml.dump")
    @patch("ask_bedrock.main.yaml.safe_load")
    @patch("ask_bedrock.main.open", mock_open(), create=True)
    @patch("ask_bedrock.main.os.makedirs")
    @patch("ask_bedrock.main.os.path.exists")
    def test_put_config_new_file(self, mock_exists, mock_makedirs, mock_yaml_load, mock_yaml_dump):
        """Test creating a new config file with a new context."""
        # Setup
        mock_exists.return_value = False
        mock_yaml_dump.return_value = "dumped yaml"
        
        new_config = {
            "region": "eu-central-1",
            "model_id": "anthropic.claude-3-sonnet",
            "parameters": {
                "temperature": 0.7,
                "max_tokens": 2000,
            }
        }
        
        # Execute
        put_config("new-context", new_config)
        
        # Assert
        mock_makedirs.assert_called_once()
        
        expected_config = {"contexts": {"new-context": new_config}}
        mock_yaml_dump.assert_called_once_with(expected_config)
        
    @patch("ask_bedrock.main.yaml.dump")
    @patch("ask_bedrock.main.yaml.safe_load")
    @patch("ask_bedrock.main.open", mock_open(), create=True)
    @patch("ask_bedrock.main.os.path.exists")
    def test_put_config_update_existing(self, mock_exists, mock_yaml_load, mock_yaml_dump):
        """Test updating an existing config file with a new context."""
        # Setup
        mock_exists.return_value = True
        mock_yaml_load.return_value = self.test_config
        mock_yaml_dump.return_value = "dumped yaml"
        
        new_config = {
            "region": "eu-central-1",
            "model_id": "anthropic.claude-3-sonnet",
            "parameters": {
                "temperature": 0.7,
                "max_tokens": 2000,
            }
        }
        
        # Execute
        put_config("new-context", new_config)
        
        # Assert
        expected_contexts = {
            "test-context": self.test_config["contexts"]["test-context"],
            "new-context": new_config
        }
        expected_config = {"contexts": expected_contexts}
        
        # The test config should contain both contexts
        mock_yaml_dump.assert_called_once()
        called_config = mock_yaml_dump.call_args[0][0]
        self.assertEqual(called_config["contexts"]["test-context"], self.test_config["contexts"]["test-context"])
        self.assertEqual(called_config["contexts"]["new-context"], new_config)


if __name__ == "__main__":
    unittest.main()